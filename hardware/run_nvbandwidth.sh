#!/bin/bash
# GPU 显存带宽 + PCIe 带宽测试，来自 gpu_rent/常见问题/gpu_benchmark.md 的方法论
set -e

# 编译nvbandwidth需要nvcc，靠sources.sh自动探测持久化CUDA Toolkit并接进PATH/CUDA_HOME——
# 如果之前没跑过 install/download_cuda_toolkit.sh 装nvcc，这里会编译失败，需要先
# INSTALL_CUDA_TOOLKIT=1 bash install/download_cuda_toolkit.sh
source "$(dirname "${BASH_SOURCE[0]}")/../install/sources.sh"

# 工具本身(clone+编译产物)和结果都放持久化数据盘（DATA_DIR），不放 $HOME/github 或 /tmp——
# 那些路径都在根分区，是临时的，实例释放/重启就没了，得重新clone+编译
# （详见持久记忆 feedback_gpu_rental_persistent_data_disk）
DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
RESULTS_DIR="$DATA_DIR/benchmark_results"
NVBW_DIR="$DATA_DIR/tools/nvbandwidth"
mkdir -p "$RESULTS_DIR" "$(dirname "$NVBW_DIR")"

echo "=== 安装依赖 ==="
apt update && apt install -y libboost-program-options-dev cmake

if [ ! -d "$NVBW_DIR" ]; then
    echo "=== clone + 编译 nvbandwidth ==="
    # 先测GitHub直连，不通则走代理（每台实例网络环境不一样，不能假设一致——
    # 2026-07-21在fj02上实测：run_gpu_burn.sh的直连clone超时130秒才失败且无兜底，
    # 导致 set -e 直接中断整个脚本，这里同样补上代理容错）
    if curl -s -m 8 -o /dev/null -w '%{http_code}' https://github.com | grep -q '200'; then
        GITHUB_PREFIX=""
    else
        GITHUB_PREFIX="$GITHUB_PROXY_PREFIX"
    fi
    git clone "${GITHUB_PREFIX}https://github.com/NVIDIA/nvbandwidth.git" "$NVBW_DIR"
    mkdir -p "$NVBW_DIR/build"
    (cd "$NVBW_DIR/build" && cmake .. -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" && make -j4)
fi

echo "=== 运行 nvbandwidth（结果存到持久化数据盘: $RESULTS_DIR） ==="
cd "$NVBW_DIR/build"
./nvbandwidth | tee "$RESULTS_DIR/nvbandwidth_result_$(date +%Y%m%d_%H%M%S).log"

echo
echo "=== 关键字段说明 ==="
echo "PCIe H->D:        host_to_device_memcpy_ce"
echo "PCIe D->H:        device_to_host_memcpy_ce"
echo "显存内带宽:        device_local_copy"
echo
echo "⚠️ 不要和 gpu-burn 同时跑，会互相干扰结果"
