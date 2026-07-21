#!/bin/bash
# GPU 算力压测（Tensor Core FP16/BF16），来自 gpu_rent/常见问题/gpu_benchmark.md 的方法论
# 用法：./run_gpu_burn.sh [压测时长秒数，默认180]
set -e

# 编译gpu-burn需要CUDA头文件(cublas_v2.h等)，靠sources.sh自动探测持久化CUDA Toolkit
# 并接进PATH/CUDA_HOME——如果之前没跑过 install/download_cuda_toolkit.sh 装nvcc，
# 这里会编译失败，需要先 INSTALL_CUDA_TOOLKIT=1 bash install/download_cuda_toolkit.sh
source "$(dirname "${BASH_SOURCE[0]}")/../install/sources.sh"

# 工具本身(clone+编译产物)和结果都放持久化数据盘（DATA_DIR），不放 $HOME/github 或 /tmp——
# 那些路径都在根分区，是临时的，实例释放/重启就没了，得重新clone+编译
# （详见持久记忆 feedback_gpu_rental_persistent_data_disk）
DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
RESULTS_DIR="$DATA_DIR/benchmark_results"
GPUBURN_DIR="$DATA_DIR/tools/gpu-burn"
mkdir -p "$RESULTS_DIR" "$(dirname "$GPUBURN_DIR")"

DURATION="${1:-180}"

if [ ! -d "$GPUBURN_DIR" ]; then
    echo "=== clone + 编译 gpu-burn（存到持久化数据盘: $GPUBURN_DIR） ==="
    # 先测GitHub直连，不通则走代理（每台实例网络环境不一样，不能假设一致——
    # 2026-07-21在fj02上实测：直连超时130秒才失败，且这里之前没有代理兜底，
    # 导致 set -e 直接中断整个脚本，压测根本没跑起来）
    if curl -s -m 8 -o /dev/null -w '%{http_code}' https://github.com | grep -q '200'; then
        GITHUB_PREFIX=""
    else
        GITHUB_PREFIX="$GITHUB_PROXY_PREFIX"
    fi
    git clone "${GITHUB_PREFIX}https://github.com/wilicc/gpu-burn.git" "$GPUBURN_DIR"
    # gpu-burn 的 Makefile 只认 /usr 或 /usr/local/cuda 这两个硬编码路径找nvcc，
    # 不认 CUDA_HOME/PATH，必须显式传 CUDAPATH 变量给make，否则INCLUDE路径是空的
    # （拼出 -I/include 这种错误路径，实测报 cublas_v2.h 找不到）
    CUDA_ROOT=$(dirname "$(dirname "$(command -v nvcc)")")
    (cd "$GPUBURN_DIR" && make CUDAPATH="$CUDA_ROOT")
fi

echo "=== 运行 gpu-burn（Tensor Core, 显存占用90%上限, ${DURATION}秒，结果存到: $RESULTS_DIR） ==="
echo "⚠️ 注意：gpu-burn 用 WMMA API，对 Blackwell(50系) 架构没做针对性优化，"
echo "   测出来的数字可能低于该架构的真实算力（尤其是FP8），只能作为同代架构内的参考"
cd "$GPUBURN_DIR"
./gpu_burn -tc -m 90% "$DURATION" | tee "$RESULTS_DIR/gpu_burn_result_$(date +%Y%m%d_%H%M%S).log"

echo
echo "=== 关键指标说明 ==="
echo "Tensor Core 算力: 看输出里 Gflop/s 的稳定值"
echo "错误数: errors 必须为 0，非0说明硬件/驱动有问题"
echo "温度: 满载 <85°C 正常"
