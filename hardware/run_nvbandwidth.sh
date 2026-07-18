#!/bin/bash
# GPU 显存带宽 + PCIe 带宽测试，来自 gpu_rent/常见问题/gpu_benchmark.md 的方法论
set -e

NVBW_DIR="$HOME/github/nvbandwidth"

echo "=== 安装依赖 ==="
apt update && apt install -y libboost-program-options-dev

if [ ! -d "$NVBW_DIR" ]; then
    echo "=== clone + 编译 nvbandwidth ==="
    git clone https://github.com/NVIDIA/nvbandwidth.git "$NVBW_DIR"
    mkdir -p "$NVBW_DIR/build"
    (cd "$NVBW_DIR/build" && cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && make -j4)
fi

echo "=== 运行 nvbandwidth ==="
cd "$NVBW_DIR/build"
./nvbandwidth | tee "/tmp/nvbandwidth_result_$(date +%Y%m%d_%H%M%S).log"

echo
echo "=== 关键字段说明 ==="
echo "PCIe H->D:        host_to_device_memcpy_ce"
echo "PCIe D->H:        device_to_host_memcpy_ce"
echo "显存内带宽:        device_local_copy"
echo
echo "⚠️ 不要和 gpu-burn 同时跑，会互相干扰结果"
