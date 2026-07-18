#!/bin/bash
# GPU 算力压测（Tensor Core FP16/BF16），来自 gpu_rent/常见问题/gpu_benchmark.md 的方法论
# 用法：./run_gpu_burn.sh [压测时长秒数，默认180]
set -e

DURATION="${1:-180}"
GPUBURN_DIR="$HOME/github/gpu-burn"

if [ ! -d "$GPUBURN_DIR" ]; then
    echo "=== clone + 编译 gpu-burn ==="
    git clone https://github.com/wilicc/gpu-burn.git "$GPUBURN_DIR"
    (cd "$GPUBURN_DIR" && make)
fi

echo "=== 运行 gpu-burn（Tensor Core, 显存占用90%上限, ${DURATION}秒） ==="
echo "⚠️ 注意：gpu-burn 用 WMMA API，对 Blackwell(50系) 架构没做针对性优化，"
echo "   测出来的数字可能低于该架构的真实算力（尤其是FP8），只能作为同代架构内的参考"
cd "$GPUBURN_DIR"
./gpu_burn -tc -m 90% "$DURATION" | tee "/tmp/gpu_burn_result_$(date +%Y%m%d_%H%M%S).log"

echo
echo "=== 关键指标说明 ==="
echo "Tensor Core 算力: 看输出里 Gflop/s 的稳定值"
echo "错误数: errors 必须为 0，非0说明硬件/驱动有问题"
echo "温度: 满载 <85°C 正常"
