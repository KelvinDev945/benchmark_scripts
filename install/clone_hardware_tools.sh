#!/bin/bash
# [wo GPU] 提前 clone gpu-burn / nvbandwidth 源码（Step 3 硬件压测要用）。
# 只 clone，不编译——编译虽然也不需要真实GPU在场，但依赖 nvcc（由并行组里的
# download_cuda_toolkit 装），跟这个脚本并行跑的话时序没保证，干脆把编译留在
# Step 3 的 hardware/run_gpu_burn.sh / run_nvbandwidth.sh 里按需做（那两个脚本
# 已经改成分别探测"目录存在"和"二进制存在"，clone和编译各自幂等，不会重复做）。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
TOOLS_DIR="$DATA_DIR/tools"
mkdir -p "$TOOLS_DIR"

if curl -s -m 8 -o /dev/null -w '%{http_code}' https://github.com | grep -q '200'; then
    GITHUB_PREFIX=""
else
    GITHUB_PREFIX="$GITHUB_PROXY_PREFIX"
fi

if [ ! -d "$TOOLS_DIR/gpu-burn" ]; then
    echo "=== clone gpu-burn ==="
    git clone "${GITHUB_PREFIX}https://github.com/wilicc/gpu-burn.git" "$TOOLS_DIR/gpu-burn"
else
    echo "gpu-burn 已存在，跳过 clone"
fi

if [ ! -d "$TOOLS_DIR/nvbandwidth" ]; then
    echo "=== clone nvbandwidth ==="
    git clone "${GITHUB_PREFIX}https://github.com/NVIDIA/nvbandwidth.git" "$TOOLS_DIR/nvbandwidth"
else
    echo "nvbandwidth 已存在，跳过 clone"
fi

echo "[clone_hardware_tools] 完成"
