#!/bin/bash
# [with GPU] 装 flash-attn——无卡阶段没有nvcc，装不了；挂卡后nvcc才可用，这里才是真正能装的时机。
# 被 step2_with_gpu.sh 调用，也可以单独跑。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

FLASH_ATTN_LOG=$(mktemp)
if uv pip install --system -i "$PYPI_MIRROR" flash-attn --no-build-isolation > "$FLASH_ATTN_LOG" 2>&1; then
    echo "[install_flash_attn] 安装成功"
else
    # 不静默吞掉：失败时打印完整报错，方便判断是真的装不上还是偶发网络问题
    echo "[install_flash_attn] 警告：安装失败（不影响主流程，Unsloth会退回FlashAttention2/xformers），完整报错："
    cat "$FLASH_ATTN_LOG"
fi
rm -f "$FLASH_ATTN_LOG"
