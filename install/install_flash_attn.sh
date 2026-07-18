#!/bin/bash
# 装 flash-attn ——优先直接下载官方预编译wheel（几秒钟装完），只有找不到匹配wheel时
# 才退回源码编译（需要nvcc，见 download_cuda_toolkit.sh）。
#
# 背景：flash-attn 官方 GitHub Releases 预编译wheel矩阵目前覆盖到 cu12+torch2.8
# （torch2.9只有cu13的wheel，torch2.10完全没有）。install_python_deps.sh 已经把
# torch锁定在 2.8.x，就是为了让这里能命中预编译wheel、不用编译——2026-07-18在fj01上
# 实测：torch2.10时flash-attn没有匹配wheel，被迫源码编译，光编译就烧了30+分钟CPU时间。
#
# 可以在任意阶段跑（不需要GPU在场，装wheel和源码编译都只需要CPU）。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
FLASH_ATTN_RELEASE_TAG="v${FLASH_ATTN_VERSION}"  # GitHub release tag，如 v2.8.3.post1

# ---- 探测当前环境，拼出预编译wheel文件名 ----
PY_TAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
TORCH_TAG=$(python3 -c "
import torch
v = torch.__version__.split('+')[0]
major, minor = v.split('.')[:2]
print(f'{major}.{minor}')
" 2>/dev/null || echo "")
CXX11ABI=$(python3 -c "
import torch
print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')
" 2>/dev/null || echo "")

if [ -z "$TORCH_TAG" ] || [ -z "$CXX11ABI" ]; then
    echo "[install_flash_attn] 探测torch版本/ABI失败（torch没装好？），直接退回源码编译"
    WHEEL_URL=""
else
    WHEEL_NAME="flash_attn-${FLASH_ATTN_VERSION}+cu12torch${TORCH_TAG}cxx11abi${CXX11ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/${FLASH_ATTN_RELEASE_TAG}/${WHEEL_NAME}"
    echo "[install_flash_attn] 探测到 torch${TORCH_TAG} / ${PY_TAG} / cxx11abi${CXX11ABI}，尝试直接装预编译wheel："
    echo "[install_flash_attn] $WHEEL_URL"
fi

FLASH_ATTN_LOG=$(mktemp)
INSTALLED=false

if [ -n "$WHEEL_URL" ] && curl -sfI -m 15 "$WHEEL_URL" > /dev/null 2>&1; then
    if uv pip install --system "$WHEEL_URL" > "$FLASH_ATTN_LOG" 2>&1; then
        echo "[install_flash_attn] ✅ 预编译wheel安装成功（无需编译）"
        INSTALLED=true
    else
        echo "[install_flash_attn] 预编译wheel下载到了但装不上，报错："
        cat "$FLASH_ATTN_LOG"
    fi
else
    echo "[install_flash_attn] 没有找到匹配的预编译wheel（当前torch${TORCH_TAG}可能超出flash-attn wheel矩阵范围），退回源码编译"
fi

if [ "$INSTALLED" = false ]; then
    echo "[install_flash_attn] 源码编译中（需要nvcc，耗时较长，通常30分钟以上）..."
    if uv pip install --system -i "$PYPI_MIRROR" "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation > "$FLASH_ATTN_LOG" 2>&1; then
        echo "[install_flash_attn] 源码编译安装成功"
        INSTALLED=true
    else
        # 不静默吞掉：失败时打印完整报错，方便判断是真的装不上还是偶发网络问题
        echo "[install_flash_attn] 警告：安装失败（不影响主流程，Unsloth会退回FlashAttention2/xformers），完整报错："
        cat "$FLASH_ATTN_LOG"
    fi
fi

rm -f "$FLASH_ATTN_LOG"
