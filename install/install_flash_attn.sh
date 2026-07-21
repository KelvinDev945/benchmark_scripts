#!/bin/bash
# 装 flash-attn ——只下预编译wheel，不做源码编译（2026-07-21用户明确要求：不允许退回
# 源码编译，必须想办法把wheel下下来。之前源码编译在无nvcc环境下直接报错
# `CUDA_HOME environment variable is not set`，装CUDA toolkit来支持编译本身就要
# 多下几百MB+装一遍工具链，还不如死磕把wheel下完）。
#
# 背景：flash-attn 官方 GitHub Releases 预编译wheel矩阵目前覆盖到 cu12+torch2.8
# （torch2.9只有cu13的wheel，torch2.10完全没有）。install_python_deps.sh 已经把
# torch锁定在 2.8.x，就是为了让这里能命中预编译wheel。
#
# ⚠️ GitHub Release 资产会重定向到 release-assets.githubusercontent.com，直连/代理
# 速度因实例网络环境而异，且经常"下到大半又卡住"（2026-07-21在fj02上实测：gh-proxy.com
# 下到208MB/~260MB时被限速判定卡住，之前的实现一失败就整个丢弃重下，浪费已下载的部分）。
# 现在改成：
#   1. wheel下载到持久化数据盘的固定路径（不是mktemp随机名），失败/中断后用
#      `curl -C -` 断点续传，不重新下载已经下好的部分；
#   2. 同一个源允许重试多次（利用续传，每次重试都是从上次中断处继续，而不是从0开始）；
#   3. 每个源在多次续传重试后仍拿不到完整文件才换下一个源；
#   4. 全部源都试完仍未成功就报错退出（exit 1），不静默退回源码编译。
#
# 可以在任意阶段跑（不需要GPU在场）。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1}"
FLASH_ATTN_RELEASE_TAG="v${FLASH_ATTN_VERSION}"  # GitHub release tag，如 v2.8.3.post1
MAX_RETRIES_PER_SOURCE="${FLASH_ATTN_MAX_RETRIES_PER_SOURCE:-6}"

# ---- 探测当前环境，拼出预编译wheel文件名 ----
PY_TAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
TORCH_TAG=$(python3 -c "
import torch
v = torch.__version__.split('+')[0]
major, minor = v.split('.')[:2]
print(f'{major}.{minor}')
")
CXX11ABI=$(python3 -c "
import torch
print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')
")

if [ -z "$TORCH_TAG" ] || [ -z "$CXX11ABI" ]; then
    echo "[install_flash_attn] 错误：探测torch版本/ABI失败（torch没装好？install_python_deps.sh 应该先跑完）"
    exit 1
fi

WHEEL_NAME="flash_attn-${FLASH_ATTN_VERSION}+cu12torch${TORCH_TAG}cxx11abi${CXX11ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
RAW_URL="https://github.com/Dao-AILab/flash-attention/releases/download/${FLASH_ATTN_RELEASE_TAG}/${WHEEL_NAME}"
echo "[install_flash_attn] 探测到 torch${TORCH_TAG} / ${PY_TAG} / cxx11abi${CXX11ABI}，目标wheel："
echo "[install_flash_attn] $RAW_URL"

DOWNLOAD_DIR="${DATA_DIR:-/root/rivermind-data}/tmp"
mkdir -p "$DOWNLOAD_DIR"
WHEEL_TMP="$DOWNLOAD_DIR/$WHEEL_NAME"  # 固定路径（不是mktemp随机名），断点续传靠这个复用
FLASH_ATTN_LOG=$(mktemp)
INSTALLED=false

# 依次尝试：直连 → GITHUB_RELEASE_PROXY_CHAIN 里配置的各个代理（sources.sh统一管理）
for PREFIX in "" $GITHUB_RELEASE_PROXY_CHAIN; do
    URL="${PREFIX}${RAW_URL}"
    SRC_NAME="${PREFIX:-直连}"
    echo "[install_flash_attn] 尝试源: ${SRC_NAME}（断点续传，最多重试 ${MAX_RETRIES_PER_SOURCE} 次）"

    for attempt in $(seq 1 "$MAX_RETRIES_PER_SOURCE"); do
        PREV_SIZE=$(stat -c%s "$WHEEL_TMP" 2>/dev/null || stat -f%z "$WHEEL_TMP" 2>/dev/null || echo 0)
        # -C -：断点续传，从 $WHEEL_TMP 已有的字节数继续下载，不重新下已经下好的部分。
        # --speed-limit/--speed-time：卡住(而非彻底失败)就让curl自己退出，好让本函数重试续传，
        # 而不是傻等 -m 设置的总超时。
        if curl -fL -C - --speed-limit 51200 --speed-time 20 -m 600 -o "$WHEEL_TMP" "$URL" 2>"$FLASH_ATTN_LOG"; then
            DOWNLOADED_SIZE=$(stat -c%s "$WHEEL_TMP" 2>/dev/null || stat -f%z "$WHEEL_TMP" 2>/dev/null || echo 0)
            if [ "$DOWNLOADED_SIZE" -gt 10000000 ]; then  # 至少10MB，防止下到一个错误页面当成功
                echo "[install_flash_attn] 下载完成（${DOWNLOADED_SIZE} 字节），来源: ${SRC_NAME}"
                if uv pip install --system "$WHEEL_TMP" > "$FLASH_ATTN_LOG" 2>&1; then
                    echo "[install_flash_attn] ✅ 预编译wheel安装成功"
                    INSTALLED=true
                else
                    echo "[install_flash_attn] wheel下载到了但装不上，报错："
                    cat "$FLASH_ATTN_LOG"
                fi
                break
            else
                echo "[install_flash_attn] 下载文件太小（${DOWNLOADED_SIZE} 字节），可能是错误页面，放弃这个源"
                rm -f "$WHEEL_TMP"
                break
            fi
        else
            CUR_SIZE=$(stat -c%s "$WHEEL_TMP" 2>/dev/null || stat -f%z "$WHEEL_TMP" 2>/dev/null || echo 0)
            if [ "$CUR_SIZE" -gt "$PREV_SIZE" ]; then
                echo "[install_flash_attn]   第${attempt}次尝试中断（${SRC_NAME}），已下载 ${CUR_SIZE} 字节，续传重试"
            else
                echo "[install_flash_attn]   第${attempt}次尝试完全没有进展（${SRC_NAME}，仍是 ${CUR_SIZE} 字节），这个源可能不通"
            fi
        fi
    done

    if [ "$INSTALLED" = true ]; then
        break
    fi
    echo "[install_flash_attn] 源 ${SRC_NAME} 重试${MAX_RETRIES_PER_SOURCE}次后仍未拿到完整wheel，换下一个源（已下载部分保留在 $WHEEL_TMP，下次可续传）"
done

rm -f "$FLASH_ATTN_LOG"

if [ "$INSTALLED" = false ]; then
    echo "[install_flash_attn] 错误：所有源都试过了，wheel仍未下完（部分文件保留在 $WHEEL_TMP，方便下次续传，不自动清理）"
    echo "[install_flash_attn] 按要求不退回源码编译，直接报错退出——重新跑本脚本会从断点继续"
    exit 1
fi
