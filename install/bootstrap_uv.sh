#!/bin/bash
# [wo GPU] 确保 uv 可用 + 装 modelscope/huggingface_hub（后面几个下载脚本都依赖这两个）。
# 必须同步跑在最前面（被 step1_wo_gpu.sh 并行拉起其他任务之前先跑完一次）——
# 拆出来是因为 download_model.sh / download_dataset.sh / install_python_deps.sh /
# install_flash_attn.sh 全部要用 uv，如果并行跑、每个都自己去"if ! command -v uv"
# 装一遍，会导致同一时间多个进程抢着装 uv，互相踩踏。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

if ! command -v uv >/dev/null 2>&1; then
    # 装到 $UV_INSTALL_DIR（数据盘，见 sources.sh），不用官方脚本默认的 ~/.local/bin
    # （根分区，容器重置就没了）。官方安装脚本走GitHub Releases CDN，国内网络有时会
    # 卡住，超时就换更快的路径：apt装pip + pip($PYPI_MIRROR)装uv。
    # 注意：不能靠这条管道命令自己的退出码判断成功与否——`cmd | tail -5` 这种管道，
    # `if !` 默认只看最后一段(tail)的退出码，tail 基本总是成功，会把真正失败的uv安装
    # 误判成"成功"。必须在安装尝试后重新显式检查 uv 是否真的能跑起来。
    mkdir -p "$UV_INSTALL_DIR"
    timeout "$UV_INSTALL_TIMEOUT" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' 2>&1 | tail -5 || true
    export PATH="$UV_INSTALL_DIR:$PATH"
    grep -qF "$UV_INSTALL_DIR" ~/.bashrc 2>/dev/null || echo "export PATH=\"$UV_INSTALL_DIR:\$PATH\"" >> ~/.bashrc
    if ! command -v uv >/dev/null 2>&1; then
        # 这条 fallback 路径很少触发（只有官方安装脚本连超时都失败时才走到这），
        # 装到根分区（不持久化）够用，不为这个低概率分支专门做数据盘持久化
        echo "官方安装脚本没能装出可用的uv，改用 apt装pip + pip($PYPI_MIRROR)装uv（根分区，不持久化，下次新容器需重装）"
        apt-get update -qq && apt-get install -y -qq python3-pip
        python3 -m pip install -qqq uv -i "$PYPI_MIRROR"
    fi
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "错误：两条安装路径都没能装出可用的uv，无法继续"
    exit 1
fi
echo "uv: $(uv --version)"

if ! python3 -c "import modelscope" >/dev/null 2>&1; then
    uv pip install --system -qqq -i "$PYPI_MIRROR" modelscope huggingface_hub
fi
echo "[bootstrap_uv] 完成"
