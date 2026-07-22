#!/bin/bash
# 统一分发源配置——被 install_deps_uv.sh 和 download_data_and_code.sh 一起 source。
# 改镜像/代理只改这一个文件，不用在多个脚本里分别改。
#
# 用法（在其他脚本里）：
#   source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

# ---- uv 装在持久化数据盘（$DATA_DIR/bin），不是 ~/.local/bin ----
# 官方安装脚本默认装到 ~/.local/bin，但那是根分区（30G，容器重置/实例释放就清空），
# 每次新开实例都要重新下载一遍 uv 本身。装到数据盘能跨容器重置存活，见
# download_data_and_code.sh 里 UV_INSTALL_DIR 的用法。
# 另外，`download_data_and_code.sh` 是被 step1_wo_gpu.sh 用 `bash xxx.sh` 起子进程
# 调用的，子进程里 export 的 PATH 不会传回父进程/后续脚本（子进程间不共享环境变量），
# 这里每次 source sources.sh 都补一遍 PATH，确保 install_python_deps.sh 等后续脚本
# 能找到已经装好的 uv，不用重新安装。
_SOURCES_DATA_DIR_EARLY="${DATA_DIR:-/root/rivermind-data}"
export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$_SOURCES_DATA_DIR_EARLY/bin}"
export PATH="$UV_INSTALL_DIR:$PATH"

# ---- PyPI 镜像：所有 uv/pip 安装统一走这个源 ----
export PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# ---- APT 镜像：仅作说明，实际由镜像商在系统里预配置（本项目机器目前是阿里云源），
#      这里不覆盖 /etc/apt/sources.list，只在需要时提示 ----
export APT_MIRROR_NOTE="阿里云(mirrors.aliyun.com)，由服务商预配置，脚本不修改"

# ---- HuggingFace：huggingface_hub 默认走这个镜像端点，不改代码只需设这一个环境变量 ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"  # 部分hf-mirror节点转发新版xet存储后端会401，详见环境文档

# ---- 模型下载优先级：能在 ModelScope 找到的模型优先走 ModelScope（国内速度更快，
#      且不依赖 HF_ENDPOINT 镜像的可用性）；HF专属资源（比如这次的DAPO-Math-17k数据集，
#      ModelScope上没有镜像）才走 huggingface_hub，经上面的 HF_ENDPOINT 镜像下载 ----
export PREFER_MODELSCOPE_FOR_MODELS="${PREFER_MODELSCOPE_FOR_MODELS:-true}"

# ---- GitHub：先探测直连，不通再走这个代理前缀（每台实例网络环境不一样，不能假设一致） ----
export GITHUB_PROXY_PREFIX="${GITHUB_PROXY_PREFIX:-https://ghfast.top/}"

# ---- GitHub Release 大文件下载专用代理链（跟上面git clone用的代理分开）----
# 2026-07-18在fj01上实测：GitHub Release资产会重定向到release-assets.githubusercontent.com，
# 直连这个域名只有~19KB/s（256MB的wheel根本下不完）；ghfast.top对这类资产重定向下载不work
# （直接失败）；gh-proxy.com实测能到~1.5MB/s，快80倍。所以release大文件下载单独维护一条
# 代理列表（不含"直连"，调用方自己先试直连再遍历这个列表），跟git clone用的
# GITHUB_PROXY_PREFIX区分开，因为两种下载方式(git协议 vs 直接文件下载)对代理的要求不一样。
export GITHUB_RELEASE_PROXY_CHAIN="${GITHUB_RELEASE_PROXY_CHAIN:-https://gh-proxy.com/ $GITHUB_PROXY_PREFIX}"

# ---- uv 官方安装脚本超时时间（秒）：超过这个时间就放弃走GitHub Releases CDN，
#      改用 apt(阿里云源)装pip + pip(清华源)装uv 这条更快路径 ----
export UV_INSTALL_TIMEOUT="${UV_INSTALL_TIMEOUT:-30}"

# ---- uv/pip 包缓存指到持久化数据盘 ----
# 租用实例的容器重置后，python包（torch/transformers/vllm等，装在根分区的
# /usr/local/lib/.../dist-packages）会被清空，得重新装；但如果下载缓存也在根分区，
# 连"重新下载"这一步都要再来一遍。把 UV_CACHE_DIR/PIP_CACHE_DIR 指到数据盘，
# 缓存能跨容器重置存活，下次重装能直接用缓存里的wheel，不用重新联网下载
# （详见持久记忆 feedback_gpu_rental_persistent_data_disk）。
_SOURCES_DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$_SOURCES_DATA_DIR/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$_SOURCES_DATA_DIR/.cache/pip}"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

# ---- Python 环境：--system 还是 venv，取决于当前用户能不能写系统 site-packages ----
# 云端租用实例通常是容器里的root，能直接写系统site-packages，用--system最简单。
# 但真机/个人长期机器（如kelvin-linux）跑的是普通用户，且现代Debian/Ubuntu的Python
# 默认是PEP668 "externally-managed"，`uv pip install --system`会被直接拒绝
# （2026-07-21在kelvin-linux上实测：报错"error: The interpreter at /usr is externally
# managed"，即使加--break-system-packages绕过这个限制，dist-packages目录本身是
# root所有，普通用户依然Permission denied）。这里自动探测：能写系统site-packages
# 就用--system（跟之前云端实例行为完全一致，不改变现有脚本习惯）；不能写就在数据盘
# 建一个venv，所有uv pip install改用`--python $VENV_PYTHON`而不是`--system`。
_SOURCES_DATA_DIR_FOR_VENV="${DATA_DIR:-/root/rivermind-data}"
_SITE_PACKAGES_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)
if [ -n "$_SITE_PACKAGES_DIR" ] && [ -w "$_SITE_PACKAGES_DIR" ]; then
    export UV_PYTHON_TARGET_FLAG="--system"
    export UV_PYTHON_BIN="python3"
else
    export UV_VENV_DIR="${UV_VENV_DIR:-$_SOURCES_DATA_DIR_FOR_VENV/venv}"
    if [ ! -x "$UV_VENV_DIR/bin/python3" ]; then
        mkdir -p "$(dirname "$UV_VENV_DIR")"
        # 优先用uv自己建venv（更快），uv还没装好时退回系统venv模块
        if command -v uv >/dev/null 2>&1; then
            uv venv "$UV_VENV_DIR" --python python3 >&2
        else
            python3 -m venv "$UV_VENV_DIR" >&2
        fi
    fi
    export UV_PYTHON_TARGET_FLAG="--python $UV_VENV_DIR/bin/python3"
    export UV_PYTHON_BIN="$UV_VENV_DIR/bin/python3"
    export PATH="$UV_VENV_DIR/bin:$PATH"
fi

# ---- CUDA Toolkit：优先用数据盘持久化装的，其次探测系统自带的（真机常见，见下）----
# 如果之前跑过 download_cuda_toolkit.sh（INSTALL_CUDA_TOOLKIT=1），nvcc 会装在
# $DATA_DIR/cuda-toolkit（持久化，跨容器重置存活）。这里自动探测，装了就接进PATH/CUDA_HOME，
# 没装就跳过——不需要每次都手动设置，也不会因为没装而报错。
# 2026-07-22在kelvin-linux（真机+Tailscale连接，非云端租用容器）上发现：系统本身已经
# 装了CUDA Toolkit（/usr/local/cuda-12.6/bin/nvcc），但没在默认PATH里，之前只探测数据盘
# 那条路径，导致gpu-burn编译时`command -v nvcc`找不到、cublas_v2.h报错。补一条系统路径
# 的探测（/usr/local/cuda/bin，标准CUDA安装的PATH约定，通常是个指向具体版本的符号链接）。
_CUDA_TOOLKIT_PATH="$_SOURCES_DATA_DIR/cuda-toolkit"
if [ -x "$_CUDA_TOOLKIT_PATH/bin/nvcc" ]; then
    export CUDA_HOME="$_CUDA_TOOLKIT_PATH"
    export PATH="$_CUDA_TOOLKIT_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$_CUDA_TOOLKIT_PATH/lib64:$LD_LIBRARY_PATH"
elif command -v nvcc >/dev/null 2>&1; then
    : # 已经在PATH里，不用额外处理
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

if [ "$UV_PYTHON_TARGET_FLAG" = "--system" ]; then
    echo "[sources] Python目标: --system（系统site-packages可写，跟云端root容器一致）"
else
    echo "[sources] Python目标: venv $UV_VENV_DIR（系统site-packages不可写，如真机非root用户/PEP668限制）"
fi

echo "[sources] PYPI_MIRROR=$PYPI_MIRROR"
echo "[sources] HF_ENDPOINT=$HF_ENDPOINT (HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET)"
echo "[sources] GITHUB_PROXY_PREFIX=$GITHUB_PROXY_PREFIX (仅在直连不可用时使用)"
echo "[sources] GITHUB_RELEASE_PROXY_CHAIN=$GITHUB_RELEASE_PROXY_CHAIN (Release大文件下载专用，直连太慢/失败时依次尝试)"
echo "[sources] PREFER_MODELSCOPE_FOR_MODELS=$PREFER_MODELSCOPE_FOR_MODELS"
echo "[sources] UV_CACHE_DIR=$UV_CACHE_DIR | PIP_CACHE_DIR=$PIP_CACHE_DIR（持久化数据盘，跨容器重置存活）"
if [ -n "${CUDA_HOME:-}" ]; then
    echo "[sources] 检测到持久化CUDA Toolkit: CUDA_HOME=$CUDA_HOME"
else
    echo "[sources] 未检测到持久化CUDA Toolkit（未装，或还没跑 download_cuda_toolkit.sh）"
fi

# 持久化到 ~/.bashrc，后续新开 shell 也生效（不重复追加）——注意 ~/.bashrc 本身在根分区，
# 容器重置会被清空，所以这里每次 source sources.sh 都会重新写一遍，不依赖它跨重置存活
for line in \
    "export HF_ENDPOINT=$HF_ENDPOINT" \
    "export HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET" \
    "export UV_CACHE_DIR=$UV_CACHE_DIR" \
    "export PIP_CACHE_DIR=$PIP_CACHE_DIR"
do
    grep -qF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
done
if [ -n "${CUDA_HOME:-}" ]; then
    for line in \
        "export CUDA_HOME=$CUDA_HOME" \
        "export PATH=$_CUDA_TOOLKIT_PATH/bin:\$PATH" \
        "export LD_LIBRARY_PATH=$_CUDA_TOOLKIT_PATH/lib64:\$LD_LIBRARY_PATH"
    do
        grep -qF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
    done
fi
