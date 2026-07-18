#!/bin/bash
# 统一分发源配置——被 install_deps_uv.sh 和 download_data_and_code.sh 一起 source。
# 改镜像/代理只改这一个文件，不用在多个脚本里分别改。
#
# 用法（在其他脚本里）：
#   source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

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
export PREFER_MODELSCOPE_FOR_MODELS=true

# ---- GitHub：先探测直连，不通再走这个代理前缀（每台实例网络环境不一样，不能假设一致） ----
export GITHUB_PROXY_PREFIX="${GITHUB_PROXY_PREFIX:-https://ghfast.top/}"

# ---- uv 官方安装脚本超时时间（秒）：超过这个时间就放弃走GitHub Releases CDN，
#      改用 apt(阿里云源)装pip + pip(清华源)装uv 这条更快路径 ----
export UV_INSTALL_TIMEOUT="${UV_INSTALL_TIMEOUT:-30}"

echo "[sources] PYPI_MIRROR=$PYPI_MIRROR"
echo "[sources] HF_ENDPOINT=$HF_ENDPOINT (HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET)"
echo "[sources] GITHUB_PROXY_PREFIX=$GITHUB_PROXY_PREFIX (仅在直连不可用时使用)"
echo "[sources] PREFER_MODELSCOPE_FOR_MODELS=$PREFER_MODELSCOPE_FOR_MODELS"

# 持久化到 ~/.bashrc，后续新开 shell 也生效（不重复追加）
for line in \
    "export HF_ENDPOINT=$HF_ENDPOINT" \
    "export HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
do
    grep -qF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
done
