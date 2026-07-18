#!/bin/bash
# [wo GPU] 下载 base 模型 / 训练数据集 / JustRL 评测代码。纯IO，不需要GPU。
# 被 step1_wo_gpu.sh 调用，也可以单独跑。
# 用法：DATA_DIR=/root/rivermind-data bash download_data_and_code.sh
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"
# /root/rivermind-data 是持久化数据盘，不是根分区，实例释放/重启不会丢——
# 所有模型/数据集/代码/输出都必须落在这里，不要改成根分区下的路径
# （详见持久记忆 feedback_gpu_rental_persistent_data_disk / obsidian 环境与框架.md）
DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
mkdir -p "$DATA_DIR/models" "$DATA_DIR/datasets"

if ! command -v uv >/dev/null 2>&1; then
    # 官方安装脚本走GitHub Releases CDN，国内网络有时会卡住，超时就换更快的路径：apt装pip + pip($PYPI_MIRROR)装uv
    # 注意：不能靠这条管道命令自己的退出码判断成功与否——`cmd | tail -5` 这种管道，
    # `if !` 默认只看最后一段(tail)的退出码，tail 基本总是成功，会把真正失败的uv安装
    # 误判成"成功"。必须在安装尝试后重新显式检查 uv 是否真的能跑起来。
    timeout "$UV_INSTALL_TIMEOUT" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' 2>&1 | tail -5 || true
    export PATH="$HOME/.local/bin:$PATH"
    grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    if ! command -v uv >/dev/null 2>&1; then
        echo "官方安装脚本没能装出可用的uv，改用 apt装pip + pip($PYPI_MIRROR)装uv"
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

# 注意：不用"目录存不存在"判断要不要下载——目录存在不代表下载完整了（比如中途被打断，
# 会留下 *.incomplete 文件）。snapshot_download 本身就是幂等+可续传的：已完整的文件会
# 秒过，没下完/缺失的会自动续传，所以每次都直接调用它，不用自己维护完成状态。
echo "下载模型（优先 ModelScope，国内更快，找不到退回 huggingface_hub 经 $HF_ENDPOINT 镜像）..."
if [ "$PREFER_MODELSCOPE_FOR_MODELS" = true ] && python3 -c "
from modelscope import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B')
"; then
    echo "模型已通过 ModelScope 下载/校验完成"
else
    echo "ModelScope 下载失败/未启用，改走 huggingface_hub（经 $HF_ENDPOINT 镜像）"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B')
"
fi

echo "下载训练数据集（ModelScope上没有镜像，走 huggingface_hub 经 $HF_ENDPOINT 镜像）..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='open-r1/DAPO-Math-17k-Processed', repo_type='dataset', local_dir='$DATA_DIR/datasets/DAPO-Math-17k-Processed')
"

# 显式复查：snapshot_download正常返回不代表真的下完了（比如中途网络断开但异常被吞掉的边界情况），
# 留有 *.incomplete 文件就说明没下完，必须报错退出而不是让后面的步骤在残缺数据上继续跑
INCOMPLETE_FILES=$(find "$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B" "$DATA_DIR/datasets/DAPO-Math-17k-Processed" -name '*.incomplete' 2>/dev/null)
if [ -n "$INCOMPLETE_FILES" ]; then
    echo "错误：以下文件下载不完整，重新运行本脚本以续传："
    echo "$INCOMPLETE_FILES"
    exit 1
fi
echo "已核实：模型和数据集均无残留的 .incomplete 文件"

if [ ! -d "$DATA_DIR/JustRL" ]; then
    # 先测GitHub直连，不通则走代理（每台实例网络环境不一样，不能假设一致）
    if curl -s -m 8 -o /dev/null -w '%{http_code}' https://github.com | grep -q '200'; then
        GITHUB_PREFIX=""
    else
        GITHUB_PREFIX="$GITHUB_PROXY_PREFIX"
    fi
    git clone --depth 1 "${GITHUB_PREFIX}https://github.com/thunlp/JustRL.git" "$DATA_DIR/JustRL"
else
    echo "JustRL仓库已存在，跳过"
fi

echo "[download_data_and_code] 完成 —— 模型: $DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B | 数据: $DATA_DIR/datasets/DAPO-Math-17k-Processed | 代码: $DATA_DIR/JustRL"
