#!/bin/bash
# [wo GPU] 下载 base 模型。假定 bootstrap_uv.sh 已经跑过（uv/modelscope/huggingface_hub 已装好）。
# 独立成单文件是为了能被 step1_wo_gpu.sh 跟其他下载任务并行拉起（互相没有依赖）。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"
DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
mkdir -p "$DATA_DIR/models"

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

INCOMPLETE_FILES=$(find "$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B" -name '*.incomplete' 2>/dev/null)
if [ -n "$INCOMPLETE_FILES" ]; then
    echo "错误：以下模型文件下载不完整，重新运行本脚本以续传："
    echo "$INCOMPLETE_FILES"
    exit 1
fi
echo "[download_model] 完成 —— $DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B"
