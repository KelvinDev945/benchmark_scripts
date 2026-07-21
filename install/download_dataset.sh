#!/bin/bash
# [wo GPU] 下载训练数据集。假定 bootstrap_uv.sh 已经跑过（uv/huggingface_hub 已装好）。
# 独立成单文件是为了能被 step1_wo_gpu.sh 跟其他下载任务并行拉起（互相没有依赖）。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"
DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
mkdir -p "$DATA_DIR/datasets"

echo "下载训练数据集（ModelScope上没有镜像，走 huggingface_hub 经 $HF_ENDPOINT 镜像）..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='open-r1/DAPO-Math-17k-Processed', repo_type='dataset', local_dir='$DATA_DIR/datasets/DAPO-Math-17k-Processed')
"

INCOMPLETE_FILES=$(find "$DATA_DIR/datasets/DAPO-Math-17k-Processed" -name '*.incomplete' 2>/dev/null)
if [ -n "$INCOMPLETE_FILES" ]; then
    echo "错误：以下数据集文件下载不完整，重新运行本脚本以续传："
    echo "$INCOMPLETE_FILES"
    exit 1
fi
echo "[download_dataset] 完成 —— $DATA_DIR/datasets/DAPO-Math-17k-Processed"
