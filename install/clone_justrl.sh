#!/bin/bash
# [wo GPU] Clone JustRL 评测代码仓库。不需要 uv/modelscope，只用 git+curl，
# 独立成单文件是为了能被 step1_wo_gpu.sh 跟其他下载任务并行拉起（互相没有依赖）。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"
DATA_DIR="${DATA_DIR:-/root/rivermind-data}"

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
echo "[clone_justrl] 完成 —— $DATA_DIR/JustRL"
