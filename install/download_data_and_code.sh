#!/bin/bash
# 下载基准测试需要的模型/数据集/评测代码——无卡（CPU）阶段就能跑，提前做好省挂卡计费时间
# 用法：DATA_DIR=/root/rivermind-data bash download_data_and_code.sh
set -e

DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
mkdir -p "$DATA_DIR/models" "$DATA_DIR/datasets"

echo '=== 0. 检查/装好 modelscope + huggingface_hub（下载模型/数据集要用） ==='
if ! python3 -c "import modelscope" >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install --system -qqq modelscope huggingface_hub
    else
        pip install -qqq modelscope huggingface_hub
    fi
fi

echo '=== 1. 禁用 HF 新版 xet 存储后端（部分 hf-mirror 节点转发会401，详见环境文档） ==='
export HF_HUB_DISABLE_XET=1
grep -q HF_HUB_DISABLE_XET ~/.bashrc 2>/dev/null || echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc

echo '=== 2. 下载 base 模型 DeepSeek-R1-Distill-Qwen-1.5B（走 ModelScope，国内更快） ==='
if [ ! -d "$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B" ]; then
    python3 -c "
from modelscope import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', local_dir='$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B')
"
else
    echo "已存在，跳过：$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B"
fi

echo '=== 3. 下载训练数据集 open-r1/DAPO-Math-17k-Processed（走 HuggingFace） ==='
if [ ! -d "$DATA_DIR/datasets/DAPO-Math-17k-Processed" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='open-r1/DAPO-Math-17k-Processed',
    repo_type='dataset',
    local_dir='$DATA_DIR/datasets/DAPO-Math-17k-Processed',
)
"
else
    echo "已存在，跳过：$DATA_DIR/datasets/DAPO-Math-17k-Processed"
fi

echo '=== 4. clone JustRL 仓库（评测脚本 + 9个benchmark数据） ==='
echo '    先测本实例GitHub直连是否可用，不可用则走 ghfast.top 代理（每台实例网络环境不一样，详见环境文档）'
if curl -s -m 8 -o /dev/null -w '%{http_code}' https://github.com | grep -q '200'; then
    GITHUB_PREFIX=""
    echo "GitHub 直连可用"
else
    GITHUB_PREFIX="https://ghfast.top/"
    echo "GitHub 直连不可用，改用 ghfast.top 代理"
fi

if [ ! -d "$DATA_DIR/JustRL" ]; then
    git clone --depth 1 "${GITHUB_PREFIX}https://github.com/thunlp/JustRL.git" "$DATA_DIR/JustRL"
else
    echo "已存在，跳过：$DATA_DIR/JustRL"
fi

echo '=== 全部完成 ==='
echo "模型: $DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B"
echo "数据: $DATA_DIR/datasets/DAPO-Math-17k-Processed"
echo "代码: $DATA_DIR/JustRL"
