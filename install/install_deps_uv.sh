#!/bin/bash
# 用 uv 装 GRPO 训练 + 基准测试所需的 python 依赖
# 比 pip 快很多，且这里用同一条命令统一解出所有约束，避免分次 pip install 时后装的包
# 静默破坏前面的版本约束（2026-07-17 在 hn01 上踩过这个坑，详见 obsidian
# ongoing_project_related/llm-rl-experiments/环境与框架.md）
set -e

echo '=== 1. 安装/升级 uv ==='
pip install --upgrade -qqq uv

echo '=== 2. 探测是否 Tesla T4（老架构，vllm/triton 版本要求不同） ==='
IS_T4=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | grep -q "Tesla T4"; then
    IS_T4=true
fi

if [ "$IS_T4" = true ]; then
    VLLM_PKG="vllm==0.9.2"
    TRITON_PKG="triton==3.2.0"
else
    VLLM_PKG="vllm"
    TRITON_PKG="triton"
fi

echo "=== 3. 一次性统一安装（关键：全部放进同一条 uv pip install，让依赖解析器同时看到所有约束） ==="
uv pip install --system -qqq --upgrade \
    'torch<2.11.0,>=2.4.0' \
    'transformers<=5.5.0,>=4.51.3' \
    "$VLLM_PKG" \
    unsloth trl peft accelerate bitsandbytes xformers \
    torchvision numpy pillow \
    modelscope huggingface_hub wandb gpustat

uv pip install --system -qqq "$TRITON_PKG"

echo '=== 4. 禁用 HF 新版 xet 存储后端（部分 hf-mirror 节点转发 xet 会 401，详见环境文档） ==='
export HF_HUB_DISABLE_XET=1
grep -q HF_HUB_DISABLE_XET ~/.bashrc 2>/dev/null || echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc

echo '=== 5. 验证 torch CUDA 可用 ==='
python3 -c 'import torch; print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())'

echo '=== 6. 安装 flash-attn（可选，预编译wheel不匹配当前torch/cuda/python版本时会跳过，不影响主流程） ==='
uv pip install --system -qqq flash-attn --no-build-isolation \
    || echo 'flash-attn 预编译wheel不匹配，跳过（Unsloth 默认走 FlashAttention2/xformers 也能跑）'

echo '=== 全部完成 ==='
