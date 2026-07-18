#!/bin/bash
# 用 uv 装 GRPO 训练 + 基准测试所需的 python 依赖
# 比 pip 快很多，且这里用同一条命令统一解出所有约束，避免分次 pip install 时后装的包
# 静默破坏前面的版本约束（2026-07-17 在 hn01 上踩过这个坑，详见 obsidian
# ongoing_project_related/llm-rl-experiments/环境与框架.md）
set -e

echo '=== 1. 安装/升级 uv ==='
# 有些精简镜像（比如 fj01 新实例）连 pip 都没有，不能假设 pip 一定存在。
# 优先用 uv 官方独立安装脚本（不依赖pip）；这个脚本走 GitHub Releases CDN 下载二进制，
# 部分国内网络环境下会卡住/极慢（fj01上实测卡死），设30秒超时，超时/失败就换更快的路径：
# apt(阿里云源)装pip + pip(清华源)装uv。全程目标都是尽快用上uv，pip只是一次性引导手段。
if command -v uv >/dev/null 2>&1; then
    echo "uv 已存在: $(uv --version)"
else
    if timeout 30 bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' 2>&1 | tail -5; then
        export PATH="$HOME/.local/bin:$PATH"
        grep -q '.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "官方安装脚本超时/失败，改用 apt(阿里云源)装pip + pip(清华源)装uv"
        apt-get update -qq && apt-get install -y -qq python3-pip
        python3 -m pip install -qqq uv -i https://pypi.tuna.tsinghua.edu.cn/simple
    fi
    echo "uv: $(uv --version)"
fi

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

echo '=== 3. 检查基础镜像自带的 torch 是否已满足 Unsloth 约束（torch<2.11.0,>=2.4.0） ==='
# 有些基础镜像（比如 2.9.0-cuda12.8 系列）已经预装了满足约束、且CUDA tag对得上系统驱动的torch，
# 这种情况下不应该再对 torch 用 --upgrade，否则 uv 可能把它换成约束范围内的更新版本，
# 连带引入不同 CUDA 编译版本的 torchvision/torchaudio wheel，导致CUDA版本不匹配
# （2026-07-17 在 hn01 上就是 pip install vllm 把 torch 从 cu128 悄悄换成 cu130 弄坏了 torchaudio）。
SKIP_TORCH_REINSTALL=false
if python3 -c "
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
from packaging.version import Version
v = Version(torch.__version__.split('+')[0])
sys.exit(0 if (Version('2.4.0') <= v < Version('2.11.0')) else 1)
" 2>/dev/null; then
    SKIP_TORCH_REINSTALL=true
    EXISTING_TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo "已检测到基础镜像自带 torch ${EXISTING_TORCH_VERSION}，满足约束，跳过重装以保留镜像原生 CUDA wheel 组合"
else
    echo "未检测到满足约束的 torch，将在下一步一起装"
fi

echo "=== 4. 一次性统一安装（关键：全部放进同一条 uv pip install，让依赖解析器同时看到所有约束） ==="
if [ "$SKIP_TORCH_REINSTALL" = true ]; then
    # torch 已满足约束：不传 torch 给 uv、不加 --upgrade，避免 uv 为了满足其他包的依赖声明
    # 而把已经装好、CUDA tag匹配的 torch 换掉
    uv pip install --system -qqq \
        'transformers<=5.5.0,>=4.51.3' \
        "$VLLM_PKG" \
        unsloth trl peft accelerate bitsandbytes xformers \
        torchvision numpy pillow \
        modelscope huggingface_hub wandb gpustat
else
    uv pip install --system -qqq --upgrade \
        'torch<2.11.0,>=2.4.0' \
        'transformers<=5.5.0,>=4.51.3' \
        "$VLLM_PKG" \
        unsloth trl peft accelerate bitsandbytes xformers \
        torchvision numpy pillow \
        modelscope huggingface_hub wandb gpustat
fi

uv pip install --system -qqq "$TRITON_PKG"

echo '=== 4b（附加校验）. 核实 torch 版本/CUDA tag 在装完其他包后有没有被意外换掉 ==='
python3 -c "
import torch
print('torch', torch.__version__, '| cuda build:', torch.version.cuda)
"

echo '=== 5. 禁用 HF 新版 xet 存储后端（部分 hf-mirror 节点转发 xet 会 401，详见环境文档） ==='
export HF_HUB_DISABLE_XET=1
grep -q HF_HUB_DISABLE_XET ~/.bashrc 2>/dev/null || echo 'export HF_HUB_DISABLE_XET=1' >> ~/.bashrc

echo '=== 6. 验证 torch CUDA 可用 ==='
python3 -c 'import torch; print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())'

echo '=== 7. 安装 flash-attn（可选，预编译wheel不匹配当前torch/cuda/python版本时会跳过，不影响主流程） ==='
uv pip install --system -qqq flash-attn --no-build-isolation \
    || echo 'flash-attn 预编译wheel不匹配，跳过（Unsloth 默认走 FlashAttention2/xformers 也能跑）'

echo '=== 全部完成 ==='
