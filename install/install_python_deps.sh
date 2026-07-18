#!/bin/bash
# [wo GPU] 装 torch/transformers/vllm/unsloth 等 python 依赖（flash-attn 除外，见 install_flash_attn.sh）。
# 只是解析依赖+下载wheel，不需要GPU。被 step1_wo_gpu.sh 调用，也可以单独跑。
set -e

source "$(dirname "${BASH_SOURCE[0]}")/sources.sh"

IS_T4=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | grep -q "Tesla T4"; then
    IS_T4=true
fi
if [ "$IS_T4" = true ]; then
    VLLM_PKG="vllm==0.9.2"; TRITON_PKG="triton==3.2.0"
else
    # vllm==0.19.1（之前验证过的版本）实际硬性要求 torch==2.10.0（不是范围，是精确
    # 版本锁定），跟下面torch锁定在2.8.x这个新约束直接冲突。改用vllm==0.11.0——
    # 2026-07-18在fj01上用dry-run验证过：torch2.8.x + transformers<=5.5.0 +
    # unsloth==2026.7.3 + vllm==0.11.0 能干净解析，没有回退到荒谬古董版本的问题。
    VLLM_PKG="vllm==0.11.0"; TRITON_PKG="triton"
fi
# unsloth 同理需要钉死具体版本，不能留空——2026-07-18 在 fj01 上踩过坑：当基础镜像没有
# 预装满足约束的torch时（见下面SKIP_TORCH_REINSTALL逻辑），uv要真正resolve torch，
# 一旦torch被约束在某个范围内，而vllm/unsloth不带版本号，解析器为了同时满足所有约束，
# 会一路回退到离谱的古董版本（实测碰到过 vllm==0.2.5，2023年的版本，跟现在这套依赖完全
# 不兼容，装的时候尝试从源码编译直接崩溃；unsloth不带版本号时也回退到了2024.8）。
# 这里钉死到实测能干净解析的组合，阻止解析器做这种荒谬的回退。
UNSLOTH_PKG="unsloth==2026.7.3"

# torch 目标锁定在 2.8.x（而不是Unsloth约束允许的整个<2.11.0区间）——原因：
# flash-attn 官方预编译wheel矩阵（GitHub Releases）目前只覆盖到 cu12+torch2.8，
# torch2.9+ 只有cu13的wheel、torch2.10完全没有对应wheel。2026-07-18在fj01上实测：
# 装torch2.10后flash-attn没有匹配wheel，被迫从源码编译，光编译就要烧半小时以上
# CPU时间（大量模板化CUDA kernel）。锁定2.8.x能让 install_flash_attn.sh 直接装
# 预编译wheel，几秒钟装完，不用再编译。
TORCH_CONSTRAINT='torch<2.9.0,>=2.8.0'

# 有些基础镜像（比如 2.8.0-cuda12.8 系列）已经预装了满足这个约束、且CUDA tag对得上
# 驱动的torch——这种情况不加 --upgrade，避免uv为了其他包的依赖声明把已经装好、
# CUDA tag匹配的torch意外换掉（2026-07-17 hn01上 pip install vllm 就是这么把
# torch从cu128换成cu130、弄坏torchaudio的）
SKIP_TORCH_REINSTALL=false
if python3 -c "
import sys
try:
    import torch
except ImportError:
    sys.exit(1)
from packaging.version import Version
v = Version(torch.__version__.split('+')[0])
sys.exit(0 if (Version('2.8.0') <= v < Version('2.9.0')) else 1)
" 2>/dev/null; then
    SKIP_TORCH_REINSTALL=true
    echo "基础镜像自带 torch $(python3 -c 'import torch; print(torch.__version__)') 已满足约束，跳过重装"
fi

# 全部放进同一条 uv pip install，让依赖解析器同时看到所有约束，避免分次装时后装的包
# 静默破坏前面的版本约束
if [ "$SKIP_TORCH_REINSTALL" = true ]; then
    uv pip install --system -qqq -i "$PYPI_MIRROR" \
        'transformers<=5.5.0,>=4.51.3' "$VLLM_PKG" "$UNSLOTH_PKG" \
        trl peft accelerate bitsandbytes xformers torchvision numpy pillow \
        modelscope huggingface_hub wandb gpustat
else
    uv pip install --system -qqq -i "$PYPI_MIRROR" --upgrade \
        "$TORCH_CONSTRAINT" 'transformers<=5.5.0,>=4.51.3' "$VLLM_PKG" "$UNSLOTH_PKG" \
        trl peft accelerate bitsandbytes xformers torchvision numpy pillow \
        modelscope huggingface_hub wandb gpustat
fi
uv pip install --system -qqq -i "$PYPI_MIRROR" "$TRITON_PKG"

echo "[install_python_deps] 完成："
python3 -c "
import torch
print(f'torch:        {torch.__version__} (cuda build: {torch.version.cuda})')
try:
    import transformers; print(f'transformers: {transformers.__version__}')
except Exception as e:
    print(f'transformers: 警告 导入失败 ({e})')
try:
    import vllm; print(f'vllm:         {vllm.__version__}')
except Exception as e:
    print(f'vllm:         警告 导入失败 ({e})')
"
