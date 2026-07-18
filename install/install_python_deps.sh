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
    VLLM_PKG="vllm"; TRITON_PKG="triton"
fi

# 有些基础镜像（比如 2.9.0-cuda12.8 系列）已经预装了满足Unsloth约束(torch<2.11.0,>=2.4.0)、
# 且CUDA tag对得上驱动的torch——这种情况不加 --upgrade，避免uv为了其他包的依赖声明
# 把已经装好、CUDA tag匹配的torch意外换掉（2026-07-17 hn01上 pip install vllm 就是这么把
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
sys.exit(0 if (Version('2.4.0') <= v < Version('2.11.0')) else 1)
" 2>/dev/null; then
    SKIP_TORCH_REINSTALL=true
    echo "基础镜像自带 torch $(python3 -c 'import torch; print(torch.__version__)') 已满足约束，跳过重装"
fi

# 全部放进同一条 uv pip install，让依赖解析器同时看到所有约束，避免分次装时后装的包
# 静默破坏前面的版本约束
if [ "$SKIP_TORCH_REINSTALL" = true ]; then
    uv pip install --system -qqq -i "$PYPI_MIRROR" \
        'transformers<=5.5.0,>=4.51.3' "$VLLM_PKG" \
        unsloth trl peft accelerate bitsandbytes xformers torchvision numpy pillow \
        modelscope huggingface_hub wandb gpustat
else
    uv pip install --system -qqq -i "$PYPI_MIRROR" --upgrade \
        'torch<2.11.0,>=2.4.0' 'transformers<=5.5.0,>=4.51.3' "$VLLM_PKG" \
        unsloth trl peft accelerate bitsandbytes xformers torchvision numpy pillow \
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
