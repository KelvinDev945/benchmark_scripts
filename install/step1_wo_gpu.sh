#!/bin/bash
# ============================================================
#  Step 1 [wo GPU] —— 无卡（CPU）阶段入口：按顺序跑完下面这些不需要GPU的脚本。
#  跑完直接挂卡，进入 Step 2。
# ============================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "===== [1/3] download_data_and_code.sh ====="
bash download_data_and_code.sh

echo "===== [2/3] install_python_deps.sh ====="
bash install_python_deps.sh

echo "===== [3/3] download_cuda_toolkit.sh（默认装nvcc，INSTALL_CUDA_TOOLKIT=0 可跳过） ====="
bash download_cuda_toolkit.sh

echo
echo "========== Step 1 [wo GPU] 全部完成 =========="
echo "下一步：挂卡后运行 Step 2 —— bash install/step2_with_gpu.sh"
echo "==============================================="
