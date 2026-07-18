#!/bin/bash
# ============================================================
#  Step 1 [wo GPU] —— 无卡（CPU）阶段入口：按顺序跑完下面这些不需要GPU的脚本。
#  跑完直接挂卡，进入 Step 2。
# ============================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "===== [1/4] download_data_and_code.sh ====="
bash download_data_and_code.sh

echo "===== [2/4] install_python_deps.sh（torch锁定在2.8.x，让flash-attn能用预编译wheel） ====="
bash install_python_deps.sh

echo "===== [3/4] download_cuda_toolkit.sh（默认跳过——flash-attn现在用预编译wheel不需要nvcc了） ====="
bash download_cuda_toolkit.sh

echo "===== [4/4] install_flash_attn.sh（优先装预编译wheel，几秒钟完事；没有匹配wheel才退回源码编译） ====="
bash install_flash_attn.sh

echo
echo "========== Step 1 [wo GPU] 全部完成 =========="
echo "下一步：挂卡后运行 Step 2 —— bash install/step2_with_gpu.sh"
echo "==============================================="
