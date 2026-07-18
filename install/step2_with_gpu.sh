#!/bin/bash
# ============================================================
#  Step 2 [with GPU] —— 挂卡后入口：按顺序跑完下面这些需要GPU/nvcc的脚本。
#  跑完再进 Step 3（hardware/、workload/ 下的正式基准测试）。
# ============================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./sources.sh

echo "===== [1/2] install_flash_attn.sh ====="
bash install_flash_attn.sh

echo "===== [2/2] verify_gpu_environment.sh ====="
bash verify_gpu_environment.sh

echo
echo "========== Step 2 [with GPU] 全部完成，环境验证通过 =========="
echo "下一步：Step 3 —— 跑正式基准测试（hardware/*.sh 和 workload/*.py）"
echo "==============================================================="
