#!/bin/bash
# ============================================================
#  Step 2 [with GPU] —— 挂卡后入口：只做真正需要GPU在场的验证。
#  flash-attn的编译已经挪到 Step 1（编译本身不需要GPU，只需要nvcc，
#  2026-07-18 在 fj01 上验证过：无卡阶段照样能正常编译成功）。
#  跑完这一步再进 Step 3（hardware/、workload/ 下的正式基准测试）。
# ============================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./sources.sh

echo "===== verify_gpu_environment.sh ====="
bash verify_gpu_environment.sh

echo
echo "========== Step 2 [with GPU] 全部完成，环境验证通过 =========="
echo "下一步：Step 3 —— 跑正式基准测试（hardware/*.sh 和 workload/*.py）"
echo "==============================================================="
