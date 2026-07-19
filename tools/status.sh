#!/bin/bash
# 一键状态速览：GPU占用 + 正在跑的benchmark进程 + 最近改动的日志尾部 + 最新的结果json文件。
# 设计成通用脚本随 benchmark_scripts 仓库一起clone到任何GPU实例上，不写死某台服务器的
# 路径/主机名——所有路径都从 DATA_DIR（跟仓库里其他脚本同一套约定）推导。
#
# 用法（在GPU实例上直接跑，或通过本地的 remote_status.sh 从本机一行调用）：
#   bash tools/status.sh                  # 默认看 $DATA_DIR 下最近改动的一个 *.log
#   bash tools/status.sh /path/to/xxx.log # 指定看哪个日志
set -uo pipefail

DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/outputs/benchmark_run}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-15}"
TARGET_LOG="${1:-}"

echo "===== GPU 状态 ====="
if command -v gpustat >/dev/null 2>&1; then
  gpustat --no-header 2>/dev/null || nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
else
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
fi

echo ""
echo "===== 正在跑的 benchmark 进程 ====="
pgrep -af 'sweep_max_train_batch|sweep_max_inference_length|train_only_benchmark|train_grpo_benchmark|vllm_throughput_benchmark|sweep_seqlen_train_batch|driver' \
  | grep -v 'pgrep -af' || echo "(没有匹配到正在跑的benchmark相关进程)"

echo ""
echo "===== 最近改动的日志 ====="
if [ -z "${TARGET_LOG}" ]; then
  TARGET_LOG=$(find "${DATA_DIR}" -maxdepth 1 -iname '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [ -n "${TARGET_LOG}" ] && [ -f "${TARGET_LOG}" ]; then
  echo "(${TARGET_LOG})"
  tail -n "${LOG_TAIL_LINES}" "${TARGET_LOG}"
else
  echo "(没找到日志文件，用 tools/status.sh /path/to/xxx.log 指定一个)"
fi

echo ""
echo "===== 最新的结果 json（按修改时间，最近5个） ====="
find "${OUTPUT_DIR}" -iname '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -5 | cut -d' ' -f2- \
  || echo "(没找到 ${OUTPUT_DIR})"
