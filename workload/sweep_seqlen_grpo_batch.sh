#!/bin/bash
# "训练+推理同时跑"（真实GRPO耦合场景）在一组 MAX_COMPLETION_LENGTH 下依次扫最大
# TRAIN_BATCH_SIZE，复用 sweep_max_train_batch.py 的护栏+双向搜索逻辑，只是指向
# train_grpo_benchmark.py（而不是默认的 train_only_benchmark.py）。
#
# 起点优化（2026-07-19，用户要求）：每档的起点/上界直接设成"同一长度下纯训练场景
# （不带vLLM）已测出的最大batch size"向下取最近的2的幂次——因为带vLLM主动生成的耦合
# 场景，显存压力只会比纯训练更大，batch上限不会超过纯训练那个数字，没必要重新从头搜索。
#
# 用法：
#   MODEL_PATH=... OUTPUT_DIR=... GPU_TAG=rtx4090_xxx bash sweep_seqlen_grpo_batch.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_PATH=${MODEL_PATH:-/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B}
DATA_PATH=${DATA_PATH:-/root/rivermind-data/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-/root/rivermind-data/outputs/benchmark_run}
GPU_TAG=${GPU_TAG:-unknown_gpu}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}   # 固定8，对齐GRPO的rollout N，跟推理侧sweep保持一致

# 长度 -> 起点/上界：对应长度下纯训练(不带vLLM)已测出的最大batch size，向下取最近2的幂次
declare -A SEED_BATCH=( [4096]=32 [8192]=16 [16384]=8 )

for SEQ in 4096 8192 16384; do
  START=${SEED_BATCH[$SEQ]}
  echo "=================================================="
  echo "[driver] max_completion_length=${SEQ} 开始扫描最大 TRAIN_BATCH_SIZE (训练+推理同时跑，vLLM standby已开)"
  echo "=================================================="
  echo "[driver] max_completion_length=${SEQ} 搜索起点=${START} 上界=${START}（来自同长度纯训练场景已测上限的最近2的幂次，耦合场景batch上限不会超过它）"

  MODEL_PATH=${MODEL_PATH} DATA_PATH=${DATA_PATH} OUTPUT_DIR=${OUTPUT_DIR} \
    MAX_COMPLETION_LENGTH=${SEQ} NUM_GENERATIONS=${NUM_GENERATIONS} \
    TARGET_SCRIPT=train_grpo_benchmark.py RESULT_FILE_PREFIX=benchmark_summary \
    GPU_TAG=${GPU_TAG}_grpo_seq${SEQ} SWEEP_START=${START} SWEEP_MAX=${START} \
    python3 -u workload/sweep_max_train_batch.py

  MAXBS=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/sweep_max_train_batch_${GPU_TAG}_grpo_seq${SEQ}.json'))['max_working_batch_size'])")
  echo "[driver] max_completion_length=${SEQ} 最大可用 TRAIN_BATCH_SIZE = ${MAXBS}"

  if [ "${MAXBS}" -gt 0 ]; then
    echo "[driver] max_completion_length=${SEQ} 用完整步数(BENCHMARK_STEPS=10默认)在 bs=${MAXBS} 上重跑一次记录准确耗时/显存"
    MODEL_PATH=${MODEL_PATH} DATA_PATH=${DATA_PATH} OUTPUT_DIR=${OUTPUT_DIR} \
      MAX_COMPLETION_LENGTH=${SEQ} NUM_GENERATIONS=${NUM_GENERATIONS} TRAIN_BATCH_SIZE=${MAXBS} \
      GPU_TAG=${GPU_TAG}_grpo_final_seq${SEQ}_bs${MAXBS} \
      python3 -u workload/train_grpo_benchmark.py
  else
    echo "[driver] max_completion_length=${SEQ} 连 batch_size=1 都装不下，跳过最终确认跑"
  fi
done

echo "[driver] ALL_DONE"
