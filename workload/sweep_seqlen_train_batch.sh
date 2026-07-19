#!/bin/bash
# 对一组 SEQ_LENGTH 依次跑 sweep_max_train_batch.py，找"单独训练"场景（LOAD_VLLM_ENGINE
# 由调用方决定）在每个目标序列长度下最大能撑的 TRAIN_BATCH_SIZE，然后在该 batch size 上
# 用完整步数(BENCHMARK_STEPS=3)重跑一次 train_only_benchmark.py 记录准确耗时/显存。
#
# 更长的序列长度支持的最大batch size不会超过更短长度的结果（同样的显存预算，序列越长
# 每条样本占用越多），所以每一轮把上一轮测出的上限直接当作这一轮的搜索起点+上界，
# 避免重新从1开始翻倍探测。
#
# 用法：
#   MODEL_PATH=... OUTPUT_DIR=... GPU_TAG=rtx4090_xxx LOAD_VLLM_ENGINE=0 \
#     SEQ_LENGTHS="4096 8192 16384" bash sweep_seqlen_train_batch.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL_PATH=${MODEL_PATH:-/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B}
OUTPUT_DIR=${OUTPUT_DIR:-/root/rivermind-data/outputs/benchmark_run}
GPU_TAG=${GPU_TAG:-unknown_gpu}
LOAD_VLLM_ENGINE=${LOAD_VLLM_ENGINE:-0}
SEQ_LENGTHS=${SEQ_LENGTHS:-"4096 8192 16384"}
FIRST_SWEEP_START=${FIRST_SWEEP_START:-8}
FIRST_SWEEP_MAX=${FIRST_SWEEP_MAX:-256}

PREV_MAX=""   # 上一个（更短）长度测出的上限

for SEQ in ${SEQ_LENGTHS}; do
  echo "=================================================="
  echo "[driver] seq_length=${SEQ} 开始扫描最大 batch size (LOAD_VLLM_ENGINE=${LOAD_VLLM_ENGINE})"
  echo "=================================================="

  if [ -z "${PREV_MAX}" ]; then
    START=${FIRST_SWEEP_START}
    MAX=${FIRST_SWEEP_MAX}
  else
    if [ "${PREV_MAX}" -eq 0 ]; then
      echo "[driver] 上一个长度batch_size=1都已经OOM，更长的长度必然更差，跳过整轮"
      continue
    fi
    START=${PREV_MAX}
    MAX=${PREV_MAX}
  fi

  echo "[driver] seq_length=${SEQ} 搜索起点=${START} 上界=${MAX}（上界来自更短长度的已测上限，更长的不会超过它）"

  SEQ_LENGTH=${SEQ} MODEL_PATH=${MODEL_PATH} OUTPUT_DIR=${OUTPUT_DIR} LOAD_VLLM_ENGINE=${LOAD_VLLM_ENGINE} \
    GPU_TAG=${GPU_TAG}_seq${SEQ} SWEEP_START=${START} SWEEP_MAX=${MAX} \
    python3 -u workload/sweep_max_train_batch.py

  MAXBS=$(python3 -c "import json; print(json.load(open('${OUTPUT_DIR}/sweep_max_train_batch_${GPU_TAG}_seq${SEQ}.json'))['max_working_batch_size'])")
  echo "[driver] seq_length=${SEQ} 最大可用 batch size = ${MAXBS}"
  PREV_MAX=${MAXBS}

  if [ "${MAXBS}" -gt 0 ]; then
    echo "[driver] seq_length=${SEQ} 用完整步数(BENCHMARK_STEPS=3)在 bs=${MAXBS} 上重跑一次记录准确耗时/显存"
    SEQ_LENGTH=${SEQ} MODEL_PATH=${MODEL_PATH} OUTPUT_DIR=${OUTPUT_DIR} LOAD_VLLM_ENGINE=${LOAD_VLLM_ENGINE} \
      GPU_TAG=${GPU_TAG}_final_seq${SEQ}_bs${MAXBS} TRAIN_BATCH_SIZE=${MAXBS} BENCHMARK_STEPS=3 \
      python3 -u workload/train_only_benchmark.py
  else
    echo "[driver] seq_length=${SEQ} 连 batch_size=1 都装不下，跳过最终确认跑"
  fi
done

echo "[driver] ALL_DONE"
