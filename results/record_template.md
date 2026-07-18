# 多卡基准测试结果记录

> 对应 obsidian `ongoing_project_related/llm-rl-experiments/训练方法论与实验记录.md` 里
> "实验：多卡训练/推理速度基准测试"。每测完一张卡，把数字填进下面两张表，
> 并把 `results/` 目录下产出的 json（`benchmark_summary_*.json` / `vllm_throughput_*.json` /
> gpu_burn、nvbandwidth log）一并存进这个目录，文件名带上 GPU_TAG 区分。

## 一、硬件基础测试

| GPU | Tensor Core算力(FP16, gpu-burn) | 显存内带宽(nvbandwidth) | PCIe H→D | 备注 |
|---|---|---|---|---|
| 云端 4090 Lite | 138.5 TFLOPS（84%） | 459 GB/s（45%） | 6.26 GB/s | 已有数据 |
| 云端 4090 | 139.7 TFLOPS | 459 GB/s（45%） | 25.90 GB/s | 已有数据 |
| 真实 RTX 4090 | 165 TFLOPS | 1008 GB/s | ~32 GB/s | 已有数据 |
| 云端 RTX 5090 | 111.8 TFLOPS ⚠️见WMMA说明 | 761 GB/s（42%） | 12.64 GB/s | 已有数据 |
| 真实 RTX 5090 | ~314 TFLOPS | 1792 GB/s | ~128 GB/s | 已有数据 |
| （新测的卡） | | | | |

## 二、GRPO 实际工作负载测试

| GPU | generate单步耗时(s) | backward+optim单步耗时(s) | 纯vLLM吞吐(tokens/s) | base模型加载后显存(GB) | LoRA包装后显存(GB) | 训练step后峰值显存(GB) |
|---|---|---|---|---|---|---|
| RTX 4090（hn01） | 6.28 | 2-4 | 待测 | 待测 | 待测 | 待测 |
| （新测的卡） | | | | | | |

## 三、跑法（复制粘贴用）

> `DATA_DIR=/root/rivermind-data` 是持久化数据盘，所有产出（模型/数据集/工具/结果）都落在
> 这里面，不会因为实例释放/重启丢失（详见持久记忆 feedback_gpu_rental_persistent_data_disk）。

```bash
export DATA_DIR="/root/rivermind-data"
export GPU_TAG="rtx3090"   # 每张卡换一下这个标签
export MODEL_PATH="$DATA_DIR/models/DeepSeek-R1-Distill-Qwen-1.5B"
export DATA_PATH="$DATA_DIR/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet"
export OUTPUT_DIR="$DATA_DIR/outputs/benchmark_run"

# 1. 硬件基础测试（结果自动存到 $DATA_DIR/benchmark_results/）
bash hardware/run_gpu_burn.sh 180
bash hardware/run_nvbandwidth.sh

# 2. GRPO 训练+推理耗时/显存
python3 workload/train_grpo_benchmark.py

# 3. 纯 vLLM 推理吞吐
python3 workload/vllm_throughput_benchmark.py
```

跑完把 `$OUTPUT_DIR/benchmark_summary_$GPU_TAG.json`、`$OUTPUT_DIR/vllm_throughput_$GPU_TAG.json`
和 `$DATA_DIR/benchmark_results/gpu_burn_result_*.log`、`$DATA_DIR/benchmark_results/nvbandwidth_result_*.log`
scp 回本地存进这个 `results/` 目录。
