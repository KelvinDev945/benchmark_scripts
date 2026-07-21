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
| 云端 A100-PCIE-40GB（fj02，2026-07-21） | 120.5 TFLOPS（理论312 TFLOPS的38.6%） | 686.76 GB/s（理论~1555 GB/s的44%） | 12.31 GB/s（D→H 13.15 GB/s） | errors:0，峰值温度60°C；降频比例跟同期4090/5090系列（40-47%区间）一致，再次印证云端虚拟化平台的统一限速规律 |
| （新测的卡） | | | | |

## 二、GRPO 实际工作负载测试

| GPU | generate单步耗时(s) | backward+optim单步耗时(s) | 纯vLLM吞吐(tokens/s) | base模型加载后显存(GB) | LoRA包装后显存(GB) | 训练step后峰值显存(GB) |
|---|---|---|---|---|---|---|
| RTX 4090（hn01） | 6.28 | 2-4 | 待测 | 待测 | 待测 | 待测 |
| A100-PCIE-40GB（fj02，2026-07-21，默认1024长度） | 10.81（占比90.3%） | 1.16（占比9.7%，单步总计11.97s） | 待测（下一步跑vllm_throughput_benchmark.py） | 33.69 | 33.69（几乎无变化） | 34.34（nvidia-smi真实峰值，占40GB的85.9%；vLLM standby休眠期间训练本身只占4.43-4.56GB） |
| ~~A100-PCIE-40GB（fj02，2026-07-21，sweep探测1步数据，已作废）~~ | ~~138.46~~ | ~~247.06（单步总计385.52s）~~ | — | — | — | **⚠️数据来自`SWEEP_PROBE_STEPS=1`单步探测，backward+optim混入未知的run-to-run差异（不只是编译开销，具体原因未查明），已被下一行的正式3步稳态数据取代，不要引用本行数字** |
| A100-PCIE-40GB（fj02，2026-07-21，seq4096/bs28，**正式`BENCHMARK_STEPS=3`稳态**，取代上一行） | 136.85 | **39.37**（单步总计**182.06s**；wake=0.90s sleep=4.94s） | 待测 | N/A | N/A | **39.005GB（真实峰值，占40GB的97.5%）**；对比4090同配置总耗时456.43s，**A100快2.5倍**（backward+optim快5.2倍：39.37s vs 205.65s）。⚠️两次"step1"耗时对不上（sweep探测的step1 backward=247s vs 本次复测的step1 backward=63s），怀疑跟torch.compile/Triton磁盘编译缓存复用有关，未验证，**待在4090上用同样方法论(3步稳态+清缓存)复现验证**，详见obsidian笔记 |
| A100-PCIE-40GB（fj02，2026-07-21，seq4096/bs28，**新脚本`FORCE_FULL_LENGTH=1`重跑，验证"token长度不一致"假说**） | 137.01 | 39.71（单步总计**182.32s**；wake=0.86s sleep=4.74s） | 待测 | N/A | N/A | **39.247GB(97.6%)**；跟上一行(旧脚本)几乎完全一致，差仅0.26s——**"token长度不一致"假说被排除**，bs=28场景模型本来就自然跑满长度 |
| A100-PCIE-40GB（fj02，2026-07-21，seq8192/bs8，新脚本`FORCE_FULL_LENGTH=1`，对齐4090历史配置） | 112.01 | 22.65（单步总计**138.44s**；wake=0.74s sleep=3.05s） | 待测 | N/A | N/A | **38.134GB(95.3%)**；对比4090同配置总耗时176.63s，**A100快27.6%** |
| A100-PCIE-40GB（fj02，2026-07-21，seq8192最大batch size） | — | — | — | N/A | N/A | **bs=16 单独复测成功，38.32GB(95.8%)**；⚠️`sweep_max_train_batch.py`一开始误报`max=4`（内部子进程间显存清理不彻底导致前序OOM污染后续探测，见obsidian笔记[[feedback_unsloth_vllm_standby_pitfalls]]陷阱5），sweep结果不可盲信，需人工复测验证 |
| A100-PCIE-40GB（fj02，2026-07-21，seq8192/bs16，正式3步稳态） | 208.05 | 46.23（单步总计**260.29s**；wake=0.875s sleep=5.125s） | 待测 | N/A | N/A | **39.38GB(98.5%)**；⚠️第2/3步波动大(generate差约70s)，可信度低于其他组；吞吐512样本/260.29s=1.97样本/s，比bs8的1.85样本/s略高(+6.4%)，但数据不稳定不宜过度解读 |
| A100-PCIE-40GB（fj02，2026-07-21，纯vLLM推理吞吐，`vllm_throughput_benchmark.py`，NUM_PROMPTS=8，len8192） | — | — | **1122.02 tokens/s** | N/A | N/A | 34.706GB(84.7%)，峰值利用率89%，**平均利用率35.8%**；对比4090历史1100.93 tokens/s(19.73GB)，仅快1.9%——低并发(8)下两卡都没被推到极限，测不出真实差距，详见obsidian笔记"A100纯vLLM推理吞吐测试"一节 |
| A100-PCIE-40GB（fj02，2026-07-21，seq4096/bs8，**USE_VLLM_STANDBY=0关闭**） | 48.22 | 11.88（单步总计**60.10s**，wake=0.00s sleep=0.00s） | 待测 | N/A | N/A | **36.06GB（真实峰值，占40GB的90.2%）**；关闭standby后wake/sleep计时器读数为0，验证了新增的单独打点逻辑正确 |
| A100-PCIE-40GB（fj02，2026-07-21，seq4096/bs8，**USE_VLLM_STANDBY=1开启**，同batch对照） | 55.25（比关闭standby慢7.03s） | 7.98（比关闭standby快3.9s；单步总计**67.11s**，比关闭standby慢7.01s/11.7%；wake=0.99s sleep=2.90s） | 待测 | N/A | N/A | **34.81GB（真实峰值，占40GB的87.0%，比关闭standby省1.25GB）**；**结论：bs=8时显存本来就够用(关闭standby也只到90.2%)，开standby在这个场景纯属负担——多花11.7%时间只省了1.25GB用不上的显存**；wake/sleep开销(3.89s)只能部分解释之前seq4096/bs28那组"backward变慢"的现象，扣除后backward本身standby版反而更快，说明还有其他未查明因素（可能是两次跑生成的实际token数因采样随机性不同，backward负载本身不完全一致，不是严格受控对比） |
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

## 四、bs大小 vs 吞吐效率（2026-07-21，A100初步发现，待4090验证）

按"每步样本数=train_batch_size×grad_accum×num_generations"折算吞吐（样本/秒），而不是只看单步总耗时：

| GPU | 配置 | 每步样本数 | 单步耗时 | 吞吐（样本/秒） |
|---|---|---|---|---|
| A100(fj02) | bs=28,standby开（⚠️数据来自sweep探测阶段仅1步，可能含warmup开销，非严格稳态） | 896 | 385.52s | 2.32 |
| A100(fj02) | bs=8,standby关（3步稳态） | 256 | 60.10s | 4.26 |
| A100(fj02) | bs=8,standby开（3步稳态） | 256 | 67.11s | 3.81 |
| 4090 | bs=?,standby开/关 | ? | ? | 待补（用户计划验证） |

**初步发现**：bs=8吞吐比bs=28高约1.6-1.8倍，跟"batch越大摊销效应越好"的直觉相反，推测与bs=28逼近97.5%显存上限时CUDA分配器碎片整理开销增大有关。**待办**：①给bs=28补跑正式`BENCHMARK_STEPS=3`稳态测试（不是sweep探测的1步数据）；②在4090上跑同样的bs大对比bs小实验，验证这个"大batch效率反而更低"现象是否是A100在高显存占用下特有的，还是所有卡的普遍规律。
