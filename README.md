# benchmark_scripts

多卡（3090/4090/5090/A100等）训练速度 + 推理速度基准测试脚本，服务于 LLM RL（GRPO）项目
的"找最佳单卡配置 + 评估便宜卡专职推理/主力卡专职训练分工方案"这个目标。

背景/完整实验设计见 obsidian 笔记：
`ongoing_project_related/llm-rl-experiments/训练方法论与实验记录.md`
里的"实验：多卡训练/推理速度基准测试"一节。

## 目录结构

```
benchmark_scripts/
├── install/
│   ├── sources.sh                   # 统一分发源配置(PyPI镜像/HF_ENDPOINT/GitHub代理)，被下面两个脚本source
│   ├── download_data_and_code.sh    # 下载base模型/训练数据集/JustRL评测代码 —— 无卡阶段
│   └── install_deps_uv.sh           # 用uv装torch/transformers/vllm/unsloth等 —— 也是无卡阶段
├── hardware/                        # 以下全部需要挂卡
│   ├── run_gpu_burn.sh              # Tensor Core 算力压测
│   └── run_nvbandwidth.sh           # 显存内带宽 + PCIe 带宽测试
├── workload/
│   ├── verify_gpu_environment.py    # 挂卡后先跑这个：快速验证模型能加载，几十秒出结果
│   ├── train_grpo_benchmark.py      # GRPO训练：耗时拆分(generate vs backward)+显存快照
│   ├── vllm_throughput_benchmark.py # 独立于训练循环的纯vLLM推理吞吐测试
│   └── train_grpo_reference.py      # 原始生产训练脚本（未插桩），仅作对照参考
└── results/
    └── record_template.md           # 结果记录表 + 跑法速查
```

## 两阶段设计（省GPU计费时间）

无卡（CPU-only）启动通常比挂卡便宜很多，`install/` 下的两个脚本**都不需要真实GPU**——
下载模型/数据集/代码是纯IO，装python依赖包也只是解析依赖+下载wheel，跟有没有插卡无关。
所以这两步应该在无卡阶段做完，挂卡后直接从 `verify_gpu_environment.py` 开始，
省下的挂卡时间就是省下来的钱。

## 环境要求

- Unsloth 当前最新版本（2026.7.3）声明的依赖约束：`torch<2.11.0,>=2.4.0`、
  `transformers<=5.5.0,>=4.51.3`——`install_deps_uv.sh` 已按这个约束一次性装好，
  不要分多次单独 `pip/uv install` 某个包，否则依赖解析器看不到全局约束，
  容易把 torch/transformers 静默升级到不兼容版本（2026-07-17 在 hn01 上踩过这个坑）。
- Python 3.12（已验证）。

## 快速开始

```bash
# ===== 阶段一：无卡(CPU)启动，先做完，省挂卡计费时间 =====
export DATA_DIR="/root/rivermind-data"
bash install/download_data_and_code.sh   # 下载模型/数据集/JustRL代码
bash install/install_deps_uv.sh          # 装torch/transformers/vllm/unsloth等python依赖

# ===== 挂卡，开始计费 =====

# ===== 阶段二：挂卡后 =====
export GPU_TAG="rtx4090"   # 每台机器/每张卡换一下这个标签，用于区分产出文件
export MODEL_PATH="/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B"
export DATA_PATH="/root/rivermind-data/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet"

python3 workload/verify_gpu_environment.py  # 先快速验证环境没问题，几十秒出结果
bash hardware/run_gpu_burn.sh 180
bash hardware/run_nvbandwidth.sh
python3 workload/train_grpo_benchmark.py
python3 workload/vllm_throughput_benchmark.py
```

结果记录方式见 `results/record_template.md`。
