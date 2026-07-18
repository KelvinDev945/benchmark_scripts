# benchmark_scripts

多卡（3090/4090/5090/A100等）训练速度 + 推理速度基准测试脚本，服务于 LLM RL（GRPO）项目
的"找最佳单卡配置 + 评估便宜卡专职推理/主力卡专职训练分工方案"这个目标。

背景/完整实验设计见 obsidian 笔记：
`ongoing_project_related/llm-rl-experiments/训练方法论与实验记录.md`
里的"实验：多卡训练/推理速度基准测试"一节。

## 三步流程

| 步骤 | 需要GPU？ | 入口脚本 | 做什么 |
|---|---|---|---|
| Step 1 | wo GPU | `install/step1_wo_gpu.sh` | 下载模型/数据集/JustRL代码 + 装torch(锁2.8.x)/transformers/vllm/unsloth等python依赖 + 装flash-attn(预编译wheel) |
| Step 2 | with GPU | `install/step2_with_gpu.sh` | 快速验证模型能加载+LoRA能包装 |
| Step 3 | with GPU | `hardware/*.sh` + `workload/*.py` | 正式基准测试：硬件跑分 + GRPO训练/推理耗时+显存+吞吐 |

Step 1 建议在**无卡（CPU-only）启动**阶段做完——下载是纯IO，装python依赖包也只是解析
依赖+下载wheel，都不需要真实GPU，无卡启动通常比挂卡便宜很多。挂卡开始计费后直接从
Step 2 开始，省下来的无卡等待时间就是省下来的钱。

**flash-attn 也在 Step 1 里装好**——`install_python_deps.sh` 把 torch 锁定在
`2.8.x`，正好命中 flash-attn 官方 GitHub Releases 的预编译wheel矩阵覆盖范围
（`cu12+torch2.8`），`install_flash_attn.sh` 会自动探测 torch/python版本 + C++11
ABI，拼出对应wheel直接下载安装，**几秒钟装完，完全不用编译**。只有在torch版本超出
wheel矩阵覆盖范围时才会退回源码编译（这种情况下才需要nvcc，可以用
`install/download_cuda_toolkit.sh`手动装，默认关闭，`INSTALL_CUDA_TOOLKIT=1`打开——
2026-07-18在fj01上实测过：torch2.10时没有匹配wheel，编译flash-attn要烧30分钟以上
CPU时间，所以正常情况下都应该走预编译wheel这条路，不需要装nvcc）。Step 2 因此只剩
一件事——真正需要GPU硬件在场的模型加载验证。

## ⚠️ 数据保存位置：`/root/rivermind-data`

这是**持久化数据盘**，不是根分区——租用的GPU实例（智川云等）根分区/`/tmp`/`$HOME`都是临时的，
实例释放或重启就会清空，只有挂载的数据盘内容会保留。**模型、数据集、JustRL代码、python依赖的
下载缓存、训练/基准测试的所有输出、gpu-burn/nvbandwidth这类工具的clone+编译产物**，全部必须
落在 `$DATA_DIR`（默认 `/root/rivermind-data`）下面，不要写到 `$HOME/github`、`/tmp` 或其他
根分区路径。所有脚本的默认值都已经这样设置，改配置时也要遵守这条。

## 目录结构

```
benchmark_scripts/
├── install/
│   ├── sources.sh                   # 统一分发源配置(PyPI镜像/HF_ENDPOINT/GitHub代理/ModelScope优先/CUDA_HOME自动探测)，被其他脚本source
│   ├── step1_wo_gpu.sh               # [wo GPU] 入口：顺序调用下面四个脚本
│   │   ├── download_data_and_code.sh #   下载base模型/数据集/JustRL代码
│   │   ├── install_python_deps.sh    #   装torch(锁2.8.x)/transformers/vllm/unsloth等（flash-attn除外）
│   │   ├── download_cuda_toolkit.sh  #   下载+装nvcc到持久化数据盘（默认关闭，只有flash-attn要编译时才需要）
│   │   └── install_flash_attn.sh     #   优先装预编译wheel(几秒钟)，没有匹配wheel才退回源码编译
│   └── step2_with_gpu.sh             # [with GPU] 入口：只做真正需要GPU在场的事
│       └── verify_gpu_environment.sh #   快速验证模型能加载+LoRA能包装
├── hardware/                         # Step 3，需要挂卡
│   ├── run_gpu_burn.sh               # Tensor Core 算力压测
│   └── run_nvbandwidth.sh            # 显存内带宽 + PCIe 带宽测试
├── workload/                         # Step 3，需要挂卡
│   ├── train_grpo_benchmark.py       # GRPO训练：耗时拆分(generate vs backward)+显存快照
│   ├── vllm_throughput_benchmark.py  # 独立于训练循环的纯vLLM推理吞吐测试
│   └── train_grpo_reference.py       # 原始生产训练脚本（未插桩），仅作对照参考
└── results/
    └── record_template.md            # 结果记录表 + 跑法速查
```

## 环境要求

- Unsloth 当前最新版本（2026.7.3）声明的依赖约束是 `torch<2.11.0,>=2.4.0`，但
  `install_python_deps.sh` 实际把 torch 锁定在更窄的 **`torch<2.9.0,>=2.8.0`**——
  原因是 flash-attn 官方预编译wheel矩阵目前只覆盖到 `cu12+torch2.8`（torch2.9只有
  cu13的wheel，torch2.10完全没有），锁在2.8.x能让 `install_flash_attn.sh` 直接装
  预编译wheel（几秒钟），不用从源码编译（2026-07-18在fj01上实测：torch2.10时
  flash-attn编译耗时30分钟以上）。基础镜像自带的torch如果已经在这个区间内会跳过重装。
  不要分多次单独 `pip/uv install` 某个包，否则依赖解析器看不到全局约束，容易把
  torch/transformers 静默升级到不兼容版本（2026-07-17 在 hn01 上踩过这个坑）。
- Python 3.11/3.12（均已验证，`install_flash_attn.sh`会自动探测python版本拼预编译wheel文件名）。

## 快速开始

```bash
export DATA_DIR="/root/rivermind-data"

# ===== Step 1 [wo GPU]：无卡阶段做完，省钱（含数据/依赖/nvcc/flash-attn编译） =====
bash install/step1_wo_gpu.sh

# ===== 挂卡，开始计费 =====

# ===== Step 2 [with GPU]：只做真正需要GPU的验证 =====
export MODEL_PATH="/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B"
bash install/step2_with_gpu.sh

# ===== Step 3：正式基准测试 =====
export GPU_TAG="rtx4090"   # 每台机器/每张卡换一下这个标签，用于区分产出文件
export DATA_PATH="/root/rivermind-data/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet"

bash hardware/run_gpu_burn.sh 180
bash hardware/run_nvbandwidth.sh
python3 workload/train_grpo_benchmark.py
python3 workload/vllm_throughput_benchmark.py
```

结果记录方式见 `results/record_template.md`。
