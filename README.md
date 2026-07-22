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
ABI，拼出对应wheel直接下载安装。下载支持**断点续传**（固定路径+`curl -C -`），
同一个源卡住/中断不会丢弃已下载部分，多个源轮流重试，**不退回源码编译**——
2026-07-18在fj01上实测过：torch2.10时没有匹配wheel被迫源码编译，要烧30分钟以上
CPU时间，所以必须走预编译wheel这条路。

**CUDA Toolkit（含nvcc）也默认在 Step 1 装好**——虽然 flash-attn 本身不需要nvcc，
但 Step 3 的 `hardware/run_gpu_burn.sh`/`run_nvbandwidth.sh` 编译时要用，装nvcc
本身不需要GPU在场，2026-07-21在fj02上因为之前默认跳过这一步，导致编译被拖到已经
挂卡计费之后才做，浪费了GPU计费时间，所以改成默认装（`install/download_cuda_toolkit.sh`，
`INSTALL_CUDA_TOOLKIT=0` 可显式关闭）。版本号（默认12.8.0）跟 `install_python_deps.sh`
锁定的 torch cu128 保持一致，改torch版本时要联动改这里。

Step 2 因此只剩一件事——真正需要GPU硬件在场的模型加载验证。

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
│   ├── sources.sh                   # 统一分发源配置(PyPI镜像/HF_ENDPOINT/GitHub代理/ModelScope优先/CUDA_HOME自动探测/uv装到数据盘)，被其他脚本source
│   ├── step1_wo_gpu.sh               # [wo GPU] 入口：先同步跑bootstrap_uv，再并行拉起5个独立任务，最后串行跑flash-attn
│   │   ├── bootstrap_uv.sh           #   （同步，最先跑）装uv+modelscope+huggingface_hub，避免并行任务互相踩踏装uv
│   │   ├── download_model.sh         #   ┐
│   │   ├── download_dataset.sh       #   │ 五个任务互相独立，并行拉起，谁先完成
│   │   ├── clone_justrl.sh           #   │ 就算谁先完成（打在不同域名/CDN上）
│   │   ├── install_python_deps.sh    #   │ 装torch(锁2.8.x)/transformers/vllm/unsloth等（flash-attn除外）
│   │   ├── download_cuda_toolkit.sh  #   ┘ 下载+装nvcc到持久化数据盘（默认装，Step3编译gpu-burn/nvbandwidth要用）
│   │   └── install_flash_attn.sh     #   （串行，依赖install_python_deps）预编译wheel+断点续传+多源重试，不退回源码编译
│   │   └── download_data_and_code.sh #   串行兼容入口（=bootstrap_uv+download_model+download_dataset+clone_justrl依次跑），单独用时保留
│   └── step2_with_gpu.sh             # [with GPU] 入口：只做真正需要GPU在场的事
│       └── verify_gpu_environment.sh #   快速验证模型能加载+LoRA能包装（依赖系统python3-dev提供Python.h，否则Triton JIT编译会失败）
├── hardware/                         # Step 3，需要挂卡
│   ├── run_gpu_burn.sh               # Tensor Core 算力压测
│   └── run_nvbandwidth.sh            # 显存内带宽 + PCIe 带宽测试
├── workload/                         # Step 3，需要挂卡
│   ├── train_grpo_benchmark.py       # GRPO训练：耗时拆分(generate vs backward)+显存快照
│   ├── train_only_benchmark.py       # 纯训练(forward+backward+optimizer)显存/耗时快照，vLLM是否常驻可配置
│   ├── vllm_throughput_benchmark.py  # 独立于训练循环的纯vLLM推理吞吐+显存测试
│   ├── train_grpo_reference.py       # 原始生产训练脚本（未插桩），仅作对照参考
│   ├── sweep_max_train_batch.py      # 二分查找最大可用TRAIN_BATCH_SIZE，带显存护栏(默认80%显存即停)
│   ├── sweep_max_inference_length.py # 二分查找最大可用推理长度(max_new_tokens)
│   └── sweep_seqlen_train_batch.sh   # 对一组SEQ_LENGTH依次跑sweep_max_train_batch.py，复用上一轮上限加速
├── tools/
│   └── status.sh                     # 一键状态速览：GPU占用+运行中进程+最近日志尾部+最新结果json，通用(路径走DATA_DIR)
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
- Python 3.10/3.11/3.12（均已验证，`install_flash_attn.sh`会自动探测python版本拼预编译wheel文件名）。
- **系统需要 `python3-dev`（提供 `Python.h`）**——2026-07-21在fj02上发现：没装这个包时，
  Triton 运行时JIT编译CUDA wrapper会报 `fatal error: Python.h: No such file or directory`，
  导致 Step 2 的 Unsloth 模型加载直接失败。跟 uv 装的 python 包无关，是系统镜像缺开发头
  文件，装对应系统Python版本的 `python3-dev`（如 `apt-get install -y python3-dev`）即可，
  这个操作本身在挂卡前后都能做（不需要GPU），建议以后并入 Step 1。
- **非root用户 / 真机（非云端租用容器）自动切venv**——2026-07-22在真实裸机(kelvin-linux)
  上发现：非root用户+现代Debian/Ubuntu默认PEP668 "externally-managed"限制下，
  `uv pip install --system`会直接报错拒绝，即使绕过限制也会因为`dist-packages`目录
  root权限而`Permission denied`。`sources.sh`已加自动探测：能写系统site-packages（云端
  容器root场景）就用`--system`，不能写就在数据盘用`uv venv`建虚拟环境，所有脚本改用
  `$UV_PYTHON_TARGET_FLAG`变量而不是硬编码`--system`，两种环境都兼容，不需要手动配置。

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
FORCE_FULL_LENGTH=1 python3 workload/vllm_throughput_benchmark.py   # 不加这个开关，模型可能几百token内自然EOS提前停止，测不到长上下文的真实压力
```

结果记录方式见 `results/record_template.md`。

**⚠️ 两个脚本"满长度"的实现方式不一样，别搞混**：
- `vllm_throughput_benchmark.py` 靠显式开关 `FORCE_FULL_LENGTH=1`（→`ignore_eos=True`）强制生成撑满 `max_new_tokens`，不开就可能只测到模型自然EOS提前结束的那一小段，几档长度测出来是同一件事（2026-07-19在fj01上踩过这个坑）。**要跟其他卡的历史数据对比，必须带上这个开关**。
- `train_grpo_benchmark.py` 没有这个开关，但早期训练阶段（rank=1、几乎没收敛）的模型输出本身就不太会自然停止，实测 `completions/clipped_ratio` 接近1.0，等效于自然跑满长度，不需要额外开关（2026-07-21在A100上验证）。
