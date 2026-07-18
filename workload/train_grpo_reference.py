"""
GRPO 训练脚本 —— JustRL 复现（实验1），单卡 Unsloth + LoRA 版本
参考：
  - JustRL 论文/训练脚本 (https://github.com/thunlp/JustRL)
  - Unsloth GRPO 教程 (https://unsloth.ai/docs/zh/kai-shi-shi-yong/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo)
  - obsidian notes: ongoing_projects/2026-07-17_llm-rl-experiments.md

与 JustRL 全量训练的关键差异（详见 notes 里的"训练模式：LoRA/QLoRA RL vs 全量 RL"）：
  - 用 LoRA 而非全量参数（RL 场景容量需求低，rank=1 起步，观察到不稳定迹象则回调到r=16）
  - 学习率是 JustRL 原始 1e-6 的约 10 倍（LoRA 学习率经验规律，但社区经验给的是1e-4~2e-4，需自行扫描对比）
  - max_prompt/completion_length 从 JustRL 的 1k/15k 大幅缩小，适配单卡 sanity check

监控：训练过程记录到 Weights & Biases（wandb），支持手机官方App查看（社区实测手机端体验远优于MLflow，
  参见notes里"训练监控工具选型"）。checkpoint 定期保存并做数量上限控制。
"""

import os
import re

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")
DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/root/rivermind-data/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/grpo_run")

# ---- Weights & Biases (wandb) 监控配置 ----
# 需要先在服务器上跑一次 `wandb login`（用你自己的 wandb API key），只用登录一次，凭证会缓存
# 手机端直接装 wandb 官方 App 登录同一账号即可远程看 reward/loss 曲线，训练异常还能收推送
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "justrl-grpo-exp1")
# run名带上关键超参，方便在 wandb 网页/App 里直接从run名区分不同sweep（rank/lr等）
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"rank{os.environ.get('LORA_RANK', '1')}_lr{os.environ.get('LEARNING_RATE', '1e-5')}",
)
os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)

# ---- checkpoint 配置 ----
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "50"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SAVE_TOTAL_LIMIT", "3"))  # 数据盘只有49G，checkpoint不能无限攒，只留最近几个

# ---- 可调超参（对应 notes 里正在验证的几个变量，全部走环境变量方便sweep） ----
LORA_RANK = int(os.environ.get("LORA_RANK", "1"))            # TODO: rank=1/16/256 三档对比
LORA_ALPHA = LORA_RANK * 2                                    # alpha = 2r（Thinking Machines 建议）
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))  # TODO: 学习率扫描 1e-5 vs 1e-4（社区经验值）
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "8"))   # 默认沿用JustRL的Rollout N=8，先不做变量
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "512"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "2048"))  # sanity check阶段先缩小，JustRL原始15360
MAX_STEPS = int(os.environ.get("MAX_STEPS", "300"))            # 教程给的sanity check参考量级
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "4"))  # 单卡先给小值
USE_FP8 = os.environ.get("USE_FP8", "0") == "1"                # 待挂卡后实测再决定是否开

# ⚠️ vLLM 的 LoRA serving kernel 对极小rank(1/2)有shape越界bug，实测崩溃（详见 hn01_llm_rl_server.md）：
# IndexError: tuple index out of range @ vllm/lora/layers/column_parallel_linear.py set_lora()
# rank<8 时自动关闭 fast_inference，走标准 HF generate() 路径（慢很多，但能跑）；rank>=8 走vLLM快速路径。
# 可用 USE_VLLM=1/0 强制覆盖这个自动判断（比如就是想对比同一rank下vLLM vs HF生成速度）。
_USE_VLLM_ENV = os.environ.get("USE_VLLM", "auto")
if _USE_VLLM_ENV == "auto":
    USE_VLLM = LORA_RANK >= 8
else:
    USE_VLLM = _USE_VLLM_ENV == "1"

# clip higher：JustRL训练脚本 clip_ratio_low=0.2/high=0.28，与Unsloth教程epsilon一致
EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28

PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def extract_boxed_answer(text: str):
    """从模型输出里提取 \\boxed{...} 内容，找不到返回 None"""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    j = i
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return text[i:j - 1].strip()


def normalize_answer(ans: str):
    if ans is None:
        return None
    return ans.strip().replace(" ", "").replace(",", "")


# ---- 用于观察 Reddit 社区提到的"奖励作弊"迹象：完成长度统计 ----
_completion_lengths = []
# ---- 格式缺失率：单独区分"没格式(\boxed{}缺失)" vs "格式对但答案错/被截断"，配合completions/clipped_ratio一起看 ----
_format_missing_flags = []


def reward_correctness(completions, answer, **kwargs):
    """答案正确 +1，格式正确但答案错 +0，完全没有\\boxed{} -1"""
    rewards = []
    for completion, gt in zip(completions, answer):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        _completion_lengths.append(len(response))
        pred = extract_boxed_answer(response)
        if pred is None:
            rewards.append(-1.0)
            _format_missing_flags.append(1.0)
        elif normalize_answer(pred) == normalize_answer(gt):
            rewards.append(1.0)
            _format_missing_flags.append(0.0)
        else:
            rewards.append(0.0)
            _format_missing_flags.append(0.0)
    return rewards


class FormatMissingRateCallback:
    """HF Trainer 的 on_log 回调：每次 trainer 往 wandb 记指标时，顺带把这批样本的格式缺失率也塞进去，
    跟 trainer 自身的 step 计数对齐，不用自己额外调用 wandb.log() 打乱 trainer 的日志节奏。"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and _format_missing_flags:
            logs["format_missing_rate"] = sum(_format_missing_flags) / len(_format_missing_flags)
            _format_missing_flags.clear()
        return control

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return kwargs.get("control")
        return noop


def reward_format(completions, **kwargs):
    """有 \\boxed{} 就给一点点格式分，鼓励模型稳定输出格式"""
    rewards = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(0.1 if "\\boxed{" in response else 0.0)
    return rewards


def main():
    # 注意：import unsloth 需要真实 GPU，无卡模式下这一步会直接报错（已在 fj01_llm_rl_server.md 记录）
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    print(f"[config] rank={LORA_RANK} alpha={LORA_ALPHA} lr={LEARNING_RATE} "
          f"num_generations={NUM_GENERATIONS} max_completion_length={MAX_COMPLETION_LENGTH} "
          f"max_steps={MAX_STEPS} fp8={USE_FP8} use_vllm={USE_VLLM} run_name={RUN_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
        load_in_4bit=False,
        load_in_fp8=USE_FP8,
        fast_inference=USE_VLLM,  # rank<8自动关闭（vLLM LoRA kernel bug），见上方说明；开启时对应notes里"GRPO瓶颈在生成"的讨论
        max_lora_rank=max(LORA_RANK, 8),
        gpu_memory_utilization=0.6,
    )

    # 全层覆盖（attention + MLP），而非只做 MLP-only —— 见 notes 里第三方复现分歧的结论
    # 如果观察到训练不稳定/输出乱码（Adapter梯度崩溃迹象，社区经验里的已知风险），
    # 优先怀疑是不是 rank 太小，把 LORA_RANK 从 1 调回 16 再试
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ---- 数据集：重新拼 prompt 成 \boxed{} 格式，与 JustRL 保持一致（而非数据集自带的 "Answer: $X" 格式）----
    dataset = load_dataset("parquet", data_files=DATA_PATH, split="train")

    def format_example(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"] + PROMPT_SUFFIX}],
            "answer": example["reward_model"]["ground_truth"],
        }

    dataset = dataset.map(format_example)

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        epsilon=EPSILON_LOW,
        epsilon_high=EPSILON_HIGH,
        temperature=1.0,
        use_vllm=USE_VLLM,
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        # ---- checkpoint 策略：数据盘只有49G，不能无限攒checkpoint ----
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,  # 只保留最近N个，自动清理旧的
        # ---- wandb 监控：把 reward/loss/entropy/kl 等指标自动上报，手机App可远程看 ----
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correctness, reward_format],
        args=training_args,
        train_dataset=dataset,
        callbacks=[FormatMissingRateCallback()],
    )

    import wandb

    wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config={
        # LoRA相关
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",  # 全层覆盖
        # 优化器/训练超参
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": 4,
        "max_steps": MAX_STEPS,
        # GRPO/rollout相关
        "num_generations": NUM_GENERATIONS,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "epsilon_low": EPSILON_LOW,
        "epsilon_high": EPSILON_HIGH,
        "temperature": 1.0,
        # 精度
        "use_fp8": USE_FP8,
        # checkpoint策略
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        # 数据/模型来源（方便跨run溯源）
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
    })

    trainer.train()

    if _completion_lengths:
        avg_len = sum(_completion_lengths) / len(_completion_lengths)
        wandb.log({"final_avg_completion_length": avg_len})
        print(f"[stats] avg completion length over training: {avg_len:.1f} tokens "
              f"(异常拉长可能是奖励作弊迹象，参考notes里的社区经验)")

    wandb.finish()

    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora"))
    print(f"[done] LoRA adapter saved to {OUTPUT_DIR}/final_lora")
    print(f"[wandb] project: {WANDB_PROJECT}, run: {RUN_NAME}")


if __name__ == "__main__":
    main()
