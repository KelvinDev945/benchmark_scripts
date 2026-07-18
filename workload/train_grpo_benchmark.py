"""
GRPO 训练侧 + 推理侧（vLLM）耗时/显存基准测试脚本
基于 gpu_rent/train_grpo.py（实验1a配置）改造，用于"多卡训练/推理速度基准测试"实验
（详见 obsidian ongoing_project_related/llm-rl-experiments/训练方法论与实验记录.md）

新增内容（相对原始 train_grpo.py）：
  1. 生命周期关键节点的显存快照（base模型加载后 / LoRA包装后 / vLLM引擎初始化后 / 首个训练step后）
  2. 每步耗时拆分：vLLM生成(rollout) vs forward+backward+optimizer step，分别计时
  3. 只跑少量步数（默认10步），跑完打印一份可直接抄进对比表的汇总

用法：
  MODEL_PATH=... DATA_PATH=... BENCHMARK_STEPS=10 python3 train_grpo_benchmark.py
"""

import os
import time
import json

# 默认路径全部落在 /root/rivermind-data —— 这是持久化数据盘，不是根分区，
# 实例释放/重启数据不会丢。不要改成根分区下的路径（详见持久记忆
# feedback_gpu_rental_persistent_data_disk / obsidian 环境与框架.md）
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")
DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/root/rivermind-data/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/benchmark_run")

LORA_RANK = int(os.environ.get("LORA_RANK", "1"))
LORA_ALPHA = LORA_RANK * 2
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "8"))
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "512"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "1024"))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
USE_FP8 = os.environ.get("USE_FP8", "0") == "1"
BENCHMARK_STEPS = int(os.environ.get("BENCHMARK_STEPS", "10"))

# 早期在 hn01 上发现 rank<8 时 vLLM 的 LoRA serving kernel 会崩溃（IndexError @
# column_parallel_linear.py set_lora()），当时用"rank<8就关vLLM"来规避。后来查明根因其实是
# torch/transformers 版本超出 Unsloth 支持范围（不是rank本身的问题），重装成正确版本组合
# （torch<2.11.0 + transformers<=5.5.0，跟 install_python_deps.sh 锁定的约束一致）后，
# rank=1 + vLLM快速路径 5步实测全部跑通，无崩溃。所以默认rank=1也默认开vLLM，
# 不再对低rank做保守降级；如果环境版本组合有问题需要临时规避，用 USE_VLLM=0 强制关闭。
USE_VLLM = os.environ.get("USE_VLLM", "1") == "1"

EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28
PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."

GPU_TAG = os.environ.get("GPU_TAG", "unknown_gpu")  # 跑的时候标一下当前是哪张卡，方便结果归档

# ---- 显存快照工具 ----
_memory_snapshots = {}


def snapshot_memory(tag):
    import torch
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    _memory_snapshots[tag] = {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "max_allocated_gb": round(max_allocated, 3),
    }
    print(f"[memory] {tag}: allocated={allocated:.2f}GB reserved={reserved:.2f}GB "
          f"max_allocated={max_allocated:.2f}GB")


# ---- 每步耗时拆分：猴子补丁 GRPOTrainer 的生成方法，单独计时 ----
_step_timings = []  # 每个元素: {"generate_s": float, "total_s": float}
_current_generate_time = [0.0]


def patch_trainer_timing(trainer):
    """
    尝试给 TRL GRPOTrainer 生成 rollout 的内部方法打点计时。
    不同 TRL 版本方法名可能不同，找不到就退化成只记录整步耗时（不拆分）。
    """
    import functools

    candidate_method_names = [
        "_generate_and_score_completions",  # 较新版本 TRL
        "_generate_completions",
        "_prepare_inputs",
    ]
    patched = False
    for name in candidate_method_names:
        if hasattr(trainer, name):
            orig = getattr(trainer, name)

            @functools.wraps(orig)
            def wrapped(*args, __orig=orig, **kwargs):
                import torch
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = __orig(*args, **kwargs)
                torch.cuda.synchronize()
                _current_generate_time[0] = time.perf_counter() - t0
                return result

            setattr(trainer, name, wrapped)
            print(f"[timing] 已对 GRPOTrainer.{name} 打点计时（生成/rollout阶段）")
            patched = True
            break
    if not patched:
        print("[timing][警告] 没找到已知的生成方法名，只能记录整步总耗时，无法拆分generate vs backward。"
              "需要根据实际安装的 trl 版本手动确认方法名（grep trl/trainer/grpo_trainer.py）")
    return patched


class StepTimingCallback:
    """记录每个 step 的总耗时，配合上面的猴子补丁拆出 generate 和 forward+backward 各自占比"""

    def __init__(self):
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        import torch
        torch.cuda.synchronize()
        self._step_start = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        import torch
        torch.cuda.synchronize()
        total = time.perf_counter() - self._step_start
        generate_s = _current_generate_time[0]
        backward_s = max(total - generate_s, 0.0)
        _step_timings.append({"step": state.global_step, "total_s": total,
                               "generate_s": generate_s, "backward_s": backward_s})
        print(f"[timing] step={state.global_step} total={total:.2f}s "
              f"generate={generate_s:.2f}s backward+optim={backward_s:.2f}s")
        if len(_step_timings) >= BENCHMARK_STEPS:
            control.should_training_stop = True
        return control

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return kwargs.get("control")
        return noop


def extract_boxed_answer(text: str):
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


def normalize_answer(ans):
    if ans is None:
        return None
    return ans.strip().replace(" ", "").replace(",", "")


def reward_correctness(completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        response = completion[0]["content"] if isinstance(completion, list) else completion
        pred = extract_boxed_answer(response)
        if pred is None:
            rewards.append(-1.0)
        elif normalize_answer(pred) == normalize_answer(gt):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def main():
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    print(f"[config] gpu_tag={GPU_TAG} rank={LORA_RANK} num_generations={NUM_GENERATIONS} "
          f"max_completion_length={MAX_COMPLETION_LENGTH} train_batch_size={TRAIN_BATCH_SIZE} "
          f"grad_accum={GRAD_ACCUM} use_vllm={USE_VLLM} fp8={USE_FP8} benchmark_steps={BENCHMARK_STEPS}")

    snapshot_memory("00_before_load")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH,
        load_in_4bit=False,
        load_in_fp8=USE_FP8,
        fast_inference=USE_VLLM,
        max_lora_rank=max(LORA_RANK, 8),
        gpu_memory_utilization=0.6,
    )
    snapshot_memory("01_after_base_model_load")  # base权重(量化/精度决定的)占用 + vLLM引擎(若fast_inference=True会在这一步一起初始化)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    snapshot_memory("02_after_lora_wrap")  # 验证"LoRA参数占用可忽略"这个假设——和01的差值应该很小

    dataset = load_dataset("parquet", data_files=DATA_PATH, split="train")

    def format_example(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"] + PROMPT_SUFFIX}],
            "answer": example["reward_model"]["ground_truth"],
        }

    dataset = dataset.map(format_example)

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=f"benchmark_{GPU_TAG}",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=BENCHMARK_STEPS + 2,  # 多留2步buffer，实际由callback提前叫停
        epsilon=EPSILON_LOW,
        epsilon_high=EPSILON_HIGH,
        temperature=1.0,
        use_vllm=USE_VLLM,
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        save_strategy="no",  # benchmark跑几步不需要存checkpoint
        report_to="none",    # benchmark跑几步不需要接wandb
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correctness],
        args=training_args,
        train_dataset=dataset,
        callbacks=[StepTimingCallback()],
    )

    patch_trainer_timing(trainer)

    snapshot_memory("03_before_first_step")
    trainer.train()
    snapshot_memory("04_after_training_steps")

    # ---- 汇总输出：可以直接抄进对比表的数字 ----
    if _step_timings:
        # 跳过第一步（有编译/初始化开销，不能代表稳态耗时）
        steady_steps = _step_timings[1:] if len(_step_timings) > 1 else _step_timings
        avg_generate = sum(s["generate_s"] for s in steady_steps) / len(steady_steps)
        avg_backward = sum(s["backward_s"] for s in steady_steps) / len(steady_steps)
        avg_total = sum(s["total_s"] for s in steady_steps) / len(steady_steps)
    else:
        avg_generate = avg_backward = avg_total = None

    summary = {
        "gpu_tag": GPU_TAG,
        "config": {
            "lora_rank": LORA_RANK,
            "num_generations": NUM_GENERATIONS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch": TRAIN_BATCH_SIZE * GRAD_ACCUM,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "use_vllm": USE_VLLM,
            "use_fp8": USE_FP8,
        },
        "avg_steady_state_seconds": {
            "generate": avg_generate,
            "backward_optim": avg_backward,
            "total": avg_total,
        },
        "memory_snapshots_gb": _memory_snapshots,
        "raw_step_timings": _step_timings,
    }

    result_path = os.path.join(OUTPUT_DIR, f"benchmark_summary_{GPU_TAG}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[summary] GPU: {GPU_TAG}")
    print(f"[summary] 稳态单步耗时（跳过首步编译开销）: "
          f"generate={avg_generate:.2f}s backward+optim={avg_backward:.2f}s total={avg_total:.2f}s"
          if avg_generate is not None else "[summary] 没有采集到有效的稳态step数据")
    print(f"[summary] 显存快照: {json.dumps(_memory_snapshots, indent=2, ensure_ascii=False)}")
    print(f"[summary] 完整结果已写入: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
