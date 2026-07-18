"""
"单独训练"场景（2026-07-19修正版）：测forward+backward+optimizer，但vLLM引擎保持加载
（fast_inference=True），显存占用包含vLLM的KV cache——因为真实GRPO训练流程里vLLM从
trainer初始化那一刻就常驻显存，不管当前是不是正在生成，所以"训练能用的显存余量"必须
把vLLM的固定占用算进去，不能像之前版本那样直接把vLLM整个关掉（那样测出来的最大batch
size会比真实情况乐观，因为真实流程里显存本来就被vLLM占掉一块）。

跟GRPOTrainer解耦：不走生成-打分-算advantage这套真实rollout流程（那部分交给
train_grpo_benchmark.py测，这里只关心"给定一个固定形状的batch，forward+backward+
optimizer.step()要多少显存/多久"）——但跳过生成不等于用随机假token：Unsloth的kernel
优化虽然对输入内容无关都生效，真实token的分布/padding模式才更贴近实际训练场景，
所以这里用DATA_PATH里真实的prompt文本构造batch（跳过的只是"生成completion"这一步，
不是"用什么当输入"），避免依赖TRL内部生成方法的接口（不同版本方法名/返回结构不稳定，
之前patch_trainer_timing已经踩过这个坑）。

用法：
  MODEL_PATH=... DATA_PATH=... TRAIN_BATCH_SIZE=4 python3 train_only_benchmark.py
"""

import os
import time
import json
import sys

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")
DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/root/rivermind-data/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/benchmark_run")
GPU_TAG = os.environ.get("GPU_TAG", "unknown_gpu")

LORA_RANK = int(os.environ.get("LORA_RANK", "1"))
LORA_ALPHA = LORA_RANK * 2
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-5"))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))
# 单条序列长度 = prompt + completion，跟 train_grpo_benchmark.py 默认配置对齐，保证可比
SEQ_LENGTH = int(os.environ.get("SEQ_LENGTH", str(512 + 1024)))
BENCHMARK_STEPS = int(os.environ.get("BENCHMARK_STEPS", "3"))
# 默认True（单卡训练+推理合一场景）；测"专职训练卡"场景（这张卡以后不跑vLLM）就设LOAD_VLLM_ENGINE=0
LOAD_VLLM_ENGINE = os.environ.get("LOAD_VLLM_ENGINE", "1") == "1"

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


def main():
    import torch
    from unsloth import FastLanguageModel

    print(f"[config] gpu_tag={GPU_TAG} rank={LORA_RANK} train_batch_size={TRAIN_BATCH_SIZE} "
          f"seq_length={SEQ_LENGTH} benchmark_steps={BENCHMARK_STEPS} load_vllm_engine={LOAD_VLLM_ENGINE} "
          f"training_kernels=unsloth（forward/backward全程走FastLanguageModel patch过的Triton kernel，"
          f"跟是否使用真实数据/是否调用生成无关）")

    snapshot_memory("00_before_load")

    # LOAD_VLLM_ENGINE 对应两种不同的部署场景，没有绝对"更真实"的一方，取决于测的是哪种：
    #   True  —— 单卡训练+推理都在一起（vLLM全程占着显存），测"这种部署下训练还能用多大batch"
    #   False —— 专职训练卡（这张卡以后永远不跑vLLM），测"这种部署下训练能用多大batch"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=SEQ_LENGTH,
        load_in_4bit=False,
        fast_inference=LOAD_VLLM_ENGINE,
        max_lora_rank=max(LORA_RANK, 8),
        gpu_memory_utilization=0.6,
    )
    snapshot_memory("01_after_base_model_load" + ("_with_vllm_engine" if LOAD_VLLM_ENGINE else "_no_vllm"))

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    snapshot_memory("02_after_lora_wrap")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    # 真实数据：跳过的是"vLLM生成completion"这一步和GRPOTrainer的reward/advantage计算，
    # 不是"用什么当输入"——用DATA_PATH里真实的prompt文本tokenize后构造batch，padding/截断到
    # SEQ_LENGTH，token分布/padding模式比纯随机token更贴近真实训练场景（Unsloth的kernel优化
    # 本身跟输入内容无关，两种数据实测出的显存/耗时应该接近，但真实数据更有说服力）
    from datasets import load_dataset
    raw_dataset = load_dataset("parquet", data_files=DATA_PATH, split="train")
    real_prompts = [raw_dataset[i]["prompt"] for i in range(min(TRAIN_BATCH_SIZE, len(raw_dataset)))]
    while len(real_prompts) < TRAIN_BATCH_SIZE:  # 数据不够就循环复用，只是要凑够batch形状
        real_prompts.append(real_prompts[len(real_prompts) % len(raw_dataset)])

    encoded = tokenizer(
        real_prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LENGTH,
    )
    real_input_ids = encoded["input_ids"].to("cuda")
    real_attention_mask = encoded["attention_mask"].to("cuda")
    # 没有真实completion（跳过了生成），直接用同一段token当labels——语义上不是真正的
    # GRPO loss，但forward+backward要走的计算图/显存分配跟真实训练一致，够用于这个benchmark
    real_labels = real_input_ids.clone()

    snapshot_memory("03_before_first_step")

    step_timings = []
    model.train()
    for step in range(BENCHMARK_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(
            input_ids=real_input_ids,
            attention_mask=real_attention_mask,
            labels=real_labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        step_timings.append(elapsed)
        print(f"[timing] step={step} total={elapsed:.2f}s loss={loss.item():.4f}")

    snapshot_memory("04_after_training_steps")

    # 跳过第一步（有编译/初始化开销），只用稳态耗时
    steady = step_timings[1:] if len(step_timings) > 1 else step_timings
    avg_step = sum(steady) / len(steady) if steady else None

    summary = {
        "gpu_tag": GPU_TAG,
        "config": {
            "lora_rank": LORA_RANK,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "seq_length": SEQ_LENGTH,
            "vllm_engine_loaded": LOAD_VLLM_ENGINE,
            "training_kernels": "unsloth",  # forward/backward全程用Unsloth patch过的Triton kernel（跟输入是否真实/是否生成无关）
            "input_data": "real_dataset_prompts",  # 2026-07-19起改用真实数据而非随机token
        },
        "avg_steady_state_seconds": avg_step,
        "memory_snapshots_gb": _memory_snapshots,
        "raw_step_timings": step_timings,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(OUTPUT_DIR, f"train_only_benchmark_{GPU_TAG}.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[summary] GPU: {GPU_TAG}")
    print(f"[summary] 稳态单步耗时（vLLM常驻但不生成，跳过首步）: "
          f"{avg_step:.3f}s" if avg_step is not None else "[summary] 没有采集到有效数据")
    print(f"[summary] 显存快照: {json.dumps(_memory_snapshots, indent=2, ensure_ascii=False)}")
    print(f"[summary] 完整结果已写入: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
