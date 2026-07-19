"""
纯 vLLM 推理吞吐基准测试——独立于 GRPO 训练循环
用于评估"某张卡专职做推理"这个分工方案的理论上限吞吐（详见训练方法论与实验记录.md 里
"多卡训练/推理速度基准测试"实验设计）。不跑任何训练，只加载模型+可选LoRA adapter，
跑固定输入的 batch generation，测 tokens/s。

用法：
  MODEL_PATH=... LORA_PATH=... python3 vllm_throughput_benchmark.py
"""

import os
import time
import json

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

# 默认路径全部落在 /root/rivermind-data —— 持久化数据盘，不是根分区，实例释放/重启不会丢
# （详见持久记忆 feedback_gpu_rental_persistent_data_disk / obsidian 环境与框架.md）
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")
LORA_PATH = os.environ.get("LORA_PATH", "")  # 留空则不加载LoRA，纯测base模型吞吐
LORA_RANK = int(os.environ.get("LORA_RANK", "1"))
GPU_TAG = os.environ.get("GPU_TAG", "unknown_gpu")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "8"))        # 固定并发请求数，默认8对齐GRPO的num_generations(rollout N)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))  # 固定生成长度，跨卡保持一致才可比
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/benchmark_run")
# 默认关闭：FIXED_PROMPT这道简单数学题模型几百个token内就会遇到EOS提前结束，导致
# MAX_NEW_TOKENS调多大实际输出token数都不变（2026-07-19实测4096/8192/16384三档
# 输出token数完全一样，测的其实是同一件事）。开启后用vLLM的ignore_eos强制生成满
# MAX_NEW_TOKENS，才能真实测出"长上下文"本身对吞吐/显存的影响。
FORCE_FULL_LENGTH = os.environ.get("FORCE_FULL_LENGTH", "0") == "1"

# 固定的测试prompt——跨卡用完全一样的输入，保证吞吐数字可比
FIXED_PROMPT = (
    "Solve the following problem step by step and put your final answer within \\boxed{}.\n"
    "A train travels 120 miles in 2 hours, then speeds up and travels another 180 miles in 2 hours. "
    "What is the average speed of the train over the entire journey in miles per hour?"
)


def main():
    from unsloth import FastLanguageModel
    from vllm import SamplingParams

    print(f"[config] gpu_tag={GPU_TAG} model={MODEL_PATH} lora_path={LORA_PATH or '(none)'} "
          f"num_prompts={NUM_PROMPTS} max_new_tokens={MAX_NEW_TOKENS}")

    snapshot_memory("00_before_load")

    # max_seq_length 要跟着 MAX_NEW_TOKENS 放大（+512给prompt留余量），否则扫描更长
    # 长度时会被这里的硬上限提前截断/拒绝，而不是真的测到显存OOM的那个边界
    max_seq_length = MAX_NEW_TOKENS + 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=max(LORA_RANK, 8),
        gpu_memory_utilization=0.85,  # 纯推理基准，不需要给训练侧留显存，可以吃更多
    )
    snapshot_memory("01_after_model_and_vllm_engine_load")  # base权重 + vLLM KV cache预留

    lora_request = None
    if LORA_PATH:
        lora_request = model.load_lora(LORA_PATH)
        print(f"[config] 已加载 LoRA adapter: {LORA_PATH}")
        snapshot_memory("02_after_lora_load")

    prompts = [FIXED_PROMPT] * NUM_PROMPTS
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
        ignore_eos=FORCE_FULL_LENGTH,  # 忽略EOS，强制生成满MAX_NEW_TOKENS而不是提前停止
    )

    # 先跑一次热身（排除首次CUDA graph捕获/编译开销），热身结果不计入统计
    print("[warmup] 跑一次热身请求（不计入统计）...")
    warmup_kwargs = {"lora_request": lora_request} if lora_request else {}
    model.fast_generate(formatted[:2], sampling_params=sampling_params, **warmup_kwargs)

    snapshot_memory("03_before_generate")

    print(f"[bench] 正式测试：{NUM_PROMPTS} 条并发请求，每条最多生成 {MAX_NEW_TOKENS} tokens...")
    t0 = time.perf_counter()
    outputs = model.fast_generate(formatted, sampling_params=sampling_params, **warmup_kwargs)
    elapsed = time.perf_counter() - t0

    snapshot_memory("04_after_generate")

    total_output_tokens = sum(len(tokenizer.encode(o.outputs[0].text)) for o in outputs)
    throughput = total_output_tokens / elapsed

    # 记录实际权重精度，而不是想当然认为是bf16——不同GPU/Unsloth版本的auto-dtype
    # 探测结果可能不同，跨卡对比时这个字段必须是实测值
    model_dtype = str(next(model.parameters()).dtype).replace("torch.", "")

    summary = {
        "gpu_tag": GPU_TAG,
        "num_prompts": NUM_PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "elapsed_seconds": round(elapsed, 3),
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_sec": round(throughput, 2),
        "dtype": model_dtype,
        "force_full_length": FORCE_FULL_LENGTH,
        "lora_loaded": bool(LORA_PATH),
        "lora_rank": LORA_RANK if LORA_PATH else None,
        "memory_snapshots_gb": _memory_snapshots,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(OUTPUT_DIR, f"vllm_throughput_{GPU_TAG}.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[summary] GPU: {GPU_TAG}")
    print(f"[summary] 总耗时: {elapsed:.2f}s, 总输出token数: {total_output_tokens}")
    print(f"[summary] 纯vLLM推理吞吐: {throughput:.2f} tokens/s")
    print(f"[summary] dtype={model_dtype}, lora_loaded={summary['lora_loaded']}, lora_rank={summary['lora_rank']}")
    print(f"[summary] 显存快照: {json.dumps(_memory_snapshots, indent=2, ensure_ascii=False)}")
    print(f"[summary] 完整结果已写入: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
