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

# 默认路径全部落在 /root/rivermind-data —— 持久化数据盘，不是根分区，实例释放/重启不会丢
# （详见持久记忆 feedback_gpu_rental_persistent_data_disk / obsidian 环境与框架.md）
MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")
LORA_PATH = os.environ.get("LORA_PATH", "")  # 留空则不加载LoRA，纯测base模型吞吐
LORA_RANK = int(os.environ.get("LORA_RANK", "1"))
GPU_TAG = os.environ.get("GPU_TAG", "unknown_gpu")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "32"))       # 固定并发请求数
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))  # 固定生成长度，跨卡保持一致才可比
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/benchmark_run")

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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=max(LORA_RANK, 8),
        gpu_memory_utilization=0.85,  # 纯推理基准，不需要给训练侧留显存，可以吃更多
    )

    lora_request = None
    if LORA_PATH:
        lora_request = model.load_lora(LORA_PATH)
        print(f"[config] 已加载 LoRA adapter: {LORA_PATH}")

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
    )

    # 先跑一次热身（排除首次CUDA graph捕获/编译开销），热身结果不计入统计
    print("[warmup] 跑一次热身请求（不计入统计）...")
    warmup_kwargs = {"lora_request": lora_request} if lora_request else {}
    model.fast_generate(formatted[:2], sampling_params=sampling_params, **warmup_kwargs)

    print(f"[bench] 正式测试：{NUM_PROMPTS} 条并发请求，每条最多生成 {MAX_NEW_TOKENS} tokens...")
    t0 = time.perf_counter()
    outputs = model.fast_generate(formatted, sampling_params=sampling_params, **warmup_kwargs)
    elapsed = time.perf_counter() - t0

    total_output_tokens = sum(len(tokenizer.encode(o.outputs[0].text)) for o in outputs)
    throughput = total_output_tokens / elapsed

    summary = {
        "gpu_tag": GPU_TAG,
        "num_prompts": NUM_PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "elapsed_seconds": round(elapsed, 3),
        "total_output_tokens": total_output_tokens,
        "throughput_tokens_per_sec": round(throughput, 2),
        "lora_loaded": bool(LORA_PATH),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(OUTPUT_DIR, f"vllm_throughput_{GPU_TAG}.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[summary] GPU: {GPU_TAG}")
    print(f"[summary] 总耗时: {elapsed:.2f}s, 总输出token数: {total_output_tokens}")
    print(f"[summary] 纯vLLM推理吞吐: {throughput:.2f} tokens/s")
    print(f"[summary] 完整结果已写入: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
