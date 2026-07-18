"""
挂卡后的快速环境验证——在跑完整基准测试之前先确认环境没问题，避免卡计费时间浪费在
一个本来就会失败的完整benchmark上。只加载模型，不跑训练/生成，几十秒内出结果。

用法：
  MODEL_PATH=... python3 verify_gpu_environment.py
"""

import os
import sys

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")


def main():
    print("=== 1. 检查 GPU 可见 ===")
    import torch
    if not torch.cuda.is_available():
        print("❌ torch.cuda.is_available() = False，没有检测到GPU，请先挂卡再运行此脚本")
        sys.exit(1)
    print(f"✅ GPU可见: {torch.cuda.get_device_name(0)}, "
          f"torch={torch.__version__} (cuda build: {torch.version.cuda})")

    print("\n=== 2. 验证 Unsloth FastLanguageModel 能否正常加载模型（含vLLM快速推理引擎） ===")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=True,
    )
    print("✅ 模型加载成功（base权重 + vLLM引擎初始化正常）")

    print("\n=== 3. 验证 LoRA 包装是否正常 ===")
    model = FastLanguageModel.get_peft_model(
        model, r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("✅ LoRA adapter 包装成功")

    print(f"\n✅✅ 环境验证全部通过，可以放心跑完整基准测试了（GPU: {torch.cuda.get_device_name(0)}）")


if __name__ == "__main__":
    main()
