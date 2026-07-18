#!/bin/bash
# [with GPU] 快速验证模型能加载 + LoRA能包装，几十秒出结果，不跑训练/生成。
# 被 step2_with_gpu.sh 调用，也可以单独跑。
set -e

# /root/rivermind-data 是持久化数据盘，不是根分区，实例释放/重启不会丢
# （详见持久记忆 feedback_gpu_rental_persistent_data_disk / obsidian 环境与框架.md）
MODEL_PATH="${MODEL_PATH:-/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B}"

echo '--- 确认 GPU 可见 ---'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || { echo '错误：没有检测到GPU，请确认已挂卡'; exit 1; }

echo '--- 验证 Unsloth 能否正常加载模型 + 包装LoRA ---'
python3 -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='$MODEL_PATH', max_seq_length=2048,
    load_in_4bit=False, fast_inference=True,
)
print('模型加载成功（base权重 + vLLM引擎初始化正常）')
model = FastLanguageModel.get_peft_model(
    model, r=8,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=16, use_gradient_checkpointing='unsloth', random_state=3407,
)
print('LoRA adapter 包装成功')
"

echo "[verify_gpu_environment] 环境验证通过"
