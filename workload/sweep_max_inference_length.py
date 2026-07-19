"""
"单独推理"场景：找这张卡纯vLLM推理下最大能撑多长的生成长度(max_new_tokens)。

用子进程跑 vllm_throughput_benchmark.py（NUM_PROMPTS给小值只是为了探测OOM/vLLM初始化失败，
不追求准确吞吐数字），每个长度一个独立子进程——避免同进程反复扫描时vLLM引擎/显存状态
在失败后难以干净复位，影响下一次测试的准确性。

用法：
  MODEL_PATH=... python3 sweep_max_inference_length.py
  可选：SWEEP_START=512 SWEEP_MAX=32768 GPU_TAG=xxx
"""

import os
import subprocess
import sys
import json

# 重定向到文件跑的时候（nohup ... > log.txt），Python默认全缓冲，进程没退出就看不到进度
sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/benchmark_run")
GPU_TAG = os.environ.get("GPU_TAG", "unknown_gpu")

SWEEP_START = int(os.environ.get("SWEEP_START", "512"))
SWEEP_MAX = int(os.environ.get("SWEEP_MAX", "32768"))  # 上限保护

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_SCRIPT = os.path.join(SCRIPT_DIR, "vllm_throughput_benchmark.py")


def try_length(max_new_tokens):
    env = os.environ.copy()
    env.update({
        "MODEL_PATH": MODEL_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "GPU_TAG": f"{GPU_TAG}_sweep_len{max_new_tokens}",
        "NUM_PROMPTS": "8",  # 对齐GRPO的num_generations(rollout N)，不用vllm_throughput_benchmark.py默认的32
        "MAX_NEW_TOKENS": str(max_new_tokens),
    })
    print(f"[sweep] 尝试 MAX_NEW_TOKENS={max_new_tokens} ...")
    result = subprocess.run(
        [sys.executable, INFER_SCRIPT],
        env=env, capture_output=True, text=True, timeout=600,
    )
    success = result.returncode == 0
    tail = (result.stdout + result.stderr)[-2000:]
    if not success:
        is_oom = "CUDA out of memory" in tail or "OutOfMemoryError" in tail or "KV cache" in tail
        print(f"[sweep] max_new_tokens={max_new_tokens} 失败（{'OOM/KV cache不够' if is_oom else '其他错误，见下方日志'}）")
        if not is_oom:
            print(tail)
    else:
        print(f"[sweep] max_new_tokens={max_new_tokens} 成功")
    return success, tail


def main():
    print(f"[config] gpu_tag={GPU_TAG} sweep_range=[{SWEEP_START}, {SWEEP_MAX}]")

    last_success = None
    length = SWEEP_START
    first_fail = None
    while length <= SWEEP_MAX:
        ok, _ = try_length(length)
        if ok:
            last_success = length
            length *= 2
        else:
            first_fail = length
            break
    else:
        print(f"[sweep] 到达上限 SWEEP_MAX={SWEEP_MAX} 仍未失败，这张卡撑得住的长度比预设上限还大，"
              f"建议调大 SWEEP_MAX 重新跑一次以找到真正的上限")

    if first_fail is None:
        result = {"gpu_tag": GPU_TAG, "max_working_length": last_success,
                  "note": f"在SWEEP_MAX={SWEEP_MAX}范围内没有失败，真实上限可能更大"}
    elif last_success is None:
        result = {"gpu_tag": GPU_TAG, "max_working_length": 0,
                  "note": f"连最小的SWEEP_START={SWEEP_START}都装不下，这张卡在当前配置下推理不了"}
    else:
        lo, hi = last_success, first_fail
        while hi - lo > 128:  # 长度用128为粒度二分，不用二分到1（意义不大，还费时间）
            mid = ((lo + hi) // 2 // 128) * 128
            if mid == lo:
                break
            ok, _ = try_length(mid)
            if ok:
                lo = mid
            else:
                hi = mid
        result = {"gpu_tag": GPU_TAG, "max_working_length": lo, "first_failing_length": hi}

    result_path = os.path.join(OUTPUT_DIR, f"sweep_max_inference_length_{GPU_TAG}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[summary] GPU: {GPU_TAG}")
    print(f"[summary] 单独推理场景，最大支持的生成长度(max_new_tokens): {result.get('max_working_length')}")
    print(f"[summary] 完整结果已写入: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
