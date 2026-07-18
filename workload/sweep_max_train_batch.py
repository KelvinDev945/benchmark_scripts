"""
"单独训练"场景：找这张卡在纯训练(不跑vLLM推理引擎)下最大能撑多大的 batch size。

用子进程跑 train_grpo_benchmark.py（USE_VLLM=0，BENCHMARK_STEPS给小值只是为了探测OOM，
不追求准确计时），每个batch size一个独立子进程——避免在同一进程里反复扫描时，
前一次OOM残留的显存碎片/未释放对象影响下一次测试的准确性。

用法：
  MODEL_PATH=... DATA_PATH=... python3 sweep_max_train_batch.py
  可选：SWEEP_START=1 SWEEP_MAX=64 GPU_TAG=xxx
"""

import os
import subprocess
import sys
import json

# 重定向到文件跑的时候（nohup ... > log.txt），Python默认全缓冲，看不到实时进度，
# 直到进程退出才一次性写入。这里强制无缓冲，配合下面用 sys.executable 重新起子进程时
# 也传 -u，两层都不缓冲。
sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/rivermind-data/models/DeepSeek-R1-Distill-Qwen-1.5B")
DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/root/rivermind-data/datasets/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/benchmark_run")
GPU_TAG = os.environ.get("GPU_TAG", "unknown_gpu")

SWEEP_START = int(os.environ.get("SWEEP_START", "1"))
SWEEP_MAX = int(os.environ.get("SWEEP_MAX", "64"))  # 上限保护，避免死循环扫到不合理的大小

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_grpo_benchmark.py")


def try_batch_size(batch_size):
    """跑一次子进程，返回 (成功与否, stdout+stderr尾部用于诊断)"""
    env = os.environ.copy()
    env.update({
        "MODEL_PATH": MODEL_PATH,
        "DATA_PATH": DATA_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "GPU_TAG": f"{GPU_TAG}_sweep_bs{batch_size}",
        "USE_VLLM": "0",           # 单独训练场景：不启动vLLM推理引擎
        "TRAIN_BATCH_SIZE": str(batch_size),
        "GRAD_ACCUM": "1",         # 只关心单次forward+backward能不能撑住这个batch，不需要梯度累积
        "BENCHMARK_STEPS": "2",    # 探测OOM只需要跑通几步，不用完整benchmark步数
    })
    print(f"[sweep] 尝试 TRAIN_BATCH_SIZE={batch_size} ...")
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        env=env, capture_output=True, text=True, timeout=600,
    )
    success = result.returncode == 0
    tail = (result.stdout + result.stderr)[-2000:]
    if not success:
        is_oom = "CUDA out of memory" in tail or "OutOfMemoryError" in tail
        print(f"[sweep] batch_size={batch_size} 失败（{'OOM' if is_oom else '其他错误，见下方日志'}）")
        if not is_oom:
            print(tail)
    else:
        print(f"[sweep] batch_size={batch_size} 成功")
    return success, tail


def main():
    print(f"[config] gpu_tag={GPU_TAG} sweep_range=[{SWEEP_START}, {SWEEP_MAX}]")

    # 第一步：指数扩大找到"第一个失败点"的上界（避免线性扫描从1开始每个都试太慢）
    last_success = None
    bs = SWEEP_START
    first_fail = None
    while bs <= SWEEP_MAX:
        ok, _ = try_batch_size(bs)
        if ok:
            last_success = bs
            bs *= 2
        else:
            first_fail = bs
            break
    else:
        print(f"[sweep] 到达上限 SWEEP_MAX={SWEEP_MAX} 仍未失败，这张卡撑得住的batch size比预设上限还大，"
              f"建议调大 SWEEP_MAX 重新跑一次以找到真正的上限")

    if first_fail is None:
        # 没触发失败，说明 last_success 就是能测到的上限（不代表绝对最大值）
        result = {"gpu_tag": GPU_TAG, "max_working_batch_size": last_success,
                  "note": f"在SWEEP_MAX={SWEEP_MAX}范围内没有失败，真实上限可能更大"}
    elif last_success is None:
        # 连SWEEP_START都装不下
        result = {"gpu_tag": GPU_TAG, "max_working_batch_size": 0,
                  "note": f"连最小的SWEEP_START={SWEEP_START}都OOM，这张卡在当前配置下训不了"}
    else:
        # 第二步：在 (last_success, first_fail) 之间二分查找精确边界
        lo, hi = last_success, first_fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            ok, _ = try_batch_size(mid)
            if ok:
                lo = mid
            else:
                hi = mid
        result = {"gpu_tag": GPU_TAG, "max_working_batch_size": lo, "first_failing_batch_size": hi}

    result_path = os.path.join(OUTPUT_DIR, f"sweep_max_train_batch_{GPU_TAG}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"[summary] GPU: {GPU_TAG}")
    print(f"[summary] 单独训练场景，最大支持的 TRAIN_BATCH_SIZE: {result.get('max_working_batch_size')}")
    print(f"[summary] 完整结果已写入: {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
