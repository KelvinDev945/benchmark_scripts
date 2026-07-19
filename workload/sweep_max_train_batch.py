"""
"单独训练"场景：找这张卡在训练时最大能撑多大的 batch size。

用子进程跑 train_only_benchmark.py（BENCHMARK_STEPS给小值只是为了探测OOM，不追求准确
计时），每个batch size一个独立子进程——避免在同一进程里反复扫描时，前一次OOM残留的
显存碎片/未释放对象影响下一次测试的准确性。

关键：train_only_benchmark.py 里 vLLM引擎保持加载(fast_inference=True)，KV cache照常
预留显存——不是"假装没有vLLM"，因为真实GRPO训练流程里vLLM从trainer初始化起就常驻显存，
这里测的是"vLLM占着它那份显存之后，训练侧还能用多大batch"，才是真实可用的上限
（2026-07-19修正：早期版本直接把vLLM关掉来测，会得出偏乐观、不符合实际流水线的数字）。

用法：
  MODEL_PATH=... python3 sweep_max_train_batch.py
  可选：SWEEP_START=1 SWEEP_MAX=64 GPU_TAG=xxx MEMORY_GUARD_RATIO=0.8

默认探测 train_only_benchmark.py（"单独训练，vLLM是否常驻由该脚本的LOAD_VLLM_ENGINE决定"）。
2026-07-19起可以改指向其他benchmark脚本复用同一套护栏+双向搜索逻辑，比如
train_grpo_benchmark.py（真实GRPO训练+推理耦合场景）：
  TARGET_SCRIPT=train_grpo_benchmark.py RESULT_FILE_PREFIX=benchmark_summary python3 sweep_max_train_batch.py
两个脚本的显存快照tag名恰好都是"04_after_training_steps"，护栏检查代码不用改；
如果以后接入的脚本tag名不同，用 MEMORY_SNAPSHOT_TAG 覆盖。跑 train_grpo_benchmark.py
且开了 UNSLOTH_VLLM_STANDBY 时，额外加 MEMORY_METRIC_FIELD=nvidia_smi_used_gb
（原因见下方 MEMORY_METRIC_FIELD 的注释）。

显存护栏（2026-07-19新增，用户人工观察到 seq_length=4096/bs=48 时已经吃到22.55GB/24GB
——只剩1.5GB余量，继续逼近OOM边界风险不小，且再往上一两档batch带来的收益有限）：
只要某个batch size成功但训练后峰值显存(max_allocated_gb)已经达到显卡总显存的
MEMORY_GUARD_RATIO（默认80%），就把这个batch size直接当作最终答案，不再继续探测更大的
batch size（不管是指数扩大阶段还是二分查找阶段）——保守上限比精确边界更重要，没必要
每次都贴着OOM线走。
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
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/rivermind-data/outputs/benchmark_run")
GPU_TAG = os.environ.get("GPU_TAG", "unknown_gpu")

SWEEP_START = int(os.environ.get("SWEEP_START", "1"))
SWEEP_MAX = int(os.environ.get("SWEEP_MAX", "64"))  # 上限保护，避免死循环扫到不合理的大小
MEMORY_GUARD_RATIO = float(os.environ.get("MEMORY_GUARD_RATIO", "0.8"))

# 探测目标可配置——同一套护栏+双向搜索逻辑既能测"单独训练"也能测"训练+推理耦合"场景
TARGET_SCRIPT_NAME = os.environ.get("TARGET_SCRIPT", "train_only_benchmark.py")
RESULT_FILE_PREFIX = os.environ.get("RESULT_FILE_PREFIX", "train_only_benchmark")
MEMORY_SNAPSHOT_TAG = os.environ.get("MEMORY_SNAPSHOT_TAG", "04_after_training_steps")
# 显存护栏读哪个字段——2026-07-19发现 UNSLOTH_VLLM_STANDBY 开启时 torch.cuda 的
# max_allocated_gb 统计会失真（CuMemAllocator手动unmap/remap绕过了caching allocator正常
# 记账，实测算出27GB/35GB这种超过物理显存总量的荒谬数字），train_grpo_benchmark.py因此
# 额外记录了基于nvidia-smi的真实物理占用nvidia_smi_used_gb，standby场景应该用这个
MEMORY_METRIC_FIELD = os.environ.get("MEMORY_METRIC_FIELD", "max_allocated_gb")
# 探测阶段跑几步——只是为了看跑不跑得通(OOM/超时与否)，不追求计时精度(精度靠最后的完整
# 步数确认跑)。真实GRPO耦合场景(train_grpo_benchmark.py)单步耗时可能到几分钟(生成
# num_generations×train_batch_size条长completion)，2步都可能超时，应该调小到1步。
SWEEP_PROBE_STEPS = int(os.environ.get("SWEEP_PROBE_STEPS", "2"))
# 单次探测子进程的超时上限——真实GRPO耦合场景单步耗时可能远超纯训练场景，600秒对重负载
# 配置(大completion length×大batch)可能不够，需要按场景调大
SWEEP_SUBPROCESS_TIMEOUT = int(os.environ.get("SWEEP_SUBPROCESS_TIMEOUT", "600"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, TARGET_SCRIPT_NAME)


def get_gpu_total_memory_gb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip().splitlines()[0]) / 1024  # MiB -> GiB


GPU_TOTAL_MEMORY_GB = get_gpu_total_memory_gb()


class MemoryGuardTriggered(Exception):
    """某个成功的 batch size 已经吃到显存护栏比例，携带这个 batch size 作为最终答案"""

    def __init__(self, batch_size, ratio, peak_gb):
        self.batch_size = batch_size
        self.ratio = ratio
        self.peak_gb = peak_gb


def try_batch_size(batch_size):
    """跑一次子进程，返回 (成功与否, stdout+stderr尾部用于诊断)。

    成功且显存占比触发护栏时，直接抛 MemoryGuardTriggered 让调用方（主流程）提前终止
    所有后续探测——不管当前处于指数扩大还是二分查找的哪个阶段。
    """
    env = os.environ.copy()
    env.update({
        "MODEL_PATH": MODEL_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "GPU_TAG": f"{GPU_TAG}_sweep_bs{batch_size}",
        "TRAIN_BATCH_SIZE": str(batch_size),
        "BENCHMARK_STEPS": str(SWEEP_PROBE_STEPS),
    })
    print(f"[sweep] 尝试 TRAIN_BATCH_SIZE={batch_size} ...")
    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT],
            env=env, capture_output=True, text=True, timeout=SWEEP_SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        # 之前这里没捕获过，超时会直接把整个sweep进程崩掉、不写任何结果文件——
        # 2026-07-19在GRPO耦合场景实测踩过：单步耗时可能到几分钟，600秒探测多个
        # 完整RL step很容易超时，必须当成一次普通失败处理，而不是让整个脚本崩溃
        print(f"[sweep] batch_size={batch_size} 失败（超时，超过{SWEEP_SUBPROCESS_TIMEOUT}秒未完成——"
              f"不代表OOM，可能只是这个配置单步耗时本身就很长，考虑调大SWEEP_SUBPROCESS_TIMEOUT或调小SWEEP_PROBE_STEPS）")
        tail = ((e.stdout or "") + (e.stderr or ""))[-2000:] if (e.stdout or e.stderr) else "(超时，无可用输出)"
        return False, tail

    tail = (result.stdout + result.stderr)[-2000:]

    # 2026-07-19发现：开UNSLOTH_VLLM_STANDBY后，进程退出时CuMemAllocator的显存池
    # 析构偶尔会崩(c10::cuda::MemPool::~MemPool())，导致子进程返回非0退出码——但这个崩溃
    # 发生在Python解释器收尾阶段，实际的训练/生成步骤早就跑完并把结果json写好了。
    # 所以不能只看returncode，先看结果json是否存在且完整，这才是"这一步是否真的成功"的
    # 依据；returncode非0只在json读不到时才当成失败信号用。
    result_json_path = os.path.join(OUTPUT_DIR, f"{RESULT_FILE_PREFIX}_{GPU_TAG}_sweep_bs{batch_size}.json")
    result_json = None
    try:
        with open(result_json_path) as f:
            result_json = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if result_json is None:
        is_oom = "CUDA out of memory" in tail or "OutOfMemoryError" in tail
        print(f"[sweep] batch_size={batch_size} 失败（{'OOM' if is_oom else '其他错误，见下方日志'}）")
        if not is_oom:
            print(tail)
        return False, tail

    if result.returncode != 0:
        print(f"[sweep] batch_size={batch_size} 子进程退出码非0(returncode={result.returncode})，"
              f"但结果json完整存在——判定为进程退出阶段的良性崩溃(如standby模式下的MemPool析构)，"
              f"训练本身视为成功")
    print(f"[sweep] batch_size={batch_size} 成功")

    # 跟显卡总显存比一下，判断要不要触发护栏。MEMORY_METRIC_FIELD 先按根级字段找
    # （比如 train_grpo_benchmark.py 用后台线程轮询nvidia-smi全程追踪到的真实峰值
    # nvidia_smi_peak_used_gb，就直接放在结果json根层级），找不到再退回嵌套在
    # memory_snapshots_gb[MEMORY_SNAPSHOT_TAG]下面那种旧格式（train_only_benchmark.py）
    try:
        if MEMORY_METRIC_FIELD in result_json:
            peak_gb = result_json[MEMORY_METRIC_FIELD]
        else:
            peak_gb = result_json["memory_snapshots_gb"][MEMORY_SNAPSHOT_TAG][MEMORY_METRIC_FIELD]
        ratio = peak_gb / GPU_TOTAL_MEMORY_GB
        print(f"[sweep] batch_size={batch_size} 训练后峰值显存 {peak_gb:.2f}GB / "
              f"总显存 {GPU_TOTAL_MEMORY_GB:.2f}GB = {ratio:.1%}")
        if ratio >= MEMORY_GUARD_RATIO:
            print(f"[sweep] 显存占比已达到护栏阈值 {MEMORY_GUARD_RATIO:.0%}，"
                  f"停止继续探测更大的batch size，把 {batch_size} 作为保守上限")
            raise MemoryGuardTriggered(batch_size, ratio, peak_gb)
    except KeyError as e:
        print(f"[sweep] 警告：结果json里没有 {MEMORY_METRIC_FIELD} 这个字段（{e}），跳过护栏检查")

    return True, tail


def main():
    print(f"[config] gpu_tag={GPU_TAG} target_script={TARGET_SCRIPT_NAME} sweep_range=[{SWEEP_START}, {SWEEP_MAX}] "
          f"memory_guard_ratio={MEMORY_GUARD_RATIO:.0%} gpu_total_memory={GPU_TOTAL_MEMORY_GB:.2f}GB")

    try:
        # 起点不再固定从1开始——SWEEP_START可以设成"已知在更短长度下测出的上限"这类先验值，
        # 直接从那附近起跳：成功就翻倍往上探（找上界），失败就从SWEEP_START以下最大的
        # 2的幂次开始高往低对半砍（找一个比SWEEP_START小的成功点），两个方向都避免了
        # 从1开始逐个爬——小batch几乎总能成功，从1爬只是白费探测次数。
        ok, _ = try_batch_size(SWEEP_START)
        if ok:
            last_success = SWEEP_START
            first_fail = None
            bs = SWEEP_START * 2
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
        else:
            # SWEEP_START本身就失败——说明真实上限比它小。不从1开始逐个爬（那样小的batch
            # 基本都能成功，纯粹浪费探测次数），改成从SWEEP_START以下最大的2的幂次开始，
            # 高往低对半砍，直到找到第一个成功点为止（跟"指数扩大"阶段的方向相反，但同样
            # 是2的倍数跳跃，不是线性扫描）
            last_success = None
            first_fail = SWEEP_START
            bs = 1
            while bs * 2 < SWEEP_START:
                bs *= 2
            while bs >= 1:
                ok2, _ = try_batch_size(bs)
                if ok2:
                    last_success = bs
                    break
                else:
                    first_fail = bs
                    bs //= 2

        if first_fail is None:
            # 没触发失败，说明 last_success 就是能测到的上限（不代表绝对最大值）
            result = {"gpu_tag": GPU_TAG, "max_working_batch_size": last_success,
                      "note": f"在SWEEP_MAX={SWEEP_MAX}范围内没有失败，真实上限可能更大"}
        elif last_success is None:
            # 连1都装不下
            result = {"gpu_tag": GPU_TAG, "max_working_batch_size": 0,
                      "note": "连 batch_size=1 都OOM，这张卡在当前配置下训不了"}
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
    except MemoryGuardTriggered as guard:
        # 不管当前处于指数扩大还是二分查找的哪一步，只要摸到护栏比例就立刻收工，
        # 不再逼近精确的OOM边界——保守上限、留够余量比"卡着线走"更重要
        result = {
            "gpu_tag": GPU_TAG,
            "max_working_batch_size": guard.batch_size,
            "note": f"显存护栏提前停止：batch_size={guard.batch_size}时训练后峰值显存已达"
                    f"{guard.peak_gb:.2f}GB（总显存{GPU_TOTAL_MEMORY_GB:.2f}GB的{guard.ratio:.1%}，"
                    f"超过护栏阈值{MEMORY_GUARD_RATIO:.0%}），未继续二分逼近精确OOM边界，"
                    f"真实上限可能比这个数字更大",
        }

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
