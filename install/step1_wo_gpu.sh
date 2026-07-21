#!/bin/bash
# ============================================================
#  Step 1 [wo GPU] —— 无卡（CPU）阶段入口。
#  跑完直接挂卡，进入 Step 2。
#
#  并行策略（2026-07-21，含CUDA toolkit并入并行组）：
#  - download_model / download_dataset / clone_justrl / install_python_deps /
#    download_cuda_toolkit 这五个任务互相没有依赖（打在不同域名/CDN上：
#    ModelScope、hf-mirror、GitHub(代理)、PyPI镜像、NVIDIA官方下载），并行拉起，
#    哪个先下完就算哪个先完成，不用互相等待，缩短总耗时。
#    download_cuda_toolkit 原来默认跳过（flash-attn改用预编译wheel后不再需要
#    nvcc），但 Step 3 的 hardware/run_gpu_burn.sh / run_nvbandwidth.sh 编译时
#    仍然要用 nvcc——2026-07-21在fj02上因为没有提前装好，编译被迫拖到已经挂卡
#    计费之后才做，浪费GPU计费时间。改成默认在 Step 1（无卡阶段）就装好，编译
#    本身不需要真实GPU在场。
#  - flash-attn 的安装（预编译wheel下载，找不到才退回源码编译）依赖
#    torch 已经装好（要探测 torch 版本/ABI 才能拼出wheel文件名），
#    所以必须等 install_python_deps 完成后再串行跑，不能提前并行。
#    这一步内部本身也保留了下载失败自动切换代理源的逻辑（直连→
#    gh-proxy.com→ghfast.top，见 sources.sh 的 GITHUB_RELEASE_PROXY_CHAIN）。
#  - bootstrap_uv（装uv+modelscope+huggingface_hub）必须最先同步跑完一次，
#    因为后面几个并行任务里有几个都要用到它，并行装uv会互相踩踏。
# ============================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
LOG_DIR="$DATA_DIR/logs/step1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "===== [0/2] bootstrap_uv.sh（同步，后面的并行任务都依赖它） ====="
bash bootstrap_uv.sh

echo "===== [1/2] 并行下载：download_model / download_dataset / clone_justrl / install_python_deps / download_cuda_toolkit ====="
echo "      （五个任务互相独立，各自日志见 $LOG_DIR/*.log，谁先完成就算谁先完成）"

declare -A PIDS
for job in download_model download_dataset clone_justrl install_python_deps download_cuda_toolkit; do
    bash "${job}.sh" > "$LOG_DIR/${job}.log" 2>&1 &
    PIDS[$job]=$!
    echo "      启动 $job（pid=${PIDS[$job]}）"
done

FAILED=()
for job in "${!PIDS[@]}"; do
    if wait "${PIDS[$job]}"; then
        echo "      ✅ $job 完成"
    else
        echo "      ❌ $job 失败，日志见 $LOG_DIR/${job}.log"
        FAILED+=("$job")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "错误：以下并行任务失败：${FAILED[*]}"
    echo "完整日志目录：$LOG_DIR/"
    exit 1
fi

echo "===== [2/2] install_flash_attn.sh（依赖上面的torch已装好，串行执行；优先装预编译wheel，找不到才退回源码编译，自动切换代理源） ====="
bash install_flash_attn.sh

echo
echo "========== Step 1 [wo GPU] 全部完成 =========="
echo "并行任务日志目录：$LOG_DIR/"
echo "下一步：挂卡后运行 Step 2 —— bash install/step2_with_gpu.sh"
echo "==============================================="
