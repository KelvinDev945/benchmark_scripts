#!/bin/bash
# [wo GPU，可选] 下载 + 安装 CUDA Toolkit（含nvcc）到持久化数据盘。
#
# 背景：`nvidia-cuda-nvcc-cu12` 这个pip包实测不含真正的nvcc可执行文件（只有ptxas等组件），
# apt的 nvidia-cuda-toolkit 版本可能跟torch用的cu128对不上；官方 .run 安装包最可靠。
#
# 关键设计：装文件本身不需要GPU在场（只有跑CUDA代码才需要），而且如果装到根分区
# （默认 /usr/local/cuda），容器重置就会被清空，每次都要重新下载+装一遍（~5.4GB）。
# 这里改成显式装到持久化数据盘（$DATA_DIR/cuda-toolkit），装一次，跨容器重置永久存活，
# 配合 sources.sh 里的 CUDA_HOME/PATH 设置，重置后不用重装就能直接用。
#
# 默认开启（无卡阶段跑，不占GPU计费时间；不想装就设 INSTALL_CUDA_TOOLKIT=0 跳过）。
# 用法：bash download_cuda_toolkit.sh          # 默认装
#       INSTALL_CUDA_TOOLKIT=0 bash download_cuda_toolkit.sh   # 跳过
set -e

if [ "${INSTALL_CUDA_TOOLKIT:-1}" != "1" ]; then
    echo "[download_cuda_toolkit] 跳过（INSTALL_CUDA_TOOLKIT=0 显式关闭）"
    exit 0
fi

DATA_DIR="${DATA_DIR:-/root/rivermind-data}"
CUDA_VERSION="${CUDA_VERSION:-12.8.0}"
CUDA_RUNFILE="cuda_${CUDA_VERSION}_570.86.10_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_RUNFILE}"
TOOLKIT_INSTALL_PATH="$DATA_DIR/cuda-toolkit"
DOWNLOAD_PATH="$DATA_DIR/tools/$CUDA_RUNFILE"

mkdir -p "$DATA_DIR/tools"

# 验证nvcc不仅"存在"，还要"能跑"且版本号对得上——安装程序退出码是0不代表真的装对了
# （比如中途磁盘满/权限问题，装出来的东西可能不完整）
verify_nvcc() {
    if [ ! -x "$TOOLKIT_INSTALL_PATH/bin/nvcc" ]; then
        echo "[download_cuda_toolkit] 验证失败：$TOOLKIT_INSTALL_PATH/bin/nvcc 不存在"
        return 1
    fi
    local output
    if ! output=$("$TOOLKIT_INSTALL_PATH/bin/nvcc" --version 2>&1); then
        echo "[download_cuda_toolkit] 验证失败：nvcc存在但跑不起来："
        echo "$output"
        return 1
    fi
    if ! echo "$output" | grep -q "release ${CUDA_VERSION%.0}"; then
        echo "[download_cuda_toolkit] 验证失败：nvcc版本号跟预期的 ${CUDA_VERSION} 对不上："
        echo "$output"
        return 1
    fi
    echo "[download_cuda_toolkit] ✅ 验证通过，nvcc 正常可用："
    echo "$output"
    return 0
}

if [ -x "$TOOLKIT_INSTALL_PATH/bin/nvcc" ] && verify_nvcc; then
    echo "[download_cuda_toolkit] 已装在持久化数据盘，跳过重装"
    exit 0
fi

if [ ! -f "$DOWNLOAD_PATH" ]; then
    echo "[download_cuda_toolkit] 下载 CUDA Toolkit ${CUDA_VERSION}（约5.4GB，存到持久化数据盘: $DOWNLOAD_PATH）..."
    curl -L -o "$DOWNLOAD_PATH" "$CUDA_URL"
else
    echo "[download_cuda_toolkit] 安装包已存在，跳过下载: $DOWNLOAD_PATH"
fi

chmod +x "$DOWNLOAD_PATH"

echo "[download_cuda_toolkit] 安装 toolkit 到持久化数据盘（只装toolkit不装驱动，不影响现有GPU驱动）: $TOOLKIT_INSTALL_PATH"
if ! "$DOWNLOAD_PATH" --toolkit --silent --toolkitpath="$TOOLKIT_INSTALL_PATH" --override; then
    echo "[download_cuda_toolkit] 错误：安装程序本身返回非0退出码，安装失败"
    exit 1
fi

echo "[download_cuda_toolkit] 验证安装结果..."
if ! verify_nvcc; then
    echo "[download_cuda_toolkit] 错误：安装程序退出码是0，但验证没通过，安装不完整"
    exit 1
fi

echo "[download_cuda_toolkit] 完成，装在: $TOOLKIT_INSTALL_PATH"
echo "下一步：sources.sh 会自动探测这个路径并接进 PATH/CUDA_HOME，无需手动操作"
