#!/bin/bash
# [wo GPU] 下载 base 模型 / 训练数据集 / JustRL 评测代码。纯IO，不需要GPU。
# 用法：DATA_DIR=/root/rivermind-data bash download_data_and_code.sh
#
# 这是串行兼容入口（单独跑这个脚本，或者不在乎并行加速时用）。step1_wo_gpu.sh
# 会绕开这个文件，直接并行拉起下面三个子脚本（各自独立、互不依赖），加速下载。
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

bash bootstrap_uv.sh
bash download_model.sh
bash download_dataset.sh
bash clone_justrl.sh
