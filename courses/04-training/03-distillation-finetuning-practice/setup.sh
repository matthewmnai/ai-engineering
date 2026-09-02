#!/usr/bin/env bash
# ============================================================
# 03 · 模型蒸馏与微调实操 —— 课程环境初始化
#
#   step1: 调用项目根目录 setup.sh 初始化运行环境（依赖 + HF 镜像 + 资源目录）
#   step2: 下载本模块所需模型（autodl / 04_training）
#   step3: 下载本模块所需数据集（autodl / 04_training）
#
# 运行:
#   bash setup.sh
# ============================================================
set -e

# ---------- 定位项目根目录（本脚本位于 courses/04-training/03-.../ ）----------
SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "============================================"
echo " 03 · 蒸馏与微调 —— 课程环境初始化"
echo " 项目根: ${PROJECT_ROOT}"
echo "============================================"

# ---------- [step1] 调用根目录 setup.sh 初始化环境 ----------
echo ""
echo "[1/3] 调用项目根 setup.sh 初始化环境 ..."
bash "${PROJECT_ROOT}/setup.sh"

# ---------- [step2] 下载本模块模型 ----------
echo ""
echo "[2/3] 下载本模块模型（autodl / 04_training）..."
python "${PROJECT_ROOT}/environment/download_models.py" --target autodl 04_training

# ---------- [step3] 下载本模块数据集 ----------
echo ""
echo "[3/3] 下载本模块数据集（autodl / 04_training）..."
python "${PROJECT_ROOT}/environment/download_datasets.py" --target autodl 04_training

echo ""
echo "============================================"
echo " ✅ 课程环境就绪！"
echo "  下一步: 参考 README.md 进入 src/ 开始练习"
echo "============================================"
