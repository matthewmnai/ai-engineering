#!/usr/bin/env bash
# ============================================================
# AI Engineering —— AutoDL / Linux GPU 环境一键部署
# 功能：conda 环境 + common 依赖 + gpu 依赖 + HF 镜像 + 自检
# 运行：
#   bash setup-autodl.sh                 # 标准路径
#   bash setup.sh                        # 根目录软链接（自动路由到此）
# ============================================================
set -e

# ---------- 定位脚本自身（兼容软链接从任意目录调用）----------
SCRIPT_SRC="${BASH_SOURCE[0]}"
while [ -L "${SCRIPT_SRC}" ]; do
  SCRIPT_DIR="$(cd -P "$(dirname "${SCRIPT_SRC}")" && pwd)"
  SCRIPT_SRC="$(readlink "${SCRIPT_SRC}")"
  case "${SCRIPT_SRC}" in
    /*) ;;
    *) SCRIPT_SRC="${SCRIPT_DIR}/${SCRIPT_SRC}" ;;
  esac
done
SCRIPT_DIR="$(cd -P "$(dirname "${SCRIPT_SRC}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="ai-engineering"
PYTHON_VER="3.10"

echo "============================================"
echo " AI Engineering —— AutoDL/GPU 环境初始化"
echo " 项目根: ${PROJECT_ROOT}"
echo " 环境名: ${ENV_NAME}  Python: ${PYTHON_VER}"
echo "============================================"

# ---------- [0] conda 可用性 ----------
if ! command -v conda >/dev/null 2>&1; then
  echo "[0/5] conda 未在 PATH，尝试 source conda.sh ..."
  for candidate in \
    /root/miniconda3/etc/profile.d/conda.sh \
    /opt/conda/etc/profile.d/conda.sh \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
      source "$candidate"; break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 找不到 conda，请先安装 Miniconda/Anaconda"; exit 1
fi

# ---------- [1] conda 环境 ----------
echo ""
echo "[1/5] 检查/创建 conda 环境: ${ENV_NAME} ..."
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "  → 已存在，跳过"
else
  conda create -y -n "${ENV_NAME}" python=${PYTHON_VER} pip
fi
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
echo "  ✅ 已激活: $(python -V 2>&1)"

# ---------- [2] pip 国内源 ----------
echo ""
echo "[2/5] 配置 pip 清华源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ---------- [3] 安装依赖（common → gpu）----------
echo ""
echo "[3/5] 安装依赖（common + gpu，首次较久）..."
pip install --upgrade pip

# common（两边通用轻量包）
pip install -r "${PROJECT_ROOT}/environment/requirements-common.txt"

# gpu（torch + CUDA 生态 —— 如果已有 CUDA torch 会跳过）
pip install -r "${PROJECT_ROOT}/environment/requirements-gpu.txt"

# CUDA 版本提示
CUDA_VER=$(python -c "import torch; print(torch.version.cuda or 'CPU')" 2>/dev/null || echo "unknown")
echo "  ✅ PyTorch CUDA 版本: ${CUDA_VER}"

# ---------- [4] HF 镜像 + 资源目录 ----------
echo ""
echo "[4/5] HF 镜像 + 资源目录..."
export HF_ENDPOINT=https://hf-mirror.com
if ! grep -qxF 'export HF_ENDPOINT=https://hf-mirror.com' ~/.bashrc; then
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
  echo "  → HF 镜像已持久化到 ~/.bashrc"
fi
# 新目录结构：按用途分，方便迁移
mkdir -p /root/autodl-tmp/ai-engineering/models/pretrained
mkdir -p /root/autodl-tmp/ai-engineering/models/finetuned
mkdir -p /root/autodl-tmp/ai-engineering/datasets/public
mkdir -p /root/autodl-tmp/ai-engineering/datasets/course
mkdir -p /root/autodl-tmp/ai-engineering/checkpoints
mkdir -p /root/autodl-tmp/ai-engineering/outputs
mkdir -p /root/autodl-tmp/ai-engineering/logs
echo "  ✅ 资源目录已就绪（AutoDL 数据盘）"
echo "     models/pretrained  基座模型（可重新下载）"
echo "     models/finetuned   微调产出（迁移必带）"
echo "     datasets/public    公开数据集"
echo "     datasets/course   课程专属数据（迁移必带）"

# ---------- [5] 自检 ----------
echo ""
echo "[5/5] 环境自检 ..."
python "${PROJECT_ROOT}/environment/verify_env.py" || true

echo ""
echo "============================================"
echo " ✅ AutoDL 环境部署完成！"
echo " ① conda activate ${ENV_NAME}"
echo " ② cp .env.example .env   （填入 API Key）"
echo " ③ python environment/download_models.py  （按需拉模型）"
echo "============================================"
