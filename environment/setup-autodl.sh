#!/usr/bin/env bash
# ============================================================
# AI Engineering —— AutoDL / Linux GPU 环境一键部署
# 功能：base 环境直接用 uv 安装 common + gpu 依赖 + HF 镜像 + 自检
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

echo "============================================"
echo " AI Engineering —— AutoDL/GPU 环境初始化"
echo " 项目根: ${PROJECT_ROOT}"
echo " 环境: conda base"
echo "============================================"

# ---------- [0] conda + uv ----------
# conda.sh source（确保 python/pip 在 PATH）
for candidate in \
  /root/miniconda3/etc/profile.d/conda.sh \
  /opt/conda/etc/profile.d/conda.sh \
  "$HOME/miniconda3/etc/profile.d/conda.sh" \
  "$HOME/anaconda3/etc/profile.d/conda.sh"; do
  if [ -f "$candidate" ]; then
    source "$candidate"; break
  fi
done
if ! command -v python >/dev/null 2>&1; then
  echo "❌ 找不到 python，请先安装 Miniconda/Anaconda"; exit 1
fi

# uv 安装（优先官方脚本，pip fallback）
if ! command -v uv >/dev/null 2>&1; then
  echo "[0/4] uv 未安装，尝试自动安装..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  elif command -v pip >/dev/null 2>&1; then
    pip install uv
  else
    echo "❌ 找不到 curl 和 pip，无法安装 uv"; exit 1
  fi
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv 安装失败，请手动安装: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1
fi

# uv pip 在 conda base（非 venv）里必须加 --system 或 export UV_SYSTEM_PYTHON=1
export UV_SYSTEM_PYTHON=1

echo "  ✅ 当前 Python: $(python -V 2>&1)"
echo "  ✅ uv 版本: $(uv --version 2>&1)"

# ---------- [1] uv 国内源 ----------
echo ""
echo "[1/4] 配置 uv 清华源..."
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# ---------- [2] 安装依赖（common → gpu）----------
echo ""
echo "[2/4] 安装依赖（common + gpu，首次较久）..."

# common（两边通用轻量包）
uv pip install -r "${PROJECT_ROOT}/environment/requirements-common.txt"

# gpu（torch + CUDA 生态 —— 如果已有 CUDA torch 会跳过）
uv pip install -r "${PROJECT_ROOT}/environment/requirements-gpu.txt"

# CUDA 版本提示
CUDA_VER=$(python -c "import torch; print(torch.version.cuda or 'CPU')" 2>/dev/null || echo "unknown")
echo "  ✅ PyTorch CUDA 版本: ${CUDA_VER}"

# ---------- [3] HF 镜像 + 资源目录 ----------
echo ""
echo "[3/4] HF 镜像 + 资源目录..."
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

# ---------- [4] 自检 ----------
echo ""
echo "[4/4] 环境自检 ..."
python "${PROJECT_ROOT}/environment/verify_env.py" || true

echo ""
echo "============================================"
echo " ✅ AutoDL 环境部署完成！"
echo " ① cp .env.example .env   （填入 API Key）"
echo " ② python environment/download_models.py  （按需拉模型）"
echo "============================================"
