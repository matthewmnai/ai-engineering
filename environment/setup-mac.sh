#!/usr/bin/env bash
# ============================================================
# AI Engineering —— Mac 本地环境一键部署
# 功能：conda 环境 + uv 安装 common 依赖 + mac 依赖 + 自检
# （Mac 不装 CUDA torch / bitsandbytes，不拉大模型，只走 API 或本地小 embedding）
# 运行：
#   bash setup-mac.sh                    # 标准路径
#   bash setup.sh                        # 根目录软链接（自动路由到此）
# ============================================================
set -e

# ---------- 定位脚本自身 ----------
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
echo " AI Engineering —— Mac 本地环境初始化"
echo " 项目根: ${PROJECT_ROOT}"
echo " 环境名: ${ENV_NAME}  Python: ${PYTHON_VER}"
echo "============================================"

# ---------- [0] conda ----------
if ! command -v conda >/dev/null 2>&1; then
  echo "[0/5] conda 未在 PATH，尝试 source conda.sh ..."
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh; do
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

# ---------- [2] uv 包管理器 + 国内源 ----------
echo ""
echo "[2/5] 检查 uv ..."
if ! command -v uv >/dev/null 2>&1; then
  echo "  uv 未安装，正在安装..."
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
echo "  ✅ uv 版本: $(uv --version 2>&1)"
export UV_SYSTEM_PYTHON=1
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
echo "  ✅ uv 清华源已配置"

# ---------- [3] 安装依赖（common → mac，不装 gpu）----------
echo ""
echo "[3/5] 安装依赖（common + mac，首次较久）..."

uv pip install -r "${PROJECT_ROOT}/environment/requirements-common.txt"
uv pip install -r "${PROJECT_ROOT}/environment/requirements-mac.txt"

# 确认 torch 是 CPU 版
TORCH_BACKEND=$(python -c "import torch; print('MPS' if torch.backends.mps.is_available() else 'CPU')" 2>/dev/null || echo "unknown")
echo "  ✅ PyTorch 后端: ${TORCH_BACKEND}"

# ---------- [4] HF 镜像 + 本地目录 ----------
echo ""
echo "[4/5] 配置 HF 镜像 + 本地缓存目录..."
export HF_ENDPOINT=https://hf-mirror.com
if ! grep -qxF 'export HF_ENDPOINT=https://hf-mirror.com' ~/.zshrc ~/.bashrc 2>/dev/null; then
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc 2>/dev/null || true
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc 2>/dev/null || true
  echo "  → HF 镜像已持久化"
fi
# 新目录结构：按用途分
mkdir -p "${PROJECT_ROOT}/models/pretrained"
mkdir -p "${PROJECT_ROOT}/models/finetuned"
mkdir -p "${PROJECT_ROOT}/datasets/public"
mkdir -p "${PROJECT_ROOT}/datasets/course"
mkdir -p "${PROJECT_ROOT}/checkpoints"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/logs"
echo "  ✅ 本地目录已就绪"
echo "     models/pretrained  基座模型（可重新下载）"
echo "     models/finetuned   微调产出"
echo "     datasets/public    公开数据集"
echo "     datasets/course   课程专属数据"

# ---------- [5] 自检 ----------
echo ""
echo "[5/5] 环境自检 ..."
python "${PROJECT_ROOT}/environment/verify_env.py" || true

echo ""
echo "============================================"
echo " ✅ Mac 环境部署完成！"
echo " ① conda activate ${ENV_NAME}"
echo " ② cp .env.example .env   （填入 API Key）"
echo " ③ （可选）python environment/download_models.py --target mac"
echo "    只下载 Mac 本地需要的小 embedding 模型"
echo "============================================"
