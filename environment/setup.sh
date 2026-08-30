#!/usr/bin/env bash
# ============================================================
# AI Engineering 项目 —— 一键环境部署
# 功能：创建/复用 conda 环境 + 装依赖 + 配镜像 + 自检
# 运行：bash environment/setup.sh
# 注意：脚本会自动 source ~/.bashrc，conda activate 才能生效
# ============================================================
set -e

# ---------- 定位脚本自身（兼容软链接被从任意目录调用的场景）----------
# 先拿到真实脚本路径（解析软链接链），再回溯 project root
SCRIPT_SRC="${BASH_SOURCE[0]}"
while [ -L "${SCRIPT_SRC}" ]; do
  SCRIPT_DIR="$(cd -P "$(dirname "${SCRIPT_SRC}")" && pwd)"
  SCRIPT_SRC="$(readlink "${SCRIPT_SRC}")"
  case "${SCRIPT_SRC}" in
    /*) ;;  # 绝对路径
    *) SCRIPT_SRC="${SCRIPT_DIR}/${SCRIPT_SRC}" ;;  # 相对路径 → 拼上软链接所在目录
  esac
done
SCRIPT_DIR="$(cd -P "$(dirname "${SCRIPT_SRC}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 把后续所有操作锁定在 PROJECT_ROOT 下，避免 cwd 错乱
cd "${PROJECT_ROOT}"

ENV_NAME="ai-engineering"
PYTHON_VER="3.10"

echo "============================================"
echo " AI Engineering —— 环境初始化"
echo " 项目根: ${PROJECT_ROOT}"
echo " 当前目录: $(pwd)"
echo " 环境名: ${ENV_NAME}  Python: ${PYTHON_VER}"
echo "============================================"

# ---------- [0] 确保 conda 可用（兼容 AutoDL / 常规机器）----------
# AutoDL 默认 shell 非 login shell，conda 可能未初始化
if ! command -v conda >/dev/null 2>&1; then
  echo "[0/5] conda 未在 PATH 中，尝试 source conda.sh ..."
  for candidate in \
    /root/miniconda3/etc/profile.d/conda.sh \
    /opt/conda/etc/profile.d/conda.sh \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
      echo "  → 找到 conda.sh: $candidate"
      source "$candidate"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 找不到 conda，请先安装 Miniconda/Anaconda"
  exit 1
fi

# ---------- [1] 创建 / 复用 conda 环境 ----------
echo ""
echo "[1/5] 检查/创建 conda 环境: ${ENV_NAME} ..."
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "  → 环境已存在，跳过创建"
else
  conda create -y -n "${ENV_NAME}" python=${PYTHON_VER} pip
fi
# 关键：bash -c 子 shell 里 source conda.sh 再 activate
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
source "${CONDA_SH}"
conda activate "${ENV_NAME}"
echo "  ✅ 已激活: $(python -V 2>&1)"

# ---------- [2] 配置 pip 国内源 ----------
echo ""
echo "[2/5] 配置 pip 国内源（清华）..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ---------- [3] 安装基础依赖 ----------
echo ""
echo "[3/5] 安装基础依赖（首次会较久，后续增量安装会快）..."
REQS="${PROJECT_ROOT}/environment/requirements-base.txt"
pip install --upgrade pip
pip install -r "${REQS}"

# ---------- [4] 配置 HuggingFace 镜像（持久化到 ~/.bashrc）----------
echo ""
echo "[4/5] 配置 HuggingFace 镜像 ..."
export HF_ENDPOINT=https://hf-mirror.com
if ! grep -qxF 'export HF_ENDPOINT=https://hf-mirror.com' ~/.bashrc; then
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
  echo "  → 已追加到 ~/.bashrc"
else
  echo "  → ~/.bashrc 中已存在，跳过"
fi

# 创建资源目录（数据盘，AutoDL 系统盘小）
mkdir -p /root/autodl-tmp/ai-engineering/hf_cache
mkdir -p /root/autodl-tmp/ai-engineering/data
mkdir -p /root/autodl-tmp/ai-engineering/logs

# ---------- [5] 环境自检 ----------
echo ""
echo "[5/5] 环境自检 ..."
python "${PROJECT_ROOT}/environment/verify_env.py" || true

echo ""
echo "============================================"
echo " ✅ 环境部署完成！"
echo " ① conda activate ${ENV_NAME}"
echo " ② cp .env.example .env    （然后填入 API Key）"
echo " ③ python environment/download_models.py  （按需拉模型）"
echo "============================================"
