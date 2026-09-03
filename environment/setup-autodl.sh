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

# ---------- 参数解析 ----------
USE_UV=0
if [ "$1" = "--uv" ]; then
  USE_UV=1
  echo "  📦 使用 uv 安装（--uv 已指定）"
else
  echo "  📦 使用 pip 安装（默认；加 --uv 可切 uv）"
fi

# ---------- [0] conda + 安装器 ----------
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

# uv 可选安装（只有 --uv 才需要）
if [ "${USE_UV}" = "1" ]; then
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
  export UV_SYSTEM_PYTHON=1
fi

echo "  ✅ 当前 Python: $(python -V 2>&1)"
if [ "${USE_UV}" = "1" ]; then
  echo "  ✅ uv 版本: $(uv --version 2>&1)"
fi

# ---------- [1] 镜像源配置 ----------
# 主源：清华 TUNA（AutoDL 容器内阿里云出网受限，实测清华稳定）
# 额外源：PyTorch 官方 CUDA wheel（torch+cu124 后缀包只有这里才有）
# 兜底：PyPI 官方（extra-index-url 里加上，清华缺包时自动回退）
echo ""
echo "[1/4] 配置镜像源（清华 TUNA + PyTorch CUDA + PyPI 兜底）..."

PRIMARY_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
EXTRA_MIRRORS="https://download.pytorch.org/whl/cu124 https://pypi.org/simple"

export PIP_INDEX_URL="${PRIMARY_MIRROR}"
export PIP_EXTRA_INDEX_URL="${EXTRA_MIRRORS}"
export UV_INDEX_URL="${PRIMARY_MIRROR}"
export UV_EXTRA_INDEX_URL="${EXTRA_MIRRORS}"

# 持久化 pip 配置（uv 也会读这个）
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<EOF
[global]
index-url = ${PRIMARY_MIRROR}
extra-index-url = ${EXTRA_MIRRORS}
trusted-host = pypi.tuna.tsinghua.edu.cn download.pytorch.org pypi.org
EOF
echo "  → pip 配置已持久化到 ~/.pip/pip.conf"
echo "  ✅ 主源: ${PRIMARY_MIRROR}"
echo "  ✅ 额外: https://download.pytorch.org/whl/cu124（CUDA wheel）"
echo "  ✅ 兜底: https://pypi.org/simple"

# ---------- [2] 安装依赖（common → gpu）----------
echo ""
echo "[2/4] 安装依赖（common + gpu，首次较久）..."

# 构造安装命令：pip vs uv
if [ "${USE_UV}" = "1" ]; then
  # uv 需要 unsafe-best-match 才能从多个源拼版本
  INSTALL="uv pip install --index-strategy unsafe-best-match"
  UNINSTALL="uv pip uninstall"
else
  # pip 默认就是从所有源拼版本，不需要额外参数
  INSTALL="pip install"
  UNINSTALL="pip uninstall"
fi

# common（两边通用轻量包）
${INSTALL} -r "${PROJECT_ROOT}/environment/requirements-common.txt"

# gpu（torch + CUDA 生态 —— 如果已有 CUDA torch 会跳过）
${INSTALL} -r "${PROJECT_ROOT}/environment/requirements-gpu.txt"

# ---------- [2.5] 版本锁定校验 ----------
# requirements-gpu.txt 已锁 unsloth_zoo==2025.3.17（无 torchao 依赖），
# 避免 pip/uv 解析到 2025.10.x+（硬绑 torchao>=0.13.0 且与 torch 2.5.1 不兼容）
echo ""
echo "[2.5/4] 版本锁校验..."
INSTALLED_ZOO=$(python -c "import unsloth_zoo; print(unsloth_zoo.__version__)" 2>/dev/null || echo "未安装")
INSTALLED_TORCHAO=$(python -c "import torchao; print(torchao.__version__)" 2>/dev/null || echo "未安装")
echo "  unsloth_zoo : ${INSTALLED_ZOO}（期望 2025.3.17）"
echo "  torchao     : ${INSTALLED_TORCHAO}（期望 未安装）"
if [ "${INSTALLED_TORCHAO}" != "未安装" ]; then
  echo "  ⚠️  torchao 意外存在，卸载中..."
  ${UNINSTALL} torchao -y 2>/dev/null || true
fi

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
