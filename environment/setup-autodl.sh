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

# ---------- [1] 双源配置 ----------
# 主源：阿里云镜像（加速通用包下载）
# 额外源：PyTorch 官方 CUDA wheel（torch+cu124 后缀包只有这里才有）
echo ""
echo "[1/4] 配置双源（阿里云 + PyTorch CUDA）..."
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
# 持久化 pip 配置（uv 也会读这个）
mkdir -p ~/.pip
if ! grep -qxF 'index-url = https://mirrors.aliyun.com/pypi/simple' ~/.pip/pip.conf 2>/dev/null; then
  cat >> ~/.pip/pip.conf <<EOF

[global]
index-url = https://mirrors.aliyun.com/pypi/simple
extra-index-url = https://download.pytorch.org/whl/cu124
trusted-host = mirrors.aliyun.com
EOF
  echo "  → pip 双源配置已持久化到 ~/.pip/pip.conf"
fi
echo "  ✅ 主源: https://mirrors.aliyun.com/pypi/simple"
echo "  ✅ CUDA: https://download.pytorch.org/whl/cu124"

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

# ---------- [2.5] torchao 卸载（和 torch 2.5.1 不兼容）----------
# unsloth_zoo 2025.10.x 硬依赖 torchao，但 torchao 0.9~0.16 全部用了 torch.int1
# （这是 torch 2.11 才引入的 dtype），一 import 就 AttributeError。
# 解法：卸载 torchao + 用 --no-deps 重装 unsloth_zoo 绕过依赖声明。
# unsloth 实际用 bitsandbytes 做量化，torchao 可安全移除。
echo ""
echo "[2.5/4] 卸 torchao（和 torch 2.5.1 不兼容）..."
${UNINSTALL} torchao -y 2>/dev/null || true
# 清理残留文件（pip/uv 卸载可能漏掉 dist-info）
rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/torchao"* 2>/dev/null || true
# 绕过 unsloth_zoo 对 torchao 的硬依赖，强制重装
if [ "${USE_UV}" = "1" ]; then
  uv pip install --no-deps "unsloth_zoo>=2025.10.9" 2>/dev/null || true
else
  pip install --no-deps "unsloth_zoo>=2025.10.9" 2>/dev/null || true
fi
echo "  ✅ torchao 已移除，unsloth_zoo 以 --no-deps 模式重装"

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
