#!/usr/bin/env bash
# ============================================================
# AI Engineering —— 跨平台安装路由脚本
# 自动检测当前系统，转发到对应 setup-*.sh
#
#   Mac        → setup-mac.sh
#   Linux/GPU  → setup-autodl.sh
#   也可以手动指定: bash setup.sh --target mac
# ============================================================
set -e

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# ---------- 参数解析 ----------
TARGET=""
USE_UV=0   # 默认用 pip，加 --uv 才切 uv

while [ $# -gt 0 ]; do
  case "$1" in
    --target)
      TARGET="$2"; shift 2 ;;
    --uv)
      USE_UV=1; shift ;;
    --help|-h)
      echo "用法: bash setup.sh [--target mac|autodl] [--uv]"
      echo ""
      echo "  --target mac|autodl   手动指定平台（默认自动检测）"
      echo "  --uv                  使用 uv 安装（默认用 pip）"
      echo ""
      echo "示例:"
      echo "  bash setup.sh                  # 自动检测 + pip"
      echo "  bash setup.sh --uv             # 自动检测 + uv"
      echo "  bash setup.sh --target mac     # Mac + pip"
      echo "  bash setup.sh --target autodl --uv"
      exit 0 ;;
    *)
      echo "❌ 未知参数: $1，可选: --target, --uv, --help"
      exit 1 ;;
  esac
done

# ---------- 系统检测 ----------
if [ -z "${TARGET}" ]; then
  case "$(uname -s)" in
    Darwin)  TARGET="mac" ;;
    Linux)   TARGET="autodl" ;;
    *)       TARGET="autodl" ;;
  esac
fi

case "${TARGET}" in
  mac)
    echo "🍎 检测到 Mac 系统 → 路由到 setup-mac.sh"
    if [ "${USE_UV}" = "1" ]; then
      exec bash "${PROJECT_ROOT}/environment/setup-mac.sh" --uv
    else
      exec bash "${PROJECT_ROOT}/environment/setup-mac.sh"
    fi
    ;;
  autodl|linux|gpu)
    echo "🐧 检测到 Linux 系统 → 路由到 setup-autodl.sh"
    if [ "${USE_UV}" = "1" ]; then
      exec bash "${PROJECT_ROOT}/environment/setup-autodl.sh" --uv
    else
      exec bash "${PROJECT_ROOT}/environment/setup-autodl.sh"
    fi
    ;;
  *)
    echo "❌ 未知 target: ${TARGET}，可选: mac / autodl"
    exit 1
    ;;
esac
