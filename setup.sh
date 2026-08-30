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

# ---------- 系统检测 ----------
TARGET=""

# 手动指定优先
if [ "$1" = "--target" ] && [ -n "$2" ]; then
  TARGET="$2"
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  echo "用法: bash setup.sh [--target mac|autodl]"
  echo ""
  echo "自动检测:"
  echo "  Mac        → setup-mac.sh      (common + mac 依赖)"
  echo "  Linux/GPU  → setup-autodl.sh   (common + gpu 依赖)"
  echo ""
  echo "手动指定:"
  echo "  bash setup.sh --target mac"
  echo "  bash setup.sh --target autodl"
  exit 0
fi

# 自动检测
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
    exec bash "${PROJECT_ROOT}/environment/setup-mac.sh" "$@"
    ;;
  autodl|linux|gpu)
    echo "🐧 检测到 Linux 系统 → 路由到 setup-autodl.sh"
    exec bash "${PROJECT_ROOT}/environment/setup-autodl.sh" "$@"
    ;;
  *)
    echo "❌ 未知 target: ${TARGET}，可选: mac / autodl"
    exit 1
    ;;
esac
