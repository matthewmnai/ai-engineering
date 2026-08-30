#!/usr/bin/env bash
# ============================================================
# AI Engineering —— AutoDL 机器间资源迁移打包脚本
#
# 迁移优先级：
#   🔴 must   —— models/finetuned + datasets/course（不可再生，必带）
#   🟡 should —— datasets/public（可重新下载但省时间）
#   🟢 optional—— models/pretrained（大，脚本可重新拉）
#   🟢 optional—— checkpoints（断点续训用）
#
# 用法:
#   bash environment/pack_migration.sh                    # 默认: 打包 must + should
#   bash environment/pack_migration.sh --level must       # 只打必带
#   bash environment/pack_migration.sh --level all        # 全量（很慢）
#   bash environment/pack_migration.sh --output /mnt/autodl-tmp/ai-migration.tar.gz
#
# 解包（在新机器上）:
#   cd /root/autodl-tmp/ai-engineering
#   tar xzf ai-migration.tar.gz
#   bash setup.sh   # 补齐缺失的 pretrained 模型
# ============================================================
set -e

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---------- 默认参数 ----------
LEVEL="should"        # must / should / all
OUTPUT="${PROJECT_ROOT}/ai-migration-$(date +%Y%m%d).tar.gz"

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --level) LEVEL="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --help|-h)
      echo "用法: bash environment/pack_migration.sh [--level must|should|all] [--output FILE]"
      echo ""
      echo "  --level must    只打包不可再生的（微调产出 + 课程数据）"
      echo "  --level should  + 公开数据集（默认，省下载时间）"
      echo "  --level all     + 基座模型 + checkpoints（全量，很大）"
      exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# ---------- 构建文件列表 ----------
INCLUDE=()

# 判断目录是否有实际内容（排除 .gitkeep 占位文件）
has_real_content() {
  local dir="$1"
  [ -d "$dir" ] || return 1
  # 过滤掉 .gitkeep 后检查是否还有文件
  local count
  count=$(find "$dir" -type f ! -name '.gitkeep' 2>/dev/null | head -1)
  [ -n "$count" ]
}

# 🔴 必带：微调产出 + 课程数据
if has_real_content "models/finetuned"; then
  INCLUDE+=("models/finetuned")
  echo "🔴 打包: models/finetuned/  (微调产出，不可再生)"
else
  echo "⏭️  跳过: models/finetuned/  (不存在或为空)"
fi

if has_real_content "datasets/course"; then
  INCLUDE+=("datasets/course")
  echo "🔴 打包: datasets/course/  (课程专属数据)"
else
  echo "⏭️  跳过: datasets/course/  (不存在或为空)"
fi

# 🟡 建议带：公开数据集
if [ "$LEVEL" = "should" ] || [ "$LEVEL" = "all" ]; then
  if has_real_content "datasets/public"; then
    INCLUDE+=("datasets/public")
    echo "🟡 打包: datasets/public/  (公开数据集，省下载时间)"
  else
    echo "⏭️  跳过: datasets/public/  (不存在或为空)"
  fi
fi

# 🟢 可不带：基座模型 + checkpoints
if [ "$LEVEL" = "all" ]; then
  if has_real_content "models/pretrained"; then
    INCLUDE+=("models/pretrained")
    echo "🟢 打包: models/pretrained/  (基座模型，很大)"
  fi
  if has_real_content "checkpoints"; then
    INCLUDE+=("checkpoints")
    echo "🟢 打包: checkpoints/  (训练中间状态)"
  fi
fi

# ---------- 检查 ----------
if [ ${#INCLUDE[@]} -eq 0 ]; then
  echo ""
  echo "❌ 没有可打包的目录，请先运行 download_models.py / 微调训练"
  exit 1
fi

# ---------- 打包 ----------
echo ""
echo "📦 输出: ${OUTPUT}"
echo "📋 内容: ${INCLUDE[*]}"

# 计算总大小
TOTAL_SIZE=$(du -sh "${INCLUDE[@]}" 2>/dev/null | tail -1 | cut -f1)
echo "📏 总大小: ${TOTAL_SIZE}"

tar czf "${OUTPUT}" "${INCLUDE[@]}"

FILE_SIZE=$(du -sh "${OUTPUT}" | cut -f1)
echo ""
echo "✅ 打包完成: ${OUTPUT}  (${FILE_SIZE})"
echo ""
echo "💡 迁移到新 AutoDL 机器:"
echo "   1. 用 autodl 的「文件上传」或 scp 传 ${OUTPUT}"
echo "   2. 在新机器上:"
echo "      cd /root/autodl-tmp/ai-engineering"
echo "      tar xzf $(basename ${OUTPUT})"
echo "      bash setup.sh                                    # 补齐环境"
echo "      python environment/download_models.py --target autodl  # 补齐缺失的基座模型"
