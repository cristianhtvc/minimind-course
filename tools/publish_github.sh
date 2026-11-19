#!/usr/bin/env bash
# ==============================================================================
# 一键将 Learn-MiniMind 发布为你的 GitHub 个人主页 (GitHub Pages)
#
# 用法 1（作为你的主个人主页 https://<username>.github.io/ ）：
#   bash tools/publish_github.sh <你的GitHub用户名>
#
# 用法 2（作为子项目主页 https://<username>.github.io/learn-minimind/ ）：
#   bash tools/publish_github.sh <你的GitHub用户名> learn-minimind
#
# 用法 3（使用 SSH 地址推送）：
#   USE_SSH=1 bash tools/publish_github.sh <你的GitHub用户名>
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

USERNAME="$1"
REPO_NAME="${2:-${USERNAME}.github.io}"

if [ -z "$USERNAME" ]; then
  echo "=============================================================================="
  echo "🧭 Learn-MiniMind GitHub 个人主页一键发布工具"
  echo "=============================================================================="
  echo "请传入你的 GitHub 用户名："
  echo ""
  echo "  方式 A（推荐：作为你的 GitHub 个人根主页 https://<用户名>.github.io/ ）："
  echo "    1. 在 GitHub 新建一个公开仓库，命名为：<你的用户名>.github.io"
  echo "    2. 运行：bash tools/publish_github.sh <你的用户名>"
  echo ""
  echo "  方式 B（作为独立项目主页 https://<用户名>.github.io/learn-minimind/ ）："
  echo "    1. 在 GitHub 新建一个公开仓库，命名为：learn-minimind"
  echo "    2. 运行：bash tools/publish_github.sh <你的用户名> learn-minimind"
  echo "=============================================================================="
  exit 1
fi

# 确保内嵌的 minimind 不包含子仓库 .git 目录，以便所有源码完整上传
if [ -d "$ROOT_DIR/minimind/.git" ]; then
  rm -rf "$ROOT_DIR/minimind/.git"
fi

if [ ! -d "$ROOT_DIR/.git" ]; then
  git init -b main
fi

if [ -z "$(git config user.name)" ]; then
  git config user.name "$USERNAME"
fi
if [ -z "$(git config user.email)" ]; then
  git config user.email "${USERNAME}@users.noreply.github.com"
fi

git add -A
if ! git diff --cached --quiet; then
  git commit -m "feat: publish Learn-MiniMind interactive LLM course & multi-device homepage"
fi

if [ "${USE_SSH:-0}" = "1" ]; then
  REMOTE_URL="git@github.com:${USERNAME}/${REPO_NAME}.git"
else
  REMOTE_URL="https://github.com/${USERNAME}/${REPO_NAME}.git"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

echo "🚀 正在推送到 $REMOTE_URL (分支: main)..."
git branch -M main
git push -u origin main

if [ "$REPO_NAME" = "${USERNAME}.github.io" ]; then
  PAGE_URL="https://${USERNAME}.github.io/"
else
  PAGE_URL="https://${USERNAME}.github.io/${REPO_NAME}/"
fi

echo ""
echo "=============================================================================="
echo "🎉 推送成功！"
echo "👉 你的多设备端学习主页地址： $PAGE_URL"
echo ""
echo "💡 首次开启提示：进入 GitHub 仓库 -> Settings -> Pages："
echo "   - Build and deployment -> Source 选择 'GitHub Actions'（或 'Deploy from a branch' -> main / root）"
echo "   - 等待约 30 秒即可在电脑、iPad、手机浏览器直接访问 $PAGE_URL 学习！"
echo "=============================================================================="
