#!/usr/bin/env bash
# AlphaDiablo 开发环境一键引导脚本
# 用途:从零(或增量)重建 DevilutionX 无头开发环境并跑通冒烟测试。
# 幂等可重复执行;上游 clone 放在系统临时目录(铁律 #3:不进项目文件夹、不开仓库)。
#
# 用法:  ./bootstrap.sh
# 产出:  $DEV_DIR/build/devilutionx.app        游戏本体(原生 Apple Silicon)
#         $DEV_DIR/build/timedemo_test          无头确定性回放测试(环境健康探针)
#
# 前置(本机已满足,重装机时脚本会自动补):
#   - Homebrew;游戏数据 MPQ 已在 ~/Library/Application Support/diasurgical/devilution/
set -euo pipefail

export PATH="/opt/homebrew/bin:$PATH"
DEV_DIR="${TMPDIR:-/tmp}/alphadiablo-dev/devilutionX"
DATA_DIR="$HOME/Library/Application Support/diasurgical/devilution"
JOBS="$(sysctl -n hw.physicalcpu)"
# 钉死的上游引擎版本 —— 排行榜与全部测试基线所用的构建源。
# 升级引擎是有意识的决定:改这个 SHA,然后重跑金标准评估、重建排行榜。
ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"

echo "==> [1/5] 检查游戏数据 ($DATA_DIR)"
ls "$DATA_DIR/diabdat.mpq" >/dev/null 2>&1 || ls "$DATA_DIR/spawn.mpq" >/dev/null 2>&1 \
  || { echo "错误:缺少 diabdat.mpq 或 spawn.mpq,先准备数据文件"; exit 1; }

echo "==> [2/5] 获取上游源码 @ ${ENGINE_REF:0:12} -> $DEV_DIR"
if [ ! -d "$DEV_DIR/.git" ]; then
  mkdir -p "$DEV_DIR"
  git -C "$DEV_DIR" init -q
  git -C "$DEV_DIR" remote add origin https://github.com/diasurgical/devilutionX.git
fi
if ! git -C "$DEV_DIR" rev-parse --quiet --verify "$ENGINE_REF^{commit}" >/dev/null; then
  git -C "$DEV_DIR" fetch --depth 1 origin "$ENGINE_REF"
fi
if [ "$(git -C "$DEV_DIR" rev-parse HEAD 2>/dev/null || true)" != "$ENGINE_REF" ]; then
  # 换版本时丢弃工作区改动对齐过去(补丁由 build.sh 幂等重涂);
  # 已在钉死版本上则不动工作区,免得每次都触发全量重编译
  git -C "$DEV_DIR" reset --hard -q "$ENGINE_REF"
fi

echo "==> [3/5] Homebrew 依赖(官方 Brewfile,幂等)"
brew bundle install --file="$DEV_DIR/Brewfile" || echo "(个别包锁冲突可忽略,下一步编译会兜底验证)"

# CI 模式:只负责 clone + 依赖,引擎编译统一交给 build.sh(免得同一引擎编两遍)
if [ "${BOOTSTRAP_CLONE_ONLY:-0}" = "1" ]; then
  echo "✅ clone-only 模式:源码就位 @ ${ENGINE_REF:0:12},跳过引擎编译与冒烟"
  exit 0
fi

echo "==> [4/5] 编译 devilutionx + timedemo_test(注意:不要用 make all,macOS 上测试资源目标必失败)"
cmake -S "$DEV_DIR" -B "$DEV_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEV_DIR/build" -j "$JOBS" --target devilutionx
cmake --build "$DEV_DIR/build" -j "$JOBS" --target timedemo_test

echo "==> [5/5] 冒烟测试:无头确定性回放"
"$DEV_DIR/build/timedemo_test"

echo ""
echo "✅ 环境就绪"
echo "   引擎:   $DEV_DIR/build/devilutionx.app"
echo "   试玩:   open '$DEV_DIR/build/devilutionx.app'   (GUI,完整版数据)"
echo "   Python: source \"$(dirname "$0")/.venv/bin/activate\"   (torch/gymnasium/SB3)"
