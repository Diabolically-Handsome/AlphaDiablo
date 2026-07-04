#!/usr/bin/env bash
# DiabloGym 一键构建:引擎(共享库+资产)+ pybind11 桥
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

# Python 解释器解析顺序:$PYTHON 环境变量 > ./.venv > ../.venv
VENV_PY="${PYTHON:-}"
[ -x "$VENV_PY" ] || VENV_PY="$PWD/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$(cd .. && pwd)/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "找不到 Python venv(设 \$PYTHON 或在仓库根/上级建 .venv)"; exit 1; }
PYBIND11_DIR="$("$VENV_PY" -m pybind11 --cmakedir)"
DEVX="${DEVILUTIONX_SRC:-$TMPDIR/alphadiablo-dev/devilutionX}"
[ -d "$DEVX/Source" ] || { echo "引擎源码缺失($DEVX),先运行 ./bootstrap.sh"; exit 1; }
JOBS="$(sysctl -n hw.physicalcpu)"

# 幂等应用引擎补丁(目前:无头城镇贴图回落修复,可回馈上游)
for patch in patches/*.patch; do
  if git -C "$DEVX" apply --reverse --check "$PWD/$patch" 2>/dev/null; then
    echo "补丁已在位: $patch"
  else
    git -C "$DEVX" apply "$PWD/$patch" && echo "已应用补丁: $patch"
  fi
done

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEVILUTIONX_SRC="$DEVX" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPython_EXECUTABLE="$VENV_PY"

# 注意:分目标构建,不用 all(macOS 上引擎测试资源目标必失败)
cmake --build build -j "$JOBS" --target devilutionx   # 出 .app → 运行时资产
cmake --build build -j "$JOBS" --target _diablogym    # pybind11 桥

echo ""
echo "✅ 构建完成"
ls -lh build/_diablogym*.so
echo "冒烟测试:  $VENV_PY tests/smoke_random_agent.py"
