#!/usr/bin/env bash
# DiabloGym 一键构建:引擎(共享库+资产)+ pybind11 桥
set -euo pipefail
cd "$(dirname "$0")"
[ "$(uname -s)" = "Darwin" ] && export PATH="/opt/homebrew/bin:$PATH" || true

# Python 解释器解析顺序:$PYTHON 环境变量 > ./.venv > ../.venv
VENV_PY="${PYTHON:-}"
[ -x "$VENV_PY" ] || VENV_PY="$PWD/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$(cd .. && pwd)/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "找不到 Python venv(设 \$PYTHON 或在仓库根/上级建 .venv)"; exit 1; }
# CMake must receive an absolute venv path while preserving the venv symlink
# itself (realpath would collapse it to the base interpreter and lose sys.prefix).
VENV_PY="$(cd "$(dirname "$VENV_PY")" && pwd -P)/$(basename "$VENV_PY")"
PYBIND11_DIR="$("$VENV_PY" -m pybind11 --cmakedir)"
PYTHON_EXT_SUFFIX="$("$VENV_PY" -c \
  'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")')"
[ -n "$PYTHON_EXT_SUFFIX" ] || { echo "当前 Python 没有 EXT_SUFFIX:$VENV_PY"; exit 1; }
PYTHON_INCLUDE_DIR="$("$VENV_PY" -c \
  'import sysconfig; print(sysconfig.get_path("include") or "")')"
[ -d "$PYTHON_INCLUDE_DIR" ] || { echo "当前 Python headers 不存在:$PYTHON_INCLUDE_DIR"; exit 1; }
DEVX="${DEVILUTIONX_SRC:-${TMPDIR:-/tmp}/alphadiablo-dev/devilutionX}"
[ -d "$DEVX/Source" ] || { echo "引擎源码缺失($DEVX),先运行 ./bootstrap.sh"; exit 1; }
if [ "$(uname -s)" = "Darwin" ]; then JOBS="$(sysctl -n hw.physicalcpu)"; else JOBS="$(nproc)"; fi
ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"

ACTUAL_REF="$(git -C "$DEVX" rev-parse HEAD)"
[ "$ACTUAL_REF" = "$ENGINE_REF" ] || {
  echo "引擎 HEAD 漂移:$ACTUAL_REF(期望 $ENGINE_REF);请重新运行 ./bootstrap.sh"
  exit 1
}

# 幂等应用引擎补丁(目前:无头城镇贴图回落修复,可回馈上游)
for patch in patches/*.patch; do
  if git -C "$DEVX" apply --ignore-space-change --reverse --check "$PWD/$patch" 2>/dev/null; then
    echo "补丁已在位: $patch"
  else
    git -C "$DEVX" apply --ignore-space-change "$PWD/$patch" && echo "已应用补丁: $patch"
  fi
done

# 用独立临时 index 从钉死 HEAD 重放登记补丁，再与真实工作树比较。
# 这样即使额外改动落在同一个已补丁文件里，也不会被“补丁已在位”误放行。
EXPECTED_INDEX="$(mktemp)"
rm -f "$EXPECTED_INDEX"
trap 'rm -f "$EXPECTED_INDEX"' EXIT
GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" read-tree "$ENGINE_REF"
for patch in patches/*.patch; do
  GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" apply --ignore-space-change \
    --cached "$PWD/$patch"
done
if ! GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" diff --quiet --; then
  echo "引擎工作树含登记补丁之外的漂移，拒绝构建:"
  GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" diff --stat --
  echo "恢复专用临时 clone: BOOTSTRAP_CLEAN=1 ./bootstrap.sh"
  exit 1
fi
UNTRACKED="$(git -C "$DEVX" ls-files --others --exclude-standard)"
[ -z "$UNTRACKED" ] || {
  echo "引擎工作树含未登记文件，拒绝构建:"
  echo "$UNTRACKED"
  echo "恢复专用临时 clone: BOOTSTRAP_CLEAN=1 ./bootstrap.sh"
  exit 1
}
rm -f "$EXPECTED_INDEX"
trap - EXIT

OSX_CMAKE_ARGS=()
OSX_DEPLOYMENT_TARGET=""
DEPLOYMENT_SOURCE=""
if [ "$(uname -s)" = "Darwin" ]; then
  OSX_DEPLOYMENT_TARGET="${CMAKE_OSX_DEPLOYMENT_TARGET:-${MACOSX_DEPLOYMENT_TARGET:-}}"
  if [ -n "$OSX_DEPLOYMENT_TARGET" ]; then
    DEPLOYMENT_SOURCE="user-explicit"
    echo "macOS deployment target(显式): $OSX_DEPLOYMENT_TARGET;将审计全部非系统 dylib"
  else
    OSX_DEPLOYMENT_TARGET="$(sw_vers -productVersion)"
    DEPLOYMENT_SOURCE="host-default"
    echo "macOS deployment target(宿主): $OSX_DEPLOYMENT_TARGET"
  fi
  [[ "$OSX_DEPLOYMENT_TARGET" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || {
    echo "非法 macOS deployment target:$OSX_DEPLOYMENT_TARGET"
    exit 1
  }
  OSX_CMAKE_ARGS=(
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=$OSX_DEPLOYMENT_TARGET"
    "-DALPHADIABLO_DEPLOYMENT_SOURCE=$DEPLOYMENT_SOURCE"
  )
fi

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEVILUTIONX_SRC="$DEVX" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPython_EXECUTABLE="$VENV_PY" \
  -DALPHADIABLO_EXPECTED_PYTHON_EXECUTABLE="$VENV_PY" \
  -DALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX="$PYTHON_EXT_SUFFIX" \
  -DALPHADIABLO_EXPECTED_PYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
  "${OSX_CMAKE_ARGS[@]}"

BUILD_IDENTITY="build/alphadiablo-python-build.txt"
[ -f "$BUILD_IDENTITY" ] || { echo "CMake 未产出 Python 构建身份:$BUILD_IDENTITY"; exit 1; }
"$VENV_PY" - "$BUILD_IDENTITY" "$VENV_PY" "$PYTHON_EXT_SUFFIX" \
  "$PYTHON_INCLUDE_DIR" "$OSX_DEPLOYMENT_TARGET" "$DEPLOYMENT_SOURCE" <<'PY'
import pathlib
import sys

manifest = {}
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, sep, value = line.partition("=")
    if not sep or key in manifest:
        raise SystemExit(f"非法 CMake Python 构建身份行:{line!r}")
    manifest[key] = value
expected = {"executable": sys.argv[2], "ext_suffix": sys.argv[3]}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(
            f"CMake Python 构建身份漂移:{key}={manifest.get(key)!r},期望 {value!r}")
includes = manifest.get("include_dirs", "").split(";")
if sys.argv[4] not in includes:
    raise SystemExit(
        f"CMake Python headers 漂移:{includes!r},期望包含 {sys.argv[4]!r}")
if sys.argv[5]:
    deployment = {
        "deployment_target": sys.argv[5], "deployment_source": sys.argv[6]}
    for key, value in deployment.items():
        if manifest.get(key) != value:
            raise SystemExit(
                f"CMake deployment 身份漂移:{key}={manifest.get(key)!r},"
                f"期望 {value!r}")
PY

# 注意:分目标构建,不用 all(macOS 上引擎测试资源目标必失败)
cmake --build build -j "$JOBS" --target devilutionx   # 出 .app → 运行时资产
cmake --build build -j "$JOBS" --target _diablogym    # pybind11 桥

BRIDGE="build/_diablogym${PYTHON_EXT_SUFFIX}"
[ -f "$BRIDGE" ] || {
  echo "构建未产出当前 Python ABI 对应模块:$BRIDGE"
  find build -maxdepth 1 -type f -name '_diablogym*.so' -print
  exit 1
}

if [ "$(uname -s)" = "Darwin" ]; then
  ENGINE_DYLIB="build/engine/liblibdevilutionx_so.dylib"
  GAME_BINARY="build/engine/devilutionx.app/Contents/MacOS/devilutionx"
  [ -f "$ENGINE_DYLIB" ] || { echo "找不到嵌入引擎 dylib:$ENGINE_DYLIB"; exit 1; }
  [ -f "$GAME_BINARY" ] || { echo "找不到资源宿主程序:$GAME_BINARY"; exit 1; }
  "$VENV_PY" cmake/audit_macos_minos.py \
    --deployment-target "$OSX_DEPLOYMENT_TARGET" \
    --search-root build \
    "$BRIDGE" "$ENGINE_DYLIB" "$GAME_BINARY"
else
  ENGINE_SO="build/engine/liblibdevilutionx_so.so"
  [ -f "$ENGINE_SO" ] || { echo "找不到嵌入引擎 so:$ENGINE_SO"; exit 1; }
  # 评测协议(eval_contract/env.py)钉死资产路径为 devilutionx.app/Contents/Resources;
  # Linux 上引擎资产落在 build/engine/assets,这里以真实拷贝(禁符号链接)摆出同一布局
  RES="build/engine/devilutionx.app/Contents/Resources"
  [ -d "build/engine/assets" ] || { echo "找不到引擎资产目录 build/engine/assets"; exit 1; }
  rm -rf "$RES" && mkdir -p "$RES"
  cp -a build/engine/assets/. "$RES/"
  echo "资产已布局: $RES ($(find "$RES" -type f | wc -l) files)"
fi

echo ""
echo "✅ 构建完成"
ls -lh "$BRIDGE"
echo "冒烟测试:  $VENV_PY tests/smoke_random_agent.py"
