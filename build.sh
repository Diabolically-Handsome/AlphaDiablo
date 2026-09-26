#!/usr/bin/env bash
# DiabloGym one-step build: engine (shared library + assets) + pybind11 bridge
set -euo pipefail
cd "$(dirname "$0")"
# A candidate can be built without replacing the installed runtime in build/.
BUILD_DIR="${ALPHADIABLO_BUILD_DIR:-build}"
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd -P)"
[ "$(uname -s)" = "Darwin" ] && export PATH="/opt/homebrew/bin:$PATH" || true

# Python interpreter lookup order: $PYTHON environment variable > ./.venv > ../.venv
VENV_PY="${PYTHON:-}"
[ -x "$VENV_PY" ] || VENV_PY="$PWD/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$(cd .. && pwd)/.venv/bin/python"
[ -x "$VENV_PY" ] || { echo "Python venv not found (set \$PYTHON or create .venv in the repo root or its parent)"; exit 1; }
# CMake must receive an absolute venv path while preserving the venv symlink
# itself (realpath would collapse it to the base interpreter and lose sys.prefix).
VENV_PY="$(cd "$(dirname "$VENV_PY")" && pwd -P)/$(basename "$VENV_PY")"
PYBIND11_DIR="$("$VENV_PY" -m pybind11 --cmakedir)"
PYTHON_EXT_SUFFIX="$("$VENV_PY" -c \
  'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")')"
[ -n "$PYTHON_EXT_SUFFIX" ] || { echo "The current Python has no EXT_SUFFIX: $VENV_PY"; exit 1; }
PYTHON_INCLUDE_DIR="$("$VENV_PY" -c \
  'import sysconfig; print(sysconfig.get_path("include") or "")')"
[ -d "$PYTHON_INCLUDE_DIR" ] || { echo "The current Python headers do not exist: $PYTHON_INCLUDE_DIR"; exit 1; }
DEVX="${DEVILUTIONX_SRC:-${TMPDIR:-/tmp}/alphadiablo-dev/devilutionX}"
[ -d "$DEVX/Source" ] || { echo "Engine source missing ($DEVX); run ./bootstrap.sh first"; exit 1; }
if [ "$(uname -s)" = "Darwin" ]; then JOBS="$(sysctl -n hw.physicalcpu)"; else JOBS="$(nproc)"; fi
ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"

ACTUAL_REF="$(git -C "$DEVX" rev-parse HEAD)"
[ "$ACTUAL_REF" = "$ENGINE_REF" ] || {
  echo "Engine HEAD drift: $ACTUAL_REF (expected $ENGINE_REF); rerun ./bootstrap.sh"
  exit 1
}

# Idempotently apply the registered engine patches (patches/*.patch)
for patch in patches/*.patch; do
  if git -C "$DEVX" apply --ignore-space-change --reverse --check "$PWD/$patch" 2>/dev/null; then
    echo "Patch already applied: $patch"
  elif git -C "$DEVX" apply --ignore-space-change --check "$PWD/$patch" 2>/dev/null; then
    git -C "$DEVX" apply --ignore-space-change "$PWD/$patch" && echo "Applied patch: $patch"
  else
    # A later registered patch can change an earlier patch's context. The
    # complete expected-index comparison below is authoritative in that case.
    echo "Patch context overlaps; deferring to the full patch-stack check: $patch"
  fi
done

# Replay the registered patches from the pinned HEAD into a separate temporary index, then compare with the real work tree.
# This way an extra change inside an already patched file cannot slip through as "patch already applied".
EXPECTED_INDEX="$(mktemp)"
rm -f "$EXPECTED_INDEX"
trap 'rm -f "$EXPECTED_INDEX"' EXIT
GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" read-tree "$ENGINE_REF"
for patch in patches/*.patch; do
  GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" apply --ignore-space-change \
    --cached "$PWD/$patch"
done
if ! GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" diff --quiet --; then
  echo "Engine work tree has drift beyond the registered patches; refusing to build:"
  GIT_INDEX_FILE="$EXPECTED_INDEX" git -C "$DEVX" diff --stat --
  echo "To restore the dedicated temporary clone: BOOTSTRAP_CLEAN=1 ./bootstrap.sh"
  exit 1
fi
UNTRACKED="$(git -C "$DEVX" ls-files --others --exclude-standard)"
[ -z "$UNTRACKED" ] || {
  echo "Engine work tree has unregistered files; refusing to build:"
  echo "$UNTRACKED"
  echo "To restore the dedicated temporary clone: BOOTSTRAP_CLEAN=1 ./bootstrap.sh"
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
    echo "macOS deployment target (explicit): $OSX_DEPLOYMENT_TARGET; auditing all non-system dylibs"
  else
    OSX_DEPLOYMENT_TARGET="$(sw_vers -productVersion)"
    DEPLOYMENT_SOURCE="host-default"
    echo "macOS deployment target (host): $OSX_DEPLOYMENT_TARGET"
  fi
  [[ "$OSX_DEPLOYMENT_TARGET" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || {
    echo "Invalid macOS deployment target: $OSX_DEPLOYMENT_TARGET"
    exit 1
  }
  OSX_CMAKE_ARGS=(
    "-DCMAKE_OSX_DEPLOYMENT_TARGET=$OSX_DEPLOYMENT_TARGET"
    "-DALPHADIABLO_DEPLOYMENT_SOURCE=$DEPLOYMENT_SOURCE"
  )
fi

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEVILUTIONX_SRC="$DEVX" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPython_EXECUTABLE="$VENV_PY" \
  -DALPHADIABLO_EXPECTED_PYTHON_EXECUTABLE="$VENV_PY" \
  -DALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX="$PYTHON_EXT_SUFFIX" \
  -DALPHADIABLO_EXPECTED_PYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
  "${OSX_CMAKE_ARGS[@]}"

BUILD_IDENTITY="$BUILD_DIR/alphadiablo-python-build.txt"
[ -f "$BUILD_IDENTITY" ] || { echo "CMake produced no Python build identity: $BUILD_IDENTITY"; exit 1; }
"$VENV_PY" - "$BUILD_IDENTITY" "$VENV_PY" "$PYTHON_EXT_SUFFIX" \
  "$PYTHON_INCLUDE_DIR" "$OSX_DEPLOYMENT_TARGET" "$DEPLOYMENT_SOURCE" <<'PY'
import pathlib
import sys

manifest = {}
for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    key, sep, value = line.partition("=")
    if not sep or key in manifest:
        raise SystemExit(f"Invalid CMake Python build identity line: {line!r}")
    manifest[key] = value
expected = {"executable": sys.argv[2], "ext_suffix": sys.argv[3]}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(
            f"CMake Python build identity drift: {key}={manifest.get(key)!r}, expected {value!r}")
includes = manifest.get("include_dirs", "").split(";")
if sys.argv[4] not in includes:
    raise SystemExit(
        f"CMake Python headers drift: {includes!r}, expected to include {sys.argv[4]!r}")
if sys.argv[5]:
    deployment = {
        "deployment_target": sys.argv[5], "deployment_source": sys.argv[6]}
    for key, value in deployment.items():
        if manifest.get(key) != value:
            raise SystemExit(
                f"CMake deployment identity drift: {key}={manifest.get(key)!r}, "
                f"expected {value!r}")
PY

# Note: build the targets separately, not `all` (the engine's test-resource target always fails on macOS)
cmake --build "$BUILD_DIR" -j "$JOBS" --target devilutionx   # produces the .app -> runtime assets
cmake --build "$BUILD_DIR" -j "$JOBS" --target _diablogym    # pybind11 bridge

BRIDGE="$BUILD_DIR/_diablogym${PYTHON_EXT_SUFFIX}"
[ -f "$BRIDGE" ] || {
  echo "The build produced no module for the current Python ABI: $BRIDGE"
  find "$BUILD_DIR" -maxdepth 1 -type f -name '_diablogym*.so' -print
  exit 1
}

if [ "$(uname -s)" = "Darwin" ]; then
  ENGINE_DYLIB="$BUILD_DIR/engine/liblibdevilutionx_so.dylib"
  GAME_BINARY="$BUILD_DIR/engine/devilutionx.app/Contents/MacOS/devilutionx"
  [ -f "$ENGINE_DYLIB" ] || { echo "Embedded engine dylib not found: $ENGINE_DYLIB"; exit 1; }
  [ -f "$GAME_BINARY" ] || { echo "Resource host binary not found: $GAME_BINARY"; exit 1; }
  "$VENV_PY" cmake/audit_macos_minos.py \
    --deployment-target "$OSX_DEPLOYMENT_TARGET" \
    --search-root "$BUILD_DIR" \
    "$BRIDGE" "$ENGINE_DYLIB" "$GAME_BINARY"
else
  ENGINE_SO="$BUILD_DIR/engine/liblibdevilutionx_so.so"
  [ -f "$ENGINE_SO" ] || { echo "Embedded engine .so not found: $ENGINE_SO"; exit 1; }
  # The evaluation contract (eval_contract/env.py) pins the asset path to devilutionx.app/Contents/Resources;
  # on Linux the engine assets land in build/engine/assets, so lay out the same structure with a real copy (no symlinks)
  RES="$BUILD_DIR/engine/devilutionx.app/Contents/Resources"
  [ -d "$BUILD_DIR/engine/assets" ] || { echo "Engine asset directory not found: $BUILD_DIR/engine/assets"; exit 1; }
  rm -rf "$RES" && mkdir -p "$RES"
  cp -a "$BUILD_DIR/engine/assets/." "$RES/"
  echo "Assets laid out: $RES ($(find "$RES" -type f | wc -l) files)"
fi

echo ""
echo "Build complete."
ls -lh "$BRIDGE"
echo "Smoke test:  $VENV_PY tests/smoke_random_agent.py"
