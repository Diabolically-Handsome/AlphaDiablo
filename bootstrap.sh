#!/usr/bin/env bash
# AlphaDiablo one-step development environment bootstrap script
# Purpose: rebuild the headless DevilutionX dev environment from scratch (or incrementally) and pass the smoke test.
# Idempotent and safe to rerun; the upstream clone lives in the system temp directory (rule #3: not in the project folder, no new repository).
#
# Usage:   ./bootstrap.sh
# Output:  $DEV_DIR/build/devilutionx.app        the game itself (native Apple Silicon)
#          $DEV_DIR/build/timedemo_test          headless deterministic replay test (environment health probe)
#
# Prerequisites (already met on this machine; on a fresh install the script adds them automatically):
#   - Homebrew; the game data MPQ is already in ~/Library/Application Support/diasurgical/devilution/
set -euo pipefail

if [ "$(uname -s)" = "Darwin" ]; then
  export PATH="/opt/homebrew/bin:$PATH"
  DATA_DIR="$HOME/Library/Application Support/diasurgical/devilution"
  JOBS="$(sysctl -n hw.physicalcpu)"
else
  DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/diasurgical/devilution"
  JOBS="$(nproc)"
fi
DEV_DIR="${TMPDIR:-/tmp}/alphadiablo-dev/devilutionX"
# Pinned upstream engine version — the build source for the leaderboard and all test baselines.
# Upgrading the engine is a deliberate decision: change this SHA, then rerun the gold-standard evaluation and rebuild the leaderboard.
ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"

echo "==> [1/5] Checking game data ($DATA_DIR)"
ls "$DATA_DIR/DIABDAT.MPQ" >/dev/null 2>&1 \
  || ls "$DATA_DIR/diabdat.mpq" >/dev/null 2>&1 \
  || ls "$DATA_DIR/spawn.mpq" >/dev/null 2>&1 \
  || { echo "Error: missing DIABDAT.MPQ/diabdat.mpq/spawn.mpq; prepare the data file first"; exit 1; }

echo "==> [2/5] Fetching upstream source @ ${ENGINE_REF:0:12} -> $DEV_DIR"
if [ ! -d "$DEV_DIR/.git" ]; then
  mkdir -p "$DEV_DIR"
  git -C "$DEV_DIR" init -q
  git -C "$DEV_DIR" remote add origin https://github.com/diasurgical/devilutionX.git
fi
if ! git -C "$DEV_DIR" rev-parse --quiet --verify "$ENGINE_REF^{commit}" >/dev/null; then
  git -C "$DEV_DIR" fetch --depth 1 origin "$ENGINE_REF"
fi
if [ "${BOOTSTRAP_CLEAN:-0}" = "1" ]; then
  # Upstream lives in a dedicated system temp clone; only the explicit recovery mode discards tracked/untracked source drift.
  # git clean runs without -x, so the large build caches listed in .gitignore are kept.
  git -C "$DEV_DIR" reset --hard -q "$ENGINE_REF"
  git -C "$DEV_DIR" clean -fd
elif [ "$(git -C "$DEV_DIR" rev-parse HEAD 2>/dev/null || true)" != "$ENGINE_REF" ]; then
  # When switching versions, discard working-tree changes to realign (build.sh reapplies the patches idempotently);
  # if already on the pinned version, leave the working tree alone so every run does not trigger a full rebuild
  git -C "$DEV_DIR" reset --hard -q "$ENGINE_REF"
fi

echo "==> [3/5] System dependencies"
if [ "$(uname -s)" = "Darwin" ]; then
  brew bundle install --file="$DEV_DIR/Brewfile" || echo "(occasional package lock conflicts can be ignored; the next build step serves as a fallback check)"
else
  # Linux (WSL2/Ubuntu): dependencies are preinstalled via apt (cmake g++ ninja libsdl2-dev libsodium-dev
  # libpng-dev libbz2-dev libfmt-dev gettext); if any are missing, the next build step will fail with an error
  for tool in cmake g++ msgfmt; do
    command -v "$tool" >/dev/null || { echo "Missing $tool; install the build dependencies with apt first"; exit 1; }
  done
fi

# CI mode: only clone + dependencies; the engine build is left entirely to build.sh (so the same engine is not compiled twice)
if [ "${BOOTSTRAP_CLONE_ONLY:-0}" = "1" ]; then
  echo "✅ clone-only mode: source ready @ ${ENGINE_REF:0:12}; skipping engine build and smoke test"
  exit 0
fi

echo "==> [4/5] Building devilutionx + timedemo_test (note: do not use make all; the test asset targets always fail on macOS)"
cmake -S "$DEV_DIR" -B "$DEV_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$DEV_DIR/build" -j "$JOBS" --target devilutionx
cmake --build "$DEV_DIR/build" -j "$JOBS" --target timedemo_test

echo "==> [5/5] Smoke test: headless deterministic replay"
"$DEV_DIR/build/timedemo_test"

echo ""
echo "✅ Environment ready"
echo "   Engine: $DEV_DIR/build/devilutionx.app"
echo "   Play:   open '$DEV_DIR/build/devilutionx.app'   (GUI, full game data)"
echo "   Python: source \"$(dirname "$0")/.venv/bin/activate\"   (torch/gymnasium/SB3)"
