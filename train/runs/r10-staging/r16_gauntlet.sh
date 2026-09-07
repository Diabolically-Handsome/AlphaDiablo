#!/usr/bin/env bash
# R16 修宪合龙后的终极执法:全树静止 → 编译 → 全套件 → 128 种子旧法锚重烤。
# 任一环节失败即停(fail-closed),日志 train/runs/r10-staging/r16-gauntlet.log。
set -u
cd ~/AlphaDiablo/diablogym || exit 2
PY=.venv/bin/python
LOG=train/runs/r10-staging/r16-gauntlet.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "R16 gauntlet start"
say "protocol source sha256:"
sha256sum python/diablogym/env.py python/diablogym/worker_env.py python/diablogym/options_env.py \
  train/train_ppo.py train/leashed_ppo.py train/eval_assembled.py train/eval_contract.py \
  train/runs/r10-staging/probe_r15_deployment.py | tee -a "$LOG"

say "step 1/3 py_compile"
if ! $PY -m py_compile python/diablogym/env.py python/diablogym/worker_env.py python/diablogym/options_env.py \
     train/train_ppo.py train/leashed_ppo.py train/eval_assembled.py train/eval_contract.py \
     train/runs/r10-staging/probe_r15_deployment.py 2>>"$LOG"; then
  say "PY_COMPILE_FAIL"; exit 1
fi
say "PY_COMPILE_OK"

say "step 2/3 full pytest suite"
$PY -m pytest tests -q -p no:cacheprovider -x 2>&1 | tail -25 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" != "0" ]; then say "SUITE_FAIL"; exit 1; fi
say "SUITE_OK"

say "step 3/3 old-law anchor rebake (128 seeds x 4, bit-identity)"
$PY train/runs/r10-staging/run_r13_rebake.py 2>&1 | tail -15 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" != "0" ]; then say "REBAKE_FAIL"; exit 1; fi
say "REBAKE_OK"
say "R16 gauntlet PASS"
