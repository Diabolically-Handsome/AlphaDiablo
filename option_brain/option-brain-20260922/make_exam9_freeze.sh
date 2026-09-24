#!/usr/bin/env bash
# WSL, CPU only. Writes the round-9 exam freeze list (PREREG-ROUND9.md decision 10, section 7; engineering review step
# 2): sha256 of every file that runs in an exam game or its broker, or decides the score. run_exam9_wave.sh refuses a
# wave unless 'sha256sum -c' of this list is clean. Written 2026-09-23; itself in the list.
# usage: make_exam9_freeze.sh [--out PATH] [--allow-missing] [--with-base-shards] [--refreeze]
#   --out PATH          write elsewhere (dry run; default exam9_config.json freeze_list). The side files (below) go
#                       next to PATH.
#   --allow-missing     dry run only (needs --out, not the official path): missing files and failed pre-checks are
#                       listed in <list>.notes instead of refusing
#   --with-base-shards  also hash the 10 base-model safetensors shards (45 GB, about 3 minutes) into a separate list
#                       exam9-freeze-base.sha256 (not in the per-wave 'sha256sum -c'; checked again after the exam);
#                       that small list is itself in the main list. Required for the official freeze.
#   --refreeze          the official list exists: keep it and its side files as <file>.superseded-<time> (after a
#                       code fix found by the smoke or the gate, PREREG sections 5-6); a first freeze refuses then.
# Pre-checks (all before anything is written; any failure refuses):
#   a. run_mixed_v4.sh's own facts-version check (its AFV code) gives each arm's facts_version of exam9_config.json:
#      5 for sft/d11facts5 (round 9b; 4 for sft/d10facts4 before the retrain) and 3 for sft/d9facts3
#      (step 0 done: d9facts3/registration.json has facts_version 3);
#   b. no python live_runner / hf_broker / option_sft process (training or evaluation may still write registration.json
#      or weights);
#   c. no <root>/live/exam9-* batch; at a first freeze also no gate9-* / smoke9-* / fidelity9-* batch;
#   d. the original Butcher probe and its results (exam9_config.json butcher_probe: probe_sha256, run1_sha256, 29 run2
#      files) are in <root>/seed-audit/ or can be copied there from butcher_probe.source (an official freeze copies
#      them; a dry run lists the source files instead);
#   e. --allow-missing only with --out other than the official path.
# The list:
#   1. option-brain code of the review's list + the exam tooling: options, render, live_runner, facts, engine_numbers,
#      undo_guard, game_executor_r3, hf_broker, option_sft, rollin_broker, quest_lottery, menu_r9, score_exam9,
#      check_exam9_wave, exam9_lib, fidelity9, seed_audit9 (.py); run_mixed_v4.sh, run_exam9_wave.sh,
#      run_exam9_chain.sh, broker_watchdog.sh, make_exam9_freeze.sh, launch_gate9.sh, launch_smoke9.sh,
#      exam9_config.json, bench/butcher_probe.py. NOT the live PREREG-ROUND9.md: its frozen copy
#      exam9-prereg-frozen.md (side file) is listed instead, so that revisions can be appended to the live file
#      (run_exam9_wave.sh and check_exam9_wave.py check live = copy + appended text).
#   2. the review's globs: strategy-brain-sft-20260922/*.py, strategist-rl-king-continuation-20260921/revision3/*.py.
#   3. the game process's import closure, traced by importing live_runner's modules in the game venv with -B and no
#      GPU (options, render, facts, undo_guard, engine_numbers, game_executor_r3, game_support_r3, and model158's
#      lazily imported leashed_ppo): session.py (r16), strategist-rl-hands runtime/model/goal/public_view,
#      config/navigation, revision1/2/3 runtimes, common/protocol/purchase_audit, ...
#   4. native and data: the build json telemetry_runtime.load_bridge reads (config.ROOT/build-r3.json), the bridge
#      and engine .so it names, model158 (config.MODEL) and the goal head (config.HEAD), every file under session.ASSETS
#      and session.DATA (DIABDAT.MPQ etc.).
#   5. brains: both adapters' adapter_model.safetensors, adapter_config.json and registration.json; the base model's
#      config/tokenizer/index files (the shards: exam9-freeze-base.sha256, --with-base-shards).
#   6. side files, written next to the list and hashed with the rest: exam9-prereg-frozen.md (copy of
#      PREREG-ROUND9.md), exam9-env.txt (python/torch/transformers/peft/bitsandbytes versions of both venvs, nvidia
#      driver, kernel), exam9-seeds.json (the two lists, halves, strata and replacement order, from exam9_config.json),
#      exam9-seed-audit.json (seed_audit9.py --include-exam-batches; any exam-seed hit refuses the freeze),
#      exam9-freeze-base.sha256 (with --with-base-shards).
#   7. every file under <root>/seed-audit/ (the Butcher probe and its results, pre-check d).
# Everything is staged first; only a complete list replaces anything. At the end it runs 'sha256sum -c' on the new
# list and prints the sha256 of the frozen prereg copy for REPORT.md (it does not edit REPORT.md).
set -u
EXP=$AD_WORKSPACE/experiments/option-brain-20260922
EXPS=${EXP%/*}
CFG=$EXP/exam9_config.json
eval "$(python3 -c "
import json, shlex
c = json.load(open('$CFG')); bp = c['butcher_probe']
print(f\"OFFICIAL={shlex.quote(c['freeze_list'])}; ROOT={shlex.quote(c['root'])}; PY_GAME={shlex.quote(c['py_game'])}; PY_MODEL={shlex.quote(c['py_model'])}\")
print(f\"BP_SRC={shlex.quote(bp['source'])}; BP_SHA={bp['probe_sha256']}; BP_RUN1={bp['run1_sha256']}; BP_N2={bp['run2_files']}\")
print('ADAPTERS=(' + ' '.join(shlex.quote(a['adapter']) for a in c['arms'].values()) + ')')
print('ARM_FV=(' + ' '.join(f\"{shlex.quote(a['adapter'])}={a['facts_version']}\" for a in c['arms'].values()) + ')')
print(f\"NEW_FV={int(c['arms']['new']['facts_version'])}\")
")"
OUT=$OFFICIAL; ALLOW=0; SHARDS=0; REFREEZE=0; OUT_GIVEN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || { echo "--out needs a path"; exit 2; }; OUT=$2; OUT_GIVEN=1; shift 2 ;;
    --allow-missing) ALLOW=1; shift ;;
    --with-base-shards) SHARDS=1; shift ;;
    --refreeze) REFREEZE=1; shift ;;
    *) sed -n '/^# usage:/,/^#   --refreeze/p' "$0"; exit 2 ;;
  esac
done
OUT=$(realpath -m "$OUT")
IS_OFFICIAL=0; [ "$OUT" = "$(realpath -m "$OFFICIAL")" ] && IS_OFFICIAL=1
OD=$(dirname "$OUT")
PRE_FAIL=""
pre() { echo "pre-check $1: $2"; }
pfail() { echo "pre-check $1: FAILED $2"; PRE_FAIL="$PRE_FAIL
# PRECHECK-FAILED $1: $2"; }
# e. --allow-missing is a dry-run flag
if [ $ALLOW = 1 ] && { [ $OUT_GIVEN = 0 ] || [ $IS_OFFICIAL = 1 ]; }; then
  echo "refused: --allow-missing needs --out with a path other than the official $OFFICIAL"; exit 2
fi
if [ $IS_OFFICIAL = 1 ] && [ $SHARDS = 0 ]; then
  echo "refused: the official freeze hashes the base-model shards: add --with-base-shards"; exit 2
fi
if [ -e "$OUT" ] && [ $REFREEZE = 0 ]; then echo "refused: $OUT exists (use --refreeze after a code fix, PREREG sections 5-6)"; exit 2; fi
if [ ! -e "$OUT" ] && [ $REFREEZE = 1 ]; then echo "refused: --refreeze but there is no $OUT"; exit 2; fi
# a. facts versions by run_mixed_v4.sh's own check
AFV_PY=$(sed -n "/^read -r -d '' AFV_PY <<'PY'\$/,/^PY\$/p" "$EXP/run_mixed_v4.sh" | sed '1d;$d')
for af in "${ARM_FV[@]}"; do
  a=${af%=*}; want=${af##*=}
  got=$(python3 -c "$AFV_PY" "$a" 2>&1)
  [ "${got%% *}" = "$want" ] && pre a "$a -> $got" || pfail a "$a needs facts version $want, run_mixed_v4.sh check says: $got"
done
# b. nothing running that could still write an adapter or a registration
BUSY=$(pgrep -af '^[^ ]*python[0-9.]* -B (live_runner|hf_broker|option_sft)\.py' | cut -c1-160)
[ -z "$BUSY" ] && pre b "no live_runner/hf_broker/option_sft python" || pfail b "running: $(echo "$BUSY" | head -3 | tr '\n' ';')"
# c. no exam batch; at a first freeze no gate/smoke/fidelity batch either
PAT='exam9-*'; [ $REFREEZE = 0 ] && PAT='exam9-* gate9-* smoke9-* fidelity9-*'
EXIST=$(for p in $PAT; do ls -d "$ROOT"/live/$p 2>/dev/null; done | tr '\n' ' ')
[ -z "$EXIST" ] && pre c "no $PAT batch in $ROOT/live" || pfail c "existing: $EXIST"
# d. the original Butcher probe and its results
SA=$ROOT/seed-audit
probe_ok() {  # <dir>: butcher_probe.py, run1.json with the recorded shas, BP_N2 files in run2/
  [ "$(sha256sum "$1/butcher_probe.py" 2>/dev/null | cut -d' ' -f1)" = "$BP_SHA" ] \
    && [ "$(sha256sum "$1/run1.json" 2>/dev/null | cut -d' ' -f1)" = "$BP_RUN1" ] \
    && [ "$(ls "$1"/run2/*.json 2>/dev/null | wc -l)" -eq "$BP_N2" ]
}
PROBE_DIR=""
if probe_ok "$SA"; then PROBE_DIR=$SA; pre d "$SA holds the original probe and results"
elif probe_ok "$BP_SRC"; then
  if [ $IS_OFFICIAL = 1 ]; then PROBE_DIR=COPY; pre d "will copy the original probe and results from $BP_SRC to $SA"
  else PROBE_DIR=$BP_SRC; pre d "dry run: listing the source $BP_SRC (an official freeze copies it to $SA)"; fi
else pfail d "neither $SA nor $BP_SRC holds butcher_probe.py ($BP_SHA), run1.json ($BP_RUN1) and $BP_N2 run2/*.json"; fi
if [ -n "$PRE_FAIL" ] && [ $ALLOW = 0 ]; then echo "refused: pre-check(s) failed (nothing written)"; exit 3; fi
# ---- stage the side files
STAGE=$(mktemp -d "$OD/.exam9-freeze-stage.XXXXXX") || { echo "refused: cannot stage in $OD"; exit 3; }
trap 'rm -rf "$STAGE"' EXIT
SIDE="exam9-prereg-frozen.md exam9-env.txt exam9-seeds.json exam9-seed-audit.json"
[ $SHARDS = 1 ] && SIDE="$SIDE exam9-freeze-base.sha256"
cp -p "$EXP/PREREG-ROUND9.md" "$STAGE/exam9-prereg-frozen.md"
python3 -c "
import json
c = json.load(open('$CFG'))
keys = ('king_seeds', 'butcher_seeds', 'halves', 'waves', 'king_seeds_with_butcher_quest', 'butcher_seeds_with_king_quest', 'replacements', 'gate', 'smoke', 'butcher_probe')
json.dump(dict({k: c[k] for k in keys}, source='exam9_config.json', quest_check='Butcher 16/16 by the tick-0 wounded-townsman probe (seed-audit/butcher_probe.py, run1.json; declared exception); King 16/16 by bridge manual_seed_has_king = quest_lottery.py'), open('$STAGE/exam9-seeds.json', 'w'), indent=1)
" || { echo "refused: could not write exam9-seeds.json"; exit 3; }
( cd "$EXP" && python3 -B seed_audit9.py --include-exam-batches --out "$STAGE/exam9-seed-audit.json" ) \
  || { echo "refused: the seed audit found exam seeds in use (see $STAGE/exam9-seed-audit.json)"; cp "$STAGE/exam9-seed-audit.json" "$OD/exam9-seed-audit.refused.json" 2>/dev/null; exit 3; }
{
  echo "# exam9 environment, $(date '+%F %T %Z'), $(uname -r)"
  for P in "$PY_GAME" "$PY_MODEL"; do
    echo "== $P"; "$P" -c "import sys; print('python', sys.version.split()[0])"
    "$P" -m pip freeze 2>/dev/null | grep -iE '^(torch|transformers|peft|bitsandbytes|accelerate|numpy|safetensors|tokenizers|mistral.common)[=@ ]'
  done
  echo "== nvidia"; nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader 2>&1
  echo "== base model"; ls -la $AD_HOME/strategy_brain_sft_20260922/base-mistral24/weights | sed 's/^/  /'
} > "$STAGE/exam9-env.txt"
BASE=$AD_HOME/strategy_brain_sft_20260922/base-mistral24/weights
if [ $SHARDS = 1 ]; then
  echo "hashing the base-model shards (about 3 minutes)"
  sha256sum "$BASE"/*.safetensors > "$STAGE/exam9-freeze-base.sha256" || { echo "refused: base shard hashing failed"; exit 3; }
fi
# ---- the file list (final paths)
LIST=$STAGE/list
{
  for f in options render live_runner facts engine_numbers undo_guard game_executor_r3 hf_broker option_sft rollin_broker \
           quest_lottery menu_r9 score_exam9 check_exam9_wave exam9_lib fidelity9 seed_audit9; do echo "$EXP/$f.py"; done
  for f in run_mixed_v4.sh run_exam9_wave.sh run_exam9_chain.sh broker_watchdog.sh make_exam9_freeze.sh launch_gate9.sh \
           launch_smoke9.sh exam9_config.json bench/butcher_probe.py; do echo "$EXP/$f"; done
  ls "$EXPS"/strategy-brain-sft-20260922/*.py "$EXPS"/strategist-rl-king-continuation-20260921/revision3/*.py
  # 3 + 4: import closure and the native/data files the runtime names (CPU, -B: no bytecode written, cwd a temp dir)
  TMPD=$(mktemp -d)
  ( cd "$TMPD" && FACTS_VERSION=$NEW_FV CUDA_VISIBLE_DEVICES= "$PY_GAME" -B -c "
import os, sys, json
E = '$EXPS'
sys.path.insert(0, E + '/option-brain-20260922'); sys.path.insert(0, E + '/strategy-brain-sft-20260922')
import options, render, facts, undo_guard, engine_numbers, game_executor_r3, game_support_r3
import config, session, telemetry_runtime
sys.path.insert(0, str(config.NETWORK_SOURCE / 'train'))
try:
    import leashed_ppo  # model.load_actor imports it when a game starts
except Exception as exc:
    print('# leashed_ppo import failed: %r' % exc)
roots = ('$AD_WORKSPACE', '$AD_HOME')
for m in list(sys.modules.values()):
    f = getattr(m, '__file__', None)
    if f and os.path.isabs(f) and os.path.exists(f):
        f = os.path.realpath(f)
        if f.startswith(roots) and '/site-packages/' not in f and '/.venv/' not in f and '/venv/' not in f:
            print(f)
build = telemetry_runtime.ROOT / 'build-r3.json'
print(build)
b = json.load(open(build))
print(b['bridge']); print(b['engine'])
print(config.MODEL); print(config.HEAD)
for d in (session.ASSETS, session.DATA):
    for dp, dn, fn in os.walk(d):
        for x in fn:
            print(os.path.join(dp, x))
" ) || echo "# IMPORT-TRACE-FAILED"
  rmdir "$TMPD" 2>/dev/null
  for a in "${ADAPTERS[@]}"; do echo "$a/adapter_model.safetensors"; echo "$a/adapter_config.json"; echo "${a%/adapter}/registration.json"; done
  ls "$BASE"/*.json
  [ -d "$SA" ] && [ "$PROBE_DIR" != "$BP_SRC" ] && find "$SA" -type f
  [ "$PROBE_DIR" = "$BP_SRC" ] && find "$BP_SRC" -type f
} | grep -v '^$' | sort -u > "$LIST"
grep -q '^# IMPORT-TRACE-FAILED' "$LIST" && { echo "refused: the import trace failed"; grep '^#' "$LIST"; exit 3; }
MISSING=$(grep -v '^#' "$LIST" | while IFS= read -r f; do [ -f "$f" ] || echo "$f"; done)
if [ -n "$MISSING" ] && [ $ALLOW = 0 ]; then
  echo "refused: missing files:"; echo "$MISSING" | sed 's/^/  /'; exit 3
fi
# ---- d (official): copy the original probe and results into seed-audit (idempotent; checked again)
if [ "$PROBE_DIR" = COPY ]; then
  mkdir -p "$SA/run2" && cp -p "$BP_SRC/butcher_probe.py" "$BP_SRC/run1.json" "$SA/" && cp -p "$BP_SRC"/run2/*.json "$SA/run2/" \
    && probe_ok "$SA" || { echo "refused: copying the Butcher probe into $SA failed"; exit 3; }
  echo "copied the original Butcher probe and results into $SA"
  find "$SA" -type f >> "$LIST"; sort -u -o "$LIST" "$LIST"
fi
# ---- write: only sha lines in the list ('sha256sum -c' warns on anything else); notes beside it
NEW=$STAGE/new.sha256
grep -v '^#' "$LIST" | while IFS= read -r f; do [ -f "$f" ] && sha256sum "$f"; done > "$NEW"
for s in $SIDE; do printf '%s  %s\n' "$(sha256sum < "$STAGE/$s" | cut -d' ' -f1)" "$OD/$s" >> "$NEW"; done
NOTES=$STAGE/new.notes
{ echo "$MISSING" | grep -v '^$' | sed 's/^/# MISSING /'; grep '^#' "$LIST"; [ -n "$PRE_FAIL" ] && echo "$PRE_FAIL" | grep -v '^$'
  [ "$PROBE_DIR" = "$BP_SRC" ] && echo "# NOTE dry run: the Butcher probe is listed at its source $BP_SRC; an official freeze copies it to $SA"
  [ $SHARDS = 0 ] && echo "# NOTE no --with-base-shards: exam9-freeze-base.sha256 not written (required for the official freeze)"; } > "$NOTES"
if [ -e "$OUT" ]; then  # --refreeze: keep the old list and side files
  T=$(date +%Y%m%d-%H%M%S)
  for f in "$OUT" "$OUT.notes" $(for s in $SIDE exam9-freeze-base.sha256; do echo "$OD/$s"; done | sort -u); do
    [ -e "$f" ] && mv "$f" "$f.superseded-$T"
  done
  echo "re-freeze: the previous list and side files are kept as *.superseded-$T"
fi
for s in $SIDE; do mv "$STAGE/$s" "$OD/$s"; done
mv "$NOTES" "$OUT.notes"; mv "$NEW" "$OUT"
N=$(wc -l < "$OUT")
echo "wrote $OUT: $N files; notes $OUT.notes ($(wc -l < "$OUT.notes") lines)$( [ -n "$MISSING" ] && echo "; $(echo "$MISSING" | grep -c .) MISSING (allowed: dry run)")$( [ -n "$PRE_FAIL" ] && echo "; PRE-CHECKS FAILED (allowed: dry run)")"
BAD=$(sha256sum -c --quiet "$OUT" 2>&1 | head -5)
[ -z "$BAD" ] && echo "sha256sum -c $OUT: clean" || echo "sha256sum -c $OUT: FAILED $BAD"
[ $SHARDS = 1 ] && echo "base shards: $OD/exam9-freeze-base.sha256 ($(wc -l < "$OD/exam9-freeze-base.sha256") files; check again after the exam with sha256sum -c)"
echo "freeze list sha256: $(sha256sum "$OUT" | cut -d' ' -f1)"
echo "for REPORT.md: PREREG-ROUND9.md frozen as $OD/exam9-prereg-frozen.md, sha256 $(sha256sum "$OD/exam9-prereg-frozen.md" | cut -d' ' -f1)"
[ -z "$BAD" ]
