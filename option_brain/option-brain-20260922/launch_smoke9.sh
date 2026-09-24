#!/usr/bin/env bash
# WSL, RTX 5090. Round-9 smoke, fidelity and determinism checks (PREREG-ROUND9.md section 5), after make_exam9_freeze.sh
# and before the gate:
#   1. both arms through the exam's own wave script (run_exam9_wave.sh <W> <arm> SMOKE: burned held-out seeds 4150053
#      King and 4150045 Butcher, 3000 ticks, 30 minutes, --decisions 6000, EXTRA_HANDS=--allow-heldout, FACTS_VERSION
#      5 / 3 (round 9b; 4 / 3 before); the same run_mixed_v4.sh command lines as section 5), each followed by check_exam9_wave.py (menu r9, facts
#      version, allow_heldout, adapter, code shas = freeze list, hands string identical for both arms);
#   2. old-arm fidelity + determinism: fidelity9.py puts 500 prompts drawn uniformly from the four round-8 held-out
#      batches twice on live/fidelity9-old<suffix>/bus; the unchanged broker with sft/d9facts3 answers; every
#      first-copy argmax must equal the logged choice and both copies must agree;
#   3. new-arm determinism: 200 test rows of the new arm's relabel dir (exam9_config.json fidelity.new_rel: relabel-d11f5
#      since round 9b) twice on live/fidelity9-new<suffix>/bus with the new arm's adapter (sft/d11facts5);
#      copies agree;
#   4. all passed: <root>/exam9-smoke-pass.json {"smoke": "pass", "freeze_sha256": <sha of the freeze list>, ...}
#      (launch_gate9.sh and run_exam9_wave.sh gate accept it only while the freeze list is unchanged); otherwise
#      exam9-smoke-fail.json. Any failure is a code problem (section 5): fix, make_exam9_freeze.sh --refreeze, redo
#      this script as smoke-r2 (batches smoke9-<arm>-r2, fidelity9-<name>-r2).
# W = smoke (default) or smoke-r<k> (k 2-9): batch suffix '' or '-r<k>'. A real run refuses unless detached (setsid
# nohup, like the chain); it first moves an existing smoke pass / fail file aside (<file>.superseded-<time>). The
# fidelity steps are skipped when a smoke step failed. Uses the GPU only when the 5090 is idle (the wave script and
# the fidelity step refuse otherwise). Written 2026-09-23; in the freeze list.
# usage: launch_smoke9.sh [--dry-run] [smoke|smoke-r<k>]   (any other argument: exit 2)
# Launch like the chain (from Git Bash, background, wsl.exe attached):
#   wsl -e bash -c 'setsid -w nohup bash $AD_WORKSPACE/experiments/option-brain-20260922/launch_smoke9.sh >> $AD_ROOT/live/smoke9.log 2>&1'
set -u
EXP=$AD_WORKSPACE/experiments/option-brain-20260922
CFG=$EXP/exam9_config.json
DRY=0; W=smoke
for x in "$@"; do
  case "$x" in
    --dry-run) DRY=1 ;;
    smoke|smoke-r[2-9]) W=$x ;;
    *) echo "usage: launch_smoke9.sh [--dry-run] [smoke|smoke-r<k>] (got '$x')"; exit 2 ;;
  esac
done
SUF=${W#smoke}
eval "$(python3 -c "
import json, shlex
c = json.load(open('$CFG'))
f = c['fidelity']
print(f\"ROOT={shlex.quote(c['root'])}; PY_MODEL={shlex.quote(c['py_model'])}; FREEZE={shlex.quote(c['freeze_list'])}; PASS={shlex.quote(c['smoke_pass_file'])}\")
print(f\"OLD={shlex.quote(c['arms']['old']['adapter'])}; NEW={shlex.quote(c['arms']['new']['adapter'])}\")
print(f\"OLD_N={f['old_n']}; NEW_N={f['new_n']}; REP={f['repeat']}; OLD_BATCHES={shlex.quote(' '.join(f['old_batches']))}\")
print(f\"NEW_SRC={shlex.quote(f['new_rel'] + ' ' + ' '.join(f['new_files']))}; GPU_UUID={shlex.quote(c['gpu_uuid'])}; GPU_BUSY={c['gpu_busy_mib']}\")
")"
FAILF=${PASS%-pass.json}-fail.json
say() { echo "$( [ $DRY = 1 ] && echo 'DRY-RUN ')$(date '+%F %T %Z') SMOKE9 $*"; }
if [ $DRY = 0 ]; then
  IGN=$(awk '/^SigIgn/{print $2}' /proc/$$/status); TTY=$(ps -o tty= -p $$ 2>/dev/null | tr -d ' ')
  if ! { [ $(( 0x${IGN:-0} & 1 )) -eq 1 ] && [ "$TTY" = "?" ]; }; then
    say "refused: not detached (SIGHUP ignored: $(( 0x${IGN:-0} & 1 )), tty $TTY); launch with setsid nohup (header)"; exit 9
  fi
  for f in "$PASS" "$FAILF"; do
    [ -e "$f" ] && { mv "$f" "$f.superseded-$(date +%Y%m%d-%H%M%S)"; say "moved aside the old $f"; }
  done
fi
settle() {
  local b=$1 t0; t0=$(date +%s)
  while pgrep -f "^[^ ]*python[0-9.]* -B hf_broker\.py --adapter [^ ]+ --bus $ROOT/live/$b/bus\$" >/dev/null \
     || pgrep -f "^bash [^ ]*broker_watchdog\.sh $b " >/dev/null; do
    [ $(( $(date +%s) - t0 )) -ge 1200 ] && return 1; sleep 10
  done
}
FAIL=""
# 1. smoke, both arms
for arm in new old; do
  b=smoke9-$arm$SUF
  if [ $DRY = 1 ]; then bash "$EXP/run_exam9_wave.sh" --dry-run "$W" $arm SMOKE; say "smoke $arm ($b) dry run exit $?"; continue; fi
  say "smoke $arm start ($b)"
  bash "$EXP/run_exam9_wave.sh" "$W" $arm SMOKE > "$ROOT/live/$b.driver.log" 2>&1
  rc=$?; say "smoke $arm driver exit $rc"
  [ -e "$ROOT/live/$b/wave.json" ] || { tail -30 "$ROOT/live/$b.driver.log"; FAIL="$FAIL smoke-$arm-refused"; continue; }
  settle $b || FAIL="$FAIL smoke-$arm-broker-did-not-exit"
  python3 -B "$EXP/check_exam9_wave.py" $b --json "$ROOT/live/$b/check.json" > "$ROOT/live/$b/check.txt" 2>&1
  rc=$?; grep -v '^PASS' "$ROOT/live/$b/check.txt" | tail -20
  [ $rc -eq 0 ] || FAIL="$FAIL smoke-$arm-check-exit-$rc"
done
# 2-3. fidelity / determinism: <name> <adapter> <expected responses> <fidelity9.py make args...>
fidelity() {
  local name=$1 adapter=$2 want=$3; shift 3
  local D=$ROOT/live/fidelity9-$name$SUF FB=$ROOT/live/fidelity9-$name$SUF/bus
  if [ $DRY = 1 ]; then
    say "would run: python3 -B $EXP/fidelity9.py make --bus $FB $*"
    say "would run: cd $EXP && $PY_MODEL -B hf_broker.py --adapter $adapter --bus $FB  (until $want responses; then touch $FB/STOP)"
    say "would run: python3 -B $EXP/fidelity9.py compare --bus $FB --out $D/fidelity.json"
    return 0
  fi
  local BUSY MIB BAD
  BUSY=$(pgrep -af '^[^ ]*python[0-9.]* -B (live_runner|hf_broker|option_sft)\.py')
  MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  BAD=$(sha256sum -c --quiet "$FREEZE" 2>&1 | head -3)
  case "$MIB" in ''|*[!0-9]*) MIB=999999 ;; esac
  if [ -n "$BUSY" ] || [ "$MIB" -gt "$GPU_BUSY" ] || [ -n "$BAD" ] || [ -e "$D" ]; then
    say "fidelity-$name refused: busy [$BUSY] gpu $MIB MiB freeze [$BAD] dir exists [$( [ -e "$D" ] && echo yes)]"
    FAIL="$FAIL fidelity-$name-refused"; return
  fi
  python3 -B "$EXP/fidelity9.py" make --bus "$FB" "$@" || { FAIL="$FAIL fidelity-$name-make"; return; }
  ( cd "$EXP" && exec "$PY_MODEL" -B hf_broker.py --adapter "$adapter" --bus "$FB" ) > "$D/broker.log" 2>&1 &
  local BP=$! t0; t0=$(date +%s)
  say "fidelity-$name broker pid $BP"
  while [ "$(ls "$FB/responses" | grep -c '\.json$')" -lt "$want" ]; do
    kill -0 $BP 2>/dev/null || { say "fidelity-$name broker died"; tail -20 "$D/broker.log"; break; }
    [ $(( $(date +%s) - t0 )) -ge 3600 ] && { say "fidelity-$name not done in 60 min"; break; }
    sleep 10
  done
  touch "$FB/STOP"; wait $BP
  python3 -B "$EXP/fidelity9.py" compare --bus "$FB" --out "$D/fidelity.json" || FAIL="$FAIL fidelity-$name-mismatch"
}
if [ -n "$FAIL" ]; then
  say "smoke failed ($FAIL): fidelity steps skipped (the GPU is not used for them)"
else
  fidelity old "$OLD" $(( OLD_N * REP )) --n "$OLD_N" --repeat "$REP" --batches $OLD_BATCHES
  fidelity new "$NEW" $(( NEW_N * REP )) --n "$NEW_N" --repeat "$REP" --relabel $NEW_SRC
fi
if [ $DRY = 1 ]; then say "would write $PASS (bound to the freeze list sha) or $FAILF"; exit 0; fi
NOW=$(date +%s)
if [ -z "$FAIL" ]; then
  echo "{\"smoke\":\"pass\",\"wave\":\"$W\",\"time\":$NOW,\"freeze_sha256\":\"$(sha256sum "$FREEZE" | cut -d' ' -f1)\"}" > "$PASS"
  say "PASS -> $PASS"
else
  echo "{\"smoke\":\"fail\",\"wave\":\"$W\",\"time\":$NOW,\"freeze_sha256\":\"$(sha256sum "$FREEZE" 2>/dev/null | cut -d' ' -f1)\",\"why\":\"${FAIL# }\"}" > "$FAILF"
  say "FAIL:$FAIL -> $FAILF"
  exit 1
fi
