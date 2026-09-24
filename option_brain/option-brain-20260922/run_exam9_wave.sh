#!/usr/bin/env bash
# WSL. One wave of the round-9 paired final exam (PREREG-ROUND9.md; engineering review step 1), or the pre-exam gate,
# or a smoke run, all through run_mixed_v4.sh with the same flags. Written 2026-09-23 (exam9 prep); in the freeze list.
#
# usage: run_exam9_wave.sh [--dry-run] <wave> <new|old> <set> [--king "s1 s2 .."] [--butcher "s1 .."]
#   wave 1..4 with set A|B      exam waves; (wave, arm, set) must be one of exam9_config.json 'waves' (ABBA:
#                               1 new A, 2 old A, 3 old B, 4 new B); broker batch exam9-w<wave>-<arm>, out-batches
#                               exam9-w<wave>-<arm>-king / -butcher; 8 King (150000 ticks) + 8 Butcher (80000 ticks).
#   wave <1-4>r<k> with A|B     whole-wave re-run of a voided wave (PREREG 3.3), e.g. 1r2 -> batch exam9-w1r2-<arm>.
#   wave m1 with set MK         the make-up wave (class-A re-runs, PREREG 4.5): --king/--butcher list the seeds, each
#                               must be an exam seed of that mission; batch exam9-m1-<arm>; m1r<k> re-runs a voided
#                               make-up wave whole (exam9-m1r<k>-<arm>).
#   wave gate[-r<k>] with GATE  the training-seed gate (decision 7), new arm only: 11 King + 8 Butcher; batch gate9-new.
#   wave smoke[-r<k>] with SMOKE the smoke (decision 8): 4150053 King + 4150045 Butcher at 3000 ticks; smoke9-<arm>.
#   --dry-run                   run every refusal check, print the wave descriptor and the exact command lines, start
#                               nothing, write nothing; exit 0 if the wave would start, 9 if it would be refused.
# Every game: <seed>:<mission>:<ticks>:720:<out-batch> (smoke: 3000 ticks, 30 minutes, PREREG section 5), --decisions
# 6000 (run_mixed_v4.sh), FACTS_VERSION from the fixed arm table (new = sft/d11facts5 -> 5 since round 9b, before it sft/d10facts4 -> 4;
# old = sft/d9facts3 -> 3),
# EXTRA_HANDS=--allow-heldout (the r9 menu is live_runner's default; check_exam9_wave.py asserts started.json menu r9).
# Refusal checks (all run; any failure refuses): STOP file; a python live_runner/hf_broker/option_sft already running
# (pattern '^[^ ]*python[0-9.]* -B (...)\.py' cannot match this shell or pgrep: their command lines do not start
# with a python path); the 5090 already holding > gpu_busy_mib; the freeze list present, covering the required files
# (the frozen prereg copy, not the live PREREG-ROUND9.md), and 'sha256sum -c' clean; PREREG-ROUND9.md = the frozen
# copy + appended text only; the adapter's weights and run_mixed_v4.sh's own facts-version check agreeing with the
# arm table; batch and out-batch directories new; exam and make-up waves: the gate pass file with gate 'pass' or an
# owner_approval, bound to the current freeze list (its freeze_sha256); gate waves: the smoke pass file bound to the
# current freeze list; free disk; detached from any terminal (SIGHUP ignored, no controlling tty; launch with setsid
# nohup, see run_exam9_chain.sh).
# Then: live/<batch>/prereg.sha256 (PREREG, frozen copy, freeze list, config) and live/<batch>/wave.json, a broker
# watchdog (broker_watchdog.sh, detached) for the batch's bus, and exec run_mixed_v4.sh (same PID, so the watchdog can
# follow it).
set -u
EXP=$AD_WORKSPACE/experiments/option-brain-20260922
CFG=$EXP/exam9_config.json
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
[ $# -ge 3 ] || { sed -n '5,19p' "$0"; exit 2; }
W=$1; ARM=$2; SET=$3; shift 3
KING=""; BUTCHER=""; HAVE_KING=0; HAVE_BUTCHER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --king) KING=$2; HAVE_KING=1; shift 2 ;;
    --butcher) BUTCHER=$2; HAVE_BUTCHER=1; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
read -r -d '' PLAN_PY <<'PY'
import json, re, shlex, sys, time, hashlib
cfg = json.load(open(sys.argv[1]))
w, arm, st, king, butcher, have_k, have_b = sys.argv[2:9]
err = []
if arm not in cfg['arms']:
    err.append(f'arm {arm!r} is not new/old')
seeds = {}
ticks = dict(cfg['ticks'])
minutes = cfg['minutes']
if re.fullmatch(r'[1-4](r[2-9])?', w):
    if {'wave': w[0], 'arm': arm, 'half': st} not in cfg['waves']:
        err.append(f'wave {w} {arm} {st} is not in the ABBA plan {cfg["waves"]}')
    batch = f'exam9-w{w}-{arm}'
    seeds = {m: cfg['halves'].get(st, {}).get(m, []) for m in ('skeleton_king', 'butcher')}
    kind = 'exam'
elif re.fullmatch(r'm1(r[2-9])?', w):
    if st != 'MK':
        err.append('a make-up wave needs set MK')
    batch = f'exam9-{w}-{arm}'
    seeds = {'skeleton_king': [int(s) for s in king.split()], 'butcher': [int(s) for s in butcher.split()]}
    for m, allowed in (('skeleton_king', cfg['king_seeds']), ('butcher', cfg['butcher_seeds'])):
        bad = [s for s in seeds[m] if s not in allowed]
        if bad:
            err.append(f'make-up {m} seeds {bad} are not exam seeds of that mission')
    if not seeds['skeleton_king'] and not seeds['butcher']:
        err.append('a make-up wave needs --king and/or --butcher seeds')
    kind = 'makeup'
elif re.fullmatch(r'gate(-r[2-9])?', w):
    if st != 'GATE' or arm != cfg['gate']['arm']:
        err.append(f'the gate is set GATE with arm {cfg["gate"]["arm"]} only')
    batch = f'gate9-{arm}' + w[4:]
    seeds = {m: cfg['gate'][m] for m in ('skeleton_king', 'butcher')}
    kind = 'gate'
elif re.fullmatch(r'smoke(-r[2-9])?', w):
    if st != 'SMOKE':
        err.append('a smoke run is set SMOKE')
    batch = f'smoke9-{arm}' + w[5:]
    seeds = {m: cfg['smoke'][m] for m in ('skeleton_king', 'butcher')}
    ticks = {m: cfg['smoke_ticks'] for m in ticks}
    minutes = cfg['smoke_minutes']
    kind = 'smoke'
else:
    err.append(f'wave {w!r} is not 1-4, <1-4>r<k>, m1[r<k>], gate[-r<k>] or smoke[-r<k>]')
    batch, kind = 'invalid', 'invalid'
if kind in ('exam', 'gate', 'smoke') and (have_k == '1' or have_b == '1'):
    err.append('--king/--butcher are for make-up waves only')
a = cfg['arms'].get(arm, {})
outb = {'skeleton_king': f'{batch}-king', 'butcher': f'{batch}-butcher'}
games = [f'{s}:{m}:{ticks[m]}:{minutes}:{outb[m]}' for m in ('skeleton_king', 'butcher') for s in seeds.get(m, [])]
wave = dict(wave=w, set=st, kind=kind, arm=arm, adapter=a.get('adapter'), facts_version=a.get('facts_version'), broker_batch=batch,
            out_batches={m: outb[m] for m in outb if seeds.get(m)}, seeds=seeds, ticks=ticks, minutes=minutes,
            decision_limit=cfg['decision_limit'], hands=cfg['hands'], extra_hands=cfg['extra_hands'], freeze_list=cfg['freeze_list'],
            config_sha256=hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())
q = shlex.quote
print(f'BATCH={q(batch)}; KIND={q(kind)}; ADAPTER={q(a.get("adapter") or "")}; FV={q(str(a.get("facts_version") or ""))}')
print(f'EXTRA={q(cfg["extra_hands"])}; FREEZE={q(cfg["freeze_list"])}; ROOT={q(cfg["root"])}; GATEPASS={q(cfg["gate_pass_file"])}; SMOKEPASS={q(cfg["smoke_pass_file"])}')
print(f'PY_MODEL={q(cfg["py_model"])}; GPU_UUID={q(cfg["gpu_uuid"])}; GPU_BUSY={q(str(cfg["gpu_busy_mib"]))}; MIN_FREE={q(str(cfg["min_free_gb"]))}')
print(f'PREREG={q(cfg["exp"] + "/" + cfg["prereg"])}; PREREG_FROZEN={q(cfg["prereg_frozen"])}; BRIDGE={q(cfg["bridge_path"])}; ENGINE={q(cfg["engine_path"])}')
print('OUTBATCHES=(' + ' '.join(q(outb[m]) for m in outb if seeds.get(m)) + ')')
print('GAMES=(' + ' '.join(q(g) for g in games) + ')')
print(f'WAVE_JSON={q(json.dumps(wave))}')
print('PLAN_ERRORS=(' + ' '.join(q(e) for e in err) + ')')
PY
PLAN=$(python3 -c "$PLAN_PY" "$CFG" "$W" "$ARM" "$SET" "$KING" "$BUTCHER" "$HAVE_KING" "$HAVE_BUTCHER") || { echo "refused: could not read $CFG"; exit 9; }
eval "$PLAN"
TAG=""; [ $DRY = 1 ] && TAG="DRY-RUN "
# a dry run may test another freeze list (e.g. make_exam9_freeze.sh --out <scratch> --allow-missing); a real run never
if [ $DRY = 1 ] && [ -n "${EXAM9_FREEZE:-}" ]; then
  FREEZE=$EXAM9_FREEZE; PREREG_FROZEN=$(dirname "$FREEZE")/exam9-prereg-frozen.md
  echo "${TAG}using freeze list $FREEZE and frozen prereg $PREREG_FROZEN (EXAM9_FREEZE)"
fi
FAILS=0
ok() { echo "${TAG}check $1: ok${2:+ ($2)}"; }
no() { echo "${TAG}check $1: REFUSED $2"; FAILS=$((FAILS+1)); }
echo "${TAG}wave $W arm $ARM set $SET -> batch $BATCH ($KIND), ${#GAMES[@]} games, $(date '+%F %T %Z')"
for e in "${PLAN_ERRORS[@]}"; do no plan "$e"; done
# 1. STOP
[ -e "$ROOT/STOP" ] && no stop-file "$ROOT/STOP exists" || ok stop-file
# 2. nothing of ours running (games, brokers, training)
BUSY=$(pgrep -af '^[^ ]*python[0-9.]* -B (live_runner|hf_broker|option_sft)\.py' | cut -c1-160)
[ -n "$BUSY" ] && no processes "running: $(echo "$BUSY" | head -3 | tr '\n' ';')" || ok processes "no live_runner/hf_broker/option_sft python"
# 3. the 5090 idle
if command -v nvidia-smi >/dev/null 2>&1; then
  MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  case "$MIB" in
    ''|*[!0-9]*) no gpu "could not read memory.used of $GPU_UUID ('$MIB')" ;;
    *) [ "$MIB" -gt "$GPU_BUSY" ] && no gpu "5090 holds $MIB MiB (> $GPU_BUSY)" || ok gpu "5090 holds $MIB MiB" ;;
  esac
else
  no gpu "nvidia-smi not found"
fi
# 4. freeze list present, covering the required files, unchanged
if [ -f "$FREEZE" ]; then
  MISSING=""
  for f in "$EXP/live_runner.py" "$EXP/run_mixed_v4.sh" "$EXP/run_exam9_wave.sh" "$EXP/run_exam9_chain.sh" "$EXP/exam9_config.json" \
           "$EXP/exam9_lib.py" "$EXP/check_exam9_wave.py" "$EXP/score_exam9.py" "$EXP/hf_broker.py" "$EXP/facts.py" "$EXP/options.py" \
           "$EXP/broker_watchdog.sh" "$PREREG_FROZEN" "$BRIDGE" "$ENGINE" "${ADAPTER:-none}/adapter_model.safetensors"; do
    sed -E 's/^[0-9a-f]+ [ *]//' "$FREEZE" | grep -qxF "$f" || MISSING="$MISSING $f"
  done
  [ -n "$MISSING" ] && no freeze-coverage "not in $FREEZE:$MISSING" || ok freeze-coverage
  BAD=$(sha256sum -c --quiet "$FREEZE" 2>&1 | head -5 | tr '\n' ';')
  [ -n "$BAD" ] && no freeze "sha256sum -c: $BAD" || ok freeze "$(wc -l < "$FREEZE") files match $FREEZE"
else
  no freeze "no freeze list $FREEZE (make_exam9_freeze.sh)"
fi
# 4b. PREREG-ROUND9.md = the frozen copy + appended revisions only (PREREG section 7)
if [ -f "$PREREG_FROZEN" ]; then
  python3 -c "import sys;a=open(sys.argv[1],'rb').read();b=open(sys.argv[2],'rb').read();sys.exit(0 if a.startswith(b) else 1)" "$PREREG" "$PREREG_FROZEN" \
    && ok prereg-append-only "$(( $(wc -c < "$PREREG") - $(wc -c < "$PREREG_FROZEN") )) bytes appended since the freeze" \
    || no prereg-append-only "$PREREG is not $PREREG_FROZEN + appended revisions"
else
  no prereg-append-only "no frozen prereg copy $PREREG_FROZEN (make_exam9_freeze.sh)"
fi
# 5. the adapter and run_mixed_v4.sh's own facts-version check
if [ -n "$ADAPTER" ] && [ -f "$ADAPTER/adapter_model.safetensors" ]; then
  AFV_PY=$(sed -n "/^read -r -d '' AFV_PY <<'PY'\$/,/^PY\$/p" "$EXP/run_mixed_v4.sh" | sed '1d;$d')
  AFV=$(python3 -c "$AFV_PY" "$ADAPTER")
  [ "${AFV%% *}" = "$FV" ] && ok facts-version "run_mixed_v4.sh check: $AFV" || no facts-version "arm $ARM needs $FV, run_mixed_v4.sh check says: $AFV"
else
  no adapter "no weights at ${ADAPTER:-?}/adapter_model.safetensors"
fi
# 6. batch and out-batches are new
for b in "$BATCH" "${OUTBATCHES[@]}"; do
  [ -e "$ROOT/live/$b" ] && no batch-new "$ROOT/live/$b exists" || ok batch-new "$b"
done
# 7. exam and make-up waves: the gate passed (or the owner approved a paused gate) under the current freeze list;
#    gate waves: the smoke passed under the current freeze list (a re-freeze makes both stale, PREREG sections 5-6)
pass_bound() {  # <pass file> <what> <python condition on the json p>
  python3 - "$1" "$FREEZE" "$3" <<'PY'
import hashlib, json, sys
try:
    p = json.load(open(sys.argv[1]))
except (OSError, ValueError) as exc:
    print(f'unreadable: {exc!r}'); sys.exit(1)
fz = hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest() if sys.argv[2] else ''
if not eval(sys.argv[3], {}, dict(p=p)):
    print(f'content does not qualify: {json.dumps(p)[:200]}'); sys.exit(1)
if p.get('freeze_sha256') != fz:
    print(f"freeze_sha256 {str(p.get('freeze_sha256'))[:12]} != current freeze list {fz[:12]} (stale: re-frozen since)"); sys.exit(1)
print(json.dumps(p)[:200])
PY
}
if [ "$KIND" = exam ] || [ "$KIND" = makeup ]; then
  if [ ! -f "$GATEPASS" ]; then no gate-pass "no $GATEPASS (launch_gate9.sh writes it on a pass; owner approval otherwise)"
  elif [ ! -f "$FREEZE" ]; then no gate-pass "no freeze list to bind $GATEPASS to"
  else
    GP=$(pass_bound "$GATEPASS" gate "p.get('gate') == 'pass' or bool(str(p.get('owner_approval') or '').strip())")
    [ $? -eq 0 ] && ok gate-pass "$GP" || no gate-pass "$GATEPASS: $GP"
  fi
fi
if [ "$KIND" = gate ]; then
  if [ ! -f "$SMOKEPASS" ]; then no smoke-pass "no $SMOKEPASS (launch_smoke9.sh)"
  elif [ ! -f "$FREEZE" ]; then no smoke-pass "no freeze list to bind $SMOKEPASS to"
  else
    SP=$(pass_bound "$SMOKEPASS" smoke "p.get('smoke') == 'pass'")
    [ $? -eq 0 ] && ok smoke-pass "$SP" || no smoke-pass "$SMOKEPASS: $SP"
  fi
fi
# 8. disk
FREE=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
[ "${FREE:-0}" -ge "$MIN_FREE" ] && ok disk "${FREE} GB free" || no disk "${FREE:-?} GB free < $MIN_FREE GB"
# 9. detached: SIGHUP ignored (nohup) and no controlling terminal (setsid)
IGN=$(awk '/^SigIgn/{print $2}' /proc/$$/status); TTY=$(ps -o tty= -p $$ 2>/dev/null | tr -d ' ')
if [ $(( 0x${IGN:-0} & 1 )) -eq 1 ] && [ "$TTY" = "?" ]; then ok detached "SIGHUP ignored, no tty"
elif [ $DRY = 1 ]; then echo "${TAG}check detached: not detached (SigIgn $IGN, tty $TTY); a real run refuses this"
else no detached "SIGHUP not ignored or tty $TTY: launch with setsid nohup (run_exam9_chain.sh header)"; fi
WD_CMD="setsid nohup bash $EXP/broker_watchdog.sh $BATCH $ADAPTER <driver-pid> > $ROOT/live/$BATCH/watchdog.log 2>&1 &"
RUN_CMD="cd $EXP && FACTS_VERSION=$FV EXTRA_HANDS='$EXTRA' exec bash run_mixed_v4.sh $BATCH $ADAPTER ${GAMES[*]}"
if [ $DRY = 1 ]; then
  echo "${TAG}wave.json: $WAVE_JSON"
  echo "${TAG}would write: $ROOT/live/$BATCH/prereg.sha256 (PREREG, frozen copy, freeze list, config), $ROOT/live/$BATCH/wave.json"
  echo "${TAG}would start: $WD_CMD"
  echo "${TAG}would run:   $RUN_CMD"
  [ $FAILS -eq 0 ] && { echo "${TAG}result: the wave would START"; exit 0; } || { echo "${TAG}result: the wave would be REFUSED ($FAILS check(s))"; exit 9; }
fi
[ $FAILS -eq 0 ] || { echo "refused: $FAILS check(s) failed; nothing started, nothing written"; exit 9; }
mkdir -p "$ROOT/live/$BATCH"
sha256sum "$PREREG" "$PREREG_FROZEN" "$FREEZE" "$CFG" > "$ROOT/live/$BATCH/prereg.sha256"
python3 -c "import json,sys,time; w=json.loads(sys.argv[1]); w.update(time=time.time(), freeze_sha256=sys.argv[2], driver_pid=int(sys.argv[3])); open(sys.argv[4],'w').write(json.dumps(w, indent=1))" \
  "$WAVE_JSON" "$(sha256sum "$FREEZE" | cut -d' ' -f1)" "$$" "$ROOT/live/$BATCH/wave.json"
setsid nohup bash "$EXP/broker_watchdog.sh" "$BATCH" "$ADAPTER" "$$" > "$ROOT/live/$BATCH/watchdog.log" 2>&1 < /dev/null &
echo "wave $W start $(date '+%F %T %Z'): watchdog pid $!; $RUN_CMD"
cd "$EXP" && FACTS_VERSION=$FV EXTRA_HANDS="$EXTRA" exec bash run_mixed_v4.sh "$BATCH" "$ADAPTER" "${GAMES[@]}"
