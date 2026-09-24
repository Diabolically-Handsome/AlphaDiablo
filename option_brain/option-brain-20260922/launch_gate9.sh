#!/usr/bin/env bash
# WSL, RTX 5090. Round-9 pre-exam gate (PREREG-ROUND9.md decision 7, section 6): the new arm on the 19 training seeds
# of round-8 impact A (11 King, 8 Butcher), one wave, same wave script, flags and caps as the exam (run_exam9_wave.sh
# gate new GATE; batch gate9-new, or gate9-new-r<k> for the re-run after a code fix: launch_gate9.sh gate-r2), then
# check_exam9_wave.py --gate. Pause when: a deaths >= 5; b any class-B exception, or a cycle signature not in
# exam9_config.json gate.known_cycles; c the same no_progress_* type in >= 2 games; d a code sha / integrity failure;
# e the queue model projects the longest gate game at 16-game concurrency beyond 50% of the 720-minute cap; also
# >= 3 class-A games (check exit 3).
#   pass  -> <root>/exam9-gate-pass.json {"gate": "pass", "freeze_sha256": <sha of the freeze list>, ...}
#            (run_exam9_wave.sh accepts it only while the freeze list is unchanged)
#   pause -> <root>/exam9-gate-pause.json with the reasons. A code problem: fix -> make_exam9_freeze.sh --refreeze ->
#            launch_smoke9.sh smoke-r2 -> launch_gate9.sh gate-r2 (PREREG section 6). No code problem (e.g. deaths
#            are the brain's choices): the exam starts only with the owner's explicit approval, recorded in a revision
#            section of the prereg and by hand as <root>/exam9-gate-pass.json:
#              {"gate": "owner_approval", "owner_approval": "<who, when, what was approved>",
#               "pause_file": "<the pause json>", "freeze_sha256": "<sha256sum of exam9-freeze.sha256>"}
# Requires <root>/exam9-smoke-pass.json bound to the current freeze list (launch_smoke9.sh). A real run refuses unless
# detached (setsid nohup) and moves an old pause file aside. Written 2026-09-23; in the freeze list.
# usage: launch_gate9.sh [--dry-run] [gate|gate-r<k>]   (any other argument: exit 2)
# Launch (from Git Bash, background, wsl.exe attached):
#   wsl -e bash -c 'setsid -w nohup bash $AD_WORKSPACE/experiments/option-brain-20260922/launch_gate9.sh >> $AD_ROOT/live/gate9.log 2>&1'
set -u
EXP=$AD_WORKSPACE/experiments/option-brain-20260922
CFG=$EXP/exam9_config.json
DRY=0; W=gate
for x in "$@"; do
  case "$x" in
    --dry-run) DRY=1 ;;
    gate|gate-r[2-9]) W=$x ;;
    *) echo "usage: launch_gate9.sh [--dry-run] [gate|gate-r<k>] (got '$x')"; exit 2 ;;
  esac
done
eval "$(python3 -c "
import json, shlex
c = json.load(open('$CFG'))
print(f\"ROOT={shlex.quote(c['root'])}; FREEZE={shlex.quote(c['freeze_list'])}; PASS={shlex.quote(c['gate_pass_file'])}; SMOKEPASS={shlex.quote(c['smoke_pass_file'])}; ARM={c['gate']['arm']}\")
")"
B=gate9-$ARM${W#gate}
PAUSEF=$ROOT/exam9-gate-pause.json
say() { echo "$( [ $DRY = 1 ] && echo 'DRY-RUN ')$(date '+%F %T %Z') GATE9 $*"; }
smoke_ok() {  # the smoke pass file, bound to the current freeze list
  python3 - "$SMOKEPASS" "$FREEZE" <<'PY'
import hashlib, json, sys
try:
    p = json.load(open(sys.argv[1])); fz = hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()
except (OSError, ValueError) as exc:
    print(f'no usable smoke pass / freeze list: {exc!r}'); sys.exit(1)
if p.get('smoke') != 'pass' or p.get('freeze_sha256') != fz:
    print(f"smoke pass {json.dumps(p)[:160]} is not a pass under the current freeze list {fz[:12]}"); sys.exit(1)
print(f'smoke pass bound to freeze list {fz[:12]}')
PY
}
if [ $DRY = 1 ]; then
  SM=$(smoke_ok) && say "$SM" || say "would refuse: $SM (launch_smoke9.sh)"
  [ -e "$PASS" ] && say "would refuse: $PASS already exists"
  bash "$EXP/run_exam9_wave.sh" --dry-run "$W" "$ARM" GATE; say "wave dry run exit $?"
  say "would run: python3 -B $EXP/check_exam9_wave.py $B --gate --json $ROOT/live/$B/check.json"
  say "would write $PASS (exit 0, bound to the freeze list sha) or $PAUSEF (exit 1/3/4)"
  exit 0
fi
IGN=$(awk '/^SigIgn/{print $2}' /proc/$$/status); TTY=$(ps -o tty= -p $$ 2>/dev/null | tr -d ' ')
if ! { [ $(( 0x${IGN:-0} & 1 )) -eq 1 ] && [ "$TTY" = "?" ]; }; then
  say "refused: not detached (SIGHUP ignored: $(( 0x${IGN:-0} & 1 )), tty $TTY); launch with setsid nohup (header)"; exit 9
fi
SM=$(smoke_ok) || { say "refused: $SM (launch_smoke9.sh first)"; exit 9; }
[ -e "$PASS" ] && { say "refused: $PASS already exists"; exit 9; }
[ -e "$PAUSEF" ] && { mv "$PAUSEF" "$PAUSEF.superseded-$(date +%Y%m%d-%H%M%S)"; say "moved aside the old $PAUSEF"; }
say "gate $W start ($B); $SM"
bash "$EXP/run_exam9_wave.sh" "$W" "$ARM" GATE > "$ROOT/live/$B.driver.log" 2>&1
rc=$?; say "driver exit $rc"
[ -e "$ROOT/live/$B/wave.json" ] || { tail -30 "$ROOT/live/$B.driver.log"; say "refused or did not start"; exit 9; }
t0=$(date +%s)
while pgrep -f "^[^ ]*python[0-9.]* -B hf_broker\.py --adapter [^ ]+ --bus $ROOT/live/$B/bus\$" >/dev/null \
   || pgrep -f "^bash [^ ]*broker_watchdog\.sh $B " >/dev/null; do
  [ $(( $(date +%s) - t0 )) -ge 1200 ] && { say "broker/watchdog still alive after 20 min"; break; }; sleep 10
done
python3 -B "$EXP/check_exam9_wave.py" "$B" --gate --json "$ROOT/live/$B/check.json" > "$ROOT/live/$B/check.txt" 2>&1
rc=$?
grep -v '^PASS' "$ROOT/live/$B/check.txt" | tail -40
if [ $rc -eq 0 ]; then
  python3 -c "import json,sys,time,hashlib; json.dump(dict(gate='pass', batch=sys.argv[1], time=time.time(), check_sha256=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest(), freeze_sha256=hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest()), open(sys.argv[4],'w'), indent=1)" \
    "$B" "$ROOT/live/$B/check.json" "$FREEZE" "$PASS"
  say "PASS -> $PASS"
else
  python3 -c "import json,sys,time,os,hashlib; c=json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else dict(checks=[]); json.dump(dict(gate='pause', batch=sys.argv[1], exit=int(sys.argv[3]), time=time.time(), reasons=(c.get('gate') or {}).get('reasons'), failed_checks=[x['name'] for x in c['checks'] if not x['ok'] and x['severity']=='integrity'], freeze_sha256=hashlib.sha256(open(sys.argv[5],'rb').read()).hexdigest()), open(sys.argv[4],'w'), indent=1)" \
    "$B" "$ROOT/live/$B/check.json" "$rc" "$PAUSEF" "$FREEZE"
  say "PAUSE (check exit $rc) -> $PAUSEF; see $ROOT/live/$B/check.txt"
  exit $rc
fi
