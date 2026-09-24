#!/usr/bin/env bash
# WSL. The round-9 paired final exam, start to end (PREREG-ROUND9.md decisions 2-3; engineering review step 5):
# the four ABBA waves of exam9_config.json 'waves' (1 new A, 2 old A, 3 old B, 4 new B), each through
# run_exam9_wave.sh (refusal checks, sha256sum -c against the freeze list, watchdog, run_mixed_v4.sh), then
# check_exam9_wave.py on it (live/<batch>/check.json + check.txt). The chain STOPS at the first wave that is refused
# or whose check exits non-zero:
#   exit 1 (integrity): the wave is void (PREREG 3.3). Find the cause, write it into the revision section, then resume
#          with --from <wave>r2 (re-runs that wave whole with the same seeds and frozen code as exam9-w<wave>r2-<arm>,
#          then continues with the later waves).
#   exit 3 (>= 3 class-A games): the wave COUNTS, its class-A games go to the make-up wave. Find the cause, write it
#          into the revision section, then resume with --from <next wave> (or --from m after wave 4).
# After wave 4: one make-up wave per arm (m1: exam9-m1-<arm>, PREREG 4.5) for the class-A games of the primary waves
# that have no make-up in a counted make-up wave yet (identical frozen code and command line apart from --out/--tag; a
# second failure counts 0), checked the same way; a voided make-up wave is re-run whole as m1r<k> by --from m. Then
# score_exam9.py --exam (not --final) writes live/exam9-score.{json,md}.
# If the chain itself dies (WSL or host crash, reboot): the games of the wave that was running have no result.json
# (class A, make-up wave). Run the check of that wave by hand, then resume with --from <next wave>:
#   python3 -B <EXP>/check_exam9_wave.py <batch> --json <ROOT>/live/<batch>/check.json > <ROOT>/live/<batch>/check.txt
# --from refuses unless every earlier wave has a counted check (check.json exit 0 or 3, from the latest run of that
# wave, exam9_lib.check_status). No human step between waves; nothing here changes code or config. Written
# 2026-09-23; in the freeze list.
#
# usage: run_exam9_chain.sh [--dry-run] [--from <1-4|<1-4>r<k>|m>] [--print-launch]
#   --dry-run       every wave's refusal checks and command lines (run_exam9_wave.sh --dry-run), nothing started.
#   --from          resume after a stop was investigated (see above); earlier waves must have counted checks.
#   --print-launch  print the launch command and exit.
# Launch (from Git Bash, as a background task so that wsl.exe stays attached; setsid: no controlling terminal, nohup:
# SIGHUP ignored by every game, so a closed terminal is not an operator stop; >> keeps the log of an earlier start):
#   wsl -e bash -c 'setsid -w nohup bash $AD_WORKSPACE/experiments/option-brain-20260922/run_exam9_chain.sh >> $AD_ROOT/live/exam9-chain.log 2>&1'
# (with --from X: put '--from X' after run_exam9_chain.sh). Keep the machine awake and Windows Update restarts paused
# for the whole run (about 7-10 hours).
set -u
EXP=$AD_WORKSPACE/experiments/option-brain-20260922
CFG=$EXP/exam9_config.json
DRY=0; FROM=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --from) [ $# -ge 2 ] || { echo "--from needs a value"; exit 2; }; FROM=$2; shift 2 ;;
    --print-launch) sed -n '/^#   wsl -e bash/p' "$0" | sed 's/^#   //'; exit 0 ;;
    *) sed -n '/^# usage:/,/^#   --print-launch/p' "$0"; exit 2 ;;
  esac
done
case "$FROM" in 1|2|3|4|[1-4]r[2-9]|m) ;; *) echo "--from must be 1-4, <1-4>r<k> (k 2-9) or m (got '$FROM')"; exit 2 ;; esac
FROM_W=${FROM%%r*}  # 1-4 or m
ROOT=$(python3 -c "import json; print(json.load(open('$CFG'))['root'])")
WAVES=$(python3 -c "import json; print('\n'.join(f\"{w['wave']} {w['arm']} {w['half']}\" for w in json.load(open('$CFG'))['waves']))")
TAG=""; [ $DRY = 1 ] && TAG="DRY-RUN "
say() { echo "${TAG}$(date '+%F %T %Z') CHAIN $*"; }
if [ $DRY = 0 ]; then
  IGN=$(awk '/^SigIgn/{print $2}' /proc/$$/status); TTY=$(ps -o tty= -p $$ 2>/dev/null | tr -d ' ')
  if ! { [ $(( 0x${IGN:-0} & 1 )) -eq 1 ] && [ "$TTY" = "?" ]; }; then
    say "refused: not detached (SIGHUP ignored: $(( 0x${IGN:-0} & 1 )), tty $TTY); launch with --print-launch's command"; exit 9
  fi
fi
# every wave before FROM must have a counted check (latest run of that wave)
if [ "$FROM" != 1 ]; then
  PRE=$(cd "$EXP" && python3 -B -c "
import sys, exam9_lib as L
from pathlib import Path
c = L.load_config(); live = Path(c['root']) / 'live'
frm = sys.argv[1]
bad = []
for w in c['waves']:
    if frm != 'm' and int(w['wave']) >= int(frm):
        continue
    b = L.latest_wave_batch(live, w['wave'], w['arm'])
    if b is None:
        bad.append(f\"wave {w['wave']} ({w['arm']}) never started\")
        continue
    ok, ex, why = L.check_status(live, b)
    print('ok', why) if ok else bad.append(why)
for x in bad:
    print('BAD', x)
" "$FROM_W") || { say "refused: could not read the earlier waves' checks"; exit 9; }
  echo "$PRE" | sed "s/^/${TAG}  earlier wave: /"
  if echo "$PRE" | grep -q '^BAD'; then
    if [ $DRY = 1 ]; then say "a real run would refuse --from $FROM: an earlier wave has no counted check"; PRE_REFUSED=9
    else say "refused --from $FROM: an earlier wave has no counted check (see above)"; exit 9; fi
  fi
fi
settle() {  # wait until the batch's broker(s) and watchdog are gone (the next wave refuses a running broker)
  local b=$1 t0; t0=$(date +%s)
  while pgrep -f "^[^ ]*python[0-9.]* -B hf_broker\.py --adapter [^ ]+ --bus $ROOT/live/$b/bus\$" >/dev/null \
     || pgrep -f "^bash [^ ]*broker_watchdog\.sh $b " >/dev/null; do
    [ $(( $(date +%s) - t0 )) -ge 1200 ] && { say "broker or watchdog of $b still alive after 20 min"; return 1; }
    sleep 10
  done
}
run_wave() {  # <wave> <arm> <set> [make-up seed args...]
  local w=$1 arm=$2 set=$3; shift 3
  local b=exam9-w$w-$arm; case $w in m*) b=exam9-$w-$arm ;; esac
  if [ $DRY = 1 ]; then
    bash "$EXP/run_exam9_wave.sh" --dry-run "$w" "$arm" "$set" "$@"; local rc=$?
    say "wave $w $arm $set: dry run exit $rc"; DRYRC=$(( DRYRC | rc )); return 0
  fi
  say "wave $w ($arm, $set) start"
  bash "$EXP/run_exam9_wave.sh" "$w" "$arm" "$set" "$@" > "$ROOT/live/$b.driver.log" 2>&1
  local rc=$?
  say "wave $w driver exit $rc (log $ROOT/live/$b.driver.log)"
  if [ $rc -eq 9 ] || [ ! -e "$ROOT/live/$b/wave.json" ]; then
    tail -30 "$ROOT/live/$b.driver.log"; say "STOPPED: wave $w refused or did not start"; exit 9
  fi
  settle "$b" || { say "STOPPED after wave $w: broker/watchdog did not exit"; exit 5; }
  python3 -B "$EXP/check_exam9_wave.py" "$b" --json "$ROOT/live/$b/check.json" > "$ROOT/live/$b/check.txt" 2>&1
  rc=$?
  grep -v '^PASS' "$ROOT/live/$b/check.txt" | tail -40
  case $rc in
    0) say "wave $w checked ok" ;;
    3) say "STOPPED: wave $w counts but has >= 3 class-A games (check exit 3, $ROOT/live/$b/check.txt); investigate, write a revision, resume with --from <next wave|m>"; exit 3 ;;
    *) local nxt="${w%%r*}r<k>"; case $w in m*) nxt=m ;; esac
       say "STOPPED: wave $w is void (check exit $rc, $ROOT/live/$b/check.txt); investigate, write a revision, resume with --from $nxt"; exit $rc ;;
  esac
}
DRYRC=${PRE_REFUSED:-0}
say "start (from $FROM); waves: $(echo "$WAVES" | tr '\n' ';')"
if [ "$FROM_W" != m ]; then
  while read -r w arm half; do
    [ "$w" -lt "$FROM_W" ] && continue
    if [ "$w" = "$FROM_W" ] && [ "$FROM" != "$FROM_W" ]; then run_wave "$FROM" "$arm" "$half"  # e.g. 2r2
    else run_wave "$w" "$arm" "$half"; fi
  done <<< "$WAVES"
fi
# make-up wave per arm: class-A primaries (exam9_lib.classify + refine_wave, per wave) without a make-up in a counted
# make-up run; the wave is m1, or m1r<k+1> when make-up runs 1..k exist (a voided make-up wave is re-run whole)
MK=$(cd "$EXP" && python3 -B -c "
import exam9_lib as L
from pathlib import Path
c = L.load_config(); live = Path(c['root']) / 'live'
cls = {}
def classify(d):
    bb = L.broker_batch_of(d.parent)
    if str(d) not in cls:
        ev = L.broker_death_evidence(live, bb)
        res = L.refine_wave([L.classify(x, ev) for x in L.wave_game_dirs(live, bb)])
        cls.update((r['dir'], r) for r in res)
    return cls.get(str(d)) or L.classify(d)
for arm in ('new', 'old'):
    runs = L.makeup_runs(live, arm)
    counted = {name for _, name in runs if L.check_status(live, name)[0]}
    need = {'skeleton_king': [], 'butcher': []}
    for m, seeds in (('skeleton_king', c['king_seeds']), ('butcher', c['butcher_seeds'])):
        for s in seeds:
            ob, _ = L.primary_batch(c, live, arm, m, s)
            d = live / ob / f's{s}'
            if not d.is_dir() or classify(d)['cls'] != 'A':  # a counted wave has every game directory (check)
                continue
            mk = L.MISSION_KEY[m]
            if any((live / f'{name}-{mk}' / f's{s}').is_dir() for name in counted):
                continue
            need[m].append(s)
    if need['skeleton_king'] or need['butcher']:
        w = 'm1' if not runs else f'm1r{runs[-1][0] + 1}'
        print(w, arm, ','.join(map(str, need['skeleton_king'])) or '-', ','.join(map(str, need['butcher'])) or '-')
") || { say "STOPPED: the make-up list could not be computed"; exit 6; }
if [ -z "$MK" ]; then
  say "no class-A game to make up"
else
  while read -r mw arm ks bs; do
    [ "$ks" = - ] && ks=""; [ "$bs" = - ] && bs=""
    say "make-up wave $mw for $arm: King [${ks//,/ }] Butcher [${bs//,/ }]"
    run_wave "$mw" "$arm" MK --king "${ks//,/ }" --butcher "${bs//,/ }"
  done <<< "$MK"
fi
if [ $DRY = 1 ]; then say "dry run done (exit $DRYRC: 0 = every wave would start)"; exit $DRYRC; fi
python3 -B "$EXP/score_exam9.py" --exam --out-json "$ROOT/live/exam9-score.json" --out-md "$ROOT/live/exam9-score.md"
say "done; score in $ROOT/live/exam9-score.md (not --final: re-score with --final once nothing is pending)"
