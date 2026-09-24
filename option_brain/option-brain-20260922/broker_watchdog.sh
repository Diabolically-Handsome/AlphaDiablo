#!/usr/bin/env bash
# WSL. Broker watchdog for one round-9 exam / gate / smoke batch (PREREG-ROUND9.md decision 2; engineering review
# step 5). run_exam9_wave.sh starts it detached (setsid nohup) before it execs run_mixed_v4.sh. Written 2026-09-23;
# in the exam freeze list.
# usage: broker_watchdog.sh <batch> <adapter> <driver-pid>
# Every poll_seconds (exam9_config.json 'watchdog'):
#   - bus/STOP exists (run_mixed_v4.sh touches it after the last game): wait up to wait_broker_exit_seconds for every
#     broker on this bus to exit (a restarted one sees STOP itself), then exit. A bus/STOP is never overridden.
#   - the broker has been ready once (bus/broker-ready.json), no hf_broker process serves this bus, games of this bus
#     are alive, and neither bus/STOP nor <root>/STOP exists (checked again right before the restart; <root>/STOP is
#     step 1 of a deliberate stop, live_runner.py lines 55-58: then the watchdog notes it once and never restarts):
#     start the same command on the same bus ($PY_MODEL -B hf_broker.py --adapter <adapter> --bus <bus>, cwd the
#     experiment directory), at most max_restarts times and not within min_gap_seconds of the previous restart. The
#     brain is deterministic (one forward pass, argmax) and a game waits paused (up to 180 s) for its answer; the
#     request of the dead broker stays in bus/requests (hf_broker unlinks it only after answering), so the new broker
#     answers it. Each restart: a line in live/<batch>/watchdog-restarts.jsonl, the previous broker-ready.json kept as
#     broker-ready-before-restart-<k>.json, the new broker's log in broker-restart-<k>.log.
#   - the driver (run_mixed_v4.sh) is gone and no game of this bus is alive: if a broker this watchdog started is
#     still up, touch bus/STOP (nothing left to serve) and wait for it; then exit.
#   - a request older than 120 s while a broker is alive: a warning line only (a hung broker is not killed; the
#     game then ends as engineering_failure 'broker did not answer', which is class A: exam9_lib.classify, PREREG 4.5).
# Processes are matched with '^[^ ]*python[0-9.]* -B ...' so that neither pgrep nor a shell quoting the pattern matches.
set -u
EXP=$AD_WORKSPACE/experiments/option-brain-20260922
[ $# -eq 3 ] || { sed -n '5,22p' "$0"; exit 2; }
B=$1; ADAPTER=$2; DRIVER=$3
eval "$(python3 -c "
import json, shlex
c = json.load(open('$EXP/exam9_config.json')); w = c['watchdog']
print(f\"ROOT={shlex.quote(c['root'])}; PY_MODEL={shlex.quote(c['py_model'])}; POLL={w['poll_seconds']}; MAX={w['max_restarts']}; GAP={w['min_gap_seconds']}; WAIT={w['wait_broker_exit_seconds']}\")
")"
R=$ROOT/live/$B; BUS=$R/bus; JL=$R/watchdog-restarts.jsonl
cd "$EXP" || exit 2  # the broker is started from here, as run_mixed_v4.sh does
say() { echo "$(date '+%F %T %Z') $*"; }
brokers() { pgrep -f "^[^ ]*python[0-9.]* -B hf_broker\.py --adapter [^ ]+ --bus $BUS\$" | tr '\n' ' '; }
games() { pgrep -f "^[^ ]*python[0-9.]* -B live_runner\.py .*--bus $BUS " | wc -l; }
wait_exit() {
  local t0; t0=$(date +%s)
  while [ -n "$(brokers)" ] && [ $(( $(date +%s) - t0 )) -lt $WAIT ]; do sleep 5; done
  [ -n "$(brokers)" ] && say "WARNING broker(s) still alive after ${WAIT}s: $(brokers)"
}
say "watchdog for $B: bus $BUS, adapter $ADAPTER, driver pid $DRIVER, poll ${POLL}s, max $MAX restarts, gap ${GAP}s"
k=0; last=0; mine=""; gave_up=0; warned=""; stop_noted=0
while true; do
  if [ -e "$BUS/STOP" ]; then
    say "bus/STOP present; waiting for the broker(s) to exit"
    wait_exit
    say "exit ($k restart(s))"
    exit 0
  fi
  n=$(games)
  if ! kill -0 "$DRIVER" 2>/dev/null && [ "$n" -eq 0 ]; then
    if [ -n "$mine" ] && [ -n "$(brokers)" ]; then
      say "driver $DRIVER gone, no game alive, broker $(brokers)started here still up: touching bus/STOP"
      touch "$BUS/STOP"
      wait_exit
    fi
    say "driver gone and no game alive; exit ($k restart(s))"
    exit 0
  fi
  if [ -f "$BUS/broker-ready.json" ] && [ -z "$(brokers)" ] && [ "$n" -gt 0 ]; then
    now=$(date +%s)
    if [ -e "$BUS/STOP" ] || [ -e "$ROOT/STOP" ]; then
      [ "${stop_noted:-0}" = 1 ] || say "broker dead, STOP present ($( [ -e "$ROOT/STOP" ] && echo "$ROOT/STOP" || echo bus/STOP)): deliberate stop, no restart"
      stop_noted=1
    elif [ $k -ge $MAX ]; then
      [ $gave_up = 0 ] && say "GIVING UP: broker dead again after $k restarts; games will end 'broker did not answer' (class A)"
      gave_up=1
    elif [ $(( now - last )) -ge $GAP ]; then
      k=$((k+1)); last=$now
      pend=$(ls "$BUS/requests" 2>/dev/null | grep -c '\.json$')
      prev=$R/broker.log; [ $k -gt 1 ] && prev=$R/broker-restart-$((k-1)).log
      say "broker dead ($n games alive, $pend pending requests); last lines of $prev:"
      tail -5 "$prev" 2>/dev/null | sed 's/^/    /'
      cp "$BUS/broker-ready.json" "$BUS/broker-ready-before-restart-$k.json"
      nohup "$PY_MODEL" -B hf_broker.py --adapter "$ADAPTER" --bus "$BUS" > "$R/broker-restart-$k.log" 2>&1 < /dev/null &
      bp=$!; mine="$mine $bp"
      say "restart $k: broker pid $bp, log $R/broker-restart-$k.log"
      echo "{\"k\":$k,\"time\":$now,\"pid\":$bp,\"games_alive\":$n,\"pending_requests\":$pend,\"log\":\"$R/broker-restart-$k.log\"}" >> "$JL"
    fi
  fi
  if [ -n "$(brokers)" ]; then
    old=$(find "$BUS/requests" -name '*.json' -mmin +2 2>/dev/null | head -1)
    if [ -n "$old" ] && [ "$old" != "$warned" ]; then say "WARNING request $old waits > 120 s with a broker alive (hung?)"; warned=$old; fi
  fi
  sleep "$POLL"
done
