#!/bin/bash
# B1-E6: standard launcher for case-level drivers (PREREG-B1 E6; after the v32 OPS incident the incident patch was
# promoted to a standard driver clause). Before this the repo had no committed launcher (v32 was launched by hand).
#
# It guarantees three things:
#   1) Orphaning: nohup + background + disown, so after the shell exits the driver's PPID becomes 1 (launchd)
#      and an SSH/terminal disconnect no longer kills a leg; operator check: ps -o ppid= -p <PID> should print 1.
#      (For system-level supervision or start at boot, use the launchd plist template under ops/ in this directory.)
#      PID bookkeeping (E8 correction): the receipt pid is the driver process itself -- on macOS
#      caffeinate execs the driver in its own process and forks a child that holds the power assertion; see the assertion note below.
#   2) No sleep: caffeinate -is wraps the whole driver process (-i prevents idle sleep, -s prevents
#      system sleep while on AC power; it does not prevent lid-close sleep, see caffeinate -d and OPS-launcher.md).
#      After launch this script asserts that caffeinate is in the process tree and exits non-zero if it is not.
#   3) Log redirection: stdout+stderr are appended to <case-dir>/driver.<UTC timestamp>.log,
#      and a launch receipt (PID/command/log path) is written to <case-dir>/launch_receipt.json.
#
# Heartbeat checks (operator procedure, thresholds written down; see docs/protocol/OPS-launcher.md):
#   while a training leg runs: mtime of train/runs/<leg>/progress.jsonl older than
#     120 s = WARN (sampling slowed / stuck window), 600 s = DEAD (handle per the P lines);
#   driver evaluation/export phase: driver log mtime older than 900 s = WARN,
#     1800 s = DEAD;
#   caffeinate check: pgrep -f "caffeinate -is" non-empty, otherwise WARN (sleep risk).
#
# Usage:
#   train/launch_case.sh <case-dir> <driver.py> [driver args...]
# Example:
#   train/launch_case.sh train/runs/infra-b1 train/run_b1_infra.py
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <case-dir> <driver.py> [args...]" >&2
  exit 64
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE_DIR="$1"; shift
DRIVER="$1"; shift
PY="$ROOT/.venv/bin/python"

[ -x "$PY" ] || { echo "missing venv python: $PY" >&2; exit 66; }
[ -f "$ROOT/$DRIVER" ] || [ -f "$DRIVER" ] || { echo "missing driver: $DRIVER" >&2; exit 66; }
mkdir -p "$ROOT/$CASE_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/$CASE_DIR/driver.$STAMP.log"

cd "$ROOT"
nohup caffeinate -is "$PY" "$DRIVER" "$@" >>"$LOG" 2>&1 &
PID=$!
disown "$PID"

# caffeinate assertion (E8 fix; the B1 verdict OPS section recorded a minor "checked the wrong PID"):
#   measured macOS caffeinate semantics -- caffeinate execs the given utility (the driver) in its own process
#   and keeps the power assertion in a forked child; so $PID (=$!, via the nohup->caffeinate exec chain)
#   is the driver itself and caffeinate is a child of $PID. The old assertion "ps -o command= -p $PID
#   should be caffeinate" checked the wrong PID: on a successful launch the $PID command line shows the driver,
#   so the assertion failed (a B1 live false alarm, exit 71: launch_receipt was not written while the driver ran fine).
#   Fix = assert that caffeinate is in the process tree of $PID (the process itself or its child, so other
#   caffeinate implementations also pass); orphaning/caffeinate -is/log/receipt/heartbeat behaviour is unchanged.
sleep 1
if ! ps -p "$PID" >/dev/null 2>&1; then
  echo "died on launch: see log $LOG" >&2
  tail -n 20 "$LOG" >&2 || true
  exit 70
fi
if ! { ps -o command= -p "$PID" | grep -q "caffeinate" \
       || pgrep -P "$PID" -f caffeinate >/dev/null 2>&1; }; then
  echo "caffeinate assertion failed: no caffeinate in the process tree of PID $PID" >&2
  exit 71
fi

RECEIPT="$ROOT/$CASE_DIR/launch_receipt.json"
printf '{"pid": %d, "driver": "%s", "args": "%s", "log": "%s", "launched_utc": "%s", "launcher": "train/launch_case.sh"}\n' \
  "$PID" "$DRIVER" "$*" "$LOG" "$STAMP" >"$RECEIPT"

echo "launched (orphaned + caffeinate -is): PID=$PID"
echo "  log: $LOG"
echo "  receipt: $RECEIPT"
echo "  operator check: ps -o ppid= -p $PID   # should be 1 after the shell exits"
echo "  heartbeat rules: docs/protocol/OPS-launcher.md (progress.jsonl mtime 120s WARN / 600s DEAD)"
