#!/bin/bash
# B1-E6:案级驱动器标准点火器(PREREG-B1 E6;承 v32 OPS 事故,由事故补丁升格为
# 驱动器标准条款)。仓内此前无任何 committed launcher(v32 系临场手作)。
#
# 保证三件事:
#   1) 孤儿化:nohup + 后台 + disown,壳退出后驱动进程 PPID 归 1(launchd),
#      SSH/终端断线不再杀腿;值守核验:ps -o ppid= -p <PID> 应为 1。
#      (若需系统级看护/开机自启,用同目录 ops/ 下 launchd plist 模板直挂。)
#      PID 簿记口径(E8 勘正):receipt 之 pid 系【驱动器本体】——macOS
#      caffeinate 会 exec 驱动器于本进程、fork 子进程持电源断言,详见下方断言注。
#   2) 防睡眠:caffeinate -is 包裹整个驱动进程(-i 防 idle sleep,-s 防
#      system sleep;盒盖仍会睡,值守须保持供电+外接或 caffeinate -d 另议)。
#      点火后本脚本自动断言 caffeinate 在进程树内,断言失败即退出非零。
#   3) 日志重定向:stdout+stderr 追加到 <case-dir>/driver.<UTC时间戳>.log,
#      并落 launch receipt(PID/命令/日志路径)到 <case-dir>/launch_receipt.json。
#
# 心跳检查细则(值守操作项,阈值成文;详见 docs/OPS-launcher.md):
#   训练腿在跑时:train/runs/<leg>/progress.jsonl 的 mtime 距今
#     > 120 秒 = WARN(采样降速/卡窗),> 600 秒 = DEAD(按 P 线处置);
#   驱动器评测/导出阶段:driver 日志 mtime 距今 > 900 秒 = WARN,
#     > 1800 秒 = DEAD;
#   caffeinate 断言:pgrep -f "caffeinate -is" 非空,否则 WARN(睡眠风险)。
#
# 用法:
#   train/launch_case.sh <case-dir> <driver.py> [driver args...]
# 例:
#   train/launch_case.sh train/runs/infra-b1 train/run_b1_infra.py
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "用法: $0 <case-dir> <driver.py> [args...]" >&2
  exit 64
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE_DIR="$1"; shift
DRIVER="$1"; shift
PY="$ROOT/.venv/bin/python"

[ -x "$PY" ] || { echo "缺 venv python: $PY" >&2; exit 66; }
[ -f "$ROOT/$DRIVER" ] || [ -f "$DRIVER" ] || { echo "缺驱动器: $DRIVER" >&2; exit 66; }
mkdir -p "$ROOT/$CASE_DIR"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$ROOT/$CASE_DIR/driver.$STAMP.log"

cd "$ROOT"
nohup caffeinate -is "$PY" "$DRIVER" "$@" >>"$LOG" 2>&1 &
PID=$!
disown "$PID"

# caffeinate 断言(E8 修复,B1 判决 OPS 段 minor"查错进程号"在案):
#   macOS caffeinate 语义实测——caffeinate 在本进程 exec 所托 utility(驱动器),
#   把电源断言留在 fork 出的子进程;故 $PID(=$!,经 nohup→caffeinate exec 链)
#   系驱动器本体,caffeinate 系 $PID 之【子】。原断言"ps -o command= -p $PID
#   应为 caffeinate"查的是错进程号:点火成功时 $PID 命令行显示为驱动器,
#   断言反而失败(B1 实弹误报 exit 71,launch_receipt 未落而驱动器安然在跑)。
#   修复 = 断言 caffeinate 在 $PID 进程树内(本体或其子进程,双形兼容其它
#   caffeinate 实现);孤儿化/caffeinate -is/日志/receipt/心跳行为面不变。
sleep 1
if ! ps -p "$PID" >/dev/null 2>&1; then
  echo "点火即死:见日志 $LOG" >&2
  tail -n 20 "$LOG" >&2 || true
  exit 70
fi
if ! { ps -o command= -p "$PID" | grep -q "caffeinate" \
       || pgrep -P "$PID" -f caffeinate >/dev/null 2>&1; }; then
  echo "caffeinate 断言失败:PID $PID 进程树内无 caffeinate" >&2
  exit 71
fi

RECEIPT="$ROOT/$CASE_DIR/launch_receipt.json"
printf '{"pid": %d, "driver": "%s", "args": "%s", "log": "%s", "launched_utc": "%s", "launcher": "train/launch_case.sh"}\n' \
  "$PID" "$DRIVER" "$*" "$LOG" "$STAMP" >"$RECEIPT"

echo "已点火(孤儿化+caffeinate -is):PID=$PID"
echo "  日志: $LOG"
echo "  回执: $RECEIPT"
echo "  值守核验: ps -o ppid= -p $PID   # 壳退出后应为 1"
echo "  心跳细则: docs/OPS-launcher.md(progress.jsonl mtime 120s WARN / 600s DEAD)"
