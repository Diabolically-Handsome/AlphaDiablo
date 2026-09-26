# OPS-launcher: launching case-level drivers and operator procedure (written for B1-E6)

**Source**: PREREG-B1 E6 (after the v32 OPS incident: v32 was launched with a hand-typed nohup and the repo had no
committed launcher; this document promotes the incident patch to a standard driver clause, committed with the freeze commit).

## Deliverables

| Item | Path | Notes |
|---|---|---|
| Standard launcher | `train/launch_case.sh` | nohup orphaning + `caffeinate -is` + log redirection + launch receipt + caffeinate assertion |
| launchd template | `train/ops/com.alphadiablo.case-driver.plist.template` | direct launchd hosting (PPID=1), three placeholders `__CASE__/__REPO__/__DRIVER__` |
| Operator procedure | this document | heartbeat thresholds written down |

## Standard launch

```bash
# B1 case (example; manually started gold-standard runs never go through the launcher and are still started by the operator)
train/launch_case.sh train/runs/infra-b1 train/run_b1_infra.py
```

Check three things right after launch:

1. **Orphaned**: `ps -o ppid= -p <PID>` (should be `1` after the shell exits);
2. **caffeinate in the tree**: `pgrep -f "caffeinate -is"` is non-empty;
3. **Log is being written**: `tail -f <case-dir>/driver.<stamp>.log`.

`caffeinate -is` prevents idle sleep and system sleep on AC power; it does not prevent sleep when a laptop lid is
closed (see `caffeinate -d` or keep the lid open).

## Heartbeat checks (thresholds written down; the operator follows the table)

| Phase | Heartbeat file | WARN | DEAD |
|---|---|---|---|
| Training leg running | `train/runs/<leg>/progress.jsonl` mtime | > 120 s | > 600 s |
| Evaluation/export/replay phase | `<case-dir>/driver.<stamp>.log` mtime | > 900 s | > 1800 s |
| Throughout | `pgrep -f "caffeinate -is"` | empty = WARN (sleep risk) | — |

- **On WARN**: observe only, do not intervene; re-check within 15 minutes; two WARNs in a row escalate to DEAD handling.
- **On DEAD**: follow the case's P lines (B1: P1 top-level exception/interruption clause) -- collect evidence first
  (`ps`, log tail, `status.json`, ledger tail), then `kill`; never restart before collecting evidence;
  restart through the driver's idempotent resume (exam_or_adopt / resume-reconciliation clauses); never hand-edit the ledger to keep a run alive.
- **Auxiliary readings**: `sps` in `train/runs/<leg>/status.json` (v32 legs ran at about 180 sps;
  below 60 sps for 5 minutes counts as WARN) and `updated_at` (same meaning as the progress mtime).

## launchd hosting (optional, system-level supervision)

```bash
# run from the repository root
sed -e "s/__CASE__/infra-b1/g" \
    -e "s#__REPO__#$(pwd)#g" \
    -e "s#__DRIVER__#train/run_b1_infra.py#g" \
    train/ops/com.alphadiablo.case-driver.plist.template \
    > ~/Library/LaunchAgents/com.alphadiablo.infra-b1.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.alphadiablo.infra-b1.plist
# unload
launchctl bootout gui/$(id -u)/com.alphadiablo.infra-b1
```

`KeepAlive=false` is deliberate: ledger budgets (2 leg launches / 2 evaluation runs) are **decided by people**,
and automatic revival could burn through the budget while nobody is watching; after a crash the operator collects
evidence per the P lines and restarts by hand.

## Relation to the driver's mutual exclusion

The driver has its own `.driver.lock` flock mutex (W8); the launcher adds no second mutex. A second process from a
duplicate launch dies on its own with exit code 4 (not idle / lock conflict) and leaves a log.
