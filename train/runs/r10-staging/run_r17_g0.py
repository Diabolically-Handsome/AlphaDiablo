"""R17.0 修正案一 G0 冒烟(run_r17_g0.py = run_r16_g0.py + 托管尺 v2)。

配方 = R16 冻结配方(r16-PREREG-FROZEN §四;G0-6 裁决 clear 1.0 / far 0)
+ 修正案一:--worker-descend-escrow-readiness-table v2(托管战备门与
readiness-v3 教练同尺)。验收(r17 合议庭 §四.5 G0-5):descends/10 窗 ≥ 1.0;
stall 份额 ≤ 25%;descend_escrow_vested > 0;零 RuntimeError(rc 0)。
用法:run_r17_g0.py <run-name> [total_steps=8192] [hp_price=0.1]
      [potion_bonus=2.0] [far_tiles=0] [a13_prior=2.0] [readiness_table=v2]
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
RUNS = ROOT / "train" / "runs"
STAGING = RUNS / "r10-staging"
LEDGER = STAGING / "r13_ledger.jsonl"
PY = str(ROOT / ".venv" / "bin" / "python")
CERT_WORKER = RUNS / "r9-reeducation" / "staging" / "worker.zip"


def log(event):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("[ledger]", json.dumps(event, ensure_ascii=False), flush=True)


def main():
    run_name = sys.argv[1]
    total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    hp_price = sys.argv[3] if len(sys.argv) > 3 else "0.1"
    potion_bonus = sys.argv[4] if len(sys.argv) > 4 else "2.0"
    far_tiles = sys.argv[5] if len(sys.argv) > 5 else "0"
    a13_prior = sys.argv[6] if len(sys.argv) > 6 else "2.0"
    readiness_table = sys.argv[7] if len(sys.argv) > 7 else "v2"
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
           "--gamma", "1.0", "--max-steps", "6000",
           "--num-envs", "4", "--n-steps", "512",
           "--total-steps", str(total_steps), "--seed", "22",
           "--device", "cpu", "--run-name", run_name,
           "--artifact-scope", "candidate",
           "--resume-from", str(CERT_WORKER),
           "--allow-environment-restart-resume", "--allow-manager-change",
           "--gradient-clip-mode", "separate-root-context-critic-v2",
           "--manager-heuristic", "readiness-v3",
           "--worker-action14-logit-bonus", "2.5",
           "--worker-dive-action11-logit-bonus", "2.0",
           "--worker-potion-action13-logit-bonus", a13_prior,
           "--worker-descend-escrow-fraction", "0.5",
           "--worker-descend-escrow-power", "1.6",
           "--worker-descend-escrow-readiness-gate",
           "--worker-descend-escrow-readiness-table", readiness_table,
           "--worker-hp-loss-price", hp_price,
           "--worker-potion-pickup-bonus", potion_bonus,
           "--worker-no-progress-timeout-credit", "zero",
           "--explore-global-hunt",
           "--progress-far-tiles", far_tiles,
           "--farm-scene-cap", "3600",
           "--reset-layer-clock-on-window",
           "--lr", "0.0001", "--ent-coef", "0.005",
           "--target-kl", "0.01", "--drink-sovereignty",
           "--worker-policy-observation-view", "dual-v4-asymmetric-v3",
           "--worker-fast-forward-reward-credit", "terminal-death-only",
           "--worker-additional-terminal-death-cost", "0.0",
           "--reward-economy", "v4",
           "--worker-learning-window-scope", "farm-dive-v1",
           "--sentinel-every", "2048", "--ckpt-every-steps", "63488"]
    log({"event": "r17_g0_launch", "run": run_name, "total_steps": total_steps,
         "hp_loss_price": hp_price, "potion_pickup_bonus": potion_bonus,
         "progress_far_tiles": far_tiles, "a13_prior": a13_prior,
         "readiness_table": readiness_table, "cmd": cmd[1:]})
    with open(STAGING / f"train-{run_name}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=ROOT, timeout=14400).returncode
    if rc != 0:
        log({"event": "R17_G0_FAILURE", "run": run_name, "rc": rc})
        raise SystemExit(1)
    audit = RUNS / run_name / "r13_dive_audit.jsonl"
    lines = [json.loads(l) for l in open(audit) if l.strip()]
    last = lines[-1] if lines else {}
    s = float(last.get("dive_share", 0.0))
    d10 = float(last.get("descends_per_10_windows", 0.0))
    windows = int(last.get("dive_live_windows", 0))
    stall_share = (float(last.get("dive_live_stalls", 0)) / windows
                   if windows else 0.0)
    vested = float(last.get("descend_escrow_vested", 0.0))
    ready_n = int(last.get("descend_escrow_ready_vested_count", 0))
    gate_file = RUNS / run_name / "r17_descend_gate.jsonl"
    gate_rows = ([json.loads(l) for l in open(gate_file) if l.strip()]
                 if gate_file.exists() else [])
    gate_summary = {
        "rows": len(gate_rows),
        "passed": sum(1 for r in gate_rows if r.get("passed")),
        "by_depth": {str(d): sum(1 for r in gate_rows if r.get("depth") == d)
                     for d in sorted({r.get("depth") for r in gate_rows})}}
    T = int(round(266240 / max(1e-6, 1.0 - s) / 2048) * 2048) if s < 1 else None
    checks = {"descends_per_10_ge_1": d10 >= 1.0,
              "stall_share_le_0.25": stall_share <= 0.25,
              "vested_gt_0": vested > 0.0}
    verdict = ("G0-5 验收通过" if all(checks.values())
               else "验收未过:" + ",".join(k for k, v in checks.items() if not v))
    log({"event": "r17_g0_result", "run": run_name, "dive_share": s,
         "descends_per_10_windows": d10, "stall_share": stall_share,
         "descend_escrow_vested": vested,
         "descend_escrow_ready_vested_count": ready_n,
         "descend_gate_summary": gate_summary,
         "descend_gate_rows": gate_rows[:64],
         "T_calibrated": T, "checks": checks,
         "audit_final": last, "verdict": verdict})
    print(json.dumps({"dive_share": s, "descends_per_10_windows": d10,
                      "stall_share": stall_share, "vested": vested,
                      "ready_vested_count": ready_n, "T": T,
                      "gate_summary": gate_summary,
                      "gate_rows": gate_rows[:64],
                      "checks": checks, "verdict": verdict, "audit": last},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
