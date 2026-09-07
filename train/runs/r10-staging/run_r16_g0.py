"""R16 修宪 G0 定标跑批:新法全开的课堂发生性验收 + 定标 s → T。

配方(r16-PREREG-DRAFT §四):认证工人续训;教练 readiness-v3;economy v4;
scope farm-dive-v1;托管 0.5×d^1.6 + 战备条件托管;a11 先验 2.0;a13 先验 2.0;
hp_loss_price 0.1 / potion_pickup_bonus 2.0(暂定,G0 校准);timeout_credit=zero;
explore_global_hunt;progress_far_tiles 3;farm_scene_cap 3600(配 max_steps 6000);
reset_layer_clock_on_window;drink_sovereignty 开放。
用法:run_r16_g0.py <run-name> [total_steps=8192] [hp_price=0.1] [potion_bonus=2.0]
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
    far_tiles = sys.argv[5] if len(sys.argv) > 5 else "3"
    a13_prior = sys.argv[6] if len(sys.argv) > 6 else "2.0"
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
    log({"event": "r16_g0_launch", "run": run_name, "total_steps": total_steps,
         "hp_loss_price": hp_price, "potion_pickup_bonus": potion_bonus,
         "progress_far_tiles": far_tiles, "a13_prior": a13_prior,
         "cmd": cmd[1:]})
    with open(STAGING / f"train-{run_name}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=ROOT, timeout=14400).returncode
    if rc != 0:
        log({"event": "R16_G0_FAILURE", "run": run_name, "rc": rc})
        raise SystemExit(1)
    audit = RUNS / run_name / "r13_dive_audit.jsonl"
    lines = [json.loads(l) for l in open(audit) if l.strip()]
    last = lines[-1] if lines else {}
    s = float(last.get("dive_share", 0.0))
    d10 = float(last.get("descends_per_10_windows", 0.0))
    T = int(round(266240 / max(1e-6, 1.0 - s) / 2048) * 2048) if s < 1 else None
    verdict = ("课堂发生性验收通过" if d10 >= 1.0 else "验收未过(<1 descend/10 窗)")
    log({"event": "r16_g0_result", "run": run_name, "dive_share": s,
         "descends_per_10_windows": d10, "T_calibrated": T,
         "audit_final": last, "verdict": verdict})
    print(json.dumps({"dive_share": s, "descends_per_10_windows": d10,
                      "T": T, "verdict": verdict, "audit": last},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
