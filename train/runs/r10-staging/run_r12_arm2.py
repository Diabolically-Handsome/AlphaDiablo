"""R12 臂二重发(修正案二:魔鬼教练→脚本教练 SMART level-margin-1)。

主席批文 2026-08-30:「就这样 开干!」。台账续写 r12_ledger.jsonl。
臂一考卷有效保留;本驱动只跑臂二训练 + 四场考试。
"""
import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
RUNS = ROOT / "train" / "runs"
STAGING = RUNS / "r10-staging"
LEDGER = STAGING / "r12_ledger.jsonl"
PY = str(ROOT / ".venv" / "bin" / "python")

CERT_WORKER = RUNS / "r9-reeducation" / "staging" / "worker.zip"
M29 = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
DEVIL = RUNS / "r10-econ-mgr" / "policy.npz"
ARM = "r12-arm2-smart"
POOLS = {"a": "2114000-2114127", "b": "2115000-2115127"}


def log(event):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("[ledger]", json.dumps(event, ensure_ascii=False), flush=True)


def run(cmd, log_name, timeout):
    log({"event": "cmd_start", "log": log_name, "cmd": [str(c) for c in cmd]})
    with open(STAGING / log_name, "w") as fh:
        rc = subprocess.run([str(c) for c in cmd], stdout=fh,
                            stderr=subprocess.STDOUT, timeout=timeout,
                            cwd=ROOT).returncode
    log({"event": "cmd_done", "log": log_name, "rc": rc})
    return rc


def agg_of(tag):
    data = json.load(open(RUNS / "eval-assembled" / f"{tag}.json"))
    rows = json.dumps(data["rows"], sort_keys=True).encode()
    a = data["agg"]
    return {"sha16": hashlib.sha256(rows).hexdigest()[:16],
            "ret": round(a["ret_mean"], 2), "l3": a["l3"], "died": a["died"],
            "kills": round(a["kills_mean"], 1)}


def main():
    log({"event": "arm2_relaunch", "amendment": "修正案二:教练=脚本 SMART(level-margin-1)",
         "authorization": "主席 2026-08-30:「就这样 开干!」"})
    rc = run([PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
              "--gamma", "1.0", "--max-steps", "3000",
              "--num-envs", "4", "--n-steps", "512",
              "--total-steps", "266240", "--seed", "22",
              "--device", "cpu", "--run-name", ARM,
              "--artifact-scope", "candidate",
              "--resume-from", str(CERT_WORKER),
              "--allow-environment-restart-resume", "--allow-manager-change",
              "--gradient-clip-mode", "separate-root-context-critic-v2",
              "--manager-heuristic", "level-margin-1",
              "--worker-action14-logit-bonus", "2.5",
              "--ckpt-every-steps", "63488",
              "--lr", "0.0001", "--ent-coef", "0.005",
              "--target-kl", "0.01", "--no-drink-sovereignty",
              "--worker-policy-observation-view", "dual-v4-asymmetric-v3",
              "--worker-fast-forward-reward-credit", "terminal-death-only",
              "--worker-additional-terminal-death-cost", "0.0",
              "--reward-economy", "v2"],
             f"train-{ARM}.log", 14400)
    if rc != 0:
        log({"event": "OPERATIONAL_FAILURE", "arm": ARM, "rc": rc})
        raise SystemExit(1)
    new_worker = RUNS / ARM / "model_candidate.zip"
    if not new_worker.is_file():
        log({"event": "STOP", "arm": ARM, "reason": "缺 model_candidate.zip"})
        raise SystemExit(1)
    wsha = hashlib.sha256(new_worker.read_bytes()).hexdigest()
    log({"event": "arm_done", "arm": ARM, "worker_sha16": wsha[:16]})
    for mgr_name, mgr, view in (("m29", M29, "legacy-v3"),
                                 ("devil", DEVIL, "raw-v4")):
        for pool, seeds in POOLS.items():
            tag = f"{ARM}-x{mgr_name}-{pool}"
            rc = run([PY, "train/eval_assembled.py",
                      "--worker", str(new_worker),
                      "--manager-npz", str(mgr),
                      "--manager-policy-observation-view", view,
                      "--seeds", seeds, "--tag", tag,
                      "--reward-economy", "v2"],
                     f"eval-{tag}.log", 7200)
            if rc != 0:
                log({"event": "STOP", "exam": tag, "rc": rc})
                raise SystemExit(1)
            log({"event": "exam", "tag": tag, **agg_of(tag)})
    log({"event": "arm2_stage_complete", "next": "全案配对分析 + 四门判决"})


if __name__ == "__main__":
    main()
