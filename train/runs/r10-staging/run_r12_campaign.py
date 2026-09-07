"""R12 战役驱动:工人在位再教育·双臂教练学(2026-08-30)。

预注册:r12-PREREG-FROZEN-20260830.md。主席批文:「双臂对比一下吧
这样有对照组才能说明一点 234没问题 就按您说的办」。

阶段:臂一(M29 常规教练)训练→四场考试;臂二(裸奔魔鬼教练)同;
判决数据落盘,终判由分析+主席终裁。基线四卷已存在(2_114/115 v2 卷)。
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

ARMS = (
    ("r12-arm1-m29", M29, "legacy-v3"),
    ("r12-arm2-devil", DEVIL, "raw-v4"),
)
POOLS = {"a": "2114000-2114127", "b": "2115000-2115127"}
TOTAL_STEPS = 266240
TRAIN_FUSE = 14400   # 4h 防僵尸(实测需求 ~42min @107sps)
EVAL_FUSE = 7200


def log(event: dict):
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
    STAGING.mkdir(parents=True, exist_ok=True)
    log({"event": "campaign_start", "campaign": "r12-worker-reeducation",
         "authorization": "主席 2026-08-30:「双臂对比…234没问题 就按您说的办」"})
    for arm, coach, coach_view in ARMS:
        rc = run([PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
                  "--gamma", "1.0", "--max-steps", "3000",
                  "--num-envs", "4", "--n-steps", "512",
                  "--total-steps", str(TOTAL_STEPS), "--seed", "22",
                  "--device", "cpu", "--run-name", arm,
                  "--artifact-scope", "candidate",
                  "--resume-from", str(CERT_WORKER),
                  "--allow-environment-restart-resume",
                  "--allow-manager-change",
                  "--gradient-clip-mode", "separate-root-context-critic-v2",
                  "--manager-npz", str(coach),
                  "--worker-action14-logit-bonus", "2.5",
                  "--ckpt-every-steps", "63488",
                  "--lr", "0.0001", "--ent-coef", "0.005",
                  "--target-kl", "0.01", "--no-drink-sovereignty",
                  "--worker-policy-observation-view", "dual-v4-asymmetric-v3",
                  "--worker-fast-forward-reward-credit", "terminal-death-only",
                  "--worker-additional-terminal-death-cost", "0.0",
                  "--reward-economy", "v2"],
                 f"train-{arm}.log", TRAIN_FUSE)
        if rc != 0:
            log({"event": "OPERATIONAL_FAILURE", "arm": arm, "rc": rc})
            raise SystemExit(1)
        new_worker = RUNS / arm / "model_candidate.zip"
        if not new_worker.is_file():
            log({"event": "STOP", "arm": arm, "reason": "缺 model_candidate.zip"})
            raise SystemExit(1)
        wsha = hashlib.sha256(new_worker.read_bytes()).hexdigest()
        log({"event": "arm_done", "arm": arm, "worker_sha16": wsha[:16]})
        # 考试:两位经理 × 两池,与既有基线同池同种子配对
        for mgr_name, mgr, view in (("m29", M29, "legacy-v3"),
                                     ("devil", DEVIL, "raw-v4")):
            for pool, seeds in POOLS.items():
                tag = f"{arm}-x{mgr_name}-{pool}"
                rc = run([PY, "train/eval_assembled.py",
                          "--worker", str(new_worker),
                          "--manager-npz", str(mgr),
                          "--manager-policy-observation-view", view,
                          "--seeds", seeds, "--tag", tag,
                          "--reward-economy", "v2"],
                         f"eval-{tag}.log", EVAL_FUSE)
                if rc != 0:
                    log({"event": "STOP", "exam": tag, "rc": rc})
                    raise SystemExit(1)
                log({"event": "exam", "tag": tag, **agg_of(tag)})
    log({"event": "campaign_stage_complete",
         "next": "配对分析 + 四门判决 + 主席终裁"})


if __name__ == "__main__":
    main()
