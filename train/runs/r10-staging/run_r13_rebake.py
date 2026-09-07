"""R13 锚重烤即认证(预注册 §4.1)。

在位修法旋转了协议 bundle sha,四份 2_114/2_115 冻结锚重烤:
默认旗(farm-only 注册)下同段种子复跑,rows 必须与冻结锚逐位相等——
等式兼任「默认关=逐位不变」的终极认证。任一失配 fail-closed 中止。
零处女种子消耗(已焚段复跑)。台账 r13_ledger.jsonl。
"""
import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path.home() / "AlphaDiablo" / "diablogym"
RUNS = ROOT / "train" / "runs"
STAGING = RUNS / "r10-staging"
LEDGER = STAGING / "r13_ledger.jsonl"
EV = RUNS / "eval-assembled"
PY = str(ROOT / ".venv" / "bin" / "python")

CERT_WORKER = RUNS / "r9-reeducation" / "staging" / "worker.zip"
M29 = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
DEVIL = RUNS / "r10-econ-mgr" / "policy.npz"

REBAKES = (
    ("r13-rebake-devil-a", DEVIL, "raw-v4", "2114000-2114127", "r10-cand-a"),
    ("r13-rebake-devil-b", DEVIL, "raw-v4", "2115000-2115127", "r10-cand-b"),
    ("r13-rebake-m29-a", M29, "legacy-v3", "2114000-2114127",
     "r10-anchor-a-rep1"),
    ("r13-rebake-m29-b", M29, "legacy-v3", "2115000-2115127",
     "r10-anchor-b-rep1"),
)


def log(event):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("[ledger]", json.dumps(event, ensure_ascii=False), flush=True)


def rows_sha16(tag):
    data = json.load(open(EV / f"{tag}.json"))
    return hashlib.sha256(
        json.dumps(data["rows"], sort_keys=True).encode()).hexdigest()[:16]


def main():
    log({"event": "rebake_start",
         "reason": "在位修法旋转协议 bundle sha(worker_env/options_env/"
                   "train_ppo/eval_assembled/leashed_ppo 五文件 R13 修法)",
         "law": "预注册 §4.1 重烤即认证:rows 逐位等式 = 默认关零漂移终极执法"})
    for tag, mgr, view, seeds, anchor in REBAKES:
        out = EV / f"{tag}.json"
        if out.exists():
            out.unlink()
        rc = subprocess.run(
            [PY, "train/eval_assembled.py",
             "--worker", str(CERT_WORKER),
             "--manager-npz", str(mgr),
             "--manager-policy-observation-view", view,
             "--seeds", seeds, "--tag", tag,
             "--reward-economy", "v2"],
            cwd=ROOT, timeout=7200,
            stdout=open(STAGING / f"eval-{tag}.log", "w"),
            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log({"event": "REBAKE_FAILURE", "tag": tag, "rc": rc})
            raise SystemExit(1)
        new_rows = json.load(open(out))["rows"]
        old_rows = json.load(open(EV / f"{anchor}.json"))["rows"]
        old_by_seed = {r["seed"]: r for r in old_rows}
        mismatches = [r["seed"] for r in new_rows
                      if r != old_by_seed.get(r["seed"])]
        sha = rows_sha16(tag)
        anchor_sha = rows_sha16(anchor)
        verdict = "PASS" if (not mismatches and sha == anchor_sha) else "FAIL"
        log({"event": "rebake", "tag": tag, "anchor": anchor,
             "rows_sha16": sha, "anchor_sha16": anchor_sha,
             "mismatch_seeds": mismatches[:8],
             "n_mismatch": len(mismatches), "verdict": verdict})
        if verdict == "FAIL":
            log({"event": "STOP",
                 "reason": "重烤等式破裂:默认路径存在漂移,fail-closed 排查"})
            raise SystemExit(1)
    log({"event": "rebake_complete",
         "verdict": "四锚逐位重现,默认关=逐位不变获终极认证,考场解锁"})


if __name__ == "__main__":
    main()
