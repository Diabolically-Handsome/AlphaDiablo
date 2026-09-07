"""R17 锚重铸(R17.1 最终字节:协议 bundle fb7d651e…,新桥 a7e47ebb…;考卷旗与 R16 完全相同,新文件 r17-anchor-*)。原 R16 文档:R16 新法锚重铸(预注册 §三:R16 = 新世界,旧法锚逐位重现 + 新法锚重铸)。

认证工人(r9 worker.zip)在 R16 考卷配置下六卷各 128 种子:
  xdevil-a/b   魔鬼经理 raw-v4,farm-dive-v1 双注册
  xm29-a/b     M29 legacy-v3,farm-only
  xm29full-a/b M29 legacy-v3,farm-dive-v1 双注册
R16 考卷配置(全部六卷同):--worker-decoding sample(逐局定种)+ 环境旗
--explore-global-hunt --progress-far-tiles 3 --farm-scene-cap 3600
--reset-layer-clock-on-window --reward-economy v4;协议 max_steps 3000 不动。
档案 r16-anchor-<exam>.json;与旧世界数字不可直接比较(台账立碑)。
用法:run_r16_anchors.py [only-exam-name ...]
"""
import hashlib
import json
import pathlib
import subprocess
import sys
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
POOLS = {"a": "2114000-2114127", "b": "2115000-2115127"}
PREFIX = "r17-anchor"
R16_EXAM_FLAGS = ["--worker-decoding", "sample",
                  "--explore-global-hunt",
                  "--farm-scene-cap", "3600", "--reset-layer-clock-on-window",
                  "--reward-economy", "v4"]
EXAMS = (
    ("xdevil-a", DEVIL, "raw-v4", "a", "farm-dive-v1"),
    ("xdevil-b", DEVIL, "raw-v4", "b", "farm-dive-v1"),
    ("xm29-a", M29, "legacy-v3", "a", "farm-only"),
    ("xm29-b", M29, "legacy-v3", "b", "farm-only"),
    ("xm29full-a", M29, "legacy-v3", "a", "farm-dive-v1"),
    ("xm29full-b", M29, "legacy-v3", "b", "farm-dive-v1"),
)


def log(event):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("[ledger]", json.dumps(event, ensure_ascii=False), flush=True)


def agg_of(tag):
    data = json.load(open(EV / f"{tag}.json"))
    rows = json.dumps(data["rows"], sort_keys=True).encode()
    a = data["agg"]
    return {"sha16": hashlib.sha256(rows).hexdigest()[:16],
            "ret": round(a["ret_mean"], 2), "l3": a["l3"], "died": a["died"],
            "kills": round(a["kills_mean"], 1),
            "r16_environment": data["meta"]["protocol"].get("r16_environment"),
            "worker_decoding": data["meta"]["protocol"].get("worker_decoding")}


def main():
    only = set(sys.argv[1:])
    log({"event": "R17_ANCHOR_CAST_START", "prefix": PREFIX,
         "worker": str(CERT_WORKER), "exam_flags": R16_EXAM_FLAGS,
         "law": "R16 新世界:新法锚与旧世界数字不可直接比较"})
    for name, mgr, view, pool, registration in EXAMS:
        if only and name not in only:
            continue
        tag = f"{PREFIX}-{name}"
        out = EV / f"{tag}.json"
        if out.exists():
            log({"event": "anchor_skip_exists", "tag": tag})
            continue
        cmd = [PY, "train/eval_assembled.py",
               "--worker", str(CERT_WORKER),
               "--manager-npz", str(mgr),
               "--manager-policy-observation-view", view,
               "--seeds", POOLS[pool], "--tag", tag] + R16_EXAM_FLAGS
        if registration != "farm-only":
            cmd += ["--worker-window-registration", registration]
        log({"event": "cmd_start", "tag": tag, "cmd": cmd[1:]})
        with open(STAGING / f"eval-{tag}.log", "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=ROOT, timeout=10800).returncode
        if rc != 0:
            log({"event": "R16_ANCHOR_FAILURE", "tag": tag, "rc": rc})
            raise SystemExit(1)
        log({"event": "r16_anchor", "tag": tag, "registration": registration,
             **agg_of(tag)})
    log({"event": "R17_ANCHOR_CAST_DONE", "prefix": PREFIX})


if __name__ == "__main__":
    main()
