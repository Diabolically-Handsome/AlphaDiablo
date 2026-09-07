"""R13 臂 α 发射驱动:live-DIVE 教室(SMART 教练 + a11 冷启动杠杆)。

主席批文 2026-08-30:「好 步数不够的话可以继续加 我在想步数可能也是
一个主要的因素 别的按您说得来」。台账 r13_ledger.jssonl 之误照 R12 惯例
写 r13_ledger.jsonl。流程:训练 → 门D探针(fail-closed)→ 六卷考试
(魔鬼×2 双注册 / M29×2 同锚协议 / M29×2 全注册信息场)→ 呈对判。

T=432,128 冻结依据:G0-2 实测 dive_share=0.383,
T=round2048(266240/(1-0.383));软顶 393,216 已越,按主席步数特批
(v0.2 修订第2条)记异常继续。
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
ARM = "r14-arm-b-wage"
TOTAL_STEPS = 401_408
POOLS = {"a": "2114000-2114127", "b": "2115000-2115127"}
DEVIL_ANCHOR_SHA = {"a": "00f4672db9c3ac69", "b": "2f21b7fd9cb38117"}


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
    data = json.load(open(EV / f"{tag}.json"))
    rows = json.dumps(data["rows"], sort_keys=True).encode()
    a = data["agg"]
    return {"sha16": hashlib.sha256(rows).hexdigest()[:16],
            "ret": round(a["ret_mean"], 2), "l3": a["l3"], "died": a["died"],
            "kills": round(a["kills_mean"], 1)}


def main():
    log({"event": "arm_b_launch", "arm": ARM, "total_steps": TOTAL_STEPS,
         "step_law": "T=round2048(266240/(1-0.337)),G0-4 定标;主席步数特批continues",
         "lever": "a11-bonus=2.0 + descend-bonus-fraction=0.25 乙案修宪(G0-4 验收 2.31/10,主席批「先做乙」)"})
    rc = run([PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
              "--gamma", "1.0", "--max-steps", "3000",
              "--num-envs", "4", "--n-steps", "512",
              "--total-steps", str(TOTAL_STEPS), "--seed", "22",
              "--device", "cpu", "--run-name", ARM,
              "--artifact-scope", "candidate",
              "--resume-from", str(CERT_WORKER),
              "--allow-environment-restart-resume", "--allow-manager-change",
              "--gradient-clip-mode", "separate-root-context-critic-v2",
              "--manager-heuristic", "level-margin-1",
              "--worker-action14-logit-bonus", "2.5",
              "--worker-dive-action11-logit-bonus", "2.0",
              "--worker-descend-bonus-fraction", "0.25",
              "--ckpt-every-steps", "63488",
              "--sentinel-every", "63488",
              "--lr", "0.0001", "--ent-coef", "0.005",
              "--target-kl", "0.01", "--no-drink-sovereignty",
              "--worker-policy-observation-view", "dual-v4-asymmetric-v3",
              "--worker-fast-forward-reward-credit", "terminal-death-only",
              "--worker-additional-terminal-death-cost", "0.0",
              "--reward-economy", "v2",
              "--worker-learning-window-scope", "farm-dive-v1"],
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

    # 门 D 探针(fail-closed):D1 学习发生 / D2 触及 DIVE 决策面
    rc = run([PY, "train/runs/r10-staging/probe_r13_identity.py",
              str(CERT_WORKER), str(new_worker),
              str(STAGING / "r13-gateD.json")],
             "probe-r13-gateD.log", 3600)
    gate_d = json.load(open(STAGING / "r13-gateD.json"))
    log({"event": "gateD", **gate_d})
    if rc != 0:
        log({"event": "STOP", "reason": "门D探针未破同一性,禁止烧考试场"})
        raise SystemExit(1)

    exams = (
        ("xdevil-a", DEVIL, "raw-v4", "a", "farm-dive-v1"),
        ("xdevil-b", DEVIL, "raw-v4", "b", "farm-dive-v1"),
        ("xm29-a", M29, "legacy-v3", "a", "farm-only"),
        ("xm29-b", M29, "legacy-v3", "b", "farm-only"),
        ("xm29full-a", M29, "legacy-v3", "a", "farm-dive-v1"),
        ("xm29full-b", M29, "legacy-v3", "b", "farm-dive-v1"),
    )
    for name, mgr, view, pool, registration in exams:
        tag = f"{ARM}-{name}"
        cmd = [PY, "train/eval_assembled.py",
               "--worker", str(new_worker),
               "--manager-npz", str(mgr),
               "--manager-policy-observation-view", view,
               "--seeds", POOLS[pool], "--tag", tag,
               "--reward-economy", "v2"]
        if registration != "farm-only":
            cmd += ["--worker-window-registration", registration]
        rc = run(cmd, f"eval-{tag}.log", 7200)
        if rc != 0:
            log({"event": "STOP", "exam": tag, "rc": rc})
            raise SystemExit(1)
        result = agg_of(tag)
        log({"event": "exam", "tag": tag, "registration": registration,
             **result})
        if name.startswith("xdevil"):
            # 开卷强制断言:R12 死因的 fail-closed 反转——魔鬼卷若与
            # 脚本代驾锚逐位一致,说明注册接线失败,整案中止排查。
            if result["sha16"] == DEVIL_ANCHOR_SHA[pool]:
                log({"event": "STOP", "exam": tag,
                     "reason": "魔鬼卷与锚逐位一致:DIVE 注册接线失败"})
                raise SystemExit(1)
    log({"event": "arm_b_stage_complete", "next": "四门配对对判 + 呈主席终裁"})


if __name__ == "__main__":
    main()
