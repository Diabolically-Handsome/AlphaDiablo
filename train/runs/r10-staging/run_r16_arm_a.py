"""R16 修宪臂 α 发射驱动:新法全开(r16-PREREG §四配方)。

流程:训练(T 由 G0 定标,argv[1])→ 门D探针(fail-closed)→ 六卷 R16 考卷
(与 run_r16_anchors.py 同配置)→ 四门对判(analyze_arm_gates.py <arm> r16-anchor)
→ 部署探针 v3(R16 形态:readiness-v3 / 主权开 / v4 / hunt / cap 3600 /
clock reset;max_steps 6000;sample;16 种子)臂 vs 认证工人(新法部署锚)。
用法:run_r16_arm_a.py <total_steps> [hp_price=0.1] [potion_bonus=2.0]
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
ARM = "r16-arm-a-constitution"
POOLS = {"a": "2114000-2114127", "b": "2115000-2115127"}
ANCHOR_PREFIX = "r16-anchor"
R16_EXAM_FLAGS = ["--worker-decoding", "sample",
                  "--explore-global-hunt",
                  "--farm-scene-cap", "3600", "--reset-layer-clock-on-window",
                  "--reward-economy", "v4"]
PROBE_FORM = json.dumps({
    "explore_global_hunt": True,
    "farm_scene_cap": 3600, "reset_layer_clock_on_window": True,
    "reward_economy": "v4", "drink_sovereignty": True,
    "manager": "readiness-v3"})
PROBE_SEEDS = "2114000-2114015"
PROBE_STEPS = "6000"


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
    total_steps = int(sys.argv[1])
    hp_price = sys.argv[2] if len(sys.argv) > 2 else "0.1"
    potion_bonus = sys.argv[3] if len(sys.argv) > 3 else "2.0"
    log({"event": "LAUNCH_R16_ARM_A", "arm": ARM, "total_steps": total_steps,
         "law": "R16 修宪:尺子 v0.2 + 门(清场比)+ 考试法(采样)+ 记账(hp/药/超时零记/v4)"
                " + 视野(hunt/cap 3600/clock reset)+ 局长 6000 + 主权开",
         "hp_loss_price": hp_price, "potion_pickup_bonus": potion_bonus})
    rc = run([PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
              "--gamma", "1.0", "--max-steps", "6000",
              "--num-envs", "4", "--n-steps", "512",
              "--total-steps", str(total_steps), "--seed", "22",
              "--device", "cpu", "--run-name", ARM,
              "--artifact-scope", "candidate",
              "--resume-from", str(CERT_WORKER),
              "--allow-environment-restart-resume", "--allow-manager-change",
              "--gradient-clip-mode", "separate-root-context-critic-v2",
              "--manager-heuristic", "readiness-v3",
              "--worker-action14-logit-bonus", "2.5",
              "--worker-dive-action11-logit-bonus", "2.0",
              "--worker-potion-action13-logit-bonus", "2.0",
              "--worker-descend-escrow-fraction", "0.5",
              "--worker-descend-escrow-power", "1.6",
              "--worker-descend-escrow-readiness-gate",
              "--worker-hp-loss-price", hp_price,
              "--worker-potion-pickup-bonus", potion_bonus,
              "--worker-no-progress-timeout-credit", "zero",
              "--explore-global-hunt",
              "--farm-scene-cap", "3600",
              "--reset-layer-clock-on-window",
              "--ckpt-every-steps", "63488",
              "--sentinel-every", "63488",
              "--lr", "0.0001", "--ent-coef", "0.005",
              "--target-kl", "0.01", "--drink-sovereignty",
              "--worker-policy-observation-view", "dual-v4-asymmetric-v3",
              "--worker-fast-forward-reward-credit", "terminal-death-only",
              "--worker-additional-terminal-death-cost", "0.0",
              "--reward-economy", "v4",
              "--worker-learning-window-scope", "farm-dive-v1"],
             f"train-{ARM}.log", 21600)
    if rc != 0:
        log({"event": "OPERATIONAL_FAILURE", "arm": ARM, "rc": rc})
        raise SystemExit(1)
    new_worker = RUNS / ARM / "model_candidate.zip"
    if not new_worker.is_file():
        log({"event": "STOP", "arm": ARM, "reason": "缺 model_candidate.zip"})
        raise SystemExit(1)
    wsha = hashlib.sha256(new_worker.read_bytes()).hexdigest()
    audit = RUNS / ARM / "r13_dive_audit.jsonl"
    last = [json.loads(l) for l in open(audit) if l.strip()][-1]
    log({"event": "arm_done", "arm": ARM, "worker_sha16": wsha[:16],
         "audit_final": last})

    rc = run([PY, "train/runs/r10-staging/probe_r13_identity.py",
              str(CERT_WORKER), str(new_worker),
              str(STAGING / "r16-gateD.json")],
             "probe-r16-gateD.log", 3600)
    gate_d = json.load(open(STAGING / "r16-gateD.json"))
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
               "--seeds", POOLS[pool], "--tag", tag] + R16_EXAM_FLAGS
        if registration != "farm-only":
            cmd += ["--worker-window-registration", registration]
        rc = run(cmd, f"eval-{tag}.log", 10800)
        if rc != 0:
            log({"event": "STOP", "exam": tag, "rc": rc})
            raise SystemExit(1)
        result = agg_of(tag)
        anchor = agg_of(f"{ANCHOR_PREFIX}-{name}")
        log({"event": "exam", "tag": tag, "registration": registration,
             **result, "anchor": anchor})
        if result["sha16"] == anchor["sha16"]:
            log({"event": "STOP", "exam": tag,
                 "reason": "臂卷与新法锚逐位一致:工人未变或接线失败"})
            raise SystemExit(1)

    gates = subprocess.run(
        [PY, "train/runs/r10-staging/analyze_arm_gates.py", ARM,
         ANCHOR_PREFIX], cwd=ROOT, capture_output=True, text=True)
    log({"event": "four_gates", "arm": ARM, "anchor_prefix": ANCHOR_PREFIX,
         "stdout": gates.stdout[-4000:], "rc": gates.returncode})

    # 部署探针 v3(R16 形态):臂 vs 认证工人(新法部署锚)
    for label, zip_path in (("arm", new_worker), ("anchor", CERT_WORKER)):
        out = STAGING / f"r16-deploy-{label}.json"
        rc = run([PY, "train/runs/r10-staging/probe_r15_deployment.py",
                  str(zip_path), PROBE_SEEDS, str(out), PROBE_STEPS,
                  "sample", PROBE_FORM], f"probe-r16-deploy-{label}.log",
                 7200)
        if rc != 0:
            log({"event": "STOP", "probe": label, "rc": rc})
            raise SystemExit(1)
        log({"event": "deploy_probe", "who": label,
             **json.load(open(out))["agg"]})
    log({"event": "R16_ARM_A_PIPELINE_DONE", "arm": ARM})


if __name__ == "__main__":
    main()
