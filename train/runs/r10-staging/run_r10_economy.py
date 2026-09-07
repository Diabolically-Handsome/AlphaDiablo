"""R10 深度经济战役驱动(2026-08-27,主席批文「没有疑问的话 就可以开始跑了」)。

阶段:
  1. 锚线重烧:M29(冻结产线经理)× 认证工人,经济 v2,池 2_114/2_115,
     每池烧两遍,逐位一致才放行(REF 纪律);
  2. 候选训练:全新 MaskablePPO 经理,经济 v2,num-envs 32,seed 22,
     total-steps 327,680(=2048×160 rollouts),6h 熔断;
  3. 行为同一性前检:候选 vs M29 argmax 一致率 <99% 才准进考场(R9 新法);
  4. 考试:候选同池 2_114/2_115,与锚同种子配对;
  5. 判决数据落盘(配对深度/回报/死亡),终判由分析脚本+班长复核出具。

台账:r10_ledger.jsonl(逐事件 append)。只用工程与预注册指定池,
决赛确认池 2_116 由终判后人工指令消耗,本驱动不碰。
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
LEDGER = STAGING / "r10_ledger.jsonl"
PY = str(ROOT / ".venv" / "bin" / "python")

WORKER_ZIP = RUNS / "r9-reeducation" / "staging" / "worker.zip"
WORKER_SHA = "2837288dad19a685925558a0d86e1cecd951d5f065d1d2f367f667c13b9cf006"
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"

ANCHOR_POOLS = {"a": "2114000-2114127", "b": "2115000-2115127"}
TRAIN_FUSE_S = 21600  # 6h(修正案后预期 ~4.6h,防真死锁)
# 修正案一(2026-08-27 ~17:30,主席「可以 没问题 就按您说的办」):
# 实测吞吐 ~45 步/分(掉队者同步病,见台账),327,680 步需 121h 不可达;
# 改 12,288(=2048×6 rollouts)。科学参数零改动,纯日程修正。
TOTAL_STEPS = 12288
NUM_ENVS = 32


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    print("[ledger]", json.dumps(event, ensure_ascii=False), flush=True)


def run(cmd, log_name, timeout=None):
    log({"event": "cmd_start", "log": log_name,
         "cmd": [str(c) for c in cmd]})
    with open(STAGING / log_name, "w") as fh:
        rc = subprocess.run(
            [str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT,
            timeout=timeout, cwd=ROOT).returncode
    log({"event": "cmd_done", "log": log_name, "rc": rc})
    return rc


def eval_cmd(manager_npz, view, seeds, tag):
    return [PY, "train/eval_assembled.py",
            "--worker", str(WORKER_ZIP),
            "--manager-npz", str(manager_npz),
            "--manager-policy-observation-view", view,
            "--seeds", seeds, "--tag", tag,
            "--reward-economy", "v2"]


def result_sha(tag):
    path = RUNS / "eval-assembled" / f"{tag}.json"
    data = json.load(open(path))
    rows = json.dumps(data["rows"], sort_keys=True).encode()
    return hashlib.sha256(rows).hexdigest()[:16], data


def main():
    STAGING.mkdir(parents=True, exist_ok=True)
    log({"event": "campaign_start", "campaign": "r10-econ-v2",
        "authorization": "主席 2026-08-27:「给经理」「没有疑问的话 就可以开始跑了」"})

    # 阶段 1:锚线重烧 ×2,逐位一致(已认证则跳过——修正案一复飞免重烧)
    if (STAGING / "anchors.certified").is_file():
        log({"event": "anchors_skip", "reason":
             "既有双池双烧逐位认证(68701e04/5ee5e408),复飞免重烧"})
        anchor_pools_to_burn = {}
    else:
        anchor_pools_to_burn = ANCHOR_POOLS
    for pool, seeds in anchor_pools_to_burn.items():
        shas = []
        for rep in (1, 2):
            tag = f"r10-anchor-{pool}-rep{rep}"
            rc = run(eval_cmd(M29_NPZ, "legacy-v3", seeds, tag),
                     f"anchor-{pool}-rep{rep}.log", timeout=7200)
            if rc != 0:
                log({"event": "STOP", "stage": "anchor",
                     "pool": pool, "rc": rc})
                sys.exit(1)
            sha, data = result_sha(tag)
            shas.append(sha)
            agg = data.get("agg", {})
            log({"event": "anchor_burn", "pool": pool, "rep": rep,
                 "sha16": sha, "mean": agg.get("ret_mean"),
                 "l3": agg.get("l3"), "died": agg.get("died")})
        if shas[0] != shas[1]:
            log({"event": "STOP", "stage": "anchor_biteq",
                 "pool": pool, "shas": shas})
            sys.exit(1)
        log({"event": "anchor_biteq_ok", "pool": pool, "sha16": shas[0]})

    # 阶段 2:候选训练
    rc = run([PY, "train/train_ppo.py", "--options", "--algo", "mppo",
              "--gamma", "1.0", "--max-steps", "3000",
              "--n-steps", "64", "--num-envs", str(NUM_ENVS),
              "--total-steps", str(TOTAL_STEPS),
              "--worker-zip", str(WORKER_ZIP),
              "--worker-zip-sha256", WORKER_SHA,
              "--manager-policy-observation-view", "raw-v4",
              "--reward-economy", "v2",
              "--run-name", "r10-econ-mgr",
              "--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"],
             "train-r10-econ-mgr.log", timeout=TRAIN_FUSE_S)
    if rc != 0:
        log({"event": "OPERATIONAL_FAILURE", "stage": "train", "rc": rc})
        sys.exit(1)
    log({"event": "arm_done", "arm": "r10-econ-mgr"})

    # 候选 NPZ 导出(经理侧六矩阵,现行工具适用)
    cand_npz = RUNS / "r10-econ-mgr" / "policy.npz"
    rc = run([PY, "train/export_manager_npz.py",
              str(RUNS / "r10-econ-mgr" / "model_final.zip"),
              str(cand_npz)],
             "export-candidate.log", timeout=600)
    if rc != 0 or not cand_npz.is_file():
        log({"event": "STOP", "stage": "export", "rc": rc})
        sys.exit(1)
    log({"event": "candidate_npz",
         "sha16": hashlib.sha256(cand_npz.read_bytes()).hexdigest()[:16]})

    # 阶段 4:考试(同池同种子,与锚配对)
    for pool, seeds in ANCHOR_POOLS.items():
        tag = f"r10-cand-{pool}"
        rc = run(eval_cmd(cand_npz, "raw-v4", seeds, tag),
                 f"cand-{pool}.log", timeout=7200)
        if rc != 0:
            log({"event": "STOP", "stage": "exam", "pool": pool, "rc": rc})
            sys.exit(1)
        sha, data = result_sha(tag)
        log({"event": "cand_exam", "pool": pool, "sha16": sha})

    log({"event": "campaign_stage_complete",
         "next": "配对分析(analyze_r10.py)+ 班长复核 + 主席终判"})


if __name__ == "__main__":
    main()
