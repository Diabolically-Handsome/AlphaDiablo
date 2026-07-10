"""v25-ALT「换届选举」驱动(docs/PREREG-v25.md 条款唯一执行者)。

流程:G-A0(仪器回归)→ M-warm 权重包 → 两臂串行训练(v22-H 原配方)→
G-A0m(逐臂 npz parity)→ 提前放弃闸(16 种子双 <75)→ 两臂满 32 →
胜者 → 配对发射判据 → GOLDEN_AUTHORIZED / VERDICT_PATH。
金牌不在此发射(值夜者手启,单臂一次)。账本:train/runs/v25/gate_ledger.jsonl。
用法:.venv/bin/python train/run_v25_election.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V25 = RUNS / "v25"
V25.mkdir(parents=True, exist_ok=True)
LEDGER = V25 / "gate_ledger.jsonl"

LEG7_ZIP = str(ROOT / "train" / "models" / "v24-worker-leg7" / "model")   # SB3 路(与存档同)
LEG7_NPZ = str(ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz")
V22H_ZIP = ROOT / "train" / "models" / "v22-h-manager" / "model_final.zip"
WARM_SD = ROOT / "train" / "models" / "v22-h-manager" / "policy_full_sd.pt"
ARCHIVE = RUNS / "eval-assembled" / "v24-G3-leg7.json"   # 参考行 92.0(逐种子)

ABANDON = 75.0        # 提前放弃闸(双臂 16 种子均 < 此值)
PAIRED_DIFF = 4.0     # 发射线:配对均差 ≥ +4
PAIRED_WINS = 18      # 且配对赢 ≥ 18/32
DEATHS_MAX = 6
FLOOR_REPRO = 85.0    # 胜者 < 85 → "重训未复现,命题未考"
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}

ARMS = {
    "v25-mfresh": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"],
    "v25-mwarm": ["--ent-coef", "0.005", "--lr", "1e-4", "--seed", "23",
                  "--bc-init", str(WARM_SD)],
}


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def run(cmd, logfile) -> int:
    with open(V25 / logfile, "w") as lf:
        return subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode


def exam(worker, tag, seeds, manager_npz=None):
    cmd = [PY, "train/eval_assembled.py", "--worker", worker, "--seeds", seeds, "--tag", tag]
    if manager_npz:
        cmd += ["--manager-npz", manager_npz]
    if run(cmd, f"exam-{tag}.log") != 0:
        return None
    d = json.loads((RUNS / "eval-assembled" / f"{tag}.json").read_text())
    d["agg"]["_sha"] = sha16(RUNS / "eval-assembled" / f"{tag}.json")
    return d


def dive_per_ep(rows) -> float:
    return sum(r["mode_seq"].count("D") for r in rows) / max(1, len(rows))


def main():
    log({"event": "start", "prereg": "docs/PREREG-v25.md v2",
         "paired_line": PAIRED_DIFF, "wins_line": PAIRED_WINS})

    # ---- G-A0:仪器回归(npz 工人 + 默认经理 ≡ v24-G3-leg7 存档,32/32)----
    ref = json.loads(ARCHIVE.read_text())
    ga0 = exam(LEG7_NPZ, "v25-GA0", "7000-7031")
    if ga0 is None:
        log({"event": "STOP", "why": "G-A0 考试进程失败"})
        return
    ref_rows = {r["seed"]: r for r in ref["rows"]}
    bad = [r["seed"] for r in ga0["rows"]
           if (abs(r["ret"] - ref_rows[r["seed"]]["ret"]) > 0.01
               or r["died"] != ref_rows[r["seed"]]["died"]
               or r["mode_seq"] != ref_rows[r["seed"]]["mode_seq"])]
    log({"event": "g_a0", "mismatch_seeds": bad, "n_ok": 32 - len(bad)})
    if bad:
        log({"event": "STOP", "why": "G-A0 位级回归失配——按预注册回退条款人工重锚"})
        return

    # ---- M-warm 权重包 ----
    if run([PY, "train/export_manager_sd.py"], "export-warm-sd.log") != 0 or not WARM_SD.exists():
        log({"event": "STOP", "why": "M-warm 权重包导出失败"})
        return
    log({"event": "warm_sd", "sha": sha16(WARM_SD)})

    # ---- 两臂串行训练(v22-H status.json 原配方)----
    npz = {}
    for name, extra in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo", "--gamma", "1.0",
               "--max-steps", "3000", "--n-steps", "64", "--num-envs", "4",
               "--total-steps", "40000", "--worker-npz", LEG7_NPZ,
               "--run-name", name] + extra
        log({"event": "arm_start", "arm": name, "cmd_extra": extra})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log")
        sp = RUNS / name / "status.json"
        steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        log({"event": "arm_done", "arm": name, "rc": rc, "steps": steps,
             "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or steps < 40_000:
            log({"event": "STOP", "why": f"{name} 训练未达标(命题未考,本版不追加重训)"})
            return
        out = RUNS / name / "policy.npz"
        if run([PY, "train/export_manager_npz.py",
                str(RUNS / name / "model_final.zip"), str(out)],
               f"export-{name}.log") != 0:
            log({"event": "STOP", "why": f"{name} G-A0m parity 失败"})
            return
        npz[name] = str(out)
        log({"event": "g_a0m", "arm": name, "npz_sha": sha16(out)})

    # ---- 提前放弃闸(16 种子)----
    s16 = {}
    for name in ARMS:
        d = exam(LEG7_ZIP, f"{name}-s16", "7000-7015", manager_npz=npz[name])
        if d is None:
            log({"event": "STOP", "why": f"{name} 初筛考试失败"})
            return
        s16[name] = d["agg"]["ret_mean"]
        log({"event": "screen16", "arm": name, "score": d["agg"]["ret_mean"],
             "died": d["agg"]["died"]})
    if all(v < ABANDON for v in s16.values()):
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"双臂初筛均 <{ABANDON}——训练失败,换届命题未考(免满 32)"})
        return

    # ---- 两臂满 32(R25.1/R25.2 口径)----
    full = {}
    for name in ARMS:
        d = exam(LEG7_ZIP, f"{name}-full32", "7000-7031", manager_npz=npz[name])
        if d is None:
            log({"event": "STOP", "why": f"{name} 满 32 考试失败"})
            return
        full[name] = d
        log({"event": "full32", "arm": name, "mean": d["agg"]["ret_mean"],
             "died": d["agg"]["died"], "dive_per_ep": round(dive_per_ep(d["rows"]), 2),
             "tau": d["agg"]["farm_tau_mean"], "override": d["agg"]["override_rate"],
             "descend": d["agg"]["farm_descend_rate"], "sha": d["agg"]["_sha"]})
    paired_wf = [full["v25-mwarm"]["rows"][i]["ret"] - full["v25-mfresh"]["rows"][i]["ret"]
                 for i in range(32)]
    log({"event": "r25_2", "paired_warm_minus_fresh_mean": round(sum(paired_wf) / 32, 2)})

    # ---- 胜者 ----
    names = list(ARMS)
    m0, m1 = full[names[0]]["agg"]["ret_mean"], full[names[1]]["agg"]["ret_mean"]
    if abs(m0 - m1) <= 0.05:
        d0, d1 = full[names[0]]["agg"]["died"], full[names[1]]["agg"]["died"]
        winner = names[0] if d0 < d1 else (names[1] if d1 < d0 else "v25-mfresh")
    else:
        winner = names[0] if m0 > m1 else names[1]
    W = full[winner]
    wa = W["agg"]
    log({"event": "winner", "arm": winner, "mean": wa["ret_mean"], "died": wa["died"]})

    # ---- 发射判据(配对 vs 存档 + 哨兵)----
    if wa["ret_mean"] < FLOOR_REPRO:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"胜者 {wa['ret_mean']} < {FLOOR_REPRO}——重训未复现参考水平,命题未考"})
        return
    diffs = [W["rows"][i]["ret"] - ref["rows"][i]["ret"] for i in range(32)]
    pd_mean = sum(diffs) / 32
    pd_wins = sum(d > 0 for d in diffs)
    dive = dive_per_ep(W["rows"])
    void = wa["override_rate"] >= R4["override_void"] or (dive > 1 and wa["died"] > 6)
    hard_ok = (wa["farm_descend_rate"] <= R4["descend"]      # 套利仪表与机械健康:恒硬闸
               and wa["cap_rate"] < R4["cap"])
    override_ok = wa["override_rate"] < R4["override_sentinel"]
    sentinels = hard_ok and override_ok
    # 预注册条件条款:DIVE>1/局 时仅 override 触线走双归因(τ̄ 本就只记不裁)
    dual_attr = dive > 1 and hard_ok and not override_ok
    launch = (pd_mean >= PAIRED_DIFF and pd_wins >= PAIRED_WINS
              and wa["died"] <= DEATHS_MAX and not void
              and (sentinels or dual_attr))
    log({"event": "launch_check", "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins,
         "died": wa["died"], "sentinels": sentinels, "data_void": void,
         "dive_per_ep": round(dive, 2), "dual_attribution": dual_attr,
         "tau_note": wa["farm_tau_mean"]})
    if launch:
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner,
             "model_npz": npz[winner], "probe32": wa["ret_mean"],
             "note": "金牌由值夜者手启,单臂一次;败臂/未发射臂永不见 9000 段"})
    elif pd_mean >= 2.0:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"配对均差 {pd_mean:.2f} ∈[+2,+4)——探针级改进,不烧牌,留工作站复赛"})
    else:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"配对均差 {pd_mean:.2f} <+2——连任,本轮交替无增益(功效限定)"})


if __name__ == "__main__":
    main()
