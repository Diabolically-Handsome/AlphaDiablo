"""v27 分腿驱动(v24 配方 ×7M;总设计师睡前嘱咐的 Otherwise 分支)。

克隆参数:LEG=427×2048(约875k,8腿共≈7M)、前缀 v27-leg、无 skip-dry;
发射判据升级为配对标准(对 v24-G3-leg7 存档 ≥+4 且赢 ≥18/32,v25/v26 先例);
SPS_FLOOR 沿 v24(标准步率适用)。

固定退火 + 双绊线,凌晨无人肉裁量;每一步裁决写 train/runs/v24/gate_ledger.jsonl。
金牌本身不在此发射:G3 判出胜者与资格后停机,由值夜者手启(金牌纪律)。
发车前审查团(wf_976f0385)22 项确认全部落地:崩溃互锁先于 G-CAL、
P* 排除受审腿、G3 override 3% 哨兵线 + ±0.05 平分带、重标定整表重排、
崩溃烧步入预算、per-attempt 尸检留档、双探针 wiring 判据、sps 同账。
用法:.venv/bin/python train/run_v24_legs.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V24 = RUNS / "v27"
V24.mkdir(parents=True, exist_ok=True)
LEDGER = V24 / "gate_ledger.jsonl"

LEG = 427 * 2048          # 874,496(v27:7M/8 腿,总设计师嘱咐)
N_LEGS = 8
BETA_SCHED = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0, 0.0]
HARD_LINE = 62.8          # 【硬-3】(0.8×G1,满32衍生;套 16 种子考更松,沿用不放松)
SOFT_MULT = 0.97          # 【软-4】
SCRIPT_SUBSET = 93.9      # 7000-7015 半池脚本/BC 已知常数(P* 集合的种子元素)
SPS_FLOOR = 1_800_000     # 实步/小时(降档条款;分子=实训步含烧步,与分母同账)
TAIL_CUT_STEPS = 244 * 2048
PROBES = (300_000, 600_000)
BC_SD = str(RUNS / "bc-worker" / "policy_sd.pt")

# G3/金评资格(v23 附录 B 解释版原文数字;override:3% 哨兵线过闸,8% 另记数据作废)
G3_MEAN = 74.6
G3_DEATHS = 6
R4 = {"farm_descend_rate": 0.0204, "override_sentinel": 0.03, "override_void": 0.08,
      "cap_rate": 0.05, "farm_tau_lo": 27.8, "farm_tau_hi": 46.4}


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def run_leg(k: int, beta: float, resume_from: str | None, leg_steps: int,
            run_name: str, attempt: int) -> dict:
    run_dir = RUNS / run_name
    stale = run_dir / "status.json"
    if stale.exists():
        stale.unlink()            # 重跑不许读上次尝试的步数
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo", "--gamma", "1.0",
           "--max-steps", "3000", "--num-envs", "4", "--n-steps", "512", "--lr", "3e-4",
           "--ent-coef", "0.005", "--seed", str(100_000 + 1000 * k),
           "--total-steps", str(leg_steps), "--run-name", run_name,
           "--distill-beta", str(beta), "--teacher-sd", BC_SD]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    else:
        cmd += ["--bc-init", BC_SD, "--freeze-policy-steps", "200000",
                "--calib-probes", ",".join(str(p) for p in PROBES)]
    t0 = time.time()
    with open(V24 / f"{run_name}.try{attempt}.log", "w") as lf:  # per-attempt 尸检留档
        rc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    sp = run_dir / "status.json"
    gsteps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
    return {"rc": rc, "dt_sec": round(dt), "global_steps": gsteps,
            "model": run_dir / "model_final.zip"}


def exam(model_path: pathlib.Path, tag: str, seeds: str) -> dict | None:
    rc = subprocess.run([PY, "train/eval_assembled.py", "--worker",
                         str(model_path).replace(".zip", ""), "--seeds", seeds,
                         "--tag", tag], cwd=ROOT,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc != 0:
        return None
    j = RUNS / "eval-assembled" / f"{tag}.json"
    agg = json.loads(j.read_text())["agg"]
    agg["_sha"] = sha16(j)
    return agg


def second_largest(vals):
    s = sorted(vals, reverse=True)
    return s[1] if len(s) > 1 else s[0]


def main():
    log({"event": "start", "leg_steps": LEG, "beta_sched": list(BETA_SCHED),
         "hard": HARD_LINE, "soft": SOFT_MULT})
    scores = []                 # 已完成腿考分(1 位小数)
    sched_idx = 0               # 软绊冻结 = 指针不进
    recalibrated = False
    burned = 0                  # 烧步(重标定 + 崩溃部分步),从腿 8 扣,入 8M 硬预算
    chain_steps = 0             # 当前计数链上应有的累计步(重标定后归零)
    extra_steps = 0             # 已烧但不在当前计数链上的实训步(sps 同账用)
    train_secs = 0.0
    prev_model = None
    leg_models = {}
    tail_cut = False
    attempts = {}
    k = 1
    while k <= N_LEGS:
        beta = BETA_SCHED[min(sched_idx, len(BETA_SCHED) - 1)]
        cap = TAIL_CUT_STEPS if (tail_cut and k >= 7) else LEG
        leg_steps = max(0, min(cap, LEG - burned)) if k == N_LEGS and burned else cap
        if k == N_LEGS and burned and leg_steps < 500_000:
            log({"event": "leg8_shrunk", "steps": leg_steps,
                 "note": "烧步扣减:P-拐杖-真自动降格半档(预注册 G-CAL/终-6)"})
        if leg_steps == 0:
            log({"event": "leg8_skipped"})
            break
        attempts[k] = attempts.get(k, 0) + 1
        run_name = f"v27-leg{k}" + ("r" if (k == 1 and recalibrated) else "")
        log({"event": "leg_start", "leg": k, "attempt": attempts[k], "beta": beta,
             "steps": leg_steps, "seed": 100_000 + 1000 * k, "resume": bool(prev_model)})
        res = run_leg(k, beta, prev_model, leg_steps, run_name, attempts[k])
        train_secs += res["dt_sec"]
        expected = chain_steps + leg_steps

        # ---- 【考-2】崩溃互锁(先于一切裁决——审查团 blocker 修正)----
        clean = res["rc"] == 0 and res["global_steps"] >= expected - 2048
        if not clean:
            partial = max(0, res["global_steps"] - chain_steps)
            burned += partial
            extra_steps += partial
            log({"event": "leg_crash", "leg": k, "attempt": attempts[k],
                 "rc": res["rc"], "global_steps": res["global_steps"],
                 "burned_partial": partial, "burned_total": burned,
                 "note": "按【终-6】原配置重跑,烧步计入 8M 硬预算(从腿 8 扣)"})
            if k == 1:
                calib = RUNS / run_name / "calib.jsonl"
                if calib.exists():   # 崩溃尝试的探针记录轮转,不污染 G-CAL 裁决
                    calib.rename(calib.with_suffix(f".try{attempts[k]}.void"))
            if attempts[k] >= 4:
                log({"event": "STOP", "why": f"腿 {k} 连崩 {attempts[k]} 次——驱动自护"
                     "停机(注:此上限系运维自护,非预注册闸门,触发即人工验尸)"})
                return
            continue
        chain_steps = res["global_steps"]

        # ---- G-CAL(仅腿 1,且仅裁决干净收官的腿——审查团 blocker 修正)----
        if k == 1:
            calib_p = RUNS / run_name / "calib.jsonl"
            recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                    if calib_p.exists() else [])
            tripped = any(r.get("tripped") for r in recs)
            probes_ok = all(any(p <= r["step"] < p + 2048 and r["g_ce"] > 0
                                and r["distill_ce"] > 0 for r in recs)
                            for p in PROBES)
            log({"event": "g_cal", "records": recs, "tripped": tripped,
                 "probes_ok": probes_ok})
            if tripped:
                if recalibrated:
                    log({"event": "STOP", "why": "G-CAL 二次触发 = 设计判死,停机写判决"})
                    return
                recalibrated = True
                burned += res["global_steps"]
                extra_steps += res["global_steps"]
                BETA_SCHED[:] = [2.0 * 0.5 ** i for i in range(6)] + [0.0, 0.0]
                log({"event": "recalibrate", "beta0": 2.0,
                     "new_sched": list(BETA_SCHED), "burned": burned,
                     "note": "唯一一次 β₀×4:整条日程按 β_k=β₀·2^{-(k-1)} 重排"
                             "(腿 7/8 钉 0 不动,拍板记录补条),烧步从腿 8 扣"})
                prev_model = None
                chain_steps = 0
                sched_idx = 0
                continue    # k 仍为 1
            if not probes_ok:
                log({"event": "STOP", "why": "G-CAL 接线失败(双探针未见 ce/g_ce>0)"
                     "——修码后按崩溃条款重跑,需人工介入"})
                return

        # ---- 腿考 ----
        agg = exam(res["model"], f"v24-leg{k}", "7000-7015")
        if agg is None:
            log({"event": "exam_crash", "leg": k, "note": "考试进程失败,按崩溃条款重考"})
            agg = exam(res["model"], f"v24-leg{k}", "7000-7015")
            if agg is None:
                log({"event": "STOP", "why": "考试连败 2 次——人工验尸"})
                return
        score = round(agg["ret_mean"], 1)
        p_star = second_largest([SCRIPT_SUBSET] + scores)   # 排除受审腿(审查团修正:
        scores.append(score)                                 # 腿 1 软绊线 = 0.97×93.9 = 91.1)
        leg_models[k] = (score, str(res["model"]), beta)
        log({"event": "leg_exam", "leg": k, "beta": beta, "score": score,
             "died": agg["died"], "diverge": agg.get("script_divergence_rate"),
             "sha": agg["_sha"], "model_sha": sha16(res["model"]),
             "p_star_prior": p_star, "global_steps": res["global_steps"]})

        # ---- 【硬-3】 ----
        if score < HARD_LINE:
            log({"event": "HARD_TRIP", "leg": k, "score": score,
                 "why": f"< {HARD_LINE},本 run 训练永久终止(回卷重训上限=0)"})
            break
        # ---- 【软-4】 ----
        if score < round(SOFT_MULT * p_star, 1):
            log({"event": "soft_trip", "leg": k, "score": score,
                 "line": round(SOFT_MULT * p_star, 1), "note": "β 冻结,日程右移"})
        else:
            sched_idx += 1
        # ---- sps 降档(分子含烧步,与分母同账;k<8 允许在腿 7 后砍腿 8)----
        rate = (res["global_steps"] + extra_steps) / max(1e-9, train_secs) * 3600
        if rate < SPS_FLOOR and k < 8 and not tail_cut:
            tail_cut = True
            log({"event": "sps_downshift", "rate_per_h": round(rate),
                 "note": "腿 7-8 各砍至 244×2048,判词同步降格(预注册)"})
        prev_model = str(res["model"])
        k += 1

    # ---- G3:候选写死 = 腿末 ckpt 按腿考分 top-2 ----
    if not leg_models:
        log({"event": "STOP", "why": "无任何完成腿"})
        return
    top2 = sorted(leg_models.items(), key=lambda kv: kv[1][0], reverse=True)[:2]
    log({"event": "g3_candidates", "cands": [(kk, v[0], v[2]) for kk, v in top2]})
    finals = []
    for kk, (sc16, mp, bt) in top2:
        agg = exam(pathlib.Path(mp), f"v27-G3-leg{kk}", "7000-7031")
        if agg is None:
            log({"event": "STOP", "why": f"G3 满32考试失败(腿 {kk})——人工验尸"})
            return
        void = agg["override_rate"] >= R4["override_void"]
        r4_ok = (agg["farm_descend_rate"] <= R4["farm_descend_rate"]
                 and agg["override_rate"] < R4["override_sentinel"]   # 3% 哨兵线过闸
                 and agg["cap_rate"] < R4["cap_rate"]
                 and R4["farm_tau_lo"] <= agg["farm_tau_mean"] <= R4["farm_tau_hi"])
        ok = agg["ret_mean"] >= G3_MEAN and agg["died"] <= G3_DEATHS and r4_ok and not void
        finals.append((kk, agg["ret_mean"], agg["died"], ok, bt,
                       agg.get("script_divergence_rate"), mp))
        log({"event": "g3_full32", "leg": kk, "mean": agg["ret_mean"],
             "died": agg["died"], "r4_ok": r4_ok, "data_void": void, "qualified": ok,
             "diverge": agg.get("script_divergence_rate"),
             "override": agg["override_rate"], "descend_rate": agg["farm_descend_rate"],
             "tau": agg["farm_tau_mean"]})
    archive = json.loads((RUNS / "eval-assembled" / "v24-G3-leg7.json").read_text())
    paired = {}
    for kk, (sc16, mp, bt) in top2:
        rows = json.loads((RUNS / "eval-assembled" / f"v27-G3-leg{kk}.json").read_text())["rows"]
        diffs = [rows[i]["ret"] - archive["rows"][i]["ret"] for i in range(32)]
        paired[kk] = (sum(diffs) / 32, sum(d > 0 for d in diffs))
        log({"event": "paired", "leg": kk, "mean_diff": round(paired[kk][0], 2),
             "wins": paired[kk][1]})
    finals = [(kk, m, dd, ok and paired[kk][0] >= 4.0 and paired[kk][1] >= 18, bt, dv, mp)
              for (kk, m, dd, ok, bt, dv, mp) in finals]
    qual = [f for f in finals if f[3]]
    if not qual:
        det = {kk: {"mean_diff": round(v[0],2), "wins": v[1]} for kk,v in paired.items()}
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"未达配对发射线(逐项:{det};线 = 均差≥+4 且 赢≥18/32 且资格闸)"})
        return
    qual.sort(key=lambda f: -f[1])
    w = qual[0]
    tie = len(qual) == 2 and abs(qual[0][1] - qual[1][1]) <= 0.05
    if tie:
        w = min(qual, key=lambda f: f[4])    # ±0.05 平分带:取 β 更低的腿
    log({"event": "GOLDEN_AUTHORIZED", "leg": w[0], "probe32_mean": w[1],
         "died": w[2], "beta_of_leg": w[4], "diverge": w[5], "model": w[6],
         "tie_band_applied": tie,
         "note": "金牌由值夜者手启,单臂一次(金牌纪律)"})


if __name__ == "__main__":
    main()
