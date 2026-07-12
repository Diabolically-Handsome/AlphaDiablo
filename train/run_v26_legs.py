"""v26「绿洲」分腿驱动(docs/PREREG-v26.md;v24 条款参数化克隆)。

克隆参数表:LEG=244×2048、PROBES=(250k,450k)、前缀 v26-leg、训练加 --skip-dry;
SPS_FLOOR/tail_cut 禁用(附录重划后的健全性上限:任一腿墙钟 >2 小时 → STOP);
G3 发射改配对判据(对 v24-G3-leg7 存档,v25 先例)。

固定退火 + 双绊线,凌晨无人肉裁量;每一步裁决写 train/runs/v26/gate_ledger.jsonl。
金牌本身不在此发射:G3 判出胜者与资格后停机,由值夜者手启(金牌纪律)。
发车前审查团(wf_976f0385)22 项确认全部落地:崩溃互锁先于 G-CAL、
P* 排除受审腿、G3 override 3% 哨兵线 + ±0.05 平分带、重标定整表重排、
崩溃烧步入预算、per-attempt 尸检留档、双探针 wiring 判据、sps 同账。
用法:.venv/bin/python train/run_v26_legs.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback
import zipfile

from eval_contract import (PROTOCOL_VERSION, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V24 = RUNS / "v26"
V24.mkdir(parents=True, exist_ok=True)
LEDGER = V24 / "gate_ledger.jsonl"
DRIVER_LOCK = V24 / ".driver.lock"
DRIVER_LOCK_PURPOSE = "v26 驱动"

LEG = 244 * 2048          # 499,712(v26 腿制,PREREG-v26 D2)
QUANTUM = 2048
N_LEGS = 8
BUDGET_STEPS = N_LEGS * LEG
BETA_SCHED = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0, 0.0]
HARD_LINE = 62.8          # 【硬-3】(0.8×G1,满32衍生;套 16 种子考更松,沿用不放松)
SOFT_MULT = 0.97          # 【软-4】
SCRIPT_SUBSET = 93.9      # 7000-7015 半池脚本/BC 已知常数(P* 集合的种子元素)
SPS_FLOOR = 1_800_000     # 实步/小时(降档条款;分子=实训步含烧步,与分母同账)
TAIL_CUT_STEPS = 244 * 2048
PROBES = (250_000, 450_000)
BC_SD = str(RUNS / "bc-worker" / "policy_sd.pt")
DEFAULT_MANAGER_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
ANCHOR_SHA = "22d9442257d3a3c79feb5b40918890917772e11036e4026ca4d3cc2005318359"
ANCHOR_WORKER = ROOT / "train" / "models" / "v24-worker-leg7" / "model.zip"
ANCHOR = RUNS / "eval-assembled" / "v24-G3-leg7.json"

# G3/金评资格(v23 附录 B 解释版原文数字;override:3% 哨兵线过闸,8% 另记数据作废)
G3_MEAN = 74.6
G3_DEATHS = 6
R4 = {"farm_descend_rate": 0.0204, "override_sentinel": 0.03, "override_void": 0.08,
      "cap_rate": 0.05, "farm_tau_lo": 27.8, "farm_tau_hi": 46.4}
CALIBRATED_PROTOCOL_VERSION = 2


class OperationalFailure(RuntimeError):
    """基础设施/接线失败；与正常科学绊线区分，进程必须非零退出。"""


def budgeted_leg_steps(spent_steps: int, cap: int = LEG) -> int:
    remaining = max(0, BUDGET_STEPS - spent_steps)
    return min(cap, (remaining // QUANTUM) * QUANTUM)


def ensure_retry_budget(spent_steps: int, cap: int = LEG) -> None:
    if budgeted_leg_steps(spent_steps, cap) == 0:
        raise OperationalFailure("失败尝试已耗尽硬预算，无法完成当前腿")


def observed_attempt_steps(base: int, result: dict, allocated: int) -> int:
    observed = max(int(result.get("global_steps", 0)),
                   int(result.get("status_steps", 0)))
    delta = max(0, observed - base)
    if delta > allocated:
        raise OperationalFailure(
            f"步数越过本次配额:base={base}, observed={observed}, allocated={allocated}")
    return delta


def failed_attempt_charge(observed: int, allocated: int) -> int:
    if not 0 <= observed <= allocated:
        raise OperationalFailure("异常尝试观测步数越界")
    return allocated


def is_gcal_stop(k: int, result: dict, base: int, expected: int,
                 records: list[dict]) -> bool:
    status = int(result.get("status_steps", 0))
    return (k == 1 and result.get("rc") == 0 and base < status <= expected
            and any(r.get("tripped") for r in records))


def reset_recalibration_attempts(attempts: dict[int, int]) -> None:
    attempts[1] = 0


def run_process(cmd, logfile, timeout: int) -> int:
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return 124


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha16(p: pathlib.Path) -> str:
    return sha256(p)[:16]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v26 的 HARD_LINE/SCRIPT_SUBSET/G3/R4 静态阈值仅在 pre-v3 环境语义标定；"
            "必须先重跑 protocol-v3 基线并人工更新预注册，禁止混用旧阈值"
        )


def read_comparable_anchor() -> dict:
    """配对锚必须是当前语义下固定 leg7/default-manager 的 v2 档案。"""
    try:
        snapshot = freeze_eval_identity(ROOT, ANCHOR_WORKER, None)
        expected = expected_eval_identity(
            snapshot, tag="v24-G3-leg7", seeds=range(7000, 7032))
        document = read_eval_archive(ANCHOR, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            "v24-G3-leg7 不满足当前 schema-v2 可比性契约；"
            "环境语义变更后须用固定 leg7 worker + 默认 manager 重跑基线"
        ) from exc


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as zf:
            return int(json.loads(zf.read("data"))["num_timesteps"])
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return 0


def preflight() -> None:
    require_calibrated_protocol()
    require(pathlib.Path(BC_SD).is_file(), f"BC 教师缺失:{BC_SD}")
    read_comparable_anchor()
    for k in range(1, N_LEGS + 1):
        require(not (RUNS / f"v26-leg{k}").exists(), f"运行目录残留:v26-leg{k}")
        for tag in (f"v26-leg{k}", f"v26-G3-leg{k}"):
            require(not (RUNS / "eval-assembled" / f"{tag}.json").exists(),
                    f"评测档案已存在:{tag}")
    require(not (RUNS / "v26-leg1r").exists(), "运行目录残留:v26-leg1r")


def run_leg(k: int, beta: float, resume_from: str | None, leg_steps: int,
            run_name: str, attempt: int) -> dict:
    run_dir = RUNS / run_name
    model_path = run_dir / "model_final.zip"
    old_mtime = model_path.stat().st_mtime_ns if model_path.exists() else None
    stale = run_dir / "status.json"
    if stale.exists():
        stale.unlink()            # 重跑不许读上次尝试的步数
    for fn in ("calib.jsonl", "sentinel.jsonl"):
        p = run_dir / fn
        if p.exists():
            p.rename(p.with_suffix(f".pre{attempt}.{time.time_ns()}.void"))
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo", "--gamma", "1.0",
           "--max-steps", "3000", "--num-envs", "4", "--n-steps", "512", "--lr", "3e-4",
           "--ent-coef", "0.005", "--seed", str(100_000 + 1000 * k),
           "--total-steps", str(leg_steps), "--run-name", run_name,
           "--distill-beta", str(beta), "--teacher-sd", BC_SD, "--skip-dry"]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    else:
        cmd += ["--bc-init", BC_SD, "--freeze-policy-steps", "200000",
                "--calib-probes", ",".join(str(p) for p in PROBES)]
    t0 = time.time()
    rc = run_process(cmd, V24 / f"{run_name}.try{attempt}.log", timeout=10_800)
    dt = time.time() - t0
    sp = run_dir / "status.json"
    try:
        status_steps = int(json.loads(sp.read_text())["total_steps"]) if sp.exists() else 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        status_steps = 0
    fresh_model = model_path.exists() and model_path.stat().st_mtime_ns != old_mtime
    model_steps = zip_steps(model_path) if fresh_model else 0
    return {"rc": rc, "dt_sec": round(dt), "global_steps": model_steps,
            "status_steps": status_steps, "model": model_path, "model_fresh": fresh_model}


def exam(model_path: pathlib.Path, tag: str, seeds: str,
         *, include_rows: bool = False) -> dict | tuple[dict, list] | None:
    worker = model_path.with_suffix("") if model_path.suffix == ".zip" else model_path
    j = RUNS / "eval-assembled" / f"{tag}.json"
    existed_before = j.exists()
    try:
        lo, hi = (int(x) for x in seeds.split("-", 1))
        seed_values = list(range(lo, hi + 1))
        require(seed_values and lo >= 0, f"非法 seed 范围:{seeds}")
        snapshot = freeze_eval_identity(ROOT, worker, DEFAULT_MANAGER_NPZ)
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
        rc = run_process(
            [PY, "train/eval_assembled.py",
             "--worker", snapshot["worker"]["path"],
             "--manager-npz", snapshot["manager"]["path"],
             "--seeds", seeds, "--tag", tag],
            V24 / f"exam-{tag}.{time.time_ns()}.log", timeout=1_800)
        if rc != 0:
            raise OSError(f"评测子进程失败: rc={rc}")
        doc = read_eval_archive(j, **expected)
        verify_eval_identity(snapshot, ROOT)
        agg = doc["agg"]
        agg["_sha"] = sha16(j)
        return (agg, doc["rows"]) if include_rows else agg
    except (OSError, KeyError, TypeError, ValueError):
        if not existed_before and j.exists():
            j.rename(j.with_suffix(f".{time.time_ns()}.void"))
        return None


def second_largest(vals):
    s = sorted(vals, reverse=True)
    return s[1] if len(s) > 1 else s[0]


def rows_by_seed(rows, expected=range(7000, 7032)):
    by_seed = {r["seed"]: r for r in rows}
    require(len(rows) == len(by_seed), "评测档案含重复 seed")
    require(set(by_seed) == set(expected), "评测档案 seed 集合异常")
    return by_seed


def main():
    try:
        with exclusive_lock(DRIVER_LOCK, DRIVER_LOCK_PURPOSE):
            _main()
    except (OperationalFailure, OutputReservationError) as exc:
        log({"event": "OPERATIONAL_FAILURE", "why": str(exc)})
        raise SystemExit(2) from exc
    except Exception as exc:
        log({"event": "DRIVER_EXCEPTION", "why": repr(exc),
             "traceback": traceback.format_exc()})
        raise


def _main():
    preflight()
    log({"event": "start", "leg_steps": LEG, "beta_sched": list(BETA_SCHED),
         "hard": HARD_LINE, "soft": SOFT_MULT})
    scores = []                 # 已完成腿考分(1 位小数)
    sched_idx = 0               # 软绊冻结 = 指针不进
    recalibrated = False
    burned = 0                  # 被丢弃的重标定/崩溃步(审计口径)
    spent_steps = 0             # 所有尝试的观测新步；逐次约束 BUDGET_STEPS
    chain_steps = 0             # 当前计数链上应有的累计步(重标定后归零)
    prev_model = None
    leg_models = {}
    tail_cut = False
    attempts = {}
    k = 1
    while k <= N_LEGS:
        beta = BETA_SCHED[min(sched_idx, len(BETA_SCHED) - 1)]
        cap = LEG   # v26:tail_cut 禁用(窗口保护为唯一降档)
        leg_steps = budgeted_leg_steps(spent_steps, cap)
        if leg_steps < cap:
            log({"event": "leg_budget_shrunk", "leg": k, "steps": leg_steps,
                 "spent": spent_steps, "remaining": BUDGET_STEPS - spent_steps,
                 "note": "所有既成新步与失败/重标定烧步逐次扣减硬预算"})
        if leg_steps == 0:
            log({"event": "budget_exhausted", "leg": k, "spent": spent_steps})
            break
        attempts[k] = attempts.get(k, 0) + 1
        run_name = f"v26-leg{k}" + ("r" if (k == 1 and recalibrated) else "")
        log({"event": "leg_start", "leg": k, "attempt": attempts[k], "beta": beta,
             "steps": leg_steps, "seed": 100_000 + 1000 * k, "resume": bool(prev_model)})
        res = run_leg(k, beta, prev_model, leg_steps, run_name, attempts[k])
        expected = chain_steps + leg_steps

        # ---- 【考-2】崩溃互锁(先于一切裁决——审查团 blocker 修正)----
        calib_p = RUNS / run_name / "calib.jsonl"
        try:
            calib_recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                          if k == 1 and calib_p.exists() else [])
        except (OSError, json.JSONDecodeError):
            calib_recs = []
        gcal_stop = is_gcal_stop(k, res, chain_steps, expected, calib_recs)
        clean = (res["rc"] == 0 and res["model_fresh"]
                 and res["global_steps"] == expected)
        sampled = observed_attempt_steps(chain_steps, res, leg_steps)
        if clean:
            require(sampled == leg_steps, "干净收官的观测步数与配额不一致")
            spent_steps += sampled
            chain_steps = res["global_steps"]
        elif gcal_stop:
            spent_steps += sampled
            burned += sampled
            log({"event": "gcal_early_stop", "leg": k, "observed_steps": sampled,
                 "status_steps": res["status_steps"], "burned_total": burned,
                 "spent_total": spent_steps})
        else:
            partial = sampled
            charged = failed_attempt_charge(partial, leg_steps)
            spent_steps += charged
            burned += charged
            log({"event": "leg_crash", "leg": k, "attempt": attempts[k],
                 "rc": res["rc"], "global_steps": res["global_steps"],
                 "burned_observed": partial, "burned_charged": charged,
                 "burned_total": burned,
                 "spent_total": spent_steps,
                 "note": "按【终-6】原配置重跑；烧步实时扣减后续可开配额"})
            if k == 1:
                calib = RUNS / run_name / "calib.jsonl"
                if calib.exists():   # 崩溃尝试的探针记录轮转,不污染 G-CAL 裁决
                    calib.rename(calib.with_suffix(f".try{attempts[k]}.void"))
            ensure_retry_budget(spent_steps, cap)
            if attempts[k] >= 4:
                why = (f"腿 {k} 连崩 {attempts[k]} 次——驱动自护停机"
                       "(非预注册科学闸门,需人工验尸)")
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)
            continue

        # ---- G-CAL(仅腿 1；完整收官或三证齐全的正常早停)----
        if k == 1:
            recs = calib_recs
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
                if not gcal_stop:
                    burned += sampled
                BETA_SCHED[:] = [2.0 * 0.5 ** i for i in range(6)] + [0.0, 0.0]
                log({"event": "recalibrate", "beta0": 2.0,
                     "new_sched": list(BETA_SCHED), "burned": burned,
                     "note": "唯一一次 β₀×4:整条日程按 β_k=β₀·2^{-(k-1)} 重排"
                             "(腿 7/8 钉 0 不动,拍板记录补条),烧步实时扣后续配额"})
                prev_model = None
                chain_steps = 0
                sched_idx = 0
                reset_recalibration_attempts(attempts)
                continue    # k 仍为 1
            if not probes_ok:
                why = ("G-CAL 接线失败(双探针未见 ce/g_ce>0)"
                       "——修码后按崩溃条款重跑,需人工介入")
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)

        # ---- G-绿洲(仅完整收官腿 1；主动 G-CAL 早停须先重标定)----
        if k == 1:
            sent = RUNS / run_name / "sentinel.jsonl"
            lines = [json.loads(l) for l in sent.read_text().splitlines()
                     if '"sentinel": "v23"' in l] if sent.exists() else []
            if not lines:
                why = "G-绿洲失败:无哨兵行"
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)
            last = lines[-1]
            oasis_ok = last.get("dry", 1) == 0 and last.get("ff_dry", 0) > 0
            log({"event": "g_oasis", "dry": last.get("dry"),
                 "ff_dry": last.get("ff_dry"), "fresh": last.get("fresh"),
                 "ok": oasis_ok})
            if not oasis_ok:
                why = "G-绿洲失败:学习窗含 dry 或 ff_dry=0"
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)

        # ---- 腿考 ----
        agg = exam(res["model"], f"v26-leg{k}", "7000-7015")
        if agg is None:
            log({"event": "exam_crash", "leg": k, "note": "考试进程失败,按崩溃条款重考"})
            agg = exam(res["model"], f"v26-leg{k}", "7000-7015")
            if agg is None:
                why = "考试连败 2 次——人工验尸"
                log({"event": "STOP", "why": why})
                raise OperationalFailure(why)
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
        if res["dt_sec"] > 7200:
            why = "腿墙钟 >2 小时——健全性上限(夜航班重划,PREREG-v26 附录)"
            log({"event": "STOP", "why": why})
            raise OperationalFailure(why)
        prev_model = str(res["model"])
        k += 1

    # ---- G3:候选写死 = 腿末 ckpt 按腿考分 top-2 ----
    if not leg_models:
        why = "无任何完成腿"
        log({"event": "STOP", "why": why})
        raise OperationalFailure(why)
    top2 = sorted(leg_models.items(), key=lambda kv: kv[1][0], reverse=True)[:2]
    log({"event": "g3_candidates", "cands": [(kk, v[0], v[2]) for kk, v in top2]})
    finals = []
    g3_rows = {}
    for kk, (sc16, mp, bt) in top2:
        exam_result = exam(pathlib.Path(mp), f"v26-G3-leg{kk}", "7000-7031",
                           include_rows=True)
        if exam_result is None:
            why = f"G3 满32考试失败(腿 {kk})——人工验尸"
            log({"event": "STOP", "why": why})
            raise OperationalFailure(why)
        agg, g3_rows[kk] = exam_result
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
    # 配对发射判据(PREREG-v26 D3:对 v24-G3-leg7 逐种子 ≥+4 且赢 ≥18/32)
    archive = read_comparable_anchor()
    archive_rows = rows_by_seed(archive["rows"])
    paired = {}
    for kk, (sc16, mp, bt) in top2:
        candidate_rows = rows_by_seed(g3_rows[kk])
        diffs = [candidate_rows[s]["ret"] - archive_rows[s]["ret"]
                 for s in range(7000, 7032)]
        paired[kk] = (sum(diffs) / 32, sum(d > 0 for d in diffs))
        log({"event": "paired", "leg": kk, "mean_diff": round(paired[kk][0], 2),
             "wins": paired[kk][1]})
    finals = [(kk, m, dd, ok and paired[kk][0] >= 4.0 and paired[kk][1] >= 18, bt, dv, mp)
              for (kk, m, dd, ok, bt, dv, mp) in finals]
    qual = [f for f in finals if f[3]]
    if not qual:
        best_leg, (best_pd, best_wins) = max(paired.items(), key=lambda x: x[1][0])
        if best_pd >= 4.0 and best_wins < 18:
            band = f"均值增益但宽度未达(腿 {best_leg},赢 {best_wins}/32),不烧牌"
        elif best_pd >= 2.0:
            band = "探针级改进 [+2,+4),不烧牌留工作站复赛"
        else:
            band = "连任,本轮无增益"
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"未达配对发射线(最佳均差 {best_pd:.2f})——{band}"})
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
