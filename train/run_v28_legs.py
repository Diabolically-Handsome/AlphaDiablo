"""v28「绿洲续航」分腿驱动(docs/PREREG-v28.md 终稿;run_v26_legs.py 定向改造)。

与 v26 驱动的差异表(PREREG-v28 D2 克隆差异表逐条对应,面板 24 项裁决落地):
- 续航:全 8 腿 resume,腿 1 起点 = v26-leg6/model_final.zip;bc-init 分支物理删除
- β 恒 0.015625:BETA_SCHED/sched_idx/软绊 0.97×P*/SCRIPT_SUBSET/recalibrate 全删
- G-CAL:每腿传绝对步探针 nt_chain+250k / +450k(不可达则裁掉并入账),
  --calib-record-only 只记不裁;probes_ok 接线闸每腿执行;崩溃尝试 calib 轮转每腿
- 计步单一化:nt_chain 从 START=2,998,272(zip num_timesteps,发车断言)起,
  status.json 计数器 ~300 步滞后由 ±2048 slack 覆盖(v26-leg6 实测滞后 268)
- 收官绊线:硬绊 <62.8 → break 进 G3;连续两条干净腿 <103.1(16 种子口径)→
  break 进 G3(崩溃尝试不计不清零;任一腿 ≥103.1 清零;硬绊优先)
- 健全性 2h/腿 → break 进 G3(v26 系裸 STOP,面板判其与预算保护哲学矛盾)
- 档案不可变性:腿考 tag v28-leg{k}、G3 tag v28-G3-leg{k};exam() 拒绝覆写;
  发车断言锚 sha(v24/v26 逐种子腿考档案已因标签碰撞灭失两轮,事故入册)
- 宽度探针:腿考 JSON rows 对锚前 16(seed 7000-7015)按 seed 键配对计赢,
  并列/缺行不计赢;纯后处理零额外评测
- G3 候选:均值 top-2 ∪ 宽度 top-1(去重 ≤3);起点本尊不入池;并列规则注册
- 发射:配对(按 seed 键 join+断言)均差 ≥+4 且赢 ≥18/32 且资格;
  仅经宽度通道入池的候选加出样本副线:满 32 后 16 配对赢 ≥8/16(基线 6/16)
- VERDICT 三键分派穷尽(资格外/副线拦截/宽度达标幅度未达/内禀 ≥108.2/
  [103,108.2) 未答/退化 <103);GOLDEN_AUTHORIZED 带 model_sha+档案 sha+金评命令
- 一切非常规路径写 runs/v28/NEEDS_ATTENTION(值夜晨检第一眼)
金牌不在此发射:判出胜者与资格后停机,由值夜者手启(金牌纪律)。
用法:.venv/bin/python train/run_v28_legs.py
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
V28 = RUNS / "v28"
V28.mkdir(parents=True, exist_ok=True)
LEDGER = V28 / "gate_ledger.jsonl"
DRIVER_LOCK = V28 / ".driver.lock"
DRIVER_LOCK_PURPOSE = "v28 驱动"
EVAL = RUNS / "eval-assembled"

LEG = 244 * 2048              # 499,712(v26 腿制原封)
QUANTUM = 2048
N_LEGS = 8
BUDGET_STEPS = N_LEGS * LEG     # 只计 START 之后的新步
BETA = 0.015625               # 恒定,永不撒手(教训十九 + v27「不可定居」)
HARD_LINE = 62.8
FINISH_LINE = 103.1           # =round(0.9×114.5,1),16 种子腿考口径(勿与满32口径 103 混读)
INTRINSIC_LINE = 108.2        # 满 32 口径:起点检查点本尊的满32均值(内禀档门槛)
PLATEAU_LINE = 103.0          # 满 32 口径:[103,108.2) 落「续航无均值增益」档
START = 2_998_272             # v26-leg6 zip num_timesteps(6×499,712;发车断言)
BASE_CKPT = RUNS / "v26-leg6" / "model_final.zip"
BC_SD = str(RUNS / "bc-worker" / "policy_sd.pt")
ANCHOR = EVAL / "v24-G3-leg7.json"
ANCHOR_SHA = "22d9442257d3a3c79feb5b40918890917772e11036e4026ca4d3cc2005318359"
ANCHOR_WORKER = ROOT / "train" / "models" / "v24-worker-leg7" / "model.zip"
BASELINE = EVAL / "v26-G3-leg6.json"
BASELINE_SHA = "24a905a7baf0f70ab09a8103721757f10eec0692b64b81f9880abedd2e0325dd"
DEFAULT_MANAGER_SHA = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"

G3_MEAN = 74.6
G3_DEATHS = 6
R4 = {"farm_descend_rate": 0.0204, "override_sentinel": 0.03, "override_void": 0.08,
      "cap_rate": 0.05, "farm_tau_lo": 27.8, "farm_tau_hi": 46.4}
SIDELINE_BACK16 = 8           # 仅宽度通道候选的出样本副线(零假设 P(≥8)≈16%)
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


def candidate_probe_eligible(probes_ok: bool | None) -> bool:
    """只有显式 PASS 能进候选；空探针的 SKIPPED 必须 fail closed。"""
    return probes_ok is True


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V28 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


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
            "v28 的 HARD_LINE/G3/R4/平台与候选静态阈值仅在 pre-v3 环境语义标定；"
            "必须先重跑 protocol-v3 基线并人工更新预注册，禁止混用旧阈值"
        )


def parse_seed_range(seeds: str) -> list[int]:
    """按 eval_assembled 的 LO-HI 契约解析，拒绝额外分隔符与负数。"""
    parts = seeds.split("-") if isinstance(seeds, str) else []
    require(len(parts) == 2 and all(p.isascii() and p.isdigit() for p in parts),
            f"非法 seed 范围:{seeds!r}")
    lo, hi = (int(p) for p in parts)
    require(lo <= hi, f"非法 seed 范围:{seeds!r}")
    return list(range(lo, hi + 1))


def read_comparable_reference(path: pathlib.Path, *, tag: str,
                              worker: pathlib.Path, label: str) -> dict:
    """活动锚只接受当前语义、固定组装体和完整 runtime/content 身份。"""
    try:
        snapshot = freeze_eval_identity(ROOT, worker, None)
        expected = expected_eval_identity(
            snapshot, tag=tag, seeds=range(7000, 7032))
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            f"{label} 不满足当前 schema-v2 可比性契约；"
            "环境语义变更后须用固定 worker + 默认 manager 重跑基线"
        ) from exc


def read_comparable_anchor() -> dict:
    return read_comparable_reference(
        ANCHOR, tag="v24-G3-leg7", worker=ANCHOR_WORKER, label="续航配对锚")


def read_comparable_baseline() -> dict:
    return read_comparable_reference(
        BASELINE, tag="v26-G3-leg6", worker=BASE_CKPT, label="续航起点基线")


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as zf:
            return int(json.loads(zf.read("data"))["num_timesteps"])
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return 0


def reachable_probes(start: int, leg_steps: int,
                     offsets=(250_000, 450_000)) -> list[int]:
    """只保留能在本腿某个 rollout 收官点触发的绝对探针步。"""
    end = start + leg_steps
    probes = []
    for offset in offsets:
        target = start + offset
        first_rollout = start + ((offset + QUANTUM - 1) // QUANTUM) * QUANTUM
        if first_rollout <= end:
            probes.append(target)
    return probes


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


def preflight():
    require_calibrated_protocol()
    require(BASE_CKPT.exists(), f"起点检查点缺失:{BASE_CKPT}")
    nt = zip_steps(BASE_CKPT)
    require(nt == START, f"START 断言失败:zip num_timesteps={nt} != {START}")
    st = json.loads((RUNS / "v26-leg6" / "status.json").read_text())["total_steps"]
    require(START - 2048 <= st <= START, f"status 计数器滞后越界:{st}")
    read_comparable_anchor()
    read_comparable_baseline()
    for k in range(1, N_LEGS + 1):
        for tag in (f"v28-leg{k}", f"v28-G3-leg{k}"):
            require(not (EVAL / f"{tag}.json").exists(), f"目标档案已存在:{tag}")
        require(not (RUNS / f"v28-leg{k}").exists(),
                f"运行目录残留:v28-leg{k}(重启协议:整体归档后再发车)")
    require(not (EVAL / "v28-golden.json").exists(), "金评目标档案已存在(金牌至多一次)")
    log({"event": "preflight_ok", "start_nt": nt, "status_lag": START - st,
         "anchor_sha": sha16(ANCHOR), "baseline_sha": sha16(BASELINE)})


def run_leg(k: int, resume_from: str, leg_steps: int, probes: list[int],
            run_name: str, attempt: int, seed_k: int) -> dict:
    run_dir = RUNS / run_name
    model_path = run_dir / "model_final.zip"
    old_mtime = model_path.stat().st_mtime_ns if model_path.exists() else None
    stale = run_dir / "status.json"
    if stale.exists():
        stale.unlink()            # 重跑不许读上次尝试的步数
    for fn in ("calib.jsonl", "sentinel.jsonl"):
        p = run_dir / fn          # 尝试进场先清桌:崩溃重试/重启的陈旧记录不许假通过闸门
        if p.exists():
            p.rename(p.with_suffix(f".pre{attempt}.{time.strftime('%H%M%S')}.void"))
    cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo", "--gamma", "1.0",
           "--max-steps", "3000", "--num-envs", "4", "--n-steps", "512", "--lr", "3e-4",
           "--ent-coef", "0.005", "--seed", str(seed_k),
           "--total-steps", str(leg_steps), "--run-name", run_name,
           "--distill-beta", str(BETA), "--teacher-sd", BC_SD, "--skip-dry",
           "--resume-from", resume_from]
    if probes:
        cmd += ["--calib-probes", ",".join(str(p) for p in probes),
                "--calib-record-only"]
    t0 = time.time()
    # 挂死护栏 3h(>2h 健全线)；进程组整组终止，避免 SubprocVecEnv 孤儿继续占机。
    rc = run_process(cmd, V28 / f"{run_name}.try{attempt}.log", timeout=10_800)
    dt = time.time() - t0
    sp = run_dir / "status.json"
    try:
        gsteps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
    except Exception:
        gsteps = 0                # 半截 status(写入中被杀):按崩溃条款落账
    fresh_model = model_path.exists() and model_path.stat().st_mtime_ns != old_mtime
    model_steps = zip_steps(model_path) if fresh_model else 0
    return {"rc": rc, "dt_sec": round(dt), "global_steps": model_steps,
            "status_steps": gsteps, "model": model_path, "model_fresh": fresh_model}


def exam(model_path: pathlib.Path, tag: str, seeds: str) -> tuple[dict, list] | None:
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    try:
        seed_values = parse_seed_range(seeds)
        snapshot = freeze_eval_identity(ROOT, model_path, None)
        require(snapshot["manager"]["sha256"] == DEFAULT_MANAGER_SHA,
                "默认 manager sha 漂移")
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
        command = [PY, "train/eval_assembled.py",
                   "--worker", snapshot["worker"]["path"],
                   "--manager-npz", snapshot["manager"]["path"],
                   "--seeds", seeds, "--tag", tag]
        rc = run_process(
            command, V28 / f"{tag}.eval.{time.time_ns()}.log", timeout=1_800)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    if rc != 0:
        if out.exists():      # 半截档案轮转,给重考让路
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        document = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
        agg = document["agg"]
        agg["_sha"] = sha16(out)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    return agg, document["rows"]


def exam_retry(model_path: pathlib.Path, tag: str, seeds: str, what: str):
    r = exam(model_path, tag, seeds)
    if r is None:
        log({"event": "exam_crash", "tag": tag, "note": f"{what}失败,按崩溃条款重考一次"})
        r = exam(model_path, tag, seeds)
    return r


def breadth_wins(rows: list, anchor_by_seed: dict, lo: int, hi: int) -> int:
    by_seed = {r["seed"]: r["ret"] for r in rows}
    require(len(rows) == len(by_seed), "评测档案含重复 seed")
    require(set(range(lo, hi + 1)) <= set(by_seed), "评测档案缺少所需 seed")
    return sum(1 for s in range(lo, hi + 1)
               if s in by_seed and by_seed[s] > anchor_by_seed[s])


def main():
    try:
        with exclusive_lock(DRIVER_LOCK, DRIVER_LOCK_PURPOSE):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("运维失败:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:   # 条款⑧兜底:任何未预期异常必须入册,不许无声死亡
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    anchor = read_comparable_anchor()
    anchor_by_seed = {r["seed"]: r["ret"] for r in anchor["rows"]}
    require(len(anchor["rows"]) == len(anchor_by_seed), "锚档案含重复种子")
    require(set(anchor_by_seed) == set(range(7000, 7032)), "锚种子集合异常")
    baseline = read_comparable_baseline()
    baseline_front16 = sum(r["ret"] for r in baseline["rows"][:16]) / 16
    finish_line = round(0.9 * baseline_front16, 1)
    intrinsic_line = baseline["agg"]["ret_mean"]
    log({"event": "start", "leg_steps": LEG, "beta_const": BETA, "hard": HARD_LINE,
         "finish": finish_line, "intrinsic": intrinsic_line,
         "start_nt": START, "base": str(BASE_CKPT)})

    nt_chain = START            # SB3 真链计数(单一计数源)
    burned = 0                  # 崩溃部分步审计口径；实时扣后续可开配额
    spent_steps = 0             # START 之后所有成功/失败尝试的观测新步
    train_secs = 0.0
    prev_model = str(BASE_CKPT)
    leg_models = {}             # k -> (score16, model_path, breadth16)
    attempts = {}
    consec_low = 0
    stop_reason = None
    k = 1
    while k <= N_LEGS:
        cap = LEG
        leg_steps = budgeted_leg_steps(spent_steps, cap)
        if leg_steps < cap:
            log({"event": "leg_budget_shrunk", "leg": k, "steps": leg_steps,
                 "spent": spent_steps, "remaining": BUDGET_STEPS - spent_steps,
                 "note": "所有既成新步与失败烧步逐次扣减硬预算"})
        if leg_steps == 0:
            log({"event": "budget_exhausted", "leg": k, "spent": spent_steps})
            break
        seed_k = 281_000 + 1_000 * (k - 1)      # 唯一定义点(cmd 与日志同源)
        probes = reachable_probes(nt_chain, leg_steps)
        if len(probes) < 2:
            log({"event": "calib_trimmed", "leg": k, "probes": probes,
                 "note": "短腿探针不可达部分裁掉(预注册;probes_ok 按余量裁决)"})
        attempts[k] = attempts.get(k, 0) + 1
        run_name = f"v28-leg{k}"
        log({"event": "leg_start", "leg": k, "attempt": attempts[k], "beta": BETA,
             "steps": leg_steps, "seed": seed_k, "resume_from": prev_model,
             "probes": probes})
        res = run_leg(k, prev_model, leg_steps, probes, run_name, attempts[k], seed_k)
        train_secs += res["dt_sec"]
        expected = nt_chain + leg_steps

        # ---- 崩溃互锁(先于一切裁决;计数 = SB3 真链 ±2048 slack 覆盖 status 滞后)----
        clean = (res["rc"] == 0 and res["model_fresh"]
                 and res["global_steps"] == expected)
        sampled = observed_attempt_steps(nt_chain, res, leg_steps)
        if not clean:
            partial = sampled
            charged = failed_attempt_charge(partial, leg_steps)
            spent_steps += charged
            burned += charged
            log({"event": "leg_crash", "leg": k, "attempt": attempts[k],
                 "rc": res["rc"], "global_steps": res["global_steps"],
                 "burned_observed": partial, "burned_charged": charged,
                 "burned_total": burned,
                 "spent_total": spent_steps,
                 "note": "原配置重跑;烧步实时扣后续配额;收官计数不计不清零"})
            ensure_retry_budget(spent_steps, cap)
            if attempts[k] >= 4:  # 陈旧 calib/sentinel 由下次尝试进场清桌,此处无须轮转
                stop_reason = (f"腿 {k} 连崩 {attempts[k]} 次——训练止步(重试上限 4 系"
                               "运维自护护栏、非预注册闸门)")
                log({"event": "crash_halt", "why": stop_reason})
                attention(stop_reason)
                raise OperationalFailure(stop_reason)
            continue
        require(sampled == leg_steps, "干净收官的观测步数与配额不一致")
        spent_steps += sampled
        nt_chain = expected      # 真链推进(status 滞后不入链)

        # ---- G-绿洲(仅腿 1;3.0M 整点保证哨兵行存在,无行即失败——v26 静默跳过事故的修正)----
        if k == 1:
            sent = RUNS / run_name / "sentinel.jsonl"
            lines = []
            if sent.exists():
                for l in sent.read_text().splitlines():
                    if '"sentinel": "v23"' in l:
                        try:
                            lines.append(json.loads(l))
                        except Exception:
                            pass  # 半截行(进程被杀于写入中)跳过,不许炸死驱动
            if not lines:
                stop_reason = "G-绿洲失败:无哨兵行(v26 曾静默跳过,v28 硬性要求)"
                log({"event": "STOP", "why": stop_reason})
                attention(stop_reason)
                raise OperationalFailure(stop_reason)
            last = lines[-1]
            oasis_ok = last.get("dry", 1) == 0 and last.get("ff_dry", 0) > 0
            log({"event": "g_oasis", "dry": last.get("dry"), "ff_dry": last.get("ff_dry"),
                 "fresh": last.get("fresh"), "ok": oasis_ok})
            if not oasis_ok:
                stop_reason = "G-绿洲失败:学习窗含 dry 或 ff_dry=0"
                log({"event": "STOP", "why": stop_reason})
                attention(stop_reason)
                raise OperationalFailure(stop_reason)

        # ---- G-CAL 接线闸(每腿;只记不裁,tripped 位入账不裁决)----
        calib_p = RUNS / run_name / "calib.jsonl"
        recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                if calib_p.exists() else [])
        probes_ok = (all(any(p <= r["step"] < p + QUANTUM and r["g_ce"] > 0
                             and r["distill_ce"] > 0 for r in recs)
                         for p in probes) if probes else None)
        probe_status = ("SKIPPED" if probes_ok is None
                        else "PASS" if probes_ok else "FAIL")
        log({"event": "g_cal", "leg": k, "records": [
                {kk: r.get(kk) for kk in ("step", "g_pg", "g_ce", "teacher_diverge", "tripped")}
                for r in recs], "probes_ok": probes_ok, "probe_status": probe_status,
             "record_only": True})
        if probes_ok is False:
            stop_reason = f"G-CAL 接线失败(腿 {k} 双探针未见 ce/g_ce>0)——人工介入"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            raise OperationalFailure(stop_reason)

        # ---- 腿考 + 宽度探针(纯后处理)----
        r = exam_retry(res["model"], f"v28-leg{k}", "7000-7015", f"腿 {k} 考试")
        if r is None:
            stop_reason = f"腿 {k} 考试连败 2 次——人工验尸"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            raise OperationalFailure(stop_reason)
        agg, rows = r
        score = round(agg["ret_mean"], 1)
        anchor_by_seed = {r["seed"]: r["ret"]
                          for r in read_comparable_anchor()["rows"]}
        bw = breadth_wins(rows, anchor_by_seed, 7000, 7015)
        if candidate_probe_eligible(probes_ok):
            leg_models[k] = (score, str(res["model"]), bw)
        else:
            log({"event": "candidate_ineligible", "leg": k,
                 "why": "G-CAL probes SKIPPED；该模型不得进入候选池"})
        log({"event": "leg_exam", "leg": k, "beta": BETA, "score": score,
             "died": agg["died"], "diverge": agg.get("script_divergence_rate"),
             "breadth16": bw, "sha": agg["_sha"], "model_sha": sha16(res["model"]),
             "global_steps": res["global_steps"], "nt_chain": nt_chain})

        # ---- 绊线(硬绊优先;皆 break 进 G3——预算保护非惩罚,已训腿保留候选资格)----
        if score < HARD_LINE:
            log({"event": "HARD_TRIP", "leg": k, "score": score,
                 "why": f"< {HARD_LINE},训练永久终止,已完成腿照常进入 G3 候选池"})
            attention(f"硬绊:腿 {k} = {score}")
            break
        consec_low = consec_low + 1 if score < finish_line else 0
        if consec_low >= 2:
            log({"event": "early_finish", "leg": k, "score": score,
                 "why": f"连续 {consec_low} 条干净腿 < {finish_line}——提前收官进 G3"
                        "(预算保护,非惩罚)"})
            attention(f"提前收官于腿 {k}")
            break
        if res["dt_sec"] > 7200:
            log({"event": "sanity_finish", "leg": k, "dt_sec": res["dt_sec"],
                 "why": "腿墙钟 >2h——健全性收官进 G3(v26 裸 STOP 之修正)"})
            attention(f"健全性收官:腿 {k} 墙钟 {res['dt_sec']}s")
            break
        prev_model = str(res["model"])
        k += 1

    # ---- G3:均值 top-2 ∪ 宽度 top-1(去重 ≤3;起点本尊不入池)----
    if not leg_models:
        why = "无任何接线 PASS 的完成腿"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    by_mean = sorted(leg_models.items(), key=lambda kv: (-kv[1][0], kv[0]))
    by_breadth = sorted(leg_models.items(), key=lambda kv: (-kv[1][2], -kv[1][0], kv[0]))
    cand_legs = []
    for kk, _ in by_mean[:2] + [by_breadth[0]]:
        if kk not in cand_legs:
            cand_legs.append(kk)
    breadth_only = {by_breadth[0][0]} - {kk for kk, _ in by_mean[:2]}
    log({"event": "g3_candidates",
         "cands": [(kk, leg_models[kk][0], leg_models[kk][2]) for kk in cand_legs],
         "breadth_only": sorted(breadth_only)})

    finals = []
    for kk in cand_legs:
        sc16, mp, bw16 = leg_models[kk]
        r = exam_retry(pathlib.Path(mp), f"v28-G3-leg{kk}", "7000-7031", f"G3 腿 {kk}")
        if r is None:
            stop_reason = f"G3 满32考试连败 2 次(腿 {kk})——人工验尸"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            raise OperationalFailure(stop_reason)
        agg, rows = r
        anchor_by_seed = {row["seed"]: row["ret"]
                          for row in read_comparable_anchor()["rows"]}
        by_seed = {row["seed"]: row["ret"] for row in rows}
        require(len(rows) == len(by_seed), f"G3 腿 {kk} 含重复种子")
        require(set(by_seed) == set(range(7000, 7032)), f"G3 腿 {kk} 种子集合异常")
        diffs = [by_seed[s] - anchor_by_seed[s] for s in range(7000, 7032)]
        mean_diff = sum(diffs) / 32
        wins = sum(d > 0 for d in diffs)
        back16 = breadth_wins(rows, anchor_by_seed, 7016, 7031)
        void = agg["override_rate"] >= R4["override_void"]
        r4_ok = (agg["farm_descend_rate"] <= R4["farm_descend_rate"]
                 and agg["override_rate"] < R4["override_sentinel"]
                 and agg["cap_rate"] < R4["cap_rate"]
                 and R4["farm_tau_lo"] <= agg["farm_tau_mean"] <= R4["farm_tau_hi"])
        qual_ok = agg["ret_mean"] >= G3_MEAN and agg["died"] <= G3_DEATHS and r4_ok and not void
        sideline_ok = (kk not in breadth_only) or (back16 >= SIDELINE_BACK16)
        launch = qual_ok and mean_diff >= 4.0 and wins >= 18 and sideline_ok
        finals.append({"leg": kk, "mean": agg["ret_mean"], "died": agg["died"],
                       "mean_diff": mean_diff, "wins": wins, "back16": back16,
                       "void": void, "qual_ok": qual_ok, "sideline_ok": sideline_ok,
                       "launch": launch, "diverge": agg.get("script_divergence_rate"),
                       "model": mp, "_sha": agg["_sha"]})
        log({"event": "g3_full32", "leg": kk, "mean": agg["ret_mean"], "died": agg["died"],
             "r4_ok": r4_ok, "data_void": void, "qualified": qual_ok,
             "mean_diff": round(mean_diff, 2), "wins": wins, "back16": back16,
             "sideline_ok": sideline_ok, "launch": launch,
             "diverge": agg.get("script_divergence_rate"),
             "override": agg["override_rate"], "descend_rate": agg["farm_descend_rate"],
             "tau": agg["farm_tau_mean"], "depth_median": agg.get("depth_median")})

    # ---- 发射裁决 ----
    launchers = [f for f in finals if f["launch"]]
    if launchers:
        launchers.sort(key=lambda f: -f["mean"])
        band = [f for f in launchers if launchers[0]["mean"] - f["mean"] <= 0.05]
        w = sorted(band, key=lambda f: (-f["wins"], f["leg"]))[0]
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker "
                      f"{pathlib.Path(w['model']).with_suffix('')} --seeds 9000-9031 "
                      f"--tag v28-golden --board")
        log({"event": "GOLDEN_AUTHORIZED", "leg": w["leg"], "probe32_mean": w["mean"],
             "died": w["died"], "wins": w["wins"], "mean_diff": round(w["mean_diff"], 2),
             "diverge": w["diverge"], "model": w["model"],
             "model_sha": sha16(pathlib.Path(w["model"])), "full32_sha": w["_sha"],
             "golden_cmd": golden_cmd,
             "note": "金牌由值夜者手启,单臂一次;开牌后须回写 golden_result 事件"})
        attention(f"金牌待手启:腿 {w['leg']}(命令见 ledger)")
        return

    # ---- 不发射:三键穷尽分派(面板 blocker 修正)----
    nonvoid = [f for f in finals if not f["void"]]
    if not nonvoid:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "全候选数据作废——无胜者,资格失败(功效外)",
             "finals": [{k2: (round(v, 3) if isinstance(v, float) else v)
                         for k2, v in f.items() if k2 != "model"} for f in finals]})
        attention("判决:全候选作废")
        return
    wv = sorted(nonvoid, key=lambda f: (-f["mean"], f["leg"]))[0]
    if not wv["qual_ok"]:
        verdict = ("资格失败,宽度考题未答(功效外)——拦截原因="
                   + ("死数" if wv["died"] > G3_DEATHS else "哨兵/均值资格"))
    elif wv["wins"] >= 18 and wv["mean_diff"] >= 4.0 and not wv["sideline_ok"]:
        verdict = (f"宽度通道候选未过出样本副线(后16 {wv['back16']}/16 < "
                   f"{SIDELINE_BACK16})——不烧牌,选择膨胀防线生效")
    elif wv["wins"] >= 18 and wv["mean_diff"] < 4.0:
        verdict = "宽度达标而幅度未达——点估宽度改进,不烧牌,留工作站"
    elif wv["mean"] >= intrinsic_line:
        verdict = (f"宽度病确认内禀(功效内):均值保持/超越起点 {intrinsic_line} "
                   "而宽度未达——欠训假说否定,机制处方(锚随王走/课程采样)升格工作站")
    elif wv["mean"] >= PLATEAU_LINE:
        verdict = (f"续航无均值增益([{PLATEAU_LINE},{intrinsic_line}) 档),"
                   "宽度考题未答——不判内禀不判退化")
    else:
        verdict = "续航退化(<103),leg-6 为该配方局部峰"
    log({"event": "VERDICT_PATH", "golden_authorized": False, "verdict": verdict,
         "verdict_winner_leg": wv["leg"], "winner_mean": wv["mean"],
         "winner_wins": wv["wins"], "winner_mean_diff": round(wv["mean_diff"], 2),
         "finals": [{k2: (round(v, 3) if isinstance(v, float) else v)
                     for k2, v in f.items() if k2 != "model"} for f in finals]})
    attention(f"判决(不发射):{verdict}")


if __name__ == "__main__":
    main()
