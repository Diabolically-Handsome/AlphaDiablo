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
import pathlib
import subprocess
import time
import traceback
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V28 = RUNS / "v28"
V28.mkdir(parents=True, exist_ok=True)
LEDGER = V28 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

LEG = 244 * 2048              # 499,712(v26 腿制原封)
N_LEGS = 8
BETA = 0.015625               # 恒定,永不撒手(教训十九 + v27「不可定居」)
HARD_LINE = 62.8
FINISH_LINE = 103.1           # =round(0.9×114.5,1),16 种子腿考口径(勿与满32口径 103 混读)
INTRINSIC_LINE = 108.2        # 满 32 口径:起点检查点本尊的满32均值(内禀档门槛)
PLATEAU_LINE = 103.0          # 满 32 口径:[103,108.2) 落「续航无均值增益」档
START = 2_998_272             # v26-leg6 zip num_timesteps(6×499,712;发车断言)
BASE_CKPT = RUNS / "v26-leg6" / "model_final.zip"
BC_SD = str(RUNS / "bc-worker" / "policy_sd.pt")
ANCHOR = EVAL / "v24-G3-leg7.json"
ANCHOR_SHA = "22d9442257d3a3c7"       # 预注册钉死;漂移即 STOP
BASELINE = EVAL / "v26-G3-leg6.json"
BASELINE_SHA = "24a905a7baf0f70a"     # 宽度基线 5/16、尸检对照的来源档案

G3_MEAN = 74.6
G3_DEATHS = 6
R4 = {"farm_descend_rate": 0.0204, "override_sentinel": 0.03, "override_void": 0.08,
      "cap_rate": 0.05, "farm_tau_lo": 27.8, "farm_tau_hi": 46.4}
SIDELINE_BACK16 = 8           # 仅宽度通道候选的出样本副线(零假设 P(≥8)≈16%)


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V28 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def preflight():
    assert BASE_CKPT.exists(), f"起点检查点缺失:{BASE_CKPT}"
    with zipfile.ZipFile(BASE_CKPT) as z:
        nt = json.loads(z.read("data"))["num_timesteps"]
    assert nt == START, f"START 断言失败:zip num_timesteps={nt} != {START}"
    st = json.loads((RUNS / "v26-leg6" / "status.json").read_text())["total_steps"]
    assert START - 2048 <= st <= START, f"status 计数器滞后越界:{st}"
    for p, s in ((ANCHOR, ANCHOR_SHA), (BASELINE, BASELINE_SHA)):
        assert p.exists() and sha16(p) == s, f"存档 sha 漂移:{p}(档案不可变性条款)"
    for k in range(1, N_LEGS + 1):
        for tag in (f"v28-leg{k}", f"v28-G3-leg{k}"):
            assert not (EVAL / f"{tag}.json").exists(), f"目标档案已存在:{tag}"
        assert not (RUNS / f"v28-leg{k}").exists(), (
            f"运行目录残留:v28-leg{k}(重启协议:整体归档后再发车)")
    assert not (EVAL / "v28-golden.json").exists(), "金评目标档案已存在(金牌至多一次)"
    log({"event": "preflight_ok", "start_nt": nt, "status_lag": START - st,
         "anchor_sha": ANCHOR_SHA, "baseline_sha": BASELINE_SHA})


def run_leg(k: int, resume_from: str, leg_steps: int, probes: list[int],
            run_name: str, attempt: int, seed_k: int) -> dict:
    run_dir = RUNS / run_name
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
    with open(V28 / f"{run_name}.try{attempt}.log", "w") as lf:  # per-attempt 尸检留档
        try:  # 挂死护栏 3h(>2h 健全线;运维护栏非判决输入):超时杀进程按崩溃互锁落账
            rc = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=10_800).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    dt = time.time() - t0
    sp = run_dir / "status.json"
    try:
        gsteps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
    except Exception:
        gsteps = 0                # 半截 status(写入中被杀):按崩溃条款落账
    return {"rc": rc, "dt_sec": round(dt), "global_steps": gsteps,
            "model": run_dir / "model_final.zip"}


def exam(model_path: pathlib.Path, tag: str, seeds: str) -> tuple[dict, list] | None:
    out = EVAL / f"{tag}.json"
    assert not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写"
    with open(V28 / f"{tag}.eval.{time.strftime('%H%M%S')}.log", "w") as lf:  # 评测侧尸检留档
        try:
            rc = subprocess.run([PY, "train/eval_assembled.py", "--worker",
                                 str(model_path).replace(".zip", ""), "--seeds", seeds,
                                 "--tag", tag], cwd=ROOT,
                                stdout=lf, stderr=subprocess.STDOUT,
                                timeout=1_800).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    if rc != 0:
        if out.exists():      # 半截档案轮转,给重考让路
            out.rename(out.with_suffix(f".{time.strftime('%H%M%S')}.void"))
        return None
    j = json.loads(out.read_text())
    agg = j["agg"]
    agg["_sha"] = sha16(out)
    return agg, j["rows"]


def exam_retry(model_path: pathlib.Path, tag: str, seeds: str, what: str):
    r = exam(model_path, tag, seeds)
    if r is None:
        log({"event": "exam_crash", "tag": tag, "note": f"{what}失败,按崩溃条款重考一次"})
        r = exam(model_path, tag, seeds)
    return r


def breadth_wins(rows: list, anchor_by_seed: dict, lo: int, hi: int) -> int:
    by_seed = {r["seed"]: r["ret"] for r in rows}
    return sum(1 for s in range(lo, hi + 1)
               if s in by_seed and by_seed[s] > anchor_by_seed[s])


def main():
    try:
        _main()
    except Exception as e:   # 条款⑧兜底:任何未预期异常必须入册,不许无声死亡
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    anchor = json.loads(ANCHOR.read_text())
    anchor_by_seed = {r["seed"]: r["ret"] for r in anchor["rows"]}
    assert set(anchor_by_seed) == set(range(7000, 7032)), "锚种子集合异常"
    log({"event": "start", "leg_steps": LEG, "beta_const": BETA, "hard": HARD_LINE,
         "finish": FINISH_LINE, "start_nt": START, "base": str(BASE_CKPT)})

    nt_chain = START            # SB3 真链计数(单一计数源)
    burned = 0                  # 崩溃部分步,从腿 8 扣,入新步硬预算 8×499,712
    train_secs = 0.0
    prev_model = str(BASE_CKPT)
    leg_models = {}             # k -> (score16, model_path, breadth16)
    attempts = {}
    consec_low = 0
    stop_reason = None
    k = 1
    while k <= N_LEGS:
        cap = LEG
        leg_steps = max(0, min(cap, LEG - burned)) if k == N_LEGS and burned else cap
        if k == N_LEGS and burned and leg_steps < LEG:
            log({"event": "leg8_shrunk", "steps": leg_steps, "note": "烧步扣减(预注册)"})
        if leg_steps == 0:
            log({"event": "leg8_skipped"})
            break
        seed_k = 281_000 + 1_000 * (k - 1)      # 唯一定义点(cmd 与日志同源)
        probes = [p for p in (nt_chain + 250_000, nt_chain + 450_000)
                  if p + 2048 <= nt_chain + leg_steps]
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
        clean = res["rc"] == 0 and res["global_steps"] >= expected - 2048
        if not clean:
            partial = max(0, res["global_steps"] - nt_chain)
            burned += partial
            log({"event": "leg_crash", "leg": k, "attempt": attempts[k],
                 "rc": res["rc"], "global_steps": res["global_steps"],
                 "burned_partial": partial, "burned_total": burned,
                 "note": "原配置重跑;烧步入新步硬预算(从腿 8 扣);收官计数不计不清零"})
            if attempts[k] >= 4:  # 陈旧 calib/sentinel 由下次尝试进场清桌,此处无须轮转
                stop_reason = (f"腿 {k} 连崩 {attempts[k]} 次——训练止步(重试上限 4 系"
                               "运维自护护栏、非预注册闸门);已完成腿按预算保护条款照常进 G3")
                log({"event": "crash_halt", "why": stop_reason})
                attention(stop_reason)
                break
            continue
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
                return
            last = lines[-1]
            oasis_ok = last.get("dry", 1) == 0 and last.get("ff_dry", 0) > 0
            log({"event": "g_oasis", "dry": last.get("dry"), "ff_dry": last.get("ff_dry"),
                 "fresh": last.get("fresh"), "ok": oasis_ok})
            if not oasis_ok:
                stop_reason = "G-绿洲失败:学习窗含 dry 或 ff_dry=0"
                log({"event": "STOP", "why": stop_reason})
                attention(stop_reason)
                return

        # ---- G-CAL 接线闸(每腿;只记不裁,tripped 位入账不裁决)----
        calib_p = RUNS / run_name / "calib.jsonl"
        recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                if calib_p.exists() else [])
        probes_ok = all(any(p <= r["step"] < p + 2048 and r["g_ce"] > 0
                            and r["distill_ce"] > 0 for r in recs)
                        for p in probes)
        log({"event": "g_cal", "leg": k, "records": [
                {kk: r[kk] for kk in ("step", "g_pg", "g_ce", "teacher_diverge", "tripped")}
                for r in recs], "probes_ok": probes_ok, "record_only": True})
        if probes and not probes_ok:
            stop_reason = f"G-CAL 接线失败(腿 {k} 双探针未见 ce/g_ce>0)——人工介入"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            return

        # ---- 腿考 + 宽度探针(纯后处理)----
        r = exam_retry(res["model"], f"v28-leg{k}", "7000-7015", f"腿 {k} 考试")
        if r is None:
            stop_reason = f"腿 {k} 考试连败 2 次——人工验尸"
            log({"event": "STOP", "why": stop_reason})
            attention(stop_reason)
            return
        agg, rows = r
        score = round(agg["ret_mean"], 1)
        bw = breadth_wins(rows, anchor_by_seed, 7000, 7015)
        leg_models[k] = (score, str(res["model"]), bw)
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
        consec_low = consec_low + 1 if score < FINISH_LINE else 0
        if consec_low >= 2:
            log({"event": "early_finish", "leg": k, "score": score,
                 "why": f"连续 {consec_low} 条干净腿 < {FINISH_LINE}——提前收官进 G3"
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
        log({"event": "STOP", "why": "无任何完成腿"})
        attention("无任何完成腿")
        return
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
            return
        agg, rows = r
        by_seed = {row["seed"]: row["ret"] for row in rows}
        assert set(by_seed) == set(range(7000, 7032)), f"G3 腿 {kk} 种子集合异常"
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
                      f"{w['model'].replace('.zip', '')} --seeds 9000-9031 "
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
    elif wv["mean"] >= INTRINSIC_LINE:
        verdict = ("宽度病确认内禀(功效内):均值保持/超越起点 108.2 而宽度未达"
                   "——欠训假说否定,机制处方(锚随王走/课程采样)升格工作站")
    elif wv["mean"] >= PLATEAU_LINE:
        verdict = "续航无均值增益([103,108.2) 档),宽度考题未答——不判内禀不判退化"
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
