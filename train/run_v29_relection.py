"""v29「经理再教育」驱动(docs/PREREG-v29.md 条款唯一执行者;run_v25_election.py 定向改造)。

克隆差异表(PREREG-v29 D2 逐条对应):
- 班底:工人 = v28-worker-leg1(zip+npz);锚 = v28-G3-leg1.json(112.4,sha 钉死)
- M-warm 臂与 warm-sd 导出段物理删除;两臂 = fresh(ent .02)/explore(ent .08),160k 决策
- FLOOR_REPRO 85→103.4;标签 v29-*;配对按 seed 键 join + 集合断言
- v28 运维护栏全套:顶层异常兜底、训练 4h/评测 30min 超时、exam 拒覆写+.void 轮转
  +评测日志、preflight(锚 sha/工人在位/目标档案含 v29-golden 不存在)、NEEDS_ATTENTION
- 满 32 事件新增深度仪表:depth≥2 种子数 / DIVE/局 / 下楼奖金兑现/局
- 不发射档穷尽(v28 学费):资格外档+递补、点估增益档(≥+4 且 <18 赢)、探针级、连任
- GOLDEN_AUTHORIZED 带金评命令原文(含 --manager-npz)+ 双 sha
金牌不在此发射(值夜者手启,单臂一次)。账本:train/runs/v29/gate_ledger.jsonl。
用法:.venv/bin/python train/run_v29_relection.py
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

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V29 = RUNS / "v29"
V29.mkdir(parents=True, exist_ok=True)
LEDGER = V29 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

W_ZIP = str(ROOT / "train" / "models" / "v28-worker-leg1" / "model_final")  # eval 侧自动补 .zip
W_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
ARCHIVE = EVAL / "v28-G3-leg1.json"        # 参考行 112.4(现任 v22-H × 同一工人,逐种子)
ARCHIVE_SHA = "6fc6a44c7862424a"

STEPS = 160_000       # 4× v22-H 自身预算(时钟锚 27.5 决策/s ≈ 97 分钟/臂)
ABANDON = 75.0        # 提前放弃闸(双臂 16 种子均 < 此值)
PAIRED_DIFF = 4.0     # 发射线:配对均差 ≥ +4
PAIRED_WINS = 18      # 且配对赢 ≥ 18/32
DEATHS_MAX = 6
FLOOR_REPRO = 103.9   # 胜者 < 此值 → "重训未复现参考水平,命题未考"(=0.9239×112.4,严格沿 v25 比例 85/92)
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}

ARMS = {
    "v29-mfresh": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"],
    "v29-mexplore": ["--ent-coef", "0.08", "--lr", "3e-4", "--seed", "24"],
}


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V29 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def run(cmd, logfile, timeout) -> int:
    with open(V29 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)   # 连锅端:SubprocVecEnv 孙进程防孤儿
            proc.wait()
            return 124    # 挂死护栏:按崩溃/失败落账(运维护栏,非判决输入)


def zip_steps(p: pathlib.Path) -> int:
    """SB3 真链读数(面板 blocker 修正:160000 整除 256,status 节流计数必滞后)。"""
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    assert not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写"
    cmd = [PY, "train/eval_assembled.py", "--worker", worker, "--seeds", seeds, "--tag", tag]
    if manager_npz:
        cmd += ["--manager-npz", manager_npz]
    if run(cmd, f"exam-{tag}.{time.strftime('%H%M%S')}.log", timeout=1_800) != 0:
        if out.exists():    # 半截档案轮转,给重考让路
            out.rename(out.with_suffix(f".{time.strftime('%H%M%S')}.void"))
        return None
    d = json.loads(out.read_text())
    d["agg"]["_sha"] = sha16(out)
    return d


def exam_retry(worker, tag, seeds, manager_npz=None):
    d = exam(worker, tag, seeds, manager_npz)
    if d is None:
        log({"event": "exam_crash", "tag": tag, "note": "评测失败,按崩溃条款重考一次"})
        d = exam(worker, tag, seeds, manager_npz)
    return d


def dive_per_ep(rows) -> float:
    return sum(r["mode_seq"].count("D") for r in rows) / max(1, len(rows))


def depth2_count(rows) -> int:
    return sum(1 for r in rows if r["depth"] >= 2)


def bonus_per_ep(rows) -> float:
    # 下楼奖金兑现:depth=d 兑现 8×(1+2+…+(d−1));d≤1 为 0
    return sum(8 * sum(range(1, r["depth"])) for r in rows) / max(1, len(rows))


def by_seed(rows) -> dict:
    m = {r["seed"]: r for r in rows}
    assert set(m) == set(range(7000, 7032)), "种子集合异常(须为 7000-7031)"
    return m


def preflight():
    assert pathlib.Path(W_ZIP + ".zip").exists(), "工人 zip 缺失"
    assert W_NPZ.exists(), "工人 npz 缺失(发车日 parity 0/1000 导出件)"
    assert ARCHIVE.exists() and sha16(ARCHIVE) == ARCHIVE_SHA, "锚档案 sha 漂移"
    tags = ["v29-GA0", "v29-golden"] + [f"{a}-{s}" for a in ARMS for s in ("s16", "full32")]
    for t in tags:
        assert not (EVAL / f"{t}.json").exists(), f"目标档案已存在:{t}(重启协议:先 .void)"
    for a in ARMS:
        assert not (RUNS / a).exists(), f"运行目录残留:{a}(重启协议:先归档)"
    log({"event": "preflight_ok", "archive_sha": ARCHIVE_SHA,
         "worker_npz_sha": sha16(W_NPZ)})


def main():
    try:
        _main()
    except Exception as e:   # 条款兜底:任何未预期异常必须入册,不许无声死亡
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    log({"event": "start", "prereg": "docs/PREREG-v29.md", "steps": STEPS,
         "paired_line": PAIRED_DIFF, "wins_line": PAIRED_WINS, "floor": FLOOR_REPRO})
    ref = json.loads(ARCHIVE.read_text())
    ref_rows = by_seed(ref["rows"])

    # ---- G-A0:仪器回归(npz 工人 + 默认经理 ≡ 112.4 锚,32/32)----
    ga0 = exam_retry(str(W_NPZ), "v29-GA0", "7000-7031")
    if ga0 is None:
        log({"event": "STOP", "why": "G-A0 考试进程连败"})
        attention("G-A0 考试进程连败")
        return
    bad = [s for s, r in by_seed(ga0["rows"]).items()
           if (abs(r["ret"] - ref_rows[s]["ret"]) > 0.01
               or r["died"] != ref_rows[s]["died"]
               or r["mode_seq"] != ref_rows[s]["mode_seq"])]
    log({"event": "g_a0", "mismatch_seeds": bad, "n_ok": 32 - len(bad)})
    if bad:
        log({"event": "STOP", "why": "G-A0 位级回归失配——按预注册回退条款人工重锚"})
        attention(f"G-A0 失配种子:{bad}")
        return

    # ---- 两臂串行训练 ----
    npz = {}
    for name, extra in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo", "--gamma", "1.0",
               "--max-steps", "3000", "--n-steps", "64", "--num-envs", "4",
               "--total-steps", str(STEPS), "--worker-npz", str(W_NPZ),
               "--run-name", name] + extra
        log({"event": "arm_start", "arm": name, "cmd_extra": extra})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log", timeout=14_400)   # 4h 挂死护栏
        sp = RUNS / name / "status.json"
        try:
            steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        except Exception:
            steps = 0
        nt = zip_steps(RUNS / name / "model_final.zip")   # 达标闸唯一计步源(SB3 真链)
        log({"event": "arm_done", "arm": name, "rc": rc, "nt_zip": nt,
             "steps_status": steps, "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or nt != STEPS:
            log({"event": "STOP", "why": f"{name} 训练未达标(rc={rc}, nt_zip={nt}, "
                 f"status={steps})——命题未考,本版不追加重训(v25 条款)"})
            attention(f"{name} 训练未达标")
            return
        out = RUNS / name / "policy.npz"
        if run([PY, "train/export_manager_npz.py",
                str(RUNS / name / "model_final.zip"), str(out)],
               f"export-{name}.log", timeout=600) != 0:
            log({"event": "STOP", "why": f"{name} npz 导出/parity 失败"})
            attention(f"{name} G-A0m parity 失败")
            return
        npz[name] = str(out)
        log({"event": "g_a0m", "arm": name, "npz_sha": sha16(out)})

    # ---- 提前放弃闸(16 种子)----
    s16 = {}
    for name in ARMS:
        d = exam_retry(W_ZIP, f"{name}-s16", "7000-7015", manager_npz=npz[name])
        if d is None:
            log({"event": "STOP", "why": f"{name} 初筛考试连败"})
            attention(f"{name} 初筛考试连败")
            return
        s16[name] = d["agg"]["ret_mean"]
        log({"event": "screen16", "arm": name, "score": d["agg"]["ret_mean"],
             "died": d["agg"]["died"]})
    if all(v < ABANDON for v in s16.values()):
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"双臂初筛均 <{ABANDON}——训练失败,再教育命题未考(免满 32)"})
        attention("判决:训练失败,命题未考")
        return

    # ---- 两臂满 32(深度仪表随行)----
    full = {}
    for name in ARMS:
        d = exam_retry(W_ZIP, f"{name}-full32", "7000-7031", manager_npz=npz[name])
        if d is None:
            log({"event": "STOP", "why": f"{name} 满 32 考试连败"})
            attention(f"{name} 满 32 考试连败")
            return
        full[name] = d
        a = d["agg"]
        log({"event": "full32", "arm": name, "mean": a["ret_mean"], "died": a["died"],
             "dive_per_ep": round(dive_per_ep(d["rows"]), 2),
             "depth2_seeds": depth2_count(d["rows"]),
             "bonus_per_ep": round(bonus_per_ep(d["rows"]), 2),
             "depth_median": a.get("depth_median"), "tau": a["farm_tau_mean"],
             "override": a["override_rate"], "descend": a["farm_descend_rate"],
             "sha": a["_sha"]})
    fr = by_seed(full["v29-mfresh"]["rows"])
    ex = by_seed(full["v29-mexplore"]["rows"])
    r29_2 = sum(ex[s]["ret"] - fr[s]["ret"] for s in fr) / 32
    log({"event": "r29_2", "paired_explore_minus_fresh_mean": round(r29_2, 2)})

    # ---- 逐臂资格判定(面板 blocker 修正:胜者只从过资格臂中取)----
    def qual_of(d):
        a = d["agg"]
        dpe_ = dive_per_ep(d["rows"])
        void_ = (a["override_rate"] >= R4["override_void"]
                 or (dpe_ > 1 and a["died"] > 6))
        hard_ok_ = a["farm_descend_rate"] <= R4["descend"] and a["cap_rate"] < R4["cap"]
        override_ok_ = a["override_rate"] < R4["override_sentinel"]
        dual_ = dpe_ > 1 and hard_ok_ and not override_ok_   # v25 双归因条款
        ok_ = (a["died"] <= DEATHS_MAX and not void_
               and ((hard_ok_ and override_ok_) or dual_))
        return {"qual_ok": ok_, "void": void_, "dual_attr": dual_}

    quals = {n: qual_of(full[n]) for n in ARMS}
    log({"event": "quals", **{n: quals[n] for n in ARMS}})
    pool = [n for n in ARMS if quals[n]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "双臂资格失败(死/哨兵/作废)——无胜者,再教育命题未答(功效外)",
             "arms": {n: {"mean": full[n]["agg"]["ret_mean"],
                          "died": full[n]["agg"]["died"], **quals[n]} for n in ARMS}})
        attention("判决:双臂资格失败,无胜者(深度仪表已随 full32 事件入册)")
        return
    prelim = max(ARMS, key=lambda n: full[n]["agg"]["ret_mean"])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim],
             "note": "均值胜者资格拦截,按 D3-2 由过资格臂递补"})
    ms = {n: full[n]["agg"]["ret_mean"] for n in pool}
    band = [n for n in pool if max(ms.values()) - ms[n] <= 0.05]
    if len(band) > 1:
        dmin = min(full[n]["agg"]["died"] for n in band)
        band = [n for n in band if full[n]["agg"]["died"] == dmin]
        winner = "v29-mfresh" if "v29-mfresh" in band else band[0]
    else:
        winner = band[0]
    W = full[winner]
    wa = W["agg"]
    wrows = by_seed(W["rows"])
    dual_attr = quals[winner]["dual_attr"]
    log({"event": "winner", "arm": winner, "mean": wa["ret_mean"], "died": wa["died"],
         "substituted": winner != prelim})

    # ---- 深度副判(科学结论,不动王座;PREREG-v29 D3-7)----
    d2, dpe = depth2_count(W["rows"]), dive_per_ep(W["rows"])
    if d2 >= 12 and 0.5 <= dpe <= 3 and wa["died"] <= DEATHS_MAX:
        depth_verdict = "深度经济已学(≥12 图摸 2 层,DIVE 份额入带)"
    elif d2 <= 7:
        depth_verdict = "再教育未解锁深度(≤基线 7)"
    else:
        depth_verdict = (f"带外(depth2={d2}, dive={dpe:.2f}, died={wa['died']}),"
                         "入册不叙事")
    log({"event": "depth_verdict", "depth2_seeds": d2, "dive_per_ep": round(dpe, 2),
         "bonus_per_ep": round(bonus_per_ep(W["rows"]), 2), "verdict": depth_verdict,
         "note": "副判;王座与 Mark-I 认定另按 D3-6 与 ROADMAP 条款(防过度叙事)"})

    # ---- 复现地板 ----
    if wa["ret_mean"] < FLOOR_REPRO:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"胜者 {wa['ret_mean']} < {FLOOR_REPRO}——重训未复现参考水平,"
                    f"再教育命题未考;深度副判:{depth_verdict}"})
        attention("判决:未复现参考水平")
        return

    # ---- 发射判据(配对 vs 112.4 锚,按 seed 键;胜者资格已由 pool 保证)----
    diffs = [wrows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    pd_mean = sum(diffs) / 32
    pd_wins = sum(d > 0 for d in diffs)
    launch = pd_mean >= PAIRED_DIFF and pd_wins >= PAIRED_WINS
    log({"event": "launch_check", "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins,
         "died": wa["died"], "dive_per_ep": round(dpe, 2),
         "dual_attribution": dual_attr, "tau_note": wa["farm_tau_mean"]})

    P_LINE = ("P线速查(按序判定):死>6→回退;金≥101.2且死≤4→P29-登基;"
              "∈(97.2,101.2)且死≤4→点估增益王座不动;>97.2且死5-6→持平(安全性);"
              "∈[93.9,97.2]→持平;<93.9→回退")
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker {W_ZIP} "
                      f"--manager-npz {npz[winner]} --seeds 9000-9031 "
                      f"--tag v29-golden --board")
        dual_note = ("【双归因未裁】override 触线经双归因路径放行——烧牌前须人工完成"
                     "配比漂移 vs 真退化裁定并回写 dual_attr_ruling 事件,先裁后烧;"
                     if dual_attr else "")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wa["ret_mean"],
             "died": wa["died"], "wins": pd_wins, "mean_diff": round(pd_mean, 2),
             "manager_npz": npz[winner], "manager_npz_sha": sha16(npz[winner]),
             "full32_sha": wa["_sha"], "golden_cmd": golden_cmd, "p_line": P_LINE,
             "note": dual_note + "金牌由值夜者手启,单臂一次;败臂/未发射臂永不见"
                     " 9000 段;开牌后回写 golden_result 事件"})
        attention(dual_note + f"金牌待手启:{winner}(命令与 P 线速查见 ledger);"
                  f"深度副判:{depth_verdict}")
        return

    # ---- 不发射:穷尽分派(胜者已过资格;无胜者档在前;宽度移动注记随行)----
    wins_note = (f"(宽度移动注记:赢 {pd_wins}/32 ≥14,不改判档)"
                 if pd_wins >= 14 else "")
    if pd_mean >= PAIRED_DIFF and pd_wins < PAIRED_WINS:
        verdict = (f"均值增益 +{pd_mean:.2f} 而宽度未达(赢 {pd_wins}/32 < 18)"
                   "——点估增益,不烧牌,留工作站复赛")
    elif pd_mean >= 2.0:
        verdict = (f"配对均差 {pd_mean:.2f} ∈[+2,+4)——探针级改进,不烧牌"
                   f"(赢 {pd_wins}/32){wins_note}")
    else:
        verdict = (f"配对均差 {pd_mean:.2f} <+2——现任连任,再教育无增益(功效限定)"
                   f"(赢 {pd_wins}/32){wins_note}")
    log({"event": "VERDICT_PATH", "golden_authorized": False,
         "verdict": verdict, "depth_verdict": depth_verdict,
         "winner": winner, "winner_mean": wa["ret_mean"],
         "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins})
    attention(f"判决(不发射):{verdict};深度副判:{depth_verdict}")


if __name__ == "__main__":
    main()
