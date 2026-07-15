"""v32「喝药主权」驱动(docs/PREREG-v32-喝药主权.md 条款唯一执行者;
run_v31_neweducation.py 骨架 × run_v30_relay.py 腿谱 定向融合)。

克隆差异表(PREREG-v32 逐条):
- 预检序列:G0 两式(恒等重放+幽灵)→ bc_worker 重生成(R1 闸)→
  KING_SD 再生+parity → in-case refs 两发 + 对 v31 参照逐种子位级对账
  (REF_BITEQ,取代 R 走廊健全闸)→ 双腿
- 双腿同种子对照(唯一变量 = --no-drink-sovereignty),v30 腿谱逐字
  (lr 3e-4 / ent .005 / M29 治下 / β=0.015625 / skip-dry / 自锚覆写),
  皆直接 resume 王 zip(legacy 口第二注册用例);nt 闸 3,997,696
- R2 锚消费:仅全文 sha + 锚桥条款(协议已变,全式复验注定失配——
  throne/script 数字经桥证对表,strict json 读取)
- R32-主判(sov×M29 vs ref-science 三档)+ R32-拆分(sov−ctrl)+
  a12 仪表(worker_action_hist["12"]/局)
金牌不在此发射(值守手启)。账本:train/runs/v32/gate_ledger.jsonl。
用法:.venv/bin/python train/run_v32_sovereign.py
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

from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           strict_json_loads, verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V32 = RUNS / "v32"
V32.mkdir(parents=True, exist_ok=True)
LEDGER = V32 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

KING_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
KING_ZIP_SHA = "2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42"
KING_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
KING_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
H_NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
BC_SD = RUNS / "bc-worker" / "policy_sd.pt"
KING_SD = V32 / "king_anchor_sd.pt"
TRAJ_BASELINE = ROOT / "docs" / "assets" / "g0v32_traj_baseline.json"
TRAJ_BASELINE_SHA = "2f567d7abfc5b4714cac4ff3236b09ab67e22485d2b05b52d9dc956692a26d6f"
THRONE_BASELINE = ROOT / "docs" / "assets" / "g0v32_traj_baseline_throne.json"
THRONE_BASELINE_SHA = "e235e96972f8f2eeaaebec2bc8970c3e3f77b715f26c364250c42d8d37131cfc"
ZEROFLIP_REPORT = ROOT / "train" / "runs" / "probe-zeroflip" / "report.json"
ZEROFLIP_SHA = "b2a2062304b081621e25d5b348e2965bf2f6fb375ed04694462ba1d8b7f6e31b"

# v31 参照档(REF_BITEQ 位级对账目标;全文 sha 钉死)
V31_REF = {
    "launch": (EVAL / "v31-ref-launch.json",
               "df17db995661c3994215c614ac4beeb6e4670bd1527d51a1198d69d960e01541"),
    "science": (EVAL / "v31-ref-science.json",
                "d842a8fa75c3b9234f4197a3f9fdc41458e8efeacf1d878934c349cf533c2489"),
}
# R2 锚(桥条款消费:strict json + 全文 sha;协议已变禁全式复验)
R2_BRIDGE = {
    "throne": (EVAL / "r2-throne.json",
               "2324648a416cbc0cb858858b006b5f797ee9c63422f13cb86c86462948f28c63"),
    "script": (EVAL / "r2-script.json",
               "71c298e05b6bf19ea94ced26c68b67b9e05303f22637357b610b15c6fb21a7f7"),
}

LEG_STEPS = 499_712
NT_TARGET = 3_997_696          # = king 3,497,984 + 499,712(增量语义)
BETA = 0.015625
SEED = 303_000
ABANDON_FRAC = (75.0, 112.4)
FLOOR_FRAC = (85.0, 92.0)
PAIRED_DIFF = 4.0
PAIRED_WINS = 18
DEATHS_MAX = 6
SURV_DIED_LINE = 3             # R32-主判:ref-science died 基线(v31 实测)
SURV_MEAN_BAND = -2.0          # 均值不降口径:配对均差 ≥ −2
A12_USE_LINE = 0.1             # "给了主权不用"档:a12/局 < 0.1
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 3

LEGS = {"v32-sov": [], "v32-ctrl": ["--no-drink-sovereignty"]}
CASE_RT: dict | None = None


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V32 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OperationalFailure(message)


def runtime_five(snapshot) -> dict:
    rt = snapshot["runtime"]
    return {"bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
            "game_data": rt["content"]["game_data"]["sha256"],
            "assets": rt["content"]["assets"]["sha256"],
            "protocol": rt["python_protocol"]["sha256"]}


def assert_case_runtime(snapshot, where: str):
    require(CASE_RT is not None, "CASE_RUNTIME 未落定")
    current = runtime_five(snapshot)
    if current != CASE_RT:
        log({"event": "CASE_HALT_RUNTIME_DRIFT", "where": where,
             "case": CASE_RT, "current": current})
        attention(f"案级运行时漂移({where}),停机呈报")
        raise OperationalFailure(f"案级运行时漂移({where})")


def run(cmd, logfile, timeout) -> int:
    with open(V32 / logfile, "w") as lf:
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


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def read_bridge(name: str) -> float:
    path, expected = R2_BRIDGE[name]
    require(path.is_file() and sha256(path) == expected,
            f"R2 桥档 sha 漂移/缺失:{name}")
    return float(strict_json_loads(path.read_bytes())["agg"]["ret_mean"])


def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"exam:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    d["agg"]["_sha"] = sha16(out)
    return d


def validate_adopted(tag, worker, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    lo, hi = (int(x) for x in seeds.split("-", 1))
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"adopt:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag,
                                      seeds=list(range(lo, hi + 1)))
    d = read_eval_archive(out, **expected)
    verify_eval_identity(snapshot, ROOT)
    d["agg"]["_sha"] = sha16(out)
    return d


def exam_or_adopt(events, worker, tag, seeds, manager_npz=None):
    """考发终局条款(v31 参照终局推广):落档+台账即冻结,续跑采信,禁重烧。"""
    out = EVAL / f"{tag}.json"
    prior = [e for e in events
             if e.get("event") == "exam_ok" and e.get("tag") == tag]
    if out.exists():
        require(bool(prior), f"{tag} 残档在位而台账无 exam_ok,停机呈报")
        d = validate_adopted(tag, worker, seeds, manager_npz)
        require(d["agg"]["_sha"] == prior[-1]["sha"],
                f"{tag} 档案与台账 exam_ok sha 失配")
        log({"event": "exam_adopted", "tag": tag, "sha": d["agg"]["_sha"]})
        return d
    require(not prior, f"{tag} 台账在册而档案缺失(REF_INVALID 型),停机呈报")
    return exam_retry(worker, tag, seeds, manager_npz)


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


def by_seed(rows, lo=7000, hi=7032) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "种子集合异常(含重复 seed)")
    require(set(m) == set(range(lo, hi)), f"种子集合异常(须为 {lo}-{hi - 1})")
    return m


def a12_per_ep(agg) -> float:
    hist = agg.get("worker_action_hist", {}) or {}
    return round(int(hist.get("12", 0)) / max(1, int(agg.get("n", 32))), 2)


def biteq(new_doc, ref_name: str):
    """REF_BITEQ:新参照与 v31 同名参照逐种子逐字段位级对账。"""
    path, expected = V31_REF[ref_name]
    require(path.is_file() and sha256(path) == expected,
            f"v31 参照档 sha 漂移:{ref_name}")
    old = strict_json_loads(path.read_bytes())
    old_rows, new_rows = old["rows"], new_doc["rows"]
    require(len(old_rows) == len(new_rows) == 32, "参照行数异常")
    diffs = []
    for o, n in zip(sorted(old_rows, key=lambda r: r["seed"]),
                    sorted(new_rows, key=lambda r: r["seed"])):
        if o != n:
            diffs.append(o["seed"])
    core = ("ret_mean", "ret_median", "died", "depth_median", "kills_mean",
            "farm_tau_mean", "override_rate", "cap_rate")
    agg_diff = [k for k in core if old["agg"].get(k) != new_doc["agg"].get(k)]
    if diffs or agg_diff:
        log({"event": "CASE_HALT_G0", "via": "REF_BITEQ", "ref": ref_name,
             "row_diff_seeds": diffs, "agg_diff": agg_diff})
        attention(f"REF_BITEQ 失败({ref_name}):E1 对现任栈非惰性,G0 失义")
        raise SystemExit(7)
    log({"event": "REF_BITEQ", "ref": ref_name, "rows": 32, "agg_core": "equal"})


def leg_starts(events, leg) -> int:
    return sum(1 for e in events
               if e.get("event") == "leg_start" and e.get("leg") == leg)


def read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    out = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OperationalFailure(f"台账第 {i} 行不可解析: {exc}") from exc
    return out


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def preflight(events):
    global CASE_RT
    require(PROTOCOL_VERSION == CALIBRATED_PROTOCOL_VERSION, "契约版本漂移")
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if l != "?? train/leaderboard-assembled-v3.md"]
    require(not dirty, f"W1: 工作树不净 {dirty}")
    head = git("rev-parse", "HEAD")
    for path in ("docs/PREREG-v32-喝药主权.md", "train/run_v32_sovereign.py"):
        touch = git("log", "-1", "--format=%H", "--", path)
        require(touch == head, f"W1: {path} 最后触碰 != HEAD")
    freezes = [e for e in events if e.get("event") == "FREEZE_SHA"]
    if freezes:
        require(freezes[-1]["sha"] == head, "W1: HEAD != 台账最后 FREEZE_SHA")
    require(KING_ZIP.is_file() and sha256(KING_ZIP) == KING_ZIP_SHA, "王 zip 漂移")
    require(sha256(KING_NPZ) == KING_NPZ_SHA, "王 npz 漂移")
    require(sha256(M29_NPZ) == M29_NPZ_SHA, "M29 npz 漂移")
    require(zip_steps(KING_ZIP) == 3_497_984, "王 zip 步数账异常")
    for name, (path, expected) in {**V31_REF, **R2_BRIDGE}.items():
        require(path.is_file() and sha256(path) == expected,
                f"档案钉死失配:{name}")
    require(ZEROFLIP_REPORT.is_file()
            and sha256(ZEROFLIP_REPORT) == ZEROFLIP_SHA,
            "零翻转探针报告钉死失配(锚桥证①落档件)")
    require(THRONE_BASELINE.is_file()
            and sha256(THRONE_BASELINE) == THRONE_BASELINE_SHA,
            "王座轨迹闭合基线钉死失配(锚桥证①升格件)")
    tags = (["v32-ref-launch", "v32-ref-science", "v32-golden"]
            + [f"{leg}-{s}" for leg in LEGS for s in ("s16", "full32", "m29")])
    for t in tags:
        if not any(e.get("event") == "exam_ok" and e.get("tag") == t
                   for e in events):
            require(not (EVAL / f"{t}.json").exists(),
                    f"W9: 目标档案已存在:{t}(重启协议:先 .void)")
    for leg in LEGS:
        if leg_starts(events, leg) == 0:
            require(not (RUNS / leg).exists(), f"运行目录残留:{leg}")
    snapshot = freeze_eval_identity(ROOT, str(KING_NPZ), None)
    CASE_RT = runtime_five(snapshot)
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt:
        require(prior_rt[0]["five"] == CASE_RT, "W10: 续跑运行时对账失配")
    else:
        log({"event": "CASE_RUNTIME", "five": CASE_RT})
    if not freezes:
        log({"event": "FREEZE_SHA", "sha": head})
    log({"event": "preflight_ok", "king_zip": KING_ZIP_SHA[:16],
         "bc_sd": sha16(BC_SD) if BC_SD.exists() else None,
         "throne_bridge": read_bridge("throne"),
         "script_bridge": read_bridge("script")})
    return head


def stage_done(events, name) -> bool:
    return any(e.get("event") == name for e in events)


def _main():
    events = read_ledger()
    if any(e.get("event") in ("VERDICT_PATH", "GOLDEN_AUTHORIZED")
           for e in events):
        print("案已结/待手启金评:幂等退出", flush=True)
        return
    preflight(events)

    # ---- S1 G0 两式(每次发车都重验——护栏而非一次性仪式)----
    require(TRAJ_BASELINE.is_file()
            and sha256(TRAJ_BASELINE) == TRAJ_BASELINE_SHA,
            "G0 基线档钉死失配(须为冻结 commit 入库件)")
    if not stage_done(events, "G0_BASELINE"):
        log({"event": "G0_BASELINE", "sha": TRAJ_BASELINE_SHA,
             "seeds": "7000-7015",
             "captured": "变更前(E1 施工前,stash 纯净码)"})
    if not stage_done(events, "E1_SCOPE_NOTE"):
        log({"event": "E1_SCOPE_NOTE",
             "note": "belt 前置系批文外裁量施工(④丙精神+14号先例+幽灵实测"
                     "必要性),无专项应答;呈报单列待追认"})
    require(run([PY, "train/probe_g0_traj.py", "replay"],
                f"g0-replay.{time.time_ns()}.log", 1_800) == 0,
            "G0-恒等 重放失败(king)")
    require(run([PY, "train/probe_g0_traj.py", "replay", "throne"],
                f"g0-replay-throne.{time.time_ns()}.log", 1_800) == 0,
            "G0-恒等 重放失败(throne,锚桥证①升格件)")
    log({"event": "G0_REPLAY", "verdict": "PASS", "seeds": "7000-7015",
         "workers": ["king", "throne"]})
    require(run([PY, "train/probe_g0_ghost.py"],
                f"g0-ghost.{time.time_ns()}.log", 1_800) == 0,
            "G0-幽灵 失败")
    log({"event": "G0_GHOST", "verdict": "PASS"})

    # ---- S2 bc_worker 重生成(R1 闸;新 impl 世界的教师回执)----
    if stage_done(events, "BC_REGEN"):
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "train"))
        from train_ppo import _validate_bc_report
        rec = _validate_bc_report(BC_SD, "data_gate", verify_replay=False)
        prior_bc = [e for e in events if e.get("event") == "BC_REGEN"][-1]
        require(rec["policy_sha256"] == prior_bc.get(
                    "policy_sha256", rec["policy_sha256"]),
                "BC_SD 跨发车身份链断裂")
    if not stage_done(events, "BC_REGEN"):
        require(run([PY, "train/bc_worker.py"],
                    f"bc-regen.{time.time_ns()}.log", 3_600) == 0,
                "bc_worker 重生成失败(R-W FAIL 型,全案停机)")
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "train"))
        from train_ppo import _validate_bc_report
        rec = _validate_bc_report(BC_SD, "data_gate", verify_replay=False)
        log({"event": "BC_REGEN", "held_out_top1": rec["held_out_top1"],
             "pairs": rec["pairs"], "policy_sha256": rec["policy_sha256"]})

    # ---- S3 KING_SD 再生 + 教师宣誓 ----
    if stage_done(events, "KING_SD_OK"):
        prior_ks = [e for e in events if e.get("event") == "KING_SD_OK"][-1]
        require(KING_SD.exists()
                and sha256(KING_SD) == prior_ks.get("sha256", ""),
                "KING_SD 跨发车身份链断裂")
    if not stage_done(events, "KING_SD_OK"):
        require(run([PY, "train/export_manager_sd.py", str(KING_ZIP),
                     str(KING_SD)], "export-king-sd.log", 600) == 0
                and KING_SD.exists(), "KING_SD 导出失败")
        require(run([PY, "train/check_teacher_parity.py", str(KING_SD),
                     str(KING_NPZ)], "parity-king.log", 600) == 0,
                "G-KL-W:王锚 sd 与 npz 宣誓失败")
        log({"event": "KING_SD_OK", "sha256": sha256(KING_SD)})

    # ---- S4 in-case refs + REF_BITEQ(全案级位级 G0)----
    refs = {}
    for name, manager in (("launch", None), ("science", str(M29_NPZ))):
        tag = f"v32-ref-{name}"
        already = any(e.get("event") == "exam_ok" and e.get("tag") == tag
                      for e in events)
        d = exam_or_adopt(events, str(KING_NPZ), tag, "7000-7031",
                          manager_npz=manager)
        require(d is not None, f"{tag} 参照考试连败")
        biteq(d, name)
        if not already:
            log({"event": "exam_ok", "tag": tag, "mean": d["agg"]["ret_mean"],
                 "died": d["agg"]["died"], "sha": d["agg"]["_sha"]})
            events.append({"event": "exam_ok", "tag": tag,
                           "sha": d["agg"]["_sha"]})
        refs[name] = d
    R = refs["launch"]["agg"]["ret_mean"]
    abandon = round(R * ABANDON_FRAC[0] / ABANDON_FRAC[1], 1)
    floor_repro = round(R * FLOOR_FRAC[0] / FLOOR_FRAC[1], 1)
    log({"event": "refs", "R": R, "abandon": abandon, "floor": floor_repro,
         "science_ref": refs["science"]["agg"]["ret_mean"],
         "science_died": refs["science"]["agg"]["died"]})
    ref_rows = by_seed(refs["launch"]["rows"])
    sci_rows = by_seed(refs["science"]["rows"])

    # ---- S5 双腿串行(同种子;唯一变量 = 主权旋钮)----
    npz = {}
    for leg, extra in LEGS.items():
        model_path = RUNS / leg / "model_final.zip"
        out = RUNS / leg / "policy.npz"
        if model_path.exists() and zip_steps(model_path) == NT_TARGET:
            if not out.exists():
                # 补导出通道:训练已达标仅 npz 缺——不记 leg_start 不烧额度
                require(run([PY, "train/export_worker_npz.py", str(model_path),
                             str(out)], f"export-{leg}.retry.log", 600) == 0
                        and out.exists(), f"{leg} 补导出失败,停机呈报")
                log({"event": "npz_exported", "leg": leg,
                     "sha256": sha256(out), "note": "补导出(非重点火)"})
            log({"event": "leg_skip_complete", "leg": leg})
            npz[leg] = str(out)
            continue
        require(leg_starts(events, leg) < 2, f"{leg} 点火额度耗尽(台账制)")
        cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo",
               "--gamma", "1.0", "--max-steps", "3000", "--n-steps", "512",
               "--num-envs", "4", "--lr", "3e-4", "--ent-coef", "0.005",
               "--seed", str(SEED), "--total-steps", str(LEG_STEPS),
               "--run-name", leg, "--distill-beta", str(BETA),
               "--teacher-sd", str(BC_SD), "--teacher-override", str(KING_SD),
               "--skip-dry", "--manager-npz", str(M29_NPZ),
               "--resume-from", str(KING_ZIP), "--allow-legacy-resume",
               "--calib-probes", "3747984,3947984",
               "--calib-record-only"] + extra
        ev = {"event": "leg_start", "leg": leg, "seed": SEED,
              "sovereignty": not extra}
        log(ev)
        events.append(ev)
        t0 = time.time()
        rc = run(cmd, f"train-{leg}.log", timeout=21_600)
        nt = zip_steps(model_path)
        log({"event": "leg_done", "leg": leg, "rc": rc, "nt_zip": nt,
             "dt_min": round((time.time() - t0) / 60, 1)})
        require(rc == 0 and nt == NT_TARGET,
                f"{leg} 训练未达标(rc={rc}, nt={nt}, 目标={NT_TARGET})"
                "——命题未考,本版不追加重训")
        require(run([PY, "train/export_worker_npz.py", str(model_path),
                     str(out)], f"export-{leg}.log", 600) == 0 and out.exists(),
                f"{leg} npz 导出失败")
        npz[leg] = str(out)
        log({"event": "npz_exported", "leg": leg, "sha256": sha256(out)})

    # ---- S6 考试:s16 → full32(H)→ full32(M29)----
    s16 = {}
    for leg in LEGS:
        tag = f"{leg}-s16"
        already = any(e.get("event") == "exam_ok" and e.get("tag") == tag
                      for e in events)
        d = exam_or_adopt(events, npz[leg], tag, "7000-7015")
        require(d is not None, f"{leg} 初筛连败")
        s16[leg] = d["agg"]["ret_mean"]
        if not already:
            log({"event": "exam_ok", "tag": tag, "mean": d["agg"]["ret_mean"],
                 "died": d["agg"]["died"], "sha": d["agg"]["_sha"]})
            events.append({"event": "exam_ok", "tag": tag,
                           "sha": d["agg"]["_sha"]})
    if all(v < abandon for v in s16.values()):
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"双腿初筛均 <{abandon}——训练失败或跨治下转移失败"
                    "(训练在 M29 治下,初筛在 H 治下),主权命题未考",
             "s16": s16, "R": R})
        attention("判决:训练失败,命题未考")
        return
    full, m29 = {}, {}
    for leg in LEGS:
        tag = f"{leg}-full32"
        already = any(e.get("event") == "exam_ok" and e.get("tag") == tag
                      for e in events)
        d = exam_or_adopt(events, npz[leg], tag, "7000-7031")
        require(d is not None, f"{leg} 满 32 连败")
        full[leg] = d
        a = d["agg"]
        if not already:
            events.append({"event": "exam_ok", "tag": tag, "sha": a["_sha"]})
            log({"event": "exam_ok", "tag": tag, "mean": a["ret_mean"],
                 "died": a["died"], "a12_per_ep": a12_per_ep(a),
                 "a13_per_ep": round(int(a.get("worker_action_hist", {})
                                         .get("13", 0)) / 32, 2),
                 "rwin_per_ep": round(sum(r["mode_seq"].count("R")
                                          for r in d["rows"]) / 32, 2),
                 "depth2": depth2_count(d["rows"]),
                 "dive": round(dive_per_ep(d["rows"]), 2),
                 "override": a["override_rate"], "sha": a["_sha"]})
        tag2 = f"{leg}-m29"
        already2 = any(e.get("event") == "exam_ok" and e.get("tag") == tag2
                       for e in events)
        d2 = exam_or_adopt(events, npz[leg], tag2, "7000-7031",
                           manager_npz=str(M29_NPZ))
        require(d2 is not None, f"{leg} M29 深潜考连败")
        m29[leg] = d2
        a2 = d2["agg"]
        if not already2:
            events.append({"event": "exam_ok", "tag": tag2, "sha": a2["_sha"]})
            log({"event": "exam_ok", "tag": tag2, "mean": a2["ret_mean"],
                 "died": a2["died"], "a12_per_ep": a12_per_ep(a2),
                 "a13_per_ep": round(int(a2.get("worker_action_hist", {})
                                         .get("13", 0)) / 32, 2),
                 "rwin_per_ep": round(sum(r["mode_seq"].count("R")
                                          for r in d2["rows"]) / 32, 2),
                 "depth2": depth2_count(d2["rows"]),
                 "sha": a2["_sha"]})

    # ---- R32-拆分(sov−ctrl,同种子逐对,只记不裁)----
    sov_rows = by_seed(full["v32-sov"]["rows"])
    ctrl_rows = by_seed(full["v32-ctrl"]["rows"])
    split = [sov_rows[s]["ret"] - ctrl_rows[s]["ret"] for s in sorted(sov_rows)]
    tri_ctrl = [ctrl_rows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    tri_sov = [sov_rows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    sp_mean = sum(split) / 32
    log({"event": "R32_SPLIT",
         "paired_sov_minus_ctrl": round(sp_mean, 2),
         "wins_sov": sum(x > 0 for x in split),
         "direction": ("方向未判定(|均差|<2,承 R30.6 判读先例)"
                       if abs(sp_mean) < 2.0 else "方向读数"),
         "triangle": {"ctrl_minus_ref": round(sum(tri_ctrl) / 32, 2),
                      "sov_minus_ref": round(sum(tri_sov) / 32, 2),
                      "note": "三角对账;禁以 sov−ctrl 单读数代言主权效应"},
         "died": {"sov": full["v32-sov"]["agg"]["died"],
                  "ctrl": full["v32-ctrl"]["agg"]["died"]},
         "a12": {"sov": a12_per_ep(full["v32-sov"]["agg"]),
                 "ctrl": a12_per_ep(full["v32-ctrl"]["agg"])},
         "口径": "同种子仅保证初始权重与 env 流等价,主权掩码自首个重归一化"
                 "分布即改变抽样,轨迹随即分岔——'唯一变量'系配置层陈述;"
                 "两腿步数账恒等(各+499712),免双口径"})

    # ---- R32-主判(sov×M29 vs ref-science;三档穷尽,独立于发射)----
    sm = m29["v32-sov"]
    sm_rows = by_seed(sm["rows"])
    surv_diffs = [sm_rows[s]["ret"] - sci_rows[s]["ret"] for s in sorted(sci_rows)]
    surv_mean = sum(surv_diffs) / 32
    sm_died = sm["agg"]["died"]
    sm_a12 = a12_per_ep(sm["agg"])
    ctrl_m29_died = m29["v32-ctrl"]["agg"]["died"]
    if (sm_a12 >= A12_USE_LINE and sm_died < SURV_DIED_LINE
            and surv_mean >= SURV_MEAN_BAND):
        if ctrl_m29_died < SURV_DIED_LINE:
            main_verdict = (f"存活改善与主权同现(重训连带候选,ctrl×M29 died "
                            f"{ctrl_m29_died} 亦<{SURV_DIED_LINE},禁用因果动词):"
                            f"died {sm_died} ∧ 均差 {surv_mean:.2f} ∧ "
                            f"a12/局 {sm_a12}≥{A12_USE_LINE}")
        else:
            main_verdict = (f"主权兑换存活:died {sm_died}<{SURV_DIED_LINE} ∧ "
                            f"配对均差 {surv_mean:.2f}≥{SURV_MEAN_BAND} ∧ "
                            f"a12/局 {sm_a12}≥{A12_USE_LINE}(ctrl×M29 died "
                            f"{ctrl_m29_died} 未同现)——批文成功线兑现")
    elif (sm_a12 < A12_USE_LINE and sm_died <= SURV_DIED_LINE
          and surv_mean >= SURV_MEAN_BAND):
        note2 = ("(低用量与存活改善并现,归因未证,死亡种子集合差随判词)"
                 if sm_a12 > 0 and sm_died < SURV_DIED_LINE else "")
        main_verdict = (f"给了主权不用:a12/局 {sm_a12}<{A12_USE_LINE} 且 "
                        f"died {sm_died}≤{SURV_DIED_LINE} ∧ 均差 "
                        f"{surv_mean:.2f}≥{SURV_MEAN_BAND}——④丙 无害无效,"
                        f"转 ④乙 议程{note2}")
    else:
        if sm_a12 >= A12_USE_LINE and sm_died <= SURV_DIED_LINE \
                and surv_mean >= SURV_MEAN_BAND:
            sub = "用而未兑现存活、未劣化(禁用'有害')" + (
                ",回报改善而存活未证" if surv_mean > 0 else "")
        elif sm_a12 >= A12_USE_LINE:
            sub = "用而劣化"
        else:
            sub = f"低用量劣化(重训/协议连带候选,a12={sm_a12})"
        main_verdict = (f"档3({sub}):a12/局 {sm_a12},died {sm_died}"
                        f"(线 {SURV_DIED_LINE}),均差 {surv_mean:.2f}"
                        f"(带 {SURV_MEAN_BAND})——触发解耦条款,"
                        "转捆绑评估案(批文'不行的话'义)")
    ref_dead = {r["seed"] for r in refs["science"]["rows"] if r["died"]}
    sov_dead = {r["seed"] for r in sm["rows"] if r["died"]}
    log({"event": "R32_MAIN", "verdict": main_verdict, "died": sm_died,
         "a12_per_ep": sm_a12, "paired_mean_vs_sciref": round(surv_mean, 2),
         "wins": sum(x > 0 for x in surv_diffs),
         "binom_note": "P(died≤2|n=32,p=3/32)≈0.41——died 档位系方向读数"
                       "非显著性证据(与 18/32 线注记同纪律)",
         "ctrl_m29": {"mean": m29["v32-ctrl"]["agg"]["ret_mean"],
                      "died": ctrl_m29_died,
                      "a12": a12_per_ep(m29["v32-ctrl"]["agg"])},
         "death_seed_sets": {"rescued": sorted(ref_dead - sov_dead),
                             "new_deaths": sorted(sov_dead - ref_dead)},
         "depth_vectors": {
             "sov_m29": [r["depth"] for r in sorted(sm["rows"],
                                                    key=lambda r: r["seed"])],
             "sciref": [r["depth"] for r in sorted(refs["science"]["rows"],
                                                   key=lambda r: r["seed"])]},
         "belt_econ": {"sov_m29_a13": round(int(sm["agg"].get(
                           "worker_action_hist", {}).get("13", 0)) / 32, 2),
                       "sciref_a13": round(int(refs["science"]["agg"].get(
                           "worker_action_hist", {}).get("13", 0)) / 32, 2)}})

    # ---- 资格/胜者/地板/发射(v31 梯子逐字)----
    def qual_of(d):
        a = d["agg"]
        dpe_ = dive_per_ep(d["rows"])
        void_ = (a["override_rate"] >= R4["override_void"]
                 or (dpe_ > 1 and a["died"] > 6))
        hard_ok_ = a["farm_descend_rate"] <= R4["descend"] and a["cap_rate"] < R4["cap"]
        override_ok_ = a["override_rate"] < R4["override_sentinel"]
        dual_ = dpe_ > 1 and hard_ok_ and not override_ok_
        ok_ = (a["died"] <= DEATHS_MAX and not void_
               and ((hard_ok_ and override_ok_) or dual_))
        return {"qual_ok": ok_, "void": void_, "dual_attr": dual_}

    quals = {n: qual_of(full[n]) for n in LEGS}
    cap_note = {}
    for n in LEGS:
        a = full[n]["agg"]
        if (not quals[n]["qual_ok"] and a["died"] <= DEATHS_MAX
                and a["cap_rate"] >= R4["cap"]):
            cap_note[n] = ("失格仅因 cap_rate——闸门语义漂移候选(主权饮药"
                           "延长存活推高撞帽,线未重标定,本案禁调线)")
    log({"event": "quals", **quals,
         **({"cap_drift_note": cap_note} if cap_note else {})})
    pool = [n for n in LEGS if quals[n]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "双腿资格失败——无胜者,换届命题未答(功效外)",
             "arms_full32": {n: full[n]["agg"]["ret_mean"] for n in LEGS},
             "R": R})
        attention("判决:双腿资格失败(R32 主判/拆分已入册)")
        return
    prelim = max(LEGS, key=lambda n: full[n]["agg"]["ret_mean"])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim]})
    ms = {n: full[n]["agg"]["ret_mean"] for n in pool}
    band = [n for n in pool if max(ms.values()) - ms[n] <= 0.05]
    if len(band) > 1:
        dmin = min(full[n]["agg"]["died"] for n in band)
        band = [n for n in band if full[n]["agg"]["died"] == dmin]
        winner = "v32-sov" if "v32-sov" in band else band[0]
    else:
        winner = band[0]
    W = full[winner]
    wa = W["agg"]
    wrows = by_seed(W["rows"])
    log({"event": "winner", "leg": winner, "mean": wa["ret_mean"],
         "died": wa["died"], "substituted": winner != prelim})
    if wa["ret_mean"] < floor_repro:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"胜者 {wa['ret_mean']} < {floor_repro}(=(85/92)×{R})"
                    "——未复现参考水平,换届命题未考",
             "arms_full32": {n: full[n]["agg"]["ret_mean"] for n in LEGS},
             "R": R, "main_verdict": main_verdict})
        attention("判决:未复现参考水平(R32 主判/拆分已入册)")
        return
    diffs = [wrows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    pd_mean = sum(diffs) / 32
    pd_wins = sum(d > 0 for d in diffs)
    launch = pd_mean >= PAIRED_DIFF and pd_wins >= PAIRED_WINS
    log({"event": "launch_check", "paired_mean": round(pd_mean, 2),
         "paired_wins": pd_wins, "died": wa["died"],
         "multiple_comparison": "18/32 线第 5 次开奖;P(≥18|p=.5)≈30%"
                                "(精确 0.2983;v29-v31 之 ≈43% 系"
                                "P(≥17) 笔误,随本案判词勘误注记)"})
    throne = read_bridge("throne")
    script_ref = read_bridge("script")
    p_hi = round(throne + 4.0, 1)
    P_LINE = (f"P线(锚桥对表:王座在位锚值 {throne} 系 r2-throne R2 重测值,"
              f"重测不改名分,王座头衔仍系在位组装体;脚本参照 {script_ref}):死>6→回退;"
              f"金≥{p_hi}且死≤4→P32-登基;∈({throne},{p_hi})且死≤4→点估;"
              f">{throne}且死5-6→持平(安全性);∈[{script_ref},{throne}]→持平;"
              f"<{script_ref}→回退。名分流转照 v31 附录成文版,两闸分行宣示。"
              "金池开牌史:金牌实开 v22/v23/v24 三次,R2 定锚五发系测量"
              "暴露,v31 未开;本案若发射为金牌第 4 次实开、金池累计第 9 次"
              "暴露,固定池偏置随判词")
    if launch:
        golden_cmd = (f"{PY} train/eval_assembled.py --worker {npz[winner]} "
                      f"--manager-npz {H_NPZ} --seeds 9000-9031 "
                      f"--tag v32-golden --board")
        log({"event": "GOLDEN_AUTHORIZED", "leg": winner,
             "probe32_mean": wa["ret_mean"], "died": wa["died"],
             "wins": pd_wins, "mean_diff": round(pd_mean, 2),
             "worker_npz_sha": sha16(npz[winner]), "full32_sha": wa["_sha"],
             "golden_cmd": golden_cmd, "p_line": P_LINE,
             "main_verdict": main_verdict,
             "note": "金牌值守手启单臂一次;知会出处=战役级纪律,无本案专项"
                     "应答(v31 条款);锚桥三证(零翻转落档/REF_BITEQ×2/"
                     "幽灵断言E)在册;王座锚桥系分布外推(限定随判词)"})
        attention(f"金牌待手启:{winner};R32 主判:{main_verdict}")
        return
    wins_note = (f"(宽度注记:赢 {pd_wins}/32 ≥14)" if pd_wins >= 14 else "")
    if pd_mean >= PAIRED_DIFF:
        verdict = f"均值增益 +{pd_mean:.2f} 而宽度未达(赢 {pd_wins}/32)——点估增益不烧牌"
    elif pd_mean >= 2.0:
        verdict = f"配对均差 {pd_mean:.2f} ∈[+2,+4)——探针级改进不烧牌{wins_note}"
    else:
        verdict = f"配对均差 {pd_mean:.2f} <+2——现任连任{wins_note}"
    log({"event": "VERDICT_PATH", "golden_authorized": False,
         "verdict": verdict, "main_verdict": main_verdict,
         "winner": winner, "arms_full32": {n: full[n]["agg"]["ret_mean"]
                                           for n in LEGS},
         "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins, "R": R})
    attention(f"判决(不发射):{verdict};R32 主判:{main_verdict}")


def main():
    try:
        with exclusive_lock(V32 / ".driver.lock", "v32 驱动"):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("运维失败:\n" + str(e))
        raise SystemExit(2) from e
    except SystemExit:
        raise
    except Exception as e:
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
