"""v31「新世界再教育」驱动(docs/PREREG-v31-新世界再教育.md 条款唯一执行者;
run_v29_relection.py 定向改造)。

克隆差异表(PREREG-v31 逐条对应;面板 wf_8e3a60e2 两 BLOCK 修正在内):
- 时代:CALIBRATED_PROTOCOL_VERSION 2→3;R2 四锚显式路径 + 全文 sha +
  当前运行时可比性全式复验(freeze→expected→read→verify,v30 先例全式)
- train_ppo.py 增补:经理侧 options resume 口(冻结先决 RESUME_SMOKE +
  G-R31 已入台账);cont 臂 --total-steps 160000(增量语义),nt 闸 320000
- 仪器口径:全案评测 worker 一律 npz;G-A0 位级回归撤 → D3-0 参照健全闸
  (died≤8 ∧ R∈[100,170])+ CASE_RUNTIME 案级运行时对账(每考先决)
- 参照终局条款:refs 一经有效落档即案级冻结,续跑采信不重烧
- 裁决线现场推导:ABANDON=(75/112.4)×R、FLOOR=(85/92)×R(分数为正典)
- arm_done 含 steps_status 诊断字段(nt_zip 唯一计步源不变)
金牌不在此发射(值守手启,单臂一次)。账本:train/runs/v31/gate_ledger.jsonl。
用法:.venv/bin/python train/run_v31_neweducation.py
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

import numpy as np

from eval_contract import (PROTOCOL_VERSION, OperationalFailure, OutputReservationError,
                           exclusive_lock, expected_eval_identity,
                           freeze_eval_identity, read_eval_archive,
                           verify_eval_identity)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "train" / "runs"
V31 = RUNS / "v31"
V31.mkdir(parents=True, exist_ok=True)
LEDGER = V31 / "gate_ledger.jsonl"
EVAL = RUNS / "eval-assembled"

W_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
W_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
W24_NPZ = ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz"
W24_NPZ_SHA = "a31fa7c6b18b5c3593f4e1753d97aac9386689aa6ad8b158c526b673c57fbc2a"
M29_NPZ = ROOT / "train" / "models" / "v29-manager-mfresh" / "policy.npz"
M29_NPZ_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
M29_ZIP = RUNS / "v29-mfresh" / "model_final.zip"
M29_ZIP_SHA = "9d5820bfb951f6ba122b98ebad707f75a5471b4ce35dd09c3e8105775a3097ee"

# R2 ANCHOR_GRANT 消费(显式路径 + 全文 sha + 运行时可比性全式复验)
R2_ANCHORS = {
    "r2-launch": ("8ab6b51065105a6719c1348d561c5fe4b15f674f27869e6870c0bf813642dff1",
                  "W", None),
    "r2-science": ("8d6a05d3517a6481c756109158b446b3eb7bae8858d02841b42c43a972e273f5",
                   "W", "M29"),
    "r2-throne": ("2324648a416cbc0cb858858b006b5f797ee9c63422f13cb86c86462948f28c63",
                  "W24", None),
    "r2-script": ("71c298e05b6bf19ea94ced26c68b67b9e05303f22637357b610b15c6fb21a7f7",
                  "script", None),
}

ABANDON_FRAC = (75.0, 112.4)     # 正典 = 分数;0.6673 系展示舍入
FLOOR_FRAC = (85.0, 92.0)
PAIRED_DIFF = 4.0
PAIRED_WINS = 18
DEATHS_MAX = 6
REF_DIED_MAX = 8                 # D3-0 参照健全闸
R_CORRIDOR = (100.0, 170.0)      # D3-0(面板注册端点)
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 3

ARMS = {
    # cli_steps 系 --total-steps(resume 腿增量语义);nt_target 系达标闸
    "v31-mfresh": {
        "cli_steps": 160_000, "nt_target": 160_000,
        "extra": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"]},
    "v31-mcont": {
        "cli_steps": 160_000, "nt_target": 320_000,
        "extra": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "26",
                  "--resume-from", str(M29_ZIP), "--allow-legacy-resume"]},
}

CASE_RT: dict | None = None      # CASE_RUNTIME 案级运行时五 sha(preflight 落定)


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V31 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    """统一预检通道(面板 minor):失败一律 OperationalFailure → 退出码 2。"""
    if not condition:
        raise OperationalFailure(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v31 裁决线在 protocol-v3 世界标定(R2 锚);协议再升版须重锚")


def runtime_five(snapshot) -> dict:
    rt = snapshot["runtime"]
    return {"bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
            "game_data": rt["content"]["game_data"]["sha256"],
            "assets": rt["content"]["assets"]["sha256"],
            "protocol": rt["python_protocol"]["sha256"]}


def assert_case_runtime(snapshot, where: str):
    """案级运行时对账(R2 W10 之案内版):参照与臂考跨时可比之先决。"""
    require(CASE_RT is not None, "CASE_RUNTIME 未落定")
    current = runtime_five(snapshot)
    if current != CASE_RT:
        log({"event": "CASE_HALT_RUNTIME_DRIFT", "where": where,
             "case": CASE_RT, "current": current})
        attention(f"案级运行时漂移({where}),参照与臂考不可比,停机呈报")
        raise OperationalFailure(f"案级运行时漂移({where})")


def anchor_spec(code):
    if code == "W":
        return str(W_NPZ)
    if code == "W24":
        return str(W24_NPZ)
    if code == "M29":
        return str(M29_NPZ)
    return code            # "script"


def read_r2_anchor(tag: str) -> dict:
    """R2 锚消费全式:全文 sha + freeze→expected→read→verify(v30 先例)。"""
    path = EVAL / f"{tag}.json"
    expected_sha, wcode, mcode = R2_ANCHORS[tag]
    require(path.is_file() and sha256(path) == expected_sha,
            f"R2 锚 sha 漂移/缺失:{tag}")
    manager = anchor_spec(mcode) if mcode else None
    try:
        snapshot = freeze_eval_identity(ROOT, anchor_spec(wcode), manager)
        expected = expected_eval_identity(snapshot, tag=tag,
                                          seeds=range(9000, 9032))
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            f"R2 锚 {tag} 与当前运行时不可比,须另立重锚案(科学读数终局,"
            f"禁止重烧):{exc}") from exc
    return document


def run(cmd, logfile, timeout) -> int:
    with open(V31 / logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # 连锅端:SubprocVecEnv 孙进程防孤儿
            except ProcessLookupError:
                pass
            proc.wait()
            return 124    # 挂死护栏:按崩溃/失败落账(运维护栏,非判决输入)


def zip_steps(p: pathlib.Path) -> int:
    """SB3 真链读数(唯一计步源;status 节流计数必滞后)。"""
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
        return 0


def exam(worker, tag, seeds, manager_npz=None):
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    require(seed_values and lo >= 0, f"非法 seed 范围:{seeds}")
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"exam:{tag}")      # 案级对账先决,不按崩溃重考
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800) != 0:
        if out.exists():    # 半截档案轮转,给重考让路
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


def exam_retry(worker, tag, seeds, manager_npz=None):
    d = exam(worker, tag, seeds, manager_npz)
    if d is None:
        log({"event": "exam_crash", "tag": tag, "note": "评测失败,按崩溃条款重考一次"})
        d = exam(worker, tag, seeds, manager_npz)
    return d


def validate_ref_archive(tag, worker, manager_npz):
    """参照档案续跑采信复验(参照终局条款)。"""
    out = EVAL / f"{tag}.json"
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    assert_case_runtime(snapshot, f"ref-adopt:{tag}")
    expected = expected_eval_identity(snapshot, tag=tag,
                                      seeds=range(7000, 7032))
    d = read_eval_archive(out, **expected)
    verify_eval_identity(snapshot, ROOT)
    d["agg"]["_sha"] = sha16(out)
    return d


def ref_exam_or_adopt(events, tag, worker, manager_npz=None):
    """参照发:一经有效落档即案级冻结;续跑采信,禁 .void 重烧。"""
    out = EVAL / f"{tag}.json"
    on_ledger = [e for e in events
                 if e.get("event") == "ref_archive" and e.get("tag") == tag]
    if out.exists():
        require(bool(on_ledger),
                f"{tag} 案前残档在位而台账无 ref_archive 记录,停机呈报")
        try:
            d = validate_ref_archive(tag, worker, manager_npz)
        except OperationalFailure:
            raise
        except Exception as exc:
            log({"event": "REF_INVALID", "tag": tag,
                 "why": (str(exc).splitlines() or ["?"])[0]})
            attention(f"{tag} 参照档案复验不过(REF_INVALID),全案停机呈报")
            raise OperationalFailure(f"{tag} 参照档案复验不过") from exc
        require(d["agg"]["_sha"] == on_ledger[-1]["sha16"],
                f"{tag} 档案与台账 ref_archive sha 失配")
        log({"event": "ref_adopted", "tag": tag, "sha16": d["agg"]["_sha"]})
        return d
    require(not on_ledger, f"{tag} 台账在册而档案缺失(REF_INVALID),停机呈报")
    d = exam_retry(worker, tag, "7000-7031", manager_npz)
    if d is None:
        raise OperationalFailure(f"{tag} 参照考试连败(残档已 .void 封存;"
                                 "禁以金池 r2-launch 跨池替充基准)")
    log({"event": "ref_archive", "tag": tag, "sha16": d["agg"]["_sha"]})
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
    require(len(rows) == len(m), "种子集合异常(含重复 seed)")
    require(set(m) == set(range(7000, 7032)), "种子集合异常(须为 7000-7031)")
    return m


def depth_gauge(d) -> dict:
    a = d["agg"]
    return {"died": a["died"], "depth2_seeds": depth2_count(d["rows"]),
            "dive_per_ep": round(dive_per_ep(d["rows"]), 2),
            "bonus_per_ep": round(bonus_per_ep(d["rows"]), 2),
            "override": a["override_rate"], "descend": a["farm_descend_rate"],
            "cap": a["cap_rate"]}


def preflight(events):
    global CASE_RT
    require_calibrated_protocol()
    require(W_NPZ.exists() and sha256(W_NPZ) == W_NPZ_SHA,
            "工人 npz 缺失或 sha 漂移")
    require(W24_NPZ.exists() and sha256(W24_NPZ) == W24_NPZ_SHA,
            "v24-leg7 npz(王座锚身份件)缺失或 sha 漂移")
    require(M29_NPZ.exists() and sha256(M29_NPZ) == M29_NPZ_SHA,
            "M29 经理 npz(归档件)缺失或 sha 漂移")
    require(M29_ZIP.exists() and sha256(M29_ZIP) == M29_ZIP_SHA,
            "M29 检查点(cont 臂初始化)缺失或 sha 漂移")
    require(zip_steps(M29_ZIP) == 160_000, "M29 检查点步数账异常")
    anchors = {tag: read_r2_anchor(tag) for tag in R2_ANCHORS}   # 全式复验
    # G-R31:零训练加载导出 == 归档 npz(加载即本尊;每次发车重证)
    g31 = V31 / "g_r31.npz"
    if g31.exists():
        g31.unlink()
    require(run([PY, "train/export_manager_npz.py", str(M29_ZIP), str(g31)],
                f"g-r31.{time.time_ns()}.log", timeout=600) == 0 and g31.exists(),
            "G-R31 导出失败")
    a, b = np.load(g31), np.load(M29_NPZ)
    require(set(a.files) == set(b.files)
            and all(np.array_equal(a[k], b[k]) for k in a.files),
            "G-R31 位级失配:M29 zip 加载导出 != 归档 npz")
    log({"event": "G_R31", "bitwise": "6/6", "source_zip_sha16": sha16(M29_ZIP)})
    # 冻结先决在册断言(面板 BLOCK:冒烟不过 → 本案不冻结)
    require(any(e.get("event") == "RESUME_SMOKE" and e.get("rc") == 0
                for e in events), "RESUME_SMOKE 事件缺席(面板冻结先决)")
    for t in ["v31-golden"] + [f"{a_}-{s}" for a_ in ARMS for s in ("s16", "full32")]:
        require(not (EVAL / f"{t}.json").exists(),
                f"目标档案已存在:{t}(重启协议:先 .void)")
    for a_ in ARMS:
        require(not (RUNS / a_).exists(), f"运行目录残留:{a_}(重启协议:先归档)")
    # CASE_RUNTIME 案级运行时落定/对账
    snapshot = freeze_eval_identity(ROOT, str(W_NPZ), None)
    five = runtime_five(snapshot)
    prior_rt = [e for e in events if e.get("event") == "CASE_RUNTIME"]
    if prior_rt:
        require({k: prior_rt[0][k] for k in five} == five,
                "CASE_RUNTIME 续跑对账失配,停机呈报")
    else:
        log({"event": "CASE_RUNTIME", **five})
    CASE_RT = five
    # 现任身份钉死:freeze_eval_identity(默认经理)内建 DEFAULT_MANAGER_SHA256
    # 断言,且 r2-launch 全式复验同走默认经理——参照发经理 == 现任经理成立。
    log({"event": "preflight_ok", "worker_npz_sha": sha16(W_NPZ),
         "m29_npz_sha": sha16(M29_NPZ), "m29_zip_sha": sha16(M29_ZIP),
         "default_manager_sha16": snapshot["manager"]["sha256"][:16],
         "r2_anchor_sha16": {t: R2_ANCHORS[t][0][:16] for t in R2_ANCHORS}})
    return anchors


def main():
    try:
        with exclusive_lock(V31 / ".driver.lock", "v31 驱动"):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("运维失败:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:   # 条款兜底:任何未预期异常必须入册,不许无声死亡
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡:\n" + traceback.format_exc())
        raise


def _read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    events = []
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise OperationalFailure(f"台账第 {i} 行不可解析,停机呈报: {exc}") from exc
    return events


def _main():
    events = _read_ledger()
    anchors = preflight(events)
    launch_gold = anchors["r2-launch"]["agg"]["ret_mean"]
    throne = anchors["r2-throne"]["agg"]["ret_mean"]
    script_ref = anchors["r2-script"]["agg"]["ret_mean"]

    # ---- 参照两发(终局条款:落档即冻结,续跑采信)----
    ref_l = ref_exam_or_adopt(events, "v31-ref-launch", str(W_NPZ))
    ref_s = ref_exam_or_adopt(events, "v31-ref-science", str(W_NPZ),
                              manager_npz=str(M29_NPZ))
    R = ref_l["agg"]["ret_mean"]
    # D3-0 参照健全闸(G-A0 仪器闸的换世界等价替身;运维护栏,非科学裁决)
    ref_gauge = depth_gauge(ref_l)
    log({"event": "ref_sanity", "R": R, "corridor": R_CORRIDOR,
         "died": ref_l["agg"]["died"], "died_max": REF_DIED_MAX,
         "ref_launch_gauge": ref_gauge,
         "ref_science_gauge": depth_gauge(ref_s),
         "cross_pool_note": {"r2_launch_golden": launch_gold,
                             "口径": "探针池 vs 金池,只记不裁;133.9 在此仅作断路器输入"}})
    if not (R_CORRIDOR[0] <= R <= R_CORRIDOR[1]) or ref_l["agg"]["died"] > REF_DIED_MAX:
        log({"event": "CASE_HALT_REF_CORRIDOR", "R": R,
             "died": ref_l["agg"]["died"]})
        attention("探针池参照崩塌/仪器异常,裁决线现场推导失义——停机呈报,"
                  "不训臂、不烧金种子;复测须另案重冻结")
        raise OperationalFailure("D3-0 参照健全闸未过")
    abandon = round(R * ABANDON_FRAC[0] / ABANDON_FRAC[1], 1)
    floor_repro = round(R * FLOOR_FRAC[0] / FLOOR_FRAC[1], 1)
    log({"event": "refs", "ref_launch_mean": R,
         "ref_science_mean": ref_s["agg"]["ret_mean"],
         "abandon_line": f"{abandon} =(75/112.4)×{R}",
         "floor_repro": f"{floor_repro} =(85/92)×{R}",
         "ref_launch_sha": ref_l["agg"]["_sha"],
         "ref_science_sha": ref_s["agg"]["_sha"]})
    ref_rows = by_seed(ref_l["rows"])
    refsci_rows = by_seed(ref_s["rows"])

    # ---- 两臂串行训练 ----
    npz = {}
    for name, spec in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo", "--gamma", "1.0",
               "--max-steps", "3000", "--n-steps", "64", "--num-envs", "4",
               "--total-steps", str(spec["cli_steps"]), "--worker-npz", str(W_NPZ),
               "--run-name", name] + spec["extra"]
        log({"event": "arm_start", "arm": name, "cli_steps": spec["cli_steps"],
             "nt_target": spec["nt_target"], "cmd_extra": spec["extra"]})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log", timeout=14_400)   # 4h 挂死护栏
        sp = RUNS / name / "status.json"
        try:
            steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        except Exception:
            steps = 0
        nt = zip_steps(RUNS / name / "model_final.zip")   # 达标闸唯一计步源
        log({"event": "arm_done", "arm": name, "rc": rc, "nt_zip": nt,
             "steps_status": steps, "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or nt != spec["nt_target"]:
            why = (f"{name} 训练未达标(rc={rc}, nt_zip={nt}, "
                   f"nt_target={spec['nt_target']})——命题未考,"
                   "本版不追加重训(v25 条款)")
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        out = RUNS / name / "policy.npz"
        if run([PY, "train/export_manager_npz.py",
                str(RUNS / name / "model_final.zip"), str(out)],
               f"export-{name}.log", timeout=600) != 0 or not out.exists():
            why = f"{name} npz 导出/parity 失败"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        npz[name] = str(out)
        log({"event": "npz_exported", "arm": name, "npz_sha": sha16(out)})

    # ---- 提前放弃闸(16 种子)----
    s16 = {}
    for name in ARMS:
        d = exam_retry(str(W_NPZ), f"{name}-s16", "7000-7015", manager_npz=npz[name])
        if d is None:
            why = f"{name} 初筛考试连败"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        s16[name] = d["agg"]["ret_mean"]
        log({"event": "screen16", "arm": name, "score": d["agg"]["ret_mean"],
             "died": d["agg"]["died"]})
    if all(v < abandon for v in s16.values()):
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"双臂初筛均 <{abandon}(=(75/112.4)×{R})——训练失败,"
                    "换届命题未考(免满 32)",
             "s16": s16, "R": R,
             "science_note": "放弃闸路径下 D3-8 续训净变观察如实缺席"})
        attention("判决:训练失败,命题未考")
        return

    # ---- 两臂满 32(深度仪表随行)----
    full = {}
    for name in ARMS:
        d = exam_retry(str(W_NPZ), f"{name}-full32", "7000-7031", manager_npz=npz[name])
        if d is None:
            why = f"{name} 满 32 考试连败"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        full[name] = d
        a = d["agg"]
        log({"event": "full32", "arm": name, "mean": a["ret_mean"], "died": a["died"],
             "dive_per_ep": round(dive_per_ep(d["rows"]), 2),
             "depth2_seeds": depth2_count(d["rows"]),
             "bonus_per_ep": round(bonus_per_ep(d["rows"]), 2),
             "depth_median": a.get("depth_median"), "tau": a["farm_tau_mean"],
             "override": a["override_rate"], "descend": a["farm_descend_rate"],
             "sha": a["_sha"]})
    fr = by_seed(full["v31-mfresh"]["rows"])
    co = by_seed(full["v31-mcont"]["rows"])
    r31_2 = sum(co[s]["ret"] - fr[s]["ret"] for s in fr) / 32
    means = {n: full[n]["agg"]["ret_mean"] for n in ARMS}
    log({"event": "r31_2", "paired_cont_minus_fresh_mean": round(r31_2, 2),
         "arms_full32": means, "R": R,
         "口径": "v3 从零 vs v2 存量续训,整体路径对比,禁单因素归因;"
                "单次配对描述量,禁作显著性解读;步数账双口径义务:"
                "cont 总步数 2×(320k 累计)∧ 新世界步数相等(各 160k)"})

    # ---- 科学观察(D3-8):恒算 cont 对 ref-science(存量续训相对前身净变)----
    cont_sci = [co[s]["ret"] - refsci_rows[s]["ret"] for s in sorted(refsci_rows)]
    log({"event": "science_observation",
         "cont_vs_m29ref_paired_mean": round(sum(cont_sci) / 32, 2),
         "wins": sum(x > 0 for x in cont_sci),
         "口径": "只记不裁;存量续训相对其前身(M29 v3 探针读数)之净变"})

    # ---- 逐臂资格判定 ----
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
    log({"event": "quals", **{n: quals[n] for n in ARMS},
         "line_note": "死亡闸/作废线对深潜行为在 v3 系统性偏紧(线未重标定,"
                      "已知偏置);凡以此裁决之判词须并引 depth 分布与逐种子 died"})
    pool = [n for n in ARMS if quals[n]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "双臂资格失败(死/哨兵/作废)——无胜者,换届命题未答(功效外)",
             "arms": {n: {"mean": means[n], "died": full[n]["agg"]["died"],
                          **quals[n]} for n in ARMS},
             "r31_2": round(r31_2, 2), "R": R})
        attention("判决:双臂资格失败,无胜者(深度仪表已随 full32 事件入册)")
        return
    prelim = max(ARMS, key=lambda n: means[n])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim],
             "note": "均值胜者资格拦截,按 D3-2 由过资格臂递补"})
    ms = {n: means[n] for n in pool}
    band = [n for n in pool if max(ms.values()) - ms[n] <= 0.05]
    if len(band) > 1:
        dmin = min(full[n]["agg"]["died"] for n in band)
        band = [n for n in band if full[n]["agg"]["died"] == dmin]
        winner = "v31-mfresh" if "v31-mfresh" in band else band[0]
    else:
        winner = band[0]
    W = full[winner]
    wa = W["agg"]
    wrows = by_seed(W["rows"])
    dual_attr = quals[winner]["dual_attr"]
    log({"event": "winner", "arm": winner, "mean": wa["ret_mean"], "died": wa["died"],
         "substituted": winner != prelim,
         "residual_note": "胜者系 max-of-2 顺序统计量,发射一类错 ≈×2(残余#1)"})

    # ---- 深度副判(v29 D3-7 三档 + v31 现场基线条款)----
    d2, dpe = depth2_count(W["rows"]), dive_per_ep(W["rows"])
    field_d2 = ref_gauge["depth2_seeds"]
    if d2 >= 12 and 0.5 <= dpe <= 3 and wa["died"] <= DEATHS_MAX:
        if field_d2 >= 12:
            depth_verdict = (f"达 v2 承继线(≥12),但现场基线已在线上"
                             f"(ref-launch depth2={field_d2}),无鉴别力")
        else:
            depth_verdict = "深度经济已学(v3 校准注记:区分度弱化,不增叙事)"
    elif d2 <= 7:
        depth_verdict = f"未解锁深度(≤基线 7;现场基线 ref depth2={field_d2})"
    else:
        depth_verdict = (f"带外(depth2={d2}, dive={dpe:.2f}, died={wa['died']};"
                         f"现场基线 ref depth2={field_d2}),入册不叙事")
    log({"event": "depth_verdict", "depth2_seeds": d2, "dive_per_ep": round(dpe, 2),
         "bonus_per_ep": round(bonus_per_ep(W["rows"]), 2),
         "field_baseline_depth2": field_d2, "verdict": depth_verdict,
         "note": "副判;王座与 Mark-I 认定另按 D3-6 与 ROADMAP 条款(防过度叙事);"
                 "三档边界系 v2 承继数字"})

    # ---- 复现地板 ----
    if wa["ret_mean"] < floor_repro:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"胜者 {wa['ret_mean']} < {floor_repro}(=(85/92)×{R})——"
                    f"重训未复现参考水平,命题未考;深度副判:{depth_verdict}",
             "arms_full32": means, "r31_2": round(r31_2, 2), "R": R})
        attention("判决:未复现参考水平")
        return

    # ---- 发射判据(配对 vs v31-ref-launch,按 seed 键)----
    diffs = [wrows[s]["ret"] - ref_rows[s]["ret"] for s in sorted(ref_rows)]
    pd_mean = sum(diffs) / 32
    pd_wins = sum(x > 0 for x in diffs)
    launch = pd_mean >= PAIRED_DIFF and pd_wins >= PAIRED_WINS
    log({"event": "launch_check", "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins,
         "died": wa["died"], "dive_per_ep": round(dpe, 2),
         "dual_attribution": dual_attr, "tau_note": wa["farm_tau_mean"],
         "multiple_comparison_note": "同池 18/32 线第 5 次挑战者开奖"
                                     "(11→16→17→v30 未及→本案);"
                                     "P(赢≥18|p=.5)≈43% 注记随判词;"
                                     "本案基线系现场推导之 R 非历史档案,一并注记",
         "line_note": "+4/18/死6 系 v2 承继线,v3 错误率未重标定;临线判决须携注记"})

    p_hi = round(throne + 4.0, 1)
    P_LINE = (f"P线速查(按序判定;王座在位锚值 {throne} / 脚本参照 {script_ref},"
              f"R2 金池档案现场读取,重测不改名分):死>6→回退;"
              f"金≥{p_hi}且死≤4→P31-登基;∈({throne},{p_hi})且死≤4→点估增益王座不动;"
              f">{throne}且死5-6→持平(安全性);∈[{script_ref},{throne}]→持平;"
              f"<{script_ref}→回退。金池开牌史:金牌实开 v22/v23/v24 三次,"
              "R2 定锚五发系测量暴露;本案若发射为金牌第 4 次实开、金池累计第 9 次暴露,"
              "固定池偏置随判词。名分流转照 D3-6 条款:发射线动现任组装体,"
              "P 线动王座,两名分分行宣示")
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker {W_NPZ} "
                      f"--manager-npz {npz[winner]} --seeds 9000-9031 "
                      f"--tag v31-golden --board")
        dual_note = ("【双归因未裁】override 触线经双归因路径放行——烧牌前须人工完成"
                     "配比漂移 vs 真退化裁定并回写 dual_attr_ruling 事件,先裁后烧;"
                     if dual_attr else "")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wa["ret_mean"],
             "died": wa["died"], "wins": pd_wins, "mean_diff": round(pd_mean, 2),
             "arms_full32": means, "r31_2": round(r31_2, 2), "R": R,
             "manager_npz": npz[winner], "manager_npz_sha": sha16(npz[winner]),
             "full32_sha": wa["_sha"], "golden_cmd": golden_cmd, "p_line": P_LINE,
             "知会出处": "战役级金评纪律先例(v24-v30;裁量已在 R2/v31 案由入册)",
             "note": dual_note + "金牌由值守手启,单臂一次;败臂/未发射臂永不见"
                     " 9000 段;开牌后回写 golden_result 事件"})
        attention(dual_note + f"金牌待手启:{winner}(命令与 P 线速查见 ledger);"
                  f"深度副判:{depth_verdict}")
        return

    # ---- 不发射:穷尽分派 ----
    wins_note = (f"(宽度移动注记:赢 {pd_wins}/32 ≥14,不改判档)"
                 if pd_wins >= 14 else "")
    cont_note = ("cont 败限定:超参未做续训适配(lr 无日程、优化器状态承继、"
                 "熵沿 fresh 配方),溃败系路径级结果,不得读作旧知识必为包袱"
                 "(v25 裸微调先例);" if winner != "v31-mcont" else "")
    if pd_mean >= PAIRED_DIFF and pd_wins < PAIRED_WINS:
        verdict = (f"均值增益 +{pd_mean:.2f} 而宽度未达(赢 {pd_wins}/32 < 18)"
                   "——点估增益,不烧牌")
    elif pd_mean >= 2.0:
        verdict = (f"配对均差 {pd_mean:.2f} ∈[+2,+4)——探针级改进,不烧牌"
                   f"(赢 {pd_wins}/32){wins_note}")
    else:
        verdict = (f"配对均差 {pd_mean:.2f} <+2——现任连任,新世界再教育无增益"
                   f"(功效限定)(赢 {pd_wins}/32){wins_note}")
    log({"event": "VERDICT_PATH", "golden_authorized": False,
         "verdict": verdict, "depth_verdict": depth_verdict,
         "winner": winner, "winner_mean": wa["ret_mean"],
         "paired_mean": round(pd_mean, 2), "paired_wins": pd_wins,
         "arms_full32": means, "r31_2": round(r31_2, 2), "R": R,
         "cont_note": cont_note})
    attention(f"判决(不发射):{verdict};深度副判:{depth_verdict}")


if __name__ == "__main__":
    main()
