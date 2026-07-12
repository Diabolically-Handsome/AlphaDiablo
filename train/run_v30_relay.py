"""v30「工人接力」双臂驱动(docs/PREREG-v30.md 终稿条款唯一执行者)。

结构:preflight(sha 全链)→ king 锚 sd 导出 + G-KL-W 保真闸 → G-A0W(组装
回归 ≡140.3 档案)→ 双臂串行(各 2 腿 resume 链;臂间唯一变量 = 皮筋教师:
king 自锚 / bc 老教师)→ 双臂满 32(M29 组装)→ 资格/递补(v29)→ 科学主判
(四档按序)→ 地板 → 双锚发射(含反搭便车先决 + 深层死伤合取)→
GOLDEN_AUTHORIZED / 不发射穷尽分派。金牌手启(单臂一次)。
面板 33 项裁决全落地:身份链三闩、逐臂 G-KL 闸、G-绿洲 ff_dry 降级、
哨兵=v29 qual_of 口径、白天航班时钟重标(健全 4h/超时 4.5h)、绊线 124.95
钉死、评测一律 --manager-npz、发射线加新证据合取、学不动档具名。
用法:.venv/bin/python train/run_v30_relay.py
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
V30 = RUNS / "v30"
V30.mkdir(parents=True, exist_ok=True)
LEDGER = V30 / "gate_ledger.jsonl"
DRIVER_LOCK = V30 / ".driver.lock"
DRIVER_LOCK_PURPOSE = "v30 接力驱动"
EVAL = RUNS / "eval-assembled"

M29_NPZ = RUNS / "v29-mfresh" / "policy.npz"
M29_SHA = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
W_ZIP = ROOT / "train" / "models" / "v28-worker-leg1" / "model_final.zip"
W_SHA = "2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42"
W_NPZ = ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz"
W_NPZ_SHA = "976b6c05edaa0a32bb30bd372782e1201c72b029cedcbb3a5bf2361d34f27f8a"
START_NT = 3_497_984            # v28-leg1 zip num_timesteps(=7×499,712,发车断言)
SCI_ANCHOR = EVAL / "v29-mfresh-full32.json"    # 140.3(科学锚:同经理单变量)
SCI_SHA = "08633101c010a2975b9001a71660bb50d43e681842e3fe1befdfdfd48f99ce63"
LAUNCH_ANCHOR = EVAL / "v28-G3-leg1.json"       # 112.4(发射锚:现任组装体)
LAUNCH_SHA = "6fc6a44c7862424ab5f71ff3a5031adfd34a3e33f9f4f2f8aee781a07711e59d"
SCREEN_BASE_JSON = EVAL / "v29-mfresh-s16.json"  # 绊线基线 147.0(16 种子,M29 组装)
BC_SD = RUNS / "bc-worker" / "policy_sd.pt"
KING_SD = V30 / "king_anchor_sd.pt"             # 稳定路径(面板 major:禁时间戳)

LEG = 244 * 2048                # 499,712(量子 2048 = n_steps×num_envs,v28 口径)
QUANTUM = 2048
LEGS = 2
BUDGET_STEPS = LEGS * LEG       # 每臂独立的新步硬预算
BETA = 0.015625
HARD_LINE = 62.8                # 历版灾难地板(140 刻度下更保守,保留)
TRIP_LINE = 124.95              # = 0.85×147.0(v29-mfresh-s16,16 种子 M29 口径)
SCREEN_BASE = 147.0
FLOOR = 129.1                   # = 0.92×140.3(谱系比例先例 v25/v29;独立闸)
PD112_LINE, WINS112_LINE, DEATHS_MAX = 4.0, 18, 6
PRE140_LINE = 2.0               # 反搭便车先决(面板 blocker:抬至与科学"有效"同线)
D2DEATH_LAUNCH_MAX = 3          # 发射合取:深层死伤不得劣于基线 3(防叙事合流)
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 2
ARMS = {"king": {"base_seed": 301_000, "override": str(KING_SD)},
        "bc": {"base_seed": 305_000, "override": None}}


class OperationalFailure(RuntimeError):
    """基础设施/接线失败；与正常科学绊线区分，进程必须非零退出。"""


def budgeted_leg_steps(spent_steps: int, cap: int = LEG) -> int:
    remaining = max(0, BUDGET_STEPS - spent_steps)
    return min(cap, (remaining // QUANTUM) * QUANTUM)


def ensure_retry_budget(spent_steps: int, cap: int = LEG) -> None:
    if budgeted_leg_steps(spent_steps, cap) == 0:
        raise OperationalFailure("失败尝试已耗尽硬预算，无法完成当前腿")


def observed_attempt_steps(base: int, global_steps: int, status_steps: int,
                           allocated: int) -> int:
    observed = max(int(global_steps), int(status_steps))
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
    """只有显式 PASS 能成为授权输入；空探针的 SKIPPED 必须 fail closed。"""
    return probes_ok is True


def missing_authorization_arms(models: dict) -> list[str]:
    """双臂实验缺任一臂都不得降级成单臂授权。"""
    return sorted(set(ARMS) - set(models))


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def attention(why: str):
    with open(V30 / "NEEDS_ATTENTION", "a") as f:
        f.write(time.strftime("%F %T ") + why + "\n")


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sha16(p) -> str:
    return sha256(p)[:16]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v30 relay 的 HARD_LINE/资格/R4/科学与发射静态阈值仅在 pre-v3 环境语义标定；"
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


def require_sha256(path: pathlib.Path, expected_sha: str, name: str) -> None:
    require(path.is_file() and sha256(path) == expected_sha,
            f"{name} sha 漂移/缺失:{path}")


def read_comparable_reference(path: pathlib.Path, *, tag: str, worker,
                              manager, seeds: range, name: str) -> dict:
    """活动科学/发射锚只接受当前语义和固定组装体的 schema-v2 档案。"""
    try:
        snapshot = freeze_eval_identity(ROOT, worker, manager)
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seeds)
        document = read_eval_archive(path, **expected)
        verify_eval_identity(snapshot, ROOT)
        return document
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise OperationalFailure(
            f"{name} 不满足当前 schema-v2 可比性契约；"
            "环境语义变更后须用固定 worker/manager 重跑基线"
        ) from exc


def read_comparable_science() -> dict:
    return read_comparable_reference(
        SCI_ANCHOR, tag="v29-mfresh-full32", worker=W_ZIP,
        manager=M29_NPZ, seeds=range(7000, 7032), name="科学锚")


def read_comparable_launch() -> dict:
    return read_comparable_reference(
        LAUNCH_ANCHOR, tag="v28-G3-leg1", worker=W_ZIP,
        manager=None, seeds=range(7000, 7032), name="发射锚")


def read_comparable_screen() -> dict:
    return read_comparable_reference(
        SCREEN_BASE_JSON, tag="v29-mfresh-s16", worker=W_ZIP,
        manager=M29_NPZ, seeds=range(7000, 7016), name="绊线基线")


def zip_steps(p: pathlib.Path) -> int:
    try:
        with zipfile.ZipFile(p) as z:
            return int(json.loads(z.read("data"))["num_timesteps"])
    except Exception:
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


def run(cmd, logfile, timeout) -> int:
    with open(V30 / logfile, "w") as lf:
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


def exam(model_path, tag, seeds):
    require_sha256(M29_NPZ, M29_SHA, "M29 npz")
    out = EVAL / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    try:
        seed_values = parse_seed_range(seeds)
        snapshot = freeze_eval_identity(ROOT, model_path, M29_NPZ)
        expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
        command = [PY, "train/eval_assembled.py",
                   "--worker", snapshot["worker"]["path"],
                   "--manager-npz", snapshot["manager"]["path"],
                   "--seeds", seeds, "--tag", tag]
        rc = run(command, f"exam-{tag}.{time.time_ns()}.log", timeout=1_800)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    if rc != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
        d["agg"]["_sha"] = sha16(out)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    return d


def exam_retry(model_path, tag, seeds):
    d = exam(model_path, tag, seeds)
    if d is None:
        log({"event": "exam_crash", "tag": tag, "note": "评测失败,重考一次"})
        d = exam(model_path, tag, seeds)
    return d


def by_seed(rows) -> dict:
    m = {r["seed"]: r for r in rows}
    require(len(rows) == len(m), "种子集合异常(含重复 seed)")
    require(set(m) == set(range(7000, 7032)), "种子集合异常(须 7000-7031)")
    return m


def metrics(d) -> dict:
    rows, a = d["rows"], d["agg"]
    by_seed(rows)  # 在任何均值/死数进入选模前先封死缺行与重复行
    hist = a.get("worker_action_hist") or {}
    counts = ({int(k): int(v) for k, v in hist.items()} if isinstance(hist, dict)
              else {i: int(v) for i, v in enumerate(hist)})   # JSON 字典键系字符串(06:32 尸检)
    calls = sum(counts.values())
    return {"mean": a["ret_mean"], "died": a["died"],
            "dive_per_ep": round(sum(r["mode_seq"].count("D") for r in rows) / len(rows), 2),
            "depth2_seeds": sum(1 for r in rows if r["depth"] >= 2),
            "d2_deaths": sum(1 for r in rows if r["died"] and r["depth"] >= 2),
            "bonus_per_ep": round(sum(8 * sum(range(1, r["depth"])) for r in rows) / len(rows), 2),
            "a13_share": round(counts.get(13, 0) / calls, 4) if calls else None,
            "tau": a["farm_tau_mean"], "override": a["override_rate"],
            "descend": a["farm_descend_rate"], "cap": a["cap_rate"],
            "depth_median": a.get("depth_median"), "sha": a["_sha"]}


def qual_of(d) -> dict:
    a = d["agg"]
    dpe = sum(r["mode_seq"].count("D") for r in d["rows"]) / len(d["rows"])
    void = a["override_rate"] >= R4["override_void"] or (dpe > 1 and a["died"] > 6)
    hard_ok = a["farm_descend_rate"] <= R4["descend"] and a["cap_rate"] < R4["cap"]
    override_ok = a["override_rate"] < R4["override_sentinel"]
    dual = dpe > 1 and hard_ok and not override_ok
    ok = a["died"] <= DEATHS_MAX and not void and ((hard_ok and override_ok) or dual)
    return {"qual_ok": ok, "void": void, "dual_attr": dual}


def preflight():
    require_calibrated_protocol()
    for p, s, name in ((M29_NPZ, M29_SHA, "M29 npz"), (W_ZIP, W_SHA, "工人 zip"),
                       (W_NPZ, W_NPZ_SHA, "工人 npz")):
        require_sha256(p, s, name)
    require(zip_steps(W_ZIP) == START_NT, "START 断言失败(工人 zip num_timesteps)")
    require(BC_SD.exists(), "BC 教师 sd 缺失(bc 臂教师随 zip 驮带此绝对路径)")
    read_comparable_science()
    read_comparable_launch()
    read_comparable_screen()
    tags = ["v30-GA0W", "v30-golden"] + [f"v30-{a}-leg{k}" for a in ARMS for k in (1, 2)] \
        + [f"v30-{a}-full32" for a in ARMS]
    for t in tags:
        require(not (EVAL / f"{t}.json").exists(), f"目标档案已存在:{t}(重启协议先 .void)")
    for a in ARMS:
        for k in (1, 2):
            require(not (RUNS / f"v30-{a}-leg{k}").exists(), f"运行目录残留:v30-{a}-leg{k}")
    log({"event": "preflight_ok", "bc_sd_sha": sha16(BC_SD), "start_nt": START_NT})


def main():
    try:
        with exclusive_lock(DRIVER_LOCK, DRIVER_LOCK_PURPOSE):
            _main()
    except (OperationalFailure, OutputReservationError) as e:
        log({"event": "OPERATIONAL_FAILURE", "why": str(e)})
        attention("运维失败:\n" + str(e))
        raise SystemExit(2) from e
    except Exception as e:
        log({"event": "DRIVER_EXCEPTION", "why": repr(e)})
        attention("驱动异常死亡:\n" + traceback.format_exc())
        raise


def _main():
    preflight()
    science_doc = read_comparable_science()
    science_base = science_doc["agg"]["ret_mean"]
    floor = round(0.92 * science_base, 1)
    d2death_launch_max = sum(
        r["died"] and r["depth"] >= 2 for r in science_doc["rows"])
    screen_base = read_comparable_screen()["agg"]["ret_mean"]
    trip_line = round(0.85 * screen_base, 2)
    log({"event": "start", "prereg": "docs/PREREG-v30.md", "leg": LEG, "legs_per_arm": LEGS,
         "trip": trip_line, "screen_base": screen_base,
         "floor": floor, "science_base": science_base, "pre140": PRE140_LINE,
         "launch": [PD112_LINE, WINS112_LINE, DEATHS_MAX, d2death_launch_max]})

    # ---- king 锚 sd 导出 + G-KL-W 保真闸(身份链三闩之一)----
    require_sha256(W_ZIP, W_SHA, "工人 zip")
    if run([PY, "train/export_manager_sd.py", str(W_ZIP), str(KING_SD)],
           "export-king-sd.log", timeout=600) != 0 or not KING_SD.exists():
        why = "king 锚 sd 导出失败"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    king_sha = sha256(KING_SD)
    require_sha256(W_NPZ, W_NPZ_SHA, "工人 npz")
    if run([PY, "train/check_teacher_parity.py", str(KING_SD), str(W_NPZ)],
           "gklw.log", timeout=600) != 0:
        why = "G-KL-W 保真闸失配——自锚教师不忠于王"
        log({"event": "STOP", "why": why})
        attention("G-KL-W 失配")
        raise OperationalFailure(why)
    require(sha256(KING_SD) == king_sha, "G-KL-W 身份链断裂(闸后文件被改)")
    log({"event": "g_kl_w", "king_sd_sha": king_sha[:16], "parity": "0/1000"})

    # ---- G-A0W:组装回归(v28-leg1 npz × M29 npz 满 32 ≡ 140.3 档案)----
    require_sha256(W_NPZ, W_NPZ_SHA, "工人 npz")
    ga0 = exam_retry(str(W_NPZ), "v30-GA0W", "7000-7031")
    if ga0 is None:
        why = "G-A0W 考试连败"
        log({"event": "STOP", "why": why})
        attention(why)
        raise OperationalFailure(why)
    ref_sci = by_seed(read_comparable_science()["rows"])
    bad = [s for s, r in by_seed(ga0["rows"]).items()
           if (abs(r["ret"] - ref_sci[s]["ret"]) > 0.01 or r["died"] != ref_sci[s]["died"]
               or r["depth"] != ref_sci[s]["depth"]
               or r["mode_seq"] != ref_sci[s]["mode_seq"])]
    log({"event": "g_a0w", "mismatch_seeds": bad, "n_ok": 32 - len(bad)})
    if bad:
        why = "G-A0W 位级回归失配——人工重锚"
        log({"event": "STOP", "why": why})
        attention(f"G-A0W 失配:{bad}")
        raise OperationalFailure(why)

    # ---- 双臂串行:各 2 腿 resume 链 ----
    arm_models = {}
    for arm, cfg in ARMS.items():
        require_sha256(W_ZIP, W_SHA, "工人 zip")
        nt_chain = START_NT
        prev = str(W_ZIP)
        burned = 0
        spent_steps = 0
        attempts = {}
        halted = None
        k = 1
        while k <= LEGS:
            leg_steps = budgeted_leg_steps(spent_steps)
            if leg_steps < LEG:
                log({"event": "leg_budget_shrunk", "arm": arm, "leg": k,
                     "steps": leg_steps, "spent": spent_steps,
                     "remaining": BUDGET_STEPS - spent_steps,
                     "note": "所有既成新步与失败烧步逐次扣减每臂硬预算"})
            if leg_steps == 0:
                log({"event": "budget_exhausted", "arm": arm, "leg": k,
                     "spent": spent_steps})
                break
            seed_k = cfg["base_seed"] + 1_000 * (k - 1)
            probes = reachable_probes(nt_chain, leg_steps)
            if len(probes) < 2:
                log({"event": "calib_trimmed", "arm": arm, "leg": k,
                     "probes": probes, "note": "烧步缩腿后不可达探针已裁掉"})
            attempts[k] = attempts.get(k, 0) + 1
            run_name = f"v30-{arm}-leg{k}"
            run_dir = RUNS / run_name
            for fn in ("status.json", "calib.jsonl", "sentinel.jsonl"):
                p = run_dir / fn
                if p.exists():
                    p.rename(p.with_suffix(f".pre{attempts[k]}.{time.strftime('%H%M%S')}.void")) \
                        if fn != "status.json" else p.unlink()
            cmd = [PY, "train/train_ppo.py", "--worker", "--algo", "mppo", "--gamma", "1.0",
                   "--max-steps", "3000", "--num-envs", "4", "--n-steps", "512",
                   "--lr", "3e-4", "--ent-coef", "0.005", "--seed", str(seed_k),
                   "--total-steps", str(leg_steps), "--run-name", run_name,
                   "--distill-beta", str(BETA), "--teacher-sd", str(BC_SD), "--skip-dry",
                   "--manager-npz", str(M29_NPZ), "--resume-from", prev,
                   "--allow-manager-change",
                   "--calib-probes", ",".join(str(p) for p in probes),
                   "--calib-record-only"]
            require_sha256(M29_NPZ, M29_SHA, "M29 npz")
            if cfg["override"]:
                require(sha256(KING_SD) == king_sha, "king 锚 sd 漂移(逐腿身份链)")
                cmd += ["--teacher-override", cfg["override"]]
            log({"event": "leg_start", "arm": arm, "leg": k, "attempt": attempts[k],
                 "seed": seed_k, "steps": leg_steps, "resume_from": prev,
                 "probes": probes, "teacher": cfg["override"] or "BC(zip 驮带)"})
            model_path = run_dir / "model_final.zip"
            old_mtime = model_path.stat().st_mtime_ns if model_path.exists() else None
            t0 = time.time()
            rc = run(cmd, f"{run_name}.try{attempts[k]}.log", timeout=16_200)  # 4.5h
            dt = round(time.time() - t0)
            fresh_model = model_path.exists() and model_path.stat().st_mtime_ns != old_mtime
            nt = zip_steps(model_path) if fresh_model else 0
            try:
                st = json.loads((run_dir / "status.json").read_text())["total_steps"]
            except Exception:
                st = 0
            expected = nt_chain + leg_steps
            clean = rc == 0 and fresh_model and nt == expected
            sampled = observed_attempt_steps(nt_chain, nt, st, leg_steps)
            if not clean:
                partial = sampled
                charged = failed_attempt_charge(partial, leg_steps)
                spent_steps += charged
                burned += charged
                log({"event": "leg_crash", "arm": arm, "leg": k, "attempt": attempts[k],
                     "rc": rc, "nt_zip": nt, "expected": expected,
                     "burned_observed": partial, "burned_charged": charged,
                     "burned_total": burned,
                     "spent_total": spent_steps})
                ensure_retry_budget(spent_steps)
                if attempts[k] >= 4:
                    halted = f"腿 {k} 连崩 4 次(运维自护,非注册闸门)"
                    log({"event": "crash_halt", "arm": arm, "why": halted})
                    attention(f"{arm} {halted}")
                    raise OperationalFailure(f"{arm} {halted}")
                continue
            require(sampled == leg_steps, "干净收官的观测步数与配额不一致")
            spent_steps += sampled
            nt_chain = expected
            if dt > 7_200:
                log({"event": "SLOW_MACHINE", "arm": arm, "leg": k, "dt_sec": dt,
                     "note": "白天慢时段,只记不裁(健全收官线 4h)"})
                attention(f"{arm} 腿 {k} 墙钟 {dt}s(慢机,未触线)")
            # G-绿洲(仅腿 1;面板 major:ff_dry 降级为记录)
            if k == 1:
                sent = run_dir / "sentinel.jsonl"
                lines = []
                if sent.exists():
                    for l in sent.read_text().splitlines():
                        if '"sentinel": "v23"' in l:
                            try:
                                lines.append(json.loads(l))
                            except Exception:
                                pass
                if not lines:
                    halted = "G-绿洲无哨兵行"
                    log({"event": "STOP_ARM", "arm": arm, "why": halted})
                    attention(f"{arm}:{halted}")
                    raise OperationalFailure(f"{arm}:{halted}")
                last = lines[-1]
                if last.get("dry", 1) != 0:
                    halted = "G-绿洲失败:学习窗含 dry(skip_dry 真不变量被破)"
                    log({"event": "STOP_ARM", "arm": arm, "why": halted,
                         "dry": last.get("dry")})
                    attention(f"{arm}:{halted}")
                    raise OperationalFailure(f"{arm}:{halted}")
                if last.get("ff_dry", 0) == 0:
                    log({"event": "g_oasis_note", "arm": arm, "ff_dry": 0,
                         "note": "M29 分布下开腿 ff_dry=0 系合法形态(v22-H 标定出身),记录不裁"})
                    attention(f"{arm} 腿1 ff_dry=0(记录,晨检肉眼裁)")
                log({"event": "g_oasis", "arm": arm, "dry": last.get("dry"),
                     "ff_dry": last.get("ff_dry"), "fresh": last.get("fresh"), "ok": True})
            # G-CAL 接线闸(只记不裁;probes_ok 每腿)
            calib_p = run_dir / "calib.jsonl"
            recs = ([json.loads(l) for l in calib_p.read_text().splitlines()]
                    if calib_p.exists() else [])
            probes_ok = (all(any(p <= r["step"] < p + QUANTUM and r["g_ce"] > 0
                                 and r["distill_ce"] > 0 for r in recs)
                             for p in probes) if probes else None)
            probe_status = ("SKIPPED" if probes_ok is None
                            else "PASS" if probes_ok else "FAIL")
            log({"event": "g_cal", "arm": arm, "leg": k, "probes_ok": probes_ok,
                 "probe_status": probe_status,
                 "records": [{kk: r.get(kk) for kk in
                              ("step", "g_ce", "teacher_diverge", "tripped")} for r in recs]})
            if probes_ok is False:
                halted = f"G-CAL 接线失败(腿 {k})"
                log({"event": "STOP_ARM", "arm": arm, "why": halted})
                attention(f"{arm}:{halted}")
                raise OperationalFailure(f"{arm}:{halted}")
            # 腿考(M29 组装,16 种子)
            r = exam_retry(run_dir / "model_final.zip", f"v30-{arm}-leg{k}", "7000-7015")
            if r is None:
                halted = f"腿 {k} 考试连败"
                log({"event": "STOP_ARM", "arm": arm, "why": halted})
                attention(f"{arm}:{halted}")
                raise OperationalFailure(f"{arm}:{halted}")
            score = round(r["agg"]["ret_mean"], 1)
            log({"event": "leg_exam", "arm": arm, "leg": k, "score": score,
                 "died": r["agg"]["died"], "diverge": r["agg"].get("script_divergence_rate"),
                 "sha": r["agg"]["_sha"], "model_sha": sha16(run_dir / "model_final.zip"),
                 "nt_chain": nt_chain, "dt_sec": dt})
            prev = str(run_dir / "model_final.zip")
            if candidate_probe_eligible(probes_ok):
                arm_models[arm] = prev        # 最后一条接线 PASS 的干净收官腿
            else:
                log({"event": "candidate_ineligible", "arm": arm, "leg": k,
                     "why": "G-CAL probes SKIPPED；该模型不得形成可授权臂"})
            if score < HARD_LINE:
                log({"event": "HARD_TRIP", "arm": arm, "leg": k, "score": score})
                attention(f"{arm} 硬绊:腿 {k} = {score}")
                break
            # 长训练期间 runtime/content 若变化，旧 screen v2 会在这里 fail closed。
            screen_base = read_comparable_screen()["agg"]["ret_mean"]
            trip_line = round(0.85 * screen_base, 2)
            if score < trip_line:
                log({"event": "trip_halt", "arm": arm, "leg": k, "score": score,
                     "line": trip_line,
                     "note": f"单腿 <0.85×{screen_base},该臂止训(臂间独立)"})
                attention(f"{arm} 绊线止训:腿 {k} = {score}")
                break
            if dt > 14_400:
                log({"event": "sanity_finish", "arm": arm, "leg": k, "dt_sec": dt,
                     "note": "墙钟 >4h 健全收官(白天航班重标)"})
                attention(f"{arm} 健全收官:腿 {k} {dt}s")
                break
            k += 1
        if arm not in arm_models:
            log({"event": "arm_dead", "arm": arm, "why": halted or "无干净腿"})
            attention(f"{arm} 臂无可考模型")

    missing_arms = missing_authorization_arms(arm_models)
    if missing_arms:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "缺任一接线 PASS 臂，禁止授权",
             "missing_arms": missing_arms})
        why = f"缺少可授权臂:{missing_arms}"
        attention(why)
        raise OperationalFailure(why)

    # ---- 双臂满 32(M29 组装)----
    full, mets = {}, {}
    for arm, mp in arm_models.items():
        d = exam_retry(pathlib.Path(mp), f"v30-{arm}-full32", "7000-7031")
        if d is None:
            why = f"{arm} 满 32 连败——人工验尸"
            log({"event": "STOP", "why": why})
            attention(why)
            raise OperationalFailure(why)
        full[arm] = d
        mets[arm] = metrics(d)
        log({"event": "full32", "arm": arm, **mets[arm]})
    if len(full) == 2:
        fk, fb = by_seed(full["king"]["rows"]), by_seed(full["bc"]["rows"])
        r30_6 = sum(fk[s]["ret"] - fb[s]["ret"] for s in fk) / 32
        r30_6w = sum(fk[s]["ret"] > fb[s]["ret"] for s in fk)
        log({"event": "r30_6", "king_minus_bc_mean": round(r30_6, 2), "king_wins": r30_6w,
             "note": "|均差|<2 判方向未判定(预注册判读规则)"})

    # ---- 资格/胜者/递补(v29 D3-2)----
    quals = {a: qual_of(full[a]) for a in full}
    log({"event": "quals", **{a: quals[a] for a in full}})
    pool = [a for a in full if quals[a]["qual_ok"]]
    if not pool:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": "双臂资格失败——无胜者,接力命题未答(功效外)",
             "arms": {a: mets[a] for a in full}})
        attention("判决:双臂资格失败")
        return
    prelim = max(full, key=lambda a: full[a]["agg"]["ret_mean"])
    if prelim not in pool:
        log({"event": "substitution", "blocked": prelim, "why": quals[prelim]})
    ms = {a: full[a]["agg"]["ret_mean"] for a in pool}
    band = [a for a in pool if max(ms.values()) - ms[a] <= 0.05]
    if len(band) > 1:
        dmin = min(full[a]["agg"]["died"] for a in band)
        band = [a for a in band if full[a]["agg"]["died"] == dmin]
        winner = "king" if "king" in band else band[0]
    else:
        winner = band[0]
    W, wm = full[winner], mets[winner]
    wrows = by_seed(W["rows"])
    log({"event": "winner", "arm": winner, "mean": wm["mean"], "died": wm["died"],
         "substituted": winner != prelim})

    # ---- 双锚配对 ----
    science_doc = read_comparable_science()
    launch_doc = read_comparable_launch()
    ref_sci = by_seed(science_doc["rows"])
    ref_launch = by_seed(launch_doc["rows"])
    science_base = science_doc["agg"]["ret_mean"]
    launch_base = launch_doc["agg"]["ret_mean"]
    floor = round(0.92 * science_base, 1)
    d2death_launch_max = sum(
        r["died"] and r["depth"] >= 2 for r in science_doc["rows"])
    d112 = [wrows[s]["ret"] - ref_launch[s]["ret"] for s in sorted(ref_launch)]
    pd112, wins112 = sum(d112) / 32, sum(x > 0 for x in d112)
    d140 = [wrows[s]["ret"] - ref_sci[s]["ret"] for s in sorted(ref_sci)]
    pd140, wins140 = sum(d140) / 32, sum(x > 0 for x in d140)
    log({"event": "paired", "vs112_mean": round(pd112, 2), "vs112_wins": wins112,
         "vs140_mean": round(pd140, 2), "vs140_wins": wins140})
    log({"event": "draw_ledger", "note": "同池 18/32 线第 4 次挑战者开奖(11→16→17→本案);"
         "台账只记不裁;P(赢≥18|p=.5)≈43% 注记;发射线新证据合取已加(先决 ≥+2 与 d2death≤3)"})

    # ---- 科学主判(四档按序;面板:有效档绑曝露守护)----
    d2, d2d = wm["depth2_seeds"], wm["d2_deaths"]
    if d2 >= 12 and d2d <= 1 and pd140 >= 2.0:
        sci = f"接力有效(曝露≥12 ∧ 深层死伤≤1 ∧ 对{science_base}≥+2)"
    elif d2d >= 3 and pd140 < 0:
        sci = "接力无效(短板非本处方可修——课③④线索)"
    elif d2d in (2, 3) and abs(pd140) < 2.0:
        sci = "信号稀释/学不动档(命题未判定)"
    elif d2 < 12:
        sci = (f"带外(曝露塌缩:depth2={d2}<12;d2死={d2d},"
               f"对{science_base}={pd140:+.2f})入册不叙事")
    else:
        sci = f"带外(d2死={d2d},对{science_base}={pd140:+.2f})入册不叙事"
    log({"event": "science_verdict", "verdict": sci, "depth2": d2, "d2_deaths": d2d,
         "pd140": round(pd140, 2), "a13": wm["a13_share"], "dive": wm["dive_per_ep"],
         "note": "科学主判与发射/王座/Mark-I 互不改写(防叙事合流尾注)"})

    # ---- 地板(独立闸;科学主判已出,双向都算答案)----
    if wm["mean"] < floor:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "verdict": f"胜者 {wm['mean']} < 地板 {floor}(=0.92×{science_base})——重训未复现"
                        f"起点水平,接力发射流程终止;科学主判:{sci}"})
        attention(f"判决:未及地板;科学主判:{sci}")
        return

    # ---- 发射(按序:先决 → 深层死伤 → 四联)----
    launch = (pd140 >= PRE140_LINE and d2d <= d2death_launch_max
              and pd112 >= PD112_LINE and wins112 >= WINS112_LINE)
    if launch:
        golden_cmd = (f"{PY} {ROOT / 'train' / 'eval_assembled.py'} --worker "
                      f"{pathlib.Path(arm_models[winner]).with_suffix('')} --manager-npz {M29_NPZ} "
                      f"--seeds 9000-9031 --tag v30-golden --board")
        dual_note = ("【双归因未裁,先裁后烧:回写 dual_attr_ruling 后方可手启】"
                     if quals[winner]["dual_attr"] else "")
        sci_note = ("" if sci.startswith("接力有效")
                    else f"【科学主判非'有效'({sci}),判词禁用接力成功措辞,"
                         f"d2死={d2d} 对基线 3】")
        log({"event": "GOLDEN_AUTHORIZED", "arm": winner, "probe32_mean": wm["mean"],
             "died": wm["died"], "vs112": [round(pd112, 2), wins112],
             "vs140": [round(pd140, 2), wins140], "model": arm_models[winner],
             "model_sha": sha16(pathlib.Path(arm_models[winner])),
             "full32_sha": wm["sha"], "golden_cmd": golden_cmd,
             "p_line": "按序:死>6回退;≥101.2且死≤4登基;(97.2,101.2)且死≤4点估;"
                       ">97.2且死5-6持平安全性;[93.9,97.2]持平;<93.9回退",
             "mark1": "= 科学主判有效 ∧ v29三条件副判在胜者满32复判已学 ∧ P30登基",
             "note": dual_note + sci_note + "金池史上第4次实开;单臂一次;开牌后回写 golden_result"})
        attention(dual_note + f"金牌待手启:{winner};科学主判:{sci}")
        return

    # ---- 不发射:穷尽分派(按序)----
    quad = (f"(对{launch_base} {pd112:+.2f}/赢{wins112},"
            f"对{science_base} {pd140:+.2f},d2死{d2d})")
    wins_note = f"(宽度移动注记:赢 {wins112}/32 ≥14,不改判档)" if wins112 >= 14 else ""
    if pd140 < PRE140_LINE:
        verdict = (f"新证据不足档:对{science_base} {pd140:+.2f} < +2"
                   f"——存量垫烧被拦,不烧牌{quad}")
    elif d2d > d2death_launch_max:
        verdict = (f"深层死伤未改善拦截档:d2死 {d2d} > 基线 "
                   f"{d2death_launch_max}——不烧牌{quad}")
    elif pd112 >= PD112_LINE and wins112 < WINS112_LINE:
        verdict = f"均值增益而宽度未达——点估增益,不烧牌{quad}{wins_note}"
    elif pd112 >= 2.0:
        verdict = f"探针级改进,不烧牌{quad}{wins_note}"
    else:
        verdict = f"现任组装体连任,接力无发射级增益(功效限定){quad}"
    log({"event": "VERDICT_PATH", "golden_authorized": False, "verdict": verdict,
         "science_verdict": sci, "winner": winner, "winner_mean": wm["mean"]})
    attention(f"判决(不发射):{verdict};科学主判:{sci}")


if __name__ == "__main__":
    main()
