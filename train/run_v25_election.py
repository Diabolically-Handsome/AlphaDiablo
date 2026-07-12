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
V25 = RUNS / "v25"
V25.mkdir(parents=True, exist_ok=True)
LEDGER = V25 / "gate_ledger.jsonl"

LEG7_ZIP = str(ROOT / "train" / "models" / "v24-worker-leg7" / "model")   # SB3 路(与存档同)
LEG7_NPZ = str(ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz")
V22H_ZIP = ROOT / "train" / "models" / "v22-h-manager" / "model_final.zip"
WARM_SD = ROOT / "train" / "models" / "v22-h-manager" / "policy_full_sd.pt"
WARM_MANIFEST = WARM_SD.with_name(f"{WARM_SD.name}.manifest.json")
ARCHIVE = RUNS / "eval-assembled" / "v24-G3-leg7.json"   # 参考行 92.0(逐种子)
ARCHIVE_SHA = "22d9442257d3a3c79feb5b40918890917772e11036e4026ca4d3cc2005318359"
LEG7_ZIP_SHA = "ac65d4eb91fdb678f38ba7ea502812353c7c83afa0551e89ea4a12b13d55e781"
LEG7_NPZ_SHA = "a31fa7c6b18b5c3593f4e1753d97aac9386689aa6ad8b158c526b673c57fbc2a"
V22H_ZIP_SHA = "f3b579d2b0c9b613045692435a46702d1a9e8de8fc62e155c651f565d8bd6f1a"

ABANDON = 75.0        # 提前放弃闸(双臂 16 种子均 < 此值)
EXPECTED_STEPS = 40_192  # ceil(40_000 / (64×4)) × (64×4)
PAIRED_DIFF = 4.0     # 发射线:配对均差 ≥ +4
PAIRED_WINS = 18      # 且配对赢 ≥ 18/32
DEATHS_MAX = 6
FLOOR_REPRO = 85.0    # 胜者 < 85 → "重训未复现,命题未考"
R4 = {"descend": 0.0204, "override_sentinel": 0.03, "override_void": 0.08, "cap": 0.05}
CALIBRATED_PROTOCOL_VERSION = 2

ARMS = {
    "v25-mfresh": ["--ent-coef", "0.02", "--lr", "3e-4", "--seed", "22"],
    "v25-mwarm": ["--ent-coef", "0.005", "--lr", "1e-4", "--seed", "23",
                  "--bc-init", str(WARM_SD), "--init-source", "checkpoint"],
}


def log(event: dict):
    event = {"t": time.strftime("%H:%M:%S"), **event}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[ledger] {event}", flush=True)


def sha16(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]


def sha256(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_calibrated_protocol() -> None:
    if PROTOCOL_VERSION != CALIBRATED_PROTOCOL_VERSION:
        raise OperationalFailure(
            "v25 的 ABANDON/资格/R4 等裁决线仅在 pre-v3 语义标定；"
            "必须先重跑 protocol-v3 基线并人工更新预注册，禁止混用旧阈值"
        )


def read_comparable_anchor() -> dict:
    """活动裁决锚必须与当前环境和固定组装体同一 schema-v2 身份。"""
    try:
        snapshot = freeze_eval_identity(ROOT, LEG7_ZIP, None)
        expected = expected_eval_identity(
            snapshot, tag="v24-G3-leg7", seeds=range(7000, 7032))
        document = read_eval_archive(ARCHIVE, **expected)
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
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile):
        return 0


def by_seed(rows, expected=range(7000, 7032)) -> dict:
    result = {r["seed"]: r for r in rows}
    require(len(rows) == len(result), "评测档案含重复 seed")
    require(set(result) == set(expected), "评测档案 seed 集合异常")
    return result


def preflight() -> None:
    require_calibrated_protocol()
    leg7_zip = pathlib.Path(LEG7_ZIP + ".zip")
    leg7_npz = pathlib.Path(LEG7_NPZ)
    require(leg7_zip.is_file() and sha256(leg7_zip) == LEG7_ZIP_SHA,
            "v24 leg7 工人 zip 缺失或 sha 漂移")
    require(leg7_npz.is_file() and sha256(leg7_npz) == LEG7_NPZ_SHA,
            "v24 leg7 工人 npz 缺失或 sha 漂移")
    require(V22H_ZIP.is_file() and sha256(V22H_ZIP) == V22H_ZIP_SHA,
            "v22-H 经理 zip 缺失或 sha 漂移")
    read_comparable_anchor()
    tags = ["v25-GA0", "v25-golden"] + [
        f"{arm}-{suffix}" for arm in ARMS for suffix in ("s16", "full32")]
    for tag in tags:
        require(not (RUNS / "eval-assembled" / f"{tag}.json").exists(),
                f"评测档案已存在:{tag}")
    for arm in ARMS:
        require(not (RUNS / arm).exists(), f"运行目录残留:{arm}")


def validate_warm_export() -> None:
    """导出成功还必须证明产物来自本次钉死的 v22-H checkpoint。"""
    try:
        manifest = strict_json_loads(WARM_MANIFEST.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OperationalFailure("M-warm 导出清单缺失/不可读") from exc
    if not isinstance(manifest, dict):
        raise OperationalFailure("M-warm 导出清单不是 JSON 对象")
    checks = (
        (manifest.get("schema_version") == 1
         and manifest.get("artifact_type") == "checkpoint_policy_state",
         "M-warm 导出清单类型异常"),
        (manifest.get("artifact_sha256") == sha256(WARM_SD),
         "M-warm 权重与导出清单 sha 不匹配"),
        (manifest.get("source_checkpoint") == str(V22H_ZIP.resolve())
         and manifest.get("source_checkpoint_sha256") == V22H_ZIP_SHA,
         "M-warm 导出清单未绑定钉死的 v22-H checkpoint"),
        (isinstance(manifest.get("tensor_count"), int)
         and not isinstance(manifest.get("tensor_count"), bool)
         and manifest["tensor_count"] > 0,
         "M-warm 导出清单 tensor_count 非法"),
    )
    for passed, message in checks:
        if not passed:
            raise OperationalFailure(message)


def run(cmd, logfile, timeout=1_800) -> int:
    with open(V25 / logfile, "w") as lf:
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


def exam(worker, tag, seeds, manager_npz=None):
    out = RUNS / "eval-assembled" / f"{tag}.json"
    require(not out.exists(), f"档案不可变性:{out} 已存在,拒绝覆写")
    lo, hi = (int(x) for x in seeds.split("-", 1))
    seed_values = list(range(lo, hi + 1))
    require(seed_values and lo >= 0, f"非法 seed 范围:{seeds}")

    # 发车前冻结 worker/manager、SB3 真步数、bridge 与评测源码；子进程
    # 必须回交与这份快照完全一致的 schema-v2 档案。
    snapshot = freeze_eval_identity(ROOT, worker, manager_npz)
    expected = expected_eval_identity(snapshot, tag=tag, seeds=seed_values)
    worker_arg = (worker if snapshot["worker"]["kind"] in {"script", "bc"}
                  else snapshot["worker"]["path"])
    cmd = [PY, "train/eval_assembled.py", "--worker", str(worker_arg),
           "--manager-npz", snapshot["manager"]["path"],
           "--seeds", seeds, "--tag", tag]
    if run(cmd, f"exam-{tag}.log") != 0:
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    try:
        d = read_eval_archive(out, **expected)
        verify_eval_identity(snapshot, ROOT)
        d["agg"]["_sha"] = sha16(out)
    except (OSError, KeyError, TypeError, ValueError):
        if out.exists():
            out.rename(out.with_suffix(f".{time.time_ns()}.void"))
        return None
    return d


def dive_per_ep(rows) -> float:
    return sum(r["mode_seq"].count("D") for r in rows) / max(1, len(rows))


def main():
    try:
        with exclusive_lock(V25 / ".driver.lock", "v25 驱动"):
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
    log({"event": "start", "prereg": "docs/PREREG-v25.md v2",
         "paired_line": PAIRED_DIFF, "wins_line": PAIRED_WINS})

    # ---- G-A0:仪器回归(npz 工人 + 默认经理 ≡ v24-G3-leg7 存档,32/32)----
    ref = read_comparable_anchor()
    floor_repro = round(ref["agg"]["ret_mean"] * 85.0 / 92.0, 1)
    ga0 = exam(LEG7_NPZ, "v25-GA0", "7000-7031")
    if ga0 is None:
        why = "G-A0 考试进程失败"
        log({"event": "STOP", "why": why})
        raise OperationalFailure(why)
    ref_rows = by_seed(ref["rows"])
    ga0_rows = by_seed(ga0["rows"])
    bad = [s for s, r in ga0_rows.items()
           if (abs(r["ret"] - ref_rows[s]["ret"]) > 0.01
               or r["died"] != ref_rows[s]["died"]
               or r["depth"] != ref_rows[s]["depth"]
               or r["mode_seq"] != ref_rows[s]["mode_seq"])]
    log({"event": "g_a0", "mismatch_seeds": bad, "n_ok": 32 - len(bad)})
    if bad:
        why = "G-A0 位级回归失配——按预注册回退条款人工重锚"
        log({"event": "STOP", "why": why})
        raise OperationalFailure(why)

    # ---- M-warm 权重包 ----
    if run([PY, "train/export_manager_sd.py"], "export-warm-sd.log", timeout=600) != 0 or not WARM_SD.exists():
        why = "M-warm 权重包导出失败"
        log({"event": "STOP", "why": why})
        raise OperationalFailure(why)
    validate_warm_export()
    log({"event": "warm_sd", "sha": sha16(WARM_SD)})

    # ---- 两臂串行训练(v22-H status.json 原配方)----
    npz = {}
    for name, extra in ARMS.items():
        cmd = [PY, "train/train_ppo.py", "--options", "--algo", "mppo", "--gamma", "1.0",
               "--max-steps", "3000", "--n-steps", "64", "--num-envs", "4",
               "--total-steps", str(EXPECTED_STEPS), "--worker-npz", LEG7_NPZ,
               "--run-name", name] + extra
        log({"event": "arm_start", "arm": name, "cmd_extra": extra})
        t0 = time.time()
        rc = run(cmd, f"train-{name}.log", timeout=14_400)
        sp = RUNS / name / "status.json"
        try:
            steps = json.loads(sp.read_text())["total_steps"] if sp.exists() else 0
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            steps = 0
        nt = zip_steps(RUNS / name / "model_final.zip")
        log({"event": "arm_done", "arm": name, "rc": rc, "steps": steps,
             "nt_zip": nt,
             "dt_min": round((time.time() - t0) / 60, 1)})
        if rc != 0 or nt != EXPECTED_STEPS:
            why = (f"{name} 训练未达标(rc={rc},zip={nt},"
                   f"期望={EXPECTED_STEPS};命题未考,本版不追加重训)")
            log({"event": "STOP", "why": why})
            raise OperationalFailure(why)
        out = RUNS / name / "policy.npz"
        if run([PY, "train/export_manager_npz.py",
                str(RUNS / name / "model_final.zip"), str(out)],
               f"export-{name}.log", timeout=600) != 0 or not out.exists():
            why = f"{name} G-A0m parity 失败"
            log({"event": "STOP", "why": why})
            raise OperationalFailure(why)
        npz[name] = str(out)
        log({"event": "g_a0m", "arm": name, "npz_sha": sha16(out)})

    # ---- 提前放弃闸(16 种子)----
    s16 = {}
    for name in ARMS:
        d = exam(LEG7_ZIP, f"{name}-s16", "7000-7015", manager_npz=npz[name])
        if d is None:
            why = f"{name} 初筛考试失败"
            log({"event": "STOP", "why": why})
            raise OperationalFailure(why)
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
            why = f"{name} 满 32 考试失败"
            log({"event": "STOP", "why": why})
            raise OperationalFailure(why)
        full[name] = d
        log({"event": "full32", "arm": name, "mean": d["agg"]["ret_mean"],
             "died": d["agg"]["died"], "dive_per_ep": round(dive_per_ep(d["rows"]), 2),
             "tau": d["agg"]["farm_tau_mean"], "override": d["agg"]["override_rate"],
             "descend": d["agg"]["farm_descend_rate"], "sha": d["agg"]["_sha"]})
    warm_rows = by_seed(full["v25-mwarm"]["rows"])
    fresh_rows = by_seed(full["v25-mfresh"]["rows"])
    paired_wf = [warm_rows[s]["ret"] - fresh_rows[s]["ret"]
                 for s in range(7000, 7032)]
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
    # 训练历时可能跨越代码/engine/content 变更；裁决前按当前身份重读锚。
    ref = read_comparable_anchor()
    ref_rows = by_seed(ref["rows"])
    floor_repro = round(ref["agg"]["ret_mean"] * 85.0 / 92.0, 1)
    if wa["ret_mean"] < floor_repro:
        log({"event": "VERDICT_PATH", "golden_authorized": False,
             "why": f"胜者 {wa['ret_mean']} < {floor_repro}——重训未复现参考水平,命题未考"})
        return
    winner_rows = by_seed(W["rows"])
    diffs = [winner_rows[s]["ret"] - ref_rows[s]["ret"] for s in range(7000, 7032)]
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
