"""v23 组装体评测:冻结 H 经理 + {脚本|BC|PPO} FARM 工人(docs/PREREG-v23.md)。

用法:
  H7 基线:  .venv/bin/python train/eval_assembled.py --worker script --seeds 7000-7031
  G0'' 回归:… --worker script --seeds 7000-7031 --check-probe docs/assets/window_econ_v23_probe.json
  G1 BC 重放:… --worker bc --seeds 7000-7031
  G3 初筛:  … --worker train/runs/<run>/ckpt/model_XXX_steps --seeds 7000-7015
  金评(唯一一次):… --worker <胜者> --seeds 9000-9031 --board
协议:argmax(经理 numpy 前向 = G0' 位级对账过的同一段代码)、3000 微步、
回报 = 经理不折现账本。R4 哨兵(换层率/override/cap/τ̄)一并产出。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from eval_contract import (DEFAULT_MANAGER_SHA256, UINT32_MAX, EvalContractError,
                           OutputReservationError, PROTOCOL_VERSION,
                           PROTOCOL_SOURCE_FILES,
                           bridge_binary_path, checkpoint_num_timesteps_bytes,
                           exclusive_lock,
                           expected_eval_identity, file_identity, make_meta,
                           loaded_engine_binary_path,
                           read_eval_archive, recompute_agg,
                           reserve_output, resolve_checkpoint_file, runtime_identity,
                           script_worker_identity, strict_json_loads,
                           validate_eval_archive,
                           verify_file_identity)
from evaluate import (assembled_leaderboard_row, ensure_leaderboard_compatible,
                      freeze_standalone_contract, upsert_leaderboard_rows,
                      versioned_row_key)

if TYPE_CHECKING:
    from diablogym import NumpyManager

NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
NPZ_SHA = DEFAULT_MANAGER_SHA256
LB = ROOT / "train" / f"leaderboard-assembled-v{PROTOCOL_VERSION}.md"
OUTDIR = ROOT / "train" / "runs" / "eval-assembled"
LB_LOCK = ROOT / "train" / "runs" / "eval-locks" / "leaderboard-assembled.lock"
FARM = 0
_NATIVE_RUNTIME = None

ASSEMBLED_BOARD_SOURCE_FILES = (*PROTOCOL_SOURCE_FILES, "train/evaluate.py")
ASSEMBLED_BOARD_PROTOCOL = {
    "name": "diablogym.standalone.assembled",
    "seeds": list(range(9000, 9032)),
    "environment": "OptionsEnv",
    "max_steps": 3000,
    "action_selection": "argmax_with_action_masks",
    "manager_forward": "numpy_tanh_mlp",
    "reward": "undiscounted_manager_ledger",
    "worker_kinds": ["script", "bc_state_dict", "numpy_policy", "sb3_checkpoint"],
}
ASSEMBLED_LEADERBOARD_HEADER = (
    f"# Assembled-agent board protocol v{PROTOCOL_VERSION} — 32 fixed seeds\n\n"
    "Protocol: OptionsEnv, 3000 micro-steps, argmax + masks, seeds 9000-9031.\n"
    "Every row is bound to its immutable schema-v2 evaluation archive.\n\n"
    "| run | ret mean | ret med | died | depth med | notes |\n"
    "|---|---|---|---|---|---|\n"
)


def assembled_board_contract() -> dict:
    return freeze_standalone_contract(
        evaluator=f"standalone-assembled-v{PROTOCOL_VERSION}",
        protocol=ASSEMBLED_BOARD_PROTOCOL,
        source_files=ASSEMBLED_BOARD_SOURCE_FILES)


def _native_runtime(expected_runtime: dict | None = None):
    """在身份冻结后才映射 bridge/engine，并立即核对实际加载路径。"""
    global _NATIVE_RUNTIME
    if _NATIVE_RUNTIME is None:
        from diablogym import NumpyManager, OptionsEnv, bridge
        from diablogym.options_env import FARM as actual_farm

        if actual_farm != FARM:
            raise EvalContractError(
                f"OptionsEnv FARM 编号漂移:{actual_farm} != {FARM}")
        _NATIVE_RUNTIME = (NumpyManager, OptionsEnv, bridge)
    manager_class, env_class, loaded_bridge = _NATIVE_RUNTIME
    if expected_runtime is not None:
        try:
            expected_path = pathlib.Path(
                expected_runtime["bridge"]["path"]).resolve()
            actual_path = pathlib.Path(loaded_bridge.__file__).resolve()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise EvalContractError("冻结的 bridge runtime identity 结构异常") from exc
        if actual_path != expected_path:
            raise EvalContractError(
                f"实际加载 bridge 路径与冻结身份不一致:{actual_path} != {expected_path}")
        loaded_engine_binary_path(expected_runtime["engine"]["path"])
        if runtime_identity(ROOT, actual_path) != expected_runtime:
            raise EvalContractError(
                "native import 期间 bridge、engine、游戏内容或协议源码发生变化")
    return manager_class, env_class, loaded_bridge


def np_policy_from_sd(source: str | pathlib.Path | bytes,
                      expected_sha256: str | None = None) -> NumpyManager:
    import torch
    payload = (source if isinstance(source, bytes)
               else pathlib.Path(source).read_bytes())
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"BC state_dict SHA 不匹配:{actual_sha256} != {expected_sha256}")
    sd = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    required = (
        "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
        "action_net.weight", "action_net.bias",
    )
    if not isinstance(sd, dict) or any(k not in sd for k in required):
        missing = [k for k in required if not isinstance(sd, dict) or k not in sd]
        raise ValueError(f"BC state_dict 缺少策略头键:{missing}")
    tensors = [sd[k].detach().cpu() for k in required]
    if not all(torch.isfinite(t).all().item() for t in tensors):
        raise ValueError("BC state_dict 含 NaN/Inf")
    w0, b0, w1, b1, wa, ba = tensors
    if (w0.ndim != 2 or tuple(w0.shape)[1] != 298 or b0.shape != w0.shape[:1]
            or w1.shape != (w0.shape[0], w0.shape[0]) or b1.shape != w0.shape[:1]
            or wa.shape != (15, w0.shape[0]) or ba.shape != (15,)):
        raise ValueError("BC 工人策略形状异常(须 298→hidden→hidden→15)")
    manager_class, _env_class, _bridge = _native_runtime()
    m = manager_class.__new__(manager_class)
    m.w0, m.b0, m.w1, m.b1, m.wa, m.ba = (
        t.numpy().astype(np.float32, copy=False) for t in tensors)
    m.source_sha256 = actual_sha256
    return m


def capture_passed_bc(sd_path: pathlib.Path) -> tuple[bytes, bytes]:
    """Capture policy/report once and validate the gate against those bytes."""
    report = sd_path.with_name("bc_report.json")
    try:
        policy_payload = sd_path.read_bytes()
        report_payload = report.read_bytes()
    except OSError as exc:
        raise ValueError(f"BC 闸门报告缺失/不可读:{report}") from exc
    from train_ppo import _validate_bc_report
    _validate_bc_report(
        sd_path, "data_gate", policy_payload=policy_payload,
        report_payload=report_payload)
    return policy_payload, report_payload


def worker_label(spec: str) -> str:
    if spec in {"script", "bc"}:
        return spec
    if pathlib.Path(spec).suffix.lower() == ".npz":
        return pathlib.Path(spec).parent.name
    return pathlib.Path(spec).stem


def load_worker(spec: str, protocol_bundle_sha256: str):
    """返回 (workers、标签、身份清单)。spec ∈ script | bc | *.npz | SB3 zip。"""
    if spec == "script":
        return None, "script", script_worker_identity(protocol_bundle_sha256)
    if spec == "bc":
        sd_path = (ROOT / "train" / "runs" / "bc-worker" / "policy_sd.pt").resolve()
        policy_payload, report_payload = capture_passed_bc(sd_path)
        policy_sha256 = hashlib.sha256(policy_payload).hexdigest()
        identity = {
            "kind": "bc_state_dict", "path": str(sd_path),
            "sha256": policy_sha256, "num_timesteps": None,
            "gate_report_sha256": hashlib.sha256(report_payload).hexdigest(),
        }
        net = np_policy_from_sd(policy_payload, policy_sha256)
        return {FARM: lambda obs, mask: net.choose(obs, mask)}, "bc", identity
    if pathlib.Path(spec).suffix.lower() == ".npz":
        path = pathlib.Path(spec).resolve()
        manager_class, _env_class, _bridge = _native_runtime()
        net = manager_class(path)  # 单次读取：同一字节串用于身份与前向。
        identity = {
            "kind": "numpy_policy", "path": str(path),
            "sha256": net.source_sha256, "num_timesteps": None,
            "gate_report_sha256": None,
        }
        return ({FARM: lambda obs, mask: net.choose(obs, mask)},
                pathlib.Path(spec).parent.name, identity)
    from sb3_contrib import MaskablePPO
    checkpoint = resolve_checkpoint_file(spec)
    checkpoint_payload = checkpoint.read_bytes()
    identity = {
        "kind": "sb3_checkpoint", "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        "num_timesteps": checkpoint_num_timesteps_bytes(
            checkpoint_payload, str(checkpoint)),
        "gate_report_sha256": None,
    }
    model = MaskablePPO.load(io.BytesIO(checkpoint_payload), device="cpu")

    def w(obs, mask):
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(a)
    return {FARM: w}, pathlib.Path(spec).stem, identity


def parse_seeds(s: str):
    try:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("种子范围须为 LO-HI(例如 7000-7031)")
    if lo < 0 or hi < lo or hi > UINT32_MAX:
        raise argparse.ArgumentTypeError(
            f"种子范围须满足 0 <= LO <= HI <= {UINT32_MAX}")
    return list(range(lo, hi + 1))


def safe_tag(tag: str) -> str:
    """档案标签只能是文件名，禁止路径穿越或意外写到 OUTDIR 之外。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) or tag in {".", ".."}:
        raise argparse.ArgumentTypeError("tag 只能包含字母、数字、点、下划线和连字符")
    return tag


def evaluate(workers, seeds, manager_npz=None, manager_sha256=None):
    manager_class, env_class, _bridge = _native_runtime()
    manager_path = pathlib.Path(manager_npz) if manager_npz else NPZ
    expected_manager_sha256 = manager_sha256 or (NPZ_SHA if not manager_npz else None)
    mgr = manager_class(
        str(manager_path), expected_sha256=expected_manager_sha256)
    # evaluate() 也会被测试/法证脚本在同一进程重复调用。不要原地包裹调用方的
    # workers 字典，否则第二次调用会叠加 instrumentation，甚至闭包到旧 env。
    active_workers = dict(workers) if workers else None
    env = env_class(max_steps=3000, workers=active_workers)
    engage = None
    if active_workers:
        # 参与度取证(2026-07-10 法证会审的后续):调用数/动作直方/与脚本分歧率,
        # 让评测文件自带"worker 真的在开车"的证据,顺带量出 PPO 漂离教师的距离
        from diablogym.options_env import dispatch
        inner = active_workers[FARM]
        engage = {"calls": 0, "hist": Counter(), "diverge": 0}

        def instrumented(obs, mask, _inner=inner):
            a = int(_inner(obs, mask))
            engage["calls"] += 1
            engage["hist"][int(a)] += 1
            s = dispatch("farm", env.env._raw, bool(env.env.action_masks()[14]))
            if a != s:
                engage["diverge"] += 1
            return a

        active_workers[FARM] = instrumented   # env 持同一 dict 引用,替换副本生效
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            R = 0.0
            farm = Counter()
            allw = Counter()
            seq = ""
            while not (done or trunc):
                opt = mgr.choose(obs, env.action_masks())
                obs, r, done, trunc, info = env.step(opt)
                R += float(r)
                oe = info["option_extra"]
                allw["n"] += 1
                allw["beats"] += oe["beats"]
                allw["overrides"] += oe["overrides"]
                allw["cap"] += oe["reason"] == "cap"
                if oe["opt"] == FARM:
                    farm["n"] += 1
                    farm["tau"] += oe["tau"]
                    farm["descend"] += oe["reason"] == "descend"
                seq = oe["mode_seq"]
            raw = env.env._raw
            rows.append({
                "seed": seed, "ret": round(R, 2),
                "depth": raw["dungeon_level"],
                "died": bool(raw.get("dead")), "kills": env.env._ep_kills,
                "farm_n": farm["n"],
                "farm_tau_mean": round(
                    farm["tau"] / max(1, farm["n"]), 1),
                # 汇总必须使用未舍入的分子；按局先舍入再加权会在 R4 阈值附近翻闸。
                "farm_tau_sum": farm["tau"],
                "farm_descend": farm["descend"], "windows": allw["n"],
                "beats": allw["beats"], "overrides": allw["overrides"],
                "cap": allw["cap"], "mode_seq": seq,
            })
            print(f"  seed {seed}: ret {R:.1f} depth {raw['dungeon_level']} "
                  f"died {bool(raw.get('dead'))} "
                  f"farmτ̄ {rows[-1]['farm_tau_mean']}", flush=True)
    finally:
        try:
            env.close()
        finally:
            # OptionsEnv 持有 active_workers，而 instrumentation 闭包又捕获
            # env；主动清空评测专用副本，打断 env→dict→fn→env 环。调用方
            # 原 workers 从未被修改。
            if active_workers is not None:
                active_workers.clear()
    return rows, engage


def digest(rows):
    return recompute_agg(rows)


def compare_probe_rows(rows, reference_rows):
    """G0'' 对账；两侧 seed 必须精确为同一个集合，禁止子集假 PASS。"""
    current = {row["seed"]: row for row in rows}
    reference = {row["seed"]: row for row in reference_rows}
    if len(current) != len(rows):
        raise ValueError("G0'' 当前评测含重复 seed")
    if len(reference) != len(reference_rows):
        raise ValueError("G0'' 参考档案含重复 seed")
    if set(current) != set(reference):
        missing = sorted(set(current) - set(reference))
        extra = sorted(set(reference) - set(current))
        raise ValueError(f"G0'' seed 集合不精确一致:参考缺 {missing},参考多 {extra}")
    bad = []
    for seed in sorted(current):
        row, expected = current[seed], reference[seed]
        if (abs(row["ret"] - expected["ep_R"]) > 0.6
                or row["depth"] != expected["depth"]
                or row["died"] != expected["died"]):
            bad.append((seed, row["ret"], expected["ep_R"],
                        row["depth"], expected["depth"]))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, help="script | bc | *.npz | ckpt 路径")
    ap.add_argument("--manager-npz", default=None,
                    help="v25:经理 npz(默认 v22-h——旧档回归口径不变)")
    ap.add_argument("--seeds", type=parse_seeds, default=parse_seeds("7000-7031"))
    ap.add_argument("--board", action="store_true",
                    help=f"金评后写 {LB.name}（旧 leaderboard-hier.md 只读）")
    ap.add_argument("--check-probe", default=None,
                    help="G0'':对 v23 前探针存档逐种子回归(ret±0.6/depth/died 全等)")
    ap.add_argument("--tag", type=safe_tag, default=None)
    args = ap.parse_args()

    # CLI 必须是尚未映射原生扩展的新进程；否则磁盘 SHA 无法证明已映射页的
    # 字节身份。所有官方驱动均以子进程启动本入口。
    if "_diablogym" in sys.modules:
        raise EvalContractError(
            "eval_assembled 必须在未预载 diablogym bridge 的新进程中运行")

    seeds = args.seeds
    seed_label = f"{seeds[0]}-{seeds[-1]}"
    label = worker_label(args.worker)
    tag = args.tag or safe_tag(f"{label}-{seed_label}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{tag}.json"
    if args.board and seeds != list(range(9000, 9032)):
        ap.error("--board 只允许金评种子 9000-9031，拒绝用探针池污染排行榜")
    board_contract = assembled_board_contract() if args.board else None
    if board_contract is not None:
        ensure_leaderboard_compatible(
            LB, board_contract, initial_text=ASSEMBLED_LEADERBOARD_HEADER)
    board_guard = (exclusive_lock(OUTDIR / ".gold-evaluation.lock", "金评发车")
                   if args.board else contextlib.nullcontext())
    try:
        # 从发车前一直持有到原子 commit 与读后复验完成；同 tag 的并发进程
        # 无法同时越过 exists 检查再互相 os.replace。
        with reserve_output(out_path), board_guard:
            bridge_path = bridge_binary_path(ROOT)
            runtime = runtime_identity(ROOT, bridge_path)
            if (board_contract is not None
                    and board_contract["runtime"] != runtime):
                raise EvalContractError("组装体榜合同与评测 runtime 冻结结果不一致")
            _native_runtime(runtime)
            workers, loaded_label, worker_id = load_worker(
                args.worker, runtime["python_protocol"]["sha256"])
            if loaded_label != label:
                raise EvalContractError("worker 标签推导与加载结果不一致")
            manager_path = pathlib.Path(args.manager_npz).resolve() if args.manager_npz else NPZ
            manager_id = file_identity("numpy_policy", manager_path)
            if not args.manager_npz and manager_id["sha256"] != NPZ_SHA:
                raise EvalContractError(
                    f"默认经理 npz sha 漂移:{manager_id['sha256']} != {NPZ_SHA}")

            # 使用身份清单中已经 resolve 的同一文件，避免自定义 symlink 在
            # 哈希与实际 NumpyManager.load 之间改指向。
            rows, engage = evaluate(
                workers, seeds, manager_npz=str(manager_path),
                manager_sha256=manager_id["sha256"])
            agg = digest(rows)
            if engage:
                agg["worker_calls"] = engage["calls"]
                agg["worker_action_hist"] = dict(sorted(engage["hist"].items()))
                agg["worker_divergences"] = engage["diverge"]
                agg["script_divergence_rate"] = round(
                    engage["diverge"] / max(1, engage["calls"]), 4)
                print(f"  参与度:worker 调用 {engage['calls']},"
                      f"动作直方 {agg['worker_action_hist']},"
                      f"与脚本分歧率 {agg['script_divergence_rate']}")
            print(f"{tag}: ret {agg['ret_mean']} (med {agg['ret_median']}) "
                  f"died {agg['died']}/{agg['n']} depth_med {agg['depth_median']} | "
                  f"R4: 换层率 {agg['farm_descend_rate']} override {agg['override_rate']} "
                  f"cap {agg['cap_rate']} farmτ̄ {agg['farm_tau_mean']}")

            if args.check_probe:
                probe_doc = strict_json_loads(
                    pathlib.Path(args.check_probe).read_text(encoding="utf-8"))
                try:
                    ref_rows = probe_doc["argmax_episodes"]
                except (KeyError, TypeError) as exc:
                    raise ValueError("G0'' 参考档案缺少 argmax_episodes") from exc
                bad = compare_probe_rows(rows, ref_rows)
                print(f"G0'' 回归:{len(rows) - len(bad)}/{len(rows)} 一致"
                      + (f";失配 {bad}" if bad else " —— PASS"))
                if bad:
                    raise SystemExit(1)

            # 文件型输入和运行时代码在长评测期间若被替换，本档案必须作废。
            verify_file_identity(worker_id)
            verify_file_identity(manager_id)
            if runtime_identity(ROOT, bridge_path) != runtime:
                raise EvalContractError("评测期间 bridge、engine 或协议源码发生变化")

            document = {
                "schema_version": 2,
                "meta": make_meta(tag=tag, seeds=seeds, worker=worker_id,
                                  manager=manager_id, runtime=runtime),
                "agg": agg,
                "rows": rows,
            }
            expected = expected_eval_identity(
                {"worker": worker_id, "manager": manager_id, "runtime": runtime},
                tag=tag, seeds=seeds)
            validate_eval_archive(document, **expected)
            payload = json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False)

            # 同目录临时文件 + fsync + os.replace：SIGKILL 不留下半截正式 JSON。
            tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, out_path)
            finally:
                tmp.unlink(missing_ok=True)
            try:
                loaded = read_eval_archive(out_path, **expected)
                verify_file_identity(worker_id)
                verify_file_identity(manager_id)
                if runtime_identity(ROOT, bridge_path) != runtime:
                    raise EvalContractError(
                        "档案提交期间 bridge、engine 或协议源码发生变化")
            except Exception:
                out_path.rename(out_path.with_suffix(f".{time.time_ns()}.void"))
                raise
            agg = loaded["agg"]
            print(f"已存并复验 {out_path}")

            if args.board:
                archive_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
                row_key = versioned_row_key(tag, archive_sha)
                visible = (
                    f"| {row_key} | {agg['ret_mean']} | {agg['ret_median']} | "
                    f"{agg['died']}/{agg['n']} | {agg['depth_median']} | "
                    f"worker={worker_id['kind']}; L3+ {agg['l3']}; "
                    f"kills {agg['kills_mean']}; 换层率 {agg['farm_descend_rate']}; "
                    f"override {agg['override_rate']} |")
                row = assembled_leaderboard_row(
                    visible, row_key=row_key, contract=board_contract,
                    archive_path=str(out_path), archive_sha256=archive_sha,
                    worker_sha256=worker_id["sha256"],
                    manager_sha256=manager_id["sha256"])
                upsert_leaderboard_rows(
                    LB, {row_key: row}, contract=board_contract,
                    initial_text=ASSEMBLED_LEADERBOARD_HEADER,
                    lock_path=LB_LOCK)
                print(f"已写入 {LB.name}")
    except OutputReservationError as exc:
        ap.error(f"{exc}（请等待并发评测结束，或按重启协议归档旧结果）")


if __name__ == "__main__":
    main()
