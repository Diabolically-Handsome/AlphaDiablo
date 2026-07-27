"""v22 发车前探针:G1b 机械保真 / G2 词表充分性 / G3 评估段参考行 / G4 预算校准。

用法:.venv/bin/python train/probe_options.py --oracle /path/to/oracle_mountain.json
默认只写诊断 JSON；全部闸通过后显式加 --write-board 才更新排行榜。

闸门(预注册):
  G1b wrapper-rush(恒 DIVE)对神谕 rush 臂逐种子 |Δ| ≤ max(5%, 1.0)
  G2  教师(榨干旗或 clvl≥dlvl+2 → DIVE)均值 ≥36 且 配对胜 wrapper-retire ≥24/32
  G3  三参考臂 9000-9031 成绩写入当前协议版本的 hierarchy leaderboard
  G4  τ̄ 与吞吐实测 → 双币种停车规则定数
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from eval_contract import (OutputReservationError, reserve_output, sha256_file,
                           strict_json_loads)
from evaluate import (atomic_write_text, contract_sha256,
                      ensure_leaderboard_compatible, scripted_leaderboard_row,
                      require_fresh_native_runtime,
                      upsert_leaderboard_rows, validated_episode_extra,
                      verify_loaded_native_runtime, verify_standalone_contract,
                      versioned_row_key)
from evaluate_options import (LB, LB_LOCK, LEADERBOARD_HEADER,
                              hierarchy_contract)

PROBE_SEEDS = list(range(7000, 7032))
EVAL_SEEDS = list(range(9000, 9032))
# 不能在冻结 runtime contract 前导入 diablogym：原生扩展/engine 一旦映射，
# 随后只哈希磁盘路径就可能把“已加载旧字节”误记成“磁盘新字节”。数值是
# 评估协议的一部分；真实导入后还会逐项核对 options_env 的公开常量。
FARM, DIVE = 0, 1
OptionsEnv = None


def _masked_option_or_first_legal(requested, mask) -> int:
    """Keep a legal proposal or deterministically choose the first legal option."""
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != (3,):
        raise ValueError(
            f"probe option 动作掩码形状异常:{valid.shape} != (3,)")
    legal = np.flatnonzero(valid)
    if len(legal) == 0:
        raise ValueError("probe option 动作掩码全假")
    if (
        isinstance(requested, (int, np.integer))
        and not isinstance(requested, (bool, np.bool_))
    ):
        candidate = int(requested)
        if 0 <= candidate < 3 and bool(valid[candidate]):
            return candidate
    return int(legal[0])


def _options_env_class():
    if OptionsEnv is not None:  # 单元测试显式注入；生产默认为 None。
        return OptionsEnv
    from diablogym import OptionsEnv as env_class
    from diablogym.options_env import DIVE as actual_dive, FARM as actual_farm

    if (actual_farm, actual_dive) != (FARM, DIVE):
        raise RuntimeError(
            "OptionsEnv 选项编号与 probe protocol 不一致:"
            f"actual={(actual_farm, actual_dive)}, expected={(FARM, DIVE)}")
    return env_class


def run_policy(env, choose, seed):
    obs, _ = env.reset(seed=seed)
    done = trunc = False
    R, taus, info = 0.0, [], {}
    while not (done or trunc):
        m = env.action_masks()
        opt = _masked_option_or_first_legal(choose(env, m), m)
        obs, r, done, trunc, info = env.step(opt)
        R += r
        taus.append(info["option_extra"]["tau"])
    ex = validated_episode_extra(info, seed)
    if not math.isfinite(float(R)):
        raise RuntimeError(f"seed {seed} 累计回报含 NaN/Inf")
    oe = info.get("option_extra")
    if not isinstance(oe, dict) or "mode_seq" not in oe:
        raise RuntimeError(f"seed {seed} 缺少完整 option_extra")
    return {"ret": round(R, 2), "depth": ex["depth"],
            "died": ex["died"], "kills": ex["kills"],
            "decisions": len(taus), "tau_mean": round(sum(taus) / max(1, len(taus)), 1),
            "tau_sum": sum(taus),
            "mode_seq": oe["mode_seq"]}


POLICIES = {
    "wrapper-retire": lambda env, m: FARM,
    "wrapper-rush": lambda env, m: DIVE,
    "teacher": lambda env, m: DIVE if (env.exhausted or
                                       env.env._raw["char_level"] >= env.env._raw["dungeon_level"] + 2)
                              else FARM,
}


def validate_oracle_rush(oracle) -> dict[int, float]:
    """冻结 G1b 所需的精确 seed→return 映射；重复/额外/非有限值全拒绝。"""
    try:
        entries = oracle["arms"]["rush"]
    except (KeyError, TypeError) as exc:
        raise ValueError("oracle 缺少 arms.rush") from exc
    if not isinstance(entries, list):
        raise ValueError("oracle arms.rush 必须是列表")
    by_seed: dict[int, float] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "error" in entry:
            continue
        try:
            seed = entry["seed"]
            value = entry["snaps"]["3000"]["ret"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"oracle rush[{index}] 结构异常") from exc
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"oracle rush[{index}].seed 非整数")
        if seed in by_seed:
            raise ValueError(f"oracle rush 含重复 seed:{seed}")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"oracle seed {seed} ret 非数值")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"oracle seed {seed} ret 非有限")
        by_seed[seed] = value
    expected = set(PROBE_SEEDS)
    if set(by_seed) != expected:
        missing = sorted(expected - set(by_seed))
        extra = sorted(set(by_seed) - expected)
        raise ValueError(f"oracle rush seed 集合异常:missing={missing},extra={extra}")
    return by_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", type=pathlib.Path, required=True,
                    help="oracle_mountain.json 路径（不再依赖作者机器的 /private/tmp）")
    ap.add_argument("--output", type=pathlib.Path,
                    default=ROOT / "train" / "runs" / "probes" / "probe_v22.json")
    ap.add_argument("--write-board", action="store_true",
                    help="全部闸门 PASS 后更新 9000 段参考行；默认只产出诊断 JSON")
    args = ap.parse_args()
    require_fresh_native_runtime("probe_options.py")
    output_path = args.output.resolve()
    if output_path == args.oracle.resolve():
        ap.error("--output 不能覆盖 --oracle")
    if output_path in {LB.resolve(), LB_LOCK.resolve()}:
        ap.error("--output 不能覆盖当前协议排行榜或其锁文件")
    try:
        oracle_payload = args.oracle.read_bytes()
        oracle = strict_json_loads(oracle_payload)
        oracle_rush = validate_oracle_rush(oracle)
        oracle_sha256 = hashlib.sha256(oracle_payload).hexdigest()
    except (OSError, ValueError) as e:
        ap.error(f"无法读取 oracle: {e}")

    contract = hierarchy_contract()
    if args.write_board:
        ensure_leaderboard_compatible(
            LB, contract, initial_text=LEADERBOARD_HEADER)
    try:
        # probe 明细也是发布证据：同一目标只允许创建一次，避免旧档案被静默
        # 替换；需要重跑时显式选择新的 --output 或先归档旧文件。
        with reserve_output(args.output):
            env = _options_env_class()(max_steps=3000)
            verify_loaded_native_runtime(contract)
            try:
                out, g1b, g2 = _collect_probe(
                    oracle_rush, env, args.oracle.resolve(), oracle_sha256,
                    contract)
            finally:
                # close 失败也必须发生在任何 JSON/排行榜正式发布之前。
                env.close()
            if (out.get("meta", {}).get("contract") != contract
                    or out.get("meta", {}).get("contract_sha256")
                    != contract_sha256(contract)):
                raise RuntimeError("probe 结果未绑定发车前 standalone contract")
            verify_standalone_contract(contract)
            return _publish_probe(args, out, g1b, g2)
    except OutputReservationError as exc:
        ap.error(str(exc))


def _collect_probe(oracle_rush: dict[int, float], env, oracle_path: pathlib.Path,
                   oracle_sha256: str, contract: dict) -> tuple[dict, bool, bool]:
    out = {
        "meta": {
            "oracle_path": str(oracle_path), "oracle_sha256": oracle_sha256,
            "contract": contract, "contract_sha256": contract_sha256(contract),
        },
        "probe": {}, "eval_refs": {},
    }
    t_wall0 = time.time()

    # ---- 探针段 7000-7031 ----
    for name, pol in POLICIES.items():
        eps = []
        for seed in PROBE_SEEDS:
            eps.append({"seed": seed, **run_policy(env, pol, seed)})
        out["probe"][name] = eps
        rs = [e["ret"] for e in eps]
        print(f"[probe] {name}: mean {sum(rs)/32:.1f} med {statistics.median(rs):.1f} "
              f"died {sum(e['died'] for e in eps)}/32 "
              f"depth_med {statistics.median(e['depth'] for e in eps)} "
              f"decisions_med {statistics.median(e['decisions'] for e in eps)}", flush=True)

    # G1b:wrapper-rush vs 神谕 rush(3000 快照)
    fails = []
    for e in out["probe"]["wrapper-rush"]:
        ref = oracle_rush[e["seed"]]
        if abs(e["ret"] - ref) > max(0.05 * abs(ref), 1.0):
            fails.append((e["seed"], e["ret"], ref))
    g1b = len(fails) == 0
    print(f"G1b {'PASS' if g1b else 'FAIL'}: wrapper-rush 对神谕 rush 逐种子偏差超限 {len(fails)}/32 "
          + (f"首例 {fails[0]}" if fails else ""), flush=True)

    # G2:教师充分性
    t_rets = [e["ret"] for e in out["probe"]["teacher"]]
    r_rets = [e["ret"] for e in out["probe"]["wrapper-retire"]]
    t_mean = sum(t_rets) / 32
    retire_by_seed = {e["seed"]: e["ret"] for e in out["probe"]["wrapper-retire"]}
    wins = sum(e["ret"] > retire_by_seed[e["seed"]] for e in out["probe"]["teacher"])
    g2 = t_mean >= 36 and wins >= 24
    grey = 34 <= t_mean < 36
    print(f"G2 {'PASS' if g2 else ('GREY' if grey else 'FAIL')}: 教师均值 {t_mean:.1f}(线 36,灰带 [34,36)) "
          f"配对胜 retire {wins}/32(线 24)", flush=True)

    # ---- G3:评估段参考行 9000-9031 ----
    for name, pol in POLICIES.items():
        eps = [{"seed": s, **run_policy(env, pol, s)} for s in EVAL_SEEDS]
        out["eval_refs"][name] = eps
        rs = [e["ret"] for e in eps]
        print(f"[eval] {name}: mean {sum(rs)/32:.1f} died {sum(e['died'] for e in eps)}/32 "
              f"depth_med {statistics.median(e['depth'] for e in eps)}", flush=True)

    # G4:预算校准
    wall = time.time() - t_wall0
    all_probe = [e for eps in out["probe"].values() for e in eps]
    all_eps = all_probe + [e for eps in out["eval_refs"].values() for e in eps]
    tau_bar = sum(e["tau_sum"] for e in all_probe) / max(1, sum(e["decisions"] for e in all_probe))
    total_micro = sum(e["tau_sum"] for e in all_eps)
    micro_per_s = total_micro / wall
    print(f"G4: τ̄≈{tau_bar:.0f} 微拍/选项,吞吐≈{micro_per_s:.0f} micro/s(单 env),"
          f"40k 管理器步 ≈ {40_000 * tau_bar / 1e6:.1f}M 微步;"
          f"4-env 预计墙钟 ≈ {40_000 * tau_bar / (micro_per_s * 2.5) / 60:.0f} 分钟", flush=True)

    return out, g1b, g2


def _publish_probe(args, out: dict, g1b: bool, g2: bool) -> int:
    try:
        contract = out["meta"]["contract"]
        oracle_path = out["meta"]["oracle_path"]
        expected_oracle_sha = out["meta"]["oracle_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("probe 输出缺少完整 provenance") from exc
    verify_standalone_contract(contract)
    try:
        current_oracle_sha = sha256_file(oracle_path)
    except OSError as exc:
        raise RuntimeError("probe 发布前 oracle 不可读") from exc
    if current_oracle_sha != expected_oracle_sha:
        raise RuntimeError("probe oracle 在诊断 JSON 发布前发生变化")
    payload = json.dumps(out, default=float, allow_nan=False, sort_keys=True)
    if args.output.exists():
        raise OutputReservationError(f"探针档案已存在，拒绝覆写:{args.output}")
    atomic_write_text(args.output, payload)
    print(f"探针明细已存 {args.output}", flush=True)

    if not (g1b and g2):
        print("闸门未全部 PASS：拒绝写排行榜", flush=True)
        return 1

    # 评估段参考行入新表
    if not args.write_board:
        print("G3: 未指定 --write-board，排行榜未改动", flush=True)
        print("GATES: G1b=PASS G2=PASS", flush=True)
        return 0
    contract = out["meta"]["contract"]
    oracle_sha256 = out["meta"]["oracle_sha256"]
    rows = {}
    for name in POLICIES:
        eps = out["eval_refs"][name]
        rs = sorted(e["ret"] for e in eps)
        label = f"{name} (scripted ref)"
        result_sha256 = hashlib.sha256(json.dumps(
            eps, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")).hexdigest()
        scripted_identity = hashlib.sha256(
            f"{oracle_sha256}\0{result_sha256}".encode("ascii")).hexdigest()
        key = versioned_row_key(label, scripted_identity)
        visible = (f"| {key} | {sum(rs)/32:.1f} | "
                   f"{statistics.median(rs):.1f} | "
                   f"{sum(e['died'] for e in eps)}/32 | "
                   f"{statistics.median(e['depth'] for e in eps)} | "
                   "G3 reference |")
        rows[key] = scripted_leaderboard_row(
            visible, row_key=key, contract=contract, policy=name,
            oracle_path=out["meta"]["oracle_path"],
            oracle_sha256=oracle_sha256, result_sha256=result_sha256)
    upsert_leaderboard_rows(
        LB, rows, contract=contract, initial_text=LEADERBOARD_HEADER,
        lock_path=LB_LOCK)
    print(f"G3: 参考行已写入 {LB.name}", flush=True)
    print("GATES: G1b=PASS G2=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
