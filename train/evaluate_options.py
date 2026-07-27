"""分层/平面双路径版本化评测（榜单版本取自 PROTOCOL_VERSION）。

用法:
  分层臂:.venv/bin/python train/evaluate_options.py train/runs/<run>/model_final --options
  平面臂:.venv/bin/python train/evaluate_options.py train/runs/<run>/model_final --flat-clock
协议:32 种子 9000-9031、argmax+掩码、3000 微步、空载、引擎钉死、
回报 = 未折现局回报(神谕账本口径)。
"""
from __future__ import annotations

import argparse
import io
import math
import pathlib
import statistics
import sys
from collections import Counter
from collections.abc import Mapping

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from evaluate import (checkpoint_snapshot, contract_sha256,
                      ensure_leaderboard_compatible, freeze_standalone_contract,
                      model_leaderboard_row, upsert_leaderboard_rows,
                      require_fresh_native_runtime,
                      validated_episode_extra, verify_checkpoint_identity,
                      verify_loaded_native_runtime, verify_standalone_contract,
                      versioned_row_key)
from eval_contract import PROTOCOL_VERSION

SEEDS = list(range(9000, 9032))
LB = (pathlib.Path(__file__).resolve().parent
      / f"leaderboard-hierarchy-v{PROTOCOL_VERSION}.md")
LB_LOCK = ROOT / "train" / "runs" / "eval-locks" / "leaderboard-hierarchy.lock"

HIERARCHY_SOURCE_FILES = (
    "train/evaluate.py", "train/evaluate_options.py", "train/probe_options.py",
    "train/models.py",
)
HIERARCHY_PROTOCOL = {
    "name": "diablogym.standalone.hierarchy",
    "evaluation_seeds": SEEDS,
    "probe_seeds": list(range(7000, 7032)),
    "max_micro_steps": 3000,
    "ticks_per_step": 4,
    "start_in_dungeon": True,
    "hero_class": 0,
    "include_raw": False,
    "descend_ladder": True,
    "death_ladder": True,
    "disable_level_backtracking": True,
    "modes": ["options-manager", "flat-stagnation-clock", "scripted-reference"],
    "action_selection": "deterministic argmax with action/option masks",
    "reward": "undiscounted episode ledger",
}

LEADERBOARD_HEADER = (
    f"# Hierarchy standalone board protocol v{PROTOCOL_VERSION} — 32 fixed seeds\n\n"
    "Protocol: 3000 micro-steps, argmax + option masks, seeds 9000-9031,\n"
    "idle machine, engine pinned, world = v20 rules (ladder + death price\n"
    "+ auto stat-spend). Return = UNDISCOUNTED episode reward (the oracle\n"
    "ledger). Reference rows are scripted policies via the same wrapper.\n\n"
    "| run | ret mean | ret med | died | depth med | notes |\n"
    "|---|---|---|---|---|---|\n"
)


def hierarchy_contract() -> dict:
    return freeze_standalone_contract(
        evaluator=f"standalone-hierarchy-v{PROTOCOL_VERSION}",
        protocol=HIERARCHY_PROTOCOL,
        source_files=HIERARCHY_SOURCE_FILES)


def evaluate(model_path: str, hier: bool, *,
             contract: Mapping | None = None):
    require_fresh_native_runtime("evaluate_options.py")
    if contract is None:
        contract = hierarchy_contract()

    from sb3_contrib import MaskablePPO

    from diablogym import DiabloGymEnv, OptionsEnv, StagnationClockWrapper
    verify_loaded_native_runtime(contract)

    checkpoint, payload, model_sha256 = checkpoint_snapshot(model_path)
    model = MaskablePPO.load(io.BytesIO(payload), device="cpu")
    if hier:
        env = OptionsEnv(max_steps=3000)
    else:
        env = StagnationClockWrapper(DiabloGymEnv(
            ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
            include_raw=False, descend_ladder=True, death_ladder=True))
    rows = []
    opt_share = Counter()
    reasons = Counter()
    try:
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            R, info = 0.0, {}
            seq = ""
            while not (done or trunc):
                a, _ = model.predict(
                    obs, action_masks=env.action_masks(), deterministic=True)
                obs, reward, done, trunc, info = env.step(int(a))
                R += float(reward)
                if hier:
                    oe = info["option_extra"]
                    opt_share[int(a)] += 1
                    reasons[oe["reason"]] += 1
                    seq = oe["mode_seq"]
            ex = validated_episode_extra(info, seed)
            if not math.isfinite(R):
                raise RuntimeError(f"seed {seed} 累计回报含 NaN/Inf")
            raw = env.env._raw
            rows.append({"seed": seed, "ret": round(R, 2),
                         "depth": ex["depth"],
                         "died": ex["died"],
                         "kills": ex["kills"],
                         "belt_at_end": raw.get("belt_heals", 0),
                         "mode_seq": seq})
    finally:
        env.close()
    verify_standalone_contract(contract)
    verify_checkpoint_identity(checkpoint, model_sha256)
    rets = sorted(r["ret"] for r in rows)

    def farm_before_dive(seq: str) -> bool:
        clean = seq.replace("†", "")
        return "F" in clean and "D" in clean and clean.index("F") < clean.index("D")

    agg = {
        "ret_mean": round(sum(rets) / len(rows), 1),
        "ret_median": statistics.median(rets),
        "died": sum(r["died"] for r in rows),
        "depth_median": statistics.median(r["depth"] for r in rows),
        "l3": sum(r["depth"] >= 3 for r in rows),
        "kills_mean": round(sum(r["kills"] for r in rows) / len(rows), 1),
        "spiral_seqs": sum(farm_before_dive(r["mode_seq"]) for r in rows) if hier else None,
        "opt_share": {k: v for k, v in opt_share.items()},
        "reasons": dict(reasons),
        "model": str(checkpoint),
        "model_sha256": model_sha256,
        "mode": "options-manager" if hier else "flat-stagnation-clock",
        "contract": contract,
        "contract_sha256": contract_sha256(contract),
    }
    return agg, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--options", action="store_true", help="评估分层经理")
    mode.add_argument("--flat-clock", action="store_true", help="评估平面+停滞钟策略")
    args = ap.parse_args()
    model_path = args.model_path
    hier = args.options
    contract = hierarchy_contract()
    ensure_leaderboard_compatible(
        LB, contract, initial_text=LEADERBOARD_HEADER)
    agg, rows = evaluate(model_path, hier, contract=contract)
    if (agg.get("contract") != contract
            or agg.get("contract_sha256") != contract_sha256(contract)):
        raise RuntimeError("层级评估结果未绑定发车前 standalone contract")
    name = pathlib.Path(model_path).parent.name or pathlib.Path(model_path).stem
    row_key = versioned_row_key(name, agg["model_sha256"])
    print(f"{name}: ret {agg['ret_mean']} (med {agg['ret_median']}) died {agg['died']}/32 "
          f"depth_med {agg['depth_median']} L3+ {agg['l3']} kills {agg['kills_mean']}")
    if hier:
        print(f"  选项份额 {agg['opt_share']} 终止原因 {agg['reasons']} 螺旋序列局数 {agg['spiral_seqs']}/32")
    note = ("hier" if hier else "flat+clock")
    visible = (f"| {row_key} | {agg['ret_mean']} | {agg['ret_median']} | "
               f"{agg['died']}/32 | {agg['depth_median']} | {note}; "
               f"L3+ {agg['l3']}; kills {agg['kills_mean']} |")
    row = model_leaderboard_row(
        visible, row_key=row_key, contract=contract, model_path=agg["model"],
        model_sha256=agg["model_sha256"], mode=agg["mode"])
    upsert_leaderboard_rows(
        LB, {row_key: row}, contract=contract,
        initial_text=LEADERBOARD_HEADER, lock_path=LB_LOCK)
    print(f"已写入 {LB.name}")


if __name__ == "__main__":
    main()
