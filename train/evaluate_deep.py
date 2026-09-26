"""Deep-water chapter 32-seed gold standard (the deep-water variant of the current protocol).

How it differs from the main board (train/evaluate.py), and why the two are not comparable:
  - max_steps 3000 (main board 1500): the armor/potion economy needs a longer accounting period;
  - metrics are depth-first: median deepest level, counts reaching L2/L3/L4, deaths, mean kills;
  - checkpoints are MaskablePPO (the masked stack since v16); predict must pass action_masks
    (the mask is part of the policy distribution; omitting it = a different policy).
Everything else in the protocol is unchanged: seeds 9000-9031 are used only for final evaluation, argmax, idle machine, engine pinned to
ENGINE_REF. Results go to the leaderboard of the current protocol version; old boards are read-only.

Usage (from the repository root):
  .venv/bin/python train/evaluate_deep.py train/runs/<run>/model_final
"""

from __future__ import annotations

import argparse
import io
import pathlib
import statistics as s
import sys
import time
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
LEADERBOARD = (pathlib.Path(__file__).resolve().parent
               / f"leaderboard-deep-v{PROTOCOL_VERSION}.md")
LEADERBOARD_LOCK = ROOT / "train" / "runs" / "eval-locks" / "leaderboard-deep.lock"

DEEP_SOURCE_FILES = ("train/evaluate.py", "train/evaluate_deep.py", "train/models.py")
DEEP_PROTOCOL = {
    "name": "diablogym.standalone.deep",
    "seeds": SEEDS,
    "max_steps": 3000,
    "ticks_per_step": 4,
    "start_in_dungeon": True,
    "hero_class": 0,
    "include_raw": False,
    "descend_ladder": True,
    "death_ladder": False,
    "disable_level_backtracking": True,
    "action_selection": "MaskablePPO deterministic argmax with action masks",
}

LEADERBOARD_HEADER = (
    f"# Deep-water leaderboard protocol v{PROTOCOL_VERSION} — 32 fixed seeds, "
    "3000-step episodes\n\n"
    "Protocol: argmax + action masks, seeds 9000-9031 (never used for\n"
    "training or hyper-parameter selection), 3000 steps/episode, idle\n"
    "machine, engine pinned to `ENGINE_REF` in bootstrap.sh, reward\n"
    "world = depth-progressive descent ladder (8×N per level). NOT\n"
    "comparable to train/leaderboard-v3.md (1500-step episodes).\n"
    "See train/evaluate_deep.py.\n\n"
    "| run | depth med | depth max | ≥L2 | ≥L3 | ≥L4 | deaths | mean kills | median |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def deep_contract() -> dict:
    return freeze_standalone_contract(
        evaluator=f"standalone-deep-v{PROTOCOL_VERSION}", protocol=DEEP_PROTOCOL,
        source_files=DEEP_SOURCE_FILES)


def evaluate(model_path: str, *, contract: Mapping | None = None):
    require_fresh_native_runtime("evaluate_deep.py")
    if contract is None:
        contract = deep_contract()

    from sb3_contrib import MaskablePPO

    from diablogym import DiabloGymEnv
    import models  # noqa: F401  (registers the custom extractor; must be importable at load time)
    verify_loaded_native_runtime(contract)

    checkpoint, payload, model_sha256 = checkpoint_snapshot(model_path)
    model = MaskablePPO.load(io.BytesIO(payload), device="cpu")
    env = DiabloGymEnv(ticks_per_step=4, max_steps=3000,
                       start_in_dungeon=True, include_raw=False,
                       descend_ladder=True)
    kills, depths, deaths = [], [], 0
    t0 = time.time()
    try:
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            info = {}
            while not (done or trunc):
                a, _ = model.predict(obs, action_masks=env.action_masks(),
                                     deterministic=True)
                obs, _reward, done, trunc, info = env.step(int(a))
            ex = validated_episode_extra(info, seed)
            kills.append(ex["kills"])
            depths.append(ex["depth"])
            deaths += ex["died"]
    finally:
        env.close()

    verify_standalone_contract(contract)
    verify_checkpoint_identity(checkpoint, model_sha256)

    return {
        "model": str(checkpoint),
        "model_sha256": model_sha256,
        "depth_median": s.median(depths),
        "depth_max": max(depths),
        "l2": sum(d >= 2 for d in depths),
        "l3": sum(d >= 3 for d in depths),
        "l4": sum(d >= 4 for d in depths),
        "deaths": deaths,
        "kills_mean": round(s.mean(kills), 1),
        "kills_median": s.median(kills),
        "secs": round(time.time() - t0, 1),
        "mode": "masked-deep",
        "contract": contract,
        "contract_sha256": contract_sha256(contract),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    args = ap.parse_args()
    model_path = args.model_path
    contract = deep_contract()
    ensure_leaderboard_compatible(
        LEADERBOARD, contract, initial_text=LEADERBOARD_HEADER)
    r = evaluate(model_path, contract=contract)
    if (r.get("contract") != contract
            or r.get("contract_sha256") != contract_sha256(contract)):
        raise RuntimeError("deep-water evaluation result is not bound to the pre-launch standalone contract")
    name = pathlib.Path(model_path).parent.name or pathlib.Path(model_path).stem
    row_key = versioned_row_key(name, r["model_sha256"])
    visible = (f"| {row_key} | {r['depth_median']} | "
               f"{r['depth_max']} | {r['l2']} | {r['l3']} | {r['l4']} | "
               f"{r['deaths']} | {r['kills_mean']} | {r['kills_median']} |")
    line = model_leaderboard_row(
        visible, row_key=row_key, contract=contract, model_path=r["model"],
        model_sha256=r["model_sha256"], mode=r["mode"])
    print(f"depth median {r['depth_median']} | deepest {r['depth_max']} | "
          f"L2 {r['l2']} | L3 {r['l3']} | L4 {r['l4']} | deaths {r['deaths']} | "
          f"mean kills {r['kills_mean']} | kills median {r['kills_median']}  [{r['secs']}s]")
    upsert_leaderboard_rows(
        LEADERBOARD, {row_key: line}, contract=contract,
        initial_text=LEADERBOARD_HEADER,
        lock_path=LEADERBOARD_LOCK)
    print(f"wrote {LEADERBOARD.name}")


if __name__ == "__main__":
    main()
