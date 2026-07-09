"""v22 分层/平面双路径金标准评测(写 leaderboard-hier.md)。

用法:
  分层臂:.venv/bin/python train/evaluate_options.py train/runs/<run>/model_final --options
  平面臂:.venv/bin/python train/evaluate_options.py train/runs/<run>/model_final --flat-clock
协议:32 种子 9000-9031、argmax+掩码、3000 微步、空载、引擎钉死、
回报 = 未折现局回报(神谕账本口径)。
"""
from __future__ import annotations

import pathlib
import statistics
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SEEDS = list(range(9000, 9032))
LB = pathlib.Path(__file__).resolve().parent / "leaderboard-hier.md"


def evaluate(model_path: str, hier: bool):
    from sb3_contrib import MaskablePPO

    from diablogym import DiabloGymEnv, OptionsEnv, StagnationClockWrapper

    model = MaskablePPO.load(model_path, device="cpu")
    if hier:
        env = OptionsEnv(max_steps=3000)
    else:
        env = StagnationClockWrapper(DiabloGymEnv(
            ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
            include_raw=False, descend_ladder=True, death_ladder=True))
    rows = []
    opt_share = Counter()
    reasons = Counter()
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        R, info = 0.0, {}
        seq = ""
        while not (done or trunc):
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, r, done, trunc, info = env.step(int(a))
            R += float(r)
            if hier:
                oe = info["option_extra"]
                opt_share[int(a)] += 1
                reasons[oe["reason"]] += 1
                seq = oe["mode_seq"]
        ex = info.get("episode_extra", {})
        raw = env.env._raw if hier else env.env._raw
        rows.append({"seed": seed, "ret": round(R, 2),
                     "depth": ex.get("depth", raw["dungeon_level"]),
                     "died": bool(ex.get("died", False)),
                     "kills": ex.get("kills", 0),
                     "belt_at_end": raw.get("belt_heals", 0),
                     "mode_seq": seq})
    rets = sorted(r["ret"] for r in rows)
    agg = {
        "ret_mean": round(sum(rets) / 32, 1), "ret_median": rets[16],
        "died": sum(r["died"] for r in rows),
        "depth_median": statistics.median(r["depth"] for r in rows),
        "l3": sum(r["depth"] >= 3 for r in rows),
        "kills_mean": round(sum(r["kills"] for r in rows) / 32, 1),
        "spiral_seqs": sum(1 for r in rows
                           if "F" in r["mode_seq"] and "D" in r["mode_seq"]
                           and r["mode_seq"].index("F") < r["mode_seq"].replace("†", "").index("D")) if hier else None,
        "opt_share": {k: v for k, v in opt_share.items()},
        "reasons": dict(reasons),
    }
    return agg, rows


def main():
    model_path = sys.argv[1]
    hier = "--options" in sys.argv
    agg, rows = evaluate(model_path, hier)
    name = pathlib.Path(model_path).parent.name
    print(f"{name}: ret {agg['ret_mean']} (med {agg['ret_median']}) died {agg['died']}/32 "
          f"depth_med {agg['depth_median']} L3+ {agg['l3']} kills {agg['kills_mean']}")
    if hier:
        print(f"  选项份额 {agg['opt_share']} 终止原因 {agg['reasons']} 螺旋序列局数 {agg['spiral_seqs']}/32")
    lines = LB.read_text().splitlines(keepends=True)
    last_row = max(i for i, l in enumerate(lines) if l.startswith("|"))
    note = ("hier" if hier else "flat+clock")
    lines.insert(last_row + 1,
                 f"| {name} | {agg['ret_mean']} | {agg['ret_median']} | {agg['died']}/32 | "
                 f"{agg['depth_median']} | {note}; L3+ {agg['l3']}; kills {agg['kills_mean']} |\n")
    LB.write_text("".join(lines))
    print(f"已写入 {LB.name}")


if __name__ == "__main__":
    main()
