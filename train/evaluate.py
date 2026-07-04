"""32 种子标准评估(评估制度 v2,2026-07-05 起为唯一金标准)。

教训:8 种子的运气波动曾把 run6/run8 高估近一倍(15.6/13.2 → 8.8/8.4)。
种子集固定为 9000-9031,永不用于训练超参挑选之外的用途。

用法:
  ../.venv/bin/python train/evaluate.py train/runs/<run>/model_final
  自动识别 RecurrentPPO(路径含 lstm)与自定义特征提取器;结果追加进 leaderboard.md
"""

from __future__ import annotations

import pathlib
import statistics as s
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SEEDS = list(range(9000, 9032))
LEADERBOARD = pathlib.Path(__file__).resolve().parent / "leaderboard.md"


def evaluate(model_path: str, recurrent: bool | None = None):
    from diablogym import DiabloGymEnv
    import models  # noqa: F401  (注册自定义提取器,load 时需要可导入)

    if recurrent is None:
        recurrent = "lstm" in model_path.lower()
    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path, device="cpu")
    else:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device="cpu")

    env = DiabloGymEnv(ticks_per_step=4, max_steps=1500,
                       start_in_dungeon=True, include_raw=False)
    kills, zeros, depth2 = [], 0, 0
    t0 = time.time()
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        st = None
        ep_start = np.ones((1,), dtype=bool)
        done = trunc = False
        info = {}
        while not (done or trunc):
            if recurrent:
                a, st = model.predict(obs, state=st, episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(int(a))
        ex = info.get("episode_extra", {})
        k = ex.get("kills", 0)
        kills.append(k)
        zeros += (k == 0)
        depth2 += (ex.get("depth", 1) >= 2)

    result = {
        "model": model_path,
        "mean": round(s.mean(kills), 1),
        "median": s.median(kills),
        "max": max(kills),
        "zero": f"{zeros}/{len(SEEDS)}",
        "depth2": depth2,
        "secs": round(time.time() - t0, 1),
    }
    return result


def main():
    model_path = sys.argv[1]
    r = evaluate(model_path)
    line = (f"| {pathlib.Path(model_path).parent.name} | {r['mean']} | {r['median']} | "
            f"{r['max']} | {r['zero']} | {r['depth2']} |")
    print(f"均击杀 {r['mean']} | 中位 {r['median']} | 最高 {r['max']} | "
          f"零杀 {r['zero']} | 到2层 {r['depth2']}  [{r['secs']}s]")
    if not LEADERBOARD.exists():
        LEADERBOARD.write_text(
            "# 排行榜(32 种子确定性评估,seeds 9000-9031)\n\n"
            "| run | 均击杀 | 中位 | 最高 | 零杀 | 到2层 |\n|---|---|---|---|---|---|\n"
        )
    with open(LEADERBOARD, "a") as f:
        f.write(line + "\n")
    print(f"已写入 {LEADERBOARD.name}")


if __name__ == "__main__":
    main()
