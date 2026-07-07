"""深水区章 32 种子金标准(评估制度 v2 的深水变体,2026-07-08 起)。

与主榜(train/evaluate.py)的差异,也是不可比的原因:
  - max_steps 3000(主榜 1500)——护甲/药水经济需要更长的会计周期;
  - 指标以深度为纲:最深层中位数、到 L2/L3/L4 计数、战死、均杀;
  - 检查点为 MaskablePPO(v16 起的掩码栈),predict 必须带 action_masks
    (掩码是策略分布的一部分,不带 = 换了一个策略)。
其余协议不动:种子 9000-9031 只用于终评、argmax、空载机器、引擎钉死
ENGINE_REF。结果写入 train/leaderboard-deep.md(新表,与主榜互不冒充)。

用法(仓库根目录):
  .venv/bin/python train/evaluate_deep.py train/runs/<run>/model_final
"""

from __future__ import annotations

import pathlib
import statistics as s
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

SEEDS = list(range(9000, 9032))
LEADERBOARD = pathlib.Path(__file__).resolve().parent / "leaderboard-deep.md"


def evaluate(model_path: str):
    from sb3_contrib import MaskablePPO

    from diablogym import DiabloGymEnv
    import models  # noqa: F401  (注册自定义提取器,load 时需要可导入)

    model = MaskablePPO.load(model_path, device="cpu")
    env = DiabloGymEnv(ticks_per_step=4, max_steps=3000,
                       start_in_dungeon=True, include_raw=False,
                       descend_ladder=True)
    kills, depths, deaths = [], [], 0
    t0 = time.time()
    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        info = {}
        while not (done or trunc):
            a, _ = model.predict(obs, action_masks=env.action_masks(),
                                 deterministic=True)
            obs, r, done, trunc, info = env.step(int(a))
        ex = info.get("episode_extra", {})
        kills.append(ex.get("kills", 0))
        depths.append(ex.get("depth", 1))
        deaths += bool(ex.get("died", False))

    return {
        "model": model_path,
        "depth_median": s.median(depths),
        "depth_max": max(depths),
        "l2": sum(d >= 2 for d in depths),
        "l3": sum(d >= 3 for d in depths),
        "l4": sum(d >= 4 for d in depths),
        "deaths": deaths,
        "kills_mean": round(s.mean(kills), 1),
        "kills_median": s.median(kills),
        "secs": round(time.time() - t0, 1),
    }


def main():
    model_path = sys.argv[1]
    r = evaluate(model_path)
    line = (f"| {pathlib.Path(model_path).parent.name} | {r['depth_median']} | "
            f"{r['depth_max']} | {r['l2']} | {r['l3']} | {r['l4']} | "
            f"{r['deaths']} | {r['kills_mean']} | {r['kills_median']} |")
    print(f"深度中位 {r['depth_median']} | 最深 {r['depth_max']} | "
          f"L2 {r['l2']} | L3 {r['l3']} | L4 {r['l4']} | 战死 {r['deaths']} | "
          f"均杀 {r['kills_mean']} | 杀中位 {r['kills_median']}  [{r['secs']}s]")
    if not LEADERBOARD.exists():
        LEADERBOARD.write_text(
            "# Deep-water leaderboard — 32 fixed seeds, 3000-step episodes\n\n"
            "Protocol: argmax + action masks, seeds 9000-9031 (never used for\n"
            "training or hyper-parameter selection), 3000 steps/episode, idle\n"
            "machine, engine pinned to `ENGINE_REF` in bootstrap.sh, reward\n"
            "world = depth-progressive descent ladder (8×N per level). NOT\n"
            "comparable to train/leaderboard.md (1500-step episodes).\n"
            "See train/evaluate_deep.py.\n\n"
            "| run | depth med | depth max | ≥L2 | ≥L3 | ≥L4 | deaths | mean kills | median |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
    lines = LEADERBOARD.read_text().splitlines(keepends=True)
    last_row = max(i for i, l in enumerate(lines) if l.startswith("|"))
    lines.insert(last_row + 1, line + "\n")
    LEADERBOARD.write_text("".join(lines))
    print(f"已写入 {LEADERBOARD.name}")


if __name__ == "__main__":
    main()
