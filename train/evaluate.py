"""32 种子标准评估(评估制度 v2,2026-07-05 起为唯一金标准)。

协议(全部条件都是结果的一部分,缺一不可比):
  - 种子集固定 9000-9031,只用于最终评估,永不参与训练/调参;
  - argmax 确定性策略,max_steps=1500,ticks_per_step=4;
  - 引擎源码钉死在 bootstrap.sh 的 ENGINE_REF(换引擎版本必须重建整张排行榜);
  - 空载机器上运行:引擎的回合推进读真实墙钟(nthread_has_500ms_passed),
    高负载下个别 tick 会少推一个逻辑回合导致轨迹漂移——2026-07-05 实测:
    空载下跨进程 4 次评估逐种子位级一致;训练同机并行时中位数曾漂过 0.5。

教训:8 种子的运气波动曾把 run6/run8 分别高估 77%/57%(15.6→8.8,13.2→8.4)。

用法(仓库根目录):
  .venv/bin/python train/evaluate.py train/runs/<run>/model_final
  自动识别 RecurrentPPO(路径含 lstm)与自定义特征提取器;结果写入 leaderboard.md
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
            "# Leaderboard — deterministic evaluation, 32 fixed seeds\n\n"
            "Protocol: argmax policy, seeds 9000-9031 (never used for training or\n"
            "hyper-parameter selection), 1500 steps/episode, idle machine, engine\n"
            "pinned to `ENGINE_REF` in bootstrap.sh. See train/evaluate.py.\n\n"
            "| run | mean kills | median | max | zero-kill | reached L2 |\n"
            "|---|---|---|---|---|---|\n"
        )
    # 插到表格最后一行之后(表格后面还有脚注/长局探针说明,不能盲目追加到文件尾)
    lines = LEADERBOARD.read_text().splitlines(keepends=True)
    last_row = max(i for i, l in enumerate(lines) if l.startswith("|"))
    lines.insert(last_row + 1, line + "\n")
    LEADERBOARD.write_text("".join(lines))
    print(f"已写入 {LEADERBOARD.name}")


if __name__ == "__main__":
    main()
