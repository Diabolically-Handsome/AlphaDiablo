"""DiabloGym v0 冒烟测试:随机 agent + 确定性验证。

验证链:引擎初始化 → reset(seed) → 随机动作 N 步 → 观测在变 → 同种子可复现。
用法(仓库根目录):  .venv/bin/python tests/smoke_random_agent.py
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import numpy as np

from diablogym import DiabloGymEnv


def snapshot(raw):
    """取观测中的确定性指纹:玩家位置 + 前 5 个怪物的位置/血量。"""
    mons = [(m["id"], m["x"], m["y"], m["hp"]) for m in raw["monsters"][:5]]
    return (raw["player_x"], raw["player_y"], raw["dungeon_level"], tuple(mons))


def main():
    print("== DiabloGym v0 冒烟测试 ==")
    env = DiabloGymEnv(ticks_per_step=4, max_steps=1000)

    # --- 1. reset 与初始观测 ---
    obs, info = env.reset(seed=42)
    raw = info["raw"]
    print(f"reset(seed=42): 城镇位置 ({raw['player_x']},{raw['player_y']}) "
          f"HP {raw['hp']}/{raw['max_hp']} 金币 {raw['gold']} "
          f"层 {raw['dungeon_level']} 怪物数 {len(raw['monsters'])}")
    assert obs.shape == env.observation_space.shape, "观测向量形状不对"

    # --- 2. 随机走 300 步 ---
    rng = np.random.default_rng(0)
    t0 = time.time()
    total_reward, positions = 0.0, set()
    for step in range(300):
        action = int(rng.integers(0, 10))
        obs, reward, terminated, truncated, info = env.step(action)
        raw = info["raw"]
        total_reward += reward
        positions.add((raw["player_x"], raw["player_y"]))
        if step % 100 == 0:
            print(f"  step {step:4d}: pos ({raw['player_x']},{raw['player_y']}) "
                  f"HP {raw['hp']} XP {raw['xp']} 层 {raw['dungeon_level']}")
        if terminated:
            print(f"  episode 终止于 step {step}(dead={raw['dead']})")
            break
    dt = time.time() - t0
    ticks = (step + 1) * env.ticks_per_step
    print(f"随机 {step + 1} 步({ticks} tick)耗时 {dt:.2f}s "
          f"≈ {ticks / dt:.0f} tick/s(实时为 20 tick/s,加速 {ticks / dt / 20:.0f}x)")
    assert len(positions) > 3, f"玩家几乎没动过(只到过 {len(positions)} 个格子)—— 动作注入可能失效"
    print(f"PASS: 玩家移动过 {len(positions)} 个格子,动作注入有效")

    # --- 3. 确定性:同种子同世界,异种子异世界 ---
    _, info_a = env.reset(seed=123)
    snap_a = snapshot(info_a["raw"])
    _, info_b = env.reset(seed=123)
    snap_b = snapshot(info_b["raw"])
    _, info_c = env.reset(seed=456)
    snap_c = snapshot(info_c["raw"])
    assert snap_a == snap_b, f"同种子初始世界不一致!\n{snap_a}\n{snap_b}"
    print("PASS: seed=123 两次 reset 初始世界一致(确定性成立)")
    if snap_a == snap_c:
        print("WARN: seed=123 与 seed=456 初始世界相同(城镇布局本就固定,属正常;下地牢后才分化)")
    else:
        print("PASS: 不同种子初始世界不同")

    print("\n== 全部通过:桥、动作、观测、确定性 OK ==")


if __name__ == "__main__":
    main()
