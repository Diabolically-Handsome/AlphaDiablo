"""任务副本必须保持概念主线深度，且场景切换不得伪造奖励/击杀。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge


env = DiabloGymEnv(
    max_steps=100,
    start_in_dungeon=True,
    include_raw=False,
    descend_ladder=True,
    death_ladder=True,
)
try:
    env.reset(seed=314159)
    if bridge.probe_is_spawn():
        print("SKIP: spawn.mpq 不含正式任务 set-level 资产")
    else:
        # Skeleton King 的任务副本枚举恰为 1，但概念返回层是主线 L3。
        bridge.probe_warp_main_level(3)
        env.step(0)
        assert env._raw["dungeon_level"] == 3
        assert env._raw["is_set_level"] is False

        kills = env._ep_kills
        bridge.probe_enter_set_level(1)
        _, reward, done, trunc, _ = env.step(0)
        assert not done and not trunc
        assert env._raw["dungeon_level"] == 3
        assert env._raw["engine_level"] == 1
        assert env._raw["is_set_level"] is True
        assert env._raw["set_level_id"] == 1
        assert env._ep_kills == kills, "主层怪物在换图时被误计为击杀"
        assert reward == 0.0, f"进入同深度任务副本产生虚假奖励: {reward}"

        bridge.probe_return_set_level()
        _, reward, done, trunc, _ = env.step(0)
        assert not done and not trunc
        assert env._raw["dungeon_level"] == 3
        assert env._raw["engine_level"] == 3
        assert env._raw["is_set_level"] is False
        assert env._ep_kills == kills, "返回主层时任务怪物被误计为击杀"
        assert reward == 0.0, f"返回同深度主层产生虚假奖励: {reward}"
        print("PASS: set-level 概念深度/场景身份/奖励与击杀隔离成立")
finally:
    env.close()
