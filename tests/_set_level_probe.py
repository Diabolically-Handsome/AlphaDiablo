"""Quest set-levels must keep the conceptual main-line depth, and scene changes must not fake rewards/kills."""

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
        print("SKIP: spawn.mpq does not contain the official quest set-level assets")
    else:
        # The Skeleton King's quest set-level enum is exactly 1, but its conceptual return level is main-line L3.
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
        assert env._ep_kills == kills, "main-level monsters were miscounted as kills on the map change"
        assert reward == 0.0, f"entering a same-depth quest set-level produced a fake reward: {reward}"

        bridge.probe_return_set_level()
        _, reward, done, trunc, _ = env.step(0)
        assert not done and not trunc
        assert env._raw["dungeon_level"] == 3
        assert env._raw["engine_level"] == 3
        assert env._raw["is_set_level"] is False
        assert env._ep_kills == kills, "quest monsters were miscounted as kills when returning to the main level"
        assert reward == 0.0, f"returning to the same-depth main level produced a fake reward: {reward}"
        print("PASS: set-level conceptual depth/scene identity/reward and kill isolation hold")
finally:
    env.close()
