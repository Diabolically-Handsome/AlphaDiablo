"""完整版 MPQ：通关必需剧情交互必须在 15 动作契约内真实可达。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge
from diablogym.options_env import dispatch


def require_live(env, done, trunc, label):
    if done or trunc:
        raw = env._raw
        raise AssertionError(
            f"{label} 提前结束: done={done}, trunc={trunc}, "
            f"dead={raw['dead']}, victory={raw['victory']}, "
            f"depth={raw['dungeon_level']}, stage={raw['betrayer_quest_stage']}")


env = DiabloGymEnv(
    max_steps=30000,
    start_in_dungeon=True,
    include_raw=False,
    descend_ladder=True,
    death_ladder=True,
)
try:
    env.reset(seed=314159)
    if bridge.probe_is_spawn():
        print("SKIP: spawn.mpq 不含 Lazarus/Vile/L16 正式剧情资产")
    else:
        # 直接跳过与本探针无关的 L1-L14 战斗，但从 L15 法杖台开始走真实
        # 物体/物品/任务状态机：operate stand → pickup staff → 单向 Cain
        # 等价交付 → quest.position 入口。无敌仅隔离战斗噪声，不移动玩家、
        # 不改任务状态，也不替宏操作任何目标。
        bridge.probe_warp_main_level(15)
        env.step(0)
        bridge.probe_invincible(True)
        initial = [p["kind"] for p in env._raw["progression_targets"]]
        assert initial == ["lazarus_stand"], initial
        for _ in range(100):
            _, _, done, trunc, _ = env.step(11)
            require_live(env, done, trunc, "L15 法杖/入口链")
            if env._raw["is_set_level"]:
                break
        assert env._raw["is_set_level"]
        assert env._raw["set_level_id"] == 5
        assert env._raw["monotonic_quest_turn_in_used"] is True
        assert env._raw["betrayer_quest_stage"] == 3

        # 换图会按上游规则重置玩家临时标志，探针重新隔离战斗。这里改按
        # action 10，验证平坦/FARM 路径与 action 11/DIVE 共用同一剧情宏。
        bridge.probe_invincible(True)
        for _ in range(100):
            _, _, done, trunc, _ = env.step(10)
            require_live(env, done, trunc, "Vile 两书/中央法阵链")
            if env._raw["betrayer_quest_stage"] >= 6:
                break
        assert env._raw["betrayer_quest_stage"] >= 6
        assert not env._raw["progression_targets"]
        print("PASS: 法杖台→法杖→单向 Cain 交付→L15 入口→Vile 两书/法阵可达")

        # Lazarus 战斗本身已有交战宏覆盖；返回主层后跳到 L16，验证四个
        # switch 必须逐个真实 operate。DIVE dispatch 会先清贴身怪，再推进
        # 机关；满级+无敌只缩短探针并防随机死亡，不改碰撞/门/机关状态。
        bridge.probe_return_set_level()
        env.step(0)
        bridge.probe_warp_main_level(16)
        env.step(0)
        bridge.probe_invincible(True)
        bridge.probe_add_experience(2_000_000_000)
        bridge.step(ticks=1)
        switches = [p for p in env._raw["progression_targets"]
                    if p["kind"] == "diablo_switch"]
        assert len(switches) == 4, switches
        for _ in range(800):
            action = dispatch("dive", env._raw, False)
            _, _, done, trunc, _ = env.step(action)
            require_live(env, done, trunc, "L16 switch 链")
            if not env._raw["progression_targets"]:
                break
        assert not env._raw["progression_targets"], env._raw["progression_targets"]
        print("PASS: L16 四组机关可达并逐个真实操作，Diablo 房间不再结构性封死")
finally:
    env.close()
