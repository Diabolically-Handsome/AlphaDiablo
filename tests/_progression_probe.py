"""Full MPQ: the story interactions required for completion must be truly reachable within the 15-action contract."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import DiabloGymEnv, bridge
from diablogym.options_env import dispatch


def require_live(env, done, trunc, label):
    if done or trunc:
        raw = env._raw
        raise AssertionError(
            f"{label} ended early: done={done}, trunc={trunc}, "
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
        print("SKIP: spawn.mpq does not contain the official Lazarus/Vile/L16 story assets")
    else:
        # Skip the L1-L14 combat unrelated to this probe, but from the L15 staff stand on, walk the real
        # object/item/quest state machine: operate stand → pickup staff → one-way Cain
        # equivalent hand-over → quest.position entrance. Invincibility only isolates combat noise; it does not move the player,
        # change quest state, or operate any target in place of the macro.
        bridge.probe_warp_main_level(15)
        env.step(0)
        bridge.probe_invincible(True)
        initial = [p["kind"] for p in env._raw["progression_targets"]]
        assert initial == ["lazarus_stand"], initial
        for _ in range(100):
            _, _, done, trunc, _ = env.step(11)
            require_live(env, done, trunc, "L15 staff/entrance chain")
            if env._raw["is_set_level"]:
                break
        assert env._raw["is_set_level"]
        assert env._raw["set_level_id"] == 5
        assert env._raw["monotonic_quest_turn_in_used"] is True
        assert env._raw["betrayer_quest_stage"] == 3

        # The map change resets the player's temporary flags per upstream rules, so the probe isolates combat again. This time it uses
        # action 10, verifying that the flat/FARM path shares the same story macro with action 11/DIVE.
        bridge.probe_invincible(True)
        for _ in range(100):
            _, _, done, trunc, _ = env.step(10)
            require_live(env, done, trunc, "Vile two books/central pentagram chain")
            if env._raw["betrayer_quest_stage"] >= 6:
                break
        assert env._raw["betrayer_quest_stage"] >= 6
        assert not env._raw["progression_targets"]
        print("PASS: staff stand → staff → one-way Cain hand-over → L15 entrance → Vile two books/pentagram reachable")

        # The Lazarus fight itself is already covered by the engage macro; after returning to the main level, jump to L16 and verify that the four
        # switches must each be really operated. DIVE dispatch first clears adjacent monsters, then advances the
        # mechanisms; max level + invincibility only shorten the probe and prevent random deaths, without changing collision/door/mechanism state.
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
            require_live(env, done, trunc, "L16 switch chain")
            if not env._raw["progression_targets"]:
                break
        assert not env._raw["progression_targets"], env._raw["progression_targets"]
        print("PASS: the four L16 mechanism groups are reachable and each really operated; the Diablo room is no longer structurally sealed")
finally:
    env.close()
