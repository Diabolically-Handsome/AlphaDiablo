"""Remembering an observation does not grant gold, time or descent permission."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym

from diablogym.options_env import OptionsEnv
from diablogym.resource_sustain_gold import SustainGoldMemoryService


def state(items=(), *, x=10, ready=False):
    return dict(dungeon_level=1, is_set_level=False, player_x=x, player_y=10,
                hp=70, max_hp=70, gold=100, monsters=[],
                resource_state=dict(enabled=True, gold_items=list(items),
                    readiness=dict(ready=ready, failures=[] if ready else ["armor"])))


def gold(active_id=4):
    return dict(active_id=active_id, seed_hi=1, seed_lo=2, create_info=3, base_id=0,
                x=11, y=10, value=50)


def environment(raw, steps=1000):
    return SimpleNamespace(_raw=raw, _steps=steps,
        _plan_descend_path=lambda raw, x, y, **kw: [(x, y, False)])


def bridge():
    return SimpleNamespace(configure_town_service=lambda value: None, WM_DIABPREVLVL=1)


def service():
    result = SustainGoldMemoryService()
    result.start(state(), 1000, "farm_cap", farm_scene_steps=3600)
    return result


def test_previous_observation_causes_real_walk_then_current_identity_pickup():
    result = service()
    result.gold_memory.observe(state([gold()]), 900)
    env = environment(state())
    old = deepcopy(env._raw)
    assert result.command(env, bridge()) == ("walk", 11, 10)
    assert env._raw == old and env._raw["gold"] == 100
    env._raw = state([gold(active_id=8)], x=11)
    env._steps = 1001
    assert result.command(env, bridge()) == ("gold", 8, 1, 2, 3, 0)
    assert result.gold_collected == 0  # Request acceptance does not fabricate wealth.
    result.receipt(("gold", 8, 1, 2, 3, 0), {"accepted": False, "reason": "stale_item"})
    assert result.gold_collected == 0 and env._raw["gold"] == 100


def test_missing_on_tile_gold_is_retired_without_a_fake_pickup():
    result = service()
    result.gold_memory.observe(state([gold()]), 900)
    env = environment(state(x=11, ready=True))
    assert result.command(env, bridge()) == ("complete",)
    assert not result.gold_memory.candidates()
    assert result.gold_collected == 0
    assert result.reason == "ready_without_town"


def test_collect_budget_hands_off_to_real_trip_without_extending_total():
    result = service()
    result.gold_memory.observe(state([gold()]), 900)
    result._stairs = lambda env, message: ("walk", 9, 10)
    assert result.command_microstep_deadline == 1300
    env = environment(state(), steps=1300)
    assert result.command(env, bridge()) == ("walk", 9, 10)
    assert result.phase == "outbound" and result.collect_stop_reason == "collect_budget"
    assert result.command_microstep_deadline == 2500
    assert result.steps == 300 and result.service_microstep_cap == 1500


def test_no_unobserved_gold_or_unready_shortcut_is_created():
    result = service()
    result._stairs = lambda env, message: ("walk", 9, 10)
    env = environment(state())
    assert result.command(env, bridge()) == ("walk", 9, 10)
    assert result.active and result.phase == "outbound"
    assert result.gold_collected == 0 and not result.gold_memory.candidates()


def test_options_selects_new_memory_per_reset_and_keeps_legacy_wire():
    base = SimpleNamespace(observation_space=gym.spaces.Box(-1, 1, shape=(295,)),
                           _raw=None, _steps=0)
    with patch("diablogym.options_env.DiabloGymEnv", return_value=base) as constructor:
        options = OptionsEnv(resource_protocol="l2-town-v1", resource_service_policy="sustain-v3", max_steps=6000)
    first = options.resource_service.gold_memory
    base.resource_observation_callback(base, state([gold()]), 1)
    assert len(first.candidates()) == 1
    options._reset_wrapper_state()
    assert options.resource_service.gold_memory is not first
    assert not options.resource_service.gold_memory.candidates()
    assert options.max_steps == 6000 and options.farm_scene_cap == 3600
    assert "resource_service_policy" not in constructor.call_args.kwargs


def test_same_scene_historical_gold_never_bypasses_finite_service_budget():
    result = service()
    result.gold_memory.observe(state([gold()]), 900)
    env = environment(state(), steps=2500)
    assert result.command(env, bridge()) == ("finish", "resource_service_cap")
    assert result.gold_collected == 0
