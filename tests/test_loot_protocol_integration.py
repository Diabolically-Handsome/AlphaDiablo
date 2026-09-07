"""Loot protocol integration using native doubles; no game or model execution."""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import Mock, patch
import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "tests"), str(ROOT / "python")]
from diablogym.env import DiabloGymEnv
from diablogym.options_env import OptionsEnv, WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC
from test_resource_recovery_commands import environment, state
import eval_contract


class DummyEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_steps = kwargs["max_steps"]
        self._raw = None
        self.observation_space = gym.spaces.Box(-1, 1, (295,), dtype=np.float32)


def options(**changes):
    values = dict(resource_protocol="l2-town-v1", resource_purchase_mode="full",
        resource_service_policy="sustain-loot-v1", worker_time_protocol="completion-l2-v1",
        max_steps=6000, worker_observation_view=WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
    values.update(changes)
    with patch("diablogym.options_env.DiabloGymEnv", DummyEnv):
        return OptionsEnv(**values)


def test_explicit_service_uses_physical_clock_and_preserves_actor_contract():
    from diablogym.resource_sustain_loot import SustainLootService
    owner = options()
    assert isinstance(owner.resource_service, SustainLootService)
    assert owner.max_steps == 6000 and owner.env.max_steps == 12000
    assert owner.farm_scene_cap == 3600
    assert owner.env.kwargs["resource_loot_economy"] is True
    assert owner.env.kwargs["resource_ordinary_armor_scope"] is True
    assert owner.env.kwargs["resource_preserve_equipment_readiness"] is True
    assert owner.action_space.n == 3
    assert owner.resource_service.collect_microstep_cap == 900
    assert owner.resource_service.service_microstep_cap == 3000
    callback = owner.env.resource_observation_callback
    with patch.object(owner.resource_service, "observe") as observe:
        callback(owner.env, {"a": 1}, 17)
    observe.assert_called_once_with({"a": 1}, 17)


@pytest.mark.parametrize("changes", [
    {"worker_time_protocol": "legacy"}, {"resource_purchase_mode": "armor"},
    {"resource_protocol": "off"}, {"max_steps": 12000},
    {"resource_loot_economy": False}, {"resource_ordinary_armor_scope": False},
    {"resource_preserve_equipment_readiness": False},
])
def test_incompatible_new_protocol_is_rejected_before_native_init(changes):
    with pytest.raises(ValueError):
        options(**changes)


def test_old_completion_policy_does_not_enable_loot_or_second_trip():
    from diablogym.resource_sustain_completion import SustainCompletionService
    owner = options(resource_service_policy="sustain-v6")
    assert type(owner.resource_service) is SustainCompletionService
    assert "resource_loot_economy" not in owner.env.kwargs
    assert not hasattr(owner.resource_service, "maybe_start")


def test_new_recipe_distinct_and_old_recipe_unchanged():
    old = eval_contract.resource_service_recipe("l2-town-v1", "full", "sustain-v6")
    before = deepcopy(old)
    new = eval_contract.resource_service_recipe("l2-town-v1", "full", "sustain-loot-v1")
    assert old == before and old["service_microstep_cap"] == 1500
    assert new["max_town_trips"] == 2
    assert new["service_microstep_cap"] == 3000
    assert new["collect_command_window_microsteps"] == 900
    assert new["native_loot_economy"] is True
    assert new["potion_target_source"] == old["potion_target_source"]
    assert "python/diablogym/resource_sustain_loot.py" in eval_contract.PROTOCOL_SOURCE_FILES


@pytest.mark.parametrize("kind, args", [("sell", ("smith", 0, 1, 2, 0, 3, 5)),
                                       ("loot", (7, 10, 10, 1, 2, 3, 4))])
@pytest.mark.parametrize("accepted", [True, False])
def test_economic_commands_have_native_receipt_and_exactly_one_real_tick(kind, args, accepted):
    env = environment([state(gold=105 if accepted and kind == "sell" else 100)])
    env._resource_loot_economy = True
    receipt = {"accepted": accepted, "reason": "fixture", "price": 0,
               "gold_received": 5 if accepted and kind == "sell" else 0}
    target = "act_sell_inventory_item" if kind == "sell" else "act_pickup_loot_at"
    with patch("diablogym.bridge." + target, create=True, return_value=receipt) as native, \
         patch("diablogym.bridge.act_wait") as wait:
        raw, beats, observed = env._execute_resource_command((kind, *args))
    native.assert_called_once_with(*args)
    assert beats == 1 and env._resource_actual_microsteps == 11
    assert observed == receipt
    assert wait.call_count == int(not accepted)


@pytest.mark.parametrize("kind", ["loot", "sell"])
def test_economic_commands_cannot_mutate_after_deadline_or_with_flag_off(kind):
    target = "act_sell_inventory_item" if kind == "sell" else "act_pickup_loot_at"
    for expired in (True, False):
        env = environment([], deadline=10 if expired else 100)
        env._resource_loot_economy = expired
        with patch("diablogym.bridge." + target, create=True) as native:
            if expired:
                _, beats, receipt = env._execute_resource_command((kind,))
                assert beats == 0 and not receipt["accepted"]
            else:
                with pytest.raises(ValueError):
                    env._execute_resource_command((kind,))
            native.assert_not_called()
        assert env._resource_actual_microsteps == 10


def test_native_configuration_and_observation_fail_closed_for_loot_identity():
    env = DiabloGymEnv.__new__(DiabloGymEnv)
    env.resource_protocol = "l2-town-v1"
    env._resource_loot_economy = True
    env._resource_preserve_equipment_readiness = True
    env.resource_ordinary_armor_scope = True
    with patch("diablogym.bridge.end_game"), patch("diablogym.bridge.configure_resource_protocol") as configure:
        env._configure_native_resource_protocol()
    configure.assert_called_once_with(True, ordinary_armor_scope=True,
                                      preserve_equipment_readiness=True, loot_economy=True)
    raw = {"resource_state": {"enabled": True, "ordinary_armor_scope": True,
                             "preserve_equipment_readiness": True, "loot_economy": True}}
    env._validate_native_resource_flags(raw)
    del raw["resource_state"]["loot_economy"]
    with pytest.raises(RuntimeError, match="loot economy identity"):
        env._validate_native_resource_flags(raw)


@pytest.mark.parametrize("terminal", ["dead", "game_over", "victory", "hp"])
@pytest.mark.parametrize("kind", ["loot", "sell"])
def test_loot_sale_never_execute_or_wait_after_terminal(terminal, kind):
    env = environment([])
    env._resource_loot_economy = True
    env._raw[terminal] = 0 if terminal == "hp" else True
    target = "act_sell_inventory_item" if kind == "sell" else "act_pickup_loot_at"
    with patch("diablogym.bridge." + target, create=True) as native, \
         patch("diablogym.bridge.act_wait") as wait:
        _, beats, receipt = env._execute_resource_command((kind,))
    assert beats == 0 and not receipt["accepted"]
    native.assert_not_called()
    wait.assert_not_called()


@pytest.mark.parametrize("kind", ["loot", "sell"])
def test_zero_dispatch_economic_receipt_retains_identity_and_real_wallet(kind):
    env = environment([], deadline=10)
    env._resource_loot_economy = True
    env._raw["gold"] = 37
    identity = (11, 12, 13, 14)
    command = (("sell", "smith", 2, *identity, 30) if kind == "sell"
               else ("loot", 9, 20, 21, *identity))
    with patch("diablogym.bridge.act_sell_inventory_item", create=True) as sell, \
         patch("diablogym.bridge.act_pickup_loot_at", create=True) as loot, \
         patch("diablogym.bridge.act_wait") as wait:
        _, beats, receipt = env._execute_resource_command(command)
    assert beats == 0 and receipt["native_executed"] is False
    assert receipt["source"] == "environment-precheck"
    assert receipt["gold_before"] == receipt["gold_after"] == 37
    assert receipt["received"] == receipt["price"] == 0
    assert tuple(receipt[k] for k in ("seed_hi", "seed_lo", "create_info", "base_id")) == identity
    if kind == "sell":
        assert receipt["index"] == 2 and receipt["quoted_price"] == 30
    sell.assert_not_called()
    loot.assert_not_called()
    wait.assert_not_called()
