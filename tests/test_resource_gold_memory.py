"""Passive gold memory contracts: no game run or additional native beat."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from diablogym.env import DiabloGymEnv
from diablogym.resource_gold_memory import ObservedGoldMemory


def pile(**changes):
    return dict(active_id=8, seed_hi=1, seed_lo=2, create_info=257,
                base_id=0, x=12, y=13, value=11, **changes)


def raw(items=(), *, depth=1, is_set=False, position=(10, 10), enabled=True):
    return dict(dungeon_level=depth, is_set_level=is_set,
                player_x=position[0], player_y=position[1], gold=100,
                resource_state=dict(enabled=enabled, gold_items=list(items)))


def test_only_actual_l1_observations_are_remembered_without_mutation():
    memory = ObservedGoldMemory()
    for state in (raw([pile()], depth=0), raw([pile()], depth=2),
                  raw([pile()], is_set=True), raw([pile()], enabled=False)):
        memory.observe(state, 1)
    assert memory.candidates() == []
    state = raw([pile()]); before = deepcopy(state)
    memory.observe(state, 2)
    assert state == before
    candidate = memory.candidates()[0]
    assert (candidate["first_seen"], candidate["last_seen"]) == (2, 2)
    state["resource_state"]["gold_items"][0]["value"] = 999
    candidate["x"] = 999
    assert memory.candidates()[0]["value"] == 11
    assert memory.candidates()[0]["x"] == 12


def test_same_active_slot_with_different_seed_never_aliases():
    memory = ObservedGoldMemory(); first = pile(); other = {**pile(), "seed_lo": 99}
    memory.observe(raw([first]), 1)
    memory.observe(raw([other]), 2)
    assert len(memory.candidates()) == 2
    memory.mark_attempted(first, "stale", 2)
    assert memory.candidates()[0]["seed_lo"] == 99
    assert len(memory.candidates(include_attempted=True)) == 2


def test_identity_keeps_latest_handle_position_and_value():
    memory = ObservedGoldMemory(); memory.observe(raw([pile()]), 3)
    moved = {**pile(), "active_id": 45, "x": 20, "value": 7}
    memory.observe(raw([moved]), 8)
    candidate = memory.candidates()[0]
    assert (candidate["first_seen"], candidate["last_seen"], candidate["active_id"],
            candidate["x"], candidate["value"]) == (3, 8, 45, 20, 7)
    assert memory.telemetry()["remembered_identities"] == 1


def test_town_and_return_preserve_memory_but_new_episode_reset_clears_it():
    memory = ObservedGoldMemory(); memory.observe(raw([pile()]), 5)
    memory.observe(raw([], depth=0), 6)
    memory.observe(raw([]), 7)
    assert len(memory.candidates()) == 1
    assert not memory.mark_gone(pile(), raw([]), 7)  # off-screen is unknown
    memory.reset()
    memory.observe(raw([]), 0)
    assert memory.candidates() == []
    assert memory.telemetry()["attempt_events"] == 0
    independent = ObservedGoldMemory()
    memory.observe(raw([pile()]), 1)
    assert independent.candidates() == []


def test_stale_attempt_is_not_gone_or_earned_income():
    memory = ObservedGoldMemory(); memory.observe(raw([pile()]), 1)
    assert memory.mark_attempted(pile(), "stale", 2)
    info = memory.telemetry()
    assert info["candidate_identities"] == 0 and info["gone_identities"] == 0
    assert info["attempt_reasons"] == {"stale": 1}
    assert not any(key in info for key in ("income", "gold_collected", "earned_gold"))
    assert not memory.mark_gone(pile(), raw([], depth=0, position=(12, 13)), 3)
    assert not memory.mark_gone(pile(), raw([pile()], position=(12, 13)), 3)
    assert memory.mark_gone(pile(), raw([], position=(12, 13)), 3)
    assert memory.telemetry()["gone_identities"] == 1
    assert memory.telemetry()["native_gone_confirmations"] == 1
    assert memory.candidates(include_attempted=True) == []


def test_older_observation_and_missing_native_gold_field_cannot_confirm_absence():
    memory = ObservedGoldMemory(); memory.observe(raw([pile()]), 5)
    memory.observe(raw([{**pile(), "value": 1}]), 4)
    assert memory.candidates()[0]["value"] == 11
    assert memory.telemetry()["out_of_order_observations"] == 1
    assert not memory.mark_gone(pile(), raw([], position=(12, 13)), 4)
    no_native_list = raw([], position=(12, 13)); del no_native_list["resource_state"]["gold_items"]
    assert not memory.mark_gone(pile(), no_native_list, 6)


def test_optional_observer_sees_same_real_beat_before_audit_and_cannot_add_time():
    memory = ObservedGoldMemory(); state = raw([pile()]); before = deepcopy(state); calls = []
    env = SimpleNamespace(resource_protocol="l2-town-v1", ticks_per_step=4,
                          _resource_actual_microsteps=7)
    def observe(current_env, observed, step):
        assert current_env is env and observed is state
        calls.append(("memory", step)); memory.observe(observed, step)
    def audit(current_env, observed, step):
        assert observed is state and memory.candidates()[0]["last_seen"] == step
        calls.append(("audit", step))
    env.resource_observation_callback = observe
    env.resource_tick_callback = audit
    with patch("diablogym.env.bridge.step", return_value=state) as native:
        assert DiabloGymEnv._step_native(env) is state
        native.assert_called_once_with(ticks=4)
    assert env._resource_actual_microsteps == 8
    assert calls == [("memory", 8), ("audit", 8)]
    assert state == before


def test_default_missing_callback_preserves_audit_and_protocol_off_calls_neither():
    state = raw([]); audit = Mock()
    env = SimpleNamespace(resource_protocol="l2-town-v1", ticks_per_step=4,
                          _resource_actual_microsteps=7, resource_tick_callback=audit)
    with patch("diablogym.env.bridge.step", return_value=state) as native:
        assert DiabloGymEnv._step_native(env) is state
        native.assert_called_once_with(ticks=4)
    audit.assert_called_once_with(env, state, 8)
    env.resource_protocol = "off"; observer = Mock(); env.resource_observation_callback = observer
    audit.reset_mock()
    with patch("diablogym.env.bridge.step", return_value=state) as native:
        assert DiabloGymEnv._step_native(env) is state
        native.assert_called_once_with(ticks=4)
    assert env._resource_actual_microsteps == 8
    observer.assert_not_called(); audit.assert_not_called()
