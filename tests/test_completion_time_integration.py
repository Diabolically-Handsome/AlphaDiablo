"""Completion clock integration through real step/settle and Worker owners.

Only native-state fixtures are used: no bridge initialization, policy or game.
"""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tests")]
from diablogym import bridge
from diablogym.env import DiabloGymEnv
from diablogym.options_env import (OptionsEnv, FARM, DIVE,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC, DUAL_WORKER_TIME_REMAINING_FEATURE)
from diablogym.worker_env import WorkerWindowEnv
from test_env_v4_semantics import raw, execution_step_fixture
from test_dual_worker_observation import _options_fixture
from test_resource_terminal_boundary import close_through_options


def native_state(level=1, *, busy=False, dead=False):
    state = raw(hp=0 if dead else 70)
    state.update(engine_level=level, dungeon_level=level, dead=dead,
                 resource_state={"enabled": True, "protocol": "l2-town-v1",
                     "readiness": {"ready": True, "failures": []},
                     "transition": {"sequence": 7, "accepted": True,
                         "pretransition_ready": True, "source_depth": 1,
                         "target_depth": 2, "source_is_set": False,
                         "target_is_set": False}})
    if busy:
        state.update(player_mode=bridge.PM_WALK_NORTHWARDS,
                     future_x=state["player_x"] + 1)
    return state


def options_fixture(*, protocol="completion-l2-v1"):
    env, reward = execution_step_fixture(native_state())
    del env._settle_to_idle  # Exercise production settle, not the shell shortcut.
    env.resource_protocol = "l2-town-v1"
    env.resource_purchase_mode = "full"
    env.max_steps = 6000
    env._steps = env._resource_actual_microsteps = 0
    env._resource_service_deadline = 3000
    env._resource_pending_command = None
    env._resource_terminal_reason = None
    env._resource_max_main_depth = 1
    env._resource_scene_ledgers = {}
    env._resource_transition_receipts = []
    env._progress_anchors = set(env._visited)
    env._resource_calibration = None
    env._record_visit = Mock()
    options = OptionsEnv.__new__(OptionsEnv)
    options.env = env
    options.max_steps = 6000
    options.worker_time_protocol = protocol
    options.resource_protocol = "l2-town-v1"
    options.resource_purchase_mode = "full"
    options.resource_service_policy = "sustain-v6"
    options.resource_calibration = None
    options.farm_scene_cap = 3600
    options._reset_wrapper_state()
    # Gold-memory callback is independently covered; retain its exact single call.
    return options, env, reward


def position_clock(options, step, *, level=1):
    """Advance only the pure clock to prepare a physical boundary fixture."""
    scene = {"engine_level": level, "is_set_level": False}
    for tick in range(options._completion_clock.state.micro_step + 1, step + 1):
        options._completion_clock.observe_native_tick(tick, scene, scene)
    options._completion_previous_scene = scene
    options.env._steps = options.env._resource_actual_microsteps = step
    options.env._raw = native_state(level)


def run_step(env, state_at_tick, *, action=0):
    ticks = []
    def native_step(*, ticks):
        assert ticks == 4
        index = env._resource_actual_microsteps + 1
        observed = state_at_tick(index)
        calls.append(index)
        return deepcopy(observed)
    calls = ticks
    with patch.object(bridge, "step", side_effect=native_step), \
         patch.object(bridge, "act_wait"), \
         patch.object(bridge, "observe", side_effect=AssertionError("extra observe")):
        result = env.step(action)
    return result, ticks


def test_last_arrival_tick_extends_only_physical_deadline_and_settles_same_action():
    options, env, reward = options_fixture()
    position_clock(options, 11999)
    callbacks = []
    env.resource_tick_callback = lambda inner, state, tick: callbacks.append(
        (tick, inner.max_steps, options._completion_clock.state.micro_step,
         state["player_mode"]))
    result, ticks = run_step(env, lambda t: native_state(2, busy=t < 12002))
    assert ticks == [12000, 12001, 12002]
    assert env._steps == env._resource_actual_microsteps == 12002
    assert result[2:4] == (False, False) and env._decision_idle(env._raw)
    assert options._completion_clock.state.first_arrival_micro_step == 12000
    assert env.max_steps == 13800 and options.max_steps == 6000
    assert [v[:3] for v in callbacks] == [(t, 13800, t) for t in ticks]
    reward.assert_called_once()  # One learner transition; all settle ticks charged.


@pytest.mark.parametrize("idle_last", [False, True])
def test_early_arrival_shortens_same_action_settle_and_stops_exactly(idle_last):
    options, env, reward = options_fixture()
    result, ticks = run_step(env, lambda t: native_state(2,
        busy=not (idle_last and t == 1801)))
    assert len(ticks) == 1801 and ticks[-1] == 1801
    assert env.max_steps == env._steps == env._resource_actual_microsteps == 1801
    assert options._completion_clock.state.first_arrival_micro_step == 1
    assert result[2:4] == (not idle_last, idle_last)
    assert result[4]["time_limit_bootstrap_safe"] is idle_last
    assert result[4]["unsettled_budget_terminal"] is (not idle_last)
    assert close_through_options(env, result)[1] == (not idle_last, idle_last, False)
    reward.assert_called_once()


@pytest.mark.parametrize("arrival_at_limit", [False, True])
def test_native_death_wins_over_arrival_extension_and_time_limit(arrival_at_limit):
    options, env, _ = options_fixture()
    position_clock(options, 11999)
    result, ticks = run_step(env, lambda t: native_state(2 if arrival_at_limit else 1,
                                                       busy=True, dead=True))
    assert ticks == [12000] and result[2:4] == (True, False)
    assert result[4]["episode_extra"]["died"] is True
    assert not result[4].get("time_limit_bootstrap_safe", False)
    assert env.max_steps == (13800 if arrival_at_limit else 12000)
    assert close_through_options(env, result)[1] == (True, False, False)


@pytest.mark.parametrize("busy", [False, True])
def test_resource_failure_at_completion_deadline_keeps_true_terminal(busy):
    options, env, _ = options_fixture()
    position_clock(options, 12000)
    env._raw = native_state(busy=busy)
    with patch.object(bridge, "step", side_effect=AssertionError("zero tick finish")), \
         patch.object(bridge, "act_wait", side_effect=AssertionError("zero tick finish")):
        result = env.step_resource(("finish", "resource_service_cap"))
    assert result[2:4] == (True, False)
    assert result[4]["resource_failure_terminal"] is True
    assert result[4]["time_limit_bootstrap_safe"] is False
    assert result[4]["unsettled_budget_terminal"] is busy
    assert result[4]["episode_extra"]["died"] is False
    assert options._completion_clock.state.micro_step == 12000
    assert close_through_options(env, result)[1] == (True, False, False)


def prefix_fixture(options, *, budget=5):
    worker = WorkerWindowEnv.__new__(WorkerWindowEnv)
    worker.oe = options
    worker.worker_time_protocol = "completion-l2-v1"
    worker.learning_window_scope = "earned-dive-suffix-v1"
    worker.action_space = gym.spaces.Discrete(15)
    worker.stats = {}
    worker.prefix_max_attempts = 4
    worker.prefix_max_microsteps = budget
    worker.prefix_worker = Mock()
    worker.prefix_worker.diablogym_worker_observation_view = "dual-v4-asymmetric-v3"
    worker.prefix_worker.diablogym_worker_action12_mode = "environment-mask"
    worker._initialize_prefix_state()
    worker._prefix_begin_attempt(123)
    worker._prefix_after_reset()
    worker._alive = True
    return worker


def test_prefix_owner_caps_busy_animation_and_releases_without_resetting_clock():
    options, env, _ = options_fixture()
    worker = prefix_fixture(options, budget=5)
    result, ticks = run_step(env, lambda t: native_state(busy=True))
    assert ticks == [1, 2, 3, 4, 5] and result[2:4] == (True, False)
    assert env.max_steps == worker._prefix_deadline == 5
    assert env._completion_prefix_active
    assert options._completion_clock.state.physical_deadline == 12000
    assert worker._prefix_account_clock() == 5
    worker._prefix_restore_limit()
    assert env.max_steps == 12000 and not env._completion_prefix_active
    assert options._completion_clock.state.micro_step == 5 and options.max_steps == 6000
    assert worker._prefix_lifetime_microsteps == 5
    worker._prefix_validate_limits(active=False)


def test_prefix_early_l2_fails_without_extending_and_preserves_incomplete_tick_evidence():
    options, env, reward = options_fixture()
    worker = prefix_fixture(options, budget=5)
    options._win = {"window_id": 3}
    with pytest.raises(RuntimeError, match="before learner handoff") as failure:
        run_step(env, lambda t: native_state(2, busy=True))
    assert env.max_steps == 5 and env._resource_actual_microsteps == 1
    assert env._steps == 0 and options._completion_clock.state.micro_step == 0
    evidence = deepcopy(env._completion_time_failure)
    assert evidence == {"reason": "L2_before_learner_handoff",
        "actual_native_microstep": 1,
        "source_scene": {"engine_level": 1, "is_set_level": False},
        "observed_scene": {"engine_level": 2, "is_set_level": False},
        "window_incomplete": True, "physical_deadline_unchanged": 5}
    worker._prefix_abort_error(failure.value)
    assert worker._prefix_attempt["completion_time_failure"] == evidence
    assert worker._prefix_attempt["status"] == "engineering_error"
    assert worker._prefix_attempt["window_incomplete"] and not worker._alive
    assert worker._prefix_lifetime_microsteps == 0  # Not a fabricated closed window.
    reward.assert_not_called()


def test_handoff_owner_accepts_arrival_deadline_then_next_reset_restores_arrival_budget():
    options, env, _ = options_fixture()
    worker = prefix_fixture(options)
    worker._prefix_restore_limit()
    result, _ = run_step(env, lambda t: native_state(2))
    assert result[2:4] == (False, False) and env.max_steps == 1801
    worker._prefix_validate_limits(active=False)
    env.max_steps += 1
    with pytest.raises(RuntimeError, match="outside its owner"):
        worker._prefix_validate_limits(active=False)
    env.max_steps -= 1
    options._mgr_obs = lambda obs: obs
    reset_seen = []
    def base_reset(*, seed, options):
        reset_seen.append((env.max_steps,
            env.resource_time_info_callback()["clock"].copy()))
        env._raw = native_state()
        env._steps = env._resource_actual_microsteps = 0
        env._episode_ended = False
        return np.zeros(295, dtype=np.float32), {"worker_time": env.resource_time_info_callback()}
    env.reset = base_reset
    _, info = options.reset(seed=456)
    assert reset_seen == [(12000, {"micro_step": 0,
        "first_arrival_micro_step": None, "physical_deadline": 12000})]
    assert info["worker_time"]["clock"]["first_arrival_micro_step"] is None
    assert env.max_steps == 12000 and options.max_steps == 6000
    assert options._completion_previous_scene == {"engine_level": 1, "is_set_level": False}
    worker._prefix_validate_limits(active=False)


@pytest.mark.parametrize("elapsed,mode", [(101, FARM), (4701, DIVE), (6100, DIVE)])
def test_full_actor_and_masks_are_identical_for_physical_caps_and_fixed_6000_clock(elapsed, mode):
    options, _, _ = _options_fixture()
    options.max_steps = 6000
    options.env._steps = elapsed
    options.dive_live_sovereignty = True
    options._win.update(opt=mode, t0=elapsed - 3)
    raw_before, window_before = deepcopy(options.env._raw), deepcopy(options._win)
    observations, masks = [], []
    for protocol, limit in [("legacy", 6000), ("completion-l2-v1", 12000),
                            ("completion-l2-v1", elapsed + 5)]:
        options.worker_time_protocol = protocol
        options.env.max_steps = limit
        observations.append(options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC))
        masks.append(options._worker_masks().copy())
    for obs, mask in zip(observations[1:], masks[1:]):
        np.testing.assert_array_equal(obs, observations[0])
        np.testing.assert_array_equal(mask, masks[0])
    assert observations[0].shape == (13012,)
    assert observations[0][DUAL_WORKER_TIME_REMAINING_FEATURE] == np.float32(max(0, 1-elapsed/6000))
    assert options.env._raw == raw_before and options._win == window_before


@pytest.mark.parametrize("busy", [False, True])
def test_default_legacy_keeps_original_deadline_and_native_step_result(busy):
    results = []
    for explicit in (False, True):
        options, env, _ = options_fixture(protocol="legacy")
        if not explicit:
            del options.worker_time_protocol
            options._reset_wrapper_state()
        assert not hasattr(env, "resource_time_callback")
        env._steps = env._resource_actual_microsteps = 5999
        result, ticks = run_step(env, lambda t: native_state(busy=busy))
        assert ticks == [6000] and env.max_steps == options.max_steps == 6000
        assert result[2:4] == (busy, not busy)
        assert options.completion_time_telemetry() is None
        results.append(result)
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1:] == results[1][1:]


def test_external_tick_callback_replacement_does_not_replace_internal_time_or_memory():
    options, env, _ = options_fixture()
    seen = []
    original_memory = env.resource_observation_callback
    def memory(inner, state, tick):
        seen.append(("memory", tick, options._completion_clock.state.micro_step))
        original_memory(inner, state, tick)
    env.resource_observation_callback = memory
    env.resource_tick_callback = lambda inner, state, tick: seen.append(
        ("first", tick, options._completion_clock.state.micro_step))
    run_step(env, lambda t: native_state())
    env.resource_tick_callback = lambda inner, state, tick: seen.append(
        ("second", tick, options._completion_clock.state.micro_step))
    run_step(env, lambda t: native_state())
    assert seen == [("memory", 1, 1), ("first", 1, 1),
                    ("memory", 2, 2), ("second", 2, 2)]
    assert options._completion_clock.state.micro_step == env._steps == 2


def test_resource_local_command_budget_is_not_extended_by_completion_settle():
    options, env, _ = options_fixture()
    env._resource_pending_command = ("drink",)
    env._resource_actual_microsteps = 1
    busy = native_state(busy=True)
    with patch.object(bridge, "step", side_effect=AssertionError("local budget already spent")), \
         patch.object(bridge, "act_wait", side_effect=AssertionError("no extra fence")):
        final, micro = env._settle_to_idle(busy, 1, max_beats=1,
                                         start_scene=(1, False, 0, 1))
    assert final is busy and micro == 1
