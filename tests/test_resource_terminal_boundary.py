"""Service failure boundaries through the real base step and window consumers.

Pure native-state fixtures only; never initialize a game or train a policy.
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
from diablogym.options_env import OptionsEnv, RESUPPLY, KILL_PATIENCE
from diablogym.worker_env import WorkerWindowEnv
from test_env_v4_semantics import raw, execution_step_fixture


def environment(*, steps=4500, busy=False, dead=False, protocol="l2-town-v1"):
    state = raw(hp=0 if dead else 70)
    state.update(dead=dead, gold=123, resource_state={
        "enabled": protocol != "off", "protocol": protocol,
        "readiness": {"ready": False, "failures": ["armor"]},
        "transition": {"sequence": 3, "accepted": True},
    })
    if busy:
        state.update(player_mode=bridge.PM_WALK_NORTHWARDS,
                     future_x=state["player_x"] + 1)
    env, reward = execution_step_fixture(state)
    env.resource_protocol = protocol
    env.resource_purchase_mode = "full"
    env.max_steps = 6000
    env._steps = env._resource_actual_microsteps = steps
    env._resource_service_deadline = steps
    env._resource_pending_command = None
    env._resource_terminal_reason = None
    env._resource_max_main_depth = 1
    env._resource_calibration = None
    env._step_native = Mock(side_effect=AssertionError("unexpected native tick"))
    # A zero-tick administrative command must not enter the settle path at all.
    env._settle_to_idle = Mock(side_effect=AssertionError("unexpected settle"))
    return env, reward


def close_through_options(env, result, *, no_progress=0):
    """Consume real base info without manufacturing the boundary markers."""
    _, value, terminated, truncated, info = result
    options = OptionsEnv.__new__(OptionsEnv)
    options.env = env
    options.resource_protocol = env.resource_protocol
    options.action_space = gym.spaces.Discrete(3)
    options.action_masks = lambda: np.ones(3, dtype=bool)
    options._decisions = 0
    options.exhausted = False
    options._cap_hits = 0
    options.mode_seq = []
    options.farm_scene_steps = 3600
    options.layer_clock = no_progress
    options._win_begin(RESUPPLY)
    options._win.update(done=terminated, trunc=truncated, last_info=info,
                        R=value, W=value)
    reason = options._win_term(terminated, truncated, 0)
    assert reason == ("death" if env._raw["dead"] else "end")
    extra, base_info, done, trunc = options._win_end(reason)
    assert base_info == info and options._win is None
    boundary = WorkerWindowEnv._worker_boundary(done, trunc, extra, base_info)
    return extra, boundary


@pytest.mark.parametrize("steps,busy,dead", [
    (4500, False, False), (4500, True, False), (4500, False, True),
    (6000, False, False), (6000, True, False), (6000, True, True),
])
def test_resource_failure_is_zero_tick_nonbootstrap_terminal(steps, busy, dead):
    env, reward = environment(steps=steps, busy=busy, dead=dead)
    before = deepcopy(env._raw)
    with patch.object(bridge, "step") as step, patch.object(bridge, "act_wait") as wait:
        result = env.step_resource(("finish", "resource_service_cap"))
    _, value, terminated, truncated, info = result
    assert (terminated, truncated, value) == (True, False, 0.0)
    assert info["resource_failure_terminal"] is True
    assert info["time_limit_bootstrap_safe"] is False
    assert info["resource_protocol"]["terminal_reason"] == "resource_service_cap"
    assert info["episode_extra"]["died"] is dead
    assert info["resource_action_audit"]["micro_steps"] == 0
    assert info["action_effect_audit"]["stall_cost_applied"] is False
    assert env._steps == env._resource_actual_microsteps == steps
    assert env._raw == before and env._resource_pending_command is None
    assert env._episode_ended
    expected_unsettled = steps == 6000 and busy and not dead
    if steps == 6000:
        assert info["budget_exhausted"] is True
        assert info["unsettled_budget_terminal"] is expected_unsettled
    else:
        assert not info.get("budget_exhausted", False)
        assert not info.get("unsettled_budget_terminal", False)
    extra, boundary = close_through_options(env, result)
    assert extra["base_done"] is True and extra["base_trunc"] is False
    assert extra["budget_boundary"] is expected_unsettled
    assert boundary == (True, False, False)
    reward.assert_called_once()
    env._step_native.assert_not_called()
    env._settle_to_idle.assert_not_called()
    step.assert_not_called()
    wait.assert_not_called()


@pytest.mark.parametrize("reason", ["resource_unreachable", "resource_healer_unavailable"])
def test_failure_reason_is_preserved_without_forging_death_or_time_limit(reason):
    env, _ = environment()
    result = env.step_resource(("finish", reason))
    assert result[2:4] == (True, False)
    assert result[4]["resource_protocol"]["terminal_reason"] == reason
    assert result[4]["resource_failure_terminal"] is True
    assert result[4]["episode_extra"]["died"] is False
    assert close_through_options(env, result)[1] == (True, False, False)


@pytest.mark.parametrize("busy", [False, True])
def test_complete_is_zero_tick_and_does_not_end_a_live_episode(busy):
    env, _ = environment(busy=busy)
    before = deepcopy(env._raw)
    result = env.step_resource(("complete",))
    assert result[1:4] == (0.0, False, False)
    assert not result[4].get("resource_failure_terminal", False)
    assert env._resource_terminal_reason is None and not env._episode_ended
    assert env._raw == before and env._steps == 4500
    env._step_native.assert_not_called()
    env._settle_to_idle.assert_not_called()


@pytest.mark.parametrize("busy", [False, True])
def test_complete_at_real_episode_limit_keeps_original_idle_busy_classification(busy):
    env, _ = environment(steps=6000, busy=busy)
    result = env.step_resource(("complete",))
    assert result[2:4] == (busy, not busy)
    assert not result[4].get("resource_failure_terminal", False)
    assert result[4]["time_limit_bootstrap_safe"] is (not busy)
    assert result[4]["unsettled_budget_terminal"] is busy
    assert close_through_options(env, result)[1] == (busy, not busy, False)


@pytest.mark.parametrize("busy", [False, True])
def test_off_historical_time_limit_uses_real_step_and_original_markers(busy):
    env, _ = environment(steps=5999, busy=busy, protocol="off")
    env._resource_service_deadline = 6000
    env._step_native = Mock(return_value=deepcopy(env._raw))
    # Isolate this one ordinary action; the supplied final state determines
    # settled vs unsettled just as in the existing base-step fixture.
    env._settle_to_idle = Mock(side_effect=lambda current, beats, **kwargs: (current, beats))
    with patch.object(bridge, "act_wait") as wait:
        result = env.step(0)
    assert result[2:4] == (busy, not busy)
    assert result[4]["time_limit_bootstrap_safe"] is (not busy)
    assert result[4]["unsettled_budget_terminal"] is busy
    assert "resource_failure_terminal" not in result[4]
    assert "resource_protocol" not in result[4]
    assert env._steps == 6000
    env._step_native.assert_called_once()
    wait.assert_called_once()
    assert close_through_options(env, result)[1] == (busy, not busy, False)


def test_failure_preserves_real_unsettled_budget_no_progress_fact():
    env, _ = environment(steps=6000, busy=True)
    result = env.step_resource(("finish", "resource_service_cap"))
    extra, boundary = close_through_options(env, result, no_progress=KILL_PATIENCE)
    assert result[4]["unsettled_budget_terminal"] is True
    assert extra["budget_boundary"] and extra["timeout_without_progress"]
    assert boundary == (True, False, True)


def test_worker_still_rejects_the_original_unmarked_truncation():
    extra = {"base_done": True, "base_trunc": True, "budget_boundary": False,
             "no_progress_micro_steps": 0, "timeout_without_progress": False}
    with pytest.raises(RuntimeError, match="truncated.*TimeLimit"):
        WorkerWindowEnv._worker_boundary(False, True, extra, {})
