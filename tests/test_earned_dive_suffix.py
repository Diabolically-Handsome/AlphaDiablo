"""Earned DIVE suffix boundaries: pure controller/Worker fixtures, no game."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "train"), str(ROOT / "tests")]
from diablogym import bridge
from diablogym.env import DiabloGymEnv, _scene_identity
from diablogym.options_env import (OptionsEnv, FARM, DIVE,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC, DUAL_WORKER_TIME_REMAINING_FEATURE)
from diablogym.worker_env import WorkerWindowEnv
from test_dual_worker_observation import _options_fixture
from test_env_v4_semantics import raw, execution_step_fixture


@pytest.mark.parametrize("depth,option,elapsed", [(1, FARM, 101), (1, DIVE, 4701), (2, FARM, 5200)])
def test_physical_prefix_limit_does_not_leak_into_full_actor_or_masks(depth, option, elapsed):
    options, _, _ = _options_fixture()
    options.max_steps = 6000
    options.env.max_steps = 6000
    options.env._steps = elapsed
    options.env._raw["dungeon_level"] = depth
    options.dive_live_sovereignty = True
    options._win.update(opt=option, t0=elapsed - 3)
    before_raw, before_window = deepcopy(options.env._raw), deepcopy(options._win)
    observation = options._worker_policy_observation(WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
    mask = options._worker_masks().copy()
    options.env.max_steps = elapsed + 7
    shortened = options._worker_policy_observation(WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
    assert observation.shape == (13012,)
    np.testing.assert_array_equal(shortened, observation)
    np.testing.assert_array_equal(options._worker_masks(), mask)
    assert observation[DUAL_WORKER_TIME_REMAINING_FEATURE] == np.float32(1 - elapsed / 6000)
    assert options.env._raw == before_raw and options._win == before_window
    assert options.max_steps == 6000


@pytest.mark.parametrize("scene_change", [False, True])
def test_existing_macro_and_settle_charge_exact_physical_cap_before_boundary(scene_change):
    state = raw()
    env, reward = execution_step_fixture(state)
    env.max_steps = 3
    env._resource_actual_microsteps = 0
    # Exercise the real settle, not the helper fixture's abbreviated version.
    del env._settle_to_idle
    events = []
    def step_native():
        env._resource_actual_microsteps += 1
        current = deepcopy(state)
        current.update(player_mode=bridge.PM_WALK_NORTHWARDS, future_x=state["player_x"] + 1)
        if scene_change and env._resource_actual_microsteps >= 2:
            current["dungeon_level"] = 2
        events.append(("tick", env._resource_actual_microsteps))
        return current
    env._step_native = step_native
    env._record_visit = Mock()
    with patch("diablogym.env.bridge.act_wait", side_effect=lambda: events.append(("wait", None))):
        _, value, terminated, truncated, info = env.step(0)
    assert env._steps == env._resource_actual_microsteps == 3
    assert [v for k, v in events if k == "tick"] == [1, 2, 3]
    assert terminated and not truncated
    assert env._raw["player_mode"] == bridge.PM_WALK_NORTHWARDS
    assert env._episode_ended
    assert value == 0.0
    reward.assert_called_once()


from diablogym.options_env import DUAL_WORKER_ACTION_MASK_SLICE, DUAL_WORKER_MANAGER_MASK_SLICE
from diablogym.worker_env import PrefixBudgetExceeded

R16 = "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"
SCOPE = "earned-dive-suffix-v1"


def parent_callback():
    calls = []
    def callback(obs, mask):
        calls.append((obs.copy(), mask.copy()))
        return 9
    callback.calls = calls
    callback.source_sha256 = R16
    callback.diablogym_worker_observation_view = "dual-v4-asymmetric-v3"
    callback.diablogym_worker_action12_mode = "environment-mask"
    callback.episode_reseed = Mock()
    callback.on_beat = None
    return callback


class ScriptedOptions:
    """Finite native-like events; Worker methods and accounting stay real."""
    def __init__(self, episodes, **kwargs):
        self.episodes = deepcopy(episodes)
        self.max_steps = kwargs["max_steps"]
        self.env = SimpleNamespace(max_steps=self.max_steps, _steps=0,
            _resource_actual_microsteps=0, observation_space=gym.spaces.Box(-1, 1, (295,)),
            _raw={}, _decision_idle=DiabloGymEnv._decision_idle,
            _econ_episode_max_depth=1)
        self._workers = {7: "untouched"}
        self._win = None
        self.exhausted = False
        self.reset_seeds, self.executed, self.windows_closed = [], [], []
        self.begin_ids, self.observation_calls, self.mask_calls = [], 0, 0
        self.next_id = 1
        self.history = {"visited": [(17, 21)], "fuse": (9, 4), "map_marker": 67}

    def reset(self, *, seed, options=None):
        self.reset_seeds.append(seed)
        self.events = self.episodes.pop(0)
        self._win = None
        self.env._steps = self.env._resource_actual_microsteps = 0
        self.env._raw = raw()
        self.env._raw.update(dungeon_level=1, is_set_level=False,
            resource_state={"enabled": True, "readiness": {"ready": True, "failures": []}})
        return np.zeros(13012, np.float32), {}

    def resource_option_choice(self):
        return self.events[0]["opt"]

    def _win_begin(self, option):
        assert self._win is None
        self.event = self.events.pop(0)
        assert self.event["opt"] == option
        self._win = dict(window_id=self.next_id, opt=option, t0=self.env._steps,
            R=0.0, W=0.0, worker_wage=0.0, worker_kills=0, beats=0,
            recovery_actions=0, last_recovery_action=None)
        self.next_id += 1
        self.begin_ids.append(self._win["window_id"])

    def advance(self, ticks, *, reward=0.0, wage=0.0, update=None):
        taken = min(ticks, self.env.max_steps - self.env._steps)
        assert taken >= 0
        self.env._steps += taken
        self.env._resource_actual_microsteps += taken
        self._win["beats"] += taken
        self._win["R"] += reward
        self._win["W"] += wage
        self._win["worker_wage"] += wage
        if update: self.env._raw.update(deepcopy(update))
        return taken

    def _consume_fuse_recovery(self):
        return None

    def _drain(self):
        self.advance(self.event.get("opening_ticks", 0),
            reward=self.event.get("opening_reward", 0.0),
            update=self.event.get("opening_update"))
        reason = self.event.get("opening_reason")
        if self.env._steps >= self.env.max_steps: reason = "cap"
        return SimpleNamespace(reason=reason) if reason else None

    def _worker_policy_observation(self, *args, **kwargs):
        self.observation_calls += 1
        obs = np.zeros(13012, np.float32)
        obs[:3] = [self.env._steps, self.env._raw["dungeon_level"], self._win["window_id"]]
        obs[DUAL_WORKER_ACTION_MASK_SLICE] = 1
        obs[DUAL_WORKER_MANAGER_MASK_SLICE] = self.event.get("manager_mask", [1, 1, 0])
        return obs

    def _worker_obs(self):
        return self._worker_policy_observation()

    def _worker_masks(self):
        self.mask_calls += 1
        return np.ones(15, bool)

    def _win_step_worker(self, action):
        self.executed.append(action)
        self.advance(self.event.get("ticks", 1), reward=self.event.get("reward", 2.0),
            wage=self.event.get("wage", 2.0), update=self.event.get("update"))
        reason = self.event.get("reason")
        if self.env._steps >= self.env.max_steps: reason = "cap"
        return SimpleNamespace(reason=reason, requested_action=action, executed_action=action,
            fuse_tripped=False, fuse_requested_action=None, action14_audit=None,
            action_effect_audit={"material_effects": ["test-effect"]})

    def _win_end(self, reason):
        win = self._win
        self._win = None
        done = bool(self.event.get("done", False))
        trunc = reason == "cap"
        extra = dict(win, reason=reason, R=win["R"], tau=win["beats"],
            base_done=done or trunc, base_trunc=trunc, budget_boundary=trunc)
        info = {"dead": bool(self.env._raw.get("dead")), "terminal_death_reward": -9.0 if done else 0.0,
                "episode_extra": {"died": bool(self.env._raw.get("dead"))}}
        self.windows_closed.append(deepcopy(extra))
        return extra, info, done, trunc

    def step(self, option):
        self._win_begin(option)
        ending = self._consume_fuse_recovery()
        if ending is None: ending = self._drain()
        if ending is None:
            callback = self._workers.get(option)
            action = callback(self._worker_policy_observation(), self._worker_masks()) if callable(callback) else 0
            ending = self._win_step_worker(action)
            if ending.reason is None: ending.reason = "cap" if self.env._steps >= self.env.max_steps else "window"
        extra, info, done, trunc = self._win_end(ending.reason)
        return None, extra["R"], done, trunc, {**info, "option_extra": extra}

    def close(self):
        pass


def worker_fixture(episodes, *, attempts=4, budget=100, max_steps=6000, **overrides):
    parent = parent_callback()
    kwargs = dict(manager_npz=None, manager_heuristic="readiness-v1", max_steps=max_steps,
        learning_window_scope=SCOPE, policy_observation_view="dual-v4-asymmetric-v3",
        resource_protocol="l2-town-v1", resource_service_policy="sustain-v6",
        prefix_worker=parent, prefix_worker_sha256=R16,
        prefix_max_attempts=attempts, prefix_max_microsteps=budget, seed_scope="train")
    kwargs.update(overrides)
    with patch("diablogym.worker_env.OptionsEnv", side_effect=lambda **kw: ScriptedOptions(episodes, **kw)):
        worker = WorkerWindowEnv(**kwargs)
    # Existing boundary/death algebra is separately tested in its own suite;
    # this fixture supplies its explicit result for synthetic native episodes.
    worker._worker_boundary = lambda done, trunc, extra, info: (bool(done), bool(trunc), False)
    worker._terminal_death_reward = lambda done, trunc, info: -9.0 if info.get("dead") else 0.0
    return worker, parent


def test_handoff_keeps_same_window_history_and_excludes_all_prefix_pay():
    worker, parent = worker_fixture([[dict(opt=FARM, ticks=3, reward=13, wage=7),
        dict(opt=DIVE, opening_ticks=2, opening_reward=5, ticks=1, wage=2, reward=2)]])
    obs, info = worker.reset(seed=2164000)
    win = worker.oe._win
    history = deepcopy(worker.oe.history)
    assert obs[0] == 5 and win["window_id"] == 2 and win["t0"] == 3
    assert len(parent.calls) == 1 and worker.oe._workers == {7: "untouched"}
    assert worker.oe.env.max_steps == worker.oe.max_steps == 6000
    ledger = worker.get_prefix_ledger()
    assert ledger["lifetime"]["prefix_microsteps"] == 5
    assert ledger["lifetime"]["prefix_raw_reward"] == 18
    assert ledger["attempts"][0]["status"] == "handoff"
    assert ledger["attempts"][0]["learner_reward_credited"] == 0
    _, reward, done, trunc, transition = worker.step(11)
    assert reward == transition["transition_reward"] == transition["worker_wage"] == 2
    assert not done and not trunc and worker.oe._win is win
    assert worker.oe.history == history and len(parent.calls) == 1
    assert worker.stats["dive_live_steps"] == 1
    ledger["attempts"][0]["status"] = "mutated"
    assert worker.get_prefix_ledger()["attempts"][0]["status"] == "handoff"


def test_opening_damage_cannot_handoff_and_recovery_inside_same_window_does_not_recheck():
    unready = {"resource_state": {"enabled": True, "readiness": {"ready": False, "failures": ["health"]}}}
    ready = {"resource_state": {"enabled": True, "readiness": {"ready": True, "failures": []}}}
    worker, parent = worker_fixture([[dict(opt=DIVE, opening_ticks=1, opening_update=unready,
        update=ready, reason="stall"), dict(opt=DIVE)]])
    obs, _ = worker.reset(seed=2164000)
    assert obs[2] == 2 and len(parent.calls) == 1
    assert worker.oe.begin_ids == [1, 2] and len(worker.oe.windows_closed) == 1
    assert [x["eligible"] for x in worker.get_prefix_ledger()["attempts"][0]["dive_openings"]] == [False, True]


@pytest.mark.parametrize("opening_update", [{"dungeon_level": 2}, {"dungeon_level": 2, "dead": True}])
def test_actual_main_depth_before_handoff_is_rejected_after_window_accounting(opening_update):
    worker, parent = worker_fixture([[dict(opt=DIVE, opening_ticks=1,
        opening_reason="scene", opening_update=opening_update)]])
    with pytest.raises(RuntimeError, match="early_main_depth"):
        worker.reset(seed=2164000)
    assert len(parent.calls) == 0 and len(worker.oe.windows_closed) == 1
    assert worker.get_prefix_ledger()["attempts"][0]["status"] == "early_main_depth_before_handoff"
    with pytest.raises(PrefixBudgetExceeded): worker.reset()
    assert len(worker.oe.reset_seeds) == 1


def test_opening_death_cost_is_separate_and_attempt_cap_blocks_new_episode():
    worker, parent = worker_fixture([[dict(opt=DIVE, opening_ticks=1, opening_reward=-9,
        opening_reason="death", opening_update={"dead": True}, done=True)]], attempts=1)
    with pytest.raises(PrefixBudgetExceeded, match="attempt_budget"):
        worker.reset(seed=2164000)
    assert parent.calls == [] and worker.stats["prefix_microsteps"] == 1
    assert worker.stats["reset_ff_terminal_death_reward"] == -9
    assert worker.stats["credited_ff_terminal_death_reward"] == 0
    assert worker.stats.get("dive_live_steps", 0) == 0
    assert len(worker.oe.windows_closed) == 1
    with pytest.raises(PrefixBudgetExceeded): worker.reset(seed=2164001)
    assert len(worker.oe.reset_seeds) == 1


@pytest.mark.parametrize("option", [FARM, DIVE, 2])
def test_lifetime_cap_settles_window_at_exact_tick_then_seals_without_learner_terminal(option):
    event = dict(opt=option, ticks=99, reward=6, wage=4)
    if option == DIVE:
        event.update(opening_update={"resource_state": {"enabled": True, "readiness": {"ready": False, "failures": ["health"]}}})
    worker, parent = worker_fixture([[event]], budget=3)
    with pytest.raises(PrefixBudgetExceeded, match="microstep_budget"):
        worker.reset(seed=2164000)
    ledger = worker.get_prefix_ledger()
    assert ledger["lifetime"]["prefix_microsteps"] == 3
    assert ledger["attempts"][0]["status"] == "microstep_budget_exhausted"
    assert len(worker.oe.windows_closed) == 1 and worker.oe._win is None
    assert worker.oe.env._steps == 3 and worker.oe.env.max_steps == 6000
    assert worker.stats.get("dive_live_steps", 0) == 0
    calls = list(worker.oe.executed)
    with pytest.raises(PrefixBudgetExceeded): worker.reset()
    with pytest.raises(gym.error.ResetNeeded): worker.step(0)
    assert worker.oe.executed == calls and len(worker.oe.reset_seeds) == 1


def test_normal_episode_cap_is_not_lifetime_exhaustion_and_cost_survives_next_reset():
    worker, parent = worker_fixture([[dict(opt=FARM, ticks=7)], [dict(opt=DIVE)]],
        budget=20, attempts=3, max_steps=7)
    obs, _ = worker.reset(seed=2164000)
    ledger = worker.get_prefix_ledger()
    assert [a["status"] for a in ledger["attempts"]] == ["prefix_terminal", "handoff"]
    assert ledger["lifetime"]["prefix_microsteps"] == 7
    assert ledger["lifetime"]["prefix_attempts"] == 2
    assert ledger["failure"] is None and obs[0] == 0
    assert len(worker.oe.reset_seeds) == 2


def test_handoff_does_not_reset_per_environment_lifetime_attempt_budget():
    worker, parent = worker_fixture([[dict(opt=DIVE)]], attempts=1)
    worker.reset(seed=2164000)
    with pytest.raises(PrefixBudgetExceeded): worker.reset(seed=2164001)
    assert len(worker.oe.reset_seeds) == 1 and worker.stats["prefix_attempts"] == 1


@pytest.mark.parametrize("option", [FARM, DIVE])
def test_after_handoff_deeper_farm_and_dive_are_live_learner_windows(option):
    worker, parent = worker_fixture([[dict(opt=DIVE), dict(opt=option)]])
    worker.reset(seed=2164000)
    worker.oe._win = None
    worker.oe.env._raw["dungeon_level"] = 2
    outcome = worker._advance_to_learning_window()
    assert outcome.obs is not None and worker.oe._win["opt"] == option
    assert parent.calls == []
    _, reward, done, trunc, info = worker.step(9)
    assert reward == 2 and not done and not trunc
    assert worker.stats["farm_live_steps" if option == FARM else "dive_live_steps"] == 1


def test_foreign_deadline_change_fails_closed_without_restoring_foreign_value():
    worker, parent = worker_fixture([[dict(opt=DIVE)]])
    worker._new_episode(2164000)
    worker.oe.env.max_steps -= 1
    with pytest.raises(RuntimeError, match="outside its owner"):
        worker._advance_to_learning_window()
    assert worker.oe.env.max_steps == 99 and parent.calls == []
    assert worker.get_prefix_ledger()["attempts"][0]["status"] == "engineering_error"


@pytest.mark.parametrize("overrides", [
    {"prefix_max_attempts": 0}, {"prefix_max_attempts": True},
    {"prefix_max_microsteps": 0}, {"prefix_max_microsteps": 1.5},
    {"prefix_worker_sha256": "0" * 64}, {"prefix_worker": None},
    {"resource_protocol": "off", "resource_service_policy": "legacy-v1"},
    {"resource_service_policy": "sustain-v5"}, {"skip_dry": 0.5},
    {"policy_observation_view": "legacy-v3"}, {"drink_sovereignty": False},
])
def test_bad_prefix_configuration_rejected_before_options_or_native_constructor(overrides):
    parent = parent_callback()
    args = dict(manager_npz=None, manager_heuristic="readiness-v1", learning_window_scope=SCOPE,
        policy_observation_view="dual-v4-asymmetric-v3", resource_protocol="l2-town-v1",
        resource_service_policy="sustain-v6", prefix_worker=parent,
        prefix_worker_sha256=R16, prefix_max_attempts=2, prefix_max_microsteps=100)
    args.update(overrides)
    with patch("diablogym.worker_env.OptionsEnv") as constructor:
        with pytest.raises(ValueError): WorkerWindowEnv(**args)
    constructor.assert_not_called()


@pytest.mark.parametrize("scope", ["farm-only", "farm-dive-v1"])
def test_old_scope_has_no_prefix_lifetime_or_callback_state(scope):
    with patch("diablogym.worker_env.OptionsEnv", side_effect=lambda **kw: ScriptedOptions([], **kw)) as constructor:
        worker = WorkerWindowEnv(None, manager_heuristic="readiness-v1", learning_window_scope=scope)
    assert worker.get_prefix_ledger() == {"enabled": False}
    assert not any(k.startswith("prefix_") for k in worker.stats)
    assert not hasattr(worker, "prefix_worker")
    assert "prefix_worker" not in constructor.call_args.kwargs
    with patch("diablogym.worker_env.OptionsEnv") as constructor:
        with pytest.raises(ValueError):
            WorkerWindowEnv(None, manager_heuristic="readiness-v1", learning_window_scope=scope,
                            prefix_max_attempts=1)
    constructor.assert_not_called()


def test_zero_lifetime_capacity_blocks_before_seed_reset_or_any_command():
    worker, parent = worker_fixture([[dict(opt=DIVE)]], budget=1)
    worker._prefix_lifetime_microsteps = 1
    worker.stats["prefix_microsteps"] = 0  # public telemetry cannot reopen capacity
    before = deepcopy(worker._rng.bit_generator.state)
    with pytest.raises(PrefixBudgetExceeded): worker.reset(seed=2164000)
    assert worker._rng.bit_generator.state == before
    assert worker.oe.reset_seeds == [] and worker.oe.executed == []
    parent.episode_reseed.assert_not_called()


def test_opening_boundary_at_exact_lifetime_cap_cannot_return_a_ready_learner_state():
    worker, parent = worker_fixture([[dict(opt=DIVE, opening_ticks=3, opening_reward=11)]], budget=3)
    with pytest.raises(PrefixBudgetExceeded): worker.reset(seed=2164000)
    ledger = worker.get_prefix_ledger()
    assert ledger["attempts"][0]["status"] == "microstep_budget_exhausted"
    assert ledger["attempts"][0]["raw_reward"] == 11
    assert ledger["lifetime"]["prefix_handoffs"] == 0
    assert parent.calls == [] and len(worker.oe.windows_closed) == 1


def test_real_sb3_collection_contains_only_suffix_transitions_and_suffix_gae_without_optimizer():
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    class Continue(BaseCallback):
        def _on_step(self): return True
    worker, parent = worker_fixture([[dict(opt=FARM, ticks=3, reward=100, wage=70),
        dict(opt=DIVE, opening_ticks=2, opening_reward=-20, ticks=1, reward=2, wage=2)]])
    env = DummyVecEnv([lambda: worker])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = MaskablePPO("MlpPolicy", env, n_steps=2, batch_size=2, gamma=0.9,
        gae_lambda=0.8, n_epochs=1, policy_kwargs={"net_arch": [8]}, seed=2164000,
        device="cpu", verbose=0)
    parameters = {key: value.clone() for key, value in model.policy.state_dict().items()}
    try:
        _, callback = model._setup_learn(2, callback=Continue())
        assert model.collect_rollouts(env, callback, model.rollout_buffer, n_rollout_steps=2)
        buffer = model.rollout_buffer
        assert model.num_timesteps == 2 and buffer.full
        assert len(parent.calls) == 1 and parent.calls[0][0][0] == 0
        np.testing.assert_array_equal(buffer.observations[:, 0, 0], [5, 6])
        np.testing.assert_array_equal(buffer.rewards[:, 0], [2, 2])
        np.testing.assert_array_equal(buffer.episode_starts[:, 0], [1, 0])
        assert worker.stats["dive_live_steps"] == 2
        assert worker.get_prefix_ledger()["lifetime"]["prefix_microsteps"] == 5
        with torch.no_grad():
            last = float(model.policy.predict_values(torch.as_tensor(model._last_obs)).item())
        values = buffer.values[:, 0]
        last_adv = 2 + 0.9 * last - values[1]
        first_adv = 2 + 0.9 * values[1] - values[0] + 0.9 * 0.8 * last_adv
        np.testing.assert_allclose(buffer.advantages[:, 0], [first_adv, last_adv], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(buffer.returns[:, 0], values + [first_adv, last_adv], rtol=1e-6, atol=1e-6)
        assert model.policy.optimizer.state == {}
        assert all(torch.equal(value, model.policy.state_dict()[key]) for key, value in parameters.items())
    finally:
        env.close()
        torch.set_num_threads(old_threads)



def test_public_counter_clear_cannot_erase_authoritative_lifetime_receipt_or_reopen_budget():
    worker, parent = worker_fixture([[dict(opt=FARM, ticks=3), dict(opt=DIVE, opening_ticks=1)]], attempts=1)
    worker.reset(seed=2164000)
    worker.stats["prefix_attempts"] = 0
    worker.stats["prefix_microsteps"] = 0
    with pytest.raises(PrefixBudgetExceeded) as failure: worker.reset(seed=2164001)
    assert len(worker.oe.reset_seeds) == 1
    ledger = worker.get_prefix_ledger()
    assert ledger["lifetime"]["prefix_attempts"] == 1
    assert ledger["lifetime"]["prefix_microsteps"] == 4
    assert failure.value.audit["lifetime"]["prefix_attempts"] == 1
    assert failure.value.audit["lifetime"]["prefix_microsteps"] == 4


def test_set_level_ready_state_is_not_a_main_l1_handoff():
    worker, parent = worker_fixture([[dict(opt=DIVE, opening_update={"is_set_level": True}, reason="scene"),
                                     dict(opt=DIVE, opening_update={"is_set_level": False})]])
    worker.reset(seed=2164000)
    openings = worker.get_prefix_ledger()["attempts"][0]["dive_openings"]
    assert [x["eligible"] for x in openings] == [False, True]
    assert len(parent.calls) == 1 and worker.oe._win["window_id"] == 2


def test_repeated_zero_native_tick_windows_fail_instead_of_spinning_for_an_unreachable_handoff():
    worker, parent = worker_fixture([[dict(opt=FARM, ticks=0) for _ in range(40)]])
    with pytest.raises(RuntimeError, match="zero-native-tick"):
        worker.reset(seed=2164000)
    assert worker.stats["prefix_microsteps"] == 0
    assert len(worker.oe.windows_closed) == 32 and len(parent.calls) == 32
    assert worker.get_prefix_ledger()["attempts"][0]["status"] == "engineering_error"
    with pytest.raises(PrefixBudgetExceeded): worker.reset()
    assert len(worker.oe.reset_seeds) == 1
