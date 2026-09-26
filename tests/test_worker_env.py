"""v23 G0' unit tests: mechanical fidelity of WorkerWindowEnv (plain script assertions, same convention as G0).

Precondition: train/models/v22-h-manager/policy.npz (produced by export_manager_npz.py).
(a) a scripted worker driving WorkerWindowEnv == OptionsEnv + frozen H run directly (seeds 7000-7007, bit for bit);
(b) worker mask (11 always masked, 12 depends on sovereignty and the belt, 14 passed through);
(c) wage identity sum(w) == R - bonus, and option_extra.worker_wage/worker_kills
        accumulate only the real policy transitions of _win_step_worker;
(d) numpy manager == SB3 predict (1000 obs);
(e) a natural window close continues into the next FARM; only a real underlying end or a hidden animation-budget interruption is
        terminated, and only an idle base-episode time limit is truncated.
"""
from collections import Counter
import hashlib
import pathlib
import json
import sys
import tempfile
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import NumpyManager, OptionsEnv, WorkerWindowEnv, bridge
from diablogym.env import terminal_death_reward_component
from diablogym.options_env import DIVE, FARM, dispatch
import diablogym.worker_env as worker_env_module
from diablogym.worker_env import (
    _AdvanceOutcome,
    _coerce_additional_terminal_death_cost,
    _coerce_fast_forward_reward_credit,
    _derive_p_skip_rng,
    BC_RESERVED_SEED_RANGES,
    EVAL_RESERVED_SEED_RANGES,
    HISTORICAL_BURNED_BC_SEED_RANGES,
    is_reserved_eval_seed,
    is_reserved_train_seed,
    sample_train_seed,
)

NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
assert NPZ.exists(), f"missing {NPZ}: run train/export_manager_npz.py first"

SEEDS = list(range(7000, 7008))


def win_sig(extra):
    return (extra["opt"], extra["tau"], extra["reason"], round(extra["R"], 3))


def native_trajectory_state_digest(options_env):
    """Hash every mutable fact that can steer the next scripted beat."""
    base = options_env.env
    combat_floor = [
        [list(key), list(value)]
        for key, value in sorted(base._combat_hp_floor.items())
    ]
    payload = {
        "raw": base._raw,
        "base": {
            "steps": int(base._steps),
            "kills": int(base._ep_kills),
            "visited": [list(point) for point in sorted(base._visited)],
            "exploration_progress": int(base.exploration_progress),
            "softwalls_opened": int(base.softwalls_opened),
            "explore_target": base._explore_target,
            "explore_blocked_targets": [
                list(point)
                for point in sorted(base._explore_blocked_targets)
            ],
            "engage_blocked_keys": [
                list(key) for key in sorted(base._engage_blocked_keys)
            ],
            "combat_hp_floor": combat_floor,
        },
        "options": {
            "layer_clock": int(options_env.layer_clock),
            "legacy_layer_clock": int(options_env._legacy_layer_clock),
            "exhausted": bool(options_env.exhausted),
            "legacy_exhausted": bool(options_env._legacy_exhausted),
            "farm_scene_steps": int(options_env.farm_scene_steps),
            "farm_scene": options_env._farm_scene,
            "fuse_signature": options_env._fuse_sig,
            "fuse_count": int(options_env._fuse),
            "window": options_env._win,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def decision_boundary_busy(raw):
    """An unencoded 295/298-dim execution state must never leak into the next worker decision."""
    return (
        int(raw.get("player_mode", -1)) != bridge.PM_STAND
        or int(raw.get("dest_action", bridge.ACTION_NONE))
        != bridge.ACTION_NONE
        or int(raw.get("walkpath0", bridge.WALK_NONE))
        != bridge.WALK_NONE
        or int(raw.get("future_x", raw["player_x"])) != int(raw["player_x"])
        or int(raw.get("future_y", raw["player_y"])) != int(raw["player_y"])
    )


# --- (d) numpy manager == SB3 predict ---
from sb3_contrib import MaskablePPO

sb3_mgr = MaskablePPO.load(str(ROOT / "train" / "models" / "v22-h-manager" / "model_final"),
                           device="cpu")
np_mgr = NumpyManager(str(NPZ))
rng = np.random.default_rng(1)
mask3 = np.ones(3, dtype=bool)
for i in range(1000):
    o = rng.standard_normal(303).astype(np.float32)
    a_np = np_mgr.choose(o, mask3)
    a_sb, _ = sb3_mgr.predict(o, action_masks=mask3, deterministic=True)
    assert a_np == int(a_sb), f"obs {i}: numpy {a_np} != sb3 {int(a_sb)}"
print("G0'.d PASS: numpy manager and SB3 predict agree bit for bit on 1000 obs")

with tempfile.TemporaryDirectory() as td:
    bad = {name: getattr(np_mgr, name).copy()
           for name in ("w0", "b0", "w1", "b1", "wa", "ba")}
    bad["w0"][0, 0] = np.nan
    bad_path = pathlib.Path(td) / "bad.npz"
    np.savez(bad_path, **bad)
    try:
        NumpyManager(str(bad_path))
    except ValueError as exc:
        assert "NaN/Inf" in str(exc)
    else:
        raise AssertionError("NumpyManager accepted non-finite weights")
print("G0'.d2 PASS: corrupt/NaN weights fail loud at the load boundary")

# --- (d3) pure unit: fast-forward DIVE reward/death must be the consequence of the previous worker transition ---
class _ScriptedOptions:
    def __init__(self, events):
        self.events = list(events)
        self.exhausted = False
        self._win = None
        self._next_window_id = 2
        self._terminal_obs = np.full(298, -1.0, dtype=np.float32)
        self.env = types.SimpleNamespace(
            death_ladder=True,
            _raw={"dead": False, "dungeon_level": 2},
        )

    def step(self, option):
        event = self.events.pop(0)
        assert option == event["opt"]
        extra = {
            "window_id": self._next_window_id,
            "opt": option,
            "reason": event.get("reason", "done"),
            "R": float(event["reward"]),
            "base_done": bool(event.get("done") or event.get("trunc")),
            "base_trunc": bool(event.get("trunc") and not event.get("done")),
            "budget_boundary": bool(event.get("budget_boundary", False)),
            "no_progress_micro_steps": int(
                event.get("no_progress_micro_steps", 0)),
            "timeout_without_progress": bool(
                event.get("timeout_without_progress", False)),
        }
        self._next_window_id += 1
        base_info = dict(event.get("base_info", {"episode_seed": 123}))
        if event.get("done") or event.get("trunc"):
            base_info.setdefault("episode_extra", {
                "xp": 99, "kills": 3, "char_level": 2,
                "depth": 2, "died": bool(event.get("done")), "gold": 7,
            })
            self.env._raw = {
                "dead": bool(base_info["episode_extra"]["died"]),
                "dungeon_level": int(base_info["episode_extra"]["depth"]),
            }
        base_info["option_extra"] = extra
        return (np.zeros(303, dtype=np.float32), float(event["reward"]),
                bool(event.get("done")), bool(event.get("trunc")),
                base_info)

    def _win_begin(self, option):
        assert option == FARM
        self._win = {
            "window_id": self._next_window_id, "R": 0.0,
            "recovery_actions": 0, "last_recovery_action": None,
        }
        self._next_window_id += 1

    @staticmethod
    def _drain():
        return None

    @staticmethod
    def _consume_fuse_recovery():
        return None

    def _worker_obs(self):
        if self._win is None:
            return self._terminal_obs.copy()
        return np.full(298, float(self._win["window_id"]), dtype=np.float32)


def _scripted_advance(manager_options, events):
    fake = object.__new__(WorkerWindowEnv)
    fake._alive = True
    fake.oe = _ScriptedOptions(events)
    options = iter(manager_options)
    fake._mgr_choose = lambda: next(options)
    fake._skip_dry_draw = lambda: False
    fake._log = lambda extra, fast_forward: None
    fake.stats = {"windows": 0, "dry": 0, "fresh": 0, "ff_dry": 0,
                  "ff_terminals": 0}
    return fake, fake._advance_to_learning_window()


fake, advanced = _scripted_advance(
    [DIVE, FARM], [{"opt": DIVE, "reward": 17.5}])
assert not advanced.terminated and not advanced.truncated
assert advanced.reward == 17.5 and advanced.extras[0]["opt"] == DIVE
assert advanced.obs is not None and fake.oe._win is not None

fake, advanced = _scripted_advance(
    [DIVE], [{"opt": DIVE, "reward": -24.0, "done": True, "reason": "death"}])
assert advanced.terminated and not advanced.truncated
assert advanced.reward == -24.0
assert advanced.terminal_death_reward == terminal_death_reward_component(
    dead=True, dungeon_level=2, death_ladder=True)
assert advanced.extras[-1]["reason"] == "death"
assert advanced.obs is not None and not fake._alive
assert advanced.terminal_base_info["episode_extra"]["kills"] == 3

# The Worker's rebuild path must really delegate to env.py's shared pure function instead of keeping a second copy of
# the numeric formula. The remaining episode/raw account checks still fail closed at the Worker boundary.
shared_calls = []
original_death_component = worker_env_module.terminal_death_reward_component
try:
    worker_env_module.terminal_death_reward_component = (
        lambda **kwargs:
        shared_calls.append(kwargs) or -123.0
    )
    rebuilt = fake._terminal_death_reward(
        advanced.terminated,
        advanced.truncated,
        advanced.terminal_base_info,
    )
finally:
    worker_env_module.terminal_death_reward_component = (
        original_death_component)
assert rebuilt == -123.0
# R12 economy act: the delegated call carries the economy spec explicitly (the stand-in uses the v1 default, numbers unchanged).
from diablogym.env import REWARD_ECONOMY_V1 as _ECON_V1
assert shared_calls == [{
    "dead": True,
    "dungeon_level": 2,
    "death_ladder": True,
    "economy": _ECON_V1,
}]
print("G0'.d3 PASS: the DIVE reward and the death that follows are returned in full in the same fast-forward result")

# --- (d4) both Worker terminal paths must hand the underlying episode_extra to Monitor ---
class _StepOptions:
    def __init__(self, *, direct_terminal: bool, base_info: dict):
        self._win = {"window_id": 1, "W": 0.0}
        self._direct_terminal = direct_terminal
        self._base_info = base_info
        self.env = types.SimpleNamespace(
            death_ladder=True,
            _raw={
                "dead": direct_terminal,
                "dungeon_level": int(
                    base_info["episode_extra"]["depth"]),
            },
        )

    def _win_step_worker(self, action):
        # The underlying FARM tick of a direct death already contains the existing -16 death penalty;
        # the last tick of the indirect path still has only the +2.5 worker wage.
        self._win["W"] += -13.5 if self._direct_terminal else 2.5
        return types.SimpleNamespace(
            reason="death" if self._direct_terminal else "levelup",
            requested_action=int(action), executed_action=int(action),
            fuse_tripped=False, fuse_requested_action=None,
        )

    def _worker_obs(self):
        return np.zeros(298, dtype=np.float32)

    def _win_end(self, reason):
        extra = {
            "window_id": 1, "reason": reason, "R": self._win["W"],
            "base_done": self._direct_terminal, "base_trunc": False,
            "budget_boundary": False,
            "no_progress_micro_steps": 0,
            "timeout_without_progress": False,
        }
        self._win = None
        return extra, (self._base_info if self._direct_terminal else {}), \
            self._direct_terminal, False


class _OpenWindowOptions:
    """A FARM window whose opening recovery was already booked before policy."""

    def __init__(self):
        self._win = {"window_id": 1, "W": 5.0}

    def _win_step_worker(self, action):
        self._win["W"] += 2.5
        return types.SimpleNamespace(
            reason=None,
            requested_action=int(action),
            executed_action=int(action),
            fuse_tripped=False,
            fuse_requested_action=None,
        )

    def _worker_obs(self):
        return np.zeros(298, dtype=np.float32)


def _step_shell(options, *, fast_forward_reward_credit="none",
                additional_terminal_death_cost=0.0):
    shell = object.__new__(WorkerWindowEnv)
    shell.oe = options
    shell.action_space = __import__("gymnasium").spaces.Discrete(15)
    shell.skip_dry = 0.0
    shell._pending_skip_dry_probability = None
    shell._pending_skip_dry_remaining_env_steps = 0
    shell._episode_seed = 321
    shell._alive = True
    shell._log = lambda extra, fast_forward: None
    shell.fast_forward_reward_credit = fast_forward_reward_credit
    shell.additional_terminal_death_cost = additional_terminal_death_cost
    shell.stats = {
        "reasons": {},
        "transition_ff_reward": 0.0,
        "direct_terminal_deaths": 0,
        "transition_ff_terminal_deaths": 0,
        "reset_ff_terminal_deaths": 0,
        "manual_ff_terminal_deaths": 0,
        "direct_existing_terminal_death_reward": 0.0,
        "direct_additional_terminal_death_reward": 0.0,
        "transition_ff_terminal_death_reward": 0.0,
        "transition_ff_additional_terminal_death_reward": 0.0,
        "credited_ff_terminal_death_reward": 0.0,
        "reset_ff_terminal_death_reward": 0.0,
        "reset_ff_additional_terminal_death_reward": 0.0,
        "additional_terminal_death_reward": 0.0,
    }
    return shell


terminal_base = {
    "episode_seed": 321,
    "episode_extra": {
        "xp": 12, "kills": 4, "char_level": 3,
        "depth": 2, "died": True, "gold": 55,
    },
}
opening_shell = _step_shell(_OpenWindowOptions())
_, opening_wage, term, trunc, opening_info = opening_shell.step(9)
assert not term and not trunc
assert opening_shell.oe._win["W"] == 7.5
assert opening_wage == 2.5
assert opening_info["worker_wage"] == 2.5
assert opening_info["transition_reward"] == 2.5


class _GearReceiptOptions(_OpenWindowOptions):
    def _win_step_worker(self, action):
        assert int(action) == 14
        self._win["W"] += 0.5
        return types.SimpleNamespace(
            reason=None,
            requested_action=14,
            executed_action=14,
            fuse_tripped=False,
            fuse_requested_action=None,
            action14_audit={
                "accepted": True,
                "commit_attempts": 1,
                "utility_before": 100,
                "utility_after": 137,
                "utility_delta": 37,
            },
        )


gear_shell = _step_shell(_GearReceiptOptions())
_, gear_wage, term, trunc, gear_info = gear_shell.step(14)
assert not term and not trunc and gear_wage == 0.5
assert gear_info["executed_action"] == 14
assert gear_info["action14_audit"] == {
    "accepted": True,
    "commit_attempts": 1,
    "utility_before": 100,
    "utility_after": 137,
    "utility_delta": 37,
}

direct_shell = _step_shell(_StepOptions(
    direct_terminal=True, base_info=terminal_base))
_, direct_reward, term, trunc, direct_info = direct_shell.step(9)
assert term and not trunc
assert direct_reward == -13.5
assert direct_info["credited_fast_forward_reward"] == 0.0
assert direct_info["existing_terminal_death_reward"] == -16.0
assert direct_info["additional_terminal_death_reward"] == 0.0
assert direct_info["total_terminal_death_reward"] == -16.0
assert direct_info["episode_extra"] == terminal_base["episode_extra"]
assert direct_info["terminal_base_info"] == terminal_base
assert direct_shell.stats["direct_terminal_deaths"] == 1
assert direct_shell.stats["direct_existing_terminal_death_reward"] == -16.0

ff_shell = _step_shell(_StepOptions(
    direct_terminal=False, base_info=terminal_base))
ff_extra = {
    # Even if the whole manager/script segment has a positive net return, only the negative death score may be extracted;
    # the +37 positive gain must never go to the worker.
    "window_id": 2, "opt": DIVE, "reason": "death", "R": 37.0,
    "base_done": True, "base_trunc": False,
    "budget_boundary": False,
    "no_progress_micro_steps": 0,
    "timeout_without_progress": False,
}
ff_shell._advance_to_learning_window = lambda: _AdvanceOutcome(
    np.zeros(298, dtype=np.float32), 37.0, True, False, (ff_extra,),
    terminal_base_info=terminal_base, terminal_death_reward=-16.0)
_, reward, term, trunc, ff_info = ff_shell.step(9)
assert term and not trunc and reward == 2.5
assert ff_info["fast_forward_reward"] == 37.0
assert ff_info["credited_fast_forward_reward"] == 0.0
assert ff_info["transition_reward"] == 2.5
assert ff_info["existing_terminal_death_reward"] == -16.0
assert ff_info["additional_terminal_death_reward"] == 0.0
assert ff_info["total_terminal_death_reward"] == -16.0
assert ff_info["episode_extra"] == terminal_base["episode_extra"]
assert ff_info["terminal_base_info"] == terminal_base
assert ff_info["terminal_option_extra"] == ff_extra
assert ff_shell.stats["transition_ff_terminal_deaths"] == 1
assert ff_shell.stats["transition_ff_terminal_death_reward"] == -16.0

# R7 explicit mode: the existing fast-forward death penalty and the user-configured extra death cost can be passed back,
# but each is counted only once; a direct FARM death only adds the cost and does not repeat the existing penalty.
ff_credit_shell = _step_shell(
    _StepOptions(direct_terminal=False, base_info=terminal_base),
    fast_forward_reward_credit="terminal-death-only",
    additional_terminal_death_cost=3.0,
)
ff_credit_shell._advance_to_learning_window = lambda: _AdvanceOutcome(
    np.zeros(298, dtype=np.float32), 37.0, True, False, (ff_extra,),
    terminal_base_info=terminal_base, terminal_death_reward=-16.0)
_, reward, term, trunc, credit_info = ff_credit_shell.step(9)
assert term and not trunc and reward == 2.5 - 16.0 - 3.0
assert credit_info["credited_fast_forward_reward"] == -19.0
assert credit_info["existing_terminal_death_reward"] == -16.0
assert credit_info["additional_terminal_death_reward"] == -3.0
assert credit_info["total_terminal_death_reward"] == -19.0
assert credit_info["transition_reward"] == reward
assert ff_credit_shell.stats["transition_ff_terminal_deaths"] == 1
assert (ff_credit_shell.stats[
    "transition_ff_additional_terminal_death_reward"] == -3.0)
assert ff_credit_shell.stats["credited_ff_terminal_death_reward"] == -19.0
assert ff_credit_shell.stats["additional_terminal_death_reward"] == -3.0

direct_cost_shell = _step_shell(
    _StepOptions(direct_terminal=True, base_info=terminal_base),
    fast_forward_reward_credit="terminal-death-only",
    additional_terminal_death_cost=3.0,
)
_, reward, term, trunc, direct_cost_info = direct_cost_shell.step(9)
assert term and not trunc and reward == -13.5 - 3.0
assert direct_cost_info["credited_fast_forward_reward"] == 0.0
assert direct_cost_info["existing_terminal_death_reward"] == -16.0
assert direct_cost_info["additional_terminal_death_reward"] == -3.0
assert direct_cost_info["total_terminal_death_reward"] == -19.0
assert direct_cost_info["transition_reward"] == reward
assert direct_cost_shell.stats["direct_terminal_deaths"] == 1
assert direct_cost_shell.stats[
    "direct_additional_terminal_death_reward"] == -3.0
assert direct_cost_shell.stats["additional_terminal_death_reward"] == -3.0

nondeath_base = {
    **terminal_base,
    "episode_extra": {
        **terminal_base["episode_extra"],
        "died": False,
    },
}
nondeath_extra = {
    **ff_extra,
    "reason": "end",
}
nondeath_shell = _step_shell(
    _StepOptions(direct_terminal=False, base_info=nondeath_base),
    fast_forward_reward_credit="terminal-death-only",
    additional_terminal_death_cost=3.0,
)
nondeath_shell._advance_to_learning_window = lambda: _AdvanceOutcome(
    np.zeros(298, dtype=np.float32), 37.0, True, False, (nondeath_extra,),
    terminal_base_info=nondeath_base, terminal_death_reward=0.0)
_, reward, term, trunc, nondeath_info = nondeath_shell.step(9)
assert term and not trunc and reward == 2.5
assert nondeath_info["credited_fast_forward_reward"] == 0.0
assert nondeath_info["existing_terminal_death_reward"] == 0.0
assert nondeath_info["additional_terminal_death_reward"] == 0.0
assert nondeath_info["total_terminal_death_reward"] == 0.0

assert _coerce_fast_forward_reward_credit("none") == "none"
assert (_coerce_fast_forward_reward_credit("terminal-death-only")
        == "terminal-death-only")
for invalid_mode in (None, False, "death", "terminal_death_only"):
    try:
        _coerce_fast_forward_reward_credit(invalid_mode)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted an illegal fast-forward credit mode: {invalid_mode!r}")
for invalid_cost in (-1, float("nan"), float("inf")):
    try:
        _coerce_additional_terminal_death_cost(invalid_cost)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted an illegal additional death cost: {invalid_cost!r}")
print("G0'.d4 PASS: the default old semantics do not collect the fast-forward return; terminal-death-only "
      "passes only the real negative death score, and the extra death cost applies exactly once inside and outside FARM; "
      "existing window-open recovery accounts are not mixed into the first worker wage tick")

# --- (d5) WorkerSentinel must aggregate the whole-window reason spectrum and the manager fast-forward sub-spectrum separately ---
sys.path.insert(0, str(ROOT / "train"))
from train_ppo import WorkerSentinelCallback

with tempfile.TemporaryDirectory() as td:
    sentinel = WorkerSentinelCallback(pathlib.Path(td), every=500_000)
    per_env_stats = [
        {
            "windows": 3, "dry": 1, "fresh": 2,
            "ff_windows": 1, "ff_dry": 0, "ff_terminals": 1,
            "episodes": 1, "reseeds": 0,
            "interrupted_resets": 0, "manual_ff_calls": 0,
            "reasons": {"death": 2, "stall": 1},
            "ff_reasons": {"death": 1},
            "direct_terminal_deaths": 1,
            "transition_ff_terminal_deaths": 1,
            "reset_ff_terminal_deaths": 0,
            "manual_ff_terminal_deaths": 0,
            "direct_no_progress_timeouts": 0,
            "transition_ff_no_progress_timeouts": 0,
            "reset_ff_no_progress_timeouts": 0,
            "manual_ff_no_progress_timeouts": 0,
            "transition_ff_reward": 0.0,
            "reset_ff_reward": 0.0,
            "manual_ff_reward": 0.0,
            "direct_existing_terminal_death_reward": -8.0,
            "direct_additional_terminal_death_reward": -32.0,
            "transition_ff_terminal_death_reward": -8.0,
            "transition_ff_additional_terminal_death_reward": -32.0,
            "credited_ff_terminal_death_reward": -40.0,
            "reset_ff_terminal_death_reward": 0.0,
            "reset_ff_additional_terminal_death_reward": 0.0,
            "additional_terminal_death_reward": -64.0,
            "direct_no_progress_timeout_failure_reward": 0.0,
            "transition_ff_no_progress_timeout_failure_reward": 0.0,
            "reset_ff_no_progress_timeout_failure_reward": 0.0,
            "manual_ff_no_progress_timeout_failure_reward": 0.0,
            "credited_no_progress_timeout_failure_reward": 0.0,
        },
        {
            "windows": 7, "dry": 2, "fresh": 5,
            "ff_windows": 3, "ff_dry": 1, "ff_terminals": 2,
            "episodes": 2, "reseeds": 0,
            "interrupted_resets": 0, "manual_ff_calls": 0,
            "reasons": {"death": 3, "descend": 4},
            "ff_reasons": {"death": 2, "descend": 1},
            "direct_terminal_deaths": 1,
            "transition_ff_terminal_deaths": 2,
            "reset_ff_terminal_deaths": 0,
            "manual_ff_terminal_deaths": 0,
            "direct_no_progress_timeouts": 0,
            "transition_ff_no_progress_timeouts": 0,
            "reset_ff_no_progress_timeouts": 0,
            "manual_ff_no_progress_timeouts": 0,
            "transition_ff_reward": 0.0,
            "reset_ff_reward": 0.0,
            "manual_ff_reward": 0.0,
            "direct_existing_terminal_death_reward": -8.0,
            "direct_additional_terminal_death_reward": -32.0,
            "transition_ff_terminal_death_reward": -16.0,
            "transition_ff_additional_terminal_death_reward": -64.0,
            "credited_ff_terminal_death_reward": -80.0,
            "reset_ff_terminal_death_reward": 0.0,
            "reset_ff_additional_terminal_death_reward": 0.0,
            "additional_terminal_death_reward": -96.0,
            "direct_no_progress_timeout_failure_reward": 0.0,
            "transition_ff_no_progress_timeout_failure_reward": 0.0,
            "reset_ff_no_progress_timeout_failure_reward": 0.0,
            "manual_ff_no_progress_timeout_failure_reward": 0.0,
            "credited_no_progress_timeout_failure_reward": 0.0,
        },
    ]
    def sentinel_get_attr(name):
        if name == "stats":
            return per_env_stats
        if name == "fast_forward_reward_credit":
            return ["terminal-death-only"] * len(per_env_stats)
        if name == "additional_terminal_death_cost":
            return [32.0] * len(per_env_stats)
        raise AssertionError(f"unexpected get_attr({name!r})")

    sentinel.model = types.SimpleNamespace(
        get_env=lambda: types.SimpleNamespace(
            get_attr=sentinel_get_attr),
        distill_beta=None,
        _last_distill_ce=None,
        _last_diverge=None,
    )
    sentinel.num_timesteps = 123
    sentinel.action_counts = np.zeros(15, dtype=np.int64)
    sentinel._emit(final=False)
    line = json.loads(
        (pathlib.Path(td) / "sentinel.jsonl").read_text().strip())
    assert line["reasons"] == {"death": 5, "stall": 1, "descend": 4}
    assert line["ff_reasons"] == {"death": 3, "descend": 1}
    assert line["transition_ff_terminal_death_reward"] == -24.0
    assert line["credited_ff_terminal_death_reward"] == -120.0
    assert line["additional_terminal_death_reward"] == -160.0
print("G0'.d5 PASS: the sentinel aggregates whole-window reasons and fast-forward ff_reasons")

# --- (a)+(c) equivalence + wage identity ---
oe = OptionsEnv(
    max_steps=3000,
    # The frozen M29 manager was trained on protocol-v3.  Fresh manager
    # training now intentionally defaults to raw-v4, so the parity arm must
    # name the same legacy deployment view that WorkerWindowEnv enforces.
    manager_observation_view="legacy-v3",
)
runA = {}
for seed in SEEDS:
    obs, _ = oe.reset(seed=seed)
    done = trunc = False
    wins = []
    while not (done or trunc):
        opt = np_mgr.choose(obs, oe.action_masks())
        obs, r, done, trunc, info = oe.step(opt)
        wins.append(win_sig(info["option_extra"]))
    runA[seed] = {"wins": wins, "steps": oe.env._steps,
                  "seq": info["option_extra"]["mode_seq"]}

wwe = WorkerWindowEnv(
    str(NPZ), max_steps=3000, rng_seed=0, log_windows=True,
    seed_scope="replay")
runB = {}
natural_continuations = 0
dive_continuations = 0
fast_forward_terminals = 0
worker_decision_boundaries = 0
excluded_opening_windows = 0
for seed in SEEDS:
    n0 = len(wwe.window_log)
    obs, _ = wwe.reset(seed=seed)
    credited_wage = Counter()
    credited_kills = Counter()
    while obs is not None:
        worker_decision_boundaries += 1
        assert not decision_boundary_busy(wwe.oe.env._raw), (
            seed, wwe.oe.env._steps, wwe.oe.env._raw)
        worker_mask, nearest = wwe.oe._worker_masks_and_distance()
        a = dispatch(
            "farm", wwe.oe.env._raw, bool(worker_mask[14]),
            action_mask=worker_mask,
            nearest_engageable_distance=nearest,
        )
        worker_window = wwe.oe._win
        window_id = int(worker_window["window_id"])
        worker_kills_before = int(worker_window["worker_kills"])
        obs2, w, term, trunc, info = wwe.step(a)
        # Worker policy reward may additionally contain a terminal
        # no-progress failure cost; the option ledger's worker_wage remains
        # the action-attributable FARM wage.
        credited_wage[window_id] += float(info["worker_wage"])
        credited_kills[window_id] += (
            int(worker_window["worker_kills"]) - worker_kills_before)
        assert info["requested_action"] == a
        assert info["transition_reward"] == w
        if not info["overridden"]:
            effect_audit = info["action_effect_audit"]
            assert isinstance(effect_audit, dict), info
            assert info["executed_action"] == (
                a if effect_audit["request_executed"] else None)
            if a == 14:
                gear_audit = info["action14_audit"]
                assert isinstance(gear_audit, dict), info
                if gear_audit["accepted"]:
                    assert info["executed_action"] == 14
        if info["farm_window_end"]:
            ex = info["option_extra"]
            assert ex["window_id"] == info["window_id"]
            assert abs(ex["worker_wage"] - credited_wage[window_id]) < 1e-6, ex
            assert ex["worker_kills"] == credited_kills[window_id], ex
            ff = info["fast_forward_extras"]
            expected_ff = (sum(float(e["R"]) for e in ff)
                           + float(info["next_window_opening_reward"]))
            assert abs(info["fast_forward_reward"] - expected_ff) < 1e-6, info
            timeout_failure = float(
                info["no_progress_timeout_failure_reward"])
            if info["worker_no_progress_timeout"] and not ex["base_done"]:
                assert (
                    info["credited_fast_forward_reward"]
                    == timeout_failure
                ), info
            else:
                assert info["credited_fast_forward_reward"] == 0.0, info
            direct_timeout_failure = (
                timeout_failure
                if ex["base_done"]
                else 0.0
            )
            expected_policy_reward = (
                float(info["worker_wage"])
                + float(info["credited_fast_forward_reward"])
                + direct_timeout_failure
            )
            assert abs(w - expected_policy_reward) < 1e-6, info
            if not ex["base_done"]:
                if term or trunc:
                    # A real underlying end after a natural FARM must come from the manager/script fast-forward
                    # within the same transition, not from the next reset.
                    assert ff and ff[-1]["base_done"], info
                    assert info["terminal_option_extra"] == ff[-1]
                    fast_forward_terminals += 1
                else:
                    natural_continuations += 1
                    assert info["next_window_id"] is not None
                    assert wwe.oe._win is not None
                    assert info["next_window_id"] == wwe.oe._win["window_id"]
            if any(e["opt"] == 1 for e in ff):
                dive_continuations += 1
        obs = wwe.next_window() if (term or trunc) else obs2
    entries = wwe.window_log[n0:]
    from diablogym.env import DESCEND_UNIT
    for e in entries:
        assert abs(e["W"] - (e["R"] - e["bonus"])) < 1e-6, e   # (c) the ledger is self-consistent
        assert 0 <= e["worker_kills"] <= e["kills_delta"], e
        if e["window_id"] in credited_wage:
            assert abs(e["worker_wage"]
                       - credited_wage[e["window_id"]]) < 1e-6, e
            assert e["worker_kills"] == credited_kills[e["window_id"]], e
            if (e["recovery_actions"] > 0
                    and abs(e["W"] - e["worker_wage"]) > 1e-12):
                excluded_opening_windows += 1
        else:
            # Frozen DIVE/RESUPPLY, curriculum proxy play, or windows closed directly by the window-open drain have no
            # _win_step_worker transition and must not fake network wage/kills.
            assert e["worker_wage"] == 0.0 and e["worker_kills"] == 0, e
        # (c') independent reconciliation of the wage-stripping formula: bonus == DESCEND_UNIT x sum(range(dlvl0, dlvl_end))
        # (the seven-step ladder guarantees that a level change closes the window, so the per-window sum of positive dlvl deltas collapses to the endpoint difference)
        expect = (DESCEND_UNIT * sum(range(e["dlvl0"], e["dlvl_end"]))
                  if e["dlvl_end"] > e["dlvl0"] else 0.0)
        assert abs(e["bonus"] - expect) < 1e-6, e
        if e["reason"] == "descend":
            assert e["bonus"] > 0.0, e
        elif e["opt"] == FARM and not e["base_done"]:
            assert e["bonus"] == 0.0, e   # exempt when the level-change tick coincides with the episode end (reason=death/end)
    assert sum(e["kills_delta"] for e in entries) == wwe.oe.env._ep_kills
    runB[seed] = {"wins": [win_sig(e) for e in entries],
                  "steps": wwe.oe.env._steps,
                  "seq": entries[-1]["mode_seq"],
                  "depth": int(wwe.oe.env._raw["dungeon_level"]),
                  "dry_windows": sum(bool(e["dry"]) for e in entries)}

for seed in SEEDS:
    A, B = runA[seed], runB[seed]
    assert A["wins"] == B["wins"], (
        f"seed {seed} window sequence mismatch:\nA={A['wins'][:8]}...\nB={B['wins'][:8]}...\n"
        f"len A {len(A['wins'])} B {len(B['wins'])}")
    assert A["steps"] == B["steps"] and A["seq"] == B["seq"], (seed, A["steps"], B["steps"])

# seed 7001 historically produced a unique item on its first run and a
# different non-unique item on every replay: upstream clears UniqueItemFlags
# only in the non-headless UI path.  The same audit also caught process-lived
# Monster fields, bestiary kills, and SDL-tick starter item seeds.  Burn no new
# held-out seed; replay this already-registered one three times in one native
# process and compare every scripted action boundary, not only final return.
replay_seed = 7001
replays = []
original_win_beat = oe._win_beat
for replay_index in range(3):
    action_trace = []

    def traced_win_beat(a, *, worker_authority=False):
        before = native_trajectory_state_digest(oe)
        outcome = original_win_beat(
            a, worker_authority=worker_authority)
        audit = outcome.action14_audit
        action_trace.append((
            int(a),
            bool(worker_authority),
            before,
            native_trajectory_state_digest(oe),
            outcome.reason,
            outcome.executed_action,
            bool(outcome.fuse_tripped),
            None if audit is None else (
                bool(audit["accepted"]),
                int(audit["commit_attempts"]),
                int(audit["utility_before"]),
                int(audit["utility_after"]),
                int(audit["utility_delta"]),
            ),
        ))
        return outcome

    oe._win_beat = traced_win_beat
    obs, _ = oe.reset(seed=replay_seed)
    initial_digest = native_trajectory_state_digest(oe)
    initial_observation_digest = hashlib.sha256(
        np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()
    assert oe.env._raw["monster_kill_total"] == 0, (
        replay_index, oe.env._raw["monster_kill_total"])
    done = trunc = False
    replay_windows = []
    while not (done or trunc):
        opt = np_mgr.choose(obs, oe.action_masks())
        obs, _, done, trunc, replay_info = oe.step(opt)
        replay_windows.append(win_sig(replay_info["option_extra"]))
    replays.append({
        "initial_digest": initial_digest,
        "initial_observation_digest": initial_observation_digest,
        "actions": action_trace,
        "windows": replay_windows,
        "steps": int(oe.env._steps),
        "kills": int(oe.env._ep_kills),
        "mode_seq": list(replay_info["option_extra"]["mode_seq"]),
    })
oe._win_beat = original_win_beat
assert replays[1:] == [replays[0], replays[0]], (
    "seed7001: repeated reset in the same process / trajectory not deterministic",
    [(r["initial_digest"], r["windows"], r["steps"], r["kills"])
     for r in replays],
)
print("G0'.a0 PASS: seed7001, three consecutive runs in the same process: initial state / native kill ledger / "
      "per-tick action trajectory / per-window results identical")

# protocol-v4's action10 no longer sneaks downstairs; if the frozen H still allowed FARM after exhausted,
# it would reselect 46--96 dry windows in a row on real seeds and trap the whole episode on L1. The manager
# mask must hand the exhausted state to a legal DIVE: this assembled assertion covers the real H forward pass, the mask,
# the Worker fast-forward and ordinary door-aware exploration together, and does not accept a fix that only works on a synthetic fixture.
assert all(runB[s]["depth"] >= 2 for s in SEEDS), {
    s: runB[s]["depth"] for s in SEEDS
}
assert all(runB[s]["dry_windows"] == 0 for s in SEEDS), {
    s: runB[s]["dry_windows"] for s in SEEDS
}
all_reason_hist = dict(Counter(e["reason"] for e in wwe.window_log))
ff_reason_hist = dict(Counter(
    e["reason"] for e in wwe.window_log if e["ff"]))
assert wwe.stats["reasons"] == all_reason_hist
assert wwe.stats["ff_reasons"] == ff_reason_hist
assert sum(wwe.stats["reasons"].values()) == len(wwe.window_log)
assert sum(wwe.stats["ff_reasons"].values()) == wwe.stats["ff_windows"]
print(f"G0'.a PASS: {len(SEEDS)} seeds: window sequence / tau / per-window R / mode_seq / micro-step end point identical bit for bit "
      f"({sum(len(runA[s]['wins']) for s in SEEDS)} windows in total)")
print("G0'.a2 PASS: the frozen H leaves L1 on all of 7000-7007, with no dry-window reselection after exhausted")
print("G0'.c PASS: the wage identity holds; whole-window reasons / fast-forward ff_reasons miss nothing")
assert natural_continuations > 0, "no natural FARM -> next FARM continuation transition observed on real seeds"
print(f"G0'.c2 PASS: natural FARM continuation transitions {natural_continuations}; "
      f"of which with a spontaneous DIVE consequence from the frozen H {dive_continuations}, "
      f"fast-forward underlying ends {fast_forward_terminals}")
assert worker_decision_boundaries > 0
print(f"G0'.c3 PASS: {worker_decision_boundaries} real worker decision boundaries with busy=0")
print(f"G0'.c4 PASS: window-open recovery wage exclusion is sealed by a deterministic unit test; the real seeds here also cover "
      f"{excluded_opening_windows} non-zero opening recovery windows")

# --- (b) worker mask (v32 course 4c: 12 follows the sovereignty knob, passed through by default; 11 always masked, unchanged) ---
obs, _ = wwe.reset(seed=7008)
m = wwe.action_masks()
base = wwe.oe.env.action_masks()
assert m.shape == (15,) and m.dtype == bool
assert not m[11], m
raw_b = wwe.oe.env._raw
hp_b = raw_b["hp"] / max(1, raw_b["max_hp"])
assert m[12] == (
    base[12]
    and raw_b.get("belt_heals", 0) > 0
    and 0.5 <= hp_b < 0.75
), (m[12], base[12], raw_b.get("belt_heals"), hp_b)
assert m[14] == base[14]
assert obs.shape == (298,), obs.shape
legacy = WorkerWindowEnv(str(NPZ), max_steps=3000, rng_seed=0,
                         drink_sovereignty=False, seed_scope="replay")
legacy.reset(seed=7008)
lm = legacy.action_masks()
assert not lm[11] and not lm[12], lm
legacy.close()
print("G0'.b PASS: mask always masks 11; 12 depends only on visible [0.5,0.75) + having potions "
      "(always masked under the old protocol knob); 14 passed through; worker observation 298-dim")

# --- (b1) Worker info must report fuse refusals truthfully and must not mislabel the result of action 10 as the requested action ---
fuse_env = WorkerWindowEnv(
    str(NPZ), max_steps=3000, rng_seed=0, seed_scope="replay")
fuse_env.reset(seed=7008)
requested = 10
assert fuse_env.action_masks()[requested]
fuse_env.oe._fuse_sig = fuse_env.oe._sig(requested, fuse_env.oe.env._raw)
fuse_env.oe._fuse = 24
# On real seed 7008 DIVE is legal at this point; force the manager path to show that the real DIVE window reward/consequence after the FARM
# boundary is consumed by the same worker transition.
manager_choices = iter([DIVE, FARM])
fuse_env._mgr_choose = lambda: next(manager_choices)
steps_before = fuse_env.oe.env._steps
_, _, term, trunc, finfo = fuse_env.step(requested)
assert finfo["farm_window_end"] and finfo["option_extra"]["reason"] == "fuse"
assert finfo["requested_action"] == requested
assert finfo["executed_action"] is None
assert finfo["fuse_tripped"] and finfo["fuse_requested_action"] == requested
assert finfo["overridden"] is True
# The manager fast-forward later in the same step may legally advance micro-steps; the refused worker tick itself
# must keep beats=0/R=W=0 in option_extra, proving that exploration 10 was not executed behind the scenes.
fex = finfo["option_extra"]
assert fex["beats"] == 0 and fex["R"] == 0.0 and fex["W"] == 0.0
assert fex["fuse_trips"] == 1
assert fuse_env.oe.env._steps >= steps_before
assert finfo["fast_forward_extras"]
assert finfo["fast_forward_extras"][0]["opt"] == DIVE
assert finfo["fast_forward_extras"][0]["recovery_actions"] == 1
assert finfo["fast_forward_extras"][0]["last_recovery_action"] == 11
fuse_env.close()
print("G0'.b1 PASS: the fuse refuses explicitly, and on a real seed the following DIVE consequence goes into the same transition")

# --- (b1b) a real underlying DIVE fast-forward terminal must not swallow episode_extra / the last-tick reward ---
ff_terminal_env = WorkerWindowEnv(
    str(NPZ), max_steps=3000, rng_seed=0, seed_scope="replay")
ff_terminal_env.reset(seed=7008)
requested = 10
assert ff_terminal_env.action_masks()[requested]
ff_terminal_env.oe._fuse_sig = ff_terminal_env.oe._sig(
    requested, ff_terminal_env.oe.env._raw)
ff_terminal_env.oe._fuse = 24
ff_terminal_env.oe.env._steps = ff_terminal_env.oe.max_steps - 1
ff_terminal_env._mgr_choose = lambda: DIVE
_, reward, term, trunc, terminal_info = ff_terminal_env.step(requested)
assert term or trunc
assert terminal_info["fast_forward_extras"]
terminal_dive = terminal_info["fast_forward_extras"][-1]
assert terminal_dive["opt"] == DIVE and terminal_dive["base_done"]
assert terminal_info["terminal_option_extra"] == terminal_dive
assert terminal_info["episode_extra"] == (
    terminal_info["terminal_base_info"]["episode_extra"])
assert reward == terminal_info["transition_reward"]
ff_terminal_env.close()
print("G0'.b1b PASS: on a real seed the DIVE fast-forward terminal reward / episode_extra return in the same transition")

# --- (b1c) a wait action must not claim the combat and descent gains of the frozen manager/script ---
# This is a real, exploitable old training loophole: the worker outputs only a0 and, after the FARM window closes, lets the
# frozen DIVE fight / descend for it; if continuation.reward were added back to the current transition,
# a0-only would also earn a dozen to dozens of points of positive return on these seeds. The FF account is now kept for
# conservation audits, but the policy return must equal only this FARM wage tick by tick, and a0 itself is non-positive.
a0_positive_fast_forward = 0.0
a0_boundary_steps = 0
a0_policy_return = 0.0
for seed in SEEDS:
    a0_env = WorkerWindowEnv(
        str(NPZ), max_steps=3000, rng_seed=0, seed_scope="replay")
    a0_env.reset(seed=seed)
    for _ in range(800):
        _, reward, term, trunc, a0_info = a0_env.step(0)
        assert reward == a0_info["worker_wage"]
        assert reward == a0_info["transition_reward"]
        assert reward <= 1e-9, (seed, reward, a0_info)
        a0_policy_return += float(reward)
        if a0_info["farm_window_end"]:
            a0_boundary_steps += 1
            expected_ff = (
                sum(float(e["R"]) for e in a0_info["fast_forward_extras"])
                + float(a0_info["next_window_opening_reward"])
            )
            assert abs(a0_info["fast_forward_reward"] - expected_ff) < 1e-6
            assert a0_info["credited_fast_forward_reward"] == 0.0
            a0_positive_fast_forward += max(
                0.0, float(a0_info["fast_forward_reward"]))
        if term or trunc:
            assert "episode_extra" in a0_info
            break
    a0_env.close()
assert a0_boundary_steps > 0
assert a0_positive_fast_forward > 8.0, a0_positive_fast_forward
assert a0_policy_return < 0.0, a0_policy_return
print("G0'.b1c PASS: on 8 real seeds a0-only has a non-positive wage every tick; the frozen script's positive FF "
      f"{a0_positive_fast_forward:.2f} goes only into the audit and does not leak to the policy")

# --- (b2) the Gym seed must take over the sampler of later automatic episode rolls ---
wwe_seed = WorkerWindowEnv(
    str(NPZ), max_steps=3000, rng_seed=999, seed_scope="replay")
obs, info = wwe_seed.reset(seed=7008)
assert info["episode_seed"] == 7008 and wwe_seed.np_random is not None
expected_rng = np.random.default_rng(7008)
assert sample_train_seed(wwe_seed._rng) == sample_train_seed(expected_rng)
print("G0'.b2 PASS: reset(seed) synchronises the worker's episode-roll RNG; later episodes are reproducible")

assert EVAL_RESERVED_SEED_RANGES == (
    (7000, 7032), (9000, 9032), (12000, 12032),
    (2_110_000, 2_130_000))
assert BC_RESERVED_SEED_RANGES == (
    (2000, 2128), (3000, 3384),
    (2_100_000, 2_100_128), (2_101_000, 2_101_384),
    (2_102_000, 2_102_128), (2_103_000, 2_103_384),
    # 2026-07-27: 2_102/2_104 were burned one after the other; the active range advances (amendment A2)
    (2_104_000, 2_104_128), (2_106_000, 2_106_128),
    (2_108_000, 2_108_128), (2_140_000, 2_140_128),
    (2_141_000, 2_141_384), (2_142_000, 2_142_128),
    (2_143_000, 2_143_384))
assert HISTORICAL_BURNED_BC_SEED_RANGES == (
    (100, 484), (1000, 1384))
for reserved in (
    7000, 7031, 9000, 9031, 12000, 12031,
    2_110_000, 2_120_000, 2_129_999,
):
    assert is_reserved_eval_seed(reserved)
for ordinary in (
    6999, 7032, 8999, 9032, 11999, 12032,
    2_109_999, 2_130_000, 304000,
):
    assert not is_reserved_eval_seed(ordinary)
for reserved in (
    100, 483, 1000, 1383,
    2000, 2127, 3000, 3383,
    2_100_000, 2_100_127, 2_101_000, 2_101_383,
    2_102_000, 2_102_127, 2_103_000, 2_103_383,
    2_104_000, 2_104_127, 2_106_000, 2_106_127,
    2_108_000, 2_108_127,
    2_140_000, 2_140_127, 2_141_000, 2_141_383,
    2_142_000, 2_142_127, 2_143_000, 2_143_383,
    2_110_000, 2_129_999,
):
    assert is_reserved_train_seed(reserved)
for ordinary in (99, 484, 999, 1384):
    assert not is_reserved_train_seed(ordinary)


class _ReservedPoolFirstRng:
    def __init__(self):
        self.values = iter((
            100, 483, 1000, 1383,
            2_100_000, 2_101_383,
            2_102_000, 2_103_383,
            42,
        ))
        self.calls = 0

    def integers(self, low, high):
        assert (low, high) == (0, 2**31)
        self.calls += 1
        return next(self.values)


reserved_first_rng = _ReservedPoolFirstRng()
assert sample_train_seed(reserved_first_rng) == 42
assert reserved_first_rng.calls == 9
print("G0'.b2a PASS: all historical pools, the new BC pools and the R7 eval bank share the training rejection table")

train_scope = WorkerWindowEnv(
    str(NPZ), max_steps=3000, rng_seed=0, seed_scope="train")
try:
    train_scope.reset(seed=2_120_000)
except ValueError as exc:
    assert "refuses reserved seed" in str(exc)
else:
    raise AssertionError("an explicit reset in ordinary training bypassed the R7 reserved pool")
train_scope.close()

scope_shell = object.__new__(WorkerWindowEnv)
scope_shell.seed_scope = "bc-v1"
scope_shell._rng = np.random.default_rng(0)
scope_shell._p_rng = np.random.default_rng(0)
scope_shell.oe = types.SimpleNamespace(reset=lambda *, seed, options: None)
scope_shell.stats = {"episodes": 0}
scope_shell.seed_scope = "train"
for burned_seed in (100, 1000):
    try:
        scope_shell._new_episode(seed=burned_seed)
    except ValueError as exc:
        assert "refuses reserved seed" in str(exc)
    else:
        raise AssertionError(
            f"ordinary training accepted a historically burned BC seed {burned_seed}")
scope_shell.seed_scope = "replay"
scope_shell._new_episode(seed=100)
assert scope_shell._episode_seed == 100
scope_shell.seed_scope = "bc-v1"
try:
    scope_shell._new_episode(seed=2_100_000)
except ValueError as exc:
    assert "bc-v1 only allows the registered pool" in str(exc)
else:
    raise AssertionError("the BC-v1 scope accepted a burned v1 seed")
scope_shell._new_episode(seed=2_142_000)
scope_shell.seed_scope = "bc-v2"
try:
    scope_shell._new_episode(seed=2_101_000)
except ValueError as exc:
    assert "bc-v2 only allows the registered pool" in str(exc)
else:
    raise AssertionError("the BC-v2 scope accepted a burned v2 seed")
scope_shell._new_episode(seed=2_143_000)
try:
    scope_shell._new_episode()
except RuntimeError as exc:
    assert "automatic rollover into an unregistered seed" in str(exc)
else:
    raise AssertionError("the BC scope rolled into an ordinary training seed automatically")
print("G0'.b2a2 PASS: explicit train/BC seed scopes block reserved-pool bypasses and demonstration escapes")

# The p_skip stream must be re-derived for each actual episode seed; the second auto episode
# must not keep consuming the first episode's stream.
seed_shell = object.__new__(WorkerWindowEnv)
seed_shell._rng = np.random.default_rng(404)
seed_shell._p_rng = np.random.default_rng(0)
seed_shell.seed_scope = "train"
seed_shell.oe = types.SimpleNamespace(reset=lambda *, seed, options: None)
seed_shell.stats = {"episodes": 0}
reference_train_rng = np.random.default_rng(404)
expected_s1 = sample_train_seed(reference_train_rng)
expected_s2 = sample_train_seed(reference_train_rng)
seed_shell._new_episode()
assert seed_shell._episode_seed == expected_s1
assert seed_shell._p_rng.bit_generator.state == (
    _derive_p_skip_rng(expected_s1).bit_generator.state)
seed_shell._new_episode()
assert seed_shell._episode_seed == expected_s2
assert seed_shell._p_rng.bit_generator.state == (
    _derive_p_skip_rng(expected_s2).bit_generator.state)
print("G0'.b2b PASS: the dedicated p_skip stream is rebuilt per real episode_seed; the second episode can be replayed independently")

# --- (b3) a reset in the middle of a window must discard the old episode and must not overwrite an unsettled window and mix accounts ---
episodes_before = wwe_seed.stats["episodes"]
interrupted_before = wwe_seed.stats["interrupted_resets"]
assert wwe_seed.oe._win is not None
_, reset_info = wwe_seed.reset()
assert wwe_seed.stats["episodes"] == episodes_before + 1
assert wwe_seed.stats["interrupted_resets"] == interrupted_before + 1
assert wwe_seed.oe._win is not None and reset_info["episode_seed"] == wwe_seed._episode_seed
print("G0'.b3 PASS: a reset in the middle of an active window forces a new underlying episode without mixing manager/wage state")

# --- (e) real underlying boundary semantics + VecEnv TimeLimit.truncated only when idle ---
seen_natural = natural_continuations > 0
seen_term = seen_trunc = False
# Natural FARM -> next FARM is already proven transition by transition by the main 7000--7007 replay above.
# The short-budget sweep here covers only underlying terminal/truncation; it must no longer rely on the same
# seed leaking different drops/boundaries across resets to create natural windows "by luck".
# 150 micro-ticks would make the DIVE forced after exhausted hit the TimeLimit within the same transition,
# so "natural boundary -> next FARM" could not be observed; 600 covers both the continuation and the truncation paths.
wwe_s = WorkerWindowEnv(
    str(NPZ), max_steps=600, rng_seed=3, log_windows=False,
    seed_scope="replay")
obs, _ = wwe_s.reset(seed=101)
for _ in range(2000):
    worker_mask, nearest = wwe_s.oe._worker_masks_and_distance()
    a = dispatch(
        "farm", wwe_s.oe.env._raw, bool(worker_mask[14]),
        action_mask=worker_mask,
        nearest_engageable_distance=nearest,
    )
    obs, w, term, trunc, info = wwe_s.step(a)
    if info["farm_window_end"] and not info["option_extra"]["base_done"]:
        if not (term or trunc):
            seen_natural = True
        else:
            assert info["fast_forward_extras"]
            assert info["fast_forward_extras"][-1]["base_done"]
    if term or trunc:
        assert term != trunc, (term, trunc)
        assert "episode_extra" in info, info
        assert info["terminal_base_info"]["episode_extra"] == info["episode_extra"]
        if trunc:
            terminal = info.get("terminal_option_extra")
            assert terminal is not None and terminal["base_trunc"], info
            seen_trunc = True
        else:
            seen_term = True
        if seen_term and seen_trunc:
            break
        obs, _ = wwe_s.reset()
assert seen_trunc, "no truncated path observed in a 600-step episode"
assert seen_natural, "a natural FARM boundary is still wrongly marked terminal/truncated"

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

vec = DummyVecEnv([lambda: Monitor(WorkerWindowEnv(
    str(NPZ), max_steps=150, rng_seed=5, seed_scope="replay"))])
vobs = vec.reset()
tl_seen = False
for _ in range(3000):
    vobs, vr, vdone, vinfos = vec.step(np.asarray([10]))
    if vdone[0]:
        if vinfos[0].get("TimeLimit.truncated"):
            assert "terminal_observation" in vinfos[0]
            tl_seen = True
            break
vec.close()
assert tl_seen, "VecEnv did not inject TimeLimit.truncated (the SB3 bootstrap branch depends on it)"
print("G0'.e PASS: natural window closes continue and can bootstrap; underlying ends / unsafe budget interruptions are "
      "terminated, only idle time limits are truncated; "
      f"VecEnv still injects the safe TimeLimit.truncated (this sweep terminated={seen_term})")

print("G0' ALL PASS")
