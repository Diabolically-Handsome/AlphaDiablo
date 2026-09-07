"""R18-B2: the training window and the trainer accept completion-l2-r18c.

The window half loads ``worker_env`` through a private package alias whose
``__path__`` points at ``python/diablogym`` (the house pattern of
tests/test_resource_retreat.py), plus a stub ``bridge`` submodule, so nothing
here builds, resets or steps the native engine.  The trainer half imports
``train_ppo`` and only reads its pure identity/validation helpers.

House law under test: with ``legacy`` or ``completion-l2-v1`` nothing moved --
the same guards fire, the same keywords are forwarded, and the contract key
``worker_time_recipe`` still equals ``COMPLETION_L2_V1.as_dict()`` so every
existing checkpoint keeps matching.  ``completion-l2-r18c`` is a separate
identity: its recipe differs, and a v1 checkpoint therefore cannot resume into
an r18c arm (or the reverse) under any drift allowance.
"""
import argparse
import ast
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "python"), str(ROOT / "tests")]

PACKAGE = "_r18b2_completion_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(ROOT / "python/diablogym")]
    sys.modules[PACKAGE] = package
if PACKAGE + ".bridge" not in sys.modules:
    bridge_stub = ModuleType(PACKAGE + ".bridge")
    bridge_stub.WM_DIABPREVLVL = 1027  # a fake engine trigger id; never used here
    sys.modules[PACKAGE + ".bridge"] = bridge_stub

worker_env = importlib.import_module(PACKAGE + ".worker_env")
clock = importlib.import_module(PACKAGE + ".completion_clock")

import prefix_worker
import train_ppo as training
from test_completion_training_contract import contract, factory_args, valid_args
from test_resource_sustain_contract import DummyEnv

V1 = "completion-l2-v1"
R18C = "completion-l2-r18c"
PROTOCOLS = (V1, R18C)
SCOPE = "earned-dive-suffix-v1"
ARRIVAL = clock.COMPLETION_L2_V1.arrival_microsteps  # 12000, shared by both recipes


# ---- the training window (WorkerWindowEnv) ----

class StubOptions:
    """Enough of a completion-protocol OptionsEnv for the wrapper; no engine."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_steps = kwargs.get("max_steps", 6000)
        self.env = SimpleNamespace(
            max_steps=ARRIVAL, _steps=0, _resource_actual_microsteps=0, _raw=None,
            _completion_prefix_active=False, _completion_prefix_deadline=None,
            observation_space=SimpleNamespace(shape=(295,)))
        self._completion_clock = SimpleNamespace(
            state=SimpleNamespace(physical_deadline=ARRIVAL))


def prefix_callback():
    """The verified R16 parent's public surface; the wrapper only inspects it."""
    def callback(observation, action_mask):
        return 9
    callback.source_sha256 = worker_env._PREFIX_WORKER_SHA256
    callback.diablogym_worker_observation_view = "dual-v4-asymmetric-v3"
    callback.diablogym_worker_action12_mode = "environment-mask"
    callback.episode_reseed = Mock()
    callback.on_beat = None
    return callback


def build(protocol, **overrides):
    arguments = dict(
        manager_npz=None, manager_heuristic="readiness-v1", max_steps=6000,
        learning_window_scope=SCOPE, policy_observation_view="dual-v4-asymmetric-v3",
        resource_protocol="l2-town-v1", resource_purchase_mode="full",
        resource_service_policy="sustain-v6", prefix_worker=prefix_callback(),
        prefix_worker_sha256=worker_env._PREFIX_WORKER_SHA256,
        prefix_max_attempts=2, prefix_max_microsteps=100,
        worker_time_protocol=protocol)
    arguments.update(overrides)
    with patch.object(worker_env, "OptionsEnv",
                      side_effect=lambda **kw: StubOptions(**kw)) as options:
        return worker_env.WorkerWindowEnv(**arguments), options


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_window_accepts_both_recipes_and_forwards_exactly_that_protocol(protocol):
    env, options = build(protocol)
    assert env.worker_time_protocol == protocol
    options.assert_called_once()
    assert env.oe.kwargs["worker_time_protocol"] == protocol
    assert "worker_time_recipe" not in env.oe.kwargs


def test_legacy_default_forwards_no_time_keyword_at_all():
    env, options = build("legacy", prefix_worker=None, prefix_worker_sha256=None,
                         prefix_max_attempts=None, prefix_max_microsteps=None,
                         learning_window_scope="farm-only",
                         policy_observation_view=None)
    assert env.worker_time_protocol == "legacy"
    options.assert_called_once()
    assert "worker_time_protocol" not in env.oe.kwargs


@pytest.mark.parametrize("value", [
    "completion-l2", "completion-l2-v2", "completion-l2-r18d", "COMPLETION-L2-R18C",
    "r18c", "off", "", None, 1, True])
def test_unknown_protocols_are_refused_before_the_options_constructor(value):
    with patch.object(worker_env, "OptionsEnv") as options:
        with pytest.raises(ValueError, match="Unknown worker_time_protocol"):
            build(value)
    options.assert_not_called()


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("scope", ["farm-only", "farm-dive-v1"])
def test_both_recipes_need_the_earned_dive_suffix_scope(protocol, scope):
    with patch.object(worker_env, "OptionsEnv") as options:
        with pytest.raises(ValueError, match="require earned-dive-suffix-v1"):
            build(protocol, learning_window_scope=scope, prefix_worker=None,
                  prefix_worker_sha256=None, prefix_max_attempts=None,
                  prefix_max_microsteps=None)
    options.assert_not_called()


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_inactive_prefix_limit_reads_the_live_clock_of_either_recipe(protocol):
    """Site A: after arrival the physical deadline is the clock's, not the original."""
    env, _ = build(protocol)
    deadline = ARRIVAL + clock.COMPLETION_RECIPES[protocol].followup_microsteps
    assert deadline == (13800 if protocol == V1 else 21000)
    env.oe._completion_clock.state.physical_deadline = deadline
    env.oe.env.max_steps = deadline
    env._prefix_validate_limits(active=False)
    env.oe.env.max_steps = deadline + 1
    with pytest.raises(RuntimeError, match="outside its owner"):
        env._prefix_validate_limits(active=False)


def test_legacy_prefix_limit_still_ignores_any_clock():
    env, _ = build("legacy")
    env.oe._completion_clock.state.physical_deadline = 21000
    env._prefix_validate_limits(active=False)  # the original limit, unchanged
    env.oe.env.max_steps = 21000
    with pytest.raises(RuntimeError, match="outside its owner"):
        env._prefix_validate_limits(active=False)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_prefix_reset_and_restore_toggle_the_completion_prefix_flags(protocol):
    """Sites B and C: the prefix owns the physical limit under either recipe."""
    env, _ = build(protocol)
    env._prefix_wall_start = 0.0
    env._prefix_attempt = {}
    env._prefix_after_reset()
    assert env._prefix_deadline == 100 and env.oe.env.max_steps == 100
    assert env.oe.env._completion_prefix_active is True
    assert env.oe.env._completion_prefix_deadline == 100
    env._prefix_restore_limit()
    assert env._prefix_deadline is None and env.oe.env.max_steps == ARRIVAL
    assert env.oe.env._completion_prefix_active is False
    assert env.oe.env._completion_prefix_deadline is None


def test_legacy_prefix_reset_never_touches_the_completion_flags():
    env, _ = build("legacy")
    env._prefix_wall_start = 0.0
    env._prefix_attempt = {}
    env._prefix_after_reset()
    assert env.oe.env._completion_prefix_active is False
    assert env.oe.env._completion_prefix_deadline is None
    env._prefix_restore_limit()
    assert env.oe.env._completion_prefix_active is False
    assert env.oe.env._completion_prefix_deadline is None


# ---- the trainer (train/train_ppo.py) ----

@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_trainer_returns_the_immutable_recipe_of_each_protocol(protocol):
    expected = clock.COMPLETION_RECIPES[protocol].as_dict()
    recipe = training._worker_time_identity(protocol)
    assert recipe == expected and recipe["protocol"] == protocol
    recipe["followup_microsteps"] = 1
    assert training._worker_time_identity(protocol) == expected  # detached copy


def test_r18c_moves_only_the_protocol_and_the_followup():
    v1 = training._worker_time_identity(V1)
    r18c = training._worker_time_identity(R18C)
    assert v1 == clock.COMPLETION_L2_V1.as_dict()  # every old checkpoint still matches
    assert {key for key in set(v1) | set(r18c)
            if v1[key] != r18c[key]} == {"protocol", "followup_microsteps"}
    assert v1["followup_microsteps"] == 1800 and r18c["followup_microsteps"] == 9000
    assert r18c["arrival_microsteps"] == 12000 and r18c["actor_denominator"] == 6000
    assert r18c["service_microsteps"] == 3000 and r18c["farm_microsteps"] == 3600


def test_legacy_is_still_the_absent_identity():
    assert training._worker_time_identity() is None
    assert training._worker_time_identity("legacy") is None


@pytest.mark.parametrize("value", [
    None, False, 1, "off", "completion-l2", "completion-l2-v2", "completion-l2-r18d",
    "COMPLETION-L2-R18C"])
def test_trainer_still_refuses_unknown_protocols(value):
    with pytest.raises(ValueError):
        training._worker_time_identity(value)


def r18c_args(**overrides):
    args = valid_args()
    args.worker_time_protocol = R18C
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_r18c_passes_the_same_world_requirements_v1_passes():
    assert training._validate_worker_time_args(r18c_args()) is None


@pytest.mark.parametrize("key,value", [
    ("worker", False), ("worker_learning_window_scope", "farm-dive-v1"),
    ("worker_learning_window_scope", "farm-only"), ("resource_protocol", "off"),
    ("resource_purchase_mode", "armor"), ("resource_service_policy", "sustain-v5"),
    ("max_steps", 12000), ("farm_scene_cap", 7200),
])
def test_r18c_rejects_the_same_alternate_worlds_v1_rejects(key, value):
    with pytest.raises(ValueError):
        training._validate_worker_time_args(r18c_args(**{key: value}))


@pytest.mark.parametrize("retreat", ["off", "retreat-v1"])
@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_the_time_recipe_and_the_retreat_arm_stay_independent(protocol, retreat):
    args = valid_args()
    args.worker_time_protocol = protocol
    args.resource_retreat = retreat
    assert training._validate_worker_time_args(args) is None


def time_contract(protocol):
    args = valid_args()
    args.worker_time_protocol = protocol
    return contract(args)


def test_the_v1_contract_is_untouched_and_r18c_moves_only_the_two_time_keys():
    v1, r18c = time_contract(V1), time_contract(R18C)
    assert v1["worker_time_protocol"] == V1
    assert v1["worker_time_recipe"] == clock.COMPLETION_L2_V1.as_dict()
    assert r18c["worker_time_recipe"] == clock.COMPLETION_L2_R18C.as_dict()
    assert {key for key in set(v1) | set(r18c)
            if v1.get(key) != r18c.get(key)} == {"worker_time_protocol",
                                                 "worker_time_recipe"}


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_same_protocol_resume_validates_exactly_as_before(protocol):
    current = time_contract(protocol)
    training._validate_resume_contract(copy.deepcopy(current), current)


@pytest.mark.parametrize("saved,current", [(V1, R18C), (R18C, V1)])
def test_cross_recipe_resume_fails_under_every_allowance(saved, current):
    with pytest.raises(ValueError, match="separately identified initialization"):
        training._validate_resume_contract(
            time_contract(saved), time_contract(current), allow_manager_change=True,
            allow_legacy_resume=True, allow_optimizer_reset=True,
            allow_target_kl_change=True, allow_environment_restart=True)


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_a_hand_edited_recipe_is_refused_for_both_protocols(protocol):
    current = time_contract(protocol)
    saved = copy.deepcopy(current)
    saved["worker_time_recipe"]["followup_microsteps"] += 1
    with pytest.raises(ValueError, match=f"exact {protocol} recipe"):
        training._validate_resume_contract(saved, current, allow_environment_restart=True)


def test_the_environment_restart_whitelist_never_covers_the_time_identity():
    assert not ({"worker_time_protocol", "worker_time_recipe"}
                & set(training._ENVIRONMENT_RESTART_ALLOWED_DRIFT))


def test_cli_choices_carry_both_recipes_and_still_default_to_legacy():
    tree = ast.parse((ROOT / "train/train_ppo.py").read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and node.value.args
             and isinstance(node.value.args[0], ast.Constant)
             and node.value.args[0].value == "--worker-time-protocol"]
    assert len(nodes) == 1
    ap = argparse.ArgumentParser()
    exec(compile(ast.Module(nodes, type_ignores=[]), "<actual-cli-node>", "exec"),
         {"ap": ap})
    assert ap.parse_args([]).worker_time_protocol == "legacy"
    for protocol in PROTOCOLS:
        assert ap.parse_args(
            ["--worker-time-protocol", protocol]).worker_time_protocol == protocol
    for rejected in ("completion-l2", "completion-l2-r18d", "wide"):
        with pytest.raises(SystemExit):
            ap.parse_args(["--worker-time-protocol", rejected])


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_make_env_forwards_one_explicit_protocol_and_no_recipe(protocol):
    kwargs = factory_args()
    kwargs["worker_time_protocol"] = protocol
    callback = Mock()
    with patch.object(prefix_worker, "load_prefix_worker", return_value=callback), \
            patch("diablogym.WorkerWindowEnv", side_effect=lambda **k: DummyEnv(**k)):
        env = training.make_env(**kwargs)
    try:
        received = env.unwrapped.constructor_kwargs
        assert received["worker_time_protocol"] == protocol
        assert received["max_steps"] == 6000 and received["farm_scene_cap"] == 3600
        assert "worker_time_recipe" not in received
    finally:
        env.close()
