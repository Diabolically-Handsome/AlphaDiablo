"""Completion time contract wiring; no models, native resets, or training."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "python"), str(ROOT / "tests")]
import train_ppo as training
import prefix_worker
from diablogym.completion_clock import COMPLETION_L2_V1
from test_resource_sustain_contract import DummyEnv

SCOPE = "earned-dive-suffix-v1"
PROTOCOL = "completion-l2-v1"
R16 = "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"


def valid_args():
    return SimpleNamespace(
        worker=True, options=False, flat_clock=False, arch="mlp", max_steps=6000,
        num_envs=4, n_steps=512, gamma=1.0, lr=1e-4, ent_coef=.005,
        skip_dry=False, no_drink_sovereignty=False, dry_curriculum_schedule=None,
        bc_aux_lambda=0., bc_aux_demos=None, bc_aux_liveness_preflight=False,
        distill_beta=0., calib_record_only=False, worker_learning_window_scope=SCOPE,
        resource_protocol="l2-town-v1", resource_purchase_mode="full",
        resource_service_policy="sustain-v6", farm_scene_cap=3600,
        worker_time_protocol=PROTOCOL, worker_prefix_model="unused-parent.zip",
        worker_prefix_max_attempts=64, worker_prefix_max_microsteps=384000,
        worker_depth_shaping_unit=24., reward_economy="v4")


def contract(args):
    model = SimpleNamespace(max_grad_norm=.5, action_space=gym.spaces.Discrete(15),
        observation_space=gym.spaces.Box(-1, 1, (13012,)), device="cpu")
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        return training._training_contract(args, model, batch_size=256)


def factory_args():
    return dict(worker=True, manager_heuristic="readiness-v3", max_steps=6000,
        worker_learning_window_scope=SCOPE, resource_protocol="l2-town-v1",
        resource_purchase_mode="full", resource_service_policy="sustain-v6",
        dive_blocker_recovery="adjacent-v1", farm_scene_cap=3600,
        worker_policy_observation_view="dual-v4-asymmetric-v3",
        worker_prefix_model="unused-parent.zip", worker_prefix_sha256=R16,
        worker_prefix_max_attempts=64, worker_prefix_max_microsteps=384000)


def test_legacy_identity_is_absent_and_does_not_import_clock():
    original = __import__
    def guarded(name, *a, **kw):
        assert name != "diablogym.completion_clock"
        return original(name, *a, **kw)
    with patch("builtins.__import__", side_effect=guarded):
        assert training._worker_time_identity() is None
        training._validate_worker_time_args(SimpleNamespace())


def test_recipe_is_complete_detached_and_binds_original_observation_clock():
    expected = dict(protocol=PROTOCOL, actor_denominator=6000,
        arrival_microsteps=12000, followup_microsteps=1800,
        collect_command_window_microsteps=900, service_microsteps=3000,
        farm_microsteps=3600)
    recipe = training._worker_time_identity(PROTOCOL)
    assert recipe == expected == COMPLETION_L2_V1.as_dict()
    recipe["arrival_microsteps"] = 1
    assert training._worker_time_identity(PROTOCOL) == expected


@pytest.mark.parametrize("value", [None, False, 1, "off", "wide", "completion-l2-v2"])
def test_unknown_protocol_is_not_silently_legacy(value):
    with pytest.raises(ValueError):
        training._worker_time_identity(value)


@pytest.mark.parametrize("key,value", [
    ("worker", False), ("worker_learning_window_scope", "farm-dive-v1"),
    ("worker_learning_window_scope", "farm-only"), ("resource_protocol", "off"),
    ("resource_purchase_mode", "armor"), ("resource_service_policy", "sustain-v5"),
    ("max_steps", 12000), ("farm_scene_cap", 7200),
])
def test_completion_config_rejects_alternate_world_before_construction(key, value):
    args = valid_args()
    setattr(args, key, value)
    with pytest.raises(ValueError):
        training._validate_worker_time_args(args)
    kw = factory_args()
    kw.update(worker_time_protocol=PROTOCOL)
    kw[key] = value
    with patch("diablogym.WorkerWindowEnv") as constructor, \
            patch.object(prefix_worker, "load_prefix_worker") as load:
        with pytest.raises(ValueError):
            training.make_env(**kw)
    constructor.assert_not_called()
    load.assert_not_called()


def test_cli_real_argument_node_has_legacy_default_and_two_explicit_choices():
    tree = ast.parse((ROOT / "train/train_ppo.py").read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and node.value.args
             and isinstance(node.value.args[0], ast.Constant)
             and node.value.args[0].value == "--worker-time-protocol"]
    assert len(nodes) == 1
    ap = argparse.ArgumentParser()
    exec(compile(ast.Module(nodes, type_ignores=[]), "<actual-cli-node>", "exec"), {"ap": ap})
    assert ap.parse_args([]).worker_time_protocol == "legacy"
    assert ap.parse_args(["--worker-time-protocol", PROTOCOL]).worker_time_protocol == PROTOCOL
    with pytest.raises(SystemExit):
        ap.parse_args(["--worker-time-protocol", "wide"])


def test_actual_contract_changes_only_two_time_keys_and_legacy_omits_them():
    args = valid_args()
    args.worker_time_protocol = "legacy"
    old = contract(args)
    del args.worker_time_protocol
    assert contract(args) == old
    args.worker_time_protocol = PROTOCOL
    new = contract(args)
    assert {key for key in set(old) | set(new) if old.get(key) != new.get(key)} == {
        "worker_time_protocol", "worker_time_recipe"}
    assert new["worker_time_recipe"] == COMPLETION_L2_V1.as_dict()
    assert new["max_steps"] == 6000 and new["farm_scene_cap"] == 3600
    assert new["gamma"] == 1 and new["algorithm_recipe"]["gae_lambda"] == .95
    assert new["resource_service_recipe"] == old["resource_service_recipe"]
    assert "worker_time_protocol" not in old and "worker_time_recipe" not in old


@pytest.mark.parametrize("mutation", ["legacy_to_new", "new_to_legacy", "missing_parent",
    "missing_recipe", "stray_recipe", "wrong_protocol", "changed_budget", "float_budget", "extra_key"])
def test_full_resume_refuses_time_drift_even_with_all_allowances(mutation):
    current = contract(valid_args())
    saved = deepcopy(current)
    if mutation == "legacy_to_new":
        saved.pop("worker_time_protocol"); saved.pop("worker_time_recipe")
    elif mutation == "new_to_legacy":
        current.pop("worker_time_protocol"); current.pop("worker_time_recipe")
    elif mutation == "missing_parent": saved = None
    elif mutation == "missing_recipe": saved.pop("worker_time_recipe")
    elif mutation == "stray_recipe": saved.pop("worker_time_protocol")
    elif mutation == "wrong_protocol": saved["worker_time_protocol"] = "wide"
    elif mutation == "changed_budget": saved["worker_time_recipe"]["arrival_microsteps"] = 12001
    elif mutation == "float_budget": saved["worker_time_recipe"]["arrival_microsteps"] = 12000.
    else: saved["worker_time_recipe"]["hidden_override"] = True
    with pytest.raises(ValueError):
        training._validate_resume_contract(saved, current, allow_manager_change=True,
            allow_legacy_resume=True, allow_optimizer_reset=True,
            allow_target_kl_change=True, allow_environment_restart=True)


def test_exact_time_resume_and_legacy_missing_null_equivalence_remain_valid():
    current = contract(valid_args())
    training._validate_resume_contract(deepcopy(current), current)
    old = {"worker_learning_window_scope": "farm-dive-v1"}
    training._validate_resume_contract(old, {**old, "worker_time_protocol": None,
                                            "worker_time_recipe": None})


@pytest.mark.parametrize("protocol", ["legacy", PROTOCOL])
def test_factory_passes_one_explicit_protocol_without_overriding_observation_or_caps(protocol):
    kw = factory_args()
    kw["worker_time_protocol"] = protocol
    callback = Mock()
    with patch.object(prefix_worker, "load_prefix_worker", return_value=callback) as load, \
            patch("diablogym.WorkerWindowEnv", side_effect=lambda **k: DummyEnv(**k)):
        env = training.make_env(**kw)
    try:
        received = env.unwrapped.constructor_kwargs
        assert received["max_steps"] == 6000 and received["farm_scene_cap"] == 3600
        assert received["prefix_worker"] is callback
        if protocol == PROTOCOL: assert received["worker_time_protocol"] == PROTOCOL
        else: assert "worker_time_protocol" not in received
        assert "worker_time_recipe" not in received
        load.assert_called_once_with("unused-parent.zip", expected_sha256=R16)
    finally:
        env.close()


def test_main_config_and_spawn_kwargs_retain_both_prefix_and_time_identity():
    tree = ast.parse((ROOT / "train/train_ppo.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_main")
    start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "prefix_env_kwargs" for t in node.targets))
    stop = next(i for i in range(start, len(main.body)) if isinstance(main.body[i], ast.Expr)
                and isinstance(main.body[i].value, ast.Call)
                and isinstance(main.body[i].value.func, ast.Name)
                and main.body[i].value.func.id == "print")
    nodes = main.body[start:stop]
    for protocol in ("legacy", PROTOCOL):
        args = valid_args(); args.worker_time_protocol = protocol; args.dive_blocker_recovery = "off"
        scope = dict(args=args, config={}, worker_prefix_identity={"source_sha256": R16,
                     "max_attempts": 64, "max_microsteps": 384000},
                     _worker_time_identity=training._worker_time_identity)
        exec(compile(ast.Module(nodes, type_ignores=[]), "<actual-config-spawn-nodes>", "exec"), scope)
        kw, config = scope["prefix_env_kwargs"], scope["config"]
        assert kw["worker_prefix_sha256"] == R16 and kw["worker_prefix_max_attempts"] == 64
        if protocol == PROTOCOL:
            assert kw["worker_time_protocol"] == config["worker_time_protocol"] == PROTOCOL
            assert config["worker_time_recipe"] == COMPLETION_L2_V1.as_dict()
        else:
            assert "worker_time_protocol" not in kw and "worker_time_recipe" not in config


def test_clock_implementation_is_in_the_training_identity():
    assert "python/diablogym/completion_clock.py" in training._IMPLEMENTATION_SOURCE_FILES
    assert "python/diablogym/resource_sustain_completion.py" in training._IMPLEMENTATION_SOURCE_FILES
