"""Earned-suffix identity and isolated prefix RNG, without native initialization."""
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "train"), str(ROOT / "tests")]
import eval_contract as contract
import prefix_worker as prefix

SCOPE = "earned-dive-suffix-v1"
R16 = "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"


def rng_state():
    numpy = np.random.get_state()
    return random.getstate(), (numpy[0], numpy[1].copy(), *numpy[2:]), torch.get_rng_state().clone()


def assert_rng_equal(before):
    after = rng_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and before[1][2:] == after[1][2:]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])


def draw_all():
    return random.random(), float(np.random.random()), float(torch.rand(()))


class SampledModel:
    """A regular predict/seed interface which deliberately consumes all RNGs."""
    def __init__(self):
        self.observation_space = SimpleNamespace(shape=(13012,))
        self.action_space = SimpleNamespace(n=15)
        self.policy = SimpleNamespace(set_training_mode=Mock(), requires_grad_=Mock())
        self.draws, self.calls = [], []
        self.fail_predict = False

    def predict(self, obs, *, action_masks, deterministic):
        assert deterministic is False
        values = draw_all()
        self.draws.append(values)
        self.calls.append((obs, action_masks))
        if self.fail_predict:
            raise RuntimeError("predict failure after RNG consumption")
        allowed = np.flatnonzero(action_masks)
        return np.asarray(allowed[int(values[2] * len(allowed))]), None

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


@pytest.fixture
def loaded_prefix(tmp_path):
    # Model loading is mocked; byte-SHA rejection is tested separately below.
    # The fake digest permits a tiny pure fixture instead of requiring a ZIP.
    path = tmp_path / "parent.zip"
    path.write_bytes(b"pure-prefix-fixture")
    model = SampledModel()
    def load(payload):
        assert payload == b"pure-prefix-fixture"
        draw_all()
        return model
    before = rng_state()
    with patch.object(prefix.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: R16)), \
            patch.object(prefix, "_load_model", side_effect=load):
        callback = prefix.load_prefix_worker(path, expected_sha256=R16)
    assert_rng_equal(before)
    return callback, model


def test_loader_contract_and_freeze_do_not_consume_learner_rng(loaded_prefix):
    callback, model = loaded_prefix
    assert callback.source_sha256 == R16 == contract.WORKER_PREFIX_R16_SHA256
    assert callback.diablogym_worker_observation_view == "dual-v4-asymmetric-v3"
    assert callback.diablogym_worker_action12_mode == "environment-mask"
    assert callback.on_beat is None
    model.policy.set_training_mode.assert_called_once_with(False)
    model.policy.requires_grad_.assert_called_once_with(False)


def test_reseed_predict_private_sequence_persists_and_exactly_replays(loaded_prefix):
    callback, model = loaded_prefix
    obs = np.arange(13012, dtype=np.float32)
    mask = np.zeros(15, dtype=bool); mask[[0, 9, 11, 14]] = True
    before_obs, before_mask = obs.copy(), mask.copy()
    before = rng_state()
    callback.episode_reseed(2164000)
    assert_rng_equal(before)
    actions = []
    for _ in range(5):
        actions.append(callback(obs, mask))
        assert_rng_equal(before)
    draws = model.draws.copy()
    assert len(set(draws)) == 5
    # Perturb learner's streams. Reseeding still reproduces the private prefix,
    # while every callback must preserve these *new* caller states.
    draw_all(); current = rng_state()
    callback.episode_reseed(2164000)
    replay = [callback(obs, mask) for _ in range(5)]
    assert actions == replay and model.draws[5:] == draws
    assert_rng_equal(current)
    np.testing.assert_array_equal(obs, before_obs)
    np.testing.assert_array_equal(mask, before_mask)
    assert all(a is obs and b is mask for a, b in model.calls)
    assert all(mask[action] for action in actions)


def test_two_private_workers_interleave_without_consuming_each_others_streams(loaded_prefix):
    first, first_model = loaded_prefix
    second_model = SampledModel()
    with patch.object(prefix.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: R16)), \
            patch.object(prefix.Path, "read_bytes", return_value=b"second"), \
            patch.object(prefix, "_load_model", return_value=second_model):
        second = prefix.load_prefix_worker("second.zip", expected_sha256=R16)
    obs = np.zeros(13012, np.float32); mask = np.ones(15, bool)
    first.episode_reseed(19); second.episode_reseed(23)
    before = rng_state()
    for _ in range(3):
        first(obs, mask); second(obs, mask)
    first_draws, second_draws = first_model.draws.copy(), second_model.draws.copy()
    first.episode_reseed(19); second.episode_reseed(23)
    for _ in range(3): first(obs, mask)
    for _ in range(3): second(obs, mask)
    assert first_model.draws[3:] == first_draws and second_model.draws[3:] == second_draws
    assert_rng_equal(before)


def test_predict_exception_restores_all_caller_rng(loaded_prefix):
    callback, model = loaded_prefix
    callback.episode_reseed(321)
    before = rng_state(); model.fail_predict = True
    with pytest.raises(RuntimeError, match="predict failure"):
        callback(np.zeros(13012, np.float32), np.ones(15, bool))
    assert_rng_equal(before)


def test_load_exception_after_random_draws_restores_all_caller_rng(tmp_path):
    path = tmp_path / "parent.zip"; path.write_bytes(b"bad archive")
    def broken(payload):
        draw_all()
        raise RuntimeError("load failed")
    before = rng_state()
    with patch.object(prefix.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: R16)), \
            patch.object(prefix, "_load_model", side_effect=broken):
        with pytest.raises(RuntimeError, match="load failed"):
            prefix.load_prefix_worker(path, expected_sha256=R16)
    assert_rng_equal(before)


@pytest.mark.parametrize("expected", [None, "0" * 64, R16.upper()])
def test_unregistered_parent_identity_rejected_before_read_or_load(expected):
    with patch.object(prefix.Path, "read_bytes") as read, patch.object(prefix, "_load_model") as load:
        with pytest.raises(ValueError):
            prefix.load_prefix_worker("absent.zip", expected_sha256=expected)
    read.assert_not_called(); load.assert_not_called()


def test_changed_parent_bytes_rejected_before_model_load(tmp_path):
    path = tmp_path / "wrong.zip"; path.write_bytes(b"not the registered R16")
    with patch.object(prefix, "_load_model") as load:
        with pytest.raises(ValueError, match="bytes differ"):
            prefix.load_prefix_worker(path, expected_sha256=R16)
    load.assert_not_called()


@pytest.mark.parametrize("shape,actions", [((298,), 15), ((13012,), 16)])
def test_bad_loaded_io_contract_rejected_without_rng_leak(tmp_path, shape, actions):
    path = tmp_path / "parent.zip"; path.write_bytes(b"fixture")
    model = SampledModel(); model.observation_space.shape = shape; model.action_space.n = actions
    before = rng_state()
    with patch.object(prefix.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: R16)), \
            patch.object(prefix, "_load_model", return_value=model):
        with pytest.raises(ValueError, match="13012"):
            prefix.load_prefix_worker(path, expected_sha256=R16)
    assert_rng_equal(before)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "7"])
def test_seed_contract_and_forbidden_callback_hook_fail_without_prediction(loaded_prefix, seed):
    callback, model = loaded_prefix
    before = rng_state()
    with pytest.raises(ValueError): callback.episode_reseed(seed)
    callback.on_beat = lambda: None
    with pytest.raises(RuntimeError, match="on_beat"):
        callback(np.zeros(13012, np.float32), np.ones(15, bool))
    assert model.calls == []
    assert_rng_equal(before)


def test_recipe_exact_scope_budget_and_handoff_contract():
    recipe = contract.worker_prefix_recipe(SCOPE, model_sha256=R16, max_attempts=7, max_microsteps=21000)
    assert recipe == {
        "version": "earned-dive-suffix-prefix-v1", "source_sha256": R16,
        "observation_view": "dual-v4-asymmetric-v3", "action12_mode": "environment-mask",
        "sampling": "original-sampled-predict", "rng_isolation": "persistent-private-python-numpy-torch-cpu-v1",
        "episode_reseed": "actual-episode-seed", "max_attempts": 7, "max_microsteps": 21000,
        "budget_scope": "per-env-lifetime-across-resets",
        "handoff": "first-main-l1-live-dive-ready-idle-after-normal-drain",
        "prefix_training": "excluded", "on_beat": None,
        "physical_clock_start": "normal-reset-return-main-l1-formal-beat0",
        "initial_town_navigation": "excluded-from-formal-microsteps-inherited-reset-boundary",
        "wall_clock_scope": "complete-reset-and-prefix",
    }


@pytest.mark.parametrize("scope", ["farm-only", "farm-dive-v1"])
def test_old_scope_omits_recipe_and_rejects_any_prefix_parameters(scope):
    assert contract.worker_prefix_recipe(scope) is None
    for extras in ({"model_sha256": R16}, {"max_attempts": 1}, {"max_microsteps": 1}):
        with pytest.raises(ValueError): contract.worker_prefix_recipe(scope, **extras)


@pytest.mark.parametrize("field,value", [("model_sha256", None), ("model_sha256", "0" * 64),
    ("max_attempts", None), ("max_attempts", 0), ("max_attempts", True),
    ("max_microsteps", None), ("max_microsteps", 0), ("max_microsteps", 1.5)])
def test_recipe_rejects_missing_or_ambiguous_lifetime_budget(field, value):
    kwargs = dict(model_sha256=R16, max_attempts=2, max_microsteps=6000); kwargs[field] = value
    with pytest.raises(ValueError): contract.worker_prefix_recipe(SCOPE, **kwargs)


import hashlib
import io
import json
import zipfile
import gymnasium as gym
import train_ppo as training
import migrate_resource_candidate as migration
from test_resource_sustain_contract import DummyEnv


def recipe(**overrides):
    args = dict(model_sha256=R16, max_attempts=3, max_microsteps=18000)
    args.update(overrides)
    return contract.worker_prefix_recipe(SCOPE, **args)


@pytest.mark.parametrize("mutation", ["old_to_new", "new_to_old", "attempts", "microsteps", "forged", "missing", "legacy"])
def test_ordinary_resume_never_bypasses_prefix_scope_or_budget(mutation):
    old = {"worker_learning_window_scope": "farm-dive-v1"}
    earned = {"worker_learning_window_scope": SCOPE, "worker_prefix": recipe()}
    previous, current = deepcopy(earned), deepcopy(earned)
    if mutation == "old_to_new": previous = old
    elif mutation == "new_to_old": current = old
    elif mutation == "attempts": current["worker_prefix"] = recipe(max_attempts=4)
    elif mutation == "microsteps": current["worker_prefix"] = recipe(max_microsteps=18001)
    elif mutation == "forged": current["worker_prefix"]["prefix_training"] = "included"
    elif mutation == "missing": current.pop("worker_prefix")
    else: previous = None
    with pytest.raises(ValueError):
        training._validate_resume_contract(previous, current,
            allow_manager_change=True, allow_legacy_resume=True, allow_optimizer_reset=True,
            allow_target_kl_change=True, allow_environment_restart=True)
    training._validate_worker_prefix_resume_identity(earned, deepcopy(earned))


def test_old_identity_never_opens_prefix_file_and_rejects_stray_flag():
    with patch.object(training, "_capture_file_sha256") as capture:
        assert training._worker_prefix_identity(SimpleNamespace(worker_learning_window_scope="farm-dive-v1")) is None
        with pytest.raises(ValueError):
            training._worker_prefix_identity(SimpleNamespace(worker_learning_window_scope="farm-dive-v1",
                                                             worker_prefix_model="must-not-open.zip"))
    capture.assert_not_called()


@pytest.mark.parametrize("scope", ["farm-only", "farm-dive-v1", SCOPE])
def test_environment_factory_loads_prefix_once_only_for_explicit_scope(scope):
    callback = Mock()
    args = dict(worker=True, manager_heuristic="readiness-v1", max_steps=6000,
        worker_learning_window_scope=scope, resource_protocol="l2-town-v1",
        resource_service_policy="sustain-v6", dive_blocker_recovery="adjacent-v1",
        worker_policy_observation_view="dual-v4-asymmetric-v3")
    if scope == SCOPE:
        args.update(worker_prefix_model="fixed-parent.zip", worker_prefix_sha256=R16,
                    worker_prefix_max_attempts=3, worker_prefix_max_microsteps=18000)
    with patch.object(prefix, "load_prefix_worker", return_value=callback) as load, \
            patch("diablogym.WorkerWindowEnv", side_effect=lambda **kw: DummyEnv(**kw)):
        env = training.make_env(**args)
    try:
        kwargs = env.unwrapped.constructor_kwargs
        if scope == SCOPE:
            load.assert_called_once_with("fixed-parent.zip", expected_sha256=R16)
            assert kwargs["prefix_worker"] is callback
            assert kwargs["prefix_max_attempts"] == 3 and kwargs["prefix_max_microsteps"] == 18000
        else:
            load.assert_not_called()
            assert not any(k.startswith("prefix_") for k in kwargs)
    finally: env.close()


def test_training_contract_omits_prefix_for_legacy_but_binds_new_recipe():
    args = SimpleNamespace(worker=True, options=False, flat_clock=False, arch="mlp", max_steps=6000,
        num_envs=4, n_steps=512, gamma=0.99, lr=3e-4, ent_coef=0.02, skip_dry=False,
        no_drink_sovereignty=False, dry_curriculum_schedule=None, bc_aux_lambda=0.0,
        bc_aux_demos=None, bc_aux_liveness_preflight=False, distill_beta=0.0,
        calib_record_only=False, worker_learning_window_scope="farm-dive-v1")
    model = SimpleNamespace(max_grad_norm=0.5, action_space=gym.spaces.Discrete(15),
        observation_space=gym.spaces.Box(-1, 1, (13012,)), device="cpu")
    old = training._training_contract(args, model, batch_size=256)
    assert "worker_prefix" not in old
    args.worker_learning_window_scope = SCOPE
    args.worker_prefix_model = "fixed.zip"
    args.worker_prefix_max_attempts = 3; args.worker_prefix_max_microsteps = 18000
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        new = training._training_contract(args, model, batch_size=256)
    assert new["worker_prefix"] == recipe()
    assert new["worker_learning_window_scope"] == SCOPE
    assert {key for key in set(new) | set(old) if old.get(key) != new.get(key)} == {
        "worker_learning_window_scope", "worker_prefix"}


@pytest.fixture
def registered_parent_data():
    path = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
    if not path.is_file(): pytest.skip("registered R16 metadata fixture unavailable")
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == R16
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return json.loads(archive.read("data"))


def test_migration_has_exact_new_operation_and_only_two_additional_authorized_keys(registered_parent_data):
    source = registered_parent_data["diablogym_contract"]
    before = deepcopy(source); implementation = "a" * 64
    old = migration.target_contract(source, implementation, "sustain-v6", "adjacent-v1")
    new = migration.target_contract(source, implementation, "sustain-v6", "adjacent-v1", SCOPE, recipe())
    assert source == before
    assert {k for k in set(new) | set(old) if new.get(k) != old.get(k)} == {
        "worker_learning_window_scope", "worker_prefix"}
    migration.validate_target_contract(source, old, implementation)
    migration.validate_target_contract(source, new, implementation)
    assert migration.operation_for("sustain-v6", "adjacent-v1", SCOPE) == \
        "r16-to-sustain-v6-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v1"
    assert migration.operation_for("sustain-v6", "adjacent-v1") == \
        "r16-to-sustain-v6-dive-adjacent-v1-weights-only-v1"
    for field, value in [("gamma", 0.1), ("action_n", 16), ("max_steps", 10000)]:
        with pytest.raises(ValueError):
            migration.validate_target_contract(source, {**new, field: value}, implementation)
    for policy, recovery in [("sustain-v5", "adjacent-v1"), ("sustain-v6", "off")]:
        with pytest.raises(ValueError):
            migration.target_contract(source, implementation, policy, recovery, SCOPE, recipe())
    forged = recipe(); forged["sampling"] = "deterministic"
    with pytest.raises(ValueError):
        migration.target_contract(source, implementation, "sustain-v6", "adjacent-v1", SCOPE, forged)
    with pytest.raises(ValueError):
        migration.target_contract(source, implementation, "sustain-v6", "adjacent-v1", None, recipe())



def prefix_cli_args(source):
    fixed_parent_keys = ["worker_policy_observation_view", "reward_economy", "farm_scene_cap",
        "worker_action14_logit_bonus", "worker_dive_action11_logit_bonus", "worker_potion_action13_logit_bonus",
        "worker_hp_loss_price", "worker_potion_pickup_bonus", "worker_no_progress_timeout_credit",
        "worker_descend_escrow_fraction", "worker_descend_escrow_power", "worker_descend_escrow_readiness_gate",
        "reset_layer_clock_on_window"]
    args = {key: source[key] for key in fixed_parent_keys}
    args.update(worker=True, algo="mppo", device="cpu", seed=2164000, max_steps=6000,
        worker_learning_window_scope=SCOPE, worker_prefix_model="registered-parent.zip",
        worker_prefix_max_attempts=3, worker_prefix_max_microsteps=18000,
        resource_protocol="l2-town-v1", resource_purchase_mode="full", resource_service_policy="sustain-v6",
        dive_blocker_recovery="adjacent-v1", drink_sovereignty=True,
        distill_beta=0.0, bc_aux_lambda=0.0, bc_aux_demos=None, bc_aux_liveness_preflight=False)
    return SimpleNamespace(**args)


def test_cli_valid_prefix_preserves_real_parent_fixed_algorithm_and_clock(registered_parent_data):
    args = prefix_cli_args(registered_parent_data["diablogym_contract"])
    assert args.reset_layer_clock_on_window is True
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        training._validate_worker_prefix_args(args)


@pytest.mark.parametrize("key,value", [("device", "cuda"), ("worker", False), ("seed", None),
    ("max_steps", 10000), ("reset_layer_clock_on_window", False), ("drink_sovereignty", False),
    ("farm_scene_cap", 1800), ("worker_action14_logit_bonus", 0.0), ("distill_beta", 1.0),
    ("worker_hp_loss_price", 0.0), ("deep", True), ("deep_start_curriculum", {"preset": 2}),
    ("teacher_override", True), ("skip_dry", 1.0), ("resource_service_policy", "sustain-v5")])
def test_cli_rejects_unregistered_algorithm_or_alternate_start_before_environment(registered_parent_data, key, value):
    args = prefix_cli_args(registered_parent_data["diablogym_contract"])
    setattr(args, key, value)
    with patch.object(training, "_capture_file_sha256", return_value=R16):
        with pytest.raises(ValueError): training._validate_worker_prefix_args(args)
