"""JSON metadata and synthetic reset/save boundaries; no actual model load or game."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import migrate_completion_candidate as migration
import migrate_resource_candidate as legacy

IMPL = "a" * 64
POLICY = "b" * 64
DEFAULT_PARENT = Path("/home/laure/r20_sustain_20260904/candidate-earned-v6-terminal/train/runs/"
                      "r20-earned-depth24-terminal-four-rollout-s2168000/model_candidate.zip")


@pytest.fixture(scope="module")
def parent_data():
    # Only the JSON data member is decoded. No torch/SB3 checkpoint loader runs.
    path = Path(os.environ.get("DIABLOGYM_COMPLETION_PARENT_FIXTURE", str(DEFAULT_PARENT)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == migration.PARENT_SHA256
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    assert legacy.json_sha256(data["diablogym_contract"]) == migration.PARENT_CONTRACT_SHA256
    return data


def pair(data):
    target = migration.target_contract(data["diablogym_contract"], IMPL)
    return target, migration.make_receipt(data, target, POLICY)


def synthetic_model(data):
    model = SimpleNamespace(**deepcopy(data))
    model.n_steps, model.n_envs = 512, 4
    model.observation_space = SimpleNamespace(shape=(13012,))
    model.action_space = SimpleNamespace(n=15)
    model.gradient_clip_mode = "separate-root-context-critic-v2"
    model.policy = SimpleNamespace(optimizer=SimpleNamespace(state={"old_moment": 3}),
        mlp_extractor=SimpleNamespace(actor_context_enabled=True))
    model.rollout_buffer = SimpleNamespace(pos=512, full=True)
    def reset_buffer():
        model.rollout_buffer.pos, model.rollout_buffer.full = 0, False
    model.rollout_buffer.reset = Mock(side_effect=reset_buffer)
    model.set_random_seed = Mock()
    model._assert_critic_migration_contract = Mock()
    return model


def optimizer_reset(model, learning_rate):
    assert learning_rate == 0.0001
    model.policy.optimizer.state = {}


def test_exact_parent_and_every_original_rollout_validated_without_mutation(parent_data):
    from leashed_ppo import validate_worker_onpolicy_pg_receipt
    before, rng = deepcopy(parent_data), random.getstate()
    with patch("leashed_ppo.validate_worker_onpolicy_pg_receipt",
               wraps=validate_worker_onpolicy_pg_receipt) as check:
        migration.validate_parent_metadata(parent_data)
    assert check.call_count == 4
    for call in check.call_args_list:
        assert call.kwargs == {"expected_samples": 2048}
    assert [r["qualifies"] for r in parent_data["_worker_onpolicy_pg_rollout_receipts"]] == [
        True, False, True, True]
    assert parent_data == before and random.getstate() == rng


def test_target_only_adds_time_contract_and_changes_implementation(parent_data):
    before = deepcopy(parent_data)
    target, receipt = pair(parent_data)
    source = parent_data["diablogym_contract"]
    assert {k for k in target if target[k] != source.get(k)} == migration.ALLOWED_CONTRACT_KEYS
    assert target["worker_time_protocol"] == "completion-l2-v1"
    assert target["worker_time_recipe"] == migration.time_recipe()
    assert target["max_steps"] == 6000
    assert target["worker_depth_shaping_unit"] == 24.0
    assert target["worker_prefix"] == source["worker_prefix"]
    assert target["resource_service_recipe"] == source["resource_service_recipe"]
    assert target["worker_descend_escrow_fraction"] == 0.5
    assert target["worker_descend_escrow_power"] == 1.6
    legacy.validate_inherited_receipt(receipt, target)
    assert receipt["parent_checkpoint_sha256"] == migration.PARENT_SHA256
    assert receipt["parent_num_timesteps"] == 8192
    assert receipt["inherited_resource_warm_start_receipt"] == parent_data["_resource_warm_start_receipt"]
    assert receipt["parent_completion"]["rollout_receipts"] == parent_data["_worker_onpolicy_pg_rollout_receipts"]
    assert receipt["optimizer_state"] == "reset-empty"
    assert receipt["exact_trajectory_continuation"] is False
    assert receipt["publication_status"] == "INITIALIZATION_ONLY_NOT_TRAINED"
    assert parent_data == before
    receipt["parent_completion"]["rollout_receipts"][0]["qualifies"] = False
    target["worker_time_recipe"]["arrival_microsteps"] = 1
    assert migration.time_recipe()["arrival_microsteps"] == 12000
    assert parent_data == before


@pytest.mark.parametrize("key", list(migration.PARENT_COUNTERS) + list(migration.PARENT_WARMUP))
def test_wrong_or_missing_parent_counter_rejected(parent_data, key):
    data = deepcopy(parent_data)
    del data[key]
    with pytest.raises(ValueError, match="counter/state"):
        migration.validate_parent_metadata(data)


@pytest.mark.parametrize("key,value", [
    ("num_timesteps", 2048), ("_n_updates", 10),
    ("_ppo_optimizer_steps_completed", 320), ("_actor_optimizer_steps_completed", 0),
    ("_last_completed_ppo_rollout_steps", 6144), ("_worker_onpolicy_pg_joint_rollouts", 3),
    ("_worker_onpolicy_pg_qualifying_rollouts", 4), ("_critic_warmup_completed", 0),
    ("_worker_onpolicy_pg_pending_receipts", [{"pending": True}]),
    ("_bc_aux_pending_action_receipts", [1]),
])
def test_partial_unupdated_or_forged_parent_is_rejected(parent_data, key, value):
    data = deepcopy(parent_data); data[key] = value
    with pytest.raises(ValueError):
        migration.validate_parent_metadata(data)


@pytest.mark.parametrize("field", [
    "_resource_warm_start_receipt", "_worker_onpolicy_pg_rollout_receipts", "_resume_lineage",
])
def test_historical_evidence_cannot_be_replaced_or_rehashed(parent_data, field):
    data = deepcopy(parent_data)
    if isinstance(data[field], list):
        data[field][0]["optimizer_steps"] = 81
    else:
        data[field]["parent_checkpoint_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        migration.validate_parent_metadata(data)


def test_receipt_validator_false_is_not_waived_by_fixed_hash(parent_data):
    with patch("leashed_ppo.validate_worker_onpolicy_pg_receipt", return_value=False):
        with pytest.raises(ValueError, match="rollout receipt"):
            migration.validate_parent_metadata(parent_data)


@pytest.mark.parametrize("key,value", [
    ("max_steps", 12000), ("worker_depth_shaping_unit", 48.0), ("gamma", 0.99),
    ("worker_descend_escrow_fraction", 0), ("worker_descend_escrow_power", 1),
    ("worker_descend_escrow_readiness_gate", False), ("farm_scene_cap", 7200),
    ("resource_service_policy", "diagnostic-protection-v1"), ("action_n", 16),
    ("worker_learning_window_scope", "farm-dive-v1"), ("extra_null_field", None),
])
def test_target_drift_rejected_even_when_receipt_hash_resealed(parent_data, key, value):
    target, receipt = pair(parent_data)
    target[key] = value
    receipt["target_contract_sha256"] = migration.json_sha256(target)
    with pytest.raises(ValueError, match="only implementation"):
        legacy.validate_inherited_receipt(receipt, target)


@pytest.mark.parametrize("key", list(migration.time_recipe()))
def test_every_recipe_field_bound_and_cannot_reseal_receipt(parent_data, key):
    target, receipt = pair(parent_data)
    target["worker_time_recipe"][key] = "changed"
    receipt["worker_time_recipe"] = deepcopy(target["worker_time_recipe"])
    receipt["target_contract_sha256"] = migration.json_sha256(target)
    with pytest.raises(ValueError):
        legacy.validate_inherited_receipt(receipt, target)


@pytest.mark.parametrize("key,value", [
    ("parent_checkpoint_sha256", legacy.PARENT_SHA256), ("parent_num_timesteps", legacy.PARENT_STEPS),
    ("parent_contract_sha256", legacy.PARENT_CONTRACT_SHA256),
    ("inherited_resource_warm_start_receipt_sha256", "f" * 64),
    ("current_world_training_steps", 8192), ("exact_trajectory_continuation", True),
    ("historical_warmup_is_current_training", True), ("optimizer_state", "preserved"),
    ("publication_status", "TRAINED"), ("policy_sha256", "invalid"),
    ("worker_time_protocol", "legacy"),
])
def test_lineage_and_zero_world_claims_are_exact(parent_data, key, value):
    target, receipt = pair(parent_data)
    receipt[key] = value
    with pytest.raises(ValueError):
        legacy.validate_inherited_receipt(receipt, target)


def test_old_operation_and_source_still_exact_and_cannot_bypass_new_migration(parent_data):
    inherited = deepcopy(parent_data["_resource_warm_start_receipt"])
    legacy.validate_inherited_receipt(inherited, parent_data["diablogym_contract"])
    assert legacy.PARENT_SHA256 == "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"
    assert legacy.PARENT_STEPS == 4089856
    target, receipt = pair(parent_data)
    for wrong in (inherited, {**receipt, "operation": "unregistered-completion-v2"}):
        with pytest.raises(ValueError):
            legacy.validate_inherited_receipt(wrong, target)
    with pytest.raises(ValueError, match="8192-step parent"):
        migration.target_contract(inherited["source_contract"], IMPL)
    with pytest.raises(ValueError, match="registered R16"):
        legacy.target_contract(parent_data["diablogym_contract"], IMPL)


def test_reset_reuses_original_empty_adam_zero_counters_and_correct_immediate_parent(parent_data):
    target, receipt = pair(parent_data)
    model = synthetic_model(parent_data)
    reset = Mock(side_effect=optimizer_reset)
    with patch.dict(sys.modules, {"train_ppo": SimpleNamespace(_reset_policy_optimizer=reset)}), \
            patch.object(legacy, "policy_sha256", return_value=POLICY):
        legacy.reset_to_initialization(model, receipt, target, 2168000)
        legacy.validate_zero_state(model)
    reset.assert_called_once_with(model, 0.0001)
    assert all(getattr(model, key) == 0 for key in legacy.ZERO_COUNTERS)
    assert all(getattr(model, key) == [] for key in legacy.EMPTY_LISTS)
    assert not model.policy.optimizer.state
    assert model._last_completed_ppo_rollout_steps is None
    assert model._last_obs is None and model._last_episode_starts is None
    assert model._resume_lineage["generation"] == 4
    assert model._resume_lineage["immediate_parent_sha256"] == migration.PARENT_SHA256
    assert model._resume_lineage["immediate_parent_num_timesteps"] == 8192
    assert model._resume_lineage["optimizer_state"] == "reset"
    assert model._resource_warm_start_receipt == receipt
    assert model._resource_warm_start_receipt is not receipt
    model.rollout_buffer.reset.assert_called_once()
    model.set_random_seed.assert_called_once_with(2168000)
    model._assert_critic_migration_contract.assert_called_once()


def test_bad_lineage_cannot_partially_reset_model(parent_data):
    target, receipt = pair(parent_data)
    receipt["parent_num_timesteps"] = 2048
    model = synthetic_model(parent_data)
    reset = Mock()
    with patch.dict(sys.modules, {"train_ppo": SimpleNamespace(_reset_policy_optimizer=reset)}):
        with pytest.raises(ValueError):
            legacy.reset_to_initialization(model, receipt, target, 2168000)
    reset.assert_not_called()
    model.rollout_buffer.reset.assert_not_called()
    model.set_random_seed.assert_not_called()
    assert model.num_timesteps == 8192 and model.policy.optimizer.state == {"old_moment": 3}


@pytest.mark.parametrize("change", [None, "parent", "steps", "pending", "recipe", "receipt", "impl"])
def test_existing_manifest_capture_accepts_only_exact_new_lineage(parent_data, tmp_path, change):
    target, receipt = pair(parent_data)
    payload = b"synthetic-validator-boundary-not-a-model"
    data = dict(diablogym_contract=target, _resource_warm_start_receipt=receipt,
        **{key: 0 for key in legacy.ZERO_COUNTERS}, _last_completed_ppo_rollout_steps=None)
    manifest = dict(schema=legacy.SCHEMA, status="INITIALIZATION_ONLY_NOT_TRAINED",
        operation=migration.OPERATION, parent_sha256=migration.PARENT_SHA256,
        model_file="model_warm_start.zip", model_sha256=hashlib.sha256(payload).hexdigest(),
        ordinary_resume_eligible=False, trained_in_target_world=False, policy_sha256=POLICY,
        source_contract=receipt["source_contract"], target_contract=target, receipt=receipt,
        dive_blocker_recovery_recipe=receipt["dive_blocker_recovery_recipe"],
        depth_shaping_recipe=receipt["depth_shaping_recipe"])
    if change == "parent": manifest["parent_sha256"] = legacy.PARENT_SHA256
    elif change == "steps": data["num_timesteps"] = 8192
    elif change == "pending": data["_last_completed_ppo_rollout_steps"] = 2048
    elif change == "recipe": target["worker_time_recipe"]["followup_microsteps"] = 1
    elif change == "receipt": manifest["receipt"] = {**receipt, "parent_num_timesteps": 1}
    (tmp_path / "model_warm_start.zip").write_bytes(payload)
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    validator = Mock(return_value=data)
    with patch.dict(sys.modules, {"train_ppo": SimpleNamespace(_validate_checkpoint_bytes=validator)}):
        if change is None:
            result = legacy.capture_initialization(path, IMPL)
            assert result[0] == payload and result[2] == manifest
        else:
            with pytest.raises(ValueError):
                legacy.capture_initialization(path, "c" * 64 if change == "impl" else IMPL)
    validator.assert_called_once_with(payload, str(tmp_path / "model_warm_start.zip"), require_leashed=True)


def test_cli_is_explicit_without_general_recipe_or_resume_overrides():
    argv = ["completion-migration", "--parent", "parent.zip", "--parent-sha256", migration.PARENT_SHA256,
        "--implementation-sha256", IMPL, "--output-dir", "unused", "--seed", "2168000"]
    result = dict(status="INITIALIZATION_ONLY_NOT_TRAINED", model_sha256="c" * 64, policy_sha256=POLICY)
    with patch.object(sys, "argv", argv), patch.object(migration, "migrate", return_value=result) as call:
        migration.main()
    call.assert_called_once_with("parent.zip", parent_sha256=migration.PARENT_SHA256,
        output_dir="unused", implementation_sha256=IMPL, seed=2168000)
    with patch.object(sys, "argv", argv + ["--max-steps", "24000"]), patch.object(migration, "migrate") as call:
        with pytest.raises(SystemExit): migration.main()
    call.assert_not_called()


@pytest.mark.parametrize("mode", ["sha", "seed", "existing", "bytes"])
def test_wrong_input_fails_before_any_model_load_or_output(parent_data, tmp_path, mode):
    path = tmp_path / "parent.zip"; path.write_bytes(b"not-the-registered-parent")
    destination = tmp_path / "output"
    if mode == "existing": destination.mkdir()
    fake_loader = Mock()
    with patch("leashed_ppo.LeashedMaskablePPO.load", fake_loader):
        with pytest.raises(ValueError):
            migration.migrate(path, parent_sha256="d" * 64 if mode == "sha" else migration.PARENT_SHA256,
                output_dir=destination, implementation_sha256=IMPL,
                seed=True if mode == "seed" else 2168000)
    fake_loader.assert_not_called()
    assert destination.exists() is (mode == "existing")


def test_success_orchestration_reuses_save_reload_probe_and_zero_checks_with_synthetic_tensors(
        parent_data, tmp_path):
    import torch
    source = Path(os.environ.get("DIABLOGYM_COMPLETION_PARENT_FIXTURE", str(DEFAULT_PARENT)))
    source_before = hashlib.sha256(source.read_bytes()).hexdigest()
    model = synthetic_model(parent_data)
    tensors = {"synthetic.weight": torch.tensor([1.0, 2.0])}
    model.policy.state_dict = lambda: tensors
    saved = {}
    payload = b"synthetic-save-boundary-no-SB3-model"
    def save(current, path):
        assert current is model and not model.policy.optimizer.state
        assert all(getattr(model, key) == 0 for key in legacy.ZERO_COUNTERS)
        saved.update({key: getattr(model, key) for key in legacy.ZERO_COUNTERS})
        saved.update(diablogym_contract=deepcopy(model.diablogym_contract),
            _resource_warm_start_receipt=deepcopy(model._resource_warm_start_receipt),
            _last_completed_ppo_rollout_steps=None)
        path.write_bytes(payload)
        return path
    def checkpoint_validator(data, label, require_leashed):
        assert require_leashed is True
        if label == str(source.resolve()):
            assert hashlib.sha256(data).hexdigest() == migration.PARENT_SHA256
            return deepcopy(parent_data)
        assert data == payload
        return saved
    reset, save_call = Mock(side_effect=optimizer_reset), Mock(side_effect=save)
    validator = Mock(side_effect=checkpoint_validator)
    boundary = Mock(side_effect=lambda data: data)
    implementation = Mock(return_value=IMPL)
    probe = {"schema": "synthetic-probe-only", "logits_sha256": "c" * 64,
             "values_sha256": "d" * 64}
    original_threads = torch.get_num_threads()
    try:
        with patch.dict(sys.modules, {"train_ppo": SimpleNamespace(
                _reset_policy_optimizer=reset, _atomic_save_model=save_call,
                _implementation_bundle_sha256=implementation,
                _validate_checkpoint_bytes=validator,
                _validate_resumable_leashed_boundary=boundary)}), \
                patch("leashed_ppo.LeashedMaskablePPO.load", return_value=model) as load, \
                patch.object(legacy, "policy_probe", return_value=probe) as policy_probe:
            result = migration.migrate(source, parent_sha256=migration.PARENT_SHA256,
                output_dir=tmp_path / "new", implementation_sha256=IMPL, seed=2168000)
        assert load.call_count == 2  # Synthetic parent load plus original reload helper.
        assert all(call.kwargs.get("env") is None for call in load.call_args_list)
        assert policy_probe.call_count == 3  # Before/reset/reload, never real forward.
        assert save_call.call_count == 1 and reset.call_count == 1
        assert boundary.call_count == 1 and validator.call_count == 2
        assert implementation.call_count == 2
        assert result["status"] == "INITIALIZATION_ONLY_NOT_TRAINED"
        assert result["parent_sha256"] == migration.PARENT_SHA256
        assert result["model_sha256"] == hashlib.sha256(payload).hexdigest()
        assert result["receipt"]["policy_sha256"] == legacy.policy_sha256(model)
        assert json.loads((tmp_path / "new/manifest.json").read_text()) == result
        assert torch.equal(tensors["synthetic.weight"], torch.tensor([1.0, 2.0]))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
    finally:
        torch.set_num_threads(original_threads)
