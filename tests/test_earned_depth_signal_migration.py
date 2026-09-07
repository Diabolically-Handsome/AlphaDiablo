"""Exact depth-24 migration metadata; no model, game, forward or optimizer load."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import eval_contract as contract
import migrate_resource_candidate as migration

SCOPE = "earned-dive-suffix-v1"
IMPL = "a" * 64
PARAMETERS = "b" * 64


@pytest.fixture(scope="module")
def parent_data():
    path = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
    if not path.is_file():
        pytest.skip("registered R16 JSON metadata fixture unavailable")
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == migration.PARENT_SHA256
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))


def prefix():
    return contract.worker_prefix_recipe(SCOPE, migration.PARENT_SHA256, 64, 384000)


def target(data, unit=24.0):
    return migration.target_contract(data["diablogym_contract"], IMPL,
        "sustain-v6", "adjacent-v1", SCOPE, prefix(), unit)


# Captured from the untouched before-image using the exact R16 JSON data.
@pytest.mark.parametrize("policy,recovery,scope,target_sha,receipt_sha", [
    ("sustain-v2", "off", None,
     "a9c6cafaf725bbc1c3970d57fdf2b68a056ad568f1f200d79e60b4e203ef9285",
     "16e247587b074a24fdce0ec1d3f228656d35ddcce76689430926fd1417488ec6"),
    ("sustain-v3", "off", None,
     "16650d03dfcff6a942260a47027b8b1c385798062859993ebd1ffd4451bee219",
     "bcf0298977e095d5966c028c2021ec006053cfc9c4f5343f8c4a81d8af138225"),
    ("sustain-v4", "adjacent-v1", None,
     "6e52679cca017d63803604e0d1d003d5319bdf20d0dd0f1e38d6351e8e21f9fa",
     "733659429f7f7adc9c81fb2a809b5e16a5df51c7ed7eb0caccf1ab8389626d0b"),
    ("sustain-v5", "adjacent-v1", None,
     "b89509e97e886bb02068a086c172a8d560c6535030808ee4fc2ba86211f9850d",
     "a2dd475697958cb8d11ca25867ad1c468a90c927cf79e11e0f7ad15f5834a931"),
    ("sustain-v6", "adjacent-v1", None,
     "9f8b38b4edab33a1af6627a8bd6e845c14c06286e1592d97694baf9d866b7573",
     "7d89152e11a16b24f63b60e1760323f57238cadb23d4c55792e62235fb49c4b0"),
    ("sustain-v6", "adjacent-v1", SCOPE,
     "32bdfd4dc07a22f1581032e841841500b4acbbc2c1b3eb42abdcf70dd392ee50",
     "cbce03ec4fa8ef3bc4375c909db2ad34bb2eca2631c5ced1f23c4b1856450191"),
])
def test_old_target_and_receipt_remain_exact(parent_data, policy, recovery, scope,
                                            target_sha, receipt_sha):
    old = migration.target_contract(parent_data["diablogym_contract"], IMPL,
        policy, recovery, scope, prefix() if scope else None)
    receipt = migration.make_receipt(parent_data, old, PARAMETERS)
    assert migration.json_sha256(old) == target_sha
    assert migration.json_sha256(receipt) == receipt_sha
    assert "depth_shaping_recipe" not in receipt
    migration.validate_inherited_receipt(receipt, old)


def test_new_target_changes_only_one_numeric_field_and_does_not_mutate_source(parent_data):
    before = deepcopy(parent_data)
    old, new = target(parent_data, None), target(parent_data)
    assert {key for key in set(old) | set(new) if old.get(key) != new.get(key)} == {
        "worker_depth_shaping_unit"}
    assert new["worker_depth_shaping_unit"] == 24.0
    assert new["gamma"] == 1.0
    assert new["worker_descend_escrow_fraction"] == 0.5
    assert new["worker_descend_escrow_power"] == 1.6
    assert new["worker_descend_escrow_readiness_gate"] is True
    receipt = migration.make_receipt(parent_data, new, PARAMETERS)
    assert receipt["operation"] == migration.DEPTH_SIGNAL_OPERATION
    assert receipt["depth_shaping_recipe"] == migration.depth_shaping_recipe(24)
    assert receipt["target_contract_sha256"] == migration.json_sha256(new)
    assert receipt["optimizer_state"] == "reset-empty"
    assert receipt["current_world_training_steps"] == 0
    migration.validate_inherited_receipt(receipt, new)
    assert parent_data == before


@pytest.mark.parametrize("unit", [0, 1, 12, 24.00001, 48, -24, True, False, "24",
                                      float("nan"), float("inf")])
def test_unregistered_units_are_rejected(parent_data, unit):
    with pytest.raises(ValueError, match="unit 24"):
        target(parent_data, unit)


@pytest.mark.parametrize("policy,recovery,scope", [
    ("sustain-v5", "adjacent-v1", SCOPE),
    ("sustain-v6", "off", SCOPE),
    ("sustain-v6", "adjacent-v1", "farm-dive-v1"),
    ("sustain-v6", "adjacent-v1", None),
])
def test_depth_signal_requires_exact_scope_service_and_recovery(parent_data, policy, recovery, scope):
    with pytest.raises(ValueError, match="Depth signal requires"):
        migration.target_contract(parent_data["diablogym_contract"], IMPL,
            policy, recovery, scope, prefix() if scope == SCOPE else None, 24)


def test_parent_and_prefix_still_require_exact_registered_identity(parent_data):
    with pytest.raises(ValueError, match="registered R16 parent contract"):
        migration.target_contract({**parent_data["diablogym_contract"], "gamma": 0.9},
            IMPL, "sustain-v6", "adjacent-v1", SCOPE, prefix(), 24)
    with pytest.raises(ValueError, match="exact registered R16 prefix"):
        migration.target_contract(parent_data["diablogym_contract"], IMPL,
            "sustain-v6", "adjacent-v1", SCOPE, {**prefix(), "source_sha256": "c" * 64}, 24)


@pytest.mark.parametrize("key,value", [
    ("gamma", 0.99), ("worker_descend_bonus_fraction", 0.25),
    ("worker_descend_escrow_fraction", 0.25), ("worker_descend_escrow_power", 1.0),
    ("worker_descend_escrow_readiness_gate", False), ("worker_hp_loss_price", 0.0),
    ("worker_no_progress_timeout_credit", "death-equivalent"), ("action_n", 16),
    ("max_steps", 7800), ("resource_protocol", "off"), ("resource_purchase_mode", "armor"),
])
def test_other_target_drift_rejected_even_if_receipt_hash_resealed(parent_data, key, value):
    new = target(parent_data)
    receipt = migration.make_receipt(parent_data, new, PARAMETERS)
    new[key] = value
    receipt["target_contract_sha256"] = migration.json_sha256(new)
    with pytest.raises(ValueError):
        migration.validate_inherited_receipt(receipt, new)


def test_new_and_old_operations_cannot_cross_bind(parent_data):
    old, new = target(parent_data, None), target(parent_data)
    old_receipt = migration.make_receipt(parent_data, old, PARAMETERS)
    new_receipt = migration.make_receipt(parent_data, new, PARAMETERS)
    for receipt, destination in [(old_receipt, new), (new_receipt, old)]:
        receipt["target_contract_sha256"] = migration.json_sha256(destination)
        with pytest.raises(ValueError, match="lineage"):
            migration.validate_inherited_receipt(receipt, destination)
    old_receipt = migration.make_receipt(parent_data, old, PARAMETERS)
    old_receipt["depth_shaping_recipe"] = migration.depth_shaping_recipe(24)
    with pytest.raises(ValueError, match="depth signal recipe"):
        migration.validate_inherited_receipt(old_receipt, old)


@pytest.mark.parametrize("change", ["missing", "refund", "unit", "critic"])
def test_new_receipt_recipe_is_exact(parent_data, change):
    new = target(parent_data)
    receipt = migration.make_receipt(parent_data, new, PARAMETERS)
    if change == "missing":
        receipt.pop("depth_shaping_recipe")
    else:
        key = {"refund": "true_terminal", "unit": "unit", "critic": "critic_initialization"}[change]
        receipt["depth_shaping_recipe"][key] = "wrong"
    with pytest.raises(ValueError, match="depth signal recipe"):
        migration.validate_inherited_receipt(receipt, new)


def test_inherited_runtime_validates_new_target_without_model_forward(parent_data):
    new = target(parent_data)
    model = SimpleNamespace(diablogym_contract=new,
        _resource_warm_start_receipt=migration.make_receipt(parent_data, new, PARAMETERS),
        n_steps=512, n_envs=4, observation_space=SimpleNamespace(shape=(13012,)),
        action_space=SimpleNamespace(n=15), gradient_clip_mode="separate-root-context-critic-v2",
        policy=SimpleNamespace(mlp_extractor=SimpleNamespace(actor_context_enabled=True)),
        _actor_optimizer_steps_completed=1)
    migration.validate_inherited_runtime(model)
    model.diablogym_contract = target(parent_data, None)
    with pytest.raises(ValueError):
        migration.validate_inherited_runtime(model)


@pytest.mark.parametrize("recipe_mode", ["exact", "missing", "modified"])
def test_manifest_capture_binds_shaping_recipe_without_loading_model(parent_data, tmp_path, recipe_mode):
    new = target(parent_data)
    receipt = migration.make_receipt(parent_data, new, PARAMETERS)
    payload = b"synthetic-checkpoint-validator-boundary"
    data = {"diablogym_contract": new, "_resource_warm_start_receipt": receipt,
            **{name: 0 for name in migration.ZERO_COUNTERS},
            "_last_completed_ppo_rollout_steps": None}
    manifest = {"schema": migration.SCHEMA, "status": "INITIALIZATION_ONLY_NOT_TRAINED",
        "model_file": "model_warm_start.zip", "model_sha256": hashlib.sha256(payload).hexdigest(),
        "parent_sha256": migration.PARENT_SHA256, "operation": receipt["operation"],
        "ordinary_resume_eligible": False, "trained_in_target_world": False,
        "policy_sha256": PARAMETERS, "target_contract": new, "receipt": receipt,
        "source_contract": receipt["source_contract"],
        "dive_blocker_recovery_recipe": receipt["dive_blocker_recovery_recipe"]}
    if recipe_mode != "missing":
        manifest["depth_shaping_recipe"] = deepcopy(receipt["depth_shaping_recipe"])
    if recipe_mode == "modified":
        manifest["depth_shaping_recipe"]["true_terminal"] = "no-refund"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    (tmp_path / "model_warm_start.zip").write_bytes(payload)
    validator = Mock(return_value=data)
    with patch.dict(sys.modules, {"train_ppo": SimpleNamespace(_validate_checkpoint_bytes=validator)}):
        if recipe_mode == "exact":
            assert migration.capture_initialization(path, IMPL)[0] == payload
        else:
            with pytest.raises(ValueError, match="manifest depth signal"):
                migration.capture_initialization(path, IMPL)
    validator.assert_called_once()


@pytest.mark.parametrize("selected", [None, "24"])
def test_cli_selection_is_explicit_and_forwards_only_registered_value(parent_data, selected):
    argv = ["migration", "--parent", "unused-parent", "--parent-sha256", migration.PARENT_SHA256,
        "--implementation-sha256", IMPL, "--output-dir", "must-not-be-created", "--seed", "2168000",
        "--resource-service-policy", "sustain-v6", "--dive-blocker-recovery", "adjacent-v1",
        "--worker-learning-window-scope", SCOPE, "--worker-prefix-model",
        str(ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"),
        "--worker-prefix-max-attempts", "64", "--worker-prefix-max-microsteps", "384000"]
    if selected is not None:
        argv += ["--worker-depth-shaping-unit", selected]
    result = dict(status="INITIALIZATION_ONLY_NOT_TRAINED", model_sha256="c" * 64, policy_sha256=PARAMETERS)
    with patch.object(sys, "argv", argv), patch.object(migration, "migrate", return_value=result) as migrate:
        migration.main()
    migrate.assert_called_once()
    assert migrate.call_args.kwargs["worker_depth_shaping_unit"] == (24.0 if selected else None)
    assert migrate.call_args.kwargs["worker_prefix"] == prefix()


def test_cli_bad_unit_stops_before_migration():
    argv = ["migration", "--parent", "unused", "--parent-sha256", migration.PARENT_SHA256,
        "--implementation-sha256", IMPL, "--output-dir", "unused", "--seed", "2168000",
        "--worker-depth-shaping-unit", "12"]
    with patch.object(sys, "argv", argv), patch.object(migration, "migrate") as migrate:
        with pytest.raises(SystemExit):
            migration.main()
    migrate.assert_not_called()
