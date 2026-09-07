"""Exact completed depth-24 parent -> completion-l2-v1 weights-only initialization.

This is a new zero-Adam, zero-step run, never an exact trajectory continuation.
The explicit CLI reuses the existing save/load, tensor, probe and zero-state
checks. Importing this module or validating JSON metadata does not load a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import migrate_resource_candidate as legacy

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA256 = "f466d04b322213a244c631da9c239a60e5e50cd0b6e3a2fcda9a170734bcc2e5"
PARENT_CONTRACT_SHA256 = "2a63dca82b94e3f41dc2823d89124dd35a22e4fc1814f0a80e39e783890a1ce2"
PARENT_RECEIPT_SHA256 = "9708c3c1b2f10ccc703050d181d132e23f5a48e070b4a9e7f18ec08c09e7b7bc"
PARENT_ROLLOUTS_SHA256 = "b07e133ef5f441c01d3121e3f674bedab3e4251371f4e836ed9b16e9cc9ad539"
PARENT_LINEAGE_SHA256 = "d061dba8ea3689ec607e8fa86d8951b188597340aecf4fb17b736d16105b94f5"
PARENT_STEPS = 8192
OPERATION = "earned-depth24-8192-to-completion-l2-v1-weights-only-v1"
ALLOWED_CONTRACT_KEYS = frozenset({
    "implementation_sha256", "worker_time_protocol", "worker_time_recipe",
})
PARENT_COUNTERS = {
    "num_timesteps": 8192, "_total_timesteps": 8192, "_n_updates": 40,
    "_last_completed_ppo_rollout_steps": 8192,
    "_ppo_optimizer_steps_completed": 316, "_actor_optimizer_steps_completed": 316,
    "_worker_onpolicy_pg_joint_rollouts": 4, "_worker_onpolicy_pg_qualifying_rollouts": 3,
}
PARENT_WARMUP = {
    "_critic_warmup_start_timesteps": None, "_critic_warmup_until_timesteps": None,
    "_critic_warmup_expected_rollouts": 0, "_critic_warmup_rollouts_completed": 0,
    "_critic_warmup_optimizer_steps_completed": 0, "_critic_warmup_completed": False,
    "_critic_warmup_actor_sha256": None, "_worker_onpolicy_pg_audit_required": True,
}
require = legacy.require
json_sha256 = legacy.json_sha256


@lru_cache(maxsize=1)
def _clock_recipe():
    # Import the standalone stdlib recipe without executing diablogym.__init__.
    name = "_diablogym_completion_migration_recipe"
    spec = importlib.util.spec_from_file_location(name,
        ROOT / "python/diablogym/completion_clock.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.COMPLETION_L2_V1


def time_recipe():
    return _clock_recipe().as_dict()


def validate_source_contract(source):
    require(isinstance(source, dict) and json_sha256(source) == PARENT_CONTRACT_SHA256,
            "Only the exact completed 8192-step parent contract is eligible")


def target_contract(source, implementation_sha256):
    validate_source_contract(source)
    require(isinstance(implementation_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", implementation_sha256),
            "Invalid target implementation SHA256")
    target = deepcopy(source)
    recipe = time_recipe()
    target.update(implementation_sha256=implementation_sha256,
                  worker_time_protocol=recipe["protocol"], worker_time_recipe=recipe)
    return target


def validate_target_contract(source, target, implementation_sha256):
    require(isinstance(target, dict), "Invalid completion target contract")
    expected = target_contract(source, implementation_sha256)
    require(json_sha256(target) == json_sha256(expected),
            "Completion migration may change only implementation and the exact time recipe")
    changed = {key for key in set(source) | set(target)
               if key not in source or key not in target or source[key] != target[key]}
    require(changed <= ALLOWED_CONTRACT_KEYS, "Unexpected completion contract changes")


def validate_parent_metadata(data):
    """Validate the fixed parent's JSON evidence, including every original receipt."""
    require(isinstance(data, dict), "Invalid parent metadata")
    source = data.get("diablogym_contract")
    validate_source_contract(source)
    for key, value in {**PARENT_COUNTERS, **PARENT_WARMUP}.items():
        require(key in data and type(data[key]) is type(value) and data[key] == value,
                f"Registered parent counter/state differs: {key}")
    for key in ("_worker_onpolicy_pg_pending_receipts", "_bc_aux_pending_action_receipts"):
        require(key not in data or data[key] == [], "Parent contains unconsumed pending receipts")
    inherited = data.get("_resource_warm_start_receipt")
    require(isinstance(inherited, dict) and json_sha256(inherited) == PARENT_RECEIPT_SHA256,
            "Parent inherited R16 receipt differs")
    # This is the old operation, so it must still pass the unchanged R16 rules.
    legacy.validate_inherited_receipt(inherited, source)
    require(json_sha256(data.get("_resume_lineage")) == PARENT_LINEAGE_SHA256,
            "Parent resume lineage differs")
    receipts = data.get("_worker_onpolicy_pg_rollout_receipts")
    require(isinstance(receipts, list) and len(receipts) == 4
            and json_sha256(receipts) == PARENT_ROLLOUTS_SHA256,
            "Parent completed rollout evidence differs")
    from leashed_ppo import validate_worker_onpolicy_pg_receipt
    for index, receipt in enumerate(receipts, 1):
        require(validate_worker_onpolicy_pg_receipt(receipt, expected_samples=2048)
                and receipt["rollout_end_timesteps"] == index * 2048,
                "Invalid or non-contiguous parent rollout receipt")
    require(sum(r["optimizer_steps"] for r in receipts) == 316
            and sum(r["qualifies"] is True for r in receipts) == 3,
            "Parent optimizer/qualifying counts do not close")


def make_receipt(parent_data, target, parameters_sha):
    validate_parent_metadata(parent_data)
    source = parent_data["diablogym_contract"]
    validate_target_contract(source, target, target.get("implementation_sha256"))
    require(isinstance(parameters_sha, str) and re.fullmatch(r"[0-9a-f]{64}", parameters_sha),
            "Invalid preserved policy tensor SHA256")
    inherited = parent_data["_resource_warm_start_receipt"]
    result = {
        "schema": legacy.SCHEMA, "operation": OPERATION,
        "parent_checkpoint_sha256": PARENT_SHA256,
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "parent_num_timesteps": PARENT_STEPS,
        "source_contract": deepcopy(source), "target_contract_sha256": json_sha256(target),
        "policy_sha256": parameters_sha,
        "inherited_resource_warm_start_receipt": deepcopy(inherited),
        "inherited_resource_warm_start_receipt_sha256": PARENT_RECEIPT_SHA256,
        "parent_completion": {
            "counters": {key: deepcopy(parent_data[key]) for key in PARENT_COUNTERS},
            "inherited_runtime_state": {key: deepcopy(parent_data[key]) for key in PARENT_WARMUP},
            "rollout_receipts": deepcopy(parent_data["_worker_onpolicy_pg_rollout_receipts"]),
            "rollout_receipts_sha256": PARENT_ROLLOUTS_SHA256,
        },
        "historical_critic_warmup": deepcopy(inherited["historical_critic_warmup"]),
        "parent_resume_lineage": deepcopy(parent_data["_resume_lineage"]),
        "optimizer_state": "reset-empty", "current_world_training_steps": 0,
        "historical_warmup_is_current_training": False,
        "exact_trajectory_continuation": False,
        "environment_state_mode": "fresh-normal-l1-no-snapshot",
        "publication_status": "INITIALIZATION_ONLY_NOT_TRAINED",
        "worker_time_protocol": target["worker_time_protocol"],
        "worker_time_recipe": deepcopy(target["worker_time_recipe"]),
        "dive_blocker_recovery_recipe": deepcopy(inherited["dive_blocker_recovery_recipe"]),
        "depth_shaping_recipe": deepcopy(inherited["depth_shaping_recipe"]),
    }
    return result


def validate_inherited_receipt(receipt, target):
    require(isinstance(receipt, dict) and isinstance(target, dict),
            "Invalid completion warm-start lineage")
    completion = receipt.get("parent_completion")
    require(isinstance(completion, dict)
            and set(completion) == {"counters", "inherited_runtime_state", "rollout_receipts",
                                    "rollout_receipts_sha256"}
            and isinstance(completion.get("counters"), dict)
            and isinstance(completion.get("inherited_runtime_state"), dict),
            "Missing completion parent evidence")
    require(set(completion["counters"]) == set(PARENT_COUNTERS)
            and set(completion["inherited_runtime_state"]) == set(PARENT_WARMUP),
            "Parent evidence fields drift")
    parent = {
        **completion["counters"], **completion["inherited_runtime_state"],
        "diablogym_contract": receipt.get("source_contract"),
        "_resource_warm_start_receipt": receipt.get("inherited_resource_warm_start_receipt"),
        "_worker_onpolicy_pg_rollout_receipts": completion.get("rollout_receipts"),
        "_resume_lineage": receipt.get("parent_resume_lineage"),
    }
    expected = make_receipt(parent, target, receipt.get("policy_sha256"))
    require(json_sha256(receipt) == json_sha256(expected),
            "Completion warm-start receipt identity or semantics drift")


def migrate(parent, *, parent_sha256, output_dir, implementation_sha256, seed):
    """Explicit-only model operation; all validation also remains in the reload path."""
    require(parent_sha256 == PARENT_SHA256, "Only the exact registered 8192 parent SHA is accepted")
    require(type(seed) is int and 0 <= seed < 2**32, "Explicit uint32 seed required")
    source_path = Path(parent).resolve(strict=True)
    destination = Path(output_dir).absolute()
    require(not destination.exists(), "Warm-start output directory already exists")
    payload = source_path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == PARENT_SHA256, "Parent checkpoint SHA mismatch")
    import torch
    from leashed_ppo import LeashedMaskablePPO
    from train_ppo import (_validate_checkpoint_bytes, _validate_resumable_leashed_boundary,
        _atomic_save_model, _implementation_bundle_sha256)
    parent_data = _validate_resumable_leashed_boundary(_validate_checkpoint_bytes(
        payload, str(source_path), require_leashed=True))
    validate_parent_metadata(parent_data)
    source = parent_data["diablogym_contract"]
    target = target_contract(source, implementation_sha256)
    require(_implementation_bundle_sha256() == implementation_sha256,
            "Target implementation changed before migration")
    torch.set_num_threads(1)
    model = LeashedMaskablePPO.load(io.BytesIO(payload), device="cpu",
                                  teacher_path=None, teacher_sha256=None)
    before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
    parameters_sha = legacy.policy_sha256(model)
    probe = legacy.policy_probe(model)
    receipt = make_receipt(parent_data, target, parameters_sha)
    legacy.reset_to_initialization(model, receipt, target, seed)
    for name, value in model.policy.state_dict().items():
        require(torch.equal(before[name], value.detach()), f"Changed policy tensor: {name}")
    require(legacy.policy_probe(model) == probe, "Reset changed fixed logits or critic values")
    destination.mkdir(parents=True, exist_ok=False)
    model_path = _atomic_save_model(model, destination / "model_warm_start.zip")
    manifest = {
        "schema": legacy.SCHEMA, "operation": OPERATION,
        "status": "INITIALIZATION_ONLY_NOT_TRAINED", "seed": seed,
        "parent_path": str(source_path), "parent_sha256": PARENT_SHA256,
        "source_contract": source, "target_contract": target, "receipt": receipt,
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "policy_tensor_count": len(before), "policy_sha256": parameters_sha,
        "policy_probe": probe, "ordinary_resume_eligible": False,
        "trained_in_target_world": False,
        "dive_blocker_recovery_recipe": deepcopy(receipt["dive_blocker_recovery_recipe"]),
        "depth_shaping_recipe": deepcopy(receipt["depth_shaping_recipe"]),
    }
    restored = legacy.load_initialization(model_path.read_bytes(), manifest)
    for name, value in restored.policy.state_dict().items():
        require(torch.equal(before[name], value.detach()), f"Save/load changed policy tensor: {name}")
    require(hashlib.sha256(source_path.read_bytes()).hexdigest() == PARENT_SHA256,
            "Parent checkpoint changed during migration")
    require(_implementation_bundle_sha256() == implementation_sha256,
            "Target implementation changed during migration")
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    legacy.capture_initialization(destination / "manifest.json", implementation_sha256)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--implementation-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    result = migrate(args.parent, parent_sha256=args.parent_sha256,
        output_dir=args.output_dir, implementation_sha256=args.implementation_sha256, seed=args.seed)
    print(json.dumps({key: result[key] for key in ("status", "model_sha256", "policy_sha256")}))


if __name__ == "__main__":
    main()
