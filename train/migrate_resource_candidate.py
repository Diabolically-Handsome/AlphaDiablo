"""Exact R16 -> explicit sustain-v2/v3/v4/v5/v6 weight initialization; never a resumed trajectory.

The standalone command does not create a game or perform an optimizer update.
Its zero-step output deliberately fails the ordinary trained-resume gate.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
PARENT_SHA256 = "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"
PARENT_CONTRACT_SHA256 = "6a2fb5e652ee46c48ec9a604353525a4e98e81fc7bb6fb071d21524e30993b7f"
PARENT_STEPS = 4089856
SCHEMA = "diablogym-resource-warm-start/1"
# R18-B3b (2026-09-07): the loot warm-start schema lives in
# migrate_loot_candidate.py; its name is declared here so this module can
# accept it without importing that module at import time (it imports this one).
SCHEMA_V2 = "diablogym-resource-warm-start/2"
OPERATION = "r16-to-sustain-v2-weights-only-v1"
OPERATIONS = {"sustain-v2": OPERATION,
              "sustain-v3": "r16-to-sustain-v3-weights-only-v1",
              "sustain-v4": "r16-to-sustain-v4-weights-only-v1",
              "sustain-v5": "r16-to-sustain-v5-weights-only-v1",
              "sustain-v6": "r16-to-sustain-v6-weights-only-v1",
              # R18-B3 (2026-09-07): a new versioned operation; the sustain-v6
              # names above are frozen and are never renamed or reused.
              # R18-B3 review round: the name is RESERVED, not yet mintable --
              # operation_for refuses sustain-loot-v1 until the migration schema
              # carries worker_time_protocol (see the require there).
              "sustain-loot-v1": "r16-to-sustain-loot-v1-weights-only-v1"}
RESOURCE_KEYS = frozenset({"resource_protocol", "resource_purchase_mode",
    "resource_service_policy", "resource_service_recipe"})
BASE_ALLOWED_CONTRACT_KEYS = RESOURCE_KEYS | {"implementation_sha256"}
ALLOWED_CONTRACT_KEYS = BASE_ALLOWED_CONTRACT_KEYS | {"dive_blocker_recovery"}
EARNED_ALLOWED_CONTRACT_KEYS = ALLOWED_CONTRACT_KEYS | {"worker_learning_window_scope", "worker_prefix"}
DEPTH_SIGNAL_OPERATION = (
    "r16-to-sustain-v6-earned-dive-suffix-v1-depth24-dive-adjacent-v1-weights-only-v1")
ZERO_COUNTERS = ("num_timesteps", "_total_timesteps", "_num_timesteps_at_start",
    "_episode_num", "_n_updates", "_ppo_optimizer_steps_completed",
    "_actor_optimizer_steps_completed", "_distill_actor_rollouts_completed",
    "_worker_onpolicy_pg_joint_rollouts", "_worker_onpolicy_pg_qualifying_rollouts",
    "_critic_warmup_expected_rollouts", "_critic_warmup_rollouts_completed",
    "_critic_warmup_optimizer_steps_completed")
EMPTY_LISTS = ("_worker_onpolicy_pg_pending_receipts",
    "_worker_onpolicy_pg_rollout_receipts", "_bc_aux_pending_action_receipts")
WARMUP_FIELDS = ("_critic_warmup_start_timesteps", "_critic_warmup_until_timesteps",
    "_critic_warmup_expected_rollouts", "_critic_warmup_rollouts_completed",
    "_critic_warmup_optimizer_steps_completed", "_critic_warmup_completed",
    "_critic_warmup_actor_sha256")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def validate_source_contract(source):
    require(isinstance(source, dict) and json_sha256(source) == PARENT_CONTRACT_SHA256,
            "Only the exact registered R16 parent contract is eligible")


def depth_shaping_recipe(unit):
    """One registered signal experiment, not a general reward-drift allowance."""
    if unit is None:
        return None
    require(type(unit) in (int, float) and unit == 24.0,
            "Only the explicitly registered depth-shaping unit 24 is eligible")
    return {
        "version": "earned-depth-signal-24-v1", "unit": 24.0, "gamma": 1.0,
        "potential": "unit*(max(1,episode_max_depth)-1)",
        "transition": "potential_after-minus-potential_before",
        "true_terminal": "refund-final-potential",
        "safe_time_limit": "no-refund-existing-value-bootstrap",
        "rollout_cutoff": "no-refund-existing-value-bootstrap",
        "interrupted_reset": "outside-complete-terminal-episode-guarantee",
        "critic_initialization": "unchanged-parent-no-potential-offset",
        "existing_escrow": "unchanged-fraction0.5-power1.6-native-readiness-gate",
    }


def operation_for(service_policy, recovery="off", worker_learning_window_scope=None,
                  worker_depth_shaping_unit=None):
    from eval_contract import validate_dive_blocker_recovery, EARNED_DIVE_SUFFIX_SCOPE
    require(service_policy in OPERATIONS,
            "Only explicit sustain-v2/v3/v4/v5/v6/sustain-loot-v1 targets are eligible")
    # R18-B3 复审 (2026-09-07):sustain-loot-v1 的操作名已经登记(见 OPERATIONS),
    # 但迁移契约的词汇表(ALLOWED_CONTRACT_KEYS / target_contract)里没有
    # worker_time_protocol,而 loot 配方自本版起随完成时钟分版。真放行,就会冻结
    # 一份 time_protocol=completion-l2-v1 的收据——一句没人核实过的时钟断言;
    # 而 train_ppo._validate_resource_warm_start_args 又无条件拒绝一切 loot
    # warm-start,那份收据永远没人能消费。在迁移 schema 升版长出时钟键之前
    # fail closed。
    require(service_policy != "sustain-loot-v1",
            "sustain-loot-v1 migration requires a completion-l2 clock in the migration "
            "schema (worker_time_protocol is not in ALLOWED_CONTRACT_KEYS yet)")
    validate_dive_blocker_recovery("l2-town-v1", recovery)
    require(worker_learning_window_scope in (None, "farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE),
            "Resource initialization allows only registered farm-dive or earned-suffix scopes")
    if depth_shaping_recipe(worker_depth_shaping_unit) is not None:
        require(service_policy == "sustain-v6" and recovery == "adjacent-v1"
                and worker_learning_window_scope == EARNED_DIVE_SUFFIX_SCOPE,
                "Depth signal requires earned-dive-suffix-v1, sustain-v6 and adjacent-v1")
        return DEPTH_SIGNAL_OPERATION
    if worker_learning_window_scope == EARNED_DIVE_SUFFIX_SCOPE:
        # R18-B3 (2026-09-07): sustain-loot-v1 gets its own operation name from
        # the same template; the sustain-v6 string it produces is byte-identical
        # to the frozen one it replaces.  R18-B3 review round: loot is already
        # refused at the top of this function until the migration schema carries
        # worker_time_protocol, so the loot arm here is the single definition of
        # the reserved name, not a reachable branch.
        require(service_policy in ("sustain-v6", "sustain-loot-v1") and recovery == "adjacent-v1",
                "Earned suffix initialization requires sustain-v6 or sustain-loot-v1 plus adjacent-v1")
        return f"r16-to-{service_policy}-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v1"
    if recovery == "off":
        return OPERATIONS[service_policy]
    return f"r16-to-{service_policy}-dive-adjacent-v1-weights-only-v1"


def target_contract(source, implementation_sha256, service_policy="sustain-v2",
                    dive_blocker_recovery="off", worker_learning_window_scope=None, worker_prefix=None,
                    worker_depth_shaping_unit=None):
    from eval_contract import resource_service_recipe, worker_prefix_recipe, EARNED_DIVE_SUFFIX_SCOPE
    validate_source_contract(source)
    require(isinstance(implementation_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", implementation_sha256),
            "Invalid target implementation SHA256")
    operation_for(service_policy, dive_blocker_recovery, worker_learning_window_scope,
                  worker_depth_shaping_unit)
    if worker_learning_window_scope == EARNED_DIVE_SUFFIX_SCOPE:
        require(isinstance(worker_prefix, dict), "Earned suffix requires an exact prefix recipe")
        expected_prefix = worker_prefix_recipe(EARNED_DIVE_SUFFIX_SCOPE,
            worker_prefix.get("source_sha256"), worker_prefix.get("max_attempts"),
            worker_prefix.get("max_microsteps"))
        require(worker_prefix == expected_prefix, "Earned suffix prefix recipe drift")
    else:
        require(worker_prefix is None, "Legacy scope cannot carry prefix metadata")
    target = deepcopy(source)
    target.update(resource_protocol="l2-town-v1", resource_purchase_mode="full",
        resource_service_policy=service_policy, resource_service_recipe=
        resource_service_recipe("l2-town-v1", "full", service_policy),
        implementation_sha256=implementation_sha256)
    if dive_blocker_recovery != "off":
        target["dive_blocker_recovery"] = dive_blocker_recovery
    if worker_learning_window_scope == EARNED_DIVE_SUFFIX_SCOPE:
        target["worker_learning_window_scope"] = EARNED_DIVE_SUFFIX_SCOPE
        target["worker_prefix"] = deepcopy(expected_prefix)
    if worker_depth_shaping_unit is not None:
        require(source.get("gamma") == 1.0, "Depth signal requires the original gamma=1 contract")
        target["worker_depth_shaping_unit"] = 24.0
    return target


def validate_target_contract(source, target, implementation_sha256):
    from eval_contract import EARNED_DIVE_SUFFIX_SCOPE
    require(isinstance(target, dict), "Invalid warm-start target contract")
    expected = target_contract(source, implementation_sha256, target.get("resource_service_policy"),
                               target.get("dive_blocker_recovery") or "off",
                               target.get("worker_learning_window_scope"), target.get("worker_prefix"),
                               target.get("worker_depth_shaping_unit"))
    require(isinstance(target, dict) and all(target.get(key) == expected.get(key)
            for key in set(target) | set(expected)),
            "Warm-start target contract drift outside the exact resource migration")
    changed = {key for key in set(source) | set(target) if source.get(key) != target.get(key)}
    allowed = (EARNED_ALLOWED_CONTRACT_KEYS
               if target.get("worker_learning_window_scope") == EARNED_DIVE_SUFFIX_SCOPE
               else ALLOWED_CONTRACT_KEYS)
    if target.get("worker_depth_shaping_unit") is not None:
        allowed = allowed | {"worker_depth_shaping_unit"}
    require(changed <= allowed, "Unexpected warm-start contract changes")


def policy_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        header = json.dumps([name, str(array.dtype), list(array.shape)],
                            separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def policy_probe(model):
    import numpy as np
    import torch
    previous_threads = torch.get_num_threads()
    try:
        # The exact probe was registered with the migration CPU thread count.
        torch.set_num_threads(1)
        width = int(model.observation_space.shape[0])
        rng = np.random.default_rng(2114999)
        obs = np.vstack([np.zeros((1, width), dtype=np.float32),
                         rng.uniform(-0.1, 0.1, (3, width)).astype(np.float32)])
        model.policy.set_training_mode(False)
        with torch.no_grad():
            tensor = torch.as_tensor(obs, device=model.device)
            logits = model.policy.get_distribution(tensor,
                action_masks=np.ones((4, int(model.action_space.n)), dtype=bool)).distribution.logits
            values = model.policy.predict_values(tensor)
        require(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(values).all()),
                "Nonfinite warm-start probe")
        return {"schema": "fixed-synthetic-logits-values-v1", "input_seed": 2114999,
                "logits_sha256": hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),
                "values_sha256": hashlib.sha256(values.cpu().numpy().tobytes()).hexdigest()}
    finally:
        torch.set_num_threads(previous_threads)


def make_receipt(parent_data, target, parameters_sha):
    source = parent_data["diablogym_contract"]
    validate_source_contract(source)
    require(parent_data.get("num_timesteps") == PARENT_STEPS,
            "Parent training count differs from the registered R16")
    historical = {name: deepcopy(parent_data.get(name)) for name in WARMUP_FIELDS}
    require(historical["_critic_warmup_completed"] is True
            and historical["_critic_warmup_rollouts_completed"] == 8
            and historical["_critic_warmup_optimizer_steps_completed"] == 640,
            "Registered parent critic warmup is incomplete")
    validate_target_contract(source, target, target.get("implementation_sha256"))
    result = {"schema": SCHEMA, "operation": operation_for(target["resource_service_policy"],
        target.get("dive_blocker_recovery") or "off", target.get("worker_learning_window_scope"),
        target.get("worker_depth_shaping_unit")),
        "parent_checkpoint_sha256": PARENT_SHA256,
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "parent_num_timesteps": PARENT_STEPS,
        "source_contract": deepcopy(source), "target_contract_sha256": json_sha256(target),
        "policy_sha256": parameters_sha, "historical_critic_warmup": historical,
        "parent_resume_lineage": deepcopy(parent_data.get("_resume_lineage")),
        "optimizer_state": "reset-empty", "current_world_training_steps": 0,
        "historical_warmup_is_current_training": False,
        "exact_trajectory_continuation": False,
        "environment_state_mode": "fresh-normal-l1-no-snapshot",
        "publication_status": "INITIALIZATION_ONLY_NOT_TRAINED"}
    if target.get("dive_blocker_recovery") == "adjacent-v1":
        from eval_contract import dive_blocker_recovery_recipe
        result["dive_blocker_recovery_recipe"] = dive_blocker_recovery_recipe("adjacent-v1")
    shaping = depth_shaping_recipe(target.get("worker_depth_shaping_unit"))
    if shaping is not None:
        result["depth_shaping_recipe"] = shaping
    return result


def validate_inherited_receipt(receipt, target):
    # R18-B3b (2026-09-07): schema/2 is the sustain-loot-v1 earned-suffix family;
    # it carries the completion clock this schema has no key for. The two
    # conditions are disjoint, and anything else still falls through to the
    # unchanged schema/1 rules below (where a loot target is refused by
    # operation_for, so a mislabelled receipt cannot be relabelled into one).
    if isinstance(receipt, dict) and receipt.get("schema") == SCHEMA_V2:
        from migrate_loot_candidate import validate_inherited_receipt as validate_loot_receipt
        return validate_loot_receipt(receipt, target)
    if (isinstance(receipt, dict) and receipt.get("operation") ==
            "earned-depth24-8192-to-completion-l2-v1-weights-only-v1"):
        from migrate_completion_candidate import validate_inherited_receipt as validate_completion_receipt
        return validate_completion_receipt(receipt, target)
    require(isinstance(target, dict), "Invalid warm-start target contract")
    require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA
        and receipt.get("operation") == operation_for(target.get("resource_service_policy"),
            target.get("dive_blocker_recovery") or "off", target.get("worker_learning_window_scope"),
            target.get("worker_depth_shaping_unit"))
        and receipt.get("parent_checkpoint_sha256") == PARENT_SHA256
        and receipt.get("parent_contract_sha256") == PARENT_CONTRACT_SHA256
        and receipt.get("parent_num_timesteps") == PARENT_STEPS,
        "Invalid registered resource warm-start lineage")
    validate_target_contract(receipt.get("source_contract"), target,
                             target.get("implementation_sha256"))
    expected = target_contract(receipt["source_contract"], target["implementation_sha256"],
                                target["resource_service_policy"], target.get("dive_blocker_recovery") or "off",
                                target.get("worker_learning_window_scope"), target.get("worker_prefix"),
                                target.get("worker_depth_shaping_unit"))
    shaping = depth_shaping_recipe(target.get("worker_depth_shaping_unit"))
    require((receipt.get("depth_shaping_recipe") == shaping)
        if shaping is not None else ("depth_shaping_recipe" not in receipt),
        "Warm-start depth signal recipe missing or inconsistent")
    from eval_contract import dive_blocker_recovery_recipe
    recovery_recipe = dive_blocker_recovery_recipe(target.get("dive_blocker_recovery") or "off")
    require((receipt.get("dive_blocker_recovery_recipe") == recovery_recipe)
        if recovery_recipe is not None else ("dive_blocker_recovery_recipe" not in receipt),
        "Warm-start DIVE recovery recipe missing or inconsistent")
    require(receipt.get("target_contract_sha256") == json_sha256(expected)
        and receipt.get("optimizer_state") == "reset-empty"
        and receipt.get("current_world_training_steps") == 0
        and receipt.get("historical_warmup_is_current_training") is False
        and receipt.get("exact_trajectory_continuation") is False
        and receipt.get("environment_state_mode") == "fresh-normal-l1-no-snapshot"
        and receipt.get("publication_status") == "INITIALIZATION_ONLY_NOT_TRAINED",
        "Resource warm-start receipt semantics drift")
    historical = receipt.get("historical_critic_warmup", {})
    require(historical.get("_critic_warmup_start_timesteps") == 3497984
        and historical.get("_critic_warmup_until_timesteps") == 3514368
        and historical.get("_critic_warmup_expected_rollouts") == 8
        and historical.get("_critic_warmup_rollouts_completed") == 8
        and historical.get("_critic_warmup_optimizer_steps_completed") == 640
        and historical.get("_critic_warmup_completed") is True
        and historical.get("_critic_warmup_actor_sha256") ==
            "5e870fd86392644dc9c3bcbccee113e8d4ba46e30beecca158ea9bfa175cb7e2",
        "Inherited historical critic warmup evidence drift")
    require(isinstance(receipt.get("policy_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt["policy_sha256"]),
        "Invalid inherited policy SHA256")


def validate_inherited_runtime(model):
    """Called only by Leashed's explicitly marked inherited-warmup branch."""
    receipt = getattr(model, "_resource_warm_start_receipt", None)
    target = getattr(model, "diablogym_contract", {})
    validate_inherited_receipt(receipt, target)
    require(int(model.n_steps) == target["n_steps"]
        and int(model.n_envs) == target["num_envs"]
        and list(model.observation_space.shape) == target["observation_shape"]
        and int(model.action_space.n) == target["action_n"],
        "Inherited warm-start runtime geometry differs from the target contract")
    require(model.gradient_clip_mode == "separate-root-context-critic-v2"
        and model.policy.mlp_extractor.actor_context_enabled is True,
        "Inherited actor/context or gradient clipping is not active")
    if model._actor_optimizer_steps_completed == 0:
        require(policy_sha256(model) == receipt["policy_sha256"],
                "Policy changed before the first warm-start optimizer step")


def reset_to_initialization(model, receipt, target, seed):
    from train_ppo import _reset_policy_optimizer
    validate_inherited_receipt(receipt, target)
    before = policy_sha256(model)
    _reset_policy_optimizer(model, float(target["learning_rate"]))
    for name in ZERO_COUNTERS:
        setattr(model, name, 0)
    for name in EMPTY_LISTS:
        setattr(model, name, [])
    for name in ("_critic_warmup_start_timesteps", "_critic_warmup_until_timesteps",
            "_critic_warmup_actor_sha256", "_worker_onpolicy_pg_collection_actor_sha256",
            "_last_obs", "_last_original_obs", "_last_episode_starts",
            "ep_info_buffer", "ep_success_buffer"):
        setattr(model, name, None)
    model._critic_warmup_completed = False
    model._worker_onpolicy_pg_audit_required = True
    model._last_completed_ppo_rollout_steps = None
    model._current_progress_remaining = 1.0
    model._last_effective_distill_beta = 0.0
    model._resource_warm_start_receipt = deepcopy(receipt)
    model.diablogym_contract = deepcopy(target)
    model._resume_lineage = {
        "schema": "diablogym-resume-lineage/1",
        "generation": int((receipt.get("parent_resume_lineage") or {}).get("generation", 0)) + 1,
        "operation": receipt["operation"], "immediate_parent_sha256": receipt["parent_checkpoint_sha256"],
        "immediate_parent_num_timesteps": receipt["parent_num_timesteps"],
        "optimizer_state": "reset", "critic_state": "preserved",
        "policy_parameter_state": "checkpoint-preserved",
        "exact_trajectory_continuation": False,
        "environment_state_mode": "reinitialized-no-native-or-wrapper-snapshot",
        "rng_state_mode": "reseeded-from-explicit-cli-seed", "requested_seed": seed}
    model.rollout_buffer.reset()
    model.set_random_seed(seed)
    model.seed = seed
    require(before == policy_sha256(model), "Warm-start reset changed policy tensors")
    validate_zero_state(model)
    model._assert_critic_migration_contract()


def validate_zero_state(model):
    for name in ZERO_COUNTERS:
        require(type(getattr(model, name, None)) is int and getattr(model, name) == 0,
                f"Warm-start initialization has nonzero counter: {name}")
    for name in EMPTY_LISTS:
        require(getattr(model, name, None) == [], f"Warm-start has historical receipts: {name}")
    require(not model.policy.optimizer.state and model._last_completed_ppo_rollout_steps is None,
            "Warm-start falsely claims a trained optimizer boundary")
    require(model.rollout_buffer.pos == 0 and not model.rollout_buffer.full,
            "Warm-start contains a collected rollout")
    validate_inherited_runtime(model)


def capture_initialization(manifest_path, expected_implementation=None):
    from eval_contract import strict_json_loads
    from train_ppo import _validate_checkpoint_bytes
    path = Path(manifest_path).resolve(strict=True)
    manifest = strict_json_loads(path.read_text())
    # R18-B3b (2026-09-07): both registered warm-start schemas; every shape check
    # below is schema-independent and the receipt is validated by its own schema.
    require(manifest.get("schema") in (SCHEMA, SCHEMA_V2)
            and manifest.get("status") == "INITIALIZATION_ONLY_NOT_TRAINED",
            "Not a resource warm-start initialization manifest")
    name = manifest.get("model_file")
    require(name == "model_warm_start.zip", "Unexpected warm-start model filename")
    model_path = path.parent / name
    require(not model_path.is_symlink(), "Warm-start model must be a local immutable artifact")
    payload = model_path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == manifest.get("model_sha256"),
            "Warm-start model bytes differ from manifest")
    data = _validate_checkpoint_bytes(payload, str(model_path), require_leashed=True)
    target = data.get("diablogym_contract", {})
    receipt = data.get("_resource_warm_start_receipt")
    validate_inherited_receipt(receipt, target)
    require(manifest.get("parent_sha256") == receipt["parent_checkpoint_sha256"]
        and manifest.get("operation") == receipt["operation"]
        # R18-B3b (2026-09-07) review round: the manifest may name either
        # registered schema, but never a different one from the receipt inside
        # the checkpoint -- before schema/2 the single-value equality above made
        # a relabelled manifest impossible, and it must stay impossible.
        and manifest.get("schema") == receipt.get("schema")
        and manifest.get("ordinary_resume_eligible") is False
        and manifest.get("trained_in_target_world") is False
        and manifest.get("policy_sha256") == receipt["policy_sha256"],
        "Warm-start manifest makes false lineage/training claims")
    require((manifest.get("dive_blocker_recovery_recipe") == receipt["dive_blocker_recovery_recipe"])
        if "dive_blocker_recovery_recipe" in receipt else ("dive_blocker_recovery_recipe" not in manifest),
        "Warm-start manifest DIVE recovery recipe drift")
    require((manifest.get("depth_shaping_recipe") == receipt["depth_shaping_recipe"])
        if "depth_shaping_recipe" in receipt else ("depth_shaping_recipe" not in manifest),
        "Warm-start manifest depth signal recipe drift")
    require(target == manifest.get("target_contract")
        and receipt == manifest.get("receipt")
        and manifest.get("source_contract") == receipt["source_contract"],
        "Warm-start manifest/checkpoint contract or receipt drift")
    if expected_implementation is not None:
        require(target["implementation_sha256"] == expected_implementation,
                "Warm-start implementation differs from the active candidate")
    for name in ZERO_COUNTERS:
        require(type(data.get(name)) is int and data[name] == 0,
                f"Warm-start checkpoint is not zero-step: {name}")
    require(data.get("_last_completed_ppo_rollout_steps") is None,
            "Warm-start checkpoint claims a consumed rollout")
    return payload, data, manifest


def load_initialization(payload, manifest, *, env=None, seed=None):
    from leashed_ppo import LeashedMaskablePPO
    model = LeashedMaskablePPO.load(io.BytesIO(payload), env=env, device="cpu",
        teacher_path=None, teacher_sha256=None)
    validate_zero_state(model)
    require(policy_probe(model) == manifest["policy_probe"], "Warm-start policy probe drift")
    if seed is not None:
        model.set_random_seed(seed)
        model.seed = seed
    return model


def migrate(parent, *, parent_sha256, output_dir, implementation_sha256, seed,
            service_policy="sustain-v2", dive_blocker_recovery="off",
            worker_learning_window_scope=None, worker_prefix=None, worker_depth_shaping_unit=None):
    import torch
    from leashed_ppo import LeashedMaskablePPO
    from train_ppo import (_validate_checkpoint_bytes, _validate_resumable_leashed_boundary,
        _atomic_save_model, _implementation_bundle_sha256)
    require(parent_sha256 == PARENT_SHA256, "Only the exact registered R16 parent SHA is accepted")
    require(type(seed) is int and 0 <= seed < 2**32, "Explicit uint32 seed required")
    source_path = Path(parent).resolve(strict=True)
    destination = Path(output_dir).absolute()
    require(not destination.exists(), "Warm-start output directory already exists")
    payload = source_path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == PARENT_SHA256, "Parent checkpoint SHA mismatch")
    parent_data = _validate_resumable_leashed_boundary(_validate_checkpoint_bytes(
        payload, str(source_path), require_leashed=True))
    source = parent_data["diablogym_contract"]
    target = target_contract(source, implementation_sha256, service_policy, dive_blocker_recovery,
                             worker_learning_window_scope, worker_prefix, worker_depth_shaping_unit)
    require(_implementation_bundle_sha256() == implementation_sha256,
            "Target implementation changed before migration")
    torch.set_num_threads(1)
    model = LeashedMaskablePPO.load(io.BytesIO(payload), device="cpu",
                                  teacher_path=None, teacher_sha256=None)
    before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
    parameters_sha = policy_sha256(model)
    probe = policy_probe(model)
    receipt = make_receipt(parent_data, target, parameters_sha)
    reset_to_initialization(model, receipt, target, seed)
    for name, value in model.policy.state_dict().items():
        require(torch.equal(before[name], value.detach()), f"Changed policy tensor: {name}")
    require(policy_probe(model) == probe, "Reset changed fixed logits or critic values")
    destination.mkdir(parents=True, exist_ok=False)
    model_path = _atomic_save_model(model, destination / "model_warm_start.zip")
    manifest = {"schema": SCHEMA, "operation": receipt["operation"],
        "status": "INITIALIZATION_ONLY_NOT_TRAINED", "seed": seed,
        "parent_path": str(source_path), "parent_sha256": PARENT_SHA256,
        "source_contract": source, "target_contract": target, "receipt": receipt,
        "model_file": model_path.name, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "policy_tensor_count": len(before), "policy_sha256": parameters_sha,
        "policy_probe": probe, "ordinary_resume_eligible": False,
        "trained_in_target_world": False}
    if "dive_blocker_recovery_recipe" in receipt:
        manifest["dive_blocker_recovery_recipe"] = deepcopy(receipt["dive_blocker_recovery_recipe"])
    if "depth_shaping_recipe" in receipt:
        manifest["depth_shaping_recipe"] = deepcopy(receipt["depth_shaping_recipe"])
    restored = load_initialization(model_path.read_bytes(), manifest)
    for name, value in restored.policy.state_dict().items():
        require(torch.equal(before[name], value.detach()), f"Save/load changed policy tensor: {name}")
    require(hashlib.sha256(source_path.read_bytes()).hexdigest() == PARENT_SHA256,
            "Parent checkpoint changed during migration")
    require(_implementation_bundle_sha256() == implementation_sha256,
            "Target implementation changed during migration")
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    capture_initialization(destination / "manifest.json", implementation_sha256)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--implementation-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resource-service-policy", choices=tuple(OPERATIONS), default="sustain-v2")
    parser.add_argument("--dive-blocker-recovery", choices=("off", "adjacent-v1"), default="off")
    parser.add_argument("--worker-learning-window-scope",
                        choices=("farm-dive-v1", "earned-dive-suffix-v1"), default=None)
    parser.add_argument("--worker-prefix-model", default=None)
    parser.add_argument("--worker-prefix-max-attempts", type=int, default=None)
    parser.add_argument("--worker-prefix-max-microsteps", type=int, default=None)
    parser.add_argument("--worker-depth-shaping-unit", type=float, choices=(24.0,), default=None,
                        help="explicit earned/sustain-v6/adjacent-v1 depth-signal experiment; omitted preserves the old target")
    args = parser.parse_args()
    from eval_contract import worker_prefix_recipe, EARNED_DIVE_SUFFIX_SCOPE
    if args.worker_learning_window_scope == EARNED_DIVE_SUFFIX_SCOPE:
        require(args.worker_prefix_model is not None, "Earned suffix needs --worker-prefix-model")
        prefix_sha = hashlib.sha256(Path(args.worker_prefix_model).read_bytes()).hexdigest()
    else:
        prefix_sha = "specified" if args.worker_prefix_model is not None else None
    prefix = worker_prefix_recipe(args.worker_learning_window_scope, prefix_sha,
                                   args.worker_prefix_max_attempts, args.worker_prefix_max_microsteps)
    if args.worker_depth_shaping_unit is not None:
        operation_for(args.resource_service_policy, args.dive_blocker_recovery,
                      args.worker_learning_window_scope, args.worker_depth_shaping_unit)
    result = migrate(args.parent, parent_sha256=args.parent_sha256,
        output_dir=args.output_dir, implementation_sha256=args.implementation_sha256, seed=args.seed,
        service_policy=args.resource_service_policy, dive_blocker_recovery=args.dive_blocker_recovery,
        worker_learning_window_scope=args.worker_learning_window_scope, worker_prefix=prefix,
        worker_depth_shaping_unit=args.worker_depth_shaping_unit)
    print(json.dumps({key: result[key] for key in ("status", "model_sha256", "policy_sha256")}))


if __name__ == "__main__":
    main()
