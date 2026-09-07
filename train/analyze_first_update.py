"""Offline, fixed-data first-update objectives. Never starts an environment or optimizer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
import zipfile

import numpy as np
import torch

CONTRACT_NULL_FIELD = "worker_descend_escrow_readiness_table"
SCHEMA = "diablogym-first-update-objective-analysis/1"
ARCHIVE_SCHEMA = "diablogym-first-update-diagnostic/1"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
ARRAY_NAMES = {"last_values", "dones", "observations", "actions", "rewards", "episode_starts",
               "values", "log_probs", "advantages", "returns", "action_masks"}
COUNTERS = {"num_timesteps", "_total_timesteps", "_last_completed_ppo_rollout_steps",
            "_ppo_optimizer_steps_completed", "_actor_optimizer_steps_completed",
            "_worker_onpolicy_pg_joint_rollouts", "_worker_onpolicy_pg_qualifying_rollouts", "_n_updates"}
RUNTIME = {"torch_num_threads", "torch_num_interop_threads", "policy_training", "device",
           "gamma", "gae_lambda", "normalize_advantage"}
MANIFEST_KEYS = {"schema", "phase", "status", "publication_eligible", "ordinary_resume_eligible",
                 "ordinary_evaluation_eligible", "exact_trajectory_continuation", "implementation_sha256",
                 "training_contract", "resource_warm_start_receipt", "counters", "rollout_shape",
                 "array_order", "flatten_order", "runtime", "actor_parameter_sha256",
                 "policy_tensor_sha256", "before_archive_sha256", "arrays", "members"}
PENDING_KEYS = {"requested_action", "executed_action", "combat_effect", "transition_reward",
                "worker_no_progress_timeout", "no_progress_timeout_base_failure_reward",
                "no_progress_timeout_additional_failure_reward", "no_progress_timeout_failure_reward",
                "expected_buffer_reward", "time_limit_bootstrap", "time_limit_bootstrap_delta"}


def require(condition, message):
    if not condition:
        raise ValueError("First-update analysis refused: " + message)


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def initialization_contract_equivalence(initialization, capture):
    """Only the inherited v1 table's absent/null spelling may differ; no mutation."""
    require(isinstance(initialization, dict) and isinstance(capture, dict),
            "initialization contract/lineage mismatch")
    require(initialization.get(CONTRACT_NULL_FIELD) is None
            and capture.get(CONTRACT_NULL_FIELD) is None,
            "initialization contract/lineage mismatch: non-null readiness table")
    initial_rest = {key: value for key, value in initialization.items() if key != CONTRACT_NULL_FIELD}
    captured_rest = {key: value for key, value in capture.items() if key != CONTRACT_NULL_FIELD}
    require(json_bytes(initial_rest) == json_bytes(captured_rest),
            "initialization contract/lineage mismatch")
    return {"rule": "only-worker-descend-escrow-readiness-table-absent-or-null-v1",
            "field": CONTRACT_NULL_FIELD,
            "applied": json_bytes(initialization) != json_bytes(capture),
            "initialization_field_present": CONTRACT_NULL_FIELD in initialization,
            "capture_field_present": CONTRACT_NULL_FIELD in capture,
            "initialization_contract_sha256": sha(json_bytes(initialization)),
            "capture_contract_sha256": sha(json_bytes(capture))}


def strict_json(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("First-update analysis refused: nonfinite JSON " + value)
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def tensor_digest(state):
    digest = hashlib.sha256()
    require(isinstance(state, dict) and bool(state), "empty/non-dict policy state")
    for name, tensor in sorted(state.items()):
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor), "non-tensor policy state")
        require(tensor.device.type == "cpu" and tensor.layout == torch.strided, "unsupported policy tensor")
        require(bool(torch.isfinite(tensor).all()), "nonfinite policy tensor")
        array = tensor.detach().contiguous().numpy()
        header = json.dumps([name, str(array.dtype), list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little")); digest.update(header); digest.update(array.tobytes())
    return digest.hexdigest()


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu" and value.layout == torch.strided
                and bool(torch.isfinite(value).all()), "nonfinite/unsupported optimizer tensor")
    elif isinstance(value, dict):
        require(all(type(k) in (str, int) for k in value), "unsupported optimizer key")
        for child in value.values(): finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value: finite_tree(child)
    else:
        require(value is None or type(value) in (bool, int, str)
                or (type(value) is float and math.isfinite(value)), "unsupported optimizer value")


def read_archive(path, phase):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    expected = {"manifest.json", "state/policy.pt", "state/optimizer.pt"}
    expected |= {"gae_snapshot.npz", "pending_receipts.json"} if phase == "before" else {"committed_receipts.json"}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == expected, "archive member schema/duplicates")
        require(sum(info.file_size for info in archive.infolist()) <= MAX_ARCHIVE_BYTES, "archive expanded size exceeds 512 MiB")
        contents = {name: archive.read(name) for name in names}
    manifest = strict_json(contents["manifest.json"])
    require(isinstance(manifest, dict) and set(manifest) == MANIFEST_KEYS, "manifest key schema")
    require(manifest["schema"] == ARCHIVE_SCHEMA and manifest["phase"] == phase
            and manifest["status"] == "DIAGNOSTIC_ONLY_NOT_PUBLISHABLE", "archive phase/status")
    for key in ("publication_eligible", "ordinary_resume_eligible", "ordinary_evaluation_eligible", "exact_trajectory_continuation"):
        require(manifest[key] is False, "archive makes publication/resume claim")
    require(manifest["array_order"] == "time-major" and manifest["flatten_order"] == "t*n_envs+env", "array ordering")
    require(manifest["rollout_shape"] == [512, 4], "expected one 512 x 4 rollout")
    for key in ("implementation_sha256", "actor_parameter_sha256", "policy_tensor_sha256"):
        require(is_sha(manifest[key]), "invalid manifest digest: " + key)
    require(isinstance(manifest["counters"], dict) and set(manifest["counters"]) == COUNTERS, "counter schema")
    require(isinstance(manifest["runtime"], dict) and set(manifest["runtime"]) == RUNTIME, "runtime schema")
    runtime = manifest["runtime"]
    require(runtime["device"] == "cpu" and type(runtime["policy_training"]) is bool
            and type(runtime["normalize_advantage"]) is bool, "runtime mode/device")
    for key in ("torch_num_threads", "torch_num_interop_threads"):
        require(type(runtime[key]) is int and runtime[key] > 0, "invalid thread count")
    for key in ("gamma", "gae_lambda"):
        require(type(runtime[key]) in (float, int) and 0 <= runtime[key] <= 1, "invalid GAE parameter")
    require(isinstance(manifest["members"], dict) and set(manifest["members"]) == expected - {"manifest.json"}, "member descriptors")
    for name, descriptor in manifest["members"].items():
        require(descriptor == {"sha256": sha(contents[name]), "bytes": len(contents[name])}, "member hash/size mismatch: " + name)
    policy = torch.load(io.BytesIO(contents["state/policy.pt"]), map_location="cpu", weights_only=True)
    require(tensor_digest(policy) == manifest["policy_tensor_sha256"], "policy tensor digest")
    optimizer = torch.load(io.BytesIO(contents["state/optimizer.pt"]), map_location="cpu", weights_only=True)
    finite_tree(optimizer)
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"}
            and isinstance(optimizer["state"], dict) and isinstance(optimizer["param_groups"], list), "optimizer state schema")
    receipt_name = "pending_receipts.json" if phase == "before" else "committed_receipts.json"
    receipts = strict_json(contents[receipt_name])
    require(isinstance(receipts, list), "receipt list required")
    return {"path": str(path), "sha256": sha(payload), "manifest": manifest,
            "contents": contents, "policy": policy, "optimizer": optimizer, "receipts": receipts}


def validate_arrays(before, after):
    with zipfile.ZipFile(io.BytesIO(before["contents"]["gae_snapshot.npz"])) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == {key + ".npy" for key in ARRAY_NAMES}, "NPZ member schema")
        require(sum(info.file_size for info in archive.infolist()) <= MAX_ARCHIVE_BYTES, "NPZ expanded size exceeds 512 MiB")
    with np.load(io.BytesIO(before["contents"]["gae_snapshot.npz"]), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    expected = {key: (512, 4) for key in ARRAY_NAMES}
    expected.update(observations=(512, 4, 13012), actions=(512, 4, 1), action_masks=(512, 4, 15),
                    last_values=(4,), dones=(4,))
    descriptors = {}
    for key, array in arrays.items():
        require(array.shape == expected[key] and array.dtype.kind in "bifu" and np.isfinite(array).all(), "array shape/type/finite: " + key)
        if key == "actions":
            require(array.dtype.kind in "iu" or array.dtype == np.float32, "unsupported action dtype")
            require(np.equal(array, np.floor(array)).all() and ((array >= 0) & (array < 15)).all(), "actions must be exact integers in range")
        elif key == "dones": require(array.dtype == np.bool_, "dones must be boolean")
        else: require(array.dtype == np.float32, "expected float32: " + key)
        descriptors[key] = {"shape": list(array.shape), "dtype": str(array.dtype), "nbytes": array.nbytes,
                            "sha256": sha(array.tobytes(order="C"))}
    require(before["manifest"]["arrays"] == descriptors == after["manifest"]["arrays"], "sealed array descriptors differ")
    for key in ("action_masks", "episode_starts"):
        require(np.isin(arrays[key], [0, 1]).all(), "nonbinary " + key)
    actions = arrays["actions"].reshape(-1).astype(np.int64)
    masks = arrays["action_masks"].reshape(-1, 15)
    require(((0 <= actions) & (actions < 15)).all() and masks.any(1).all()
            and masks[np.arange(2048), actions].all(), "illegal requested action/mask")
    rows = before["receipts"]
    require(len(rows) == 2048, "pending receipt count")
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == PENDING_KEYS, "pending receipt schema")
        require(type(row["requested_action"]) is int and row["requested_action"] == int(actions[index]), "time-major action/receipt mismatch")
        executed = row["executed_action"]
        require(executed is None or (type(executed) is int and executed == row["requested_action"]), "requested/executed mismatch")
        for key in ("combat_effect", "worker_no_progress_timeout", "time_limit_bootstrap"):
            require(type(row[key]) is bool, "receipt boolean: " + key)
        require(not row["combat_effect"] or executed == 9, "combat effect without executed a9")
        for key in PENDING_KEYS - {"requested_action", "executed_action", "combat_effect", "worker_no_progress_timeout", "time_limit_bootstrap"}:
            require(type(row[key]) in (float, int) and math.isfinite(row[key]), "receipt finite number: " + key)
        expected_reward = np.float32(row["expected_buffer_reward"])
        # Mirror the original forward subtraction; its rounded delta is not invertible.
        expected_delta = np.float32(expected_reward - row["transition_reward"])
        require(float(expected_delta) == row["time_limit_bootstrap_delta"], "bootstrap delta generation mismatch")
        require(row["time_limit_bootstrap"] or expected_reward == np.float32(row["transition_reward"]),
                "non-bootstrap reward mismatch")
    require(np.array_equal(arrays["rewards"].reshape(-1), np.asarray([x["expected_buffer_reward"] for x in rows], dtype=np.float32)), "sealed reward/receipt mismatch")
    runtime = before["manifest"]["runtime"]
    advantages = np.zeros_like(arrays["advantages"])
    tail = 0
    for t in reversed(range(512)):
        nonterminal = 1.0 - (arrays["dones"].astype(np.float32) if t == 511 else arrays["episode_starts"][t+1])
        next_values = arrays["last_values"] if t == 511 else arrays["values"][t+1]
        delta = arrays["rewards"][t] + runtime["gamma"] * next_values * nonterminal - arrays["values"][t]
        tail = delta + runtime["gamma"] * runtime["gae_lambda"] * nonterminal * tail
        advantages[t] = tail
    require(np.array_equal(advantages, arrays["advantages"])
            and np.array_equal(advantages + arrays["values"], arrays["returns"]), "sealed GAE/return recurrence mismatch")
    return arrays


def validate_pair(before, after, receipt_validator):
    pre, post = before["manifest"], after["manifest"]
    require(pre["runtime"]["policy_training"] is False, "collection policy was not in evaluation mode")
    require(pre["before_archive_sha256"] is None and post["before_archive_sha256"] == before["sha256"], "before/after archive binding")
    for key in ("implementation_sha256", "training_contract", "resource_warm_start_receipt", "rollout_shape"):
        require(json_bytes(pre[key]) == json_bytes(post[key]), "pre/post identity mismatch: " + key)
    for key in RUNTIME - {"policy_training"}:
        require(pre["runtime"][key] == post["runtime"][key], "pre/post runtime mismatch: " + key)
    contract = pre["training_contract"]
    require(isinstance(contract, dict) and contract.get("implementation_sha256") == pre["implementation_sha256"]
            and contract.get("n_steps") == 512 and contract.get("num_envs") == 4
            and contract.get("observation_shape") == [13012] and contract.get("action_n") == 15, "contract geometry/implementation")
    recipe = contract.get("algorithm_recipe")
    require(isinstance(recipe, dict), "missing algorithm recipe")
    clip = recipe.get("clip_range")
    require(type(clip) in (int, float) and 0 < clip < 1, "fixed numeric clip_range required")
    pc, ac = pre["counters"], post["counters"]
    require(pc["_last_completed_ppo_rollout_steps"] is None, "before already consumed")
    for key in COUNTERS - {"_last_completed_ppo_rollout_steps"}:
        require(type(pc[key]) is int and type(ac[key]) is int, "integer counters required")
    for counters in (pc, ac):
        require(counters["num_timesteps"] == counters["_total_timesteps"] == 2048, "not a single 2048-sample trial")
    for key in COUNTERS - {"num_timesteps", "_total_timesteps", "_last_completed_ppo_rollout_steps"}:
        require(pc[key] == 0, "before is not zero-update")
    require(type(ac["_last_completed_ppo_rollout_steps"]) is int and ac["_last_completed_ppo_rollout_steps"] == 2048, "post not completed")
    require(len(after["receipts"]) == 1, "exactly one committed receipt required")
    receipt = after["receipts"][0]
    require(receipt_validator(receipt, expected_samples=2048) is True, "original formal receipt rejected")
    require(receipt["rollout_end_timesteps"] == 2048 and receipt["collection_actor_sha256"] == pre["actor_parameter_sha256"], "collection actor/receipt binding")
    require(ac["_worker_onpolicy_pg_joint_rollouts"] == 1
            and ac["_worker_onpolicy_pg_qualifying_rollouts"] == int(receipt["qualifies"])
            and ac["_ppo_optimizer_steps_completed"] == ac["_actor_optimizer_steps_completed"] == receipt["optimizer_steps"] > 0
            and ac["_n_updates"] > 0, "post counters/receipt mismatch")
    require(before["optimizer"]["state"] == {} and bool(after["optimizer"]["state"]), "optimizer before/after state boundary")
    pre_groups = before["optimizer"]["param_groups"]
    post_groups = after["optimizer"]["param_groups"]
    require(len(pre_groups) == len(post_groups) > 0, "optimizer groups")
    require([x["params"] for x in pre_groups] == [x["params"] for x in post_groups], "optimizer parameter mapping drift")
    return receipt


def bind_receipt_data(receipt, arrays, pending):
    rewards = np.asarray([x["transition_reward"] for x in pending], dtype=np.float64)
    combat = np.asarray([x["combat_effect"] for x in pending], dtype=bool)
    advantages = arrays["advantages"].reshape(-1).astype(np.float64)
    requested = arrays["actions"].reshape(-1).astype(np.int64)
    executed = [x["executed_action"] for x in pending if x["executed_action"] is not None]
    combat_rewards = rewards[combat]
    checks = {
        "transition_reward_samples": len(rewards),
        "transition_reward_nonzero_samples": int(np.count_nonzero(rewards)),
        "transition_reward_positive_samples": int(np.count_nonzero(rewards > 0)),
        "transition_reward_negative_samples": int(np.count_nonzero(rewards < 0)),
        "transition_reward_sum": float(rewards.sum()),
        "transition_reward_abs_sum": float(np.abs(rewards).sum()),
        "transition_reward_mean": float(rewards.mean()),
        "transition_reward_variance": float(np.mean(np.square(rewards-rewards.mean()))),
        "combat_effect_samples": int(combat.sum()),
        "combat_transition_reward_nonzero_samples": int(np.count_nonzero(combat_rewards)),
        "combat_transition_reward_positive_samples": int(np.count_nonzero(combat_rewards > 0)),
        "combat_transition_reward_negative_samples": int(np.count_nonzero(combat_rewards < 0)),
        "combat_transition_reward_sum": float(combat_rewards.sum()),
        "combat_transition_reward_abs_sum": float(np.abs(combat_rewards).sum()),
        "combat_positive_advantage_samples": int(np.count_nonzero(combat & (advantages > 0))),
        "gae_advantage_samples": len(advantages),
        "gae_advantage_nonzero_samples": int(np.count_nonzero(advantages)),
        "gae_advantage_mean": float(advantages.mean()),
        "gae_advantage_variance": float(np.mean(np.square(advantages-advantages.mean()))),
        "requested_action_counts": np.bincount(requested, minlength=15).tolist(),
        "executed_action_counts": np.bincount(executed, minlength=15).tolist(),
        "gae_recomputed_max_abs_delta": 0.0,
        "return_recomputed_max_abs_delta": 0.0,
    }
    for key, value in checks.items():
        require(receipt.get(key) == value, "committed receipt does not bind captured data: " + key)
    return checks


def _production_api():
    from leashed_ppo import LeashedMaskablePPO, actor_parameter_sha256, validate_worker_onpolicy_pg_receipt
    from migrate_resource_candidate import validate_zero_state
    from train_ppo import _implementation_bundle_sha256, _validate_checkpoint_bytes
    def loader(payload):
        _validate_checkpoint_bytes(payload, "explicit first-update initialization", require_leashed=True)
        model = LeashedMaskablePPO.load(io.BytesIO(payload), env=None, device="cpu", teacher_path=None, teacher_sha256=None)
        validate_zero_state(model)
        return model
    return loader, actor_parameter_sha256, validate_worker_onpolicy_pg_receipt, _implementation_bundle_sha256


@contextmanager
def private_runtime(threads):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng, previous_threads = torch.get_rng_state().clone(), torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        yield
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(torch_rng)
        torch.set_num_threads(previous_threads)


def infer(policy, arrays, chunk_rows=4):
    require(chunk_rows == 4, "fixed inference chunk is four collection environments")
    obs = arrays["observations"].reshape(-1, 13012)
    actions = arrays["actions"].reshape(-1).astype(np.int64)
    masks = arrays["action_masks"].reshape(-1, 15).astype(bool)
    outputs = {key: [] for key in ("log_probs", "entropy", "values")}
    state_before = tensor_digest(policy.state_dict())
    rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        for start in range(0, len(actions), chunk_rows):
            values, log_probs, entropy = policy.evaluate_actions(
                torch.as_tensor(obs[start:start+chunk_rows]),
                torch.as_tensor(actions[start:start+chunk_rows]).long(),
                action_masks=masks[start:start+chunk_rows])
            require(entropy is not None, "exact masked entropy required")
            for key, value in (("values", values), ("log_probs", log_probs), ("entropy", entropy)):
                value = value.detach().cpu().reshape(-1)
                require(value.shape == (chunk_rows,) and value.dtype == torch.float32
                        and bool(torch.isfinite(value).all()), "inference output geometry/finite")
                outputs[key].append(value.clone())
    require(torch.equal(rng_before, torch.get_rng_state()), "inference consumed Torch RNG")
    require(state_before == tensor_digest(policy.state_dict()), "inference mutated policy state")
    return {key: torch.cat(values) for key, values in outputs.items()}


def objectives(outputs, arrays, pending, clip_range):
    advantages = torch.as_tensor(arrays["advantages"].reshape(-1).copy())
    require(advantages.numel() > 1, "sample standard deviation requires N > 1")
    # One fixed whole-rollout baseline/std for both policies, unlike training minibatches.
    normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=True) + 1e-8)
    old_log_probs = torch.as_tensor(arrays["log_probs"].reshape(-1).copy())
    ratio = torch.exp(outputs["log_probs"] - old_log_probs)
    clipped = torch.clamp(ratio, 1-clip_range, 1+clip_range)
    raw_rewards = np.asarray([row["transition_reward"] for row in pending], dtype=np.float64)
    combat = np.asarray([row["combat_effect"] for row in pending], dtype=bool)
    q = np.where(combat, raw_rewards, 0.0)
    centered = torch.as_tensor(q - q.mean(), dtype=outputs["log_probs"].dtype)
    returns = torch.as_tensor(arrays["returns"].reshape(-1).copy())
    result = {
        "gae_clipped_surrogate_loss": float(-torch.minimum(normalized*ratio, normalized*clipped).mean()),
        "combat_centered_reference_loss": float(-(centered*outputs["log_probs"]).mean()),
        "entropy_mean": float(outputs["entropy"].mean()),
        "value_mse_fixed_returns": float(torch.square(outputs["values"]-returns).mean()),
        "ratio_mean": float(ratio.mean()),
        "clip_fraction": float((torch.abs(ratio-1) > clip_range).float().mean()),
    }
    require(all(math.isfinite(x) for x in result.values()), "nonfinite objective")
    return result


def analyze(before_path, after_path, initialization_path):
    before, after = read_archive(before_path, "before"), read_archive(after_path, "after")
    loader, actor_digest, validator, implementation_digest = _production_api()
    receipt = validate_pair(before, after, validator)
    arrays = validate_arrays(before, after)
    bind_receipt_data(receipt, arrays, before["receipts"])
    pre = before["manifest"]
    require(implementation_digest() == pre["implementation_sha256"], "local implementation differs from archives")
    init_path = Path(initialization_path).resolve(strict=True)
    require(init_path.name == "model_warm_start.zip", "explicit initialization ZIP filename required")
    payload = init_path.read_bytes()
    runtime = pre["runtime"]
    require(torch.get_num_interop_threads() == runtime["torch_num_interop_threads"], "interop thread context mismatch")
    with private_runtime(runtime["torch_num_threads"]):
        model = loader(payload)  # exactly one regular Leashed load; env=None, no optimizer step
        require(model.get_env() is None and str(model.device) == "cpu", "offline model must have no environment")
        require(model.observation_space.shape == (13012,) and model.action_space.n == 15, "loaded model geometry")
        contract_equivalence = initialization_contract_equivalence(model.diablogym_contract, pre["training_contract"])
        require(json_bytes(model._resource_warm_start_receipt) == json_bytes(pre["resource_warm_start_receipt"]), "initialization contract/lineage mismatch")
        require(tensor_digest(model.policy.state_dict()) == pre["policy_tensor_sha256"], "initialization differs from collection policy")
        require(model.policy.optimizer.state_dict()["state"] == {}, "initialization optimizer is not empty")
        require(float(model.gamma) == runtime["gamma"] and float(model.gae_lambda) == runtime["gae_lambda"]
                and model.normalize_advantage is runtime["normalize_advantage"], "initialization GAE runtime drift")
        model.policy.set_training_mode(False)
        results = {}
        for name, archive in (("before", before), ("after", after)):
            model.policy.load_state_dict(archive["policy"], strict=True)
            require(tensor_digest(model.policy.state_dict()) == archive["manifest"]["policy_tensor_sha256"], "strict policy load drift")
            require(actor_digest(model.policy, optimizer=model.policy.optimizer) == archive["manifest"]["actor_parameter_sha256"], "actor tensor digest mismatch")
            output = infer(model.policy, arrays)
            if name == "before":
                observed = output["log_probs"].numpy().astype(np.float64)
                old = arrays["log_probs"].reshape(-1).astype(np.float64)
                max_delta = float(np.max(np.abs(observed-old)))
                require(np.allclose(observed, old, rtol=1e-6, atol=1e-7), "initial policy/collection log-prob mismatch")
            results[name] = objectives(output, arrays, before["receipts"], pre["training_contract"]["algorithm_recipe"]["clip_range"])
        require(model.policy.optimizer.state_dict()["state"] == {} and model.num_timesteps == 0,
                "analysis modified optimizer/training counters")
    results["delta"] = {key: results["after"][key]-results["before"][key] for key in results["before"]}
    return {"schema": SCHEMA, "status": "PASS_DIAGNOSTIC_ONLY", "actual_samples": 2048,
            "bindings": {"before_sha256": before["sha256"], "after_sha256": after["sha256"],
                         "initialization_sha256": sha(payload), "implementation_sha256": pre["implementation_sha256"],
                         "training_contract_sha256": sha(json_bytes(pre["training_contract"])),
                         "collection_actor_sha256": pre["actor_parameter_sha256"],
                         "after_actor_sha256": after["manifest"]["actor_parameter_sha256"],
                         "committed_receipt_sha256": sha(json_bytes(receipt))},
            "initialization_contract_equivalence": contract_equivalence,
            "committed_receipt": receipt, "objectives": results,
            "objective_definition": {"gae": "fixed-full-rollout-normalized-clipped-surrogate-v1",
                                     "normalization": "float32-whole-rollout-mean-and-torch-unbiased-std-plus-1e-8",
                                     "normalization_scope": "fixed-once-for-both-policies-not-training-minibatches",
                                     "combat": "negative-mean-centered-combat-effect-times-raw-transition-reward-logprob",
                                     "value_target": "fixed-original-GAE-returns", "delta": "after-minus-before"},
            "inference": {"chunk_rows": 4, "array_order": "time-major", "device": "cpu", "training": False,
                          "torch_num_threads": runtime["torch_num_threads"], "torch_num_interop_threads": runtime["torch_num_interop_threads"]},
            "logprob_closure": {"rtol": 1e-6, "atol": 1e-7, "max_abs_delta": max_delta},
            "gae_closure": {"advantage_max_abs_delta": 0.0, "return_max_abs_delta": 0.0},
            "farm_dive_breakdown": {"available": False, "reason": "Original pending receipts have no certified FARM/DIVE label; not inferred from actions/masks."},
            "limits": {"model_loads": 1, "environment_initializations": 0, "optimizer_steps": 0, "random_action_samples": 0},
            "limitations": ["Same collected data only; no win-rate or generalization conclusion.",
                            "Full-rollout normalization differs from actual per-minibatch PPO normalization.",
                            "Combat reference is centered immediate transition reward, not the complete long-horizon objective.",
                            "No causal separation of Adam, entropy, clipping, or GAE; no automatic gate/recipe changes.",
                            "PASS describes diagnostic integrity, not publication qualification or objective improvement."]}


def write_output(path, result):
    path = Path(path).absolute()
    require(path.parent.is_dir() and not os.path.lexists(path), "output parent absent or output already exists")
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".first-update-analysis-", dir=path.parent)
    linked = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path); linked = True
        Path(temporary).unlink(); temporary = None
    except BaseException:
        if linked and temporary is not None and os.path.samefile(temporary, path): path.unlink()
        raise
    finally:
        if temporary is not None:
            try: Path(temporary).unlink(missing_ok=True)
            except OSError: pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "initialization", "output"): parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    require(not os.path.lexists(args.output), "output already exists")
    result = analyze(args.before, args.after, args.initialization)
    write_output(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).absolute())}))


if __name__ == "__main__":
    main()
