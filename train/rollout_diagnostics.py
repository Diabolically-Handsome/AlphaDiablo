"""Passive, single-update evidence capture; these ZIPs are not checkpoints."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
ARRAY_NAMES = ("last_values", "dones", "observations", "actions", "rewards",
               "episode_starts", "values", "log_probs", "advantages", "returns", "action_masks")
COUNTER_NAMES = ("num_timesteps", "_total_timesteps", "_last_completed_ppo_rollout_steps",
                 "_ppo_optimizer_steps_completed", "_actor_optimizer_steps_completed",
                 "_worker_onpolicy_pg_joint_rollouts", "_worker_onpolicy_pg_qualifying_rollouts",
                 "_n_updates")
STATUS = "DIAGNOSTIC_ONLY_NOT_PUBLISHABLE"


def _require(condition, message):
    if not condition:
        raise RuntimeError("First-update diagnostic: " + message)


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def policy_tensor_sha256(state):
    """Exactly the existing migrate_resource_candidate.policy_sha256 framing."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        header = json.dumps([name, str(array.dtype), list(array.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tree_bytes(value):
    if isinstance(value, torch.Tensor):
        _require(value.layout == torch.strided and bool(torch.isfinite(value).all()),
                 "nonfinite or unsupported state tensor")
        return value.numel() * value.element_size()
    if type(value) in (dict, OrderedDict):
        _require(all(type(key) in (str, int) for key in value), "unsupported state dictionary key")
        metadata_bytes = _tree_bytes(value._metadata) if hasattr(value, "_metadata") else 0
        return sum(_tree_bytes(item) for item in value.values()) + metadata_bytes
    if type(value) in (tuple, list):
        return sum(_tree_bytes(item) for item in value)
    _require(value is None or type(value) in (str, int, bool, float), "state tree contains non-data object")
    _require(type(value) is not float or math.isfinite(value), "nonfinite state scalar")
    return 0


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if type(value) in (dict, OrderedDict):
        result = type(value)((key, _clone_tree(item)) for key, item in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = deepcopy(value._metadata)
        return result
    if type(value) in (list, tuple):
        return type(value)(_clone_tree(item) for item in value)
    return value


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.detach().cpu()))
    if type(left) is not type(right):
        return False
    if type(left) in (dict, OrderedDict):
        return (left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
                and _tree_equal(getattr(left, "_metadata", None), getattr(right, "_metadata", None)))
    if type(left) in (tuple, list):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def _object_ids(value):
    if isinstance(value, (dict, list, tuple)):
        children = value.values() if isinstance(value, dict) else value
        return (id(value), tuple(_object_ids(item) for item in children))
    return id(value) if isinstance(value, (np.ndarray, torch.Tensor)) else None


def _torch_bytes(value):
    with io.BytesIO() as memory:
        torch.save(value, memory)
        return memory.getvalue()


def _inode(path):
    stat = Path(path).stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino


class FirstUpdateDiagnosticCallback(BaseCallback):
    """Capture one resource warm-start quantum, without sampling or optimization.

    Installation does not authorize training. The original training/publication
    gates remain in charge; even a COMPLETE diagnostic is not a candidate.
    """
    def __init__(self, run_dir, implementation_sha256):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.implementation_sha256 = implementation_sha256
        self._phase = "NEW"
        self._summary_payload = None
        self._summary_inode = None

    def _on_step(self):
        return True

    def _counters(self):
        return {name: getattr(self.model, name) for name in COUNTER_NAMES}

    def _zero(self, collected=False):
        from migrate_resource_candidate import ZERO_COUNTERS
        for name in ZERO_COUNTERS:
            value = getattr(self.model, name)
            if collected and name == "_episode_num":
                _require(type(value) is int and value >= 0, "invalid collected episode count")
                continue
            expected = self.quantum if name == "_total_timesteps" or (collected and name == "num_timesteps") else 0
            _require(type(value) is int and value == expected, "nonzero initial/update counter: " + name)
        _require(self.model._last_completed_ppo_rollout_steps is None,
                 "an optimizer-consumed rollout already exists")
        _require(self.model._worker_onpolicy_pg_rollout_receipts == [], "historical PG receipts")

    def _lineage(self):
        _require(self.model._worker_onpolicy_pg_audit_required is True, "formal PG audit is not required")
        _require(getattr(self.model, "_calib_tripped", False) is False, "calibration interrupted capture")
        _require(isinstance(self.model._resource_warm_start_receipt, dict), "resource warm-start required")
        contract = self.model.diablogym_contract
        _require(contract.get("artifact_scope") == "candidate"
                 and contract.get("implementation_sha256") == self.implementation_sha256,
                 "candidate/implementation contract mismatch")
        self.model._assert_critic_migration_contract()

    def _on_training_start(self):
        _require(self._phase == "NEW", "callback cannot be reused")
        try:
            _require(isinstance(self.implementation_sha256, str)
                     and len(self.implementation_sha256) == 64
                     and set(self.implementation_sha256) <= set("0123456789abcdef"), "invalid implementation SHA")
            self.run_dir = self.run_dir.resolve(strict=True)
            _require(self.run_dir.is_dir(), "run directory missing")
            for name in ("first_update_before.zip", "first_update_after.zip", "rollout_diagnostic.json"):
                _require(not os.path.lexists(self.run_dir / name), "output already exists: " + name)
            _require(type(self.model.n_steps) is int and type(self.model.n_envs) is int
                     and self.model.n_steps > 0 and self.model.n_envs > 0, "invalid rollout dimensions")
            self.shape = [self.model.n_steps, self.model.n_envs]
            self.quantum = self.model.n_steps * self.model.n_envs
            _require(type(self.model._total_timesteps) is int
                     and self.model._total_timesteps == self.quantum, "explicit total must be exactly one quantum")
            if "total_timesteps" in self.locals:
                _require(type(self.locals["total_timesteps"]) is int
                         and self.locals["total_timesteps"] == self.quantum, "learn target mismatch")
            self._zero()
            self._lineage()
            _require(self.model._worker_onpolicy_pg_pending_receipts == [], "pending initial receipts")
            policy = self.model.policy.state_dict()
            optimizer = self.model.policy.optimizer.state_dict()
            _require(_tree_bytes(policy) + _tree_bytes(optimizer) <= MAX_UNCOMPRESSED_BYTES, "state exceeds size cap")
            _require(optimizer.get("state") == {}, "initial optimizer is not empty")
            self._initial_policy_sha = policy_tensor_sha256(policy)
            _require(self.model._resource_warm_start_receipt.get("policy_sha256") == self._initial_policy_sha,
                     "initial policy does not match the inherited initialization")
            self._initial_optimizer = _clone_tree(optimizer)
            self._contract_bytes = _json(self.model.diablogym_contract)
            self._warm_bytes = _json(self.model._resource_warm_start_receipt)
            self._phase = "ARMED"
        except BaseException:
            self._phase = "FAILED"
            raise

    def _arrays(self, require_live):
        buffer = self.model.rollout_buffer
        snapshot = buffer._formal_gae_snapshot
        _require(type(snapshot) is dict and set(snapshot) == set(ARRAY_NAMES), "sealed snapshot schema mismatch")
        arrays = {}
        metadata = {}
        total = 0
        for name in ARRAY_NAMES:
            array = snapshot[name]
            _require(isinstance(array, np.ndarray) and array.dtype.kind in "biuf",
                     "invalid/object snapshot: " + name)
            total += array.nbytes
            _require(total <= MAX_UNCOMPRESSED_BYTES, "snapshot exceeds size cap")
            _require(bool(np.isfinite(array).all()), "nonfinite snapshot: " + name)
            if name in ("last_values", "dones"):
                _require(array.shape == (self.shape[1],), "tail shape mismatch: " + name)
            else:
                _require(array.shape[:2] == tuple(self.shape), "time-major shape mismatch: " + name)
                if name not in ("observations", "actions", "action_masks"):
                    _require(array.ndim == 2, "scalar rollout shape mismatch: " + name)
                if name == "observations":
                    _require(array.shape[2:] == tuple(self.model.diablogym_contract["observation_shape"]),
                             "observation shape differs from contract")
                if name == "actions":
                    _require(array.shape[2:] == (1,), "discrete action dimension mismatch")
                if name == "action_masks":
                    _require(array.shape[2:] == (self.model.diablogym_contract["action_n"],),
                             "mask dimension differs from contract")
                if require_live:
                    live = getattr(buffer, name)
                    _require(live.dtype == array.dtype and np.array_equal(live, array),
                             "live buffer differs from its sealed snapshot: " + name)
            arrays[name] = array
            metadata[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                              "nbytes": array.nbytes, "sha256": _sha(array.tobytes(order="C"))}
        return arrays, metadata, total

    def _runtime(self):
        _require(str(self.model.device) == "cpu", "only the registered CPU capture is supported")
        return {"torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "policy_training": self.model.policy.training, "device": "cpu",
                "gamma": self.model.gamma, "gae_lambda": self.model.gae_lambda,
                "normalize_advantage": self.model.normalize_advantage}

    def _capture(self, phase):
        from leashed_ppo import actor_parameter_sha256, validate_worker_onpolicy_pg_receipt
        before = phase == "before"
        _require(self._phase == ("ARMED" if before else "BEFORE_CAPTURED"), "invalid capture stage")
        _require(self._counters()["num_timesteps"] == self.quantum
                 and self.model._total_timesteps == self.quantum, "partial or excessive collection")
        _require(self.model.rollout_buffer.full is True, "rollout is not full")
        self._lineage()
        _require(_json(self.model.diablogym_contract) == self._contract_bytes
                 and _json(self.model._resource_warm_start_receipt) == self._warm_bytes, "lineage changed")
        arrays, array_metadata, array_bytes = self._arrays(require_live=before)
        pending = self.model._worker_onpolicy_pg_pending_receipts
        committed = self.model._worker_onpolicy_pg_rollout_receipts
        if before:
            self._zero(collected=True)
            _require(self.model.rollout_buffer.generator_ready is False, "rollout was flattened before capture")
            _require(type(pending) is list and len(pending) == self.quantum, "pending receipt count mismatch")
            actions = arrays["actions"].reshape(-1)
            _require(len(actions) == self.quantum and all(type(row.get("requested_action")) is int
                     and row["requested_action"] == action for row, action in zip(pending, actions)),
                     "pending receipts are not in raw time-major action order")
        else:
            _require(array_metadata == self._array_metadata, "sealed snapshot changed after update")
            _require(pending == [] and type(committed) is list and len(committed) == 1,
                     "update receipt is not fully committed")
            receipt = committed[0]
            _require(validate_worker_onpolicy_pg_receipt(receipt, expected_samples=self.quantum) is True,
                     "invalid canonical PG receipt")
            _require(receipt["rollout_end_timesteps"] == self.quantum
                     and receipt["collection_actor_sha256"] == self._before_actor_sha,
                     "update receipt does not bind captured collection")
            c = self._counters()
            _require(type(c["_last_completed_ppo_rollout_steps"]) is int
                     and c["_last_completed_ppo_rollout_steps"] == self.quantum
                     and type(c["_n_updates"]) is int and c["_n_updates"] > 0
                     and type(c["_ppo_optimizer_steps_completed"]) is int
                     and type(c["_actor_optimizer_steps_completed"]) is int
                     and c["_ppo_optimizer_steps_completed"] == c["_actor_optimizer_steps_completed"]
                     == receipt["optimizer_steps"] > 0
                     and type(c["_worker_onpolicy_pg_joint_rollouts"]) is int
                     and c["_worker_onpolicy_pg_joint_rollouts"] == 1
                     and type(c["_worker_onpolicy_pg_qualifying_rollouts"]) is int
                     and c["_worker_onpolicy_pg_qualifying_rollouts"] == int(receipt["qualifies"]),
                     "optimizer/receipt counters do not close")
        receipts = pending if before else committed
        receipt_bytes = _json(receipts)
        receipt_ids = _object_ids(receipts)
        array_ids = _object_ids(self.model.rollout_buffer._formal_gae_snapshot)
        counters = self._counters()
        runtime = self._runtime()
        policy_live = self.model.policy.state_dict()
        optimizer_live = self.model.policy.optimizer.state_dict()
        _require(array_bytes + _tree_bytes(policy_live) + _tree_bytes(optimizer_live)
                 + len(receipt_bytes) <= MAX_UNCOMPRESSED_BYTES, "capture exceeds size cap")
        policy = _clone_tree(policy_live)
        optimizer = _clone_tree(optimizer_live)
        policy_sha = policy_tensor_sha256(policy)
        if before:
            _require(policy_sha == self._initial_policy_sha and _tree_equal(optimizer, self._initial_optimizer),
                     "parameters/optimizer changed during collection")
        actor_sha = actor_parameter_sha256(self.model.policy, optimizer=self.model.policy.optimizer)
        members = {"state/policy.pt": _torch_bytes(policy), "state/optimizer.pt": _torch_bytes(optimizer),
                   "pending_receipts.json" if before else "committed_receipts.json": receipt_bytes}
        if before:
            with io.BytesIO() as memory:
                np.savez(memory, **{name: array.copy() for name, array in arrays.items()})
                members["gae_snapshot.npz"] = memory.getvalue()
        manifest = {"schema": "diablogym-first-update-diagnostic/1", "phase": phase, "status": STATUS,
                    "publication_eligible": False, "ordinary_resume_eligible": False,
                    "ordinary_evaluation_eligible": False, "exact_trajectory_continuation": False,
                    "implementation_sha256": self.implementation_sha256,
                    "training_contract": deepcopy(self.model.diablogym_contract),
                    "resource_warm_start_receipt": deepcopy(self.model._resource_warm_start_receipt),
                    "counters": counters, "rollout_shape": self.shape,
                    "array_order": "time-major", "flatten_order": "t*n_envs+env",
                    "runtime": runtime, "actor_parameter_sha256": actor_sha,
                    "policy_tensor_sha256": policy_sha,
                    "before_archive_sha256": None if before else self._before_archive["sha256"],
                    "arrays": array_metadata,
                    "members": {name: {"sha256": _sha(value), "bytes": len(value)} for name, value in members.items()}}
        members["manifest.json"] = _json(manifest)
        _require(sum(map(len, members.values())) <= MAX_UNCOMPRESSED_BYTES, "encoded capture exceeds size cap")
        with io.BytesIO() as memory:
            with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in members.items():
                    archive.writestr(name, value)
            payload = memory.getvalue()
        # Read-only postconditions, including receipt/array object identities.
        current_receipts = (self.model._worker_onpolicy_pg_pending_receipts if before
                            else self.model._worker_onpolicy_pg_rollout_receipts)
        _require(self._counters() == counters and _json(current_receipts) == receipt_bytes
                 and _object_ids(current_receipts) == receipt_ids
                 and self._runtime() == runtime
                 and _json(self.model.diablogym_contract) == self._contract_bytes
                 and _json(self.model._resource_warm_start_receipt) == self._warm_bytes
                 and _object_ids(self.model.rollout_buffer._formal_gae_snapshot) == array_ids
                 and self._arrays(require_live=before)[1] == array_metadata
                 and _tree_equal(policy, self.model.policy.state_dict())
                 and _tree_equal(optimizer, self.model.policy.optimizer.state_dict()), "capture mutated live data")
        evidence = {"path": str(self.run_dir / f"first_update_{phase}.zip"), "sha256": _sha(payload)}
        summary = {"schema": "diablogym-first-update-diagnostic-status/1",
                   "status": "BEFORE_CAPTURED" if before else "COMPLETE",
                   "implementation_sha256": self.implementation_sha256, "rollout_shape": self.shape,
                   "before": evidence if before else self._before_archive,
                   "after": None if before else evidence}
        self._publish(Path(evidence["path"]), payload, _json(summary))
        if before:
            self._before_archive, self._before_actor_sha = evidence, actor_sha
            self._array_metadata = array_metadata
        self._phase = summary["status"]

    def _publish(self, target, payload, summary_payload):
        summary = self.run_dir / "rollout_diagnostic.json"
        _require(not os.path.lexists(target), "archive already exists")
        if self._summary_payload is None:
            _require(not os.path.lexists(summary), "summary already exists")
        else:
            _require(summary.is_file() and _inode(summary) == self._summary_inode
                     and summary.read_bytes() == self._summary_payload, "summary ownership/bytes changed")
        temporary = []
        archive_inode = summary_inode = None
        backup = None
        def stage(data):
            fd, name = tempfile.mkstemp(prefix=".first-update-", suffix=".tmp", dir=self.run_dir)
            path = Path(name)
            temporary.append(path)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            return path
        try:
            archive_temp, summary_temp = stage(payload), stage(summary_payload)
            if self._summary_payload is not None:
                backup = stage(self._summary_payload)
            os.link(archive_temp, target)
            archive_inode = _inode(target)
            summary_inode = _inode(summary_temp)
            if self._summary_payload is None:
                os.link(summary_temp, summary)
            else:
                _require(_inode(summary) == self._summary_inode
                         and summary.read_bytes() == self._summary_payload, "summary changed during staging")
                os.replace(summary_temp, summary)
            fd = os.open(self.run_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            for path in temporary:  # backup is deliberately last.
                path.unlink(missing_ok=True)
            self._summary_payload, self._summary_inode = summary_payload, summary_inode
        except BaseException:
            if archive_inode is not None and os.path.lexists(target) and _inode(target) == archive_inode:
                target.unlink()
            if summary_inode is not None and os.path.lexists(summary) and _inode(summary) == summary_inode:
                if backup is not None and backup.exists():
                    os.replace(backup, summary)
                    self._summary_inode = _inode(summary)
                elif self._summary_payload is None:
                    summary.unlink()
            raise
        finally:
            for path in temporary:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _on_rollout_end(self):
        try:
            self._capture("before")
        except BaseException:
            self._phase = "FAILED"
            raise

    def _on_training_end(self):
        try:
            self._capture("after")
        except BaseException:
            self._phase = "FAILED"
            raise
