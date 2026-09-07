"""Retain a completed but unpublishable update as a non-checkpoint archive.

This is intentionally not a save/resume or publication fallback. The nested SB3
bytes never have a standalone disk pathname, and ordinary readers reject the
outer ZIP. Callers retain responsibility for the original training exception.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile


_COUNTERS = {
    "num_timesteps": "num_timesteps",
    "last_completed_ppo_rollout_steps": "_last_completed_ppo_rollout_steps",
    "ppo_optimizer_steps_completed": "_ppo_optimizer_steps_completed",
    "actor_optimizer_steps_completed": "_actor_optimizer_steps_completed",
    "worker_onpolicy_pg_joint_rollouts": "_worker_onpolicy_pg_joint_rollouts",
    "worker_onpolicy_pg_qualifying_rollouts": "_worker_onpolicy_pg_qualifying_rollouts",
}
_METADATA = tuple(_COUNTERS.values()) + (
    "n_steps", "n_envs", "diablogym_contract", "_resource_warm_start_receipt",
    "_worker_onpolicy_pg_audit_required", "_worker_onpolicy_pg_pending_receipts",
    "_worker_onpolicy_pg_rollout_receipts",
)
_CHECKS = {"rollout_boundary", "exact_target", "migration", "worker_pg", "asymmetric"}
# Leashed deliberately does not persist its in-flight receipt queue. Its live
# emptiness is required above; do not alter the production serializer to save it.
_TRANSIENT_METADATA = {"_worker_onpolicy_pg_pending_receipts"}


def _require(condition, message):
    if not condition:
        raise ValueError("Training diagnostic refused: " + message)


def _json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _identities(value):
    """Notice replacement as well as mutation of the original receipt objects."""
    if isinstance(value, dict):
        return (id(value), tuple((key, _identities(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (id(value), tuple(_identities(item) for item in value))
    return None


def _snapshot(model):
    metadata = {name: getattr(model, name) for name in _METADATA}
    return {
        "metadata": deepcopy(metadata),
        "identities": _identities(metadata)[1],
        "rollout_buffer_id": id(model.rollout_buffer),
        "rollout_full": model.rollout_buffer.full,
        "calib_tripped": getattr(model, "_calib_tripped", False),
    }


def _unchanged(model, before, report, report_bytes):
    after = _snapshot(model)
    _require(_json(after) == _json(before), "serialization/validation mutated live bookkeeping")
    _require(_json(report) == report_bytes, "completion report changed during archival")


def _validate(model, report, expected_implementation, receipt_validator, before):
    _require(isinstance(expected_implementation, str)
             and len(expected_implementation) == 64
             and set(expected_implementation) <= set("0123456789abcdef"),
             "invalid implementation SHA256")
    _require(isinstance(report, dict)
             and report.get("schema") == "diablogym-training-completion/1"
             and report.get("publication_eligible") is False, "not a refused completion report")
    checks = report.get("checks")
    _require(isinstance(checks, dict) and set(checks) == _CHECKS
             and all(type(value) is bool for value in checks.values()), "invalid completion checks")
    _require(all(checks[name] for name in ("rollout_boundary", "exact_target", "migration"))
             and not (checks["worker_pg"] and checks["asymmetric"]),
             "failure is outside the completed diagnostic boundary")
    counters = report.get("counters")
    _require(isinstance(counters, dict), "missing completion counters")
    for name, attribute in _COUNTERS.items():
        value = getattr(model, attribute)
        _require(type(value) is int and value >= 0
                 and type(counters.get(name)) is int and counters[name] == value,
                 "live/report counter mismatch: " + name)
    target = counters.get("target_global_steps")
    _require(type(target) is int and target > 0 and model.num_timesteps == target
             and model._last_completed_ppo_rollout_steps == target,
             "not the exact consumed target")
    _require(before["rollout_full"] is True and before["calib_tripped"] is False,
             "partial rollout or calibration interruption")
    _require(type(model.n_steps) is int and type(model.n_envs) is int
             and model.n_steps > 0 and model.n_envs > 0, "invalid rollout quantum")
    quantum = model.n_steps * model.n_envs
    contract = model.diablogym_contract
    warm = model._resource_warm_start_receipt
    _require(isinstance(contract, dict)
             and contract.get("implementation_sha256") == expected_implementation,
             "training contract implementation mismatch")
    _require(isinstance(warm, dict)
             and warm.get("schema") == "diablogym-resource-warm-start/1",
             "missing resource warm-start lineage")
    # Reuse the production lineage/partition validation; it does not predict.
    model._assert_critic_migration_contract()
    receipts = model._worker_onpolicy_pg_rollout_receipts
    _require(model._worker_onpolicy_pg_audit_required is True
             and type(model._worker_onpolicy_pg_pending_receipts) is list
             and model._worker_onpolicy_pg_pending_receipts == []
             and type(receipts) is list and bool(receipts)
             and model._worker_onpolicy_pg_joint_rollouts == len(receipts),
             "unclosed formal receipt ledger")
    for index, receipt in enumerate(receipts, 1):
        _require(receipt_validator(receipt, expected_samples=quantum) is True,
                 "invalid original PG receipt")
        _require(receipt["rollout_end_timesteps"] == index * quantum
                 and receipt["transition_reward_samples"] == quantum
                 and receipt["gae_advantage_samples"] == quantum,
                 "receipts do not continuously cover fresh training")
    qualifying = sum(receipt["qualifies"] is True for receipt in receipts)
    updates = sum(receipt["optimizer_steps"] for receipt in receipts)
    _require(len(receipts) * quantum == target, "receipt samples do not bind target")
    _require(model._worker_onpolicy_pg_qualifying_rollouts == qualifying
             and checks["worker_pg"] is (qualifying > 0), "qualifying evidence/report mismatch")
    _require(updates > 0 and updates == model._ppo_optimizer_steps_completed
             == model._actor_optimizer_steps_completed,
             "optimizer counters do not bind completed joint receipts")


def archive_refused_training(model, run_dir, completion_report, *,
                             expected_implementation, checkpoint_validator,
                             receipt_validator):
    """Exclusively publish one diagnostic ZIP, or raise without a new artifact.

    The caller must first verify the implementation on disk. Validators are the
    existing checkpoint/PG validators, not replacement or relaxed checks. This
    function does not catch or replace the caller's training exception.
    """
    directory = Path(run_dir).resolve(strict=True)
    _require(directory.is_dir(), "run directory does not exist")
    target = directory / "training_diagnostic.zip"
    _require(not os.path.lexists(target), "diagnostic output already exists")
    before = _snapshot(model)
    report_bytes = _json(completion_report)
    _validate(model, completion_report, expected_implementation, receipt_validator, before)
    _unchanged(model, before, completion_report, report_bytes)
    memory = io.BytesIO()
    try:
        model.save(memory)
        payload = memory.getvalue()
        data = checkpoint_validator(payload, "training diagnostic nested checkpoint",
                                    require_leashed=True)
        _require(isinstance(data, dict), "checkpoint validator did not return metadata")
        for name, value in before["metadata"].items():
            if name in _TRANSIENT_METADATA and name not in data:
                continue
            _require(name in data and _json(data[name]) == _json(value),
                     "serialized/live metadata mismatch: " + name)
    finally:
        try:
            _unchanged(model, before, completion_report, report_bytes)
        finally:
            memory.close()
    receipt_bytes = _json(before["metadata"]["_worker_onpolicy_pg_rollout_receipts"])
    manifest = {
        "schema": "diablogym-refused-training-diagnostic/1",
        "status": "DIAGNOSTIC_ONLY_NOT_PUBLISHABLE",
        "artifact_scope": "diagnostic-only",
        "publication_eligible": False, "ordinary_resume_eligible": False,
        "ordinary_evaluation_eligible": False, "exact_trajectory_continuation": False,
        "implementation_sha256": expected_implementation,
        "checkpoint_member": "checkpoint/model.sb3.zip",
        "checkpoint_sha256": _sha(payload), "checkpoint_bytes": len(payload),
        "completion_failure_sha256": _sha(report_bytes),
        "receipts_sha256": _sha(receipt_bytes),
        "training_contract": before["metadata"]["diablogym_contract"],
        "resource_warm_start_receipt": before["metadata"]["_resource_warm_start_receipt"],
        "counters": deepcopy(completion_report["counters"]),
        "failed_publication_checks": sorted(name for name, passed in completion_report["checks"].items()
                                             if not passed),
        "source_mutated": False,
        "usage": "Forensic retention only; extracting the nested checkpoint does not certify publication or resume.",
    }
    manifest_bytes = _json(manifest)
    temporary = None
    linked = False
    try:
        fd, temporary = tempfile.mkstemp(prefix=".training-diagnostic-", suffix=".tmp", dir=directory)
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in (("manifest.json", manifest_bytes),
                                    ("completion_failure.json", report_bytes),
                                    ("receipts.json", receipt_bytes),
                                    ("checkpoint/model.sb3.zip", payload)):
                    archive.writestr(name, value)
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        _unchanged(model, before, completion_report, report_bytes)
        # link is atomic and fails if another process has claimed the target.
        os.link(temporary, target)
        linked = True
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        # Cleanup is part of the transaction: do not return a receipt and then
        # risk raising from an unprotected finally after publication.
        Path(temporary).unlink()
        temporary = None
        return {
            "schema": "diablogym-refused-training-archive-receipt/1",
            "status": manifest["status"], "archive_path": str(target),
            "archive_sha256": archive_sha, "manifest_sha256": _sha(manifest_bytes),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "completion_failure_sha256": manifest["completion_failure_sha256"],
            "receipts_sha256": manifest["receipts_sha256"],
            "implementation_sha256": expected_implementation,
            "counters": deepcopy(manifest["counters"]),
            "publication_eligible": False, "ordinary_resume_eligible": False,
            "ordinary_evaluation_eligible": False,
        }
    except BaseException:
        if linked and temporary is not None:
            # Never remove a target replaced/created by somebody else.
            if os.path.lexists(target) and os.path.samefile(temporary, target):
                target.unlink()
        raise
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                # An existing failure is already propagating. Never replace it
                # with a second cleanup error or touch another attempt's file.
                pass
