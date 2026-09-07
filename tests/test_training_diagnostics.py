"""Diagnostic packaging fixtures only: no environment, rollout or optimizer.

The fake saver emits finite Torch state dictionaries and synthetic bookkeeping;
these are engineering fixtures, never claimed as completed training evidence.
The real production checkpoint/evaluation readers test the archive boundary.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import training_diagnostics as diagnostic
from train_ppo import _validate_checkpoint_bytes
from eval_contract import checkpoint_num_timesteps_bytes, EvalContractError
from leashed_ppo import LeashedMaskablePPO, validate_worker_onpolicy_pg_receipt

IMPLEMENTATION = "a" * 64


class SyntheticSaver:
    def __init__(self):
        self.num_timesteps = self._last_completed_ppo_rollout_steps = 8
        self.n_steps, self.n_envs = 2, 2
        self._ppo_optimizer_steps_completed = self._actor_optimizer_steps_completed = 6
        self._worker_onpolicy_pg_joint_rollouts = 2
        self._worker_onpolicy_pg_qualifying_rollouts = 0
        self._worker_onpolicy_pg_audit_required = True
        self._worker_onpolicy_pg_pending_receipts = []
        self._worker_onpolicy_pg_rollout_receipts = [
            {"rollout_end_timesteps": end, "transition_reward_samples": 4,
             "gae_advantage_samples": 4, "optimizer_steps": 3, "qualifies": False}
            for end in (4, 8)]
        self.diablogym_contract = {"implementation_sha256": IMPLEMENTATION,
                                   "fixture": "synthetic-no-training"}
        self._resource_warm_start_receipt = {
            "schema": "diablogym-resource-warm-start/1", "fixture": "synthetic"}
        self.rollout_buffer = SimpleNamespace(full=True)
        self._calib_tripped = False
        self.save_calls = self.contract_calls = 0
        self.policy_value = self.adam_value = 1.0
        self.save_error = None
        self.metadata_mutation = None
        self.live_mutation = None
        self.duplicate_member = False
        self.lineage_error = False

    def _assert_critic_migration_contract(self):
        self.contract_calls += 1
        if self.lineage_error:
            raise ValueError("synthetic invalid lineage")

    def save(self, file):
        assert isinstance(file, io.BytesIO), "no ordinary SB3 disk pathname"
        self.save_calls += 1
        if self.save_error:
            file.write(b"partial serialization")
            raise self.save_error
        data = {name: deepcopy(getattr(self, name)) for name in diagnostic._METADATA
                if name not in diagnostic._TRANSIENT_METADATA}
        data["distill_beta"] = 0.0
        if self.metadata_mutation:
            self.metadata_mutation(data)
        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr("data", json.dumps(data))
            archive.writestr("_stable_baselines3_version",
                             importlib.metadata.version("stable-baselines3"))
            for name, value in (("policy.pth", self.policy_value),
                                ("policy.optimizer.pth", self.adam_value)):
                tensor = io.BytesIO()
                torch.save({"fixture_tensor": torch.tensor([value])}, tensor)
                archive.writestr(name, tensor.getvalue())
            if self.duplicate_member:
                archive.writestr("data", json.dumps(data))
        if self.live_mutation:
            self.live_mutation(self)


def report_for(model):
    return {"schema": "diablogym-training-completion/1", "publication_eligible": False,
            "checks": {"rollout_boundary": True, "exact_target": True, "migration": True,
                       "worker_pg": False, "asymmetric": True},
            "counters": {**{key: getattr(model, attribute)
                            for key, attribute in diagnostic._COUNTERS.items()},
                         "target_global_steps": 8}}


def synthetic_receipt_validator(receipt, *, expected_samples):
    # Deliberately not a substitute for the canonical production validator.
    return (set(receipt) == {"rollout_end_timesteps", "transition_reward_samples",
                            "gae_advantage_samples", "optimizer_steps", "qualifies"}
            and expected_samples == 4 and type(receipt["qualifies"]) is bool
            and type(receipt["optimizer_steps"]) is int and receipt["optimizer_steps"] > 0)


def archive(model, path, report=None, **kwargs):
    return diagnostic.archive_refused_training(
        model, path, report_for(model) if report is None else report,
        expected_implementation=IMPLEMENTATION,
        checkpoint_validator=kwargs.get("checkpoint_validator", _validate_checkpoint_bytes),
        receipt_validator=kwargs.get("receipt_validator", synthetic_receipt_validator))


def test_nested_structure_hashes_real_readers_and_no_input_rng_mutation(tmp_path):
    model = SyntheticSaver()
    report = report_for(model)
    before = diagnostic._snapshot(model)
    report_before = deepcopy(report)
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    calls = []

    def receipt_validator(receipt, *, expected_samples):
        calls.append((id(receipt), expected_samples))
        return synthetic_receipt_validator(receipt, expected_samples=expected_samples)

    receipt = archive(model, tmp_path, report, receipt_validator=receipt_validator)
    path = tmp_path / "training_diagnostic.zip"
    assert list(tmp_path.iterdir()) == [path]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == receipt["archive_sha256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        assert set(outer.namelist()) == {"manifest.json", "completion_failure.json",
                                        "receipts.json", "checkpoint/model.sb3.zip"}
        assert outer.testzip() is None
        manifest = json.loads(outer.read("manifest.json"))
        assert manifest["publication_eligible"] is False
        assert manifest["ordinary_resume_eligible"] is False
        assert manifest["ordinary_evaluation_eligible"] is False
        assert manifest["exact_trajectory_continuation"] is False
        assert manifest["artifact_scope"] == "diagnostic-only"
        assert manifest["training_contract"] == model.diablogym_contract
        for member, key in (("checkpoint/model.sb3.zip", "checkpoint_sha256"),
                            ("completion_failure.json", "completion_failure_sha256"),
                            ("receipts.json", "receipts_sha256")):
            assert hashlib.sha256(outer.read(member)).hexdigest() == manifest[key] == receipt[key]
        assert json.loads(outer.read("completion_failure.json")) == report
        assert json.loads(outer.read("receipts.json")) == model._worker_onpolicy_pg_rollout_receipts
        assert _validate_checkpoint_bytes(outer.read("checkpoint/model.sb3.zip"), "inner", True)["num_timesteps"] == 8
    with pytest.raises(ValueError, match="缺关键成员"):
        _validate_checkpoint_bytes(payload, "outer", True)
    with pytest.raises(EvalContractError, match="不可读"):
        checkpoint_num_timesteps_bytes(payload, "outer")
    assert model.save_calls == model.contract_calls == 1
    assert calls == [(id(item), 4) for item in model._worker_onpolicy_pg_rollout_receipts]
    assert diagnostic._snapshot(model) == before and report == report_before
    assert random.getstate() == rng[0]
    np_now = np.random.get_state()
    assert np_now[0] == rng[1][0] and np.array_equal(np_now[1], rng[1][1]) and np_now[2:] == rng[1][2:]
    assert torch.equal(torch.random.get_rng_state(), rng[2])


@pytest.mark.parametrize("worker_pg,asymmetric", [(False, False), (True, False)])
def test_asymmetric_failure_is_also_diagnostic(worker_pg, asymmetric, tmp_path):
    model = SyntheticSaver()
    if worker_pg:
        model._worker_onpolicy_pg_rollout_receipts[0]["qualifies"] = True
        model._worker_onpolicy_pg_qualifying_rollouts = 1
    report = report_for(model)
    report["checks"].update(worker_pg=worker_pg, asymmetric=asymmetric)
    assert archive(model, tmp_path, report)["publication_eligible"] is False


@pytest.mark.parametrize("mutation", [
    lambda m, r: r.update(schema="other"),
    lambda m, r: r.update(publication_eligible=True),
    lambda m, r: r["checks"].update(rollout_boundary=False),
    lambda m, r: r["checks"].update(exact_target=False),
    lambda m, r: r["checks"].update(migration=False),
    lambda m, r: r["checks"].update(asymmetric=0),
    lambda m, r: r["checks"].update(worker_pg=True),
    lambda m, r: r["counters"].update(target_global_steps=9),
    lambda m, r: r["counters"].update(num_timesteps=True),
    lambda m, r: setattr(m.rollout_buffer, "full", False),
    lambda m, r: setattr(m, "_calib_tripped", True),
    lambda m, r: setattr(m, "_last_completed_ppo_rollout_steps", 4),
    lambda m, r: setattr(m, "_actor_optimizer_steps_completed", 0),
    lambda m, r: setattr(m, "_worker_onpolicy_pg_pending_receipts", [{}]),
    lambda m, r: setattr(m, "_worker_onpolicy_pg_joint_rollouts", 1),
    lambda m, r: m._worker_onpolicy_pg_rollout_receipts[0].update(qualifies=True),
    lambda m, r: m._worker_onpolicy_pg_rollout_receipts[0].update(rollout_end_timesteps=8),
    lambda m, r: m._worker_onpolicy_pg_rollout_receipts[0].update(optimizer_steps=5),
    lambda m, r: m.diablogym_contract.update(implementation_sha256="b" * 64),
    lambda m, r: setattr(m, "lineage_error", True),
])
def test_rejected_incomplete_or_inconsistent_boundary_never_serializes(tmp_path, mutation):
    model = SyntheticSaver()
    report = report_for(model)
    mutation(model, report)
    with pytest.raises(ValueError):
        archive(model, tmp_path, report)
    assert model.save_calls == 0 and list(tmp_path.iterdir()) == []


def test_canonical_pg_validator_is_not_bypassed(tmp_path):
    model = SyntheticSaver()
    # Compact synthetic receipts intentionally fail the full production schema.
    with pytest.raises(ValueError, match="invalid original PG receipt"):
        archive(model, tmp_path, receipt_validator=validate_worker_onpolicy_pg_receipt)
    assert model.save_calls == 0 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("attribute", ["policy_value", "adam_value"])
def test_nan_tensors_rejected_by_original_validator_before_disk(tmp_path, attribute):
    model = SyntheticSaver()
    setattr(model, attribute, float("nan"))
    with pytest.raises(ValueError):
        archive(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_duplicate_inner_members_rejected(tmp_path):
    model = SyntheticSaver()
    model.duplicate_member = True
    with pytest.warns(UserWarning, match="Duplicate name"), pytest.raises(ValueError, match="重复"):
        archive(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", sorted(set(diagnostic._METADATA) - diagnostic._TRANSIENT_METADATA))
def test_serialized_metadata_cannot_drift_from_live(tmp_path, field):
    model = SyntheticSaver()
    model.metadata_mutation = lambda data: data.pop(field)
    with pytest.raises(ValueError):
        archive(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_transient_pending_matches_actual_leashed_exclusion_and_rejects_saved_nonempty(tmp_path):
    # Calling this pure method requires no model initialization or environment.
    excluded = LeashedMaskablePPO._excluded_save_params(object.__new__(LeashedMaskablePPO))
    assert set(excluded) & set(diagnostic._METADATA) == diagnostic._TRANSIENT_METADATA
    model = SyntheticSaver()
    model.metadata_mutation = lambda data: data.update(_worker_onpolicy_pg_pending_receipts=[{}])
    with pytest.raises(ValueError, match="serialized/live metadata mismatch"):
        archive(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_real_leashed_save_with_synthetic_metadata_and_no_model_initialization(tmp_path):
    # Exercise inherited SB3 save and Leashed exclusions, with no constructor,
    # policy forward, environment or optimizer update. Metadata remain fixtures.
    model = object.__new__(LeashedMaskablePPO)
    model.__dict__.update(SyntheticSaver().__dict__)
    model.distill_beta = 0.0
    model._assert_critic_migration_contract = lambda: None
    model.policy = torch.nn.Module()
    model.policy.register_parameter("fixture_parameter", torch.nn.Parameter(torch.ones(1)))
    model.policy.optimizer = SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": []})
    before = diagnostic._snapshot(model)
    receipt = archive(model, tmp_path)
    assert diagnostic._snapshot(model) == before
    with zipfile.ZipFile(receipt["archive_path"]) as outer:
        payload = outer.read("checkpoint/model.sb3.zip")
    data = _validate_checkpoint_bytes(payload, "actual SB3 serializer fixture", True)
    assert "_worker_onpolicy_pg_pending_receipts" not in data
    assert data["_worker_onpolicy_pg_rollout_receipts"] == model._worker_onpolicy_pg_rollout_receipts


def test_live_receipt_replacement_during_save_rejected(tmp_path):
    model = SyntheticSaver()
    model.live_mutation = lambda m: setattr(m, "_worker_onpolicy_pg_rollout_receipts",
                                          deepcopy(m._worker_onpolicy_pg_rollout_receipts))
    with pytest.raises(ValueError, match="mutated live"):
        archive(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_serialization_exception_leaves_no_disk_checkpoint(tmp_path):
    model = SyntheticSaver()
    model.save_error = RuntimeError("synthetic serializer failure")
    with pytest.raises(RuntimeError, match="synthetic serializer failure"):
        archive(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_existing_archive_or_dangling_symlink_is_never_overwritten(tmp_path):
    target = tmp_path / "training_diagnostic.zip"
    target.write_bytes(b"previous-owned-artifact")
    model = SyntheticSaver()
    with pytest.raises(ValueError, match="already exists"):
        archive(model, tmp_path)
    assert target.read_bytes() == b"previous-owned-artifact" and model.save_calls == 0
    target.unlink()  # This test's own file only.
    target.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="already exists"):
        archive(model, tmp_path)
    assert target.is_symlink() and model.save_calls == 0


def test_atomic_claim_race_preserves_other_output_and_cleans_own_temp(tmp_path, monkeypatch):
    real_link = diagnostic.os.link
    def competing_claim(source, target):
        Path(target).write_bytes(b"competitor")
        real_link(source, target)
    monkeypatch.setattr(diagnostic.os, "link", competing_claim)
    with pytest.raises(FileExistsError):
        archive(SyntheticSaver(), tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == {"training_diagnostic.zip": b"competitor"}


@pytest.mark.parametrize("fail_at", [1, 2])
def test_fsync_failure_removes_only_this_attempt(tmp_path, monkeypatch, fail_at):
    existing = tmp_path / ".training-diagnostic-not-ours.tmp"
    existing.write_bytes(b"keep")
    calls = []
    real_fsync = diagnostic.os.fsync
    def fsync(fd):
        calls.append(fd)
        if len(calls) == fail_at:
            raise OSError("synthetic fsync failure")
        real_fsync(fd)
    monkeypatch.setattr(diagnostic.os, "fsync", fsync)
    with pytest.raises(OSError, match="fsync failure"):
        archive(SyntheticSaver(), tmp_path)
    assert list(tmp_path.iterdir()) == [existing] and existing.read_bytes() == b"keep"


def test_first_temp_unlink_failure_after_link_rolls_back_final_archive(tmp_path, monkeypatch):
    real_unlink = Path.unlink
    failed = []
    def unlink(path, *args, **kwargs):
        if path.name.startswith(".training-diagnostic-") and not failed:
            assert (tmp_path / "training_diagnostic.zip").is_file()
            failed.append(path)
            raise OSError("synthetic temp unlink failure")
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="synthetic temp unlink failure"):
        archive(SyntheticSaver(), tmp_path)
    assert len(failed) == 1 and list(tmp_path.iterdir()) == []
