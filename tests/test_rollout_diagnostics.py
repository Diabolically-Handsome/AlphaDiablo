"""Synthetic callback transitions only; no model/environment/optimizer execution."""
from copy import deepcopy
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
import rollout_diagnostics as capture
import leashed_ppo
from migrate_resource_candidate import ZERO_COUNTERS
from eval_contract import checkpoint_num_timesteps_bytes, EvalContractError

IMPLEMENTATION = "b" * 64


class FixtureOptimizer:
    def __init__(self):
        self.payload = {"state": {}, "param_groups": [{"params": [0], "lr": 0.01}]}

    def state_dict(self):
        return deepcopy(self.payload)


def fixture_model():
    policy = torch.nn.Module()
    policy.register_parameter("fixture", torch.nn.Parameter(torch.ones(2)))
    policy.optimizer = FixtureOptimizer()
    policy.train(False)
    model = SimpleNamespace(**{name: 0 for name in ZERO_COUNTERS})
    model.policy = policy
    model.n_steps, model.n_envs, model._total_timesteps = 2, 2, 4
    model._last_completed_ppo_rollout_steps = None
    model._worker_onpolicy_pg_pending_receipts = []
    model._worker_onpolicy_pg_rollout_receipts = []
    model._worker_onpolicy_pg_audit_required = True
    model._worker_onpolicy_pg_collection_actor_sha256 = None
    model._calib_tripped = False
    model.diablogym_contract = {"artifact_scope": "candidate", "implementation_sha256": IMPLEMENTATION,
                               "observation_shape": [3], "action_n": 15, "fixture": "not-training"}
    model._resource_warm_start_receipt = {"schema": "diablogym-resource-warm-start/1",
        "policy_sha256": capture.policy_tensor_sha256(policy.state_dict()), "fixture": "not-training"}
    model._assert_critic_migration_contract = lambda: None
    model.device = "cpu"
    model.gamma, model.gae_lambda, model.normalize_advantage = 1.0, 0.95, True
    model.rollout_buffer = SimpleNamespace(full=False, generator_ready=False, _formal_gae_snapshot=None)
    return model


def collect_fixture(model):
    # Distinct T/E values make accidental env-major transposition observable.
    scalar = np.arange(4, dtype=np.float32).reshape(2, 2)
    arrays = {name: scalar.copy() for name in capture.ARRAY_NAMES}
    arrays.update(last_values=np.array([9, 10], dtype=np.float32),
                  dones=np.array([False, True]), observations=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
                  actions=scalar.reshape(2, 2, 1).copy(), action_masks=np.ones((2, 2, 15), dtype=np.float32))
    model.rollout_buffer._formal_gae_snapshot = arrays
    for name, array in arrays.items():
        if name not in ("last_values", "dones"):
            setattr(model.rollout_buffer, name, array.copy())
    model.rollout_buffer.full = True
    model.num_timesteps = 4
    model._worker_onpolicy_pg_pending_receipts = [
        {"requested_action": i, "transition_reward": float(i), "fixture_time_env": [i // 2, i % 2]}
        for i in range(4)]


def consumed_fixture(model, before_sha):
    # Manually assigned synthetic endpoint, never an optimizer step.
    with torch.no_grad():
        model.policy.fixture.add_(0.125)
    model.policy.optimizer.payload["state"] = {0: {"step": torch.tensor(3.),
        "exp_avg": torch.zeros(2), "exp_avg_sq": torch.ones(2)}}
    model.policy.train(True)
    model._worker_onpolicy_pg_pending_receipts = []
    model._worker_onpolicy_pg_rollout_receipts = [{"rollout_end_timesteps": 4,
        "transition_reward_samples": 4, "gae_advantage_samples": 4, "optimizer_steps": 3,
        "qualifies": False, "collection_actor_sha256": before_sha}]
    model._worker_onpolicy_pg_joint_rollouts = 1
    model._worker_onpolicy_pg_qualifying_rollouts = 0
    model._ppo_optimizer_steps_completed = model._actor_optimizer_steps_completed = 3
    model._last_completed_ppo_rollout_steps = 4
    model._n_updates = 2
    model.rollout_buffer.generator_ready = True
    # SB3 may flatten its live arrays; the sealed arrays remain time-major.
    model.rollout_buffer.observations = model.rollout_buffer.observations.reshape(4, 3)


@pytest.fixture
def callback(tmp_path, monkeypatch):
    monkeypatch.setattr(leashed_ppo, "actor_parameter_sha256",
        lambda policy, optimizer=None: capture.policy_tensor_sha256(policy.state_dict()))
    monkeypatch.setattr(leashed_ppo, "validate_worker_onpolicy_pg_receipt",
        lambda receipt, expected_samples=None: type(receipt) is dict
        and receipt.get("transition_reward_samples") == expected_samples
        and type(receipt.get("qualifies")) is bool and receipt.get("optimizer_steps", 0) > 0)
    cb = capture.FirstUpdateDiagnosticCallback(tmp_path, IMPLEMENTATION)
    cb.init_callback(fixture_model())
    return cb


def start(cb):
    cb.on_training_start({"total_timesteps": 4}, {})


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    for name, evidence in manifest["members"].items():
        assert evidence == {"sha256": capture._sha(members[name]), "bytes": len(members[name])}
    assert sum(map(len, members.values())) <= capture.MAX_UNCOMPRESSED_BYTES
    return manifest, members


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def assert_rng_same(previous):
    assert random.getstate() == previous[0]
    now = np.random.get_state()
    assert now[0] == previous[1][0] and np.array_equal(now[1], previous[1][1]) and now[2:] == previous[1][2:]
    assert torch.equal(torch.random.get_rng_state(), previous[2])


def test_complete_pair_raw_time_major_state_weights_only_and_rng_transparency(callback, tmp_path):
    start(callback)
    model = callback.model
    collect_fixture(model)
    arrays = model.rollout_buffer._formal_gae_snapshot
    copies = {name: array.copy() for name, array in arrays.items()}
    ids = capture._object_ids(arrays), capture._object_ids(model._worker_onpolicy_pg_pending_receipts)
    rng = rng_snapshot()
    callback.on_rollout_end()
    assert_rng_same(rng)
    assert capture._object_ids(arrays) == ids[0]
    assert capture._object_ids(model._worker_onpolicy_pg_pending_receipts) == ids[1]
    for name in arrays:
        assert np.array_equal(arrays[name], copies[name])
    before, members = read_archive(tmp_path / "first_update_before.zip")
    assert set(members) == {"manifest.json", "gae_snapshot.npz", "pending_receipts.json",
                           "state/policy.pt", "state/optimizer.pt"}
    assert before["runtime"]["torch_num_threads"] == torch.get_num_threads()
    assert before["runtime"]["policy_training"] is False
    assert before["before_archive_sha256"] is None
    assert before["array_order"] == "time-major" and before["flatten_order"] == "t*n_envs+env"
    with np.load(io.BytesIO(members["gae_snapshot.npz"]), allow_pickle=False) as saved:
        assert set(saved.files) == set(capture.ARRAY_NAMES)
        for name in copies:
            assert np.array_equal(saved[name], copies[name]) and saved[name].dtype == copies[name].dtype
    assert [row["fixture_time_env"] for row in json.loads(members["pending_receipts.json"])] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert torch.load(io.BytesIO(members["state/optimizer.pt"]), weights_only=True)["state"] == {}
    assert capture.policy_tensor_sha256(torch.load(io.BytesIO(members["state/policy.pt"]), weights_only=True)) == before["policy_tensor_sha256"]
    assert model._worker_onpolicy_pg_collection_actor_sha256 is None
    consumed_fixture(model, before["actor_parameter_sha256"])
    rng = rng_snapshot()
    callback.on_training_end()
    assert_rng_same(rng)
    after, members = read_archive(tmp_path / "first_update_after.zip")
    assert set(members) == {"manifest.json", "committed_receipts.json", "state/policy.pt", "state/optimizer.pt"}
    assert after["before_archive_sha256"] == capture._sha((tmp_path / "first_update_before.zip").read_bytes())
    assert after["arrays"] == before["arrays"] and after["policy_tensor_sha256"] != before["policy_tensor_sha256"]
    assert json.loads(members["committed_receipts.json"])[0]["collection_actor_sha256"] == before["actor_parameter_sha256"]
    summary = json.loads((tmp_path / "rollout_diagnostic.json").read_bytes())
    assert summary["status"] == "COMPLETE"
    for phase in ("before", "after"):
        assert summary[phase]["sha256"] == capture._sha(Path(summary[phase]["path"]).read_bytes())
        with pytest.raises(EvalContractError):
            checkpoint_num_timesteps_bytes(Path(summary[phase]["path"]).read_bytes())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["first_update_after.zip", "first_update_before.zip", "rollout_diagnostic.json"]


@pytest.mark.parametrize("mutation", [
    lambda m: setattr(m, "num_timesteps", 1),
    lambda m: setattr(m, "_total_timesteps", 8),
    lambda m: setattr(m, "_ppo_optimizer_steps_completed", 1),
    lambda m: setattr(m, "_last_completed_ppo_rollout_steps", 0),
    lambda m: m.policy.optimizer.payload.update(state={0: {"step": torch.tensor(1.)}}),
    lambda m: m.diablogym_contract.update(artifact_scope="certified"),
    lambda m: m._resource_warm_start_receipt.update(policy_sha256="c" * 64),
])
def test_nonzero_or_unregistered_start_rejected(callback, tmp_path, mutation):
    mutation(callback.model)
    with pytest.raises(RuntimeError):
        start(callback)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", [
    lambda m: setattr(m, "num_timesteps", 3),
    lambda m: setattr(m.rollout_buffer, "full", False),
    lambda m: setattr(m.rollout_buffer, "generator_ready", True),
    lambda m: m._worker_onpolicy_pg_pending_receipts.reverse(),
    lambda m: m.rollout_buffer._formal_gae_snapshot.pop("dones"),
    lambda m: m.rollout_buffer._formal_gae_snapshot.update(dones=np.array([object(), object()])),
    lambda m: m.rollout_buffer._formal_gae_snapshot["advantages"].fill(np.nan),
    lambda m: m.rollout_buffer.observations.fill(1000),
    lambda m: setattr(m, "_actor_optimizer_steps_completed", 1),
])
def test_before_invalid_stage_order_or_array_has_no_output(callback, tmp_path, mutation):
    start(callback)
    collect_fixture(callback.model)
    mutation(callback.model)
    with pytest.raises(RuntimeError):
        callback.on_rollout_end()
    assert list(tmp_path.iterdir()) == []


def test_size_cap_and_torch_serialization_failure_leave_no_bundle(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    monkeypatch.setattr(capture, "MAX_UNCOMPRESSED_BYTES", 32)
    with pytest.raises(RuntimeError, match="size cap"):
        callback.on_rollout_end()
    assert list(tmp_path.iterdir()) == []


def test_serialization_error_propagates(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    monkeypatch.setattr(capture.torch, "save", lambda *a, **kw: (_ for _ in ()).throw(OSError("serialization failure")))
    with pytest.raises(OSError, match="serialization failure"):
        callback.on_rollout_end()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutation", [
    lambda m: setattr(m, "_last_completed_ppo_rollout_steps", None),
    lambda m: setattr(m, "_actor_optimizer_steps_completed", 2),
    lambda m: setattr(m, "_worker_onpolicy_pg_joint_rollouts", 2),
    lambda m: setattr(m, "_worker_onpolicy_pg_qualifying_rollouts", 1),
    lambda m: m._worker_onpolicy_pg_rollout_receipts[0].update(collection_actor_sha256="c" * 64),
    lambda m: m.rollout_buffer._formal_gae_snapshot["rewards"].fill(99),
    lambda m: m.policy.optimizer.payload["state"][0]["exp_avg"].fill_(float("nan")),
])
def test_incomplete_or_inconsistent_update_keeps_only_before(callback, tmp_path, mutation):
    start(callback)
    collect_fixture(callback.model)
    callback.on_rollout_end()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    consumed_fixture(callback.model, callback._before_actor_sha)
    mutation(callback.model)
    with pytest.raises(RuntimeError):
        callback.on_training_end()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_half_batch_training_end_without_before_never_creates_after(callback, tmp_path):
    start(callback)
    callback.model.num_timesteps = 2
    with pytest.raises(RuntimeError, match="stage"):
        callback.on_training_end()
    assert list(tmp_path.iterdir()) == []


def test_no_overwrite_or_callback_reuse(callback, tmp_path):
    target = tmp_path / "first_update_before.zip"
    target.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="already exists"):
        start(callback)
    assert target.read_bytes() == b"old"
    with pytest.raises(RuntimeError, match="reused"):
        start(callback)


def test_after_summary_replacement_failure_rolls_back_only_after(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    callback.on_rollout_end()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    consumed_fixture(callback.model, callback._before_actor_sha)
    real_replace = capture.os.replace
    def replace(source, target):
        if Path(target).name == "rollout_diagnostic.json":
            raise OSError("replace failure")
        real_replace(source, target)
    monkeypatch.setattr(capture.os, "replace", replace)
    with pytest.raises(OSError, match="replace failure"):
        callback.on_training_end()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_after_fsync_failure_restores_before_summary(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    callback.on_rollout_end()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    consumed_fixture(callback.model, callback._before_actor_sha)
    real_fsync = capture.os.fsync
    calls = []
    def fsync(fd):
        calls.append(fd)
        if len(calls) == 4:  # Three staged blobs, then the directory.
            raise OSError("directory fsync failure")
        return real_fsync(fd)
    monkeypatch.setattr(capture.os, "fsync", fsync)
    with pytest.raises(OSError, match="directory fsync failure"):
        callback.on_training_end()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_actual_audited_buffer_actions_dtype_and_gae_without_model_or_environment(callback, tmp_path):
    from gymnasium import spaces
    buffer = leashed_ppo._AuditedMaskableRolloutBuffer(
        2, spaces.Box(-100., 100., shape=(3,), dtype=np.float32), spaces.Discrete(15),
        device="cpu", gae_lambda=0.95, gamma=1.0, n_envs=2)
    start(callback)
    for step in range(2):
        buffer.add(np.full((2, 3), step, dtype=np.float32),
                   np.array([step * 2, step * 2 + 1], dtype=np.int64),
                   np.array([1., 2.], dtype=np.float32), np.array([step == 0] * 2),
                   torch.tensor([0.1, 0.2]), torch.tensor([-1., -2.]),
                   action_masks=np.ones((2, 15), dtype=bool))
    buffer.compute_returns_and_advantage(torch.tensor([0.3, 0.4]), np.array([False, True]))
    assert buffer.full and buffer.actions.dtype.kind in "if"
    assert buffer.actions.shape == (2, 2, 1)
    callback.model.rollout_buffer = buffer
    callback.model.num_timesteps = 4
    callback.model._episode_num = 1
    callback.model._worker_onpolicy_pg_pending_receipts = [
        {"requested_action": i, "transition_reward": float(1 + i % 2)} for i in range(4)]
    callback.on_rollout_end()
    _, members = read_archive(tmp_path / "first_update_before.zip")
    with np.load(io.BytesIO(members["gae_snapshot.npz"]), allow_pickle=False) as saved:
        assert saved["actions"].dtype == buffer.actions.dtype
        assert np.array_equal(saved["actions"].reshape(-1), [0, 1, 2, 3])
        assert np.array_equal(saved["advantages"], buffer._formal_gae_snapshot["advantages"])


def test_external_summary_edit_is_preserved_and_after_refused(callback, tmp_path):
    start(callback)
    collect_fixture(callback.model)
    callback.on_rollout_end()
    consumed_fixture(callback.model, callback._before_actor_sha)
    summary = tmp_path / "rollout_diagnostic.json"
    summary.write_bytes(b"externally modified")
    with pytest.raises(RuntimeError, match="ownership/bytes"):
        callback.on_training_end()
    assert summary.read_bytes() == b"externally modified"
    assert not (tmp_path / "first_update_after.zip").exists()


def test_original_pg_validator_failure_prevents_after(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    callback.on_rollout_end()
    consumed_fixture(callback.model, callback._before_actor_sha)
    calls = []
    def invalid(receipt, expected_samples=None):
        calls.append(expected_samples)
        return False
    monkeypatch.setattr(leashed_ppo, "validate_worker_onpolicy_pg_receipt", invalid)
    with pytest.raises(RuntimeError, match="canonical PG receipt"):
        callback.on_training_end()
    assert calls == [4] and not (tmp_path / "first_update_after.zip").exists()


def test_capture_detects_receipt_object_replacement(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    original_save = capture.torch.save
    def save(*args, **kwargs):
        original_save(*args, **kwargs)
        callback.model._worker_onpolicy_pg_pending_receipts = deepcopy(
            callback.model._worker_onpolicy_pg_pending_receipts)
    monkeypatch.setattr(capture.torch, "save", save)
    with pytest.raises(RuntimeError, match="mutated live data"):
        callback.on_rollout_end()
    assert list(tmp_path.iterdir()) == []


def test_archive_claim_race_preserves_competing_file(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    original_link = capture.os.link
    def link(source, target):
        if Path(target).name == "first_update_before.zip":
            Path(target).write_bytes(b"competing artifact")
        original_link(source, target)
    monkeypatch.setattr(capture.os, "link", link)
    with pytest.raises(FileExistsError):
        callback.on_rollout_end()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == {
        "first_update_before.zip": b"competing artifact"}


def test_temp_cleanup_failure_rolls_back_before_and_summary(callback, tmp_path, monkeypatch):
    start(callback)
    collect_fixture(callback.model)
    original_unlink = Path.unlink
    failed = []
    def unlink(path, *args, **kwargs):
        if path.name.startswith(".first-update-") and not failed:
            assert (tmp_path / "rollout_diagnostic.json").exists()
            failed.append(path)
            raise OSError("temp cleanup failure")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="temp cleanup failure"):
        callback.on_rollout_end()
    assert failed and list(tmp_path.iterdir()) == []


def test_hidden_state_dict_metadata_must_also_be_pure_data():
    from collections import OrderedDict
    state = OrderedDict(weight=torch.ones(1))
    state._metadata = {"": {"version": 1}}
    assert capture._tree_bytes(state) == 4
    state._metadata["unsafe"] = object()
    with pytest.raises(RuntimeError, match="non-data object"):
        capture._tree_bytes(state)
