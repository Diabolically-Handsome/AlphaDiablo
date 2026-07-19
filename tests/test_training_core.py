"""训练入口的自包含快速回归（不启动引擎、不依赖 ignored 训练产物）。"""

from __future__ import annotations

import json
import hashlib
import io
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
import warnings
import zipfile
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from bc_worker import split_by_episode  # noqa: E402
from diablogym import NumpyManager, WorkerWindowEnv  # noqa: E402
from models import EntityAttentionExtractor  # noqa: E402
from leashed_ppo import HUGE_NEG, LeashedMaskablePPO, build_teacher  # noqa: E402
import train_ppo as train_module  # noqa: E402
import eval_contract  # noqa: E402
from train_ppo import (  # noqa: E402
    AtomicRolloutCheckpointCallback,
    EpisodeJsonlCallback,
    _GEAR_PRESENT_INDEX,
    _RunLock,
    _atomic_save_model,
    _capture_leashed_checkpoint,
    _export_manifest_path,
    _load_bc_state_dict,
    _load_dry_anchor_demos,
    _prepare_run_dir,
    _select_batch_size,
    _is_publishable_rollout_boundary,
    _validate_checkpoint_bytes,
    _validate_checkpoint_file,
    _validate_export_manifest,
    _validate_leashed_checkpoint,
    _validate_model_recipe,
    _validate_bc_report,
    _validate_resume_contract,
)


def _valid_worker_bc_report(policy: pathlib.Path) -> dict:
    demos = policy.with_name("demos.npz")
    if not demos.exists():
        episode_id = np.repeat(
            np.asarray(train_module._WORKER_BC_DEMO_SEEDS, dtype=np.int64), 4)
        x = np.zeros((len(episode_id), 298), dtype=np.float32)
        y = np.full(len(episode_id), 9, dtype=np.int64)
        np.savez_compressed(
            demos, X=x, Y=y, episode_id=episode_id)
        state = {
            "mlp_extractor.policy_net.0.weight": torch.zeros(64, 298),
            "mlp_extractor.policy_net.0.bias": torch.zeros(64),
            "mlp_extractor.policy_net.2.weight": torch.zeros(64, 64),
            "mlp_extractor.policy_net.2.bias": torch.zeros(64),
            "action_net.weight": torch.zeros(15, 64),
            "action_net.bias": torch.nn.functional.one_hot(
                torch.tensor(9), num_classes=15).float() * 10,
        }
        torch.save(state, policy)
    with np.load(demos, allow_pickle=False) as archive:
        groups = archive["episode_id"]
        pairs = len(archive["Y"])
    _train, holdout, held_episodes = split_by_episode(groups)
    return {
        "schema_version": train_module._BC_REPORT_SCHEMA_VERSION,
        "pairs": pairs,
        "held_out_top1": 1.0,
        "held_out_pairs": int(len(holdout)),
        "held_out_episodes": [int(value) for value in sorted(held_episodes)],
        "class_recalls": {"9": 1.0},
        "class_weighted_retry": False,
        "data_gate": "PASS",
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "demos_sha256": hashlib.sha256(demos.read_bytes()).hexdigest(),
        "protocol_version": eval_contract.PROTOCOL_VERSION,
        "implementation_sha256": train_module._implementation_bundle_sha256(),
        "generator_sha256": hashlib.sha256(
            (ROOT / "train" / "bc_worker.py").read_bytes()).hexdigest(),
        "manager_npz_sha256": hashlib.sha256(
            (ROOT / "train" / "models" / "v22-h-manager" / "policy.npz")
            .read_bytes()).hexdigest(),
    }


class TinyMaskedEnv(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self):
        return np.ones(3, dtype=bool)


def _write_tiny_teacher(path: pathlib.Path, preferred_action: int = 2) -> None:
    """Write a self-contained 4→64→64→3 teacher for leash guards."""
    action_bias = torch.full((3,), -10.0)
    action_bias[preferred_action] = 10.0
    torch.save({
        "mlp_extractor.policy_net.0.weight": torch.zeros(64, 4),
        "mlp_extractor.policy_net.0.bias": torch.zeros(64),
        "mlp_extractor.policy_net.2.weight": torch.zeros(64, 64),
        "mlp_extractor.policy_net.2.bias": torch.zeros(64),
        "action_net.weight": torch.zeros(3, 64),
        "action_net.bias": action_bias,
    }, path)


class TrainingCoreTests(unittest.TestCase):
    @staticmethod
    def _plain_model(env):
        return MaskablePPO(
            "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
            seed=7, device="cpu", verbose=0)

    @staticmethod
    def _fill_masked_buffer(model, seed=11):
        rng = np.random.default_rng(seed)
        buffer = model.rollout_buffer
        buffer.reset()
        for index in range(model.n_steps):
            obs = rng.standard_normal((1, 4)).astype(np.float32)
            mask = np.ones((1, 3), dtype=bool)
            buffer.add(
                obs, np.asarray([int(rng.integers(3))]),
                np.asarray([float(rng.standard_normal())]),
                np.asarray([index % 5 == 0]), torch.zeros(1), torch.zeros(1),
                action_masks=mask)
        buffer.compute_returns_and_advantage(
            last_values=torch.zeros(1), dones=np.zeros(1))

    def test_batch_size_never_creates_singleton_tail(self):
        self.assertEqual(_select_batch_size(512, 4), 256)
        for rollout_size in range(2, 2_000):
            batch = _select_batch_size(rollout_size, 1)
            self.assertTrue(rollout_size <= batch or rollout_size % batch != 1,
                            (rollout_size, batch))

    def test_run_retry_archives_stale_outputs_but_preserves_explicit_input(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "model_final.zip").write_bytes(b"old-model")
            (run_dir / "progress.jsonl").write_text("old-progress\n")
            protected = pathlib.Path(directory) / "input-policy.npz"
            protected.write_bytes(b"active-input")
            _prepare_run_dir(run_dir, None, [protected])

            self.assertTrue(protected.exists())
            self.assertFalse((run_dir / "model_final.zip").exists())
            archives = list((run_dir / "_attempts").iterdir())
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / "model_final.zip").read_bytes(), b"old-model")
            self.assertEqual((archives[0] / "progress.jsonl").read_text(), "old-progress\n")

    def test_run_retry_rejects_input_nested_beside_stale_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory) / "run"
            ckpt = run_dir / "ckpt"
            ckpt.mkdir(parents=True)
            protected = ckpt / "init.pt"
            stale = ckpt / "model_999999_steps.zip"
            protected.write_bytes(b"active-input")
            stale.write_bytes(b"stale-checkpoint")

            with self.assertRaisesRegex(ValueError, "训练输入不能位于"):
                _prepare_run_dir(run_dir, None, [protected])
            self.assertTrue(protected.exists())
            self.assertTrue(stale.exists())
            self.assertFalse((run_dir / "_attempts").exists())

    def test_checkpoint_validation_and_leashed_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([TinyMaskedEnv])
            plain = pathlib.Path(directory) / "plain.zip"
            self._plain_model(env).save(plain)
            self.assertEqual(_validate_checkpoint_file(plain)["num_timesteps"], 0)
            with self.assertRaisesRegex(ValueError, "不是 Leashed"):
                _validate_leashed_checkpoint(plain)

            leashed = pathlib.Path(directory) / "leashed.zip"
            LeashedMaskablePPO(
                "MlpPolicy", env, n_steps=8, batch_size=8,
                distill_beta=0.0, seed=7, device="cpu", verbose=0).save(leashed)
            self.assertEqual(_validate_leashed_checkpoint(leashed)["distill_beta"], 0.0)

            broken = pathlib.Path(directory) / "broken.zip"
            broken.write_bytes(leashed.read_bytes()[:128])
            with self.assertRaisesRegex(ValueError, "不可读/不安全"):
                _validate_checkpoint_file(broken)

            with zipfile.ZipFile(leashed) as source:
                members = {name: source.read(name) for name in source.namelist()}
            original_data = json.loads(members["data"])
            for invalid_steps in (True, 1.5, "1"):
                invalid_data = dict(original_data, num_timesteps=invalid_steps)
                payload = io.BytesIO()
                with zipfile.ZipFile(payload, "w") as archive:
                    for name, member in members.items():
                        archive.writestr(
                            name,
                            json.dumps(invalid_data) if name == "data" else member)
                with self.subTest(num_timesteps=invalid_steps), \
                        self.assertRaisesRegex(ValueError, "非负普通整数"):
                    _validate_checkpoint_bytes(payload.getvalue(), "invalid-steps")

            duplicate = pathlib.Path(directory) / "duplicate.zip"
            duplicate.write_bytes(leashed.read_bytes())
            with zipfile.ZipFile(leashed) as source:
                duplicate_policy = source.read("policy.pth")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(duplicate, "a") as archive:
                    archive.writestr("policy.pth", duplicate_policy)
            with self.assertRaisesRegex(ValueError, "重复 ZIP 成员"):
                _validate_checkpoint_file(duplicate)
            env.close()

    def test_hidden_ppo_recipe_is_pinned(self):
        env = DummyVecEnv([TinyMaskedEnv])
        model = self._plain_model(env)
        model.n_epochs = 10  # _plain_model keeps other tests fast with one epoch.
        _validate_model_recipe(model)
        model.n_epochs = 9
        with self.assertRaisesRegex(ValueError, "隐含算法配方漂移"):
            _validate_model_recipe(model)
        model.n_epochs = 10
        model.clip_range = lambda progress: 0.2 if progress == 1.0 else 0.1
        with self.assertRaisesRegex(ValueError, "隐含算法配方漂移"):
            _validate_model_recipe(model)
        env.close()

    def test_training_contract_is_stdlib_json_serializable(self):
        env = DummyVecEnv([TinyMaskedEnv])
        model = self._plain_model(env)
        args = types.SimpleNamespace(
            worker=False, options=False, flat_clock=False, arch="mlp",
            max_steps=8, num_envs=1, n_steps=8, gamma=0.99, lr=3e-4,
            ent_coef=0.02, skip_dry=False, no_drink_sovereignty=False,
            # E4 rev5 改写(相应单测改写而非删除):契约新读两旗,补不在位默认
            dry_curriculum_schedule=None, bc_aux_lambda=0.0, bc_aux_demos=None,
        )
        contract = train_module._training_contract(args, model, batch_size=8)
        json.dumps(contract)
        self.assertIs(type(contract["action_n"]), int)
        self.assertTrue(all(type(value) is int
                            for value in contract["observation_shape"]))
        env.close()

    def test_resume_load_uses_captured_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([TinyMaskedEnv])
            path = pathlib.Path(directory) / "leashed.zip"
            LeashedMaskablePPO(
                "MlpPolicy", env, n_steps=8, batch_size=8,
                distill_beta=0.0, seed=7, device="cpu", verbose=0).save(path)
            payload, data, digest = _capture_leashed_checkpoint(path)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
            self.assertEqual(data["distill_beta"], 0.0)

            # A path replacement after capture cannot affect the object loaded
            # by the training entry point.
            path.write_bytes(b"replacement garbage")
            loaded = LeashedMaskablePPO.load(io.BytesIO(payload), env=env)
            self.assertEqual(loaded.num_timesteps, 0)
            env.close()

    def test_checkpoint_rejects_nonfinite_policy_and_optimizer_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([TinyMaskedEnv])
            model = self._plain_model(env)
            with torch.no_grad():
                next(model.policy.parameters()).flatten()[0] = float("nan")
            path = pathlib.Path(directory) / "nan.zip"
            model.save(path)
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                _validate_checkpoint_file(path)
            env.close()

    def test_atomic_save_preserves_canonical_on_validation_failure(self):
        class BrokenModel:
            def save(self, path):
                pathlib.Path(path).write_bytes(b"not a checkpoint")

        with tempfile.TemporaryDirectory() as directory:
            final = pathlib.Path(directory) / "model_final.zip"
            final.write_bytes(b"known-good-placeholder")
            with self.assertRaises(ValueError):
                _atomic_save_model(BrokenModel(), final)
            self.assertEqual(final.read_bytes(), b"known-good-placeholder")
            self.assertEqual(list(final.parent.glob(".*.tmp.zip")), [])

    def test_rollout_checkpoint_is_post_update_and_quantized(self):
        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([TinyMaskedEnv])
            model = self._plain_model(env)
            callback = AtomicRolloutCheckpointCallback(
                pathlib.Path(directory), every_steps=5)
            model.learn(total_timesteps=16, callback=callback)
            paths = sorted((pathlib.Path(directory) / "ckpt").glob("*.zip"))
            self.assertEqual([p.name for p in paths], [
                "model_16_steps.zip", "model_8_steps.zip",
            ])
            for expected, path in ((16, paths[0]), (8, paths[1])):
                data = _validate_checkpoint_file(path)
                self.assertEqual(data["num_timesteps"], expected)
                with zipfile.ZipFile(path) as archive:
                    optimizer = torch.load(
                        archive.open("policy.optimizer.pth"), map_location="cpu",
                        weights_only=True)
                self.assertTrue(optimizer["state"], path)
            env.close()

    def test_checkpoint_rejects_runtime_content_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = AtomicRolloutCheckpointCallback(
                pathlib.Path(directory), every_steps=8,
                implementation_sha256="a" * 64)
            callback.next_at = 8
            callback.num_timesteps = 8
            with mock.patch.object(
                    train_module, "_implementation_bundle_sha256",
                    return_value="b" * 64):
                with self.assertRaisesRegex(ValueError, "游戏内容发生漂移"):
                    callback._save_due()

    def test_gcal_rejected_rollout_cannot_publish_checkpoint_or_final(self):
        callback = AtomicRolloutCheckpointCallback(
            pathlib.Path("/unused"), every_steps=8)
        callback.next_at = 8
        callback.num_timesteps = 8
        callback.model = types.SimpleNamespace(
            _calib_tripped=True,
            rollout_buffer=types.SimpleNamespace(full=True),
        )
        with mock.patch.object(train_module, "_atomic_save_model") as save:
            callback._save_due()
            callback._on_training_end()
        save.assert_not_called()
        self.assertFalse(_is_publishable_rollout_boundary(callback.model))

        # record_only probes never arm _calib_tripped, so an otherwise complete
        # rollout remains publishable.
        callback.model._calib_tripped = False
        self.assertTrue(_is_publishable_rollout_boundary(callback.model))

    def test_run_lock_rejects_concurrent_owner_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            first = _RunLock(run_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "正被另一训练进程占用"):
                    _RunLock(run_dir)
            finally:
                first.close()
            second = _RunLock(run_dir)
            second.close()

    def test_numpy_manager_binds_hash_to_the_bytes_it_parses(self):
        source = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
        payload = source.read_bytes()
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "manager.npz"
            path.write_bytes(payload)
            manager = NumpyManager(path, expected_sha256=expected)
            self.assertEqual(manager.source_sha256, expected)
            manager.require_io_shape(303, 3, "test manager")

            path.write_bytes(payload + b"replaced")
            with self.assertRaisesRegex(ValueError, "SHA256 不匹配"):
                NumpyManager(path, expected_sha256=expected)

    def test_external_npz_shapes_fail_before_native_env_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "wrong.npz"
            np.savez_compressed(
                path,
                w0=np.zeros((2, 4), dtype=np.float32),
                b0=np.zeros(2, dtype=np.float32),
                w1=np.zeros((2, 2), dtype=np.float32),
                b1=np.zeros(2, dtype=np.float32),
                wa=np.zeros((3, 2), dtype=np.float32),
                ba=np.zeros(3, dtype=np.float32),
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "303→3"):
                WorkerWindowEnv(str(path), manager_sha256=digest)
            with self.assertRaisesRegex(ValueError, "298→15"):
                train_module.make_env(
                    options=True, worker_npz=str(path),
                    worker_npz_sha256=digest)

    def test_all_prelearn_failures_close_env_lock_and_restore_sigalrm(self):
        class FakeResource:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        original_main = train_module._main
        original_handler = signal.getsignal(signal.SIGALRM)

        def host_handler(*_):
            return None

        signal.signal(signal.SIGALRM, host_handler)
        try:
            for phase in ("model", "load", "contract", "bc", "callback"):
                lock, env = FakeResource(), FakeResource()

                def fail(resources, current_phase=phase):
                    resources.run_lock = lock
                    resources.vec_env = env
                    raise RuntimeError(f"fault:{current_phase}")

                train_module._main = fail
                with self.assertRaisesRegex(RuntimeError, f"fault:{phase}"):
                    train_module.main()
                self.assertTrue(env.closed, phase)
                self.assertTrue(lock.closed, phase)
                self.assertIs(signal.getsignal(signal.SIGALRM), host_handler)
        finally:
            train_module._main = original_main
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)

        # Exercise the real registration points too: the lock and VecEnv are
        # acquired by _main, then policy construction fails before the old
        # learn-scoped finally would have existed.
        lock, env = FakeResource(), FakeResource()
        saved = (train_module._RunLock, train_module.DummyVecEnv,
                 train_module._prepare_run_dir, train_module.PPO, sys.argv)
        try:
            train_module._RunLock = lambda _: lock
            train_module.DummyVecEnv = lambda _: env
            train_module._prepare_run_dir = lambda *_, **__: None

            def model_fault(*_, **__):
                raise RuntimeError("fault:real-model-setup")

            train_module.PPO = model_fault
            sys.argv = [
                "train_ppo.py", "--total-steps", "4", "--num-envs", "1",
                "--n-steps", "2", "--run-name", "audit-no-write",
            ]
            with self.assertRaisesRegex(RuntimeError, "fault:real-model-setup"):
                train_module.main()
            self.assertTrue(env.closed)
            self.assertTrue(lock.closed)
        finally:
            (train_module._RunLock, train_module.DummyVecEnv,
             train_module._prepare_run_dir, train_module.PPO, sys.argv) = saved

    def test_checkpoint_export_manifest_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([TinyMaskedEnv])
            source = pathlib.Path(directory) / "source.zip"
            output = pathlib.Path(directory) / "policy.pt"
            self._plain_model(env).save(source)
            run = subprocess.run(
                [sys.executable, str(ROOT / "train" / "export_manager_sd.py"),
                 str(source), str(output)], text=True, capture_output=True,
                check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            manifest = _validate_export_manifest(output)
            self.assertEqual(manifest["artifact_type"], "checkpoint_policy_state")
            self.assertEqual(_export_manifest_path(output).suffix, ".json")

            manifest_path = _export_manifest_path(output)
            duplicate = manifest_path.read_text().replace(
                "{", '{"schema_version":999,', 1)
            manifest_path.write_text(duplicate)
            with self.assertRaisesRegex(ValueError, "不可读"):
                _validate_export_manifest(output)
            manifest_path.write_text(json.dumps(manifest))

            for invalid_schema in (999, True):
                wrong_schema = dict(manifest, schema_version=invalid_schema)
                manifest_path.write_text(json.dumps(wrong_schema))
                with self.assertRaisesRegex(ValueError, "schema"):
                    _validate_export_manifest(output)
            manifest_path.write_text(json.dumps(manifest))

            output.write_bytes(output.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA"):
                _validate_export_manifest(output)
            output.write_bytes(output.read_bytes()[:-6])
            source_payload = source.read_bytes()
            source.unlink()
            with self.assertRaisesRegex(ValueError, "源 checkpoint 不可读"):
                _validate_export_manifest(output)
            source.write_bytes(source_payload)
            source.write_bytes(source_payload + b"tamper")
            with self.assertRaisesRegex(ValueError, "源 checkpoint SHA"):
                _validate_export_manifest(output)
            env.close()

    def test_checkpoint_init_requires_the_full_exact_policy_state(self):
        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([TinyMaskedEnv])
            model = self._plain_model(env)
            state = model.policy.state_dict()
            source = pathlib.Path(directory) / "source.zip"
            model.save(source)
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            partial = {key: state[key] for key in train_module._POLICY_HEAD_KEYS}
            artifact = pathlib.Path(directory) / "partial.pt"
            torch.save(partial, artifact)
            manifest = {
                "schema_version": train_module._EXPORT_MANIFEST_SCHEMA_VERSION,
                "artifact_type": "checkpoint_policy_state",
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "source_checkpoint": str(source.resolve()),
                "source_checkpoint_sha256": source_sha,
                "tensor_count": len(partial),
            }
            _export_manifest_path(artifact).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                    ValueError, "tensor_count|导出件字段.*源 checkpoint"):
                _load_bc_state_dict(
                    str(artifact), model.policy, "hypothesis", "checkpoint")

            torch.save(state, artifact)
            manifest["artifact_sha256"] = hashlib.sha256(
                artifact.read_bytes()).hexdigest()
            manifest["tensor_count"] = len(state) + 1
            _export_manifest_path(artifact).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "tensor_count"):
                _load_bc_state_dict(
                    str(artifact), model.policy, "hypothesis", "checkpoint")

            wrong_dtype = {key: value.clone() for key, value in state.items()}
            first_key = next(iter(wrong_dtype))
            wrong_dtype[first_key] = wrong_dtype[first_key].double()
            torch.save(wrong_dtype, artifact)
            manifest["artifact_sha256"] = hashlib.sha256(
                artifact.read_bytes()).hexdigest()
            manifest["tensor_count"] = len(wrong_dtype)
            _export_manifest_path(artifact).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "导出件张量.*源 checkpoint"):
                _load_bc_state_dict(
                    str(artifact), model.policy, "hypothesis", "checkpoint")
            env.close()

    def test_resume_contract_rejects_environment_and_device_drift(self):
        base = {
            "schema_version": 2, "contract_revision": 3,
            "implementation_sha256": "c" * 64,
            "mode": "worker", "num_envs": 4,
            "skip_dry": False, "manager_npz_sha256": "a" * 64,
            "device": "cpu",
        }
        _validate_resume_contract(base, dict(base))
        for key, value in (("num_envs", 2), ("skip_dry", True),
                           ("device", "mps"),
                           ("implementation_sha256", "d" * 64)):
            changed = dict(base, **{key: value})
            with self.assertRaisesRegex(ValueError, "契约漂移"):
                _validate_resume_contract(base, changed)
        changed_manager = dict(base, manager_npz_sha256="b" * 64)
        with self.assertRaises(ValueError):
            _validate_resume_contract(base, changed_manager)
        _validate_resume_contract(base, changed_manager, allow_manager_change=True)
        with self.assertRaisesRegex(ValueError, "无 training_contract"):
            _validate_resume_contract(None, base)
        _validate_resume_contract(None, base, allow_legacy_resume=True)

    def test_bc_report_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = pathlib.Path(directory) / "policy_sd.pt"
            policy.write_bytes(b"placeholder")
            report = policy.with_name("bc_report.json")
            sha = hashlib.sha256(policy.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                _validate_bc_report(policy, "data_gate")
            for record in ({"data_gate": "RUNNING", "policy_sha256": sha},
                           {"hypothesis": "PASS", "policy_sha256": sha},
                           {"unrecognized": "PASS", "policy_sha256": sha}):
                report.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    _validate_bc_report(policy, "data_gate")
            wrong_policy = _valid_worker_bc_report(policy)
            wrong_policy["policy_sha256"] = "0" * 64
            report.write_text(json.dumps(wrong_policy))
            with self.assertRaisesRegex(ValueError, "SHA"):
                _validate_bc_report(policy, "data_gate")
            report.write_text(json.dumps(_valid_worker_bc_report(policy)))
            _validate_bc_report(policy, "data_gate")

            valid_record = _valid_worker_bc_report(policy)
            for field, value, message in (
                    ("pairs", 401, "X 形状/dtype"),
                    ("held_out_pairs", valid_record["held_out_pairs"] + 1,
                     "held_out_pairs"),
                    ("held_out_episodes", [0, 1], "held_out_episodes")):
                forged = dict(valid_record, **{field: value})
                report.write_text(json.dumps(forged))
                with self.subTest(evidence_field=field), \
                        self.assertRaisesRegex(ValueError, message):
                    _validate_bc_report(policy, "data_gate")

            demos = policy.with_name("demos.npz")
            canonical_demos = demos.read_bytes()
            with np.load(io.BytesIO(canonical_demos), allow_pickle=False) as archive:
                canonical_x = archive["X"]
                canonical_y = archive["Y"]
                canonical_groups = archive["episode_id"]

            keep = canonical_groups < 120
            np.savez_compressed(
                demos, X=canonical_x[keep], Y=canonical_y[keep],
                episode_id=canonical_groups[keep])
            partial = _valid_worker_bc_report(policy)
            report.write_text(json.dumps(partial))
            with self.assertRaisesRegex(ValueError, "固定示范种子"):
                _validate_bc_report(policy, "data_gate")

            bad_y = canonical_y.copy()
            bad_y[0] = 11
            np.savez_compressed(
                demos, X=canonical_x, Y=bad_y,
                episode_id=canonical_groups)
            forbidden = _valid_worker_bc_report(policy)
            report.write_text(json.dumps(forbidden))
            with self.assertRaisesRegex(ValueError, "禁采动作 11/12"):
                _validate_bc_report(policy, "data_gate")
            demos.write_bytes(canonical_demos)
            report.write_text(json.dumps(valid_record))

            valid_policy_payload = policy.read_bytes()
            forged_state = torch.load(
                io.BytesIO(valid_policy_payload), map_location="cpu",
                weights_only=True)
            forged_state["action_net.bias"] = torch.nn.functional.one_hot(
                torch.tensor(0), num_classes=15).float() * 10
            torch.save(forged_state, policy)
            forged_metrics = dict(
                valid_record,
                policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
            report.write_text(json.dumps(forged_metrics))
            with self.assertRaisesRegex(ValueError, "held_out_top1.*重算"):
                _validate_bc_report(policy, "data_gate")
            policy.write_bytes(valid_policy_payload)
            report.write_text(json.dumps(valid_record))

            demos_payload = demos.read_bytes()
            demos.write_bytes(demos_payload + b"tamper")
            with self.assertRaisesRegex(ValueError, "demos.npz.*SHA"):
                _validate_bc_report(policy, "data_gate")
            demos.write_bytes(demos_payload)
            demos.unlink()
            with self.assertRaisesRegex(ValueError, "示范集缺失"):
                _validate_bc_report(policy, "data_gate")
            demos.write_bytes(demos_payload)

            valid = json.dumps(_valid_worker_bc_report(policy))
            report.write_text(valid.replace("{", '{"data_gate":"FAIL",', 1))
            with self.assertRaisesRegex(ValueError, "不可读"):
                _validate_bc_report(policy, "data_gate")

            unknown = _valid_worker_bc_report(policy)
            unknown["unexpected"] = True
            report.write_text(json.dumps(unknown))
            with self.assertRaisesRegex(ValueError, "字段/schema"):
                _validate_bc_report(policy, "data_gate")

            bool_schema = _valid_worker_bc_report(policy)
            bool_schema["schema_version"] = True
            report.write_text(json.dumps(bool_schema))
            with self.assertRaisesRegex(ValueError, "schema"):
                _validate_bc_report(policy, "data_gate")

            stale = _valid_worker_bc_report(policy)
            stale["protocol_version"] -= 1
            report.write_text(json.dumps(stale))
            with self.assertRaisesRegex(ValueError, "协议过期"):
                _validate_bc_report(policy, "data_gate")
            for field, message in (
                    ("implementation_sha256", "运行时不一致"),
                    ("generator_sha256", "生成器已漂移"),
                    ("manager_npz_sha256", "冻结 manager")):
                stale = _valid_worker_bc_report(policy)
                stale[field] = "0" * 64
                report.write_text(json.dumps(stale))
                with self.assertRaisesRegex(ValueError, message):
                    _validate_bc_report(policy, "data_gate")

    def test_replay_bc_report_metrics_are_finite_positive_and_self_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = pathlib.Path(directory) / "policy_sd.pt"
            policy.write_bytes(b"placeholder")
            report = policy.with_name("bc_report.json")
            common = {
                "schema_version": train_module._BC_REPORT_SCHEMA_VERSION,
                "pairs": 1_000,
                "protocol_version": eval_contract.PROTOCOL_VERSION,
                # This unit test exercises report arithmetic, not the expensive
                # live source/native/content bundle.  Pin an injected identity
                # so a concurrently rebuilt workspace cannot make the two gate
                # variants observe different implementation snapshots.
                "implementation_sha256": "a" * 64,
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            }
            cases = {
                "hypothesis": {
                    **common,
                    "teacher_demo_mean": 10.0,
                    "bc_replay_7000": 9.0,
                    "teacher_7000": 10.0,
                    "ratio": 0.9,
                    "hypothesis": "PASS",
                    "generator_sha256": hashlib.sha256(
                        (ROOT / "train" / "bc_manager.py").read_bytes()).hexdigest(),
                },
                "memoryless_hypothesis": {
                    **common,
                    "teacher_mean_demo": 10.0,
                    "bc_replay_mean_7000s": 9.0,
                    "teacher_replay_mean_7000s": 10.0,
                    "ratio": 0.9,
                    "memoryless_hypothesis": "PASS",
                    "generator_sha256": hashlib.sha256(
                        (ROOT / "train" / "bc_flat.py").read_bytes()).hexdigest(),
                },
            }
            metric_keys = {
                "hypothesis": (
                    "teacher_demo_mean", "bc_replay_7000", "teacher_7000"),
                "memoryless_hypothesis": (
                    "teacher_mean_demo", "bc_replay_mean_7000s",
                    "teacher_replay_mean_7000s"),
            }
            for gate, valid in cases.items():
                with self.subTest(gate=gate, case="valid"):
                    report.write_text(json.dumps(valid))
                    _validate_bc_report(
                        policy, gate,
                        expected_implementation_sha256=common[
                            "implementation_sha256"],
                        verify_replay=False)

                demo_key, bc_key, teacher_key = metric_keys[gate]
                for key in (demo_key, bc_key, teacher_key):
                    malformed = dict(valid, **{key: "not-a-number"})
                    report.write_text(json.dumps(malformed))
                    with self.subTest(gate=gate, case=f"nonfinite-{key}"), \
                            self.assertRaisesRegex(ValueError, "必须是数值"):
                        _validate_bc_report(
                            policy, gate,
                            expected_implementation_sha256=common[
                                "implementation_sha256"],
                            verify_replay=False)

                zero_teacher = dict(valid, **{teacher_key: 0.0})
                report.write_text(json.dumps(zero_teacher))
                with self.subTest(gate=gate, case="zero-teacher"), \
                        self.assertRaisesRegex(ValueError, "必须为正"):
                    _validate_bc_report(
                        policy, gate,
                        expected_implementation_sha256=common[
                            "implementation_sha256"],
                        verify_replay=False)

                inconsistent = dict(valid, ratio=0.95)
                report.write_text(json.dumps(inconsistent))
                with self.subTest(gate=gate, case="inconsistent-ratio"), \
                        self.assertRaisesRegex(ValueError, "指标不一致"):
                    _validate_bc_report(
                        policy, gate,
                        expected_implementation_sha256=common[
                            "implementation_sha256"],
                        verify_replay=False)

    def test_replay_bc_report_is_reexecuted_from_frozen_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = pathlib.Path(directory) / "policy_sd.pt"
            state = {
                "mlp_extractor.policy_net.0.weight": torch.zeros(64, 303),
                "mlp_extractor.policy_net.0.bias": torch.zeros(64),
                "mlp_extractor.policy_net.2.weight": torch.zeros(64, 64),
                "mlp_extractor.policy_net.2.bias": torch.zeros(64),
                "action_net.weight": torch.zeros(3, 64),
                "action_net.bias": torch.zeros(3),
            }
            torch.save(state, policy)
            implementation = "b" * 64
            record = {
                "schema_version": train_module._BC_REPORT_SCHEMA_VERSION,
                "pairs": 1_000,
                "teacher_demo_mean": 10.0,
                "bc_replay_7000": 9.0,
                "teacher_7000": 10.0,
                "ratio": 0.9,
                "hypothesis": "PASS",
                "protocol_version": eval_contract.PROTOCOL_VERSION,
                "implementation_sha256": implementation,
                "generator_sha256": hashlib.sha256(
                    (ROOT / "train" / "bc_manager.py").read_bytes()).hexdigest(),
                "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            }
            policy.with_name("bc_report.json").write_text(json.dumps(record))
            recomputed = dict(
                pairs=1_000, teacher_demo_mean=10.0,
                bc_replay_7000=1.0, teacher_7000=10.0, ratio=0.1)
            with mock.patch.object(
                    train_module, "_recompute_replay_bc_evidence",
                    return_value=recomputed) as replay:
                with self.assertRaisesRegex(ValueError, "冻结 policy.*重算不一致"):
                    _validate_bc_report(
                        policy, "hypothesis",
                        expected_implementation_sha256=implementation)
                replay.assert_called_once()

    def test_dry_anchor_demos_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "demos.npz"
            x = np.zeros((2, 298), dtype=np.float32)
            x[0, 297] = 1.0
            y = np.asarray([9, 10], dtype=np.int64)
            episode_id = np.asarray([0, 1], dtype=np.int64)
            np.savez_compressed(path, X=x, Y=y, episode_id=episode_id)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded_x, loaded_y, actual = _load_dry_anchor_demos(path, expected)
            self.assertEqual(actual, expected)
            self.assertTrue(np.array_equal(loaded_x, x))
            self.assertTrue(np.array_equal(loaded_y, y))

            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "demos SHA"):
                _load_dry_anchor_demos(path, expected)

    def test_teacher_loader_rejects_nonfinite_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "teacher.pt"
            state = {
                "mlp_extractor.policy_net.0.weight": torch.zeros(2, 4),
                "mlp_extractor.policy_net.0.bias": torch.zeros(2),
                "mlp_extractor.policy_net.2.weight": torch.zeros(2, 2),
                "mlp_extractor.policy_net.2.bias": torch.zeros(2),
                "action_net.weight": torch.zeros(3, 2),
                "action_net.bias": torch.zeros(3),
            }
            state["action_net.bias"][0] = float("nan")
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                build_teacher(str(path))

    def test_fresh_teacher_rejects_stale_expected_sha(self):
        state = {
            "mlp_extractor.policy_net.0.weight": torch.zeros(2, 4),
            "mlp_extractor.policy_net.0.bias": torch.zeros(2),
            "mlp_extractor.policy_net.2.weight": torch.zeros(2, 2),
            "mlp_extractor.policy_net.2.bias": torch.zeros(2),
            "action_net.weight": torch.zeros(3, 2),
            "action_net.bias": torch.zeros(3),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "teacher.pt"
            torch.save(state, path)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            state["action_net.bias"][0] = 1.0
            torch.save(state, path)
            env = DummyVecEnv([TinyMaskedEnv])
            with self.assertRaisesRegex(ValueError, "教师 SHA 不匹配"):
                LeashedMaskablePPO(
                    "MlpPolicy", env, n_steps=8, batch_size=8,
                    distill_beta=1.0, teacher_path=str(path),
                    teacher_sha256=expected, seed=7, device="cpu", verbose=0)
            env.close()

    def test_teacher_payload_is_not_reopened_after_hashing(self):
        state = {
            "mlp_extractor.policy_net.0.weight": torch.zeros(2, 4),
            "mlp_extractor.policy_net.0.bias": torch.zeros(2),
            "mlp_extractor.policy_net.2.weight": torch.zeros(2, 2),
            "mlp_extractor.policy_net.2.bias": torch.zeros(2),
            "action_net.weight": torch.zeros(3, 2),
            "action_net.bias": torch.tensor([5.0, 0.0, 0.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "teacher.pt"
            torch.save(state, path)
            payload = path.read_bytes()
            state["action_net.bias"] = torch.tensor([0.0, 0.0, 5.0])
            torch.save(state, path)

            from_payload = build_teacher(payload)
            from_replaced_path = build_teacher(path)
            obs = torch.zeros((1, 4))
            self.assertEqual(int(from_payload(obs).argmax()), 0)
            self.assertEqual(int(from_replaced_path(obs).argmax()), 2)

    def test_bc_loader_rejects_nonfinite_weights(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp_extractor = torch.nn.Module()
                self.mlp_extractor.policy_net = torch.nn.Sequential(
                    torch.nn.Linear(4, 2), torch.nn.Tanh(), torch.nn.Linear(2, 2))
                self.action_net = torch.nn.Linear(2, 3)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "policy.pt"
            state = Policy().state_dict()
            state["action_net.bias"][0] = float("inf")
            torch.save(state, path)
            expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch.object(
                    train_module, "_validate_bc_report",
                    return_value={"policy_sha256": expected_sha}):
                with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                    _load_bc_state_dict(str(path), Policy(), "data_gate")

    def test_bc_loader_rechecks_expected_sha_on_its_load_bytes(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp_extractor = torch.nn.Module()
                self.mlp_extractor.policy_net = torch.nn.Sequential(
                    torch.nn.Linear(4, 2), torch.nn.Tanh(), torch.nn.Linear(2, 2))
                self.action_net = torch.nn.Linear(2, 3)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "policy.pt"
            original_state = Policy().state_dict()
            torch.save(original_state, path)
            expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            replacement = pathlib.Path(directory) / "replacement.pt"
            changed_state = {k: v.clone() for k, v in original_state.items()}
            changed_state["action_net.bias"][0] += 1.0
            torch.save(changed_state, replacement)

            def replace_after_gate(policy_path, gate):
                del gate
                policy_path.write_bytes(replacement.read_bytes())
                return {"policy_sha256": expected_sha}

            with mock.patch.object(
                    train_module, "_validate_bc_report",
                    side_effect=replace_after_gate):
                with self.assertRaisesRegex(ValueError, "闸门校验后发生漂移"):
                    _load_bc_state_dict(str(path), Policy(), "data_gate")

    def test_worker_holdout_is_split_by_episode(self):
        groups = np.repeat(np.arange(40), np.arange(1, 41))
        train, holdout, held_episodes = split_by_episode(groups)
        self.assertTrue(len(train) and len(holdout) and len(held_episodes))
        self.assertTrue(set(groups[train]).isdisjoint(groups[holdout]))

    def test_attention_shape_contract_and_finite_backward(self):
        self.assertEqual(_GEAR_PRESENT_INDEX, 293)
        with self.assertRaises(ValueError):
            EntityAttentionExtractor(
                spaces.Box(-np.inf, np.inf, shape=(296,), dtype=np.float32))
        extractor = EntityAttentionExtractor(
            spaces.Box(-np.inf, np.inf, shape=(295,), dtype=np.float32),
            features_dim=32)
        obs = torch.zeros((2, 295), requires_grad=True)
        out = extractor(obs)
        self.assertEqual(tuple(out.shape), (2, 32))
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in extractor.parameters()))

    def test_final_status_uses_global_resume_target(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            callback = EpisodeJsonlCallback(run_dir, {
                "total_steps": 499_712,
                "start_steps": 7_010_304,
                "target_global_steps": 7_510_016,
            })
            callback.num_timesteps = 7_510_016
            callback._steps0 = 7_010_304
            callback.t0 = time.time() - 1
            callback._on_training_end()
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["total_steps"], 7_510_016)
            self.assertEqual(status["target_steps"], 7_510_016)
            self.assertEqual(status["leg_steps"], 499_712)
            self.assertTrue(status["training_ended"])
            self.assertFalse(status["rollout_full"])
            self.assertTrue(status["target_reached"])

    def test_cli_help_and_reserved_seed_guard(self):
        help_run = subprocess.run(
            [sys.executable, str(ROOT / "train" / "train_ppo.py"), "--help"],
            text=True, capture_output=True, check=False)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("41.5%,20%", help_run.stdout)

        bad_seed = subprocess.run(
            [sys.executable, str(ROOT / "train" / "train_ppo.py"),
             "--total-steps", "2048", "--seed", "9000"],
            text=True, capture_output=True, check=False)
        self.assertNotEqual(bad_seed.returncode, 0)
        self.assertIn("种子纪律", bad_seed.stderr)

        non_quantized = subprocess.run(
            [sys.executable, str(ROOT / "train" / "train_ppo.py"),
             "--total-steps", "1", "--num-envs", "1", "--n-steps", "8"],
            text=True, capture_output=True, check=False)
        self.assertNotEqual(non_quantized.returncode, 0)
        self.assertIn("整除", non_quantized.stderr)

    def test_zero_beta_update_matches_upstream_maskable_ppo(self):
        env = DummyVecEnv([TinyMaskedEnv])
        kwargs = dict(n_steps=8, batch_size=8, seed=7, device="cpu", verbose=0)
        leashed = LeashedMaskablePPO(
            "MlpPolicy", env, distill_beta=0.0, **kwargs)
        upstream = MaskablePPO("MlpPolicy", env, **kwargs)
        upstream.policy.load_state_dict(leashed.policy.state_dict())
        for model in (leashed, upstream):
            model._setup_learn(total_timesteps=8)
            self._fill_masked_buffer(model)
        torch.manual_seed(99)
        np.random.seed(99)
        leashed.train()
        torch.manual_seed(99)
        np.random.seed(99)
        upstream.train()
        for key, tensor in leashed.policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, upstream.policy.state_dict()[key]), key)
        env.close()

    def test_leash_rejects_invalid_beta_missing_teacher_and_empty_masks(self):
        env = DummyVecEnv([TinyMaskedEnv])
        kwargs = dict(n_steps=8, batch_size=8, n_epochs=1,
                      seed=7, device="cpu", verbose=0)
        try:
            for invalid in (-0.1, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "有限非负数"):
                    LeashedMaskablePPO(
                        "MlpPolicy", env, distill_beta=invalid, **kwargs)

            missing = LeashedMaskablePPO(
                "MlpPolicy", env, distill_beta=1.0, **kwargs)
            missing._setup_learn(total_timesteps=8)
            self._fill_masked_buffer(missing)
            with self.assertRaisesRegex(RuntimeError, "教师未挂载"):
                missing.train()

            with tempfile.TemporaryDirectory() as directory:
                teacher = pathlib.Path(directory) / "teacher.pt"
                _write_tiny_teacher(teacher)
                guarded = LeashedMaskablePPO(
                    "MlpPolicy", env, distill_beta=1.0,
                    teacher_path=str(teacher), **kwargs)
                obs = torch.zeros((2, 4))
                masks = torch.tensor([[True, True, False],
                                      [True, False, True]])
                probs = guarded._teacher_probs(obs, masks)
                self.assertEqual(float(probs[0, 2]), 0.0)
                self.assertEqual(float(probs[1, 1]), 0.0)
                fake_logp = torch.full_like(probs, HUGE_NEG)
                self.assertTrue(torch.isfinite(-(probs * fake_logp).sum(-1)).all())
                with self.assertRaisesRegex(ValueError, "全 False"):
                    guarded._teacher_probs(
                        obs, torch.zeros((2, 3), dtype=torch.bool))
        finally:
            env.close()

    def test_gcal_trip_does_not_update_the_triggering_minibatch(self):
        env = DummyVecEnv([TinyMaskedEnv])
        try:
            with tempfile.TemporaryDirectory() as directory:
                teacher = pathlib.Path(directory) / "teacher.pt"
                calibration = pathlib.Path(directory) / "calib.jsonl"
                _write_tiny_teacher(teacher, preferred_action=2)
                model = LeashedMaskablePPO(
                    "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=2,
                    distill_beta=1.0, teacher_path=str(teacher),
                    calib_probes=[0], calib_out=str(calibration),
                    seed=7, device="cpu", verbose=0)
                # Force the student to choose action 0 while the teacher chooses
                # action 2, making the >20% calibration trip deterministic.
                with torch.no_grad():
                    model.policy.action_net.weight.zero_()
                    model.policy.action_net.bias.copy_(
                        torch.tensor([10.0, -10.0, -10.0]))
                model._setup_learn(total_timesteps=8)
                self._fill_masked_buffer(model, seed=14)
                before = {key: value.detach().clone()
                          for key, value in model.policy.state_dict().items()}
                model.train()
                self.assertTrue(model._calib_tripped)
                self.assertTrue(calibration.is_file())
                for key, value in model.policy.state_dict().items():
                    self.assertTrue(torch.equal(before[key], value), key)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
