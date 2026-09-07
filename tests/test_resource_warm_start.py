"""Exact parent weight initialization and inherited PG audit (synthetic only)."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import migrate_resource_candidate as migration
import train_ppo as training
from leashed_ppo import LeashedMaskablePPO, validate_worker_onpolicy_pg_receipt
from test_critic_migration import _FormalPgReceiptEnv

PARENT = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
IMPL = "a" * 64


@unittest.skipUnless(PARENT.is_file(), "registered R16 fixture is unavailable")
class ResourceWarmStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.parent_payload = PARENT.read_bytes()
        if hashlib.sha256(cls.parent_payload).hexdigest() != migration.PARENT_SHA256:
            raise RuntimeError("Registered R16 test fixture SHA changed")
        with zipfile.ZipFile(io.BytesIO(cls.parent_payload)) as archive:
            cls.parent_data = json.loads(archive.read("data"))
        cls.source = cls.parent_data["diablogym_contract"]
        cls.target = migration.target_contract(cls.source, IMPL)
        model = LeashedMaskablePPO.load(io.BytesIO(cls.parent_payload), device="cpu",
                                       teacher_path=None, teacher_sha256=None)
        cls.parameters_sha = migration.policy_sha256(model)
        cls.before_probe = migration.policy_probe(model)
        cls.receipt = migration.make_receipt(cls.parent_data, cls.target, cls.parameters_sha)
        migration.reset_to_initialization(model, cls.receipt, cls.target, 2124000)
        buffer = io.BytesIO()
        model.save(buffer)
        cls.warm_payload = buffer.getvalue()
        cls.manifest = {"receipt": cls.receipt, "policy_probe": cls.before_probe}

    def fresh(self, env=None):
        return migration.load_initialization(self.warm_payload, self.manifest, env=env)

    def test_only_registered_source_and_four_resource_fields_migrate(self):
        changed = {k for k in set(self.source) | set(self.target)
                   if self.source.get(k) != self.target.get(k)}
        self.assertEqual(changed, migration.BASE_ALLOWED_CONTRACT_KEYS)
        self.assertEqual(migration.json_sha256(self.source), migration.PARENT_CONTRACT_SHA256)
        for field, value in (("action_n", 16), ("gamma", 0.99), ("max_steps", 10000),
                             ("worker_hp_loss_price", 0), ("distill_beta", 1)):
            invalid = {**self.target, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contract drift"):
                migration.validate_target_contract(self.source, invalid, IMPL)
        with self.assertRaisesRegex(ValueError, "exact registered R16"):
            migration.target_contract({**self.source, "max_steps": 3000}, IMPL)

    def test_v3_migration_is_separately_bound_without_loosening_any_other_fields(self):
        v3 = migration.target_contract(self.source, IMPL, "sustain-v3")
        receipt = migration.make_receipt(self.parent_data, v3, self.parameters_sha)
        migration.validate_inherited_receipt(receipt, v3)
        self.assertEqual(receipt["operation"], "r16-to-sustain-v3-weights-only-v1")
        changed = {key for key in set(self.source) | set(v3)
                   if self.source.get(key) != v3.get(key)}
        self.assertEqual(changed, migration.BASE_ALLOWED_CONTRACT_KEYS)
        self.assertEqual(v3["farm_scene_cap"], 3600)
        self.assertEqual(v3["resource_service_recipe"]["collect_microstep_cap"], 300)
        for invalid in (self.target, {**v3, "resource_service_recipe": self.target["resource_service_recipe"]},
                        {**v3, "farm_scene_cap": 1800}):
            with self.subTest(invalid=invalid["resource_service_policy"]), self.assertRaises(ValueError):
                migration.validate_inherited_receipt(receipt, invalid)
        with self.assertRaises(ValueError):
            migration.validate_inherited_receipt(self.receipt, v3)
        model = self.fresh()
        migration.reset_to_initialization(model, receipt, v3, 2164000)
        self.assertEqual(migration.policy_sha256(model), self.parameters_sha)
        self.assertEqual(model._resume_lineage["operation"], receipt["operation"])
        model._assert_critic_migration_contract()
        with self.assertRaisesRegex(ValueError, "resource_service"):
            training._validate_resume_contract(self.target, v3, allow_environment_restart=True)
        with self.assertRaisesRegex(ValueError, "Only explicit"):
            migration.target_contract(self.source, IMPL, "sustain-v999")

    def test_v4_is_an_exact_distinct_migration_and_v3_receipts_cannot_be_relabelled(self):
        v3 = migration.target_contract(self.source, IMPL, "sustain-v3", "adjacent-v1")
        v4 = migration.target_contract(self.source, IMPL, "sustain-v4", "adjacent-v1")
        receipt3 = migration.make_receipt(self.parent_data, v3, self.parameters_sha)
        receipt4 = migration.make_receipt(self.parent_data, v4, self.parameters_sha)
        self.assertEqual(receipt4["operation"],
                         "r16-to-sustain-v4-dive-adjacent-v1-weights-only-v1")
        self.assertEqual(migration.operation_for("sustain-v4"),
                         "r16-to-sustain-v4-weights-only-v1")
        self.assertEqual({key for key in set(v3) | set(v4) if v3.get(key) != v4.get(key)},
                         {"resource_service_policy", "resource_service_recipe"})
        self.assertEqual(v4["resource_service_recipe"]["collect_command_window_microsteps"], 450)
        self.assertNotIn("collect_microstep_cap", v4["resource_service_recipe"])
        self.assertEqual(v4["max_steps"], 6000)
        self.assertEqual(v4["farm_scene_cap"], 3600)
        for receipt, wrong in ((receipt3, v4), (receipt4, v3)):
            with self.assertRaises(ValueError):
                migration.validate_inherited_receipt(receipt, wrong)
        wrong = deepcopy(v4)
        wrong["resource_service_recipe"]["collect_command_window_microsteps"] = 300
        with self.assertRaisesRegex(ValueError, "contract drift"):
            migration.validate_target_contract(self.source, wrong, IMPL)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                training, "_implementation_bundle_sha256", return_value=IMPL):
            output = Path(directory) / "v4-adjacent-initialization"
            manifest = migration.migrate(PARENT, parent_sha256=migration.PARENT_SHA256,
                output_dir=output, implementation_sha256=IMPL, seed=2164000,
                service_policy="sustain-v4", dive_blocker_recovery="adjacent-v1")
            payload, data, restored = migration.capture_initialization(output / "manifest.json", IMPL)
            model = migration.load_initialization(payload, restored)
            migration.validate_zero_state(model)
            self.assertEqual(migration.policy_sha256(model), self.parameters_sha)
            self.assertEqual(migration.policy_probe(model), self.before_probe)
            self.assertEqual(data["diablogym_contract"], v4)
            manifest["operation"] = receipt3["operation"]
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "false lineage"):
                migration.capture_initialization(output / "manifest.json", IMPL)

    def test_v3_save_load_manifest_is_consistent_and_cannot_be_relabelled_v2(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                training, "_implementation_bundle_sha256", return_value=IMPL):
            output = Path(directory) / "v3-initialization"
            manifest = migration.migrate(PARENT, parent_sha256=migration.PARENT_SHA256,
                output_dir=output, implementation_sha256=IMPL, seed=2164000,
                service_policy="sustain-v3")
            self.assertEqual(manifest["operation"], "r16-to-sustain-v3-weights-only-v1")
            payload, data, restored_manifest = migration.capture_initialization(output / "manifest.json", IMPL)
            self.assertEqual(data["diablogym_contract"]["resource_service_policy"], "sustain-v3")
            model = migration.load_initialization(payload, restored_manifest)
            self.assertEqual(migration.policy_sha256(model), self.parameters_sha)
            manifest["operation"] = migration.OPERATION
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "false lineage"):
                migration.capture_initialization(output / "manifest.json", IMPL)

    def test_adjacent_recovery_adds_only_one_named_key_and_separate_recipe(self):
        for policy in ("sustain-v2", "sustain-v3", "sustain-v4"):
            off = migration.target_contract(self.source, IMPL, policy)
            on = migration.target_contract(self.source, IMPL, policy, "adjacent-v1")
            receipt = migration.make_receipt(self.parent_data, on, self.parameters_sha)
            migration.validate_inherited_receipt(receipt, on)
            self.assertEqual({key for key in set(off) | set(on) if off.get(key) != on.get(key)},
                             {"dive_blocker_recovery"})
            self.assertEqual(off["resource_service_recipe"], on["resource_service_recipe"])
            self.assertEqual(receipt["operation"], f"r16-to-{policy}-dive-adjacent-v1-weights-only-v1")
            self.assertEqual(receipt["dive_blocker_recovery_recipe"]["core_macro_microstep_cap"], 12)
            changed = {key for key in set(self.source) | set(on) if self.source.get(key) != on.get(key)}
            self.assertEqual(changed, migration.ALLOWED_CONTRACT_KEYS)
            with self.assertRaises(ValueError):
                migration.validate_inherited_receipt(receipt, off)
            bad = deepcopy(receipt)
            bad["dive_blocker_recovery_recipe"]["core_macro_microstep_cap"] = 24
            with self.assertRaisesRegex(ValueError, "DIVE recovery recipe"):
                migration.validate_inherited_receipt(bad, on)
            with self.assertRaises(ValueError):
                migration.validate_target_contract(self.source, {**on, "farm_scene_cap": 1800}, IMPL)

    def test_adjacent_recovery_manifest_roundtrip_cannot_masquerade_as_pure_gold(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                training, "_implementation_bundle_sha256", return_value=IMPL):
            output = Path(directory) / "v3-adjacent-initialization"
            manifest = migration.migrate(PARENT, parent_sha256=migration.PARENT_SHA256,
                output_dir=output, implementation_sha256=IMPL, seed=2164000,
                service_policy="sustain-v3", dive_blocker_recovery="adjacent-v1")
            payload, data, restored_manifest = migration.capture_initialization(output / "manifest.json", IMPL)
            model = migration.load_initialization(payload, restored_manifest)
            self.assertEqual(migration.policy_sha256(model), self.parameters_sha)
            self.assertEqual(data["diablogym_contract"]["dive_blocker_recovery"], "adjacent-v1")
            self.assertEqual(manifest["dive_blocker_recovery_recipe"], manifest["receipt"]["dive_blocker_recovery_recipe"])
            del manifest["dive_blocker_recovery_recipe"]
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest DIVE"):
                migration.capture_initialization(output / "manifest.json", IMPL)

    def test_zero_counters_empty_adam_same_policy_and_historical_lineage(self):
        model = self.fresh()
        migration.validate_zero_state(model)
        self.assertEqual(migration.policy_sha256(model), self.parameters_sha)
        self.assertEqual(migration.policy_probe(model), self.before_probe)
        self.assertEqual(model._resume_lineage["immediate_parent_num_timesteps"], 4089856)
        self.assertEqual(model._critic_warmup_rollouts_completed, 0)
        self.assertEqual(model._critic_warmup_optimizer_steps_completed, 0)
        self.assertTrue(model._worker_onpolicy_pg_audit_required)
        self.assertFalse(model._prepare_main_ppo_rollout())
        self.assertTrue(model._resource_warm_start_receipt["historical_critic_warmup"]["_critic_warmup_completed"])

    def test_initialization_is_not_a_fake_trained_resume_boundary(self):
        data = training._validate_checkpoint_bytes(self.warm_payload, "warm-test", True)
        with self.assertRaisesRegex(ValueError, "optimizer|rollout"):
            training._validate_resumable_leashed_boundary(data)

    def test_forged_lineage_or_current_warmup_cannot_bypass_default_guard(self):
        cases = ("receipt", "parent", "target", "warmup", "tensor", "audit")
        for case in cases:
            model = self.fresh()
            if case == "receipt":
                del model._resource_warm_start_receipt
            elif case == "parent":
                model._resource_warm_start_receipt["parent_checkpoint_sha256"] = "b" * 64
            elif case == "target":
                model.diablogym_contract["max_steps"] = 10000
            elif case == "warmup":
                model._critic_warmup_optimizer_steps_completed = 640
            elif case == "tensor":
                with torch.no_grad():
                    next(model.policy.parameters()).view(-1)[0].add_(1)
            else:
                model._worker_onpolicy_pg_audit_required = False
            with self.subTest(case=case), self.assertRaises((ValueError, RuntimeError)):
                model._assert_critic_migration_contract()

    def test_cli_warm_start_disallows_resume_and_override_shortcuts(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}")
            baseline = dict(resource_warm_start=str(manifest), worker=True, algo="mppo",
                device="cpu", seed=2124000, distill_beta=0.0, bc_aux_lambda=0.0,
                resource_protocol="l2-town-v1", resource_purchase_mode="full",
                resource_service_policy="sustain-v2")
            training._validate_resource_warm_start_args(SimpleNamespace(**baseline))
            for field, value in (("resume_from", "old.zip"), ("reset_optimizer", True),
                    ("allow_environment_restart_resume", True), ("reset_worker_critic", True),
                    ("resource_purchase_mode", "armor"), ("skip_dry", True),
                    ("deep_start_curriculum", "2:1"), ("distill_beta", 0.1)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    training._validate_resource_warm_start_args(SimpleNamespace(**{**baseline, field: value}))

    def test_current_cli_contract_can_match_target_without_any_drift_allowance(self):
        class Parsed(Exception):
            pass
        captured = {}
        def capture(args):
            captured["args"] = args
            raise Parsed()
        with patch.object(sys, "argv", ["train_ppo.py"]), patch.object(
                training, "_validate_args", side_effect=capture):
            with self.assertRaises(Parsed):
                training._main(SimpleNamespace())
        args = captured["args"]
        settings = dict(worker=True, options=False, algo="mppo", arch="mlp", device="cpu",
            max_steps=6000, num_envs=4, n_steps=512, gamma=1.0, lr=0.0001,
            ent_coef=0.005, distill_beta=0.0, skip_dry=False, no_drink_sovereignty=False,
            worker_fast_forward_reward_credit="terminal-death-only",
            worker_learning_window_scope="farm-dive-v1", worker_dive_action11_logit_bonus=2.0,
            worker_potion_action13_logit_bonus=2.0, explore_global_hunt=True,
            farm_scene_cap=3600, reset_layer_clock_on_window=True,
            worker_hp_loss_price=0.1, worker_potion_pickup_bonus=2.0,
            worker_no_progress_timeout_credit="zero", worker_descend_escrow_fraction=0.5,
            worker_descend_escrow_readiness_gate=True, worker_descend_escrow_power=1.6,
            worker_policy_observation_view="dual-v4-asymmetric-v3",
            worker_action14_logit_bonus=2.5, gradient_clip_mode="separate-root-context-critic-v2",
            artifact_scope="candidate", target_kl=0.01, reward_economy="v4",
            manager_heuristic="readiness-v3", resource_protocol="l2-town-v1",
            resource_purchase_mode="full", resource_service_policy="sustain-v2",
            resource_warm_start="manifest.json")
        for key, value in settings.items():
            setattr(args, key, value)
        args.seed = 2164000
        args.total_steps = 8192
        args.run_name = "r20-resource-engineering-test"
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text("{}")
            args.resource_warm_start = str(manifest_path)
            training._validate_args(args)
            self.assertIsNone(args.manager_npz)
            args.seed = 2124000  # reserved eval bank; never a game training seed
            with self.assertRaisesRegex(ValueError, "种子纪律"):
                training._validate_args(args)
            args.seed = 2164000
        for policy in ("sustain-v2", "sustain-v3", "sustain-v4"):
            for recovery in ("off", "adjacent-v1"):
                args.resource_service_policy = policy
                args.dive_blocker_recovery = recovery
                target = migration.target_contract(self.source, IMPL, policy, recovery)
                receipt = migration.make_receipt(self.parent_data, target, self.parameters_sha)
                current = training._training_contract(args, self.fresh(), 256, implementation_sha256=IMPL)
                with self.subTest(policy=policy, recovery=recovery):
                    training._validate_resume_contract(target, current)
                    migration.validate_inherited_receipt(receipt, current)

    def test_complete_migration_roundtrip_and_output_is_never_overwritten(self):
        original_sha = hashlib.sha256(PARENT.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                training, "_implementation_bundle_sha256", return_value=IMPL):
            output = Path(directory) / "new-candidate"
            manifest = migration.migrate(PARENT, parent_sha256=migration.PARENT_SHA256,
                output_dir=output, implementation_sha256=IMPL, seed=2124000)
            self.assertFalse(manifest["ordinary_resume_eligible"])
            payload, _data, read = migration.capture_initialization(output / "manifest.json", IMPL)
            restored = migration.load_initialization(payload, read)
            self.assertEqual(migration.policy_sha256(restored), self.parameters_sha)
            with self.assertRaisesRegex(ValueError, "already exists"):
                migration.migrate(PARENT, parent_sha256=migration.PARENT_SHA256,
                    output_dir=output, implementation_sha256=IMPL, seed=2124000)
            model_path = output / "model_warm_start.zip"
            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                migration.capture_initialization(output / "manifest.json", IMPL)
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), original_sha)

    def test_wrong_parent_is_rejected_before_model_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged.zip"
            path.write_bytes(b"untrusted")
            with patch.object(LeashedMaskablePPO, "load") as load:
                with self.assertRaisesRegex(ValueError, "Parent checkpoint SHA"):
                    migration.migrate(path, parent_sha256=migration.PARENT_SHA256,
                        output_dir=Path(directory) / "candidate", implementation_sha256=IMPL, seed=1)
                load.assert_not_called()

    def test_first_synthetic_rollout_updates_actor_and_real_pg_receipt_without_warmup(self):
        # Keep the exact registered 512 x 4 geometry; these are synthetic states,
        # not Diablo trajectories and not a candidate training result.
        space = self.fresh().observation_space
        def make_env():
            env = _FormalPgReceiptEnv(diverse_rewards=True)
            env.observation_space = space
            return env
        env = DummyVecEnv([make_env for _ in range(4)])
        try:
            model = self.fresh(env)
            model.tensorboard_log = None
            model.verbose = 0
            model.learn(total_timesteps=2048)
            self.assertEqual(model.num_timesteps, 2048)
            self.assertGreater(model._ppo_optimizer_steps_completed, 0)
            self.assertGreater(model._actor_optimizer_steps_completed, 0)
            self.assertEqual(model._critic_warmup_optimizer_steps_completed, 0)
            self.assertEqual(model._critic_warmup_rollouts_completed, 0)
            self.assertEqual(model._worker_onpolicy_pg_joint_rollouts, 1)
            receipt = model._worker_onpolicy_pg_rollout_receipts[0]
            self.assertTrue(validate_worker_onpolicy_pg_receipt(receipt, expected_samples=2048))
            self.assertGreater(receipt["optimizer_steps"], 0)
            self.assertEqual(model._last_completed_ppo_rollout_steps, 2048)
            self.assertNotEqual(migration.policy_sha256(model), self.parameters_sha)
            # Inherited warmup must not skip the final real-PG publication gate.
            with patch.object(training, "_asymmetric_worker_deployment_evidence_complete", return_value=True), \
                    patch("leashed_ppo.worker_onpolicy_pg_audit_complete", return_value=False):
                self.assertFalse(training._is_exact_training_completion(model, 2048))
            with patch.object(training, "_asymmetric_worker_deployment_evidence_complete", return_value=True), \
                    patch("leashed_ppo.worker_onpolicy_pg_audit_complete", return_value=True):
                self.assertTrue(training._is_exact_training_completion(model, 2048))
            # A learned child becomes an ordinary trained-boundary artifact;
            # the initialization manifest can never be reused for it.
            output = io.BytesIO()
            model.save(output)
            child_data = training._validate_checkpoint_bytes(output.getvalue(), "synthetic-child", True)
            training._validate_resumable_leashed_boundary(child_data)
            restored = LeashedMaskablePPO.load(io.BytesIO(output.getvalue()), device="cpu",
                teacher_path=None, teacher_sha256=None)
            restored._assert_critic_migration_contract()
            self.assertEqual(restored._worker_onpolicy_pg_joint_rollouts, 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
