"""Core regression tests for the full-game Worker critic migration."""

from __future__ import annotations

import copy
import hashlib
import pathlib
import random
import sys
import tempfile
import types
import unittest

import gymnasium as gym
import numpy as np
import platform as _platform

# 前向 logits 原始字节的平台取证 KAT(matmul/tanh 归约序随 BLAS/SIMD 而异,
# 迁移正确性由 helper 内 torch.equal(source,target) 保证;SHA 仅锚定各平台数值):
# - darwin arm64 (Mac Accelerate): 原始取证值(audit-r7-dual-smoke 07-25 在案)
# - linux x86_64 (torch 2.12.1+cpu MKL/AVX): 2026-07-27 WSL2 移植取证值
BITWISE_PROBE_SHA_BY_PLATFORM = {
    ("darwin", "arm64"):
        "67d9bd852faf53af457848d9730b7693c0e9219f098801ae3407cd4e782038c8",
    ("linux", "x86_64"):
        "cb7ca81617fbb16792903dba8456d6d29abb2d7ef353d6e3d83e509f1bfec0a1",
}
EXPECTED_BITWISE_PROBE_SHA = BITWISE_PROBE_SHA_BY_PLATFORM.get(
    (_platform.system().lower(), _platform.machine().lower()))

import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from leashed_ppo import (  # noqa: E402
    ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
    ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
    ASYMMETRIC_WORKER_FROZEN_ACTOR_PARAMETER_COUNT,
    ASYMMETRIC_WORKER_FROZEN_ACTOR_TENSOR_COUNT,
    ASYMMETRIC_WORKER_FROZEN_CONTEXT_PARAMETER_COUNT,
    ASYMMETRIC_WORKER_FROZEN_CONTEXT_TENSOR_COUNT,
    ASYMMETRIC_WORKER_FROZEN_CRITIC_PARAMETER_COUNT,
    ASYMMETRIC_WORKER_FROZEN_CRITIC_TENSOR_COUNT,
    ASYMMETRIC_WORKER_OBSERVATION_DIM,
    ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA,
    ASYMMETRIC_WORKER_SKIP_DRY_FEATURE,
    WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
    WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
    GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
    GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
    AsymmetricWorkerMaskableActorCriticPolicy,
    LeashedMaskablePPO,
    _StructuredWorkerContextEncoder,
    _legacy_worker_observation_view,
    _legacy_distillation_masks,
    _worker_onpolicy_pg_receipt_qualifies,
    actor_parameter_sha256,
    asymmetric_worker_runtime_evidence,
    clip_actor_critic_gradients,
    worker_onpolicy_pg_audit_complete,
    critic_parameter_sha256,
    strict_actor_critic_parameter_partition,
    validate_worker_onpolicy_pg_receipt,
)
from diablogym.controller_wire import (  # noqa: E402
    CONTROLLER_SNAPSHOT_MAP_CHANNELS,
    DUAL_WORKER_LAYOUT,
    DUAL_WORKER_LAYOUT_SHA256,
)
import train_ppo  # noqa: E402


class _TinyMaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self):
        return np.ones(3, dtype=bool)


class _DualMaskEnv(_TinyMaskEnv):
    def __init__(self):
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(ASYMMETRIC_WORKER_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(15)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(
            ASYMMETRIC_WORKER_OBSERVATION_DIM, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(
                ASYMMETRIC_WORKER_OBSERVATION_DIM, dtype=np.float32),
            0.0, False, False, {},
        )

    def action_masks(self):
        return np.ones(15, dtype=bool)


class _LegacyMaskEnv(_DualMaskEnv):
    def __init__(self):
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(298,), dtype=np.float32)
        self.action_space = spaces.Discrete(15)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        return np.zeros(298, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(298, dtype=np.float32), 0.0, False, False, {}


class _FormalPgReceiptEnv(_DualMaskEnv):
    def __init__(self, *, diverse_rewards=True, combat_effects=True):
        super().__init__()
        self.diverse_rewards = bool(diverse_rewards)
        self.combat_effects = bool(combat_effects)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        self.steps = 0
        return self._observation(), {}

    def _observation(self):
        observation = np.zeros(
            ASYMMETRIC_WORKER_OBSERVATION_DIM, dtype=np.float32)
        observation[0] = float(self.steps % 5) / 4.0
        fuse = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "fuse_streak")
        observation[
            fuse.start + (self.steps % fuse.width)
        ] = 1.0
        return observation

    def step(self, action):
        self.steps += 1
        reward = (
            (
                float(3 + (self.steps % 2))
                if int(action) == 9
                else -float(1 + (self.steps % 2))
            )
            if self.diverse_rewards else 0.0
        )
        return (
            self._observation(),
            reward,
            False,
            False,
            _formal_pg_info(
                int(action),
                reward,
                combat=(
                    self.combat_effects
                    and int(action) == 9
                ),
            ),
        )

    def action_masks(self):
        mask = np.zeros(15, dtype=bool)
        mask[[8, 9]] = True
        return mask


class _FormalPgTimeLimitEnv(_FormalPgReceiptEnv):
    def step(self, action):
        observation, reward, _terminated, _truncated, info = (
            super().step(action))
        return observation, reward, False, True, info


def _formal_pg_info(action, reward, *, combat=False):
    action = int(action)
    reasons = ("target_damage",) if combat else ("move",)
    return {
        "requested_action": action,
        "executed_action": action,
        "worker_wage": float(reward),
        "transition_reward": float(reward),
        "worker_no_progress_timeout": False,
        "no_progress_timeout_base_failure_reward": 0.0,
        "no_progress_timeout_additional_failure_reward": 0.0,
        "no_progress_timeout_failure_reward": 0.0,
        "action_effect_audit": {
            "requested_action": action,
            "native_attempts": 1,
            "native_accepts": 1,
            "request_executed": True,
            "material_effect": True,
            "effect_reasons": reasons,
            "same_scene": True,
            "stall_cost_applied": False,
        },
        "overridden": False,
        "fuse_tripped": False,
    }


def _model(*, n_steps=2):
    return LeashedMaskablePPO(
        "MlpPolicy",
        _TinyMaskEnv(),
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        learning_rate=1e-3,
        seed=17,
        device="cpu",
        verbose=0,
    )


def _asymmetric_model(
        *, n_steps=2, seed=17, action14_logit_bonus=0.0):
    return LeashedMaskablePPO(
        AsymmetricWorkerMaskableActorCriticPolicy,
        _DualMaskEnv(),
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        learning_rate=1e-3,
        policy_kwargs={
            "action14_logit_bonus": float(action14_logit_bonus),
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )


def _branch_loss(model):
    actor = (
        model.policy.action_net.weight.sum()
        + model.policy.action_net.bias.sum()
    )
    critic = (
        model.policy.value_net.weight.sum()
        + model.policy.value_net.bias.sum()
    )
    return actor + critic


class _DummyMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = torch.nn.Linear(2, 2)
        self.value_net = torch.nn.Linear(2, 2)


class _TrainableSharedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features_extractor = torch.nn.Linear(2, 2)
        self.mlp_extractor = _DummyMlp()
        self.action_net = torch.nn.Linear(2, 2)
        self.value_net = torch.nn.Linear(2, 1)
        self.share_features_extractor = True
        self.optimizer = torch.optim.Adam(self.parameters())


class CriticMigrationTests(unittest.TestCase):
    @staticmethod
    def _formal_pg_audit_model():
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(diverse_rewards=True),
            n_steps=4,
            batch_size=4,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            seed=1701,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        return model

    def test_dual_resume_classifies_one_time_migration_and_native_continuation(self):
        args = types.SimpleNamespace(
            worker=True,
            worker_policy_observation_view=
                train_ppo._WORKER_VIEW_DUAL_V4_ASYMMETRIC,
            legacy_worker_policy_observation_view=False,
            bc_aux_graft=False,
            bc_aux_lambda=0.0,
            bc_aux_demos=None,
            reset_worker_critic=True,
            allow_legacy_resume=True,
            allow_environment_restart_resume=False,
        )
        self.assertEqual(
            train_ppo._classify_dual_worker_resume(args, {}),
            "legacy-actor-migration",
        )

        args.reset_worker_critic = False
        args.allow_legacy_resume = False
        args.allow_environment_restart_resume = True
        runtime = train_ppo._current_asymmetric_worker_runtime_evidence()
        max_grad_norm = float(
            train_ppo._ALGORITHM_RECIPE["max_grad_norm"])
        source_checkpoint_sha256 = "a" * 64
        source_actor_sha256 = "b" * 64
        contract = {
            "schema_version": 2,
            "contract_revision": train_ppo._CONTRACT_REVISION,
            "mode": "worker",
            "worker_policy_observation_view":
                train_ppo._WORKER_VIEW_DUAL_V4_ASYMMETRIC,
            "worker_episode_boundary":
                train_ppo._WORKER_EPISODE_BOUNDARY_V24,
            "worker_window_bootstrap": "next-learning-window",
            "worker_no_progress_timeout":
                dict(train_ppo._WORKER_NO_PROGRESS_TIMEOUT_CONTRACT),
            "worker_action14_logit_bonus": 0.0,
            "distill_beta": 0.0,
            "demos_sha256": None,
            "policy_source_roles": {
                "schema": train_ppo._POLICY_SOURCE_ROLES_SCHEMA,
                "initialization": "resume-checkpoint",
                "bc_v1_direct_policy_uses": [],
                "bc_v1_dataset_uses": [],
                "distillation_teacher": "disabled",
                "worker_action14_policy_sources": [
                    "native-reward-bound-on-policy-ppo",
                ],
            },
            "observation_shape": [ASYMMETRIC_WORKER_OBSERVATION_DIM],
            "action_n": 15,
            "algorithm_recipe": {
                "max_grad_norm": max_grad_norm,
            },
            "actor_migration": {
                "method": train_ppo._ASYMMETRIC_ACTOR_INIT_METHOD,
                "context_architecture":
                    train_ppo._ASYMMETRIC_CONTEXT_ARCHITECTURE,
                "controller_layout_schema":
                    DUAL_WORKER_LAYOUT.schema,
                "controller_layout_sha256":
                    DUAL_WORKER_LAYOUT_SHA256,
                "target_actor_parameter_tensors":
                    runtime["policy"]["actor_tensor_count"],
                "target_actor_parameter_count":
                    runtime["policy"]["actor_parameter_count"],
                "context_parameter_tensors":
                    runtime["context"]["tensor_count"],
                "context_parameter_count":
                    runtime["context"]["parameter_count"],
                "context_initialization": {
                    "hidden": ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
                    "output":
                        "exact-zero-disabled-through-critic-warmup",
                },
                "actor_context_excluded_observation_features":
                    list(ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES),
                "source_checkpoint_sha256":
                    source_checkpoint_sha256,
                "source_actor_sha256": source_actor_sha256,
                "migrated_actor_sha256": "c" * 64,
            },
            "critic_migration": {
                "method": train_ppo._ASYMMETRIC_CRITIC_RESET_METHOD,
                "critic_architecture":
                    train_ppo._ASYMMETRIC_CRITIC_ARCHITECTURE,
                "controller_layout_schema":
                    DUAL_WORKER_LAYOUT.schema,
                "controller_layout_sha256":
                    DUAL_WORKER_LAYOUT_SHA256,
                "critic_parameter_tensors":
                    runtime["policy"]["critic_tensor_count"],
                "critic_parameter_count":
                    runtime["policy"]["critic_parameter_count"],
                "source_checkpoint_sha256":
                    source_checkpoint_sha256,
                "source_actor_sha256": source_actor_sha256,
                "warmup_steps": 4,
                "gradient_clip_mode":
                    GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
                "worker_onpolicy_pg_audit_schema":
                    WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
                "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout":
                    WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            },
            "gradient_clipping":
                train_ppo._root_context_critic_gradient_clipping(
                    max_grad_norm),
        }
        self.assertEqual(
            train_ppo._classify_dual_worker_resume(
                args, {"diablogym_contract": contract}),
            train_ppo._DUAL_ENV_RESTART_CONTINUATION,
        )
        drifted_runtime = copy.deepcopy(runtime)
        drifted_runtime["context"]["parameter_count"] -= 1
        with self.assertRaisesRegex(
                ValueError, "容量/张量拓扑漂移"):
            train_ppo._validate_current_dual_worker_contract(
                contract, runtime_evidence=drifted_runtime)
        args.allow_environment_restart_resume = False
        with self.assertRaisesRegex(
                ValueError, "allow-environment-restart-resume"):
            train_ppo._classify_dual_worker_resume(
                args, {"diablogym_contract": contract})
        args.allow_environment_restart_resume = True
        args.reset_worker_critic = True
        with self.assertRaisesRegex(ValueError, "禁止重复 reset critic"):
            train_ppo._classify_dual_worker_resume(
                args, {"diablogym_contract": contract})

    def test_native_dual_checkpoint_roundtrip_preserves_resume_contract(self):
        source_path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        payload = source_path.read_bytes()
        source_sha256 = hashlib.sha256(payload).hexdigest()
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-3,
            seed=991,
            device="cpu",
            verbose=0,
        )
        actor_receipt = train_ppo._initialize_asymmetric_worker_actor(
            model,
            source_checkpoint_payload=payload,
            source_checkpoint_sha256=source_sha256,
        )
        critic_receipt = train_ppo._reset_worker_critic(
            model,
            training_seed=991,
            source_checkpoint_sha256=source_sha256,
            source_actor_sha256=actor_receipt["source_actor_sha256"],
            source_critic_sha256=actor_receipt["source_critic_sha256"],
        )
        critic_receipt.update(
            model.configure_critic_migration(
                gradient_clip_mode=
                    GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
                critic_warmup_steps=2,
            ),
            optimizer_reset=True,
        )
        model._critic_migration_receipt = dict(critic_receipt)
        args = types.SimpleNamespace(
            worker=True,
            options=False,
            flat_clock=False,
            arch="mlp",
            max_steps=3000,
            num_envs=1,
            n_steps=2,
            gamma=1.0,
            lr=1e-3,
            ent_coef=0.0,
            distill_beta=0.0,
            calib_record_only=False,
            device="cpu",
            skip_dry=False,
            drink_sovereignty=False,
            worker_fast_forward_reward_credit="terminal-death-only",
            worker_additional_terminal_death_cost=32.0,
            legacy_worker_policy_observation_view=False,
            worker_policy_observation_view=
                train_ppo._WORKER_VIEW_DUAL_V4_ASYMMETRIC,
            manager_policy_observation_view="raw-v4",
            gradient_clip_mode=
                GRADIENT_CLIP_ROOT_CONTEXT_CRITIC_V2,
            artifact_scope="development",
            dry_curriculum_schedule=None,
            bc_aux_lambda=0.0,
            bc_aux_demos=None,
            bc_aux_graft=False,
            bc_aux_liveness_preflight=False,
            reset_worker_critic=False,
            allow_legacy_resume=False,
            allow_environment_restart_resume=True,
            resume_from=str(source_path),
        )
        contract = train_ppo._training_contract(
            args, model, batch_size=2,
            implementation_sha256="a" * 64,
        )
        model.diablogym_contract = contract
        # A resumable contracted checkpoint must prove that its latest
        # collected rollout has already reached an optimizer step.
        model.num_timesteps = 2
        model._last_completed_ppo_rollout_steps = 2
        model._ppo_optimizer_steps_completed = 1

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "native-dual.zip"
            model.save(path)
            _, data, _ = train_ppo._capture_leashed_checkpoint(path)
            restored = LeashedMaskablePPO.load(
                path,
                env=_DualMaskEnv(),
                device="cpu",
                teacher_path=None,
                teacher_sha256=None,
            )

        self.assertEqual(
            train_ppo._classify_dual_worker_resume(args, data),
            train_ppo._DUAL_ENV_RESTART_CONTINUATION,
        )
        self.assertEqual(
            restored._actor_migration_receipt,
            actor_receipt,
        )
        self.assertEqual(
            restored._critic_migration_receipt,
            critic_receipt,
        )
        current = train_ppo._training_contract(
            args, restored, batch_size=2,
            implementation_sha256="a" * 64,
        )
        train_ppo._validate_resume_contract(
            restored.diablogym_contract, current)
        self.assertEqual(current, contract)

    def test_asymmetric_source_validation_preserves_target_rng_streams(self):
        source_path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        payload = source_path.read_bytes()
        source_sha256 = hashlib.sha256(payload).hexdigest()
        preserved_states = []

        for seed in (2_130_000, 2_130_100):
            target = LeashedMaskablePPO(
                AsymmetricWorkerMaskableActorCriticPolicy,
                _DualMaskEnv(),
                n_steps=2,
                batch_size=2,
                n_epochs=1,
                learning_rate=1e-3,
                seed=seed,
                device="cpu",
                verbose=0,
            )
            python_before = random.getstate()
            numpy_before = np.random.get_state()
            torch_before = torch.random.get_rng_state().clone()

            receipt = train_ppo._initialize_asymmetric_worker_actor(
                target,
                source_checkpoint_payload=payload,
                source_checkpoint_sha256=source_sha256,
            )

            self.assertEqual(random.getstate(), python_before)
            numpy_after = np.random.get_state()
            self.assertEqual(numpy_after[0], numpy_before[0])
            np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
            self.assertEqual(numpy_after[2:], numpy_before[2:])
            self.assertTrue(torch.equal(
                torch.random.get_rng_state(), torch_before))
            self.assertIsNotNone(
                EXPECTED_BITWISE_PROBE_SHA,
                "本平台无取证 KAT:先取证 bitwise_probe_sha256 再入表")
            self.assertEqual(
                receipt["bitwise_probe_sha256"], EXPECTED_BITWISE_PROBE_SHA)
            preserved_states.append(
                (python_before, numpy_before, torch_before))

        self.assertNotEqual(
            preserved_states[0][0], preserved_states[1][0])
        self.assertFalse(np.array_equal(
            preserved_states[0][1][1], preserved_states[1][1][1]))
        self.assertFalse(torch.equal(
            preserved_states[0][2], preserved_states[1][2]))

    def test_asymmetric_migration_helper_signs_exact_real_v28_root(self):
        source_path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        payload = source_path.read_bytes()
        target = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-3,
            seed=23,
            device="cpu",
            verbose=0,
        )
        receipt = train_ppo._initialize_asymmetric_worker_actor(
            target,
            source_checkpoint_payload=payload,
            source_checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        )
        self.assertEqual(
            receipt["migrated_actor_sha256"],
            actor_parameter_sha256(
                target.policy, optimizer=target.policy.optimizer),
        )
        self.assertIsNotNone(
            EXPECTED_BITWISE_PROBE_SHA,
            "本平台无取证 KAT:先取证 bitwise_probe_sha256 再入表")
        self.assertEqual(
            receipt["bitwise_probe_sha256"], EXPECTED_BITWISE_PROBE_SHA)
        evidence = asymmetric_worker_runtime_evidence(target.policy)
        self.assertEqual(
            receipt["target_actor_parameter_tensors"],
            evidence["policy"]["actor_tensor_count"])
        self.assertEqual(
            receipt["context_parameter_tensors"],
            evidence["context"]["tensor_count"])
        self.assertLess(receipt["context_parameter_count"], 65_000)
        self.assertFalse(receipt["context_enabled"])
        self.assertGreater(receipt["context_hidden_nonzero"], 0)
        self.assertEqual(receipt["context_output_nonzero"], 0)
        self.assertEqual(
            receipt["schema"],
            train_ppo._ASYMMETRIC_ACTOR_INIT_SCHEMA)
        self.assertEqual(
            receipt["controller_layout_sha256"],
            DUAL_WORKER_LAYOUT_SHA256)
        self.assertEqual(
            receipt["actor_context_excluded_observation_features"],
            [ASYMMETRIC_WORKER_SKIP_DRY_FEATURE])

    def test_structured_deployment_completion_consumes_canonical_runtime_evidence(
            self):
        source_path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        payload = source_path.read_bytes()
        source_sha256 = hashlib.sha256(payload).hexdigest()
        model = _asymmetric_model(seed=77)
        actor_receipt = train_ppo._initialize_asymmetric_worker_actor(
            model,
            source_checkpoint_payload=payload,
            source_checkpoint_sha256=source_sha256,
        )
        critic_receipt = train_ppo._reset_worker_critic(
            model,
            training_seed=77,
            source_checkpoint_sha256=source_sha256,
            source_actor_sha256=actor_receipt["source_actor_sha256"],
            source_critic_sha256=actor_receipt["source_critic_sha256"],
        )
        model._critic_migration_receipt = dict(critic_receipt)
        adapter = model.policy.mlp_extractor.context_adapter
        groups = adapter.named_parameter_groups()
        partition = strict_actor_critic_parameter_partition(
            model.policy, optimizer=model.policy.optimizer)
        with torch.no_grad():
            groups["encoder"][0].view(-1)[0].add_(0.01)
            groups["interaction"][0].view(-1)[0].add_(0.01)
            for parameter in groups["output"]:
                parameter.fill_(1e-4)
            partition["critic"][0].view(-1)[0].add_(0.01)
        model.policy.mlp_extractor.enable_actor_context()
        model.distill_beta = 0.0
        model.distill_anneal_actor_rollouts = 0
        self.assertTrue(
            train_ppo._asymmetric_worker_deployment_evidence_complete(
                model))
        with torch.no_grad():
            for parameter in groups["output"]:
                parameter.zero_()
        self.assertFalse(
            train_ppo._asymmetric_worker_deployment_evidence_complete(
                model))

    def test_canonical_reconstruction_matches_seeded_training_constructor(self):
        source_path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        payload = source_path.read_bytes()
        source_sha256 = hashlib.sha256(payload).hexdigest()
        seed = 991
        model = _asymmetric_model(seed=seed)
        actor_receipt = train_ppo._initialize_asymmetric_worker_actor(
            model,
            source_checkpoint_payload=payload,
            source_checkpoint_sha256=source_sha256,
        )
        critic_receipt = train_ppo._reset_worker_critic(
            model,
            training_seed=seed,
            source_checkpoint_sha256=source_sha256,
            source_actor_sha256=actor_receipt["source_actor_sha256"],
            source_critic_sha256=actor_receipt["source_critic_sha256"],
        )
        canonical = (
            train_ppo._canonical_asymmetric_worker_migration_evidence(
                source_checkpoint_payload=payload,
                source_checkpoint_sha256=source_sha256,
                training_seed=seed,
            )
        )
        self.assertEqual(
            canonical["actor_migration"], actor_receipt)
        self.assertEqual(
            canonical["critic_reset"], critic_receipt)

    def test_asymmetric_migration_binds_payload_hash_and_real_source_topology(self):
        source_path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        payload = source_path.read_bytes()
        target = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        with self.assertRaisesRegex(ValueError, "payload/SHA"):
            train_ppo._initialize_asymmetric_worker_actor(
                target,
                source_checkpoint_payload=payload,
                source_checkpoint_sha256="0" * 64,
            )

        relu_source = LeashedMaskablePPO(
            "MlpPolicy",
            _LegacyMaskEnv(),
            n_steps=2,
            batch_size=2,
            policy_kwargs={"activation_fn": torch.nn.ReLU},
            device="cpu",
            verbose=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "relu-source.zip"
            relu_source.save(path)
            relu_payload = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "registered|注册"):
            train_ppo._initialize_asymmetric_worker_actor(
                target,
                source_checkpoint_payload=relu_payload,
                source_checkpoint_sha256=hashlib.sha256(
                    relu_payload).hexdigest(),
            )

    def test_asymmetric_policy_rejects_non_tanh_topology(self):
        with self.assertRaisesRegex(ValueError, "Tanh"):
            LeashedMaskablePPO(
                AsymmetricWorkerMaskableActorCriticPolicy,
                _DualMaskEnv(),
                n_steps=2,
                batch_size=2,
                policy_kwargs={"activation_fn": torch.nn.ReLU},
                device="cpu",
                verbose=0,
            )

    def test_structured_capacity_groups_and_softwall_decode_are_closed(self):
        model = _asymmetric_model()
        adapter = model.policy.mlp_extractor.context_adapter
        groups = adapter.named_parameter_groups()
        self.assertEqual(
            tuple(groups), ("encoder", "interaction", "output"))
        grouped = tuple(
            parameter
            for parameters in groups.values()
            for parameter in parameters
        )
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in adapter.parameters()})
        self.assertEqual(
            len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertTrue(all(groups.values()))
        self.assertEqual(
            sum(parameter.numel() for parameter in grouped),
            ASYMMETRIC_WORKER_FROZEN_CONTEXT_PARAMETER_COUNT,
        )
        self.assertEqual(
            len(grouped),
            ASYMMETRIC_WORKER_FROZEN_CONTEXT_TENSOR_COUNT,
        )

        partition = strict_actor_critic_parameter_partition(
            model.policy, optimizer=model.policy.optimizer)
        self.assertEqual(
            sum(parameter.numel() for parameter in partition["actor"]),
            ASYMMETRIC_WORKER_FROZEN_ACTOR_PARAMETER_COUNT,
        )
        self.assertEqual(
            len(partition["actor"]),
            ASYMMETRIC_WORKER_FROZEN_ACTOR_TENSOR_COUNT,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in partition["critic"]),
            ASYMMETRIC_WORKER_FROZEN_CRITIC_PARAMETER_COUNT,
        )
        self.assertEqual(
            len(partition["critic"]),
            ASYMMETRIC_WORKER_FROZEN_CRITIC_TENSOR_COUNT,
        )
        self.assertFalse(
            {id(parameter) for parameter in partition["actor"]}
            & {id(parameter) for parameter in partition["critic"]})

        wire = torch.zeros(
            8, len(CONTROLLER_SNAPSHOT_MAP_CHANNELS), 25, 25)
        softwall_index = CONTROLLER_SNAPSHOT_MAP_CHANNELS.index(
            "softwall_kind")
        wire[:, softwall_index] = (
            torch.arange(8, dtype=wire.dtype).view(8, 1, 1) / 7.0)
        decoded = _StructuredWorkerContextEncoder.decode_map_wire(wire)
        expected = torch.tensor([
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ], dtype=wire.dtype)
        self.assertTrue(torch.equal(
            decoded[:, softwall_index:softwall_index + 3, 0, 0],
            expected,
        ))

        observation = torch.zeros(
            1, ASYMMETRIC_WORKER_OBSERVATION_DIM)
        monster = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "controller_monsters")
        missile = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "controller_missiles")
        self.assertGreater(len(monster.tail_field_names), 0)
        self.assertGreater(len(missile.tail_field_names), 0)
        observation[
            0, monster.stop - len(monster.tail_field_names)
        ] = 0.375
        observation[
            0, missile.stop - len(missile.tail_field_names)
        ] = 0.625
        with torch.no_grad():
            summary = adapter.encoder(observation)
        monster_tail_start = 32 + 24 + 50
        missile_tail_start = (
            monster_tail_start + len(monster.tail_field_names) + 48)
        self.assertEqual(
            float(summary[0, monster_tail_start]), 0.375)
        self.assertEqual(
            float(summary[0, missile_tail_start]), 0.625)

    def test_runtime_evidence_is_rng_free_behavioral_and_fail_closed(self):
        model = _asymmetric_model()
        rng_before = torch.random.get_rng_state().clone()
        evidence = asymmetric_worker_runtime_evidence(model.policy)
        self.assertTrue(torch.equal(
            torch.random.get_rng_state(), rng_before))
        self.assertEqual(
            evidence["schema"],
            ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA)
        self.assertEqual(
            evidence["layout"]["schema"], DUAL_WORKER_LAYOUT.schema)
        self.assertEqual(
            evidence["layout"]["sha256"], DUAL_WORKER_LAYOUT_SHA256)
        self.assertFalse(evidence["context"]["enabled"])
        self.assertEqual(
            evidence["context"]["parameter_groups"]["output"]
            ["nonzero_count"],
            0,
        )
        self.assertTrue(
            evidence["probes"]["p_skip_preoutput_invariant"])
        self.assertEqual(
            evidence["probes"]["p_skip_preoutput_max_abs_delta"], 0.0)
        self.assertTrue(
            evidence["probes"]["p_skip_action_logits_invariant"])
        self.assertEqual(
            evidence["probes"]["p_skip_action_logits_max_abs_delta"], 0.0)
        self.assertFalse(
            evidence["probes"]["nonzero_context_action_effect"])
        for segment_name in ("current_v4_base", "wrapper_scalars"):
            scalar = evidence["probes"][
                "actor_scalar_preoutput_effects"][segment_name]
            self.assertTrue(scalar["registered_actor_input"])
            self.assertNotEqual(
                scalar["feature_index"],
                ASYMMETRIC_WORKER_SKIP_DRY_FEATURE)
            self.assertTrue(scalar["preoutput_effect"])
            self.assertGreater(
                scalar["preoutput_max_abs_delta"], 0.0)

        with torch.no_grad():
            adapter = model.policy.mlp_extractor.context_adapter
            adapter.output.weight.copy_(torch.eye(
                adapter.output.out_features,
                dtype=adapter.output.weight.dtype,
                device=adapter.output.weight.device,
            ))
            model.policy.mlp_extractor.enable_actor_context()
        live = asymmetric_worker_runtime_evidence(model.policy)
        self.assertTrue(live["context"]["enabled"])
        self.assertTrue(
            live["probes"]["forced_context_action_effect"])
        self.assertTrue(
            live["probes"]["nonzero_context_action_effect"])
        self.assertTrue(
            live["probes"]["p_skip_action_logits_invariant"])
        self.assertEqual(
            live["probes"]["p_skip_action_logits_max_abs_delta"], 0.0)
        self.assertGreater(
            live["probes"]["nonzero_context_action_logit_max_abs_delta"],
            0.0,
        )
        for segment_name in (
            "current_v4_base",
            "wrapper_scalars",
            "controller_combat",
        ):
            focused = live["probes"][
                "actor_focused_effects"][segment_name]
            self.assertTrue(focused["registered_actor_input"])
            self.assertTrue(focused["preoutput_effect"])
            self.assertTrue(focused["context_action_effect"])
            self.assertGreater(
                focused["context_action_logit_max_abs_delta"], 0.0)

        with torch.no_grad():
            adapter.output.weight[0, 0] = torch.nan
        with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
            asymmetric_worker_runtime_evidence(model.policy)

    def test_asymmetric_probe_padding_preserves_only_legacy_root(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        raw = torch.randn(5, 298)
        padded = train_ppo._probe_policy_observation_view(model, raw)
        self.assertEqual(
            tuple(padded.shape),
            (5, ASYMMETRIC_WORKER_OBSERVATION_DIM))
        self.assertTrue(torch.equal(
            padded[:, :298],
            _legacy_worker_observation_view(raw),
        ))
        self.assertEqual(
            int(torch.count_nonzero(padded[:, 298:])), 0)

    def test_asymmetric_context_and_hashes_survive_save_load(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        with torch.no_grad():
            adapter = model.policy.mlp_extractor.context_adapter
            adapter.context_projection.weight.fill_(0.125)
            adapter.context_projection.bias.fill_(0.0625)
            adapter.fusion.weight.fill_(0.125)
            adapter.fusion.bias.fill_(0.0625)
            adapter.output.weight.fill_(0.125)
            model.policy.mlp_extractor.enable_actor_context()
        actor_hash = actor_parameter_sha256(
            model.policy, optimizer=model.policy.optimizer)
        critic_hash = critic_parameter_sha256(
            model.policy, optimizer=model.policy.optimizer)
        observation = torch.randn(
            3, ASYMMETRIC_WORKER_OBSERVATION_DIM)
        with torch.no_grad():
            logits = model.policy.action_net(
                model.policy.mlp_extractor.forward_actor(observation))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "asymmetric.zip"
            model.save(path)
            restored = LeashedMaskablePPO.load(
                path, env=_DualMaskEnv(), device="cpu",
                teacher_path=None, teacher_sha256=None)
        self.assertIsInstance(
            restored.policy, AsymmetricWorkerMaskableActorCriticPolicy)
        self.assertTrue(
            restored.policy.mlp_extractor.actor_context_enabled)
        self.assertEqual(
            actor_parameter_sha256(
                restored.policy, optimizer=restored.policy.optimizer),
            actor_hash)
        self.assertEqual(
            critic_parameter_sha256(
                restored.policy, optimizer=restored.policy.optimizer),
            critic_hash)
        with torch.no_grad():
            restored_logits = restored.policy.action_net(
                restored.policy.mlp_extractor.forward_actor(observation))
        self.assertTrue(torch.equal(restored_logits, logits))

    def test_training_only_skip_probability_is_critic_only(self):
        self.assertEqual(
            ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
            (ASYMMETRIC_WORKER_SKIP_DRY_FEATURE,))
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        extractor = model.policy.mlp_extractor
        relative = ASYMMETRIC_WORKER_SKIP_DRY_FEATURE - 298
        with torch.no_grad():
            # Make the actor residual live.  Its structured scalar encoder
            # physically omits p_skip; no dense "excluded column" exists.
            adapter = extractor.context_adapter
            adapter.context_projection.weight.fill_(0.125)
            adapter.context_projection.bias.zero_()
            adapter.fusion.weight.fill_(0.125)
            adapter.fusion.bias.zero_()
            adapter.output.weight.fill_(0.125)
            extractor.enable_actor_context()

            # Route only p_skip through one explicit critic path so critic
            # sensitivity is constructive rather than a random coincidence.
            critic = extractor.value_net
            for parameter in critic.parameters():
                parameter.zero_()
            wrapper = next(
                segment for segment in DUAL_WORKER_LAYOUT.segments
                if segment.name == "wrapper_scalars")
            current = next(
                segment for segment in DUAL_WORKER_LAYOUT.segments
                if segment.name == "current_v4_base")
            scalar_index = (
                current.width
                + ASYMMETRIC_WORKER_SKIP_DRY_FEATURE
                - wrapper.start
            )
            critic.encoder.scalar_encoder.weight[0, scalar_index] = 1.0
            critic.context_projection.weight[0, 0] = 1.0
            critic.fusion.weight[0, 32] = 1.0
            critic.post.weight[0, 0] = 1.0
        low = torch.zeros(2, ASYMMETRIC_WORKER_OBSERVATION_DIM)
        high = low.clone()
        high[:, ASYMMETRIC_WORKER_SKIP_DRY_FEATURE] = 1.0
        with torch.no_grad():
            low_actor = extractor.forward_actor(low)
            high_actor = extractor.forward_actor(high)
            low_logits = model.policy.action_net(low_actor)
            high_logits = model.policy.action_net(high_actor)
            low_critic = extractor.forward_critic(low)
            high_critic = extractor.forward_critic(high)
        self.assertTrue(torch.equal(low_actor, high_actor))
        self.assertTrue(torch.equal(low_logits, high_logits))
        self.assertFalse(torch.equal(low_critic, high_critic))
        self.assertEqual(
            float(extractor._actor_context_feature_mask[relative]), 0.0)

    def test_asymmetric_actor_transplant_is_bitwise_and_context_learns(self):
        source_path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        source = LeashedMaskablePPO.load(
            source_path, device="cpu",
            teacher_path=None, teacher_sha256=None)
        target = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-3,
            seed=23,
            device="cpu",
            verbose=0,
        )
        keys = train_ppo._POLICY_HEAD_KEYS
        source_state = source.policy.state_dict()
        target_state = target.policy.state_dict()
        with torch.no_grad():
            for key in keys:
                target_state[key].copy_(source_state[key])
        self.assertFalse(
            target.policy.mlp_extractor.actor_context_enabled)
        adapter = target.policy.mlp_extractor.context_adapter
        self.assertGreater(
            int(torch.count_nonzero(
                adapter.context_projection.weight)), 0)
        self.assertGreater(
            int(torch.count_nonzero(adapter.fusion.weight)), 0)
        self.assertEqual(
            int(torch.count_nonzero(adapter.output.weight)), 0)
        for batch in (1, 4):
            legacy = torch.randn(batch, 298)
            context = torch.randn(
                batch, ASYMMETRIC_WORKER_OBSERVATION_DIM - 298)
            dual = torch.cat([legacy, context], dim=-1)
            with torch.no_grad():
                old_logits = source.policy.action_net(
                    source.policy.mlp_extractor.forward_actor(legacy))
                new_logits = target.policy.action_net(
                    target.policy.mlp_extractor.forward_actor(dual))
            self.assertTrue(torch.equal(old_logits, new_logits))

        target.policy.mlp_extractor.enable_actor_context()
        for batch in (1, 4):
            legacy = torch.randn(batch, 298)
            context = torch.randn(
                batch, ASYMMETRIC_WORKER_OBSERVATION_DIM - 298)
            dual = torch.cat([legacy, context], dim=-1)
            with torch.no_grad():
                old_logits = source.policy.action_net(
                    source.policy.mlp_extractor.forward_actor(legacy))
                enabled_logits = target.policy.action_net(
                    target.policy.mlp_extractor.forward_actor(dual))
            self.assertTrue(torch.equal(old_logits, enabled_logits))

        dual = torch.randn(3, ASYMMETRIC_WORKER_OBSERVATION_DIM)
        loss = target.policy.action_net(
            target.policy.mlp_extractor.forward_actor(dual)).sum()
        target.policy.optimizer.zero_grad()
        loss.backward()
        output_gradient = adapter.output.weight.grad
        projection_gradient = adapter.context_projection.weight.grad
        fusion_gradient = adapter.fusion.weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertGreater(float(output_gradient.abs().sum()), 0.0)
        self.assertIsNotNone(projection_gradient)
        self.assertIsNotNone(fusion_gradient)
        self.assertEqual(float(projection_gradient.abs().sum()), 0.0)
        self.assertEqual(float(fusion_gradient.abs().sum()), 0.0)

        target.policy.optimizer.step()
        target.policy.optimizer.zero_grad()
        dual = dual.detach().clone().requires_grad_(True)
        second_loss = target.policy.action_net(
            target.policy.mlp_extractor.forward_actor(dual)).square().sum()
        second_loss.backward()
        self.assertGreater(
            float(adapter.context_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(adapter.fusion.weight.grad.abs().sum()), 0.0)
        self.assertEqual(
            int(torch.count_nonzero(
                dual.grad[:, ASYMMETRIC_WORKER_SKIP_DRY_FEATURE])), 0)

    def test_context_adapter_has_true_legacy_by_context_interaction(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        adapter = model.policy.mlp_extractor.context_adapter
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if name.endswith("bias"):
                    parameter.fill_(0.125)
            adapter.output.weight.fill_(0.25)
            zero_context_residual = adapter(
                torch.zeros(
                    3, ASYMMETRIC_WORKER_OBSERVATION_DIM),
                torch.randn(
                    3, model.policy.mlp_extractor.latent_dim_pi),
            )
        self.assertTrue(torch.equal(
            zero_context_residual,
            torch.zeros_like(zero_context_residual),
        ))
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.zero_()
            adapter.encoder.scalar_encoder.weight[0, 0] = 1.0
            adapter.context_projection.weight[0, 0] = 1.0
            adapter.context_gate.weight[0, 0] = 1.0
            adapter.legacy_gate.weight[0, 0] = 1.0
            interaction_start = adapter.context_projection.out_features
            adapter.fusion.weight[0, interaction_start] = 1.0
            adapter.output.weight[0, 0] = 1.0

        first_scalar_segment = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name
            == adapter.encoder._scalar_segment_names[0])
        context_feature = first_scalar_segment.start
        self.assertNotEqual(
            context_feature, ASYMMETRIC_WORKER_SKIP_DRY_FEATURE)

        def response(legacy_value: float, context_value: float):
            context = torch.zeros(
                1, ASYMMETRIC_WORKER_OBSERVATION_DIM)
            legacy = torch.zeros(
                1, model.policy.mlp_extractor.latent_dim_pi)
            context[0, context_feature] = context_value
            legacy[0, 0] = legacy_value
            with torch.no_grad():
                return float(adapter(context, legacy)[0, 0])

        mixed_difference = (
            response(1.0, 1.0)
            - response(1.0, 0.0)
            - response(0.0, 1.0)
            + response(0.0, 0.0)
        )
        self.assertGreater(abs(mixed_difference), 1e-3)

        context = torch.zeros(
            1, ASYMMETRIC_WORKER_OBSERVATION_DIM,
            requires_grad=True)
        legacy = torch.zeros(
            1, model.policy.mlp_extractor.latent_dim_pi,
            requires_grad=True)
        context.data[0, context_feature] = 0.5
        legacy.data[0, 0] = 0.5
        adapter(context, legacy).sum().backward()
        self.assertGreater(float(context.grad.abs().sum()), 0.0)
        self.assertIsNone(legacy.grad)

    def test_legacy_teacher_cannot_backpropagate_into_context_residual(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            device="cpu",
            verbose=0,
        )
        adapter = model.policy.mlp_extractor.context_adapter
        with torch.no_grad():
            adapter.output.weight.fill_(0.125)
            model.policy.mlp_extractor.enable_actor_context()
        observations = torch.randn(
            4, ASYMMETRIC_WORKER_OBSERVATION_DIM)

        model.policy.optimizer.zero_grad()
        distillation_logits = model._student_distillation_logits(
            observations)
        distillation_logits.square().sum().backward()
        self.assertTrue(all(
            parameter.grad is None
            or int(torch.count_nonzero(parameter.grad)) == 0
            for parameter in adapter.parameters()
        ))
        self.assertGreater(
            float(model.policy.action_net.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(
                model.policy.mlp_extractor.policy_net[
                    0].weight.grad.abs().sum()),
            0.0,
        )
        model.policy.optimizer.zero_grad()
        full_logits = model._student_raw_action_logits(observations)
        full_logits.square().sum().backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in adapter.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_distillation_anneal_counts_only_joint_actor_rollouts(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _DualMaskEnv(),
            n_steps=2,
            batch_size=2,
            distill_beta=0.015625,
            distill_anneal_actor_rollouts=3,
            device="cpu",
            verbose=0,
        )
        self.assertEqual(
            model._effective_distill_beta(actor_frozen=True),
            0.015625,
        )
        self.assertEqual(
            model._effective_distill_beta(actor_frozen=False),
            0.015625,
        )
        self.assertEqual(model._distill_actor_rollouts_completed, 0)
        model._complete_main_ppo_rollout(
            actor_frozen=False, optimizer_steps=1)
        self.assertEqual(
            model._effective_distill_beta(actor_frozen=False),
            0.0078125,
        )
        model._complete_main_ppo_rollout(
            actor_frozen=False, optimizer_steps=1)
        self.assertEqual(
            model._effective_distill_beta(actor_frozen=False), 0.0)
        model._complete_main_ppo_rollout(
            actor_frozen=False, optimizer_steps=1)
        self.assertEqual(
            model._effective_distill_beta(actor_frozen=False), 0.0)

        with self.assertRaisesRegex(ValueError, "0 或 >=2"):
            LeashedMaskablePPO(
                AsymmetricWorkerMaskableActorCriticPolicy,
                _DualMaskEnv(),
                n_steps=2,
                batch_size=2,
                distill_anneal_actor_rollouts=1,
                device="cpu",
                verbose=0,
            )

    def test_legacy_exclusions_are_noop_when_actions_do_not_exist(self):
        masks = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        filtered = _legacy_distillation_masks(masks)
        self.assertTrue(torch.equal(filtered, masks))

        worker_masks = torch.ones((1, 15), dtype=torch.bool)
        worker_filtered = _legacy_distillation_masks(worker_masks)
        self.assertFalse(bool(worker_filtered[0, 12]))
        self.assertFalse(bool(worker_filtered[0, 14]))
        self.assertTrue(bool(worker_masks[0, 12]))
        self.assertTrue(bool(worker_masks[0, 14]))

    def test_legacy_exclusions_fail_loud_when_they_empty_support(self):
        masks = torch.zeros((1, 15), dtype=torch.bool)
        masks[0, 12] = True
        masks[0, 14] = True
        with self.assertRaisesRegex(ValueError, "全 False"):
            _legacy_distillation_masks(masks)

    def test_action14_prior_is_mask_bound_on_policy_and_trainable(self):
        model = _asymmetric_model(
            seed=419, action14_logit_bonus=2.5)
        obs = torch.randn(
            4, ASYMMETRIC_WORKER_OBSERVATION_DIM)
        masks = torch.ones((4, 15), dtype=torch.bool)
        masks[1, 14] = False
        masks_before = masks.clone()

        with torch.no_grad():
            model.policy.action14_logit_bonus = 0.0
            base = model.policy.get_distribution(
                obs, action_masks=masks).distribution.probs.clone()
            model.policy.action14_logit_bonus = 2.5
            boosted = model.policy.get_distribution(
                obs, action_masks=masks).distribution.probs.clone()

        self.assertTrue(torch.equal(masks, masks_before))
        self.assertEqual(float(boosted[1, 14]), 0.0)
        self.assertTrue(torch.equal(boosted[1], base[1]))
        legal = masks[:, 14]
        base_odds = base[legal, 14] / base[legal, 0]
        boosted_odds = boosted[legal, 14] / boosted[legal, 0]
        self.assertTrue(torch.allclose(
            boosted_odds / base_odds,
            torch.full_like(base_odds, float(np.exp(2.5))),
            rtol=1e-5,
            atol=1e-6,
        ))

        model.policy.optimizer.zero_grad()
        distribution = model.policy.get_distribution(
            obs[[0]], action_masks=masks[[0]])
        loss = -distribution.log_prob(torch.tensor([14])).mean()
        loss.backward()
        self.assertGreater(
            float(model.policy.action_net.weight.grad[14].abs().sum()),
            0.0,
        )
        self.assertNotEqual(
            float(model.policy.action_net.bias.grad[14]), 0.0)
        model.get_env().close()

    def test_real_v28_topology_is_exactly_six_plus_six(self):
        path = (
            ROOT / "train" / "models" / "v28-worker-leg1"
            / "model_final.zip"
        )
        self.assertTrue(path.is_file())
        model = LeashedMaskablePPO.load(
            path, device="cpu", teacher_path=None, teacher_sha256=None)
        partition = strict_actor_critic_parameter_partition(
            model.policy, optimizer=model.policy.optimizer)
        self.assertEqual(len(partition["actor"]), 6)
        self.assertEqual(len(partition["critic"]), 6)
        self.assertEqual(
            sum(parameter.numel() for parameter in partition["actor"]),
            24_271,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in partition["critic"]),
            23_361,
        )
        self.assertFalse(
            {id(parameter) for parameter in partition["actor"]}
            & {id(parameter) for parameter in partition["critic"]}
        )

    def test_value_only_reset_is_deterministic_and_preserves_actor_rng(self):
        source_sha = "a" * 64
        models = [_asymmetric_model(), _asymmetric_model()]
        critic_hashes = []
        for model in models:
            actor_before = actor_parameter_sha256(
                model.policy, optimizer=model.policy.optimizer)
            rng_before = torch.random.get_rng_state().clone()
            receipt = train_ppo._reset_worker_critic(
                model,
                training_seed=2_130_000,
                source_checkpoint_sha256=source_sha,
            )
            self.assertEqual(
                actor_parameter_sha256(
                    model.policy, optimizer=model.policy.optimizer),
                actor_before,
            )
            self.assertTrue(torch.equal(
                torch.random.get_rng_state(), rng_before))
            self.assertEqual(
                receipt["actor_sha256_before"],
                receipt["actor_sha256_after"],
            )
            self.assertNotEqual(
                receipt["critic_sha256_before"],
                receipt["critic_sha256_after"],
            )
            critic_hashes.append(critic_parameter_sha256(
                model.policy, optimizer=model.policy.optimizer))
        self.assertEqual(critic_hashes[0], critic_hashes[1])

        other = _asymmetric_model()
        train_ppo._reset_worker_critic(
            other,
            training_seed=2_130_001,
            source_checkpoint_sha256=source_sha,
        )
        self.assertNotEqual(
            critic_parameter_sha256(
                other.policy, optimizer=other.policy.optimizer),
            critic_hashes[0],
        )

    def test_critic_reset_followed_by_optimizer_reset_clears_adam(self):
        model = _asymmetric_model()
        model.policy.optimizer.zero_grad()
        _branch_loss(model).backward()
        model.policy.optimizer.step()
        self.assertTrue(model.policy.optimizer.state)
        train_ppo._reset_worker_critic(
            model,
            training_seed=7,
            source_checkpoint_sha256="b" * 64,
        )
        train_ppo._reset_policy_optimizer(model, 1e-3)
        self.assertFalse(model.policy.optimizer.state)

    def test_trainable_shared_extractor_is_rejected(self):
        policy = _TrainableSharedPolicy()
        with self.assertRaisesRegex(
                RuntimeError, "trainable shared features_extractor"):
            strict_actor_critic_parameter_partition(
                policy, optimizer=policy.optimizer)

    def test_separate_clip_does_not_let_critic_scale_actor(self):
        model = _model()
        partition = strict_actor_critic_parameter_partition(
            model.policy, optimizer=model.policy.optimizer)
        model.policy.optimizer.zero_grad()
        partition["actor"][0].grad = torch.zeros_like(
            partition["actor"][0])
        partition["actor"][0].grad.reshape(-1)[0] = 0.25
        partition["critic"][0].grad = torch.zeros_like(
            partition["critic"][0])
        partition["critic"][0].grad.reshape(-1)[0] = 250.0

        record = clip_actor_critic_gradients(
            model.policy,
            model.policy.optimizer,
            0.5,
            actor_frozen=False,
        )
        self.assertAlmostEqual(record["actor_preclip_norm"], 0.25, places=6)
        self.assertAlmostEqual(
            record["critic_preclip_norm"], 250.0, places=3)
        self.assertFalse(record["actor_clipped"])
        self.assertTrue(record["critic_clipped"])
        self.assertAlmostEqual(
            float(partition["actor"][0].grad.reshape(-1)[0]), 0.25,
            places=6,
        )
        self.assertAlmostEqual(
            float(partition["critic"][0].grad.norm()), 0.5,
            places=5,
        )

    def test_warmup_clears_actor_grad_and_persists_across_save_load(self):
        model = _model()
        receipt = model.configure_critic_migration(
            gradient_clip_mode=GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        self.assertEqual(receipt["warmup_expected_rollouts"], 2)
        self.assertEqual(
            receipt["worker_onpolicy_pg_audit_schema"],
            WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
        )
        anchor = actor_parameter_sha256(
            model.policy, optimizer=model.policy.optimizer)
        critic_before = {
            name: parameter.detach().clone()
            for name, parameter in model.policy.named_parameters()
            if "value_net" in name
        }

        model.num_timesteps = 2
        frozen = model._prepare_main_ppo_rollout()
        self.assertTrue(frozen)
        record = model._apply_main_ppo_optimizer_step(
            _branch_loss(model), actor_frozen=frozen)
        model._complete_main_ppo_rollout(
            actor_frozen=frozen, optimizer_steps=1)
        self.assertGreater(record["actor_counterfactual_norm"], 0.0)
        self.assertEqual(record["actor_preclip_norm"], 0.0)
        self.assertEqual(
            actor_parameter_sha256(
                model.policy, optimizer=model.policy.optimizer),
            anchor,
        )
        self.assertTrue(any(
            not torch.equal(
                critic_before[name], parameter.detach())
            for name, parameter in model.policy.named_parameters()
            if name in critic_before
        ))
        self.assertEqual(model._critic_warmup_rollouts_completed, 1)
        self.assertEqual(model._actor_optimizer_steps_completed, 0)

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "mid-warmup.zip"
            model.save(path)
            restored = LeashedMaskablePPO.load(
                path, env=_TinyMaskEnv(), device="cpu",
                teacher_path=None, teacher_sha256=None)
        self.assertEqual(
            restored._critic_warmup_until_timesteps, 4)
        self.assertEqual(
            restored._critic_warmup_rollouts_completed, 1)
        self.assertEqual(
            restored._critic_warmup_actor_sha256, anchor)

        restored.num_timesteps = 4
        frozen = restored._prepare_main_ppo_rollout()
        self.assertTrue(frozen)
        restored._apply_main_ppo_optimizer_step(
            _branch_loss(restored), actor_frozen=frozen)
        restored._complete_main_ppo_rollout(
            actor_frozen=frozen, optimizer_steps=1)
        self.assertTrue(restored._critic_warmup_completed)
        self.assertEqual(
            actor_parameter_sha256(
                restored.policy, optimizer=restored.policy.optimizer),
            anchor,
        )

        restored.num_timesteps = 6
        frozen = restored._prepare_main_ppo_rollout()
        self.assertFalse(frozen)
        restored._apply_main_ppo_optimizer_step(
            _branch_loss(restored), actor_frozen=frozen)
        restored._complete_main_ppo_rollout(
            actor_frozen=frozen, optimizer_steps=1)
        self.assertNotEqual(
            actor_parameter_sha256(
                restored.policy, optimizer=restored.policy.optimizer),
            anchor,
        )
        self.assertEqual(restored._actor_optimizer_steps_completed, 1)

        # A later joint rollout must not compare the now-legitimately-updated
        # actor against the frozen warmup anchor.
        restored.num_timesteps = 8
        frozen = restored._prepare_main_ppo_rollout()
        self.assertFalse(frozen)
        restored._apply_main_ppo_optimizer_step(
            _branch_loss(restored), actor_frozen=frozen)
        restored._complete_main_ppo_rollout(
            actor_frozen=frozen, optimizer_steps=1)
        self.assertEqual(restored._actor_optimizer_steps_completed, 2)

    def test_joint_rollout_records_reward_bound_pure_policy_gradient(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(diverse_rewards=True),
            n_steps=4,
            batch_size=4,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            ent_coef=0.25,
            seed=1701,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        # The canonical output is exactly zero at the first joint rollout:
        # that rollout can prove reward reaches the output, while the second
        # proves reward/PPO reaches the context projection itself.
        model.learn(total_timesteps=16)

        self.assertEqual(model._worker_onpolicy_pg_joint_rollouts, 3)
        self.assertEqual(
            model._worker_onpolicy_pg_qualifying_rollouts, 1)
        self.assertTrue(worker_onpolicy_pg_audit_complete(model))
        first, second, receipt = (
            model._worker_onpolicy_pg_rollout_receipts)
        self.assertFalse(first["qualifies"])
        self.assertFalse(second["qualifies"])
        self.assertEqual(
            first["reward_centered_context_encoder_grad_norm"],
            0.0,
        )
        self.assertEqual(
            first["reward_centered_context_interaction_grad_norm"],
            0.0,
        )
        self.assertTrue(validate_worker_onpolicy_pg_receipt(
            receipt, expected_samples=4))
        self.assertEqual(
            receipt["optimizer_steps"],
            WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT)
        runtime = asymmetric_worker_runtime_evidence(model.policy)
        self.assertTrue(runtime["context"]["enabled"])
        self.assertGreater(
            runtime["context"]["parameter_groups"]["output"]
            ["nonzero_count"],
            0,
        )
        for segment_name in (
            "current_v4_base",
            "wrapper_scalars",
            "controller_combat",
        ):
            focused = runtime["probes"][
                "actor_focused_effects"][segment_name]
            self.assertTrue(focused["preoutput_effect"])
            self.assertTrue(focused["context_action_effect"])
            self.assertGreater(
                focused["context_action_logit_max_abs_delta"], 0.0)
        self.assertEqual(
            receipt["transition_reward_source"],
            "WorkerWindowEnv.info.transition_reward",
        )
        self.assertEqual(
            sum(receipt["requested_action_counts"]), 4)
        self.assertEqual(
            sum(receipt["executed_action_counts"]), 4)
        self.assertGreaterEqual(receipt["combat_effect_samples"], 1)
        self.assertGreaterEqual(
            receipt["combat_transition_reward_nonzero_samples"], 1)
        self.assertGreaterEqual(
            receipt["combat_positive_advantage_samples"], 1)
        self.assertGreater(
            receipt["combat_reward_centered_actor_grad_norm"], 0.0)
        self.assertGreater(
            receipt[
                "combat_reward_centered_context_encoder_grad_norm"
            ],
            0.0,
        )
        self.assertGreater(
            receipt[
                "combat_reward_centered_context_interaction_grad_norm"
            ],
            0.0,
        )
        self.assertEqual(
            receipt["pure_ppo_actor_grad_measurements"],
            receipt["optimizer_steps"],
        )
        self.assertEqual(
            receipt["pure_ppo_root_grad_measurements"],
            receipt["optimizer_steps"],
        )
        self.assertEqual(
            receipt["pure_ppo_context_grad_measurements"],
            receipt["optimizer_steps"],
        )
        self.assertGreater(
            receipt["reward_centered_actor_grad_norm"], 0.0)
        self.assertGreater(
            receipt["reward_centered_context_grad_norm"], 0.0)
        self.assertGreater(
            receipt["pure_ppo_actor_grad_norm_max"], 0.0)
        self.assertGreater(
            receipt["pure_ppo_root_grad_norm_max"], 0.0)
        self.assertGreater(
            receipt["pure_ppo_context_grad_norm_max"], 0.0)
        self.assertGreater(
            receipt["reward_centered_context_encoder_grad_norm"],
            0.0,
        )
        self.assertGreater(
            receipt["pure_ppo_context_encoder_grad_norm_max"],
            0.0,
        )
        self.assertGreater(
            receipt["pure_ppo_context_interaction_grad_norm_max"],
            0.0,
        )
        self.assertEqual(
            receipt["distill_context_grad_norm_max"], 0.0)
        self.assertGreater(
            receipt["combined_context_on_pure_ppo_projection"], 0.0)
        self.assertGreater(
            receipt["combined_root_on_pure_ppo_projection"], 0.0)
        self.assertGreater(
            receipt[
                "pure_ppo_actor_on_combat_reward_projection"
            ],
            0.0,
        )
        self.assertGreater(
            receipt[
                "pure_ppo_root_on_combat_reward_projection"
            ],
            0.0,
        )
        self.assertGreater(
            receipt[
                "pure_ppo_context_on_combat_reward_projection"
            ],
            0.0,
        )
        self.assertGreater(
            receipt[
                "optimizer_delta_actor_on_combat_reward_descent_projection"
            ],
            0.0,
        )
        self.assertGreater(
            receipt[
                "optimizer_delta_root_on_combat_reward_descent_projection"
            ],
            0.0,
        )
        self.assertGreater(
            receipt[
                "optimizer_delta_context_on_combat_reward_descent_projection"
            ],
            0.0,
        )
        for partition in ("actor", "root", "context"):
            with self.subTest(optimizer_delta_partition=partition):
                self.assertGreater(
                    receipt[f"optimizer_delta_{partition}_l2"], 0.0)
                self.assertGreater(
                    receipt[
                        f"optimizer_delta_{partition}"
                        "_on_combat_reward_descent_cosine"
                    ],
                    0.0,
                )
                self.assertLessEqual(
                    abs(receipt[
                        f"optimizer_delta_{partition}"
                        "_dot_combat_reward_descent"
                    ]),
                    receipt[f"optimizer_delta_{partition}_l2"]
                    * receipt[
                        f"combat_reward_centered_{partition}_grad_norm"
                    ]
                    * (1.0 + 1e-10),
                )
        zero_samples = dict(receipt)
        zero_samples["transition_reward_samples"] = 0
        self.assertFalse(validate_worker_onpolicy_pg_receipt(
            zero_samples, expected_samples=4))
        cancelled_root = dict(receipt)
        cancelled_root["combined_root_dot_pure_ppo_sum"] = 0.0
        cancelled_root["combined_root_on_pure_ppo_projection"] = 0.0
        cancelled_root["qualifies"] = False
        self.assertFalse(
            _worker_onpolicy_pg_receipt_qualifies(cancelled_root))
        self.assertTrue(validate_worker_onpolicy_pg_receipt(
            cancelled_root, expected_samples=4))
        cancelled_root["qualifies"] = True
        self.assertFalse(validate_worker_onpolicy_pg_receipt(
            cancelled_root, expected_samples=4))
        cancelled_actual_delta = dict(receipt)
        cancelled_actual_delta[
            "optimizer_delta_root_dot_combat_reward_descent"
        ] = 0.0
        cancelled_actual_delta[
            "optimizer_delta_root_on_combat_reward_descent_projection"
        ] = 0.0
        cancelled_actual_delta[
            "optimizer_delta_root_on_combat_reward_descent_cosine"
        ] = 0.0
        cancelled_actual_delta[
            "optimizer_delta_actor_dot_combat_reward_descent"
        ] = cancelled_actual_delta[
            "optimizer_delta_context_dot_combat_reward_descent"
        ]
        actor_denominator = (
            cancelled_actual_delta[
                "combat_reward_centered_actor_grad_norm"
            ] ** 2
        )
        cancelled_actual_delta[
            "optimizer_delta_actor_on_combat_reward_descent_projection"
        ] = (
            cancelled_actual_delta[
                "optimizer_delta_actor_dot_combat_reward_descent"
            ] / actor_denominator
        )
        cancelled_actual_delta[
            "optimizer_delta_actor_on_combat_reward_descent_cosine"
        ] = (
            cancelled_actual_delta[
                "optimizer_delta_actor_dot_combat_reward_descent"
            ]
            / (
                cancelled_actual_delta["optimizer_delta_actor_l2"]
                * cancelled_actual_delta[
                    "combat_reward_centered_actor_grad_norm"
                ]
            )
        )
        cancelled_actual_delta["qualifies"] = False
        self.assertTrue(validate_worker_onpolicy_pg_receipt(
            cancelled_actual_delta, expected_samples=4))
        self.assertFalse(_worker_onpolicy_pg_receipt_qualifies(
            cancelled_actual_delta))
        cancelled_actual_delta["qualifies"] = True
        self.assertFalse(validate_worker_onpolicy_pg_receipt(
            cancelled_actual_delta, expected_samples=4))
        noncombat_ppo_direction = dict(receipt)
        noncombat_ppo_direction[
            "pure_ppo_root_dot_combat_reward_sum"
        ] = 0.0
        noncombat_ppo_direction[
            "pure_ppo_root_on_combat_reward_projection"
        ] = 0.0
        noncombat_ppo_direction[
            "pure_ppo_actor_dot_combat_reward_sum"
        ] = noncombat_ppo_direction[
            "pure_ppo_context_dot_combat_reward_sum"
        ]
        repeated_actor_denominator = (
            noncombat_ppo_direction["optimizer_steps"]
            * noncombat_ppo_direction[
                "combat_reward_centered_actor_grad_norm"
            ] ** 2
        )
        noncombat_ppo_direction[
            "pure_ppo_actor_on_combat_reward_projection"
        ] = (
            noncombat_ppo_direction[
                "pure_ppo_actor_dot_combat_reward_sum"
            ] / repeated_actor_denominator
        )
        noncombat_ppo_direction["qualifies"] = False
        self.assertTrue(validate_worker_onpolicy_pg_receipt(
            noncombat_ppo_direction, expected_samples=4))
        self.assertFalse(_worker_onpolicy_pg_receipt_qualifies(
            noncombat_ppo_direction))
        noncombat_ppo_direction["qualifies"] = True
        self.assertFalse(validate_worker_onpolicy_pg_receipt(
            noncombat_ppo_direction, expected_samples=4))
        cauchy_violation = dict(receipt)
        cauchy_violation[
            "optimizer_delta_context_dot_combat_reward_descent"
        ] = (
            cauchy_violation["optimizer_delta_context_l2"]
            * cauchy_violation[
                "combat_reward_centered_context_grad_norm"
            ]
            * 2.0
        )
        cauchy_violation[
            "optimizer_delta_context_on_combat_reward_descent_projection"
        ] = (
            cauchy_violation[
                "optimizer_delta_context_dot_combat_reward_descent"
            ] / (
                cauchy_violation[
                    "combat_reward_centered_context_grad_norm"
                ] ** 2
            )
        )
        cauchy_violation[
            "optimizer_delta_context_on_combat_reward_descent_cosine"
        ] = 2.0
        cauchy_violation[
            "optimizer_delta_actor_dot_combat_reward_descent"
        ] = (
            cauchy_violation[
                "optimizer_delta_root_dot_combat_reward_descent"
            ] + cauchy_violation[
                "optimizer_delta_context_dot_combat_reward_descent"
            ]
        )
        cauchy_violation[
            "optimizer_delta_actor_on_combat_reward_descent_projection"
        ] = (
            cauchy_violation[
                "optimizer_delta_actor_dot_combat_reward_descent"
            ] / (
                cauchy_violation[
                    "combat_reward_centered_actor_grad_norm"
                ] ** 2
            )
        )
        cauchy_violation[
            "optimizer_delta_actor_on_combat_reward_descent_cosine"
        ] = (
            cauchy_violation[
                "optimizer_delta_actor_dot_combat_reward_descent"
            ] / (
                cauchy_violation["optimizer_delta_actor_l2"]
                * cauchy_violation[
                    "combat_reward_centered_actor_grad_norm"
                ]
            )
        )
        cauchy_violation["qualifies"] = (
            _worker_onpolicy_pg_receipt_qualifies(cauchy_violation))
        self.assertFalse(validate_worker_onpolicy_pg_receipt(
            cauchy_violation, expected_samples=4))
        floating_noise_only = dict(receipt)
        for partition in ("root", "context"):
            gradient_norm = floating_noise_only[
                f"combat_reward_centered_{partition}_grad_norm"
            ]
            dot = (
                1e-15
                * floating_noise_only[
                    f"optimizer_delta_{partition}_l2"
                ]
                * gradient_norm
            )
            floating_noise_only[
                f"optimizer_delta_{partition}"
                "_dot_combat_reward_descent"
            ] = dot
            floating_noise_only[
                f"optimizer_delta_{partition}"
                "_on_combat_reward_descent_projection"
            ] = dot / (gradient_norm ** 2)
            floating_noise_only[
                f"optimizer_delta_{partition}"
                "_on_combat_reward_descent_cosine"
            ] = 1e-15
        floating_noise_only[
            "optimizer_delta_actor_dot_combat_reward_descent"
        ] = (
            floating_noise_only[
                "optimizer_delta_root_dot_combat_reward_descent"
            ] + floating_noise_only[
                "optimizer_delta_context_dot_combat_reward_descent"
            ]
        )
        floating_noise_only[
            "optimizer_delta_actor_on_combat_reward_descent_projection"
        ] = (
            floating_noise_only[
                "optimizer_delta_actor_dot_combat_reward_descent"
            ] / (
                floating_noise_only[
                    "combat_reward_centered_actor_grad_norm"
                ] ** 2
            )
        )
        floating_noise_only[
            "optimizer_delta_actor_on_combat_reward_descent_cosine"
        ] = (
            floating_noise_only[
                "optimizer_delta_actor_dot_combat_reward_descent"
            ] / (
                floating_noise_only["optimizer_delta_actor_l2"]
                * floating_noise_only[
                    "combat_reward_centered_actor_grad_norm"
                ]
            )
        )
        floating_noise_only["qualifies"] = False
        self.assertTrue(validate_worker_onpolicy_pg_receipt(
            floating_noise_only, expected_samples=4))
        self.assertFalse(_worker_onpolicy_pg_receipt_qualifies(
            floating_noise_only))
        for field, forged in (
            ("transition_reward_mean", 999.0),
            ("transition_reward_sum", -777.0),
            ("reward_centered_l2", 123.0),
            ("transition_reward_abs_sum", 0.0),
            ("pure_ppo_root_grad_norm_mean",
             receipt["pure_ppo_root_grad_norm_max"] + 1.0),
        ):
            with self.subTest(nonclosed_summary_field=field):
                tampered = dict(receipt)
                tampered[field] = forged
                self.assertFalse(validate_worker_onpolicy_pg_receipt(
                    tampered, expected_samples=4))
        for field in (
            "reward_centered_context_encoder_grad_norm",
            "reward_centered_context_interaction_grad_norm",
        ):
            with self.subTest(nonqualifying_reward_context_field=field):
                nonqualifying = dict(receipt)
                nonqualifying[field] = 0.0
                nonqualifying["qualifies"] = False
                self.assertFalse(
                    _worker_onpolicy_pg_receipt_qualifies(nonqualifying))
                self.assertTrue(validate_worker_onpolicy_pg_receipt(
                    nonqualifying, expected_samples=4))
                nonqualifying["qualifies"] = True
                self.assertFalse(validate_worker_onpolicy_pg_receipt(
                    nonqualifying, expected_samples=4))

    def test_joint_rollout_fails_fast_below_minimum_optimizer_steps(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(diverse_rewards=True),
            n_steps=4,
            batch_size=4,
            n_epochs=(
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT
                - 1
            ),
            learning_rate=1e-3,
            seed=1703,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        with self.assertRaisesRegex(
                RuntimeError, "joint rollout 至少要求 8"):
            model.learn(total_timesteps=8)
        self.assertEqual(model._critic_warmup_rollouts_completed, 1)
        self.assertTrue(model._critic_warmup_completed)
        self.assertEqual(model._worker_onpolicy_pg_joint_rollouts, 0)
        self.assertEqual(model._worker_onpolicy_pg_rollout_receipts, [])

    def test_formal_pg_info_batch_commits_exactly_one_row_per_env(self):
        env = DummyVecEnv([
            lambda: _FormalPgReceiptEnv(diverse_rewards=True),
            lambda: _FormalPgReceiptEnv(diverse_rewards=True),
        ])
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            env,
            n_steps=2,
            batch_size=4,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        model._setup_learn(total_timesteps=4)
        infos = [
            _formal_pg_info(3, 1.25),
            _formal_pg_info(7, -2.5),
        ]
        model._update_info_buffer(
            infos, dones=np.zeros(2, dtype=bool))
        self.assertEqual(
            model._worker_onpolicy_pg_pending_receipts,
            [
                {
                    "requested_action": 3,
                    "executed_action": 3,
                    "combat_effect": False,
                    "transition_reward": 1.25,
                    "worker_no_progress_timeout": False,
                    "no_progress_timeout_base_failure_reward": 0.0,
                    "no_progress_timeout_additional_failure_reward": 0.0,
                    "no_progress_timeout_failure_reward": 0.0,
                    "expected_buffer_reward": 1.25,
                    "time_limit_bootstrap": False,
                    "time_limit_bootstrap_delta": 0.0,
                },
                {
                    "requested_action": 7,
                    "executed_action": 7,
                    "combat_effect": False,
                    "transition_reward": -2.5,
                    "worker_no_progress_timeout": False,
                    "no_progress_timeout_base_failure_reward": 0.0,
                    "no_progress_timeout_additional_failure_reward": 0.0,
                    "no_progress_timeout_failure_reward": 0.0,
                    "expected_buffer_reward": -2.5,
                    "time_limit_bootstrap": False,
                    "time_limit_bootstrap_delta": 0.0,
                },
            ],
        )

        before = list(model._worker_onpolicy_pg_pending_receipts)
        with self.assertRaisesRegex(RuntimeError, "transition_reward"):
            model._update_info_buffer([
                infos[0],
                {
                    **_formal_pg_info(4, 1.0),
                    "transition_reward": None,
                },
            ], dones=np.zeros(2, dtype=bool))
        self.assertEqual(
            model._worker_onpolicy_pg_pending_receipts, before)
        mismatched_effect = _formal_pg_info(
            9, 1.0, combat=True)
        mismatched_effect["executed_action"] = None
        with self.assertRaisesRegex(
                RuntimeError, "action_effect_audit"):
            model._update_info_buffer(
                [infos[0], mismatched_effect],
                dones=np.zeros(2, dtype=bool),
            )
        self.assertEqual(
            model._worker_onpolicy_pg_pending_receipts, before)

    def test_formal_pg_timeout_row_is_terminal_nonbootstrap_and_exactly_split(
            self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(diverse_rewards=True),
            n_steps=2,
            batch_size=2,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=2,
        )
        model._setup_learn(total_timesteps=2)

        timeout_info = {
            **_formal_pg_info(9, -47.0, combat=True),
            "worker_wage": 1.0,
            "worker_no_progress_timeout": True,
            "no_progress_timeout_base_failure_reward": -16.0,
            "no_progress_timeout_additional_failure_reward": -32.0,
            "no_progress_timeout_failure_reward": -48.0,
            "TimeLimit.truncated": False,
            "time_limit_bootstrap_safe": False,
            "unsettled_budget_terminal": False,
            "existing_terminal_death_reward": 0.0,
            "additional_terminal_death_reward": 0.0,
            "total_terminal_death_reward": 0.0,
        }
        model._update_info_buffer(
            [timeout_info], dones=np.ones(1, dtype=bool))
        self.assertEqual(
            model._worker_onpolicy_pg_pending_receipts,
            [{
                "requested_action": 9,
                "executed_action": 9,
                "combat_effect": True,
                "transition_reward": -47.0,
                "worker_no_progress_timeout": True,
                "no_progress_timeout_base_failure_reward": -16.0,
                "no_progress_timeout_additional_failure_reward": -32.0,
                "no_progress_timeout_failure_reward": -48.0,
                "expected_buffer_reward": -47.0,
                "time_limit_bootstrap": False,
                "time_limit_bootstrap_delta": 0.0,
            }],
        )

        committed = list(model._worker_onpolicy_pg_pending_receipts)
        forged_rows = {
            "timelimit-bootstrap": {
                **timeout_info,
                "TimeLimit.truncated": True,
            },
            "partition-total": {
                **timeout_info,
                "no_progress_timeout_failure_reward": -47.0,
            },
            "transition-reward": {
                **timeout_info,
                "transition_reward": -46.0,
            },
            "death-overlap": {
                **timeout_info,
                "existing_terminal_death_reward": -16.0,
            },
            "false-marker": {
                **timeout_info,
                "worker_no_progress_timeout": False,
            },
        }
        for name, forged in forged_rows.items():
            with self.subTest(case=name), self.assertRaisesRegex(
                    RuntimeError, "timeout"):
                model._update_info_buffer(
                    [forged], dones=np.ones(1, dtype=bool))
            self.assertEqual(
                model._worker_onpolicy_pg_pending_receipts, committed)

    def test_formal_pg_receipt_mirrors_time_limit_bootstrap_exactly(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(diverse_rewards=True),
            n_steps=2,
            batch_size=2,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=2,
        )
        model._setup_learn(total_timesteps=2)

        def _constant_terminal_value(policy, observation):
            return torch.full(
                (observation.shape[0], 1),
                2.0,
                dtype=torch.float32,
                device=observation.device,
            )

        model.policy.predict_values = types.MethodType(
            _constant_terminal_value, model.policy)
        raw_reward = np.float32(1.25)
        expected = np.asarray([raw_reward], dtype=np.float32)
        expected[0] += model.gamma * torch.tensor(
            [2.0], dtype=torch.float32)
        model._update_info_buffer(
            [{
                **_formal_pg_info(9, float(raw_reward), combat=True),
                "terminal_observation": np.zeros(
                    ASYMMETRIC_WORKER_OBSERVATION_DIM,
                    dtype=np.float32,
                ),
                "TimeLimit.truncated": True,
            }],
            dones=np.ones(1, dtype=bool),
        )

        self.assertEqual(
            model._worker_onpolicy_pg_pending_receipts,
            [{
                "requested_action": 9,
                "executed_action": 9,
                "combat_effect": True,
                "transition_reward": float(raw_reward),
                "worker_no_progress_timeout": False,
                "no_progress_timeout_base_failure_reward": 0.0,
                "no_progress_timeout_additional_failure_reward": 0.0,
                "no_progress_timeout_failure_reward": 0.0,
                "expected_buffer_reward": float(expected[0]),
                "time_limit_bootstrap": True,
                "time_limit_bootstrap_delta":
                    float(np.float32(expected[0] - raw_reward)),
            }],
        )

    def test_formal_pg_rejects_rollout_end_actor_rewrite(self):
        model = self._formal_pg_audit_model()

        class _RewriteActor(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                actor = strict_actor_critic_parameter_partition(
                    self.model.policy,
                    optimizer=self.model.policy.optimizer,
                )["actor"]
                with torch.no_grad():
                    actor[0].reshape(-1)[0].add_(0.25)

        with self.assertRaisesRegex(
                RuntimeError, "rollout-end callback 改写 actor"):
            model.learn(total_timesteps=4, callback=_RewriteActor())
        self.assertEqual(model._worker_onpolicy_pg_rollout_receipts, [])

    def test_formal_pg_rejects_rollout_end_advantage_rewrite(self):
        model = self._formal_pg_audit_model()

        class _RewriteAdvantage(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                self.model.rollout_buffer.advantages[0, 0] += np.float32(1.0)

        with self.assertRaisesRegex(
                RuntimeError, "buffer\\.advantages 在 GAE 后被改写"):
            model.learn(total_timesteps=4, callback=_RewriteAdvantage())
        self.assertEqual(model._worker_onpolicy_pg_rollout_receipts, [])

    def test_formal_pg_rejects_collection_log_prob_rewrite(self):
        model = self._formal_pg_audit_model()

        class _RewriteCollectionLogProb(BaseCallback):
            def _on_step(self):
                return True

            def _on_rollout_end(self):
                buffer = self.model.rollout_buffer
                offset = np.float32(0.5)
                # Keep the sealed/current array equality intact so this test
                # specifically exercises the independent policy-log-prob
                # closure rather than the generic post-GAE array guard.
                buffer.log_probs[0, 0] += offset
                buffer._formal_gae_snapshot["log_probs"][0, 0] += offset

        with self.assertRaisesRegex(
                RuntimeError, "actor/log-prob 与 collection 回执不闭合"):
            model.learn(
                total_timesteps=4,
                callback=_RewriteCollectionLogProb(),
            )
        self.assertEqual(model._worker_onpolicy_pg_rollout_receipts, [])

    def test_formal_pg_fails_before_training_on_buffer_reward_tamper(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(diverse_rewards=True),
            n_steps=4,
            batch_size=4,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        begin_rollout = model._begin_worker_onpolicy_pg_rollout

        def _tampered_begin(this, *, actor_frozen):
            this.rollout_buffer.rewards[0, 0] += np.float32(0.5)
            return begin_rollout(actor_frozen=actor_frozen)

        model._begin_worker_onpolicy_pg_rollout = types.MethodType(
            _tampered_begin, model)
        with self.assertRaisesRegex(
                RuntimeError, "buffer\\.rewards 在 GAE 后被改写"):
            model.learn(total_timesteps=4)
        self.assertEqual(
            len(model._worker_onpolicy_pg_pending_receipts), 4)
        self.assertEqual(
            model._critic_warmup_rollouts_completed, 0)

    def test_formal_pg_time_limit_rollout_closes_against_real_buffer(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgTimeLimitEnv(diverse_rewards=True),
            n_steps=2,
            batch_size=2,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            seed=1704,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=2,
        )
        model.learn(total_timesteps=2)

        self.assertEqual(
            model._worker_onpolicy_pg_pending_receipts, [])
        self.assertEqual(
            model._critic_warmup_rollouts_completed, 1)
        self.assertTrue(model._critic_warmup_completed)

    def test_entropy_or_critic_update_cannot_forge_worker_pg_receipt(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(diverse_rewards=False),
            n_steps=4,
            batch_size=4,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            ent_coef=0.25,
            seed=1702,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        model.learn(total_timesteps=8)

        self.assertGreater(model._actor_optimizer_steps_completed, 0)
        self.assertEqual(model._worker_onpolicy_pg_joint_rollouts, 1)
        self.assertEqual(
            model._worker_onpolicy_pg_qualifying_rollouts, 0)
        receipt = model._worker_onpolicy_pg_rollout_receipts[0]
        self.assertTrue(validate_worker_onpolicy_pg_receipt(
            receipt, expected_samples=4))
        self.assertEqual(
            receipt["reward_centered_actor_grad_norm"], 0.0)
        self.assertEqual(
            receipt["reward_centered_context_grad_norm"], 0.0)
        self.assertFalse(receipt["qualifies"])
        self.assertFalse(worker_onpolicy_pg_audit_complete(model))

    def test_noncombat_rewards_cannot_forge_combat_pg_receipt(self):
        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            _FormalPgReceiptEnv(
                diverse_rewards=True,
                combat_effects=False,
            ),
            n_steps=4,
            batch_size=4,
            n_epochs=
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            learning_rate=1e-3,
            ent_coef=0.25,
            seed=1701,
            device="cpu",
            verbose=0,
        )
        model.configure_critic_migration(
            gradient_clip_mode=
                GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        model.learn(total_timesteps=12)

        self.assertEqual(model._worker_onpolicy_pg_joint_rollouts, 2)
        self.assertEqual(
            model._worker_onpolicy_pg_qualifying_rollouts, 0)
        self.assertFalse(worker_onpolicy_pg_audit_complete(model))
        self.assertTrue(any(
            receipt["requested_action_counts"][9] > 0
            for receipt in model._worker_onpolicy_pg_rollout_receipts
        ))
        self.assertTrue(any(
            receipt["reward_centered_actor_grad_norm"] > 0.0
            for receipt in model._worker_onpolicy_pg_rollout_receipts
        ))
        for receipt in model._worker_onpolicy_pg_rollout_receipts:
            self.assertTrue(validate_worker_onpolicy_pg_receipt(
                receipt, expected_samples=4))
            self.assertEqual(receipt["combat_effect_samples"], 0)
            self.assertEqual(
                receipt[
                    "combat_transition_reward_nonzero_samples"
                ],
                0,
            )
            self.assertEqual(
                receipt["combat_positive_advantage_samples"], 0)
            self.assertEqual(
                receipt["combat_reward_centered_actor_grad_norm"],
                0.0,
            )
            self.assertFalse(receipt["qualifies"])

    def test_warmup_rejects_endpoint_jump_and_reconfiguration(self):
        model = _model()
        model.configure_critic_migration(
            gradient_clip_mode=GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
            critic_warmup_steps=4,
        )
        with self.assertRaisesRegex(RuntimeError, "已配置"):
            model.configure_critic_migration(
                gradient_clip_mode=
                    GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
                critic_warmup_steps=4,
            )
        model.num_timesteps = 4
        with self.assertRaisesRegex(RuntimeError, "端点跳跃"):
            model._prepare_main_ppo_rollout()

    def test_warmup_rejects_non_rollout_multiple(self):
        model = _model(n_steps=3)
        with self.assertRaisesRegex(ValueError, "完整 rollout 量子"):
            model.configure_critic_migration(
                gradient_clip_mode=
                    GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
                critic_warmup_steps=4,
            )

    def test_configuration_requires_an_empty_optimizer(self):
        model = _model()
        model.policy.optimizer.zero_grad()
        _branch_loss(model).backward()
        model.policy.optimizer.step()
        self.assertTrue(model.policy.optimizer.state)
        with self.assertRaisesRegex(RuntimeError, "清空全部 state"):
            model.configure_critic_migration(
                gradient_clip_mode=
                    GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
                critic_warmup_steps=4,
            )


if __name__ == "__main__":
    unittest.main()
