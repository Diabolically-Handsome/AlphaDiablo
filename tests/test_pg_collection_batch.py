"""Collection-batch PG closure: ordering and fail-closed integration, no game."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
from leashed_ppo import (LeashedMaskablePPO, AsymmetricWorkerMaskableActorCriticPolicy,
    GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
    WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT)
from test_critic_migration import _FormalPgReceiptEnv


class _RecordingPolicy:
    def __init__(self, bad_shape=False):
        self.calls = []
        self.actions = []
        self.bad_shape = bad_shape

    def get_distribution(self, observations, action_masks):
        assert not torch.is_grad_enabled()
        self.calls.append((observations.detach().clone(), action_masks.clone()))
        def log_prob(actions):
            self.actions.append(actions.clone())
            values = observations[:, 0] + 10 * action_masks[:, 0] + actions
            return values[:, None] if self.bad_shape else values
        return SimpleNamespace(log_prob=log_prob)


class _RankedReceiptEnv(_FormalPgReceiptEnv):
    def __init__(self, rank):
        super().__init__(diverse_rewards=True)
        self.rank = rank

    def _observation(self):
        observation = super()._observation()
        observation[0] = (100 * self.rank + self.steps) / 200
        return observation


def audit_model():
    env = DummyVecEnv([lambda: _RankedReceiptEnv(0), lambda: _RankedReceiptEnv(1)])
    model = LeashedMaskablePPO(AsymmetricWorkerMaskableActorCriticPolicy, env,
        n_steps=4, batch_size=8,
        n_epochs=WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        learning_rate=1e-3, seed=1701, device="cpu", verbose=0)
    model.configure_critic_migration(gradient_clip_mode=GRADIENT_CLIP_SEPARATE_ACTOR_CRITIC_V1,
                                    critic_warmup_steps=8)
    return model, env


class CollectionBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        observations = np.zeros((3, 2, 5), dtype=np.float32)
        observations[:, :, 0] = [[0, 1], [100, 101], [200, 201]]
        masks = np.ones((3, 2, 3), dtype=np.float32)
        masks[:, :, 0] = [[0, 1], [1, 0], [0, 1]]
        model = SimpleNamespace(n_steps=3, n_envs=2, device="cpu", policy=_RecordingPolicy(),
            rollout_buffer=SimpleNamespace(observations=observations,
                action_masks=masks, generator_ready=False))
        return model, torch.tensor([0, 1, 2, 0, 1, 2])

    def test_original_time_batches_and_env_order_are_preserved_without_rng_or_input_mutation(self):
        model, actions = self.fixture()
        before_obs = model.rollout_buffer.observations.copy()
        before_masks = model.rollout_buffer.action_masks.copy()
        rng_before = torch.get_rng_state().clone()
        got = LeashedMaskablePPO._worker_pg_collection_log_probs(model, actions)
        np.testing.assert_array_equal(got, [0, 12, 112, 101, 201, 213])
        self.assertEqual(got.dtype, np.float64)
        self.assertEqual(len(model.policy.calls), 3)
        for step, (observations, masks) in enumerate(model.policy.calls):
            np.testing.assert_array_equal(observations.numpy(), before_obs[step])
            np.testing.assert_array_equal(masks.numpy(), before_masks[step].astype(bool))
            torch.testing.assert_close(model.policy.actions[step], actions.reshape(3, 2)[step], rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        np.testing.assert_array_equal(model.rollout_buffer.observations, before_obs)
        np.testing.assert_array_equal(model.rollout_buffer.action_masks, before_masks)
        self.assertFalse(model.rollout_buffer.generator_ready)

    def test_no_automatic_recovery_of_flattened_transposed_or_wrong_geometry(self):
        for mutation in ("ready", "flat_obs", "flat_masks", "transpose", "actions"):
            model, actions = self.fixture()
            if mutation == "ready":
                model.rollout_buffer.generator_ready = True
            elif mutation == "flat_obs":
                model.rollout_buffer.observations = model.rollout_buffer.observations.reshape(6, 5)
            elif mutation == "flat_masks":
                model.rollout_buffer.action_masks = model.rollout_buffer.action_masks.reshape(6, 3)
            elif mutation == "transpose":
                model.rollout_buffer.observations = model.rollout_buffer.observations.swapaxes(0, 1)
            else:
                actions = actions.reshape(3, 2)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(RuntimeError, "time/env batch"):
                LeashedMaskablePPO._worker_pg_collection_log_probs(model, actions)
            self.assertEqual(model.policy.calls, [])

    def test_malformed_distribution_batch_cannot_silently_reshape(self):
        model, actions = self.fixture()
        model.policy = _RecordingPolicy(bad_shape=True)
        with self.assertRaisesRegex(RuntimeError, "batch 形状"):
            LeashedMaskablePPO._worker_pg_collection_log_probs(model, actions)

    def test_real_multi_env_collection_replays_time_steps_before_first_optimizer(self):
        model, env = audit_model()
        calls = []
        original = model.policy.get_distribution
        def tracked(observations, action_masks=None):
            calls.append(observations[:, 0].detach().cpu().numpy().copy())
            self.assertEqual(model._ppo_optimizer_steps_completed, 0)
            return original(observations, action_masks=action_masks)
        try:
            with patch.object(model.policy, "get_distribution", side_effect=tracked):
                model.learn(total_timesteps=8)
            self.assertEqual(len(calls), 4)
            for step, values in enumerate(calls):
                np.testing.assert_array_equal(values,
                    np.asarray([step / 200, (100 + step) / 200], dtype=np.float32))
            self.assertGreater(model._ppo_optimizer_steps_completed, 0)
            self.assertEqual(model._actor_optimizer_steps_completed, 0)
            self.assertTrue(model._critic_warmup_completed)
        finally:
            env.close()

    def test_same_current_and_sealed_log_prob_tamper_still_fails_before_optimizer(self):
        class Tamper(BaseCallback):
            def _on_step(self):
                return True
            def _on_rollout_end(self):
                buffer = self.model.rollout_buffer
                for values in (buffer.log_probs, buffer._formal_gae_snapshot["log_probs"]):
                    values[1, 0] += np.float32(0.5)
        model, env = audit_model()
        try:
            with self.assertRaisesRegex(RuntimeError, "actor/log-prob 与 collection 回执不闭合"):
                model.learn(total_timesteps=8, callback=Tamper())
            self.assertEqual(model._ppo_optimizer_steps_completed, 0)
            self.assertEqual(model._actor_optimizer_steps_completed, 0)
            self.assertEqual(model._worker_onpolicy_pg_rollout_receipts, [])
        finally:
            env.close()

    def test_existing_gae_guard_still_rejects_already_flattened_flow(self):
        class FlattenFlag(BaseCallback):
            def _on_step(self):
                return True
            def _on_rollout_end(self):
                self.model.rollout_buffer.generator_ready = True
        model, env = audit_model()
        try:
            with self.assertRaisesRegex(RuntimeError, "未展开的 audited rollout buffer"):
                model.learn(total_timesteps=8, callback=FlattenFlag())
            self.assertEqual(model._ppo_optimizer_steps_completed, 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
