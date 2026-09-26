"""Self-contained fast regression for the content-case E4/E5/E6/E8 infrastructure surface (counterpart of PREREG-v33-content-case;
no engine is started and no training output is needed; follows the style of tests/test_b1_infra.py and tests/test_content_case.py).

Covers:
- E4 contract 4->5: single source of the revision number, the three leg shapes (L-base both keys disabled / L-cur
    dry_curriculum payload / L-full both payloads), the skip_dry key still holds the literal CLI flag value,
    keys added isomorphically to the contract and the config receipt, the rejection path for non-legacy continuation of old rev4 checkpoints (R8);
- E5 three instruments: (1) offline probe of dry/fresh distill_ce in separate columns (schema / formula surface / training path untouched),
    (2) dry-window behaviour instrument (dry-state entropy and action distribution / dry and fresh window wage and width aggregation / interval reset / fail-closed),
    (3) canary a12-per-episode statistics function + recorder (closed schema, fail-loud);
    with the three instruments inactive, the code path equals HEAD (knobs default 0 / mount guard / RNG and parameters untouched);
- E6 probe set pinned: BC-v1 demos bytes == the frozen constant (real file must pass, fake bytes must fail,
    both loading gates wired, assertion inside the probe construction);
- E8 launcher: bash -n syntax + key logic of the PID bookkeeping fix (fix for the false alarm "checked the wrong PID" in the OPS
    section of the B1 verdict: the caffeinate assertion now looks inside the process tree; behaviour unchanged).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np
import torch as th

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

import leashed_ppo  # noqa: E402
import train_ppo  # noqa: E402
from train_ppo import (  # noqa: E402
    _BC_AUX_MAIN_LAMBDA,
    _BC_V1_DEMOS_SHA256,
    _CONTRACT_REVISION,
    _DRY_CURRICULUM_MAIN_TABLE,
    DistillCeProbe,
    DryWindowMetricsCallback,
    _assert_bc_v1_demos_frozen,
    _contract_bc_aux,
    _contract_dry_curriculum,
    _dry_anchor_partition,
    _load_dry_anchor_demos,
    _is_exact_training_completion,
    _require_exact_training_completion,
    _reset_policy_optimizer,
    _training_contract,
    _validate_resume_contract,
    a12_canary_stats,
    record_a12_canary,
)

TRAIN_PPO = ROOT / "train" / "train_ppo.py"
LAUNCHER = ROOT / "train" / "launch_case.sh"
CANONICAL_DEMOS = ROOT / "train" / "runs" / "bc-worker" / "demos.npz"
MAIN_TABLE_LITERAL = "linear:1.0:0.5:147,hold:0.5:97"


def _run_cli(*extra_args):
    return subprocess.run(
        [sys.executable, str(TRAIN_PPO), *extra_args],
        text=True, capture_output=True, check=False)


def _leg_args(**kw):
    """Scaled-down namespace of the contract-relevant surface of the D3 shared command form of the three legs (worker/mppo shape)."""
    base = dict(worker=True, options=False, flat_clock=False, arch="mlp",
                max_steps=3000, num_envs=4, n_steps=512, gamma=1.0, lr=3e-4,
                ent_coef=0.005, skip_dry=False, dry_curriculum_schedule=None,
                no_drink_sovereignty=False, bc_aux_lambda=0.0,
                bc_aux_demos=None, bc_aux_graft=False,
                bc_aux_liveness_preflight=False,
                distill_beta=0.0, calib_record_only=False,
                reset_optimizer=False, target_kl=None,
                legacy_worker_policy_observation_view=True)
    base.update(kw)
    return types.SimpleNamespace(**base)


class ExactTrainingCompletionTests(unittest.TestCase):
    def test_full_rollout_before_or_after_frozen_target_is_not_publishable(self):
        model = types.SimpleNamespace(
            rollout_buffer=types.SimpleNamespace(full=True),
            _calib_tripped=False,
            num_timesteps=3_500_032,
        )
        self.assertFalse(
            _is_exact_training_completion(model, 3_997_696))
        model.num_timesteps = 3_997_696
        self.assertTrue(
            _is_exact_training_completion(model, 3_997_696))
        model.num_timesteps += 2_048
        self.assertFalse(
            _is_exact_training_completion(model, 3_997_696))
        model.num_timesteps = 3_997_696
        model._calib_tripped = True
        self.assertFalse(
            _is_exact_training_completion(model, 3_997_696))
        with self.assertRaisesRegex(
                RuntimeError, "did not stop exactly at the frozen target of completed updates"):
            _require_exact_training_completion(model, 3_997_696)
        model._calib_tripped = False
        _require_exact_training_completion(model, 3_997_696)


def _contract_for(args, bc_aux_demos_sha256=None):
    model = types.SimpleNamespace(
        action_space=types.SimpleNamespace(n=15), device="cpu",
        observation_space=types.SimpleNamespace(shape=(298,)),
        max_grad_norm=0.5,
        teacher_sha256=(
            "f" * 64 if float(args.distill_beta) > 0.0 else None
        ))
    return _training_contract(args, model, batch_size=256,
                              bc_aux_demos_sha256=bc_aux_demos_sha256)


class Tiny298MaskedEnv(gym.Env):
    """Engine-free micro-environment with 298-dim observations / Discrete(15) (feeds the probe a real policy forward pass)."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(298,),
                                       dtype=np.float32)
    action_space = gym.spaces.Discrete(15)
    metadata = {"render_modes": []}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(298, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(298, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self):
        mask = np.ones(15, dtype=bool)
        mask[11] = mask[12] = False
        return mask


def _real_policy(seed=7):
    from sb3_contrib import MaskablePPO

    env = DummyVecEnv([Tiny298MaskedEnv])
    return MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
                       seed=seed, device="cpu", verbose=0)


def _teacher_298(preferred_action=9):
    bias = th.full((15,), -10.0)
    bias[preferred_action] = 10.0
    sd = {
        "mlp_extractor.policy_net.0.weight": th.zeros(64, 298),
        "mlp_extractor.policy_net.0.bias": th.zeros(64),
        "mlp_extractor.policy_net.2.weight": th.zeros(64, 64),
        "mlp_extractor.policy_net.2.bias": th.zeros(64),
        "action_net.weight": th.zeros(15, 64),
        "action_net.bias": bias,
    }
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "teacher.pt"
        th.save(sd, p)
        return leashed_ppo.build_teacher(str(p))


def _fake_demos(directory, n=8):
    """Fake-bytes demonstration file (legal shape, bytes differ from the frozen constant)."""
    path = pathlib.Path(directory) / "demos.npz"
    x = np.zeros((n, 298), dtype=np.float32)
    x[: n // 2, 297] = 1.0
    y = np.full(n, 9, dtype=np.int64)
    episode_id = np.asarray([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    np.savez_compressed(path, X=x, Y=y, episode_id=episode_id)
    return path


class CurrentContractTests(unittest.TestCase):
    """The current contract still carries the contextual graft, reward credit, scope and observation view in full."""

    def test_revision_constant_is_26(self):
        self.assertEqual(_CONTRACT_REVISION, 26)

    def test_l_base_carries_disabled_keys(self):
        # L-base (--skip-dry): both keys disabled; no two versions within the same case (design decision 7).
        contract = _contract_for(_leg_args(skip_dry=True))
        self.assertEqual(contract["contract_revision"], 26)
        self.assertIs(contract["legacy_policy_observation_view"], True)
        self.assertIs(contract["skip_dry"], True)          # literal CLI flag value
        self.assertEqual(contract["dry_curriculum"], "disabled")
        self.assertEqual(contract["bc_aux"], "disabled")
        self.assertEqual(contract["actor_migration"], "disabled")
        self.assertEqual(contract["distill_beta"], 0.0)
        self.assertIsNone(contract["teacher_sha256"])
        self.assertFalse(contract["calib_record_only"])

    def test_l_cur_carries_schedule_payload(self):
        contract = _contract_for(
            _leg_args(dry_curriculum_schedule=MAIN_TABLE_LITERAL))
        self.assertIs(contract["skip_dry"], False)          # literal value; must not be overwritten by the predicate
        self.assertEqual(contract["dry_curriculum"],
                         {"schedule": MAIN_TABLE_LITERAL})
        self.assertEqual(contract["bc_aux"], "disabled")

    def test_l_full_carries_both_payloads(self):
        contract = _contract_for(
            _leg_args(dry_curriculum_schedule=MAIN_TABLE_LITERAL,
                      bc_aux_graft=True,
                      bc_aux_lambda=0.0,
                      bc_aux_demos="runs/bc-worker-v2/demos.npz",
                      bc_aux_liveness_preflight=True,
                      distill_beta=0.015625,
                      legacy_worker_policy_observation_view=False),
            bc_aux_demos_sha256="a" * 64)
        self.assertEqual(contract["dry_curriculum"],
                         {"schedule": MAIN_TABLE_LITERAL})
        self.assertEqual(
            contract["distillation"]["excluded_actions"], [12, 14])
        self.assertEqual(contract["bc_aux"],
                         {"mode":
                              "expanded-trainable-a12-contextual-mixture",
                          "lambda": 0.0,
                          "demos_sha256": "a" * 64,
                          "objective_revision": 11,
                          "circuit": train_ppo._bc_aux_circuit_spec(),
                          "king_support":
                          "legal-non12-non14-renormalized",
                          "aux_optimizer_calls_per_rollout": 0,
                          "initial_calibration":
                              "exact-five-percent-contextual-legal-support-mixture",
                          "trainable_adapter_parameters": 5,
                          "post_step_projection": {
                              "gate_parameter_abs_max": 8.0,
                              "probability_min": 0.001,
                              "probability_max": 0.95,
                          },
                          "liveness_preflight": True})
        self.assertIs(contract["legacy_policy_observation_view"], False)
        json.dumps(contract)   # serialisable with stdlib JSON (embedded in status.json/zip)

    def test_main_table_literal_matches_frozen_constant(self):
        self.assertEqual(MAIN_TABLE_LITERAL, _DRY_CURRICULUM_MAIN_TABLE)

    def test_bc_aux_payload_requires_sha_when_active(self):
        args = _leg_args(bc_aux_graft=True, bc_aux_lambda=0.0,
                         bc_aux_demos="x.npz")
        with self.assertRaisesRegex(ValueError, "bc-worker-v2 demo set sha256 is missing"):
            _contract_bc_aux(args, None)
        with self.assertRaisesRegex(ValueError, "bc-worker-v2 demo set sha256 is missing"):
            _contract_for(args)   # the contract assembly surface fails too

    def test_helper_truth_table(self):
        self.assertEqual(_contract_dry_curriculum(_leg_args()), "disabled")
        self.assertEqual(
            _contract_dry_curriculum(_leg_args(
                dry_curriculum_schedule=MAIN_TABLE_LITERAL)),
            {"schedule": MAIN_TABLE_LITERAL})
        # neither flag forces the other: either flag alone -> inactive -> disabled (single source of the E3 predicate)
        self.assertEqual(_contract_bc_aux(
            _leg_args(bc_aux_lambda=_BC_AUX_MAIN_LAMBDA), None), "disabled")
        self.assertEqual(_contract_bc_aux(
            _leg_args(bc_aux_demos="x.npz"), None), "disabled")
        self.assertEqual(_contract_bc_aux(
            _leg_args(bc_aux_graft=True), None), "disabled")
        with self.assertRaisesRegex(
                ValueError, "only accepts a structural graft"):
            _contract_bc_aux(
                _leg_args(
                    bc_aux_lambda=_BC_AUX_MAIN_LAMBDA,
                    bc_aux_demos="legacy-gradient.npz"),
                "a" * 64)

    def test_config_receipt_mirrors_contract_keys_in_source(self):
        # Keys added isomorphically to the contract and the config receipt: each of the two helpers is consumed in two places (contract + receipt);
        # the skip_dry receipt key still holds the literal CLI flag value (rev3 correction unchanged).
        src = TRAIN_PPO.read_text()
        self.assertEqual(
            src.count('"dry_curriculum": _contract_dry_curriculum(args),'), 2)
        self.assertEqual(
            src.count('"bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),'),
            2)
        self.assertIn('"skip_dry": bool(args.skip_dry),', src)   # contract key
        self.assertIn('"skip_dry": args.skip_dry,', src)          # receipt key

    def test_legacy_print_follows_constant(self):
        src = TRAIN_PPO.read_text()
        self.assertIn("contract_revision {_CONTRACT_REVISION} contract", src)
        self.assertNotIn("write a contract_revision 4 contract", src)
        self.assertNotIn("write a contract_revision 5 contract", src)   # must not be hard-coded


class Rev4CheckpointRejectionTests(unittest.TestCase):
    """E4/R8: non-legacy continuation of an old checkpoint (rev4 contract) is rejected; the known cost is recorded as is."""

    @staticmethod
    def _rev4_saved():
        # Simulate the contract stored in the zip by a leg before E4 (rev4): no dry_curriculum/bc_aux keys.
        contract = _contract_for(_leg_args(skip_dry=True))
        saved = {key: value for key, value in contract.items()
                 if key not in ("dry_curriculum", "bc_aux")}
        saved["contract_revision"] = 4
        return saved

    def test_rev4_resume_rejected_with_drift(self):
        current = _contract_for(_leg_args(skip_dry=True))
        with self.assertRaisesRegex(ValueError, "contract drift") as ctx:
            _validate_resume_contract(self._rev4_saved(), current)
        message = str(ctx.exception)
        self.assertIn("contract_revision", message)
        self.assertIn("dry_curriculum", message)
        self.assertIn("bc_aux", message)

    def test_legacy_route_unchanged(self):
        # Old artifacts without a contract still go through the explicit one-time --allow-legacy-resume migration (original R8 semantics).
        current = _contract_for(_leg_args(skip_dry=True))
        with self.assertRaisesRegex(ValueError, "has no training_contract"):
            _validate_resume_contract(None, current)
        _validate_resume_contract(None, current, allow_legacy_resume=True)

    def test_rev5_self_consistent_resume_passes(self):
        args = _leg_args(dry_curriculum_schedule=MAIN_TABLE_LITERAL)
        _validate_resume_contract(_contract_for(args), _contract_for(args))


class ContinuationOptimizerContractTests(unittest.TestCase):
    """reset is an event of this leg; only lr/target_kl are the persistent recipe."""

    def test_reset_clears_all_adam_state_and_preserves_policy_weights(self):
        model = _real_policy(seed=31)
        self.addCleanup(model.env.close)
        loss = sum(parameter.square().mean()
                   for parameter in model.policy.parameters())
        model.policy.optimizer.zero_grad()
        loss.backward()
        model.policy.optimizer.step()
        self.assertTrue(model.policy.optimizer.state)
        before = {key: value.detach().clone()
                  for key, value in model.policy.state_dict().items()}
        _reset_policy_optimizer(model, 1e-4)
        self.assertFalse(model.policy.optimizer.state)
        self.assertEqual(model.learning_rate, 1e-4)
        for key, value in model.policy.state_dict().items():
            self.assertTrue(th.equal(before[key], value), key)

    def test_reset_checkpoint_can_resume_normally_next_leg(self):
        # reset_optimizer does not enter the contract, so a reset leg's artifacts do not permanently require the next leg
        # to keep carrying the flag; its settled lr/target_kl remain persistently equal as usual.
        reset_leg = _contract_for(
            _leg_args(lr=1e-4, reset_optimizer=True, target_kl=0.02))
        normal_next = _contract_for(
            _leg_args(lr=1e-4, reset_optimizer=False, target_kl=0.02))
        self.assertNotIn("optimizer_reset", reset_leg)
        _validate_resume_contract(reset_leg, normal_next)

    def test_lr_change_requires_explicit_reset_and_target_kl_change_is_explicit(self):
        saved = _contract_for(_leg_args(lr=3e-4, target_kl=None))
        lower = _contract_for(_leg_args(lr=1e-4, target_kl=0.02,
                                        reset_optimizer=True))
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _validate_resume_contract(saved, lower)
        _validate_resume_contract(
            saved, lower, allow_optimizer_reset=True,
            allow_target_kl_change=True)


class E6FrozenDemosTests(unittest.TestCase):
    """v4: bound to the current strict PASS receipt; the historical v3 constants no longer earn training trust."""

    def test_v3_constant_is_compat_only_and_stale_report_rejected(self):
        self.assertEqual(
            _BC_V1_DEMOS_SHA256,
            "3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363")
        with tempfile.TemporaryDirectory() as d:
            fake = _fake_demos(d)
            fake.with_name("policy_sd.pt").write_bytes(b"policy")
            with mock.patch.object(
                    train_ppo, "_validate_bc_report",
                    side_effect=ValueError("BC receipt protocol outdated")):
                with self.assertRaisesRegex(ValueError, "protocol outdated"):
                    _assert_bc_v1_demos_frozen(fake)

    def test_current_pass_report_binds_live_bytes_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as d:
            fake = _fake_demos(d)
            fake.with_name("policy_sd.pt").write_bytes(b"policy")
            actual = hashlib.sha256(fake.read_bytes()).hexdigest()
            with mock.patch.object(
                    train_ppo, "_validate_bc_report",
                    return_value={"demos_sha256": actual}):
                self.assertEqual(_assert_bc_v1_demos_frozen(fake), actual)
            with mock.patch.object(
                    train_ppo, "_validate_bc_report",
                    return_value={"demos_sha256": "d" * 64}):
                with self.assertRaisesRegex(ValueError, "bytes drifted from the current PASS receipt"):
                    _assert_bc_v1_demos_frozen(fake)
            with self.assertRaisesRegex(ValueError, "unreadable"):
                _assert_bc_v1_demos_frozen(pathlib.Path(d) / "absent.npz")

    def test_probe_constructors_require_current_pass_binding(self):
        with tempfile.TemporaryDirectory() as d:
            fake = _fake_demos(d)
            with self.assertRaisesRegex(ValueError, "current weights missing"):
                DistillCeProbe(pathlib.Path(d), str(fake), every=49_152)
            with self.assertRaisesRegex(ValueError, "current weights missing"):
                DryWindowMetricsCallback(pathlib.Path(d), str(fake),
                                         every=49_152)

    def test_dry_window_gates_wired_to_current_receipt_binding(self):
        src = TRAIN_PPO.read_text()
        self.assertEqual(src.count("_assert_bc_v1_demos_frozen(demos)"), 2)
        self.assertEqual(
            src.count("expected_sha = _assert_bc_v1_demos_frozen(demos_npz)"),
            2)
        self.assertNotIn("demos_npz, _BC_V1_DEMOS_SHA256", src)
        # the v2 surface is not pinned: BC-v2 enters the auxiliary loss only through --bc-aux-demos (canonical unchanged)
        self.assertNotIn("_assert_bc_v1_demos_frozen(args.bc_aux_demos)", src)


class DryAnchorPartitionTests(unittest.TestCase):
    """worker v4 pre-dry = col296 / decoded col297 up to the cap-1 frontier."""

    def test_dual_channel_cap_minus_one_latch_decode_and_complement(self):
        from diablogym.options_env import FARM_SCENE_CAP, KILL_PATIENCE

        x = np.zeros((7, 298), dtype=np.float32)
        x[0, 296] = 1.0
        x[1, 296] = np.float32((KILL_PATIENCE - 1) / KILL_PATIENCE)
        x[2, 296] = np.float32((KILL_PATIENCE - 2) / KILL_PATIENCE)
        x[3, 297] = np.float32(
            (FARM_SCENE_CAP - 1) / FARM_SCENE_CAP)
        x[4, 297] = np.float32(
            (FARM_SCENE_CAP - 2) / FARM_SCENE_CAP)
        # The negative domain discloses "already drank in this window", but abs(value)-1 must still recover the same scene clock.
        x[5, 297] = np.float32(
            -(1.0 + (FARM_SCENE_CAP - 1) / FARM_SCENE_CAP))
        x[6, 297] = -1.5
        dry, fresh = _dry_anchor_partition(x)
        np.testing.assert_array_equal(
            dry, np.asarray([True, True, False, True, False, True, False]))
        np.testing.assert_array_equal(fresh, ~dry)
        self.assertFalse(bool((dry & fresh).any()))
        self.assertTrue(bool((dry | fresh).all()))

    def test_loader_accepts_col296_cap_minus_one_state(self):
        from diablogym.options_env import KILL_PATIENCE

        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "demos.npz"
            x = np.zeros((4, 298), dtype=np.float32)
            x[0, 296] = np.float32(
                (KILL_PATIENCE - 1) / KILL_PATIENCE)
            y = np.full(4, 9, dtype=np.int64)
            groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
            np.savez_compressed(
                p, X=x, Y=y, episode_id=groups)
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            loaded, _, actual = _load_dry_anchor_demos(p, sha)
            self.assertEqual(actual, sha)
            self.assertEqual(loaded.shape, (4, 298))

    def test_all_four_consumers_use_single_partition_helper(self):
        src = TRAIN_PPO.read_text()
        # one helper definition + four consumers: loader/sentinel/distill/drywin.
        self.assertEqual(src.count("_dry_anchor_partition("), 5)
        self.assertNotIn("X[:, 297] == 1.0", src)
        self.assertNotIn("X[:, 297] == 0.0", src)


class DistillCeProbeTests(unittest.TestCase):
    """E5-1: offline probe of dry/fresh distill_ce in separate columns (the twin DryAnchorSentinel shape)."""

    @classmethod
    def setUpClass(cls):
        cls.model = _real_policy()
        cls.teacher = _teacher_298()

    def _probe(self, run_dir, every=49_152):
        actual = hashlib.sha256(CANONICAL_DEMOS.read_bytes()).hexdigest()
        with mock.patch.object(
                train_ppo, "_assert_bc_v1_demos_frozen",
                return_value=actual):
            cb = DistillCeProbe(
                run_dir, str(CANONICAL_DEMOS), every=every)
        probe_model = types.SimpleNamespace(
            policy=self.model.policy, teacher=self.teacher, device="cpu",
            distill_beta=0.015625,
            _last_effective_distill_beta=0.015625,
            _distill_actor_rollouts_completed=0,
            _bc_aux_circuit_spec=None)
        for name in (
                "_teacher_probs",
                "_student_raw_action_logits",
                "_student_distillation_logits"):
            setattr(
                probe_model,
                name,
                types.MethodType(
                    getattr(leashed_ppo.LeashedMaskablePPO, name),
                    probe_model))
        cb.model = probe_model
        return cb

    def test_emit_schema_and_group_split(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._probe(run_dir)
            cb.num_timesteps = 3_547_136
            cb._emit(final=False)
            lines = [json.loads(line) for line in
                     (run_dir / "distill_ce_probe.jsonl").read_text()
                     .strip().splitlines()]
            self.assertEqual(len(lines), 1)
            line = lines[0]
            self.assertEqual(set(line), {
                "probe", "step",
                "dry_ce", "dry_teacher_entropy", "dry_kl", "dry_tv", "dry_n",
                "fresh_ce", "fresh_teacher_entropy", "fresh_kl", "fresh_tv",
                "fresh_n", "beta_initial", "beta",
                "distill_actor_rollouts_completed",
                "mask_mode", "demos_sha16",
            })
            self.assertEqual(line["probe"], "distill-ce")
            self.assertEqual(line["step"], 3_547_136)
            self.assertEqual(
                line["mask_mode"], "legacy-root-exclude-a12-a14")
            self.assertEqual(
                line["demos_sha16"],
                hashlib.sha256(CANONICAL_DEMOS.read_bytes()).hexdigest()[:16])
            self.assertEqual(line["beta"], 0.015625)
            self.assertEqual(line["beta_initial"], 0.015625)
            self.assertEqual(line["distill_actor_rollouts_completed"], 0)
            self.assertGreater(line["dry_n"], 0)
            self.assertGreater(line["fresh_n"], 0)
            for key in (
                    "dry_ce", "dry_teacher_entropy", "dry_kl", "dry_tv",
                    "fresh_ce", "fresh_teacher_entropy", "fresh_kl",
                    "fresh_tv"):
                self.assertTrue(np.isfinite(line[key]))
                self.assertGreaterEqual(line[key], 0.0)
            self.assertGreater(line["dry_ce"], 0.0)
            self.assertGreater(line["fresh_ce"], 0.0)

    def test_zero_touch_of_params_and_global_rng(self):
        # Training path untouched: building and firing the probe changes no policy parameter and consumes no global RNG (dedicated rng).
        with tempfile.TemporaryDirectory() as d:
            before_params = [p.detach().clone()
                            for p in self.model.policy.parameters()]
            torch_state = th.get_rng_state().clone()
            np_state = np.random.get_state()
            cb = self._probe(pathlib.Path(d))
            cb.num_timesteps = 3_547_136
            cb._emit(final=False)
            for before, after in zip(before_params,
                                     self.model.policy.parameters()):
                self.assertTrue(th.equal(before, after.detach()))
            self.assertTrue(th.equal(torch_state, th.get_rng_state()))
            after_np = np.random.get_state()
            self.assertEqual(np_state[0], after_np[0])
            self.assertTrue(np.array_equal(np_state[1], after_np[1]))
            self.assertEqual(np_state[2:], after_np[2:])

    def test_missing_teacher_fails_loud(self):
        with tempfile.TemporaryDirectory() as d:
            actual = hashlib.sha256(CANONICAL_DEMOS.read_bytes()).hexdigest()
            with mock.patch.object(
                    train_ppo, "_assert_bc_v1_demos_frozen",
                    return_value=actual):
                cb = DistillCeProbe(
                    pathlib.Path(d), str(CANONICAL_DEMOS), every=49_152)
            cb.model = types.SimpleNamespace(
                policy=self.model.policy, teacher=None, device="cpu")
            cb.num_timesteps = 100
            with self.assertRaisesRegex(ValueError, "needs the teacher present"):
                cb._emit(final=False)

    def test_cadence_aligns_to_next_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            cb = self._probe(pathlib.Path(d))
            cb.num_timesteps = 3_497_984
            cb._on_training_start()
            self.assertEqual(cb.next_at, 3_538_944)   # ((⌊·/49152⌋)+1)×49152
        with self.assertRaisesRegex(ValueError, "interval must be > 0"):
            DistillCeProbe(pathlib.Path("."), str(CANONICAL_DEMOS), every=0)

    def test_ce_formula_mirrors_leashed_rubberband(self):
        # Formula surface: mean of -sum t_probs*logp_all (mirrors leashed_ppo train() :356).
        src = TRAIN_PPO.read_text()
        self.assertIn("-(t_probs * logp_all).sum(dim=-1).mean()", src)
        leashed_src = (ROOT / "train" / "leashed_ppo.py").read_text()
        self.assertIn("-(t_probs * logp_all).sum(dim=-1).mean()", leashed_src)


class DryWindowMetricsTests(unittest.TestCase):
    """E5-2: dry-window behaviour instrument (dry-state entropy / action distribution + dry and fresh window wage and width; recorded, not enforced)."""

    @classmethod
    def setUpClass(cls):
        cls.model = _real_policy(seed=11)

    def _cb(self, run_dir, every=49_152):
        actual = hashlib.sha256(CANONICAL_DEMOS.read_bytes()).hexdigest()
        with mock.patch.object(
                train_ppo, "_assert_bc_v1_demos_frozen",
                return_value=actual):
            cb = DryWindowMetricsCallback(
                run_dir, str(CANONICAL_DEMOS), every=every)
        cb.model = types.SimpleNamespace(policy=self.model.policy,
                                         device="cpu")
        return cb

    @staticmethod
    def _info(dry, W, tau, dlvl_end):
        return {"option_extra": {"dry": dry, "W": W, "tau": tau,
                                 "dlvl_end": dlvl_end}}

    def test_window_aggregation_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._cb(run_dir)
            cb.num_timesteps = 10_000            # boundary not reached: aggregate only, do not fire
            cb.locals = {"infos": [
                self._info(True, -0.5, 20, 3), self._info(True, -0.1, 30, 3),
                self._info(False, 2.0, 40, 4), {"overridden": False}]}
            self.assertTrue(cb._on_step())
            self.assertFalse((run_dir / "drywin_metrics.jsonl").exists())
            cb.num_timesteps = 49_152            # boundary reached: fire
            cb.locals = {"infos": []}
            self.assertTrue(cb._on_step())
            line = json.loads(
                (run_dir / "drywin_metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(
                set(line), {"metrics", "step", "dry_state_entropy",
                            "dry_state_n", "dry_state_argmax_hist", "windows",
                            "mask_mode", "demos_sha16"})
            self.assertEqual(line["metrics"], "drywin")
            self.assertEqual(line["step"], 49_152)
            self.assertEqual(line["mask_mode"], "dry-anchor-legacy")
            self.assertEqual(
                line["demos_sha16"],
                hashlib.sha256(CANONICAL_DEMOS.read_bytes()).hexdigest()[:16])
            self.assertEqual(line["windows"]["dry"],
                             {"n": 2, "wage_mean": -0.3, "tau_mean": 25.0,
                              "depth_mean": 3.0})
            self.assertEqual(line["windows"]["fresh"],
                             {"n": 1, "wage_mean": 2.0, "tau_mean": 40.0,
                              "depth_mean": 4.0})
            self.assertTrue(np.isfinite(line["dry_state_entropy"]))
            hist = line["dry_state_argmax_hist"]
            self.assertEqual(len(hist), 15)
            self.assertEqual(sum(hist), line["dry_state_n"])
            self.assertEqual(hist[11], 0)   # old-criterion mask: 11/12 always masked
            self.assertEqual(hist[12], 0)

    def test_interval_reset_and_fail_closed_zero_coverage(self):
        # Interval reset: no new window after firing -> the n=0 group records n:0, with a null mean that is not dropped.
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._cb(run_dir)
            cb.locals = {"infos": [self._info(True, -0.5, 20, 3)]}
            cb.num_timesteps = 49_152
            cb._on_step()
            cb.locals = {"infos": []}
            cb.num_timesteps = 98_304
            cb._on_step()
            lines = [json.loads(raw) for raw in
                     (run_dir / "drywin_metrics.jsonl").read_text()
                     .strip().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[1]["windows"]["dry"],
                             {"n": 0, "wage_mean": None, "tau_mean": None,
                              "depth_mean": None})
            self.assertEqual(lines[1]["windows"]["fresh"]["n"], 0)

    def test_final_emit_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._cb(run_dir)
            cb.num_timesteps = 49_152
            cb.locals = {"infos": []}
            cb._on_step()
            cb._on_training_end()   # the sync point has already fired -> deduplicated, not rewritten
            lines = (run_dir / "drywin_metrics.jsonl").read_text() \
                .strip().splitlines()
            self.assertEqual(len(lines), 1)
            cb.num_timesteps = 60_000
            cb._on_training_end()   # unaligned tail point -> final row
            lines = [json.loads(raw) for raw in
                     (run_dir / "drywin_metrics.jsonl").read_text()
                     .strip().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertIs(lines[1]["final"], True)


class A12CanaryTests(unittest.TestCase):
    """E5-3: canary a12-per-episode statistics function + recorder (for offline checkpoint sequences; recorded, not enforced)."""

    def test_stats_correctness(self):
        stats = a12_canary_stats([0, 0, 4, 1])
        self.assertEqual(stats, {"episodes": 4, "a12_total": 5,
                                 "a12_per_episode": 1.25,
                                 "episodes_with_a12": 2, "a12_max": 4})
        zero = a12_canary_stats([0] * 32)
        self.assertEqual(zero["a12_per_episode"], 0.0)   # zero use is recorded faithfully as 0.0
        self.assertEqual(zero["episodes"], 32)

    def test_stats_fail_loud(self):
        for bad in ([], [1, -1], [1.5], [True], [1, None]):
            with self.assertRaises(ValueError):
                a12_canary_stats(bad)

    def test_recorder_schema_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "a12_canary.jsonl"
            stats = a12_canary_stats([0, 3])
            line = record_a12_canary(out, checkpoint_step=3_596_288,
                                     manager="M29", stats=stats,
                                     tag="lfull-canary")
            written = json.loads(out.read_text().strip())
            self.assertEqual(written, line)
            self.assertEqual(written["canary"], "a12")
            self.assertEqual(written["schema_version"], "a12-canary/1")
            self.assertEqual(written["checkpoint_step"], 3_596_288)
            self.assertEqual(written["manager"], "M29")
            self.assertEqual(written["tag"], "lfull-canary")
            self.assertEqual(written["a12_per_episode"], 1.5)
            # a row without a tag carries no tag key (the closed schema can decide)
            second = record_a12_canary(out, checkpoint_step=0,
                                       manager="H", stats=stats)
            self.assertNotIn("tag", second)

    def test_recorder_fail_loud(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "a12_canary.jsonl"
            stats = a12_canary_stats([1])
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=-1, manager="H",
                                  stats=stats)
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=1, manager="",
                                  stats=stats)
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=1, manager="H",
                                  stats={"episodes": 1})
            extra = dict(stats, rogue=1)
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=1, manager="H",
                                  stats=extra)
            self.assertFalse(out.exists())   # fail-loud writes no half line


class E5ZeroIntrusionTests(unittest.TestCase):
    """E5: inactive = the code path equals HEAD (G0-2a precondition; knobs default 0 + mount guard)."""

    def test_cli_defaults_are_off(self):
        src = TRAIN_PPO.read_text()
        self.assertIn('ap.add_argument("--distill-ce-probe-every", type=int,'
                      " default=0,", src)
        self.assertIn('ap.add_argument("--drywin-metrics-every", type=int,'
                      " default=0,", src)

    def test_mount_guards_gate_on_knobs(self):
        src = TRAIN_PPO.read_text()
        self.assertIn("if (args.worker and args.distill_ce_probe_every > 0)",
                      src)
        self.assertIn("if (args.worker and args.drywin_metrics_every > 0)", src)
        self.assertIn("([distill_ce_cb] if distill_ce_cb else [])", src)
        self.assertIn("([drywin_cb] if drywin_cb else [])", src)

    def test_a12_functions_have_no_training_path_caller(self):
        # (3) is an offline statistics component: zero calls inside the training process (exactly one definition each).
        src = TRAIN_PPO.read_text()
        self.assertEqual(src.count("def a12_canary_stats("), 1)
        self.assertEqual(src.count("a12_canary_stats("), 1)
        self.assertEqual(src.count("def record_a12_canary("), 1)
        self.assertEqual(src.count("record_a12_canary("), 1)

    def test_cli_rejects_bad_knobs_and_documents_them(self):
        help_run = _run_cli("--help")
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--distill-ce-probe-every", help_run.stdout)
        self.assertIn("--drywin-metrics-every", help_run.stdout)
        bad = _run_cli("--total-steps", "2048",
                       "--distill-ce-probe-every", "-1")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("--distill-ce-probe-every must not be negative", bad.stderr)
        nonworker = _run_cli("--total-steps", "2048",
                             "--distill-ce-probe-every", "49152")
        self.assertNotEqual(nonworker.returncode, 0)
        self.assertIn("only applies to --worker --algo mppo", nonworker.stderr)
        nonworker2 = _run_cli("--total-steps", "2048",
                              "--drywin-metrics-every", "49152")
        self.assertNotEqual(nonworker2.returncode, 0)
        self.assertIn("--drywin-metrics-every only applies to --worker",
                      nonworker2.stderr)

    def test_probe_teacher_preflight_wired(self):
        src = TRAIN_PPO.read_text()
        self.assertIn("--distill-ce-probe-every>0 requires a Leashed teacher present", src)


class LauncherFixTests(unittest.TestCase):
    """E8: fix for the false alarm in launch_case.sh PID bookkeeping (minor, OPS section of the B1 verdict)."""

    def test_bash_syntax(self):
        run = subprocess.run(["bash", "-n", str(LAUNCHER)],
                             text=True, capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_tree_assertion_replaces_wrong_pid_check(self):
        src = LAUNCHER.read_text()
        # Fix surface: the caffeinate assertion now checks "inside the $PID process tree" (compatible with the process itself or a child).
        self.assertIn('pgrep -P "$PID" -f caffeinate', src)
        self.assertIn('ps -o command= -p "$PID" | grep -q "caffeinate"', src)
        self.assertIn("no caffeinate in the process tree", src)
        self.assertNotIn("is not a caffeinate process", src)   # the old false-alarm text is gone

    def test_behavior_surface_unchanged(self):
        src = LAUNCHER.read_text()
        # Behaviour unchanged: nohup + caffeinate -is + orphaning + log + receipt + heartbeat details as before.
        self.assertIn('nohup caffeinate -is "$PY" "$DRIVER" "$@"', src)
        self.assertIn('disown "$PID"', src)
        self.assertIn("launch_receipt.json", src)
        self.assertIn("progress.jsonl", src)
        self.assertIn("exit 70", src)
        self.assertIn("exit 71", src)
        self.assertIn("exit 64", src)
        self.assertIn("set -euo pipefail", src)

    def test_liveness_check_precedes_tree_assertion(self):
        src = LAUNCHER.read_text()
        self.assertLess(src.index('if ! ps -p "$PID"'),
                        src.index('pgrep -P "$PID" -f caffeinate'))


if __name__ == "__main__":
    unittest.main()
