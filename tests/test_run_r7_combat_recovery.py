"""Pure, seed-free protocol tests for the R7 campaign launcher."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

import run_r7_combat_recovery as r7  # noqa: E402


def _training_death_evidence(
        observed: bool = True, *, cost: float = 32.0) -> dict:
    direct = 1 if observed else 0
    return {
        "schema_version": r7.TERMINAL_DEATH_EVIDENCE_SCHEMA,
        "sentinel_sha256": "9" * 64,
        "sentinel_step": r7.TARGET_STEPS,
        "opportunity_status": (
            "TRAINING_FAILURE_OBSERVED"
            if observed else "NO_TRAINING_FAILURE_OBSERVED"
        ),
        "reward_mechanism_triggered": observed,
        "fast_forward_reward_credit_mode": r7.FAST_FORWARD_CREDIT,
        "configured_additional_terminal_death_cost": cost,
        "control_path_counts": {
            "interrupted_resets": 0,
            "manual_ff_calls": 0,
        },
        "death_counts": {
            "direct": direct,
            "transition_ff": 0,
            "reset_ff": 0,
            "manual_ff": 0,
        },
        "death_rewards": {
            "direct_existing_terminal_death_reward":
                -8.0 if observed else 0.0,
            "direct_additional_terminal_death_reward":
                -cost if observed else 0.0,
            "transition_ff_terminal_death_reward": 0.0,
            "transition_ff_additional_terminal_death_reward": 0.0,
            "credited_ff_terminal_death_reward": 0.0,
            "reset_ff_terminal_death_reward": 0.0,
            "reset_ff_additional_terminal_death_reward": 0.0,
            "additional_terminal_death_reward":
                -cost if observed else 0.0,
        },
        "no_progress_timeout_counts": {
            "direct": 0,
            "transition_ff": 0,
            "reset_ff": 0,
            "manual_ff": 0,
        },
        "no_progress_timeout_rewards": {
            "direct_no_progress_timeout_failure_reward": 0.0,
            "transition_ff_no_progress_timeout_failure_reward": 0.0,
            "reset_ff_no_progress_timeout_failure_reward": 0.0,
            "manual_ff_no_progress_timeout_failure_reward": 0.0,
            "credited_no_progress_timeout_failure_reward": 0.0,
        },
    }


def _analysis(passed: bool, *, cost: float = 32.0) -> dict:
    return {
        "verdict": {"status": "PASS" if passed else "FAIL"},
        "training_death_reward_evidence":
            _training_death_evidence(cost=cost),
    }


def _worker_final_sentinel(
        *, cost: float = 32.0, direct: int = 2,
        transition: int = 3, reset: int = 1,
        manual: int = 0, direct_timeout: int = 0,
        transition_timeout: int = 0,
        reset_timeout: int = 0) -> dict:
    ff_deaths = transition + reset + manual
    ff_timeouts = transition_timeout + reset_timeout
    all_deaths = direct + ff_deaths
    all_timeouts = (
        direct_timeout + transition_timeout + reset_timeout)
    direct_existing = -8.0 * direct
    transition_existing = -8.0 * transition
    reset_existing = -8.0 * reset
    direct_additional = -cost * direct
    transition_additional = -cost * transition
    reset_additional = -cost * reset
    timeout_unit_reward = -(8.0 + cost)
    direct_timeout_reward = timeout_unit_reward * direct_timeout
    transition_timeout_reward = (
        timeout_unit_reward * transition_timeout)
    reset_timeout_reward = timeout_unit_reward * reset_timeout
    reasons = {}
    ff_reasons = {}
    if all_deaths:
        reasons["death"] = all_deaths
    if ff_deaths:
        ff_reasons["death"] = ff_deaths
    if all_timeouts:
        reasons["exhausted"] = all_timeouts
    if ff_timeouts:
        ff_reasons["exhausted"] = ff_timeouts
    return {
        "sentinel": "v23",
        "step": r7.TARGET_STEPS,
        "windows": max(2, direct + 1),
        "dry": 1,
        "fresh": max(1, direct),
        "ff_windows": ff_deaths + ff_timeouts,
        "ff_dry": 0,
        "ff_terminals": ff_deaths + ff_timeouts,
        "episodes": max(1, all_deaths + all_timeouts),
        "reseeds": reset + reset_timeout,
        "interrupted_resets": 0,
        "manual_ff_calls": 0,
        "direct_terminal_deaths": direct,
        "transition_ff_terminal_deaths": transition,
        "reset_ff_terminal_deaths": reset,
        "manual_ff_terminal_deaths": manual,
        "direct_no_progress_timeouts": direct_timeout,
        "transition_ff_no_progress_timeouts": transition_timeout,
        "reset_ff_no_progress_timeouts": reset_timeout,
        "manual_ff_no_progress_timeouts": 0,
        "transition_ff_reward": transition_existing,
        "reset_ff_reward": reset_existing,
        "manual_ff_reward": 0.0,
        "direct_existing_terminal_death_reward": direct_existing,
        "direct_additional_terminal_death_reward": direct_additional,
        "transition_ff_terminal_death_reward": transition_existing,
        "transition_ff_additional_terminal_death_reward":
            transition_additional,
        "credited_ff_terminal_death_reward":
            transition_existing + transition_additional,
        "reset_ff_terminal_death_reward": reset_existing,
        "reset_ff_additional_terminal_death_reward": reset_additional,
        "additional_terminal_death_reward":
            direct_additional + transition_additional,
        "direct_no_progress_timeout_failure_reward":
            direct_timeout_reward,
        "transition_ff_no_progress_timeout_failure_reward":
            transition_timeout_reward,
        "reset_ff_no_progress_timeout_failure_reward":
            reset_timeout_reward,
        "manual_ff_no_progress_timeout_failure_reward": 0.0,
        "credited_no_progress_timeout_failure_reward":
            direct_timeout_reward + transition_timeout_reward,
        "dry_share": round(
            1 / (1 + max(1, direct)), 4),
        "reasons": reasons,
        "ff_reasons": ff_reasons,
        "fast_forward_reward_credit_mode": r7.FAST_FORWARD_CREDIT,
        "configured_additional_terminal_death_cost": cost,
        "top1_action": 9,
        "top1_share": 0.25,
        "beta_initial": r7.DISTILL_BETA,
        "beta": 0.0,
        "distill_actor_rollouts_completed": r7.ACTOR_TRAIN_CALLS,
        "distill_ce": 1.25,
        "teacher_entropy": None,
        "distill_kl": None,
        "distill_tv": None,
        "teacher_diverge": 0.1,
        "final": True,
    }


def _formal_pg_checkpoint(
        timeout_rows: tuple[tuple[int, float, float, float], ...] = (),
) -> dict:
    """Build the compact launcher's inputs without running a real rollout."""
    optimizer_steps = (
        r7.WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT)
    endpoints = range(
        r7.START_STEPS + r7.CRITIC_WARMUP_STEPS + r7.ROLLOUT_QUANTUM,
        r7.TARGET_STEPS + 1,
        r7.ROLLOUT_QUANTUM,
    )
    receipts = []
    for index, endpoint in enumerate(endpoints):
        if index < len(timeout_rows):
            timeout_samples, timeout_base, timeout_additional, timeout_total = (
                timeout_rows[index])
        else:
            timeout_samples = 0
            timeout_base = timeout_additional = timeout_total = 0.0
        requested = [0] * 15
        executed = [0] * 15
        requested[9] = r7.ROLLOUT_QUANTUM
        executed[9] = r7.ROLLOUT_QUANTUM
        receipts.append({
            "rollout_end_timesteps": endpoint,
            "qualifies": True,
            "optimizer_steps": optimizer_steps,
            "transition_reward_nonzero_samples": 1,
            "no_progress_timeout_samples": timeout_samples,
            "no_progress_timeout_base_failure_reward_sum": timeout_base,
            "no_progress_timeout_additional_failure_reward_sum":
                timeout_additional,
            "no_progress_timeout_failure_reward_sum": timeout_total,
            "requested_action_counts": requested,
            "executed_action_counts": executed,
            "combat_effect_samples": 1,
            "combat_transition_reward_nonzero_samples": 1,
            "combat_transition_reward_positive_samples": 1,
            "combat_transition_reward_sum": 1.0,
            "combat_positive_advantage_samples": 1,
            "combat_reward_centered_actor_grad_norm": 1.0,
            "combat_reward_centered_context_encoder_grad_norm": 1.0,
            "combat_reward_centered_context_interaction_grad_norm": 1.0,
            "reward_centered_actor_grad_norm": 1.0,
            "reward_centered_context_encoder_grad_norm": 1.0,
            "reward_centered_context_interaction_grad_norm": 1.0,
            "pure_ppo_actor_grad_norm_max": 1.0,
            "pure_ppo_root_grad_norm_max": 1.0,
            "pure_ppo_context_encoder_grad_norm_max": 1.0,
            "pure_ppo_context_interaction_grad_norm_max": 1.0,
            "combined_context_on_pure_ppo_projection": 1.0,
            "combined_root_on_pure_ppo_projection": 1.0,
            "pure_ppo_actor_on_combat_reward_projection": 1.0,
            "pure_ppo_root_on_combat_reward_projection": 1.0,
            "pure_ppo_context_on_combat_reward_projection": 1.0,
            "optimizer_delta_actor_l2": 2.0 ** 0.5,
            "optimizer_delta_actor_on_combat_reward_descent_projection": 1.0,
            "optimizer_delta_actor_on_combat_reward_descent_cosine": 1.0,
            "optimizer_delta_root_l2": 1.0,
            "optimizer_delta_root_on_combat_reward_descent_projection": 1.0,
            "optimizer_delta_root_on_combat_reward_descent_cosine": 1.0,
            "optimizer_delta_context_l2": 1.0,
            "optimizer_delta_context_on_combat_reward_descent_projection": 1.0,
            "optimizer_delta_context_on_combat_reward_descent_cosine": 1.0,
        })
    return {
        "_worker_onpolicy_pg_audit_required": True,
        "_worker_onpolicy_pg_rollout_receipts": receipts,
        "_worker_onpolicy_pg_joint_rollouts": r7.ACTOR_TRAIN_CALLS,
        "_worker_onpolicy_pg_qualifying_rollouts": r7.ACTOR_TRAIN_CALLS,
        "_actor_optimizer_steps_completed":
            optimizer_steps * r7.ACTOR_TRAIN_CALLS,
    }


def _dry_curriculum_rows() -> list[dict]:
    table = r7.train_ppo._parse_dry_curriculum_schedule(r7.CURRICULUM)
    return [
        {
            "rollout_index": index,
            "p": float(probability),
            "num_timesteps":
                r7.START_STEPS + index * r7.ROLLOUT_QUANTUM,
            "boundary_preapplied": index > 0,
            "cached_dual_observation_refreshed": True,
        }
        for index, probability in enumerate(table)
    ]


class R7FrozenProtocolTests(unittest.TestCase):
    def test_artifact_hash_constants_match_disk(self):
        self.assertEqual(r7._sha256(r7.V28_ZIP), r7.V28_SHA256)
        self.assertEqual(r7._sha256(r7.M29_NPZ), r7.M29_SHA256)
        self.assertEqual(r7._sha256(r7.KING_SD), r7.KING_SHA256)

    def test_seed_pools_are_fresh_registered_and_disjoint(self):
        r7._require_seed_discipline()
        self.assertEqual(r7.BC_V1_RANGE, (2_142_000, 2_142_128))
        self.assertEqual(r7.BC_V2_RANGE, (2_143_000, 2_143_384))
        self.assertIn((2_100_000, 2_100_128),
                      r7.BC_RESERVED_SEED_RANGES)
        self.assertIn((2_101_000, 2_101_384),
                      r7.BC_RESERVED_SEED_RANGES)
        self.assertEqual(
            tuple(range(*r7.BC_V1_RANGE)),
            r7.train_ppo._WORKER_BC_DEMO_SEEDS)
        self.assertEqual(
            tuple(range(*r7.BC_V2_RANGE)),
            r7.train_ppo._BC_V2_COLLECTION_EPISODES)
        self.assertEqual(len(r7.DEV_POOLS["dev-a"]), 128)
        self.assertEqual(len(r7.DEV_POOLS["dev-b"]), 128)
        self.assertEqual(len(r7.FINAL_POOL), 256)
        eval_bank = set(range(*r7.EVAL_BANK_RANGE))
        self.assertTrue(
            set(range(*r7.BC_V1_RANGE)).isdisjoint(eval_bank))
        self.assertTrue(
            set(range(*r7.BC_V2_RANGE)).isdisjoint(eval_bank))
        self.assertEqual(r7.DEVELOPMENT_TRAIN_SEEDS,
                         (2_130_000, 2_130_100, 2_130_200))
        self.assertEqual(r7.PRODUCTION_TRAIN_SEED, 2_130_900)

    def test_curriculum_is_exact_consumed_prefix(self):
        r7._require_curriculum_discipline()
        actual = r7.train_ppo._parse_dry_curriculum_schedule(r7.CURRICULUM)
        registered = r7.train_ppo._parse_dry_curriculum_schedule(
            "linear:1.0:0.0:92,hold:0.0:30")
        self.assertEqual(len(actual), r7.TRAIN_CALLS)
        self.assertEqual(
            tuple(actual[:r7.CRITIC_WARMUP_CALLS]),
            (1.0,) * r7.CRITIC_WARMUP_CALLS,
        )
        for left, right in zip(
                actual[r7.CRITIC_WARMUP_CALLS:],
                registered[:r7.ACTOR_TRAIN_CALLS]):
            self.assertAlmostEqual(left, right, places=15)
        self.assertEqual(
            r7.LEG_STEPS,
            r7.CRITIC_WARMUP_STEPS + r7.ACTOR_LEG_STEPS,
        )

    def test_recipe_binds_full_game_archive_requirement(self):
        self.assertEqual(r7.CAMPAIGN_REVISION, 23)
        self.assertEqual(
            r7.TRAINING_RECEIPT_SCHEMA,
            "diablogym-r7-training-artifact/9",
        )
        self.assertEqual(r7.ACTION14_LOGIT_BONUS, 2.5)
        self.assertEqual(
            r7.CAMPAIGN_RECIPE["game_data_requirement"],
            {
                "mode": "full-diablo",
                "allowed_basenames": ["DIABDAT.MPQ", "diabdat.mpq"],
                "sha256": r7.FULL_GAME_DATA_SHA256,
            },
        )
        self.assertEqual(
            r7.CAMPAIGN_RECIPE[
                "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout"],
            r7.WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        )
        self.assertEqual(
            r7.CAMPAIGN_RECIPE["worker_failure_evidence_schema"],
            r7.TERMINAL_DEATH_EVIDENCE_SCHEMA,
        )
        self.assertEqual(
            r7.CAMPAIGN_RECIPE["gear_progression_gate"],
            {
                "schema": r7.GEAR_PROGRESSION_GATE_SCHEMA,
                "minimum_action14_mask_opportunities":
                    r7.MIN_ACTION14_MASK_OPPORTUNITIES,
                "required_when_informative": {
                    "minimum_requests": 1,
                    "minimum_native_successes": 1,
                    "minimum_gear_utility_delta": 1,
                },
            },
        )

    def test_full_game_identity_is_explicit_and_exact(self):
        source = {
            "path": "/tmp/DIABDAT.MPQ",
            "sha256": r7.FULL_GAME_DATA_SHA256,
        }
        with mock.patch.object(
                r7.eval_contract, "game_data_identity",
                return_value=source):
            self.assertEqual(
                r7._full_game_data_identity(),
                {
                    "mode": "full-diablo",
                    "basename": "DIABDAT.MPQ",
                    **source,
                },
            )

    def test_formal_r7_rejects_spawn_only_before_marker_or_subprocess(self):
        spawn = {
            "path": "/tmp/spawn.mpq",
            "sha256": "6" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            marker = root / "control" / "training-fired.json"
            receipt = root / "run" / "receipt.json"
            with (
                mock.patch.object(
                    r7.eval_contract, "game_data_identity",
                    return_value=spawn),
                mock.patch.object(
                    r7, "_training_fired_path", return_value=marker),
                mock.patch.object(
                    r7, "_training_receipt_path", return_value=receipt),
                mock.patch.object(
                    r7, "_bc_identity", return_value={"bc": 1}),
                mock.patch.object(r7, "_invoke") as invoke,
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "spawn/shareware"):
                    r7._run_training_once(
                        "risk32", 42, "development")
                self.assertFalse(marker.exists())
                invoke.assert_not_called()

    def test_formal_r7_rejects_wrong_full_archive_sha(self):
        source = {
            "path": "/tmp/diabdat.mpq",
            "sha256": "0" * 64,
        }
        with mock.patch.object(
                r7.eval_contract, "game_data_identity",
                return_value=source):
            with self.assertRaisesRegex(r7.CampaignError, "SHA 漂移"):
                r7._full_game_data_identity()

    def test_generic_eval_contract_still_accepts_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            spawn = pathlib.Path(directory) / "spawn.mpq"
            spawn.write_bytes(b"shareware")
            identity = r7.eval_contract.game_data_identity(directory)
        self.assertEqual(pathlib.Path(identity["path"]).name, "spawn.mpq")
        self.assertEqual(
            identity["sha256"],
            hashlib.sha256(b"shareware").hexdigest(),
        )

    def test_dry_curriculum_ledger_closes_all_130_rollouts(self):
        rows = _dry_curriculum_rows()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "dry_curriculum.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            evidence = r7._dry_curriculum_ledger_evidence(path)
        self.assertEqual(evidence["rollout_rows"], r7.TRAIN_CALLS)
        self.assertEqual(evidence["first_rollout_index"], 0)
        self.assertEqual(
            evidence["last_rollout_index"], r7.TRAIN_CALLS - 1)
        self.assertEqual(evidence["first_global_step"], r7.START_STEPS)
        self.assertEqual(
            evidence["last_global_step"],
            r7.START_STEPS
            + (r7.TRAIN_CALLS - 1) * r7.ROLLOUT_QUANTUM,
        )
        self.assertTrue(r7._is_sha256(evidence["sha256"]))

    def test_dry_curriculum_ledger_rejects_missing_or_forged_rows(self):
        mutations = {
            "missing": lambda rows: rows.pop(),
            "index": lambda rows: rows[37].__setitem__(
                "rollout_index", 36),
            "step": lambda rows: rows[64].__setitem__(
                "num_timesteps",
                rows[64]["num_timesteps"] - r7.ROLLOUT_QUANTUM),
            "probability": lambda rows: rows[99].__setitem__(
                "p", rows[99]["p"] + 1e-12),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as d:
                rows = _dry_curriculum_rows()
                mutate(rows)
                path = pathlib.Path(d) / "dry_curriculum.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                with self.assertRaises(r7.CampaignError):
                    r7._dry_curriculum_ledger_evidence(path)

    def test_training_artifact_requires_curriculum_origin_config(self):
        expected = {
            "dry_curriculum_start_index": 0,
            "dry_curriculum_start_probability": 1.0,
        }
        r7._require_config_values(dict(expected), expected, "unit")
        with self.assertRaises(r7.CampaignError):
            r7._require_config_values(
                {
                    "dry_curriculum_start_index": 1,
                    "dry_curriculum_start_probability": 1.0,
                },
                expected,
                "unit",
            )
        source = inspect.getsource(r7._training_artifact_evidence)
        self.assertIn('"dry_curriculum_start_index": 0', source)
        self.assertIn(
            '"dry_curriculum_start_probability": 1.0', source)
        self.assertIn(
            '"dry_curriculum_ledger": dry_curriculum_ledger', source)

    def test_candidate_command_is_nonpublished_independent_retrain(self):
        command = r7._training_command(
            "risk64", r7.PRODUCTION_TRAIN_SEED, "candidate")
        self.assertEqual(
            command[command.index("--artifact-scope") + 1], "candidate")
        self.assertEqual(
            command[command.index("--resume-from") + 1], str(r7.V28_ZIP))
        self.assertEqual(
            command[command.index("--seed") + 1],
            str(r7.PRODUCTION_TRAIN_SEED),
        )
        self.assertIn("--reset-worker-critic", command)
        self.assertEqual(
            command[command.index("--critic-warmup-steps") + 1],
            str(r7.CRITIC_WARMUP_STEPS),
        )
        self.assertEqual(
            command[command.index("--gradient-clip-mode") + 1],
            "separate-root-context-critic-v2",
        )
        self.assertNotIn(
            "--legacy-worker-policy-observation-view", command)
        self.assertEqual(
            command[
                command.index("--worker-policy-observation-view") + 1],
            r7.WORKER_POLICY_OBSERVATION_VIEW,
        )
        self.assertEqual(
            r7._training_model_path(
                "risk64", r7.PRODUCTION_TRAIN_SEED, "candidate").name,
            "model_candidate.zip",
        )

    def test_expected_candidate_contract_closes_all_training_inputs(self):
        bc = {"demos_sha256": "d" * 64}
        with mock.patch.object(
                r7.train_ppo, "_implementation_bundle_sha256",
                return_value="i" * 64):
            contract = r7._expected_training_contract(
                "candidate", "risk32", bc)
        self.assertEqual(contract["artifact_scope"], "candidate")
        self.assertEqual(contract["manager_npz_sha256"], r7.M29_SHA256)
        self.assertEqual(contract["teacher_sha256"], r7.KING_SHA256)
        self.assertEqual(contract["demos_sha256"], "d" * 64)
        self.assertEqual(contract["n_steps"], r7.N_STEPS)
        self.assertEqual(contract["num_envs"], r7.NUM_ENVS)
        self.assertEqual(contract["batch_size"], 256)
        self.assertEqual(contract["gamma"], 1.0)
        self.assertEqual(contract["learning_rate"], r7.LEARNING_RATE)
        self.assertEqual(contract["ent_coef"], r7.ENT_COEF)
        self.assertEqual(contract["distill_beta"], r7.DISTILL_BETA)
        self.assertEqual(
            contract["distillation"]["excluded_actions"], [12, 14])
        self.assertEqual(
            contract["worker_action14_logit_bonus"],
            r7.ACTION14_LOGIT_BONUS)
        self.assertEqual(
            contract["manager_policy_observation_view"], "legacy-v3")
        self.assertEqual(
            contract["worker_episode_boundary"],
            r7.train_ppo._WORKER_EPISODE_BOUNDARY_V24)
        self.assertEqual(
            contract["worker_window_bootstrap"],
            "next-learning-window")
        self.assertEqual(
            contract["worker_no_progress_timeout"],
            r7.train_ppo._WORKER_NO_PROGRESS_TIMEOUT_CONTRACT)
        self.assertEqual(
            contract["gradient_clipping"]["mode"],
            "separate-root-context-critic-v2")
        self.assertFalse(
            contract["legacy_policy_observation_view"])
        self.assertEqual(
            contract["worker_policy_observation_view"],
            r7.WORKER_POLICY_OBSERVATION_VIEW)
        self.assertEqual(
            contract["observation_shape"],
            [r7.DUAL_WORKER_OBSERVATION_DIM],
        )
        canonical = r7._canonical_migration_evidence(0)
        self.assertEqual(
            contract["actor_migration"]["migrated_actor_sha256"],
            canonical["actor_migration"]["migrated_actor_sha256"])
        self.assertEqual(
            contract["critic_migration"]["source_checkpoint_sha256"],
            r7.V28_SHA256)
        self.assertEqual(
            contract["critic_migration"]["warmup_steps"],
            r7.CRITIC_WARMUP_STEPS)
        self.assertEqual(
            contract["critic_migration"][
                "worker_onpolicy_pg_audit_schema"],
            r7.WORKER_ONPOLICY_PG_AUDIT_SCHEMA)
        self.assertEqual(
            contract["critic_migration"][
                "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout"],
            r7.WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT)
        self.assertEqual(
            contract["dry_curriculum"], {"schedule": r7.CURRICULUM})
        self.assertEqual(
            contract["policy_source_roles"],
            r7._recipe_document()["policy_source_roles"],
        )
        self.assertEqual(
            contract["policy_source_roles"]["bc_v1_direct_policy_uses"],
            [],
        )

    def test_current_dual_contract_rejects_arch_capacity_pg_and_clip_tamper(
            self):
        with mock.patch.object(
                r7.train_ppo, "_implementation_bundle_sha256",
                return_value="i" * 64):
            contract = r7._expected_training_contract(
                "candidate", "risk32", {"demos_sha256": "d" * 64})
        r7.train_ppo._validate_current_dual_worker_contract(contract)
        mutations = (
            (("actor_migration", "context_architecture"), "forged"),
            (("actor_migration", "context_parameter_count"), 1),
            (("critic_migration", "critic_parameter_count"), 1),
            (("critic_migration", "worker_onpolicy_pg_audit_schema"),
             "stale"),
            (("gradient_clipping", "root_max_norm"), 0.5),
            (("policy_source_roles", "bc_v1_direct_policy_uses"),
             ["policy-initialization"]),
            (("policy_source_roles", "initialization"),
             "random-initialization"),
            (("worker_no_progress_timeout", "boundary"),
             "truncated-bootstrap"),
        )
        for path, value in mutations:
            forged = json.loads(json.dumps(contract))
            cursor = forged
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                r7.train_ppo._validate_current_dual_worker_contract(
                    forged)

    def test_actor_and_critic_migration_receipts_are_cross_bound(self):
        canonical = r7._canonical_migration_evidence(
            r7.PRODUCTION_TRAIN_SEED)
        actor = canonical["actor_migration"]
        critic = {
            **canonical["critic_reset"],
            "gradient_clip_mode": "separate-root-context-critic-v2",
            "warmup_start_timesteps": r7.START_STEPS,
            "warmup_until_timesteps":
                r7.START_STEPS + r7.CRITIC_WARMUP_STEPS,
            "warmup_steps": r7.CRITIC_WARMUP_STEPS,
            "warmup_rollout_quantum": r7.ROLLOUT_QUANTUM,
            "warmup_expected_rollouts": r7.CRITIC_WARMUP_CALLS,
            "actor_sha256": actor["migrated_actor_sha256"],
            "worker_onpolicy_pg_audit_schema":
                r7.WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            "optimizer_reset": True,
        }
        r7._validate_actor_migration_receipt(
            actor, canonical_receipt=actor)
        r7._validate_critic_migration_receipt(
            critic, seed=r7.PRODUCTION_TRAIN_SEED,
            actor_receipt=actor, canonical_evidence=canonical)
        for key in (
            "context_encoder_sha256",
            "context_interaction_sha256",
            "context_output_sha256",
        ):
            forged = {**actor, key: "0" * 64}
            with self.subTest(actor_field=key), self.assertRaisesRegex(
                    r7.CampaignError, "canonical"):
                r7._validate_actor_migration_receipt(
                    forged, canonical_receipt=actor)
        forged_critic = {
            **critic,
            "critic_sha256_after": "0" * 64,
        }
        with self.assertRaisesRegex(
                r7.CampaignError, "canonical"):
            r7._validate_critic_migration_receipt(
                forged_critic,
                seed=r7.PRODUCTION_TRAIN_SEED,
                actor_receipt=actor,
                canonical_evidence=canonical)

    def test_canonical_migration_reconstruction_preserves_global_rng(self):
        import random

        import numpy as np
        import torch

        r7._CANONICAL_MIGRATION_EVIDENCE_CACHE.clear()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        evidence = r7._canonical_migration_evidence(
            r7.PRODUCTION_TRAIN_SEED)
        self.assertEqual(
            evidence["actor_migration"]["migrated_actor_sha256"],
            evidence["runtime"]["policy"]["actor_sha256"],
        )
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        self.assertTrue(np.array_equal(
            numpy_after[1], numpy_before[1]))
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(
            torch.random.get_rng_state(), torch_before))

    def test_training_artifact_requires_exact_invocation_argv(self):
        with self.assertRaisesRegex(r7.CampaignError, "invocation"):
            r7._require_config_values(
                {},
                {"invocation_argv": ["--worker"]},
                "unit",
            )
        source = inspect.getsource(r7._training_artifact_evidence)
        self.assertIn('"invocation_argv":', source)
        self.assertNotIn('if "invocation_argv" in config', source)

    def test_every_invoke_preflights_frozen_brains(self):
        with (
            mock.patch.object(
                r7, "_frozen_inputs_identity",
                side_effect=r7.CampaignError("drift")),
            mock.patch.object(r7.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(r7.CampaignError, "drift"):
                r7._invoke(["never"], "unit")
            run.assert_not_called()

    def test_frozen_input_identity_rejects_one_hash_drift(self):
        values = {
            r7.V28_ZIP: r7.V28_SHA256,
            r7.M29_NPZ: r7.M29_SHA256,
            r7.KING_SD: "0" * 64,
        }
        with mock.patch.object(r7, "_sha256", side_effect=values.__getitem__):
            with self.assertRaisesRegex(r7.CampaignError, "KING SHA"):
                r7._frozen_inputs_identity()


class R7GearProgressionGateTests(unittest.TestCase):
    @staticmethod
    def archive(
            opportunities: int, requests: int,
            successes: int, utility_delta: int) -> dict:
        return {"agg": {
            "worker_calls": max(opportunities, requests, 1),
            "worker_action14_mask_opportunities": opportunities,
            "worker_action14_requests": requests,
            "worker_action14_native_successes": successes,
            "worker_action14_gear_utility_delta": utility_delta,
        }}

    def test_no_or_sparse_drop_pool_is_explicit_and_nonfailing(self):
        no_drop = r7._gear_progression_gate(
            self.archive(0, 0, 0, 0))
        self.assertEqual(
            no_drop["opportunity_status"],
            "NO_MASK_OPPORTUNITY_OBSERVED",
        )
        self.assertFalse(no_drop["informative"])
        self.assertTrue(no_drop["passed"])

        sparse = r7._gear_progression_gate(self.archive(
            r7.MIN_ACTION14_MASK_OPPORTUNITIES - 1,
            0, 0, 0,
        ))
        self.assertEqual(
            sparse["opportunity_status"],
            "INSUFFICIENT_MASK_OPPORTUNITY",
        )
        self.assertFalse(sparse["informative"])
        self.assertTrue(sparse["passed"])

    def test_informative_zero_action14_is_a_hard_failure(self):
        gate = r7._gear_progression_gate(self.archive(
            r7.MIN_ACTION14_MASK_OPPORTUNITIES,
            0, 0, 0,
        ))
        self.assertEqual(
            gate["opportunity_status"],
            "INFORMATIVE_MASK_OPPORTUNITY",
        )
        self.assertTrue(gate["informative"])
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["checks"],
            {
                "request_observed_when_informative": False,
                "native_success_observed_when_informative": False,
                "utility_growth_observed_when_informative": False,
            },
        )

    def test_informative_native_growth_passes(self):
        gate = r7._gear_progression_gate(self.archive(
            r7.MIN_ACTION14_MASK_OPPORTUNITIES,
            2, 1, 37,
        ))
        self.assertTrue(gate["passed"])
        self.assertTrue(all(gate["checks"].values()))

    def test_gate_rejects_missing_or_impossible_ledgers(self):
        missing = self.archive(8, 1, 1, 1)
        del missing["agg"]["worker_action14_requests"]
        impossible = (
            self.archive(8, 9, 1, 1),
            self.archive(8, 1, 2, 2),
            self.archive(8, 1, 1, 0),
            self.archive(8, 1, 0, 1),
        )
        with self.assertRaises(r7.CampaignError):
            r7._gear_progression_gate(missing)
        for candidate in impossible:
            with self.subTest(candidate=candidate), self.assertRaises(
                    r7.CampaignError):
                r7._gear_progression_gate(candidate)

    def test_pair_analysis_merges_gear_failure_into_verdict(self):
        statistical = {
            "verdict": {
                "status": "PASS",
                "checks": {"paired": True},
                "failed_checks": [],
            },
        }
        candidate = self.archive(
            r7.MIN_ACTION14_MASK_OPPORTUNITIES,
            0, 0, 0,
        )
        # rev21 诊断消费真实 rows;本测试聚焦 verdict 合并,给最小配对行。
        paired_row = {"seed": 7000, "micro_steps": 100,
                      "farm_worker_wage": 1.0, "died": False}
        baseline = {"baseline": True, "rows": [dict(paired_row)]}
        candidate["rows"] = [dict(paired_row)]
        with mock.patch.object(
                r7.r7_statistics, "analyze_paired_archives",
                return_value=statistical):
            result = r7._analyze_pair(
                baseline, "a" * 64,
                candidate, "b" * 64,
                phase="development",
                death_margin=r7.DEVELOPMENT_DEATH_MARGIN,
            )
        self.assertEqual(result["verdict"]["status"], "FAIL")
        self.assertIn(
            "gear.action14_progression",
            result["verdict"]["failed_checks"],
        )
        self.assertFalse(
            result["verdict"]["checks"]["gear.action14_progression"])
        self.assertEqual(
            result["gear_progression_gate"]["schema_version"],
            r7.GEAR_PROGRESSION_GATE_SCHEMA,
        )


class R7TerminalDeathEvidenceTests(unittest.TestCase):
    def _parse(self, rows, *, recipe="risk32"):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sentinel.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row, allow_nan=True) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            return r7._terminal_death_reward_evidence(
                path, recipe=recipe)

    def test_final_sentinel_proves_observed_reward_conservation(self):
        final = _worker_final_sentinel()
        evidence = self._parse([
            {
                "sentinel": "dry-anchor",
                "step": r7.TARGET_STEPS,
                "mismatch": 0.1,
                "n": 10,
                "final": True,
            },
            final,
        ])
        self.assertEqual(
            evidence["opportunity_status"],
            "TRAINING_FAILURE_OBSERVED",
        )
        self.assertTrue(evidence["reward_mechanism_triggered"])
        self.assertEqual(
            evidence["death_counts"],
            {
                "direct": 2,
                "transition_ff": 3,
                "reset_ff": 1,
                "manual_ff": 0,
            },
        )
        self.assertEqual(
            evidence["death_rewards"][
                "credited_ff_terminal_death_reward"],
            -120.0,
        )
        self.assertRegex(evidence["sentinel_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            evidence["schema_version"],
            "diablogym-r7-terminal-death-reward-evidence/3",
        )
        self.assertEqual(
            evidence["control_path_counts"],
            {"interrupted_resets": 0, "manual_ff_calls": 0},
        )

    def test_no_training_death_is_explicit_not_a_false_trigger(self):
        evidence = self._parse([
            _worker_final_sentinel(
                direct=0, transition=0, reset=2),
        ])
        self.assertEqual(
            evidence["opportunity_status"],
            "NO_TRAINING_FAILURE_OBSERVED",
        )
        self.assertFalse(evidence["reward_mechanism_triggered"])
        self.assertEqual(evidence["death_counts"]["reset_ff"], 2)
        self.assertEqual(
            evidence["death_rewards"][
                "reset_ff_additional_terminal_death_reward"],
            -64.0,
        )
        self.assertEqual(
            evidence["death_rewards"][
                "additional_terminal_death_reward"],
            0.0,
        )

    def test_no_progress_timeout_is_learning_facing_failure_evidence(self):
        evidence = self._parse([
            _worker_final_sentinel(
                direct=0,
                transition=0,
                reset=0,
                direct_timeout=2,
                transition_timeout=1,
                reset_timeout=1,
            ),
        ])
        self.assertEqual(
            evidence["opportunity_status"],
            "TRAINING_FAILURE_OBSERVED",
        )
        self.assertTrue(evidence["reward_mechanism_triggered"])
        self.assertEqual(
            evidence["no_progress_timeout_counts"],
            {
                "direct": 2,
                "transition_ff": 1,
                "reset_ff": 1,
                "manual_ff": 0,
            },
        )
        unit = -(8.0 + 32.0)
        self.assertEqual(
            evidence["no_progress_timeout_rewards"][
                "credited_no_progress_timeout_failure_reward"
            ],
            3 * unit,
        )

    def test_final_sentinel_is_unique_exact_and_finite(self):
        valid = _worker_final_sentinel()
        cases = {}

        missing = dict(valid)
        del missing["direct_terminal_deaths"]
        cases["missing-field"] = [missing]

        extra = {**valid, "unregistered": 1}
        cases["extra-field"] = [extra]

        nonfinite = dict(valid)
        nonfinite["transition_ff_reward"] = float("nan")
        cases["nonfinite"] = [nonfinite]

        wrong_step = dict(valid)
        wrong_step["step"] -= r7.ROLLOUT_QUANTUM
        cases["wrong-step"] = [wrong_step]

        duplicate = [valid, dict(valid)]
        cases["duplicate-final"] = duplicate

        for name, rows in cases.items():
            with self.subTest(case=name), self.assertRaises(
                    r7.CampaignError):
                self._parse(rows)

    def test_final_sentinel_rejects_counter_and_reward_forgery(self):
        valid = _worker_final_sentinel()
        mutations = {
            "negative-count": ("direct_terminal_deaths", -1),
            "bool-count": ("transition_ff_terminal_deaths", True),
            "manual-path": ("manual_ff_terminal_deaths", 1),
            "manual-call": ("manual_ff_calls", 1),
            "interrupted-reset": ("interrupted_resets", 1),
            "direct-additional":
                ("direct_additional_terminal_death_reward", -63.0),
            "transition-additional":
                ("transition_ff_additional_terminal_death_reward", -95.0),
            "reset-additional":
                ("reset_ff_additional_terminal_death_reward", -31.0),
            "total-additional":
                ("additional_terminal_death_reward", -159.0),
            "credited-transition":
                ("credited_ff_terminal_death_reward", -119.0),
            "reason-ledger": ("reasons", {"death": 5}),
            "ff-reason-ledger": ("ff_reasons", {"death": 3}),
            "credit-mode": ("fast_forward_reward_credit_mode", "none"),
            "configured-cost":
                ("configured_additional_terminal_death_cost", 64.0),
            "timeout-reward-without-count":
                ("direct_no_progress_timeout_failure_reward", -40.0),
        }
        for name, (field, value) in mutations.items():
            forged = {**valid, field: value}
            with self.subTest(case=name), self.assertRaises(
                    r7.CampaignError):
                self._parse([forged])

    def test_worker_sentinel_emits_env_bound_reward_configuration(self):
        final = _worker_final_sentinel()
        stats = {
            key: final[key]
            for key in (
                *r7._WORKER_SENTINEL_COUNT_KEYS,
                *r7._WORKER_SENTINEL_REWARD_KEYS,
                "reasons", "ff_reasons",
            )
        }

        class FakeVec:
            def __init__(self, modes):
                self.modes = modes

            def get_attr(self, name):
                if name == "stats":
                    return [stats for _ in self.modes]
                if name == "fast_forward_reward_credit":
                    return list(self.modes)
                if name == "additional_terminal_death_cost":
                    return [32.0 for _ in self.modes]
                raise AssertionError(name)

        with tempfile.TemporaryDirectory() as directory:
            callback = r7.train_ppo.WorkerSentinelCallback(
                pathlib.Path(directory))
            vec = FakeVec([r7.FAST_FORWARD_CREDIT])
            callback.model = types.SimpleNamespace(
                get_env=lambda: vec,
                distill_beta=r7.DISTILL_BETA,
                _last_distill_ce=1.25,
                _last_diverge=0.1,
            )
            callback.num_timesteps = r7.TARGET_STEPS
            callback._emit(final=True)
            emitted = json.loads(
                (pathlib.Path(directory) / "sentinel.jsonl").read_text(
                    encoding="utf-8"))
            self.assertEqual(
                emitted["fast_forward_reward_credit_mode"],
                r7.FAST_FORWARD_CREDIT,
            )
            self.assertEqual(
                emitted["configured_additional_terminal_death_cost"],
                32.0,
            )
            self.assertEqual(emitted["direct_terminal_deaths"], 2)
            self.assertEqual(emitted["interrupted_resets"], 0)
            self.assertEqual(emitted["manual_ff_calls"], 0)
            self.assertEqual(
                emitted[
                    "transition_ff_additional_terminal_death_reward"],
                -96.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            callback = r7.train_ppo.WorkerSentinelCallback(
                pathlib.Path(directory))
            vec = FakeVec([r7.FAST_FORWARD_CREDIT, "none"])
            callback.model = types.SimpleNamespace(
                get_env=lambda: vec,
                distill_beta=None,
                _last_distill_ce=None,
                _last_diverge=None,
            )
            callback.num_timesteps = r7.TARGET_STEPS
            with self.assertRaisesRegex(ValueError, "credit 漂移"):
                callback._emit(final=True)


class R7WorkerOnPolicyPgEvidenceTests(unittest.TestCase):
    def _summarize(self, checkpoint):
        with mock.patch.object(
                r7, "validate_worker_onpolicy_pg_receipt",
                return_value=True):
            return r7._worker_onpolicy_pg_evidence(
                checkpoint,
                "pure-test",
                expected_additional_terminal_death_cost=32.0,
            )

    def test_compact_evidence_aggregates_timeout_split_exactly(self):
        checkpoint = _formal_pg_checkpoint((
            (2, -24.0, -64.0, -88.0),
            (1, -16.0, -32.0, -48.0),
        ))
        evidence = self._summarize(checkpoint)
        self.assertEqual(
            evidence["schema"], r7.WORKER_ONPOLICY_PG_AUDIT_SCHEMA)
        self.assertEqual(
            evidence["configured_additional_terminal_death_cost"], 32.0)
        self.assertEqual(evidence["no_progress_timeout_samples"], 3)
        self.assertEqual(
            evidence["no_progress_timeout_base_failure_reward_sum"], -40.0)
        self.assertEqual(
            evidence[
                "no_progress_timeout_additional_failure_reward_sum"],
            -96.0,
        )
        self.assertEqual(
            evidence["no_progress_timeout_failure_reward_sum"], -136.0)
        self.assertEqual(
            evidence["optimizer_delta_actor_l2_qualifying_min"],
            2.0 ** 0.5,
        )
        self.assertEqual(
            evidence["optimizer_delta_root_l2_qualifying_min"], 1.0)
        self.assertEqual(
            evidence["optimizer_delta_context_l2_qualifying_min"], 1.0)
        for partition in ("actor", "root", "context"):
            self.assertEqual(
                evidence[
                    f"optimizer_delta_{partition}"
                    "_on_combat_reward_descent_cosine_qualifying_min"
                ],
                1.0,
            )

    def test_compact_evidence_rejects_timeout_cost_or_total_forgery(self):
        mutations = {}
        wrong_cost = _formal_pg_checkpoint(((1, -8.0, -32.0, -40.0),))
        wrong_cost_receipt = wrong_cost[
            "_worker_onpolicy_pg_rollout_receipts"][0]
        wrong_cost_receipt[
            "no_progress_timeout_additional_failure_reward_sum"] = -31.0
        wrong_cost_receipt["no_progress_timeout_failure_reward_sum"] = -39.0
        mutations["configured-cost"] = wrong_cost

        wrong_total = _formal_pg_checkpoint(((1, -8.0, -32.0, -40.0),))
        wrong_total["_worker_onpolicy_pg_rollout_receipts"][0][
            "no_progress_timeout_failure_reward_sum"] = -39.0
        mutations["partition-total"] = wrong_total

        for name, checkpoint in mutations.items():
            with self.subTest(case=name), self.assertRaisesRegex(
                    r7.CampaignError, "timeout 分账"):
                self._summarize(checkpoint)


class StableIoTests(unittest.TestCase):
    def test_stable_read_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            target = root / "target.bin"
            target.write_bytes(b"payload")
            link = root / "link.bin"
            link.symlink_to(target)
            with self.assertRaisesRegex(r7.CampaignError, "符号链接"):
                r7._stable_read(link)

    def test_exclusive_json_round_trip_and_staging_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            record_path = root / "nested" / "record.json"
            r7._write_json_exclusive(record_path, {"value": 7})
            self.assertEqual(r7._stable_json(record_path), {"value": 7})

            source = root / "source.zip"
            destination = root / "stage" / "worker.zip"
            source.write_bytes(b"frozen-worker")
            digest = hashlib.sha256(b"frozen-worker").hexdigest()
            self.assertEqual(
                r7._stage_eval_file(
                    source, destination, expected_sha256=digest),
                digest,
            )
            self.assertEqual(r7._stable_read(destination), b"frozen-worker")
            self.assertEqual(destination.stat().st_mode & 0o222, 0)

            source.write_bytes(b"changed")
            with self.assertRaisesRegex(r7.CampaignError, "源 SHA 漂移"):
                r7._stage_eval_file(
                    source, destination, expected_sha256=digest)

    def test_v28_eval_staging_rebinds_fixed_sha_at_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            worker = root / "v28.zip"
            manager = root / "manager.npz"
            worker.write_bytes(b"foreign-worker")
            manager.write_bytes(b"manager")
            expected_v28_sha = hashlib.sha256(
                b"registered-v28").hexdigest()
            manager_sha = hashlib.sha256(b"manager").hexdigest()
            with (
                mock.patch.object(r7, "V28_ZIP", worker),
                mock.patch.object(
                    r7, "V28_SHA256", expected_v28_sha),
                mock.patch.object(r7, "M29_NPZ", manager),
                mock.patch.object(r7, "M29_SHA256", manager_sha),
                mock.patch.object(
                    r7, "EVAL_INPUT_DIR", root / "eval-inputs"),
                mock.patch.object(
                    r7, "_frozen_inputs_identity", return_value={}),
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "V28 baseline SHA 漂移"):
                    r7._prepare_eval_launch(
                        worker, (1, 2), "baseline-toctou")

    def test_campaign_and_registry_locks_reject_unsafe_inodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            control = root / "control"
            control.mkdir()
            target = root / "target.lock"
            target.write_bytes(b"")
            campaign_link = control / "campaign.lock"
            campaign_link.symlink_to(target)
            with (
                mock.patch.object(r7, "CONTROL_DIR", control),
                mock.patch.object(r7, "LOCK_PATH", campaign_link),
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "不可安全打开"):
                    with r7._campaign_lock():
                        self.fail("symlink lock must not be acquired")

            registry = root / "registry"
            registry.mkdir()
            registry_lock = registry / ".registry.lock"
            registry_lock.write_bytes(b"")
            hardlink = root / "registry-lock-hardlink"
            os.link(registry_lock, hardlink)
            with (
                mock.patch.object(r7, "FINAL_REGISTRY_DIR", registry),
                mock.patch.object(r7, "FINAL_REGISTRY_LOCK", registry_lock),
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "单链接普通文件"):
                    with r7._final_registry_lock():
                        self.fail("multi-link lock must not be acquired")


class R7SelectionTests(unittest.TestCase):
    def _all(self, function) -> dict[str, dict]:
        return {
            f"{pool}:{recipe}:{seed}": _analysis(
                function(pool, recipe, seed),
                cost=r7.RECIPES[recipe][
                    "additional_terminal_death_cost"])
            for pool in r7.DEV_POOLS
            for recipe in r7.RECIPE_PREFERENCE
            for seed in r7.DEVELOPMENT_TRAIN_SEEDS
        }

    def _freeze(self, analyses):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "decision.json"
            hashes = {
                key: r7._canonical_sha256(value)
                for key, value in analyses.items()
            }
            with (
                mock.patch.object(r7, "DEVELOPMENT_DECISION_PATH", path),
                mock.patch.object(
                    r7, "_analysis_file_sha256s", return_value=hashes),
            ):
                result = r7._freeze_development_decision(analyses)
                self.assertEqual(r7._stable_json(path), result)
                return result

    def test_never_selects_a_development_model_or_best_seed(self):
        decision = self._freeze(
            self._all(lambda _pool, _recipe, _seed: True))
        self.assertEqual(decision["selected_recipe"], "risk32")
        self.assertNotIn("selected_seed", decision)
        self.assertNotIn("selected_model", decision)

    def test_recipe_requires_two_same_seed_cross_pool_replications(self):
        seeds = r7.DEVELOPMENT_TRAIN_SEEDS

        def passed(pool, recipe, seed):
            if recipe == "risk32":
                return (
                    (pool == "dev-a" and seed in {seeds[0], seeds[1]})
                    or (pool == "dev-b" and seed in {seeds[0], seeds[2]})
                )
            return seed in {seeds[0], seeds[1]}

        decision = self._freeze(self._all(passed))
        self.assertFalse(decision["eligibility"]["risk32"]["eligible"])
        self.assertTrue(decision["eligibility"]["risk64"]["eligible"])
        self.assertEqual(decision["selected_recipe"], "risk64")

    def test_selection_exposes_no_death_opportunity_without_false_trigger(self):
        analyses = self._all(lambda _pool, _recipe, _seed: True)
        seed = r7.DEVELOPMENT_TRAIN_SEEDS[0]
        no_opportunity = _training_death_evidence(observed=False)
        for pool in r7.DEV_POOLS:
            analyses[f"{pool}:risk32:{seed}"][
                "training_death_reward_evidence"] = no_opportunity
        decision = self._freeze(analyses)
        self.assertEqual(
            decision["training_death_reward_evidence"][
                f"risk32:{seed}"]["opportunity_status"],
            "NO_TRAINING_FAILURE_OBSERVED",
        )
        self.assertFalse(
            decision["training_death_reward_evidence"][
                f"risk32:{seed}"]["reward_mechanism_triggered"],
        )
        self.assertEqual(
            decision["eligibility"]["risk32"][
                "training_failure_observed_replications"],
            2,
        )
        self.assertEqual(
            decision["eligibility"]["risk32"][
                "training_failure_opportunity"][str(seed)],
            "NO_TRAINING_FAILURE_OBSERVED",
        )

    def test_selected_recipe_rejects_forged_decision_or_state_sha(self):
        analyses = self._all(lambda _pool, _recipe, _seed: True)
        expected = r7._development_decision_document(analyses)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "decision.json"
            r7._write_json_exclusive(path, expected)
            state = {
                "phases": {
                    "eval_development": {
                        "status": "complete",
                        "completed": sorted(r7._development_analysis_keys()),
                        "decision_sha256": r7._sha256(path),
                        "selected_recipe": "risk32",
                    },
                },
            }
            absent = pathlib.Path(directory) / "no-adoption.json"
            with (
                mock.patch.object(r7, "DEVELOPMENT_DECISION_PATH", path),
                mock.patch.object(r7, "AMENDMENT5_PATH", absent),
                mock.patch.object(r7, "AMENDMENT6_PATH", absent),
                mock.patch.object(
                    r7, "_recompute_development_analyses",
                    return_value=analyses),
                mock.patch.object(
                    r7, "_analysis_file_sha256s",
                    return_value=expected["analysis_sha256s"]),
            ):
                self.assertEqual(
                    r7._validate_development_decision(state), expected)
                forged = {**expected, "selected_recipe": "risk64"}
                path.write_text(json.dumps(forged), encoding="utf-8")
                with self.assertRaises(r7.CampaignError):
                    r7._validate_development_decision(state)


class R7OneShotTests(unittest.TestCase):
    def test_training_fired_marker_blocks_retry_without_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            marker = root / "control" / "training-fired.json"
            receipt = root / "run" / "receipt.json"
            with (
                mock.patch.object(
                    r7, "_training_fired_path", return_value=marker),
                mock.patch.object(
                    r7, "_training_receipt_path", return_value=receipt),
                mock.patch.object(r7, "_bc_identity", return_value={"bc": 1}),
                mock.patch.object(
                    r7, "_training_fired_core",
                    return_value={"schema_version": "unit", "attempt": 1}),
                mock.patch.object(
                    r7, "_invoke",
                    side_effect=CampaignInterrupted("interrupted")),
            ):
                with self.assertRaises(CampaignInterrupted):
                    r7._run_training_once("risk32", 42, "development")
                self.assertTrue(marker.exists())
                with self.assertRaisesRegex(r7.CampaignError, "禁止重发"):
                    r7._run_training_once("risk32", 42, "development")

    def test_completed_training_is_adopted_without_refiring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            marker = root / "control" / "training-fired.json"
            receipt = root / "run" / "receipt.json"
            marker.parent.mkdir(parents=True)
            marker.write_text("{}", encoding="utf-8")
            receipt.parent.mkdir(parents=True)
            expected = {"model_sha256": "a" * 64}
            with (
                mock.patch.object(
                    r7, "_training_fired_path", return_value=marker),
                mock.patch.object(
                    r7, "_training_receipt_path", return_value=receipt),
                mock.patch.object(r7, "_bc_identity", return_value={"bc": 1}),
                mock.patch.object(
                    r7, "_capture_training_artifact",
                    return_value=expected) as capture,
                mock.patch.object(r7, "_invoke") as invoke,
            ):
                self.assertEqual(
                    r7._run_training_once(
                        "risk32", 42, "development"),
                    expected,
                )
            capture.assert_called_once_with(
                "risk32", 42, "development")
            invoke.assert_not_called()

    def test_eval_fired_marker_blocks_retry_without_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            marker = root / "eval-fired.json"
            output = root / "archive.json"
            attestation = root / "attestation.json"
            launch = {
                "campaign_recipe_sha256": "a" * 64,
                "tag": "unit-tag",
                "seeds": [1, 2],
                "source_worker_path": "/worker.zip",
                "source_worker_sha256": "b" * 64,
                "staged_worker_path": "/stage/worker.zip",
                "worker_sha256": "b" * 64,
                "staged_manager_path": "/stage/manager.npz",
                "manager_sha256": "c" * 64,
                "training_receipt_sha256": None,
                "final_bind_sha256": None,
                "implementation": {"sha": "d" * 64},
                "command": ["eval"],
                "attempt": 1,
            }
            with (
                mock.patch.object(
                    r7, "_prepare_eval_launch", return_value=launch),
                mock.patch.object(
                    r7, "_eval_fired_path", return_value=marker),
                mock.patch.object(r7, "_eval_path", return_value=output),
                mock.patch.object(
                    r7, "_eval_attestation_path",
                    return_value=attestation),
                mock.patch.object(
                    r7, "_invoke",
                    side_effect=CampaignInterrupted("interrupted")),
            ):
                with self.assertRaises(CampaignInterrupted):
                    r7._run_eval_once(
                        pathlib.Path("/worker.zip"), (1, 2), "unit-tag")
                self.assertTrue(marker.exists())
                with self.assertRaisesRegex(r7.CampaignError, "禁止.*重试"):
                    r7._run_eval_once(
                        pathlib.Path("/worker.zip"), (1, 2), "unit-tag")

    def test_completed_eval_is_attested_without_refiring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            worker = root / "worker.zip"
            worker.write_bytes(b"worker")
            marker = root / "eval-fired.json"
            marker.write_text("{}", encoding="utf-8")
            output = root / "archive.json"
            output.write_bytes(b"archive")
            attestation_path = root / "attestation.json"
            archive_sha = hashlib.sha256(b"archive").hexdigest()
            worker_sha = hashlib.sha256(b"worker").hexdigest()
            launch = {
                "tag": "unit-tag",
                "seeds": [1, 2],
                "source_worker_path": str(worker),
                "source_worker_sha256": worker_sha,
                "worker_sha256": worker_sha,
                "manager_sha256": "c" * 64,
                "training_receipt_sha256": None,
                "final_bind_sha256": None,
                "implementation": {"sha": "d" * 64},
                "command": ["eval"],
            }
            attestation = {"schema_version": "unit-attestation"}
            with (
                mock.patch.object(
                    r7, "_prepare_eval_launch",
                    return_value=launch) as prepare,
                mock.patch.object(
                    r7, "_eval_fired_path", return_value=marker),
                mock.patch.object(r7, "_eval_path", return_value=output),
                mock.patch.object(
                    r7, "_eval_attestation_path",
                    return_value=attestation_path),
                mock.patch.object(
                    r7, "_validate_eval_fired") as validate_fired,
                mock.patch.object(
                    r7, "_implementation_identity",
                    return_value=launch["implementation"]),
                mock.patch.object(
                    r7, "_validate_eval_archive",
                    return_value=({"complete": True}, archive_sha)),
                mock.patch.object(
                    r7, "_eval_attestation",
                    return_value=attestation),
                mock.patch.object(r7, "_invoke") as invoke,
            ):
                document, digest = r7._run_eval_once(
                    worker, (1, 2), "unit-tag")
            self.assertEqual(document, {"complete": True})
            self.assertEqual(digest, archive_sha)
            self.assertEqual(
                json.loads(attestation_path.read_text(encoding="utf-8")),
                attestation,
            )
            self.assertTrue(
                prepare.call_args.kwargs["require_existing_staging"])
            validate_fired.assert_called_once_with("unit-tag", launch)
            invoke.assert_not_called()

    def test_eval_archive_must_equal_frozen_staging_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            worker = root / "worker.zip"
            manager = root / "manager.npz"
            archive = root / "archive.json"
            worker.write_bytes(b"worker")
            manager.write_bytes(b"manager")
            archive.write_bytes(b"{}")
            launch = {
                "staged_worker_path": str(worker),
                "staged_manager_path": str(manager),
                "worker_sha256": hashlib.sha256(b"worker").hexdigest(),
                "manager_sha256": hashlib.sha256(b"manager").hexdigest(),
            }
            document = {
                "meta": {
                    "worker": {
                        "path": str(worker.resolve()),
                        "sha256": "0" * 64,
                    },
                    "manager": {
                        "path": str(manager.resolve()),
                        "sha256": launch["manager_sha256"],
                    },
                },
                "agg": {
                    "worker_calls": 0,
                    "worker_action14_mask_opportunities": 0,
                    "worker_action14_requests": 0,
                    "worker_action14_native_successes": 0,
                    "worker_action14_gear_utility_delta": 0,
                },
            }
            with (
                mock.patch.object(
                    r7.eval_contract, "freeze_eval_identity",
                    return_value={"snapshot": 1}),
                mock.patch.object(
                    r7.eval_contract, "expected_eval_identity",
                    return_value={}),
                mock.patch.object(
                    r7.eval_contract, "validate_eval_archive",
                    return_value=document),
                mock.patch.object(r7, "_eval_path", return_value=archive),
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "worker/manager"):
                    r7._validate_eval_archive(launch, (1, 2), "unit")

    def test_eval_archive_document_and_sha_share_one_stable_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            worker = root / "worker.zip"
            manager = root / "manager.npz"
            archive = root / "archive.json"
            worker.write_bytes(b"worker")
            manager.write_bytes(b"manager")
            original = b"{}"
            archive.write_bytes(original)
            launch = {
                "staged_worker_path": str(worker),
                "staged_manager_path": str(manager),
                "worker_sha256": hashlib.sha256(b"worker").hexdigest(),
                "manager_sha256": hashlib.sha256(b"manager").hexdigest(),
            }
            document = {
                "meta": {
                    "worker": {
                        "path": str(worker.resolve()),
                        "sha256": launch["worker_sha256"],
                    },
                    "manager": {
                        "path": str(manager.resolve()),
                        "sha256": launch["manager_sha256"],
                    },
                },
                "agg": {
                    "worker_calls": 0,
                    "worker_action14_mask_opportunities": 0,
                    "worker_action14_requests": 0,
                    "worker_action14_native_successes": 0,
                    "worker_action14_gear_utility_delta": 0,
                },
            }

            with (
                mock.patch.object(
                    r7.eval_contract, "freeze_eval_identity",
                    return_value={"snapshot": 1}),
                mock.patch.object(
                    r7.eval_contract, "expected_eval_identity",
                    return_value={}),
                mock.patch.object(
                    r7.eval_contract, "validate_eval_archive",
                    return_value=document),
                mock.patch.object(r7, "_eval_path", return_value=archive),
            ):
                frozen, digest = r7._validate_eval_archive(
                    launch, (1, 2), "unit")
            self.assertEqual(frozen, document)
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())

    def test_eval_archive_rejects_replacement_during_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            worker = root / "worker.zip"
            manager = root / "manager.npz"
            archive = root / "archive.json"
            worker.write_bytes(b"worker")
            manager.write_bytes(b"manager")
            archive.write_bytes(b"{}")
            launch = {
                "staged_worker_path": str(worker),
                "staged_manager_path": str(manager),
                "worker_sha256": hashlib.sha256(b"worker").hexdigest(),
                "manager_sha256": hashlib.sha256(b"manager").hexdigest(),
            }
            document = {
                "meta": {
                    "worker": {
                        "path": str(worker.resolve()),
                        "sha256": launch["worker_sha256"],
                    },
                    "manager": {
                        "path": str(manager.resolve()),
                        "sha256": launch["manager_sha256"],
                    },
                },
                "agg": {
                    "worker_calls": 0,
                    "worker_action14_mask_opportunities": 0,
                    "worker_action14_requests": 0,
                    "worker_action14_native_successes": 0,
                    "worker_action14_gear_utility_delta": 0,
                },
            }

            def validate_and_replace(_document, **_expected):
                archive.write_bytes(b"replacement")
                return document

            with (
                mock.patch.object(
                    r7.eval_contract, "freeze_eval_identity",
                    return_value={"snapshot": 1}),
                mock.patch.object(
                    r7.eval_contract, "expected_eval_identity",
                    return_value={}),
                mock.patch.object(
                    r7.eval_contract, "validate_eval_archive",
                    side_effect=validate_and_replace),
                mock.patch.object(r7, "_eval_path", return_value=archive),
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "路径漂移"):
                    r7._validate_eval_archive(
                        launch, (1, 2), "unit")

    def test_eval_archive_rejects_same_bytes_from_wrong_staging_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            worker = root / "worker.zip"
            manager = root / "manager.npz"
            archive = root / "archive.json"
            worker.write_bytes(b"worker")
            manager.write_bytes(b"manager")
            archive.write_bytes(b"{}")
            launch = {
                "staged_worker_path": str(worker),
                "staged_manager_path": str(manager),
                "worker_sha256": hashlib.sha256(b"worker").hexdigest(),
                "manager_sha256": hashlib.sha256(b"manager").hexdigest(),
            }
            document = {
                "meta": {
                    "worker": {
                        "path": "/wrong/worker.zip",
                        "sha256": launch["worker_sha256"],
                    },
                    "manager": {
                        "path": str(manager.resolve()),
                        "sha256": launch["manager_sha256"],
                    },
                },
            }
            with (
                mock.patch.object(
                    r7.eval_contract, "freeze_eval_identity",
                    return_value={"snapshot": 1}),
                mock.patch.object(
                    r7.eval_contract, "expected_eval_identity",
                    return_value={}),
                mock.patch.object(
                    r7.eval_contract, "validate_eval_archive",
                    return_value=document),
                mock.patch.object(r7, "_eval_path", return_value=archive),
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "worker/manager"):
                    r7._validate_eval_archive(
                        launch, (1, 2), "unit")

    def test_eval_staging_rejects_extra_sidecar_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            inputs = root / "inputs"
            worker = root / "source-worker.zip"
            manager = root / "source-manager.npz"
            worker.write_bytes(b"worker")
            manager.write_bytes(b"manager")
            manager_sha = hashlib.sha256(b"manager").hexdigest()
            with (
                mock.patch.object(r7, "EVAL_INPUT_DIR", inputs),
                mock.patch.object(r7, "M29_NPZ", manager),
                mock.patch.object(r7, "M29_SHA256", manager_sha),
                mock.patch.object(
                    r7, "_frozen_inputs_identity", return_value={}),
                mock.patch.object(
                    r7, "_implementation_identity",
                    return_value={"implementation": "unit"}),
            ):
                r7._prepare_eval_launch(
                    worker, (1, 2), "unit-tag")
                with mock.patch.object(
                        r7, "_stage_eval_file") as stage_file:
                    r7._prepare_eval_launch(
                        worker, (1, 2), "unit-tag",
                        require_existing_staging=True)
                    stage_file.assert_not_called()
                (inputs / "unit-tag" / "bc_aux_behavior_receipt.json").write_text(
                    "{}", encoding="utf-8")
                with self.assertRaisesRegex(
                        r7.CampaignError, "成员集合"):
                    r7._prepare_eval_launch(
                        worker, (1, 2), "unit-tag",
                        require_existing_staging=True)


class FinalRegistryTests(unittest.TestCase):
    def test_global_registry_blocks_second_open_and_allows_exact_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            registry = root / "registry"
            lock = registry / ".registry.lock"
            control = root / "control"
            opened = control / "final-opened.json"
            with (
                mock.patch.object(r7, "FINAL_REGISTRY_DIR", registry),
                mock.patch.object(r7, "FINAL_REGISTRY_LOCK", lock),
                mock.patch.object(r7, "CONTROL_DIR", control),
                mock.patch.object(r7, "FINAL_OPENED_PATH", opened),
                mock.patch.object(r7, "FINAL_POOL", (10, 11)),
            ):
                first = r7._register_final_pool("a" * 64, continuing=False)
                self.assertEqual(first["seeds"], [10, 11])
                r7._write_json_exclusive(opened, {"bind": "a" * 64})
                continued = r7._register_final_pool(
                    "a" * 64, continuing=True)
                self.assertEqual(continued, first)
                with self.assertRaisesRegex(
                        r7.CampaignError, "全局 registry"):
                    r7._register_final_pool("a" * 64, continuing=False)

    def test_global_registry_rejects_partial_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            registry = root / "registry"
            lock = registry / ".registry.lock"
            control = root / "control"
            historical_spec = {
                "pool_name": "r7-official-final",
                "seeds": [11, 12],
            }
            pool_sha = r7._canonical_sha256(historical_spec)
            historical = {
                "schema_version": r7.FINAL_REGISTRY_SCHEMA,
                **historical_spec,
                "pool_sha256": pool_sha,
                "campaign_recipe_sha256": "c" * 64,
                "bind_sha256": "b" * 64,
                "control_path": "/old/control",
                "opened_at_ns": 1,
                "consumption_stage": "before_baseline_evaluation",
            }
            r7._write_json_exclusive(registry / f"{pool_sha}.json", historical)
            with (
                mock.patch.object(r7, "FINAL_REGISTRY_DIR", registry),
                mock.patch.object(r7, "FINAL_REGISTRY_LOCK", lock),
                mock.patch.object(r7, "CONTROL_DIR", control),
                mock.patch.object(r7, "FINAL_POOL", (10, 11)),
            ):
                with self.assertRaisesRegex(r7.CampaignError, "重叠"):
                    r7._register_final_pool("a" * 64, continuing=False)

    def test_residue_scan_catches_corrupt_void_with_final_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = pathlib.Path(directory)
            (eval_dir / "prior-final.json.void").write_bytes(
                b'{"episode_seed":2120000')
            (eval_dir / "unrelated.json").write_text(
                '{"meta":{"protocol":{"seeds":[1,2]}}}',
                encoding="utf-8",
            )
            with (
                mock.patch.object(r7, "EVAL_DIR", eval_dir),
                mock.patch.object(r7, "FINAL_POOL", (2_120_000, 2_120_001)),
            ):
                residue = r7._find_final_eval_residue()
            self.assertEqual(residue, [
                str(eval_dir / "prior-final.json.void")])

    def test_final_persistent_eval_lock_is_allowed_only_when_registered(self):
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = pathlib.Path(directory)
            lock = eval_dir / ".official-r7-final-baseline-2120000.json.lock"
            lock.write_text("pid=123\n", encoding="ascii")
            with (
                mock.patch.object(r7, "EVAL_DIR", eval_dir),
                mock.patch.object(r7, "FINAL_POOL", (2_120_000, 2_120_001)),
            ):
                self.assertEqual(
                    r7._find_final_eval_residue(), [str(lock)])
                self.assertEqual(
                    r7._find_final_eval_residue((lock,)), [])

    def test_registry_sha_and_pool_sha_are_frozen_into_final_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            registry = root / "registry"
            lock = registry / ".registry.lock"
            control = root / "control"
            opened_path = control / "final-opened.json"
            with (
                mock.patch.object(r7, "FINAL_REGISTRY_DIR", registry),
                mock.patch.object(r7, "FINAL_REGISTRY_LOCK", lock),
                mock.patch.object(r7, "CONTROL_DIR", control),
                mock.patch.object(r7, "FINAL_OPENED_PATH", opened_path),
                mock.patch.object(r7, "FINAL_POOL", (10, 11)),
            ):
                bind_core = {
                    "schema_version": r7.FINAL_OPENED_SCHEMA,
                    "final_pool_sha256": r7._final_pool_sha256(),
                    "candidate_sha256": "c" * 64,
                }
                r7._register_final_pool(
                    r7._canonical_sha256(bind_core), continuing=False)
                opened = r7._final_opened_document(bind_core)
                r7._write_json_exclusive(opened_path, opened)
                actual, evidence = r7._validate_final_opened_registry(
                    bind_core)
                self.assertEqual(actual, opened)
                self.assertEqual(
                    actual["final_pool_sha256"],
                    r7._final_pool_sha256(),
                )
                self.assertEqual(
                    actual["final_registry_record_sha256"],
                    r7._sha256(r7._final_registry_path()),
                )
                self.assertEqual(
                    evidence["final_registry_record_sha256"],
                    actual["final_registry_record_sha256"],
                )

                registry_record = r7._stable_json(
                    r7._final_registry_path())
                registry_record["opened_at_ns"] += 1
                r7._write_json_atomic(
                    r7._final_registry_path(), registry_record)
                with self.assertRaisesRegex(
                        r7.CampaignError, "evidence|证据|registry"):
                    r7._validate_final_opened_registry(bind_core)


class PublicationTests(unittest.TestCase):
    def test_fail_never_creates_published_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            published = pathlib.Path(directory) / "published"
            with mock.patch.object(r7, "PUBLISHED_DIR", published):
                with self.assertRaisesRegex(r7.CampaignError, "只有 final PASS"):
                    r7._promote_passed_candidate(
                        {}, pathlib.Path("/unused"),
                        {"verdict": {"status": "FAIL"}},
                        baseline_sha="a" * 64,
                        candidate_sha="b" * 64,
                    )
                self.assertFalse(published.exists())

    def test_publication_receipt_embeds_registry_and_pool_identity(self):
        evidence = {
            "bind_sha256": "a" * 64,
            "final_pool_sha256": "b" * 64,
            "final_registry_path": "/registry/pool.json",
            "final_registry_record_sha256": "c" * 64,
            "final_fired_sha256": "d" * 64,
            "analysis_sha256": "e" * 64,
            "baseline_archive_sha256": "f" * 64,
            "candidate_archive_sha256": "1" * 64,
            "baseline_eval_fired_sha256": "2" * 64,
            "candidate_eval_fired_sha256": "3" * 64,
            "baseline_attestation_sha256": "4" * 64,
            "candidate_attestation_sha256": "5" * 64,
        }
        bind = {
            "recipe": "risk32",
            "production_seed": r7.PRODUCTION_TRAIN_SEED,
            "candidate_sha256": "6" * 64,
            "final_pool_sha256": evidence["final_pool_sha256"],
            "final_registry_path": evidence["final_registry_path"],
            "final_registry_record_sha256":
                evidence["final_registry_record_sha256"],
        }
        receipt = {"receipt_sha256": "7" * 64}
        analysis = {"verdict": {"status": "PASS"}}
        publication = r7._publication_document(
            bind, receipt, analysis, evidence=evidence)
        self.assertEqual(
            publication["final_pool_sha256"],
            evidence["final_pool_sha256"],
        )
        self.assertEqual(
            publication["final_registry_record_sha256"],
            evidence["final_registry_record_sha256"],
        )
        self.assertEqual(publication["final_bind"], bind)

    def test_pass_atomically_promotes_candidate_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidate = root / "candidate.zip"
            candidate.write_bytes(b"candidate")
            candidate_sha = hashlib.sha256(b"candidate").hexdigest()
            published = root / "published"
            published_model = published / "model_final.zip"
            published_receipt = published / "receipt.json"
            receipt = {"receipt_sha256": "b" * 64}
            bind = {
                "recipe": "risk32",
                "candidate_sha256": candidate_sha,
                "candidate_receipt_sha256": receipt["receipt_sha256"],
            }
            publication = {
                "schema_version": r7.PUBLICATION_RECEIPT_SCHEMA,
                "publication_status": "PUBLISHED",
                "published_model_sha256": candidate_sha,
            }
            publication_on_disk = dict(publication)
            evidence_snapshot = {
                "identity": {
                    "baseline_archive_sha256": "a" * 64,
                    "candidate_archive_sha256": "c" * 64,
                },
                "analysis": {"verdict": {"status": "PASS"}},
                "candidate_receipt": receipt,
                "candidate_payload": b"candidate",
                "path_sha256s": {},
            }
            with (
                mock.patch.object(r7, "PUBLISHED_DIR", published),
                mock.patch.object(
                    r7, "PUBLISHED_MODEL_PATH", published_model),
                mock.patch.object(
                    r7, "PUBLISHED_RECEIPT_PATH", published_receipt),
                mock.patch.object(
                    r7, "_capture_final_evidence",
                    return_value=evidence_snapshot),
                mock.patch.object(
                    r7, "_publication_document",
                    return_value=publication),
            ):
                result = r7._promote_passed_candidate(
                    bind, candidate,
                    {"verdict": {"status": "PASS"}},
                    baseline_sha="a" * 64,
                    candidate_sha="c" * 64,
                )
                result_receipt_sha256 = result["receipt_sha256"]
                publication.pop("receipt_sha256")
                (published / "unexpected.txt").write_text(
                    "residue", encoding="utf-8")
                with self.assertRaisesRegex(
                        r7.CampaignError, "成员集合"):
                    r7._promote_passed_candidate(
                        bind, candidate,
                        {"verdict": {"status": "PASS"}},
                        baseline_sha="a" * 64,
                        candidate_sha="c" * 64,
                    )
            self.assertEqual(published_model.read_bytes(), b"candidate")
            self.assertEqual(
                json.loads(published_receipt.read_text()),
                publication_on_disk,
            )
            self.assertEqual(
                result_receipt_sha256,
                hashlib.sha256(published_receipt.read_bytes()).hexdigest(),
            )

    def test_promotion_rejects_evidence_drift_before_atomic_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            candidate = root / "candidate.zip"
            candidate.write_bytes(b"candidate")
            candidate_sha = hashlib.sha256(b"candidate").hexdigest()
            published = root / "published"
            receipt = {"receipt_sha256": "b" * 64}
            bind = {
                "recipe": "risk32",
                "candidate_sha256": candidate_sha,
                "candidate_receipt_sha256": receipt["receipt_sha256"],
            }
            analysis = {"verdict": {"status": "PASS"}}
            frozen = {
                "identity": {
                    "baseline_archive_sha256": "a" * 64,
                    "candidate_archive_sha256": "c" * 64,
                    "final_registry_record_sha256": "d" * 64,
                    "final_pool_sha256": "e" * 64,
                },
                "analysis": analysis,
                "candidate_receipt": receipt,
                "candidate_payload": b"candidate",
                "path_sha256s": {},
            }
            drifted = {
                **frozen,
                "identity": {
                    **frozen["identity"],
                    "final_registry_record_sha256": "f" * 64,
                },
            }
            publication = {
                "schema_version": r7.PUBLICATION_RECEIPT_SCHEMA,
                "publication_status": "PUBLISHED",
                "published_model_sha256": candidate_sha,
            }
            with (
                mock.patch.object(r7, "PUBLISHED_DIR", published),
                mock.patch.object(
                    r7, "PUBLISHED_MODEL_PATH",
                    published / "model_final.zip"),
                mock.patch.object(
                    r7, "PUBLISHED_RECEIPT_PATH",
                    published / "receipt.json"),
                mock.patch.object(
                    r7, "_capture_final_evidence",
                    side_effect=(frozen, drifted)),
                mock.patch.object(
                    r7, "_publication_document",
                    return_value=publication),
            ):
                with self.assertRaisesRegex(
                        r7.CampaignError, "commit 前"):
                    r7._promote_passed_candidate(
                        bind, candidate, analysis,
                        baseline_sha="a" * 64,
                        candidate_sha="c" * 64,
                    )
            self.assertFalse(published.exists())

    def test_terminal_status_runs_read_only_publication_audit(self):
        state = {"terminal_status": "PASS", "phases": {}}
        audit = {
            "publication_status": "PUBLISHED",
            "published_model_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            with (
                mock.patch.object(r7, "_require_seed_discipline"),
                mock.patch.object(r7, "_load_state", return_value=state),
                mock.patch.object(
                    r7, "_implementation_identity",
                    return_value={"implementation_sha256": "b" * 64}),
                mock.patch.object(
                    r7, "_bc_identity", return_value={"policy_sha256": "c" * 64}),
                mock.patch.object(
                    r7, "_audit_terminal_publication",
                    return_value=audit) as publication_audit,
                mock.patch.object(
                    r7, "DEVELOPMENT_DECISION_PATH", root / "decision.json"),
                mock.patch.object(
                    r7, "FINAL_ANALYSIS_PATH", root / "analysis.json"),
                mock.patch("builtins.print") as output,
            ):
                r7.command_status()
            publication_audit.assert_called_once_with(state)
            health = json.loads(output.call_args.args[0])
            self.assertEqual(
                health["publication_audit"],
                {"status": "PASS", **audit},
            )


class CampaignInterrupted(RuntimeError):
    pass




class Rev21DiagnosticsTests(unittest.TestCase):
    """rev21 每步效率记分肢与存活分解(只记不裁)的纯函数回归。"""

    @staticmethod
    def _archive(rows):
        return {"rows": rows}

    def test_rate_report_and_decomposition_identity(self):
        baseline = self._archive([
            {"seed": 7000, "micro_steps": 1000,
             "farm_worker_wage": 30.0, "died": True},
            {"seed": 7001, "micro_steps": 2000,
             "farm_worker_wage": 50.0, "died": False},
            {"seed": 7002, "micro_steps": 500,
             "farm_worker_wage": 10.0, "died": True},
        ])
        candidate = self._archive([
            {"seed": 7000, "micro_steps": 2000,
             "farm_worker_wage": 80.0, "died": False},
            {"seed": 7001, "micro_steps": 1000,
             "farm_worker_wage": 20.0, "died": True},
            {"seed": 7002, "micro_steps": 500,
             "farm_worker_wage": 12.0, "died": True},
        ])
        diag = r7._rev21_diagnostics(baseline, candidate)
        self.assertTrue(diag["record_only"])
        self.assertEqual(diag["schema_version"],
                         r7.REV21_DIAGNOSTICS_SCHEMA)
        report = diag["rate_report"]
        self.assertEqual(report["n_pairs"], 3)
        # 逐种子 rate delta:7000: .04-.03=+.01;7001: .02-.025=-.005;7002: +.004
        self.assertAlmostEqual(report["rows"][0]["delta_rate"], 0.01)
        self.assertAlmostEqual(report["rows"][1]["delta_rate"], -0.005)
        self.assertAlmostEqual(report["rows"][2]["delta_rate"], 0.004)
        self.assertEqual((report["wins"], report["losses"], report["ties"]),
                         (2, 1, 0))
        self.assertEqual(report["max_leverage_seed"], 7000)
        self.assertAlmostEqual(
            report["deleveraged_mean"], (-0.005 + 0.004) / 2)
        decomp = diag["survival_decomposition"]
        total = sum(c["farm_worker_wage"] for c in candidate["rows"])             - sum(b["farm_worker_wage"] for b in baseline["rows"])
        self.assertAlmostEqual(decomp["total_delta_wage"], total)
        self.assertAlmostEqual(
            decomp["rate_component"] + decomp["time_component"], total)
        self.assertEqual(decomp["deaths_flipped_to_survived"], [7000])
        self.assertEqual(decomp["deaths_flipped_to_died"], [7001])

    def test_rejects_unpaired_or_degenerate_rows(self):
        good = {"seed": 7000, "micro_steps": 100,
                "farm_worker_wage": 1.0, "died": False}
        with self.assertRaises(r7.CampaignError):
            r7._rev21_diagnostics(
                self._archive([good]),
                self._archive([dict(good, seed=7001)]))
        with self.assertRaises(r7.CampaignError):
            r7._rev21_diagnostics(
                self._archive([good]),
                self._archive([dict(good, micro_steps=0)]))


def _a5_analysis(failed_checks, all_checks=None):
    checks = {
        name: True for name in (
            all_checks or r7.AMENDMENT5_PREREG_CHECK_KEYS)
    }
    for name in failed_checks:
        checks[name] = False
    return {
        "verdict": {
            "status": "PASS" if not failed_checks else "FAIL",
            "checks": checks,
            "failed_checks": sorted(failed_checks),
        },
    }


class Amendment5GateTest(unittest.TestCase):
    def test_constants_frozen(self):
        self.assertEqual(r7.FINAL_DEATH_MARGIN, 0.10)
        self.assertEqual(r7.DEVELOPMENT_DEATH_MARGIN, 0.05)
        self.assertEqual(
            r7.AMENDMENT5_DROPPED_CHECK,
            "deaths.noninferiority_upper_bound",
        )
        self.assertEqual(r7.AMENDMENT5_PRE_CAMPAIGN_REVISION, 21)
        self.assertEqual(len(r7.AMENDMENT5_PRE_LAUNCHER_SHA256), 64)
        self.assertEqual(len(r7.AMENDMENT5_PRE_RECIPE_SHA256), 64)
        # 修正案五后的当前身份必须已离开 rev21 快照
        self.assertNotEqual(
            r7.CAMPAIGN_RECIPE_SHA256, r7.AMENDMENT5_PRE_RECIPE_SHA256)
        self.assertEqual(
            r7.CAMPAIGN_RECIPE["statistics"]["final_death_margin"], 0.10)

    def test_leg_passes_drops_only_noninferiority(self):
        only_ni = r7._amendment5_leg_passes(
            _a5_analysis(["deaths.noninferiority_upper_bound"]))
        self.assertTrue(only_ni["passed"])
        self.assertEqual(only_ni["amendment5_failed_checks"], [])

        with_sign = r7._amendment5_leg_passes(_a5_analysis(
            ["deaths.noninferiority_upper_bound", "ret.exact_sign"]))
        self.assertFalse(with_sign["passed"])
        self.assertEqual(
            with_sign["amendment5_failed_checks"], ["ret.exact_sign"])

        observed_higher = r7._amendment5_leg_passes(_a5_analysis(
            ["deaths.noninferiority_upper_bound",
             "deaths.observed_not_higher"]))
        self.assertFalse(observed_higher["passed"])

        clean = r7._amendment5_leg_passes(_a5_analysis([]))
        self.assertTrue(clean["passed"])

    def test_leg_passes_rejects_inconsistent_verdict(self):
        broken = _a5_analysis(["deaths.noninferiority_upper_bound"])
        broken["verdict"]["status"] = "PASS"
        with self.assertRaises(r7.CampaignError):
            r7._amendment5_leg_passes(broken)
        missing = _a5_analysis([], all_checks=("ret.exact_sign",))
        with self.assertRaises(r7.CampaignError):
            r7._amendment5_leg_passes(missing)

    def _selection_fixture(self, fail_map):
        analyses = {}
        for pool in r7.DEV_POOLS:
            for recipe in r7.RECIPE_PREFERENCE:
                for seed in r7.DEVELOPMENT_TRAIN_SEEDS:
                    key = f"{pool}:{recipe}:{seed}"
                    analyses[key] = _a5_analysis(fail_map.get(key, []))
        return analyses

    def test_selection_matches_frozen_development_readings(self):
        ni = "deaths.noninferiority_upper_bound"
        seeds = r7.DEVELOPMENT_TRAIN_SEEDS
        fail_map = {
            # risk32:s2130200 双池带侧翼失败;s2130100 B 池死亡观测上升
            f"dev-a:risk32:{seeds[0]}": [ni],
            f"dev-a:risk32:{seeds[1]}": [ni],
            f"dev-a:risk32:{seeds[2]}": [
                ni, "deaths.observed_not_higher", "kills.exact_sign",
                "ret.exact_sign"],
            f"dev-b:risk32:{seeds[0]}": [ni],
            f"dev-b:risk32:{seeds[1]}": [ni, "deaths.observed_not_higher"],
            f"dev-b:risk32:{seeds[2]}": [
                ni, "deaths.observed_not_higher", "ret.exact_sign"],
            # risk64:仅 dev-a 第三种子带 kills/ret 符号失败
            f"dev-a:risk64:{seeds[0]}": [ni],
            f"dev-a:risk64:{seeds[1]}": [ni],
            f"dev-a:risk64:{seeds[2]}": [
                ni, "kills.exact_sign", "ret.exact_sign"],
            f"dev-b:risk64:{seeds[0]}": [ni],
            f"dev-b:risk64:{seeds[1]}": [ni],
            f"dev-b:risk64:{seeds[2]}": [ni],
        }
        selection = r7._amendment5_selection(self._selection_fixture(fail_map))
        self.assertEqual(selection["selected_recipe"], "risk64")
        self.assertEqual(
            selection["per_recipe"]["risk32"]["count"], 1)
        self.assertEqual(
            selection["per_recipe"]["risk32"]["seeds_passing_both_pools"],
            [seeds[0]])
        self.assertEqual(
            selection["per_recipe"]["risk64"]["count"], 2)
        self.assertEqual(
            selection["per_recipe"]["risk64"]["seeds_passing_both_pools"],
            [seeds[0], seeds[1]])

    def test_selection_prefers_risk32_when_both_qualify(self):
        selection = r7._amendment5_selection(self._selection_fixture({}))
        self.assertEqual(selection["selected_recipe"], "risk32")

    def test_selection_returns_none_when_no_recipe_qualifies(self):
        ni = "deaths.noninferiority_upper_bound"
        fail_map = {
            f"{pool}:{recipe}:{seed}": [ni, "farm_worker_wage.mean_lcb"]
            for pool in r7.DEV_POOLS
            for recipe in r7.RECIPE_PREFERENCE
            for seed in r7.DEVELOPMENT_TRAIN_SEEDS
        }
        selection = r7._amendment5_selection(self._selection_fixture(fail_map))
        self.assertIsNone(selection["selected_recipe"])

    def test_eval_tags_cover_frozen_development_plan(self):
        tags = r7._amendment5_eval_tags()
        self.assertEqual(len(tags), 14)
        self.assertEqual(len(set(tags)), 14)
        self.assertIn("r7-dev-a-baseline-v28", tags)
        self.assertIn("r7-dev-b-risk64-s2130200", tags)

    def test_adopt_command_registered(self):
        self.assertTrue(callable(r7.command_adopt_development))
        self.assertTrue(callable(r7._validate_amendment5_adoption))


class Amendment5MachineryTest(unittest.TestCase):
    """夹具战役:伪造 rev21 scientific-fail 终态,全链演练收养机器。"""

    _PATCHED = (
        "CONTROL_DIR", "STATE_PATH", "DEVELOPMENT_DECISION_PATH",
        "AMENDMENT5_PATH", "TRAINING_FIRED_DIR", "EVAL_DIR",
        "EVAL_ATTESTATION_DIR",
    )

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name)
        self._saved = {name: getattr(r7, name) for name in self._PATCHED}
        self._saved_receipt = r7._training_receipt_path
        control = root / "control"
        r7.CONTROL_DIR = control
        r7.STATE_PATH = control / "status.json"
        r7.DEVELOPMENT_DECISION_PATH = control / "development-decision.json"
        r7.AMENDMENT5_PATH = control / "amendment5-adoption.json"
        r7.TRAINING_FIRED_DIR = control / "training-fired"
        r7.EVAL_DIR = root / "eval-assembled"
        r7.EVAL_ATTESTATION_DIR = control / "eval-attestations"
        receipts = root / "receipts"
        r7._training_receipt_path = (
            lambda recipe, seed, scope:
            receipts / f"{recipe}-{seed}-{scope}.json")
        for path in (control, r7.TRAINING_FIRED_DIR, r7.EVAL_DIR,
                     r7.EVAL_ATTESTATION_DIR, receipts):
            path.mkdir(parents=True, exist_ok=True)
        self._build_rev21_terminal()

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(r7, name, value)
        r7._training_receipt_path = self._saved_receipt
        self._tmp.cleanup()

    # ---- 夹具构造 ----

    def _write_json(self, path, value):
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=1) + "\n")

    def _analysis(self, failed):
        checks = {name: True for name in r7.AMENDMENT5_PREREG_CHECK_KEYS}
        for name in failed:
            checks[name] = False
        return {
            "verdict": {
                "status": "PASS" if not failed else "FAIL",
                "checks": checks,
                "failed_checks": sorted(failed),
            },
        }

    def _frozen_fail_map(self):
        ni = "deaths.noninferiority_upper_bound"
        seeds = r7.DEVELOPMENT_TRAIN_SEEDS
        return {
            f"dev-a:risk32:{seeds[0]}": [ni],
            f"dev-a:risk32:{seeds[1]}": [ni],
            f"dev-a:risk32:{seeds[2]}": [
                ni, "deaths.observed_not_higher", "kills.exact_sign",
                "ret.exact_sign"],
            f"dev-b:risk32:{seeds[0]}": [ni],
            f"dev-b:risk32:{seeds[1]}": [ni, "deaths.observed_not_higher"],
            f"dev-b:risk32:{seeds[2]}": [
                ni, "deaths.observed_not_higher", "ret.exact_sign"],
            f"dev-a:risk64:{seeds[0]}": [ni],
            f"dev-a:risk64:{seeds[1]}": [ni],
            f"dev-a:risk64:{seeds[2]}": [
                ni, "kills.exact_sign", "ret.exact_sign"],
            f"dev-b:risk64:{seeds[0]}": [ni],
            f"dev-b:risk64:{seeds[1]}": [ni],
            f"dev-b:risk64:{seeds[2]}": [ni],
        }

    def _build_rev21_terminal(self, fail_map=None):
        fail_map = fail_map or self._frozen_fail_map()
        analysis_sha = {}
        for pool in r7.DEV_POOLS:
            for recipe in r7.RECIPE_PREFERENCE:
                for seed in r7.DEVELOPMENT_TRAIN_SEEDS:
                    key = f"{pool}:{recipe}:{seed}"
                    path = r7._analysis_path(pool, recipe, seed)
                    self._write_json(path, self._analysis(fail_map[key]))
                    analysis_sha[key] = r7._sha256(path)
        self._write_json(
            r7.DEVELOPMENT_DECISION_PATH,
            {"selected_recipe": None, "analysis_sha256s": analysis_sha},
        )
        for tag in r7._amendment5_eval_tags():
            fired = r7._eval_fired_path(tag)
            fired.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(fired, {"tag": tag, "kind": "fired"})
            archive = r7._eval_path(tag)
            self._write_json(archive, {"tag": tag, "kind": "archive"})
            self._write_json(
                r7._eval_attestation_path(tag),
                {
                    "tag": tag,
                    "archive_sha256": r7._sha256(archive),
                    "fired_sha256": r7._sha256(fired),
                },
            )
        for recipe in r7.RECIPE_PREFERENCE:
            for seed in r7.DEVELOPMENT_TRAIN_SEEDS:
                self._write_json(
                    r7._training_receipt_path(recipe, seed, "development"),
                    {"leg": f"{recipe}:{seed}", "kind": "receipt"})
                self._write_json(
                    r7._training_fired_path(recipe, seed, "development"),
                    {"leg": f"{recipe}:{seed}", "kind": "fired"})
        state = {
            "schema_version": r7.STATE_SCHEMA,
            "campaign_revision": r7.AMENDMENT5_PRE_CAMPAIGN_REVISION,
            "recipe_sha256": r7.AMENDMENT5_PRE_RECIPE_SHA256,
            "launcher_sha256": r7.AMENDMENT5_PRE_LAUNCHER_SHA256,
            "implementation": {
                **r7._implementation_identity(),
                "launcher_sha256": r7.AMENDMENT5_PRE_LAUNCHER_SHA256,
                "recipe_sha256": r7.AMENDMENT5_PRE_RECIPE_SHA256,
            },
            "phases": {
                "prepare_bc": {"status": "complete"},
                "train_development": {"status": "complete"},
                "eval_development": {
                    "status": "scientific-fail",
                    "completed": sorted(r7._development_analysis_keys()),
                    "decision_sha256": r7._sha256(
                        r7.DEVELOPMENT_DECISION_PATH),
                    "selected_recipe": None,
                },
            },
            "terminal_status": "DEVELOPMENT_SCIENTIFIC_FAIL",
        }
        self._write_json(r7.STATE_PATH, state)

    # ---- 测试 ----

    def test_adopt_migrates_selects_risk64_and_is_idempotent(self):
        r7.command_adopt_development()
        self.assertTrue(r7.AMENDMENT5_PATH.exists())
        state = r7._stable_json(r7.STATE_PATH)
        self.assertEqual(state["campaign_revision"], r7.CAMPAIGN_REVISION)
        self.assertIsNone(state["terminal_status"])
        phase = state["phases"]["eval_development"]
        self.assertEqual(phase["status"], "complete-amendment5-posthoc")
        self.assertEqual(phase["selected_recipe"], "risk64")
        self.assertIs(phase["post_hoc"], True)
        document = r7._stable_json(r7.AMENDMENT5_PATH)
        self.assertIs(document["post_hoc"], True)
        self.assertEqual(
            document["post"]["selection"]["per_recipe"]["risk64"]["count"],
            2)
        self.assertEqual(
            document["post"]["selection"]["per_recipe"]["risk32"]["count"],
            1)
        # 幂等重跑 + 改道复验
        r7.command_adopt_development()
        verdict = r7._validate_development_decision(
            r7._load_state())
        self.assertEqual(verdict["selected_recipe"], "risk64")

    def test_adopt_refuses_judgment_anchor_tamper(self):
        # 对抗复核实证的伪造路径:改单文件使 risk32 达 2/3 —— 锚定必须拦截
        seeds = r7.DEVELOPMENT_TRAIN_SEEDS
        path = r7._analysis_path("dev-b", "risk32", seeds[1])
        self._write_json(
            path,
            self._analysis(["deaths.noninferiority_upper_bound"]))
        with self.assertRaisesRegex(r7.CampaignError, "失锚"):
            r7.command_adopt_development()
        self.assertFalse(r7.AMENDMENT5_PATH.exists())
        self.assertEqual(
            r7._stable_json(r7.STATE_PATH)["terminal_status"],
            "DEVELOPMENT_SCIENTIFIC_FAIL")

    def test_adopt_crash_window_repair(self):
        pre_state_bytes = r7.STATE_PATH.read_bytes()
        r7.command_adopt_development()
        # 模拟 doc 已落、state 未迁的崩溃窗
        r7.STATE_PATH.write_bytes(pre_state_bytes)
        r7.command_adopt_development()
        state = r7._stable_json(r7.STATE_PATH)
        self.assertEqual(state["campaign_revision"], r7.CAMPAIGN_REVISION)
        self.assertEqual(
            state["phases"]["eval_development"]["selected_recipe"],
            "risk64")

    def test_post_adoption_development_commands_sealed(self):
        r7.command_adopt_development()
        state_before = r7.STATE_PATH.read_bytes()
        with self.assertRaisesRegex(r7.CampaignError, "封存"):
            r7.command_eval_development()
        with self.assertRaisesRegex(r7.CampaignError, "封存"):
            r7.command_train_development()
        # phase 记录必须原封不动(不得被 locked-failed 覆写)
        self.assertEqual(r7.STATE_PATH.read_bytes(), state_before)

    def test_validate_rejects_wiped_phase(self):
        r7.command_adopt_development()
        state = r7._stable_json(r7.STATE_PATH)
        state["phases"]["eval_development"] = {
            "status": "locked-failed", "completed": [],
            "retry_forbidden": True, "error": "simulated",
        }
        self._write_json(r7.STATE_PATH, state)
        with self.assertRaisesRegex(
                r7.CampaignError, "amendment5 state phase 未闭合"):
            r7._validate_amendment5_adoption(state)

    def test_adopt_refuses_when_no_recipe_qualifies(self):
        fail_map = {
            key: ["deaths.noninferiority_upper_bound",
                  "farm_worker_wage.mean_lcb"]
            for key in self._frozen_fail_map()
        }
        self._build_rev21_terminal(fail_map)
        with self.assertRaisesRegex(r7.CampaignError, "拒绝收养"):
            r7.command_adopt_development()
        self.assertFalse(r7.AMENDMENT5_PATH.exists())

    def test_adopt_refuses_tampered_eval_artifact_chain(self):
        tag = "r7-dev-a-baseline-v28"
        archive = r7._eval_path(tag)
        self._write_json(archive, {"tag": tag, "kind": "archive", "x": 1})
        with self.assertRaisesRegex(r7.CampaignError, "失锚"):
            r7.command_adopt_development()
        self.assertFalse(r7.AMENDMENT5_PATH.exists())


class Amendment6MachineryTest(unittest.TestCase):
    """夹具:伪造 rev22 修五收养态 + 终考事故现场,演练修六机器。"""

    _A6_PATCHED = (
        "AMENDMENT6_PATH", "FINAL_OPENED_PATH", "FINAL_FIRED_PATH",
        "FINAL_ANALYSIS_PATH", "PUBLISHED_DIR", "FINAL_REGISTRY_DIR",
        "AMENDMENT6_BURNED_FIRED_SHA256", "AMENDMENT6_INCIDENT_LOG_SHA256",
    )

    def setUp(self):
        self._a5 = Amendment5MachineryTest("setUp")
        self._a5.setUp()
        self._a6_saved = {
            name: getattr(r7, name) for name in self._A6_PATCHED}
        self._saved_model_path = r7._training_model_path
        root = pathlib.Path(self._a5._tmp.name)
        control = r7.CONTROL_DIR
        r7.AMENDMENT6_PATH = control / "amendment6-final-incident.json"
        r7.FINAL_OPENED_PATH = control / "final-pool-opened.json"
        r7.FINAL_FIRED_PATH = control / "final-candidate-fired.json"
        r7.FINAL_ANALYSIS_PATH = control / "analysis-final.json"
        r7.PUBLISHED_DIR = root / "published"
        r7.FINAL_REGISTRY_DIR = root / "final-registry"
        r7._training_model_path = (
            lambda recipe, seed, scope:
            root / "receipts" / f"{recipe}-{seed}-{scope}-model.zip")
        self._promote_to_rev22_incident(root, control)

    def tearDown(self):
        r7._training_model_path = self._saved_model_path
        for name, value in self._a6_saved.items():
            setattr(r7, name, value)
        self._a5.tearDown()

    def _promote_to_rev22_incident(self, root, control):
        wj = self._a5._write_json
        # 伪造修五收养文档(post 钉 rev22 = A6_PRE 三元组)
        state_sha = r7._sha256(r7.STATE_PATH)
        analyses = r7._amendment5_read_frozen_analyses()
        self._selection = r7._amendment5_selection(analyses)
        pre5 = r7._amendment5_pre_inventory(state_sha)
        wj(r7.AMENDMENT5_PATH, {
            "schema_version": r7.AMENDMENT5_SCHEMA,
            "amendment": 5, "plan": "B", "post_hoc": True,
            "adopted_at_ns": 1,
            "pre": pre5,
            "post": {
                "campaign_revision": r7.AMENDMENT6_PRE_CAMPAIGN_REVISION,
                "launcher_sha256": r7.AMENDMENT6_PRE_LAUNCHER_SHA256,
                "recipe_sha256": r7.AMENDMENT6_PRE_RECIPE_SHA256,
                "final_death_margin": r7.FINAL_DEATH_MARGIN,
                "development_gate": r7.AMENDMENT5_DEVELOPMENT_GATE,
                "selection": self._selection,
            },
        })
        recipe = self._selection["selected_recipe"]
        self._recipe = recipe
        # 生产件三件套
        receipt_path = r7._training_receipt_path(
            recipe, r7.PRODUCTION_TRAIN_SEED, "candidate")
        model_path = r7._training_model_path(
            recipe, r7.PRODUCTION_TRAIN_SEED, "candidate")
        wj(receipt_path, {
            "model_path": str(model_path), "kind": "production-receipt"})
        wj(r7._training_fired_path(
            recipe, r7.PRODUCTION_TRAIN_SEED, "candidate"),
            {"kind": "production-fired"})
        model_path.write_bytes(b"fixture-production-model")
        artifact = r7._stable_json(receipt_path)
        artifact["receipt_sha256"] = r7._sha256(receipt_path)
        self._artifact = artifact
        # 事故现场:被烧池 opened + 基线点火标记 + registry 一条 + 日志副本
        wj(r7.FINAL_OPENED_PATH, {
            "seeds": list(range(*r7.AMENDMENT6_BURNED_POOL)),
            "kind": "burned-opened"})
        bind_sha = r7._sha256(r7.FINAL_OPENED_PATH)
        self._bind_sha = bind_sha
        burned = (control / "eval-fired"
                  / f"{r7.AMENDMENT6_BURNED_BASELINE_TAG}.json")
        wj(burned, {"kind": "burned-baseline-fired",
                    "final_bind_sha256": bind_sha})
        r7.AMENDMENT6_BURNED_FIRED_SHA256 = r7._sha256(burned)
        r7.FINAL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        burned_seeds = list(range(*r7.AMENDMENT6_BURNED_POOL))
        pool_sha = r7._canonical_sha256(
            {"pool_name": "r7-official-final", "seeds": burned_seeds})
        wj(r7.FINAL_REGISTRY_DIR / f"{pool_sha}.json", {
            "schema_version": r7.FINAL_REGISTRY_SCHEMA,
            "pool_name": "r7-official-final",
            "seeds": burned_seeds,
            "pool_sha256": pool_sha,
            "campaign_recipe_sha256": r7.AMENDMENT6_PRE_RECIPE_SHA256,
            "bind_sha256": bind_sha,
            "control_path": str(r7.CONTROL_DIR.resolve()),
            "opened_at_ns": 1,
            "consumption_stage": "before_baseline_evaluation",
        })
        incident_dir = r7._amendment6_incident_dir()
        incident_dir.mkdir(parents=True, exist_ok=True)
        log_copy = incident_dir / "eval-final.partial.log"
        log_copy.write_text("seed 2120213: fixture tail\n")
        r7.AMENDMENT6_INCIDENT_LOG_SHA256 = r7._sha256(log_copy)
        # state 提升到 rev22 修五收养态 + 事故停点
        state = r7._stable_json(r7.STATE_PATH)
        state["campaign_revision"] = r7.AMENDMENT6_PRE_CAMPAIGN_REVISION
        state["launcher_sha256"] = r7.AMENDMENT6_PRE_LAUNCHER_SHA256
        state["recipe_sha256"] = r7.AMENDMENT6_PRE_RECIPE_SHA256
        state["implementation"] = {
            **r7._implementation_identity(),
            "launcher_sha256": r7.AMENDMENT6_PRE_LAUNCHER_SHA256,
            "recipe_sha256": r7.AMENDMENT6_PRE_RECIPE_SHA256,
        }
        state["terminal_status"] = None
        state["phases"]["eval_development"] = {
            "status": "complete-amendment5-posthoc",
            "completed": sorted(r7._development_analysis_keys()),
            "decision_sha256": pre5["decision_sha256"],
            "selected_recipe": recipe,
            "amendment5_sha256": r7._sha256(r7.AMENDMENT5_PATH),
            "post_hoc": True,
        }
        state["phases"]["train_production"] = {
            "status": "complete", "recipe": recipe,
            "seed": r7.PRODUCTION_TRAIN_SEED, "attempts": 1,
            "artifact": artifact,
        }
        state["phases"]["eval_final"] = {
            "status": "opened", "bind_sha256": bind_sha,
        }
        self._a5._write_json(r7.STATE_PATH, state)

    def test_adopt_incident_migrates_and_is_idempotent(self):
        r7.command_adopt_final_incident()
        self.assertTrue(r7.AMENDMENT6_PATH.exists())
        state = r7._stable_json(r7.STATE_PATH)
        self.assertEqual(state["campaign_revision"], r7.CAMPAIGN_REVISION)
        self.assertNotIn("eval_final", state["phases"])
        self.assertEqual(
            state["phases"]["final_incident"]["status"],
            "adopted-amendment6")
        incident_dir = r7._amendment6_incident_dir()
        self.assertTrue(
            (incident_dir
             / f"{r7.AMENDMENT6_BURNED_BASELINE_TAG}.json").exists())
        self.assertFalse(r7.FINAL_OPENED_PATH.exists())
        self.assertTrue(
            (incident_dir / r7.FINAL_OPENED_PATH.name).exists())
        # 幂等重跑 + 生产件改道复验
        r7.command_adopt_final_incident()
        receipt = r7._production_receipt(self._recipe)
        self.assertEqual(receipt, self._artifact)
        # a5 链在 a6 生效后仍闭合
        verdict = r7._validate_development_decision(r7._load_state())
        self.assertEqual(verdict["selected_recipe"], self._recipe)


    def test_all_production_receipt_sites_rerouted(self):
        # 复核 blocker 回归网:三处生产回执消费点必须全部走 _production_receipt
        for func in (r7._capture_final_evidence, r7._final_bind,
                     r7.command_eval_final):
            source = inspect.getsource(func)
            self.assertNotIn("_validate_training_artifact", source,
                             func.__name__)
        self.assertIn(
            "_production_receipt(", inspect.getsource(
                r7._capture_final_evidence))

    def test_adopt_refuses_forged_opened_document(self):
        # 锚定回归:伪造 opened(seeds 正确但字节不同)必失锚
        self._a5._write_json(r7.FINAL_OPENED_PATH, {
            "seeds": list(range(*r7.AMENDMENT6_BURNED_POOL)),
            "kind": "forged-opened"})
        with self.assertRaisesRegex(r7.CampaignError, "失锚"):
            r7.command_adopt_final_incident()
        self.assertFalse(r7.AMENDMENT6_PATH.exists())

    def test_adopt_incident_crash_window_repair(self):
        original = r7._amendment6_complete_adoption

        def _boom(document, state):
            raise RuntimeError("simulated crash after doc write")

        r7._amendment6_complete_adoption = _boom
        try:
            with self.assertRaises(RuntimeError):
                r7.command_adopt_final_incident()
        finally:
            r7._amendment6_complete_adoption = original
        self.assertTrue(r7.AMENDMENT6_PATH.exists())
        self.assertEqual(
            r7._stable_json(r7.STATE_PATH)["campaign_revision"],
            r7.AMENDMENT6_PRE_CAMPAIGN_REVISION)
        r7.command_adopt_final_incident()
        state = r7._stable_json(r7.STATE_PATH)
        self.assertEqual(state["campaign_revision"], r7.CAMPAIGN_REVISION)
        self.assertEqual(
            state["phases"]["final_incident"]["status"],
            "adopted-amendment6")

    def test_adopt_refuses_if_candidate_ever_fired(self):
        tag = r7.AMENDMENT6_BURNED_BASELINE_TAG.replace(
            "baseline", "candidate")
        self._a5._write_json(
            r7.CONTROL_DIR / "eval-fired" / f"{tag}.json",
            {"kind": "candidate-fired"})
        with self.assertRaisesRegex(r7.CampaignError, "候选必须从未"):
            r7.command_adopt_final_incident()
        self.assertFalse(r7.AMENDMENT6_PATH.exists())

    def test_validation_catches_production_receipt_tamper(self):
        r7.command_adopt_final_incident()
        receipt_path = r7._training_receipt_path(
            self._recipe, r7.PRODUCTION_TRAIN_SEED, "candidate")
        self._a5._write_json(receipt_path, {"kind": "tampered"})
        with self.assertRaisesRegex(r7.CampaignError, "生产回执漂移"):
            r7.command_adopt_final_incident()

    def test_train_production_sealed_after_adoption(self):
        r7.command_adopt_final_incident()
        state_before = r7.STATE_PATH.read_bytes()
        with self.assertRaisesRegex(r7.CampaignError, "禁止重训"):
            r7.command_train_production()
        self.assertEqual(r7.STATE_PATH.read_bytes(), state_before)


if __name__ == "__main__":
    unittest.main()
