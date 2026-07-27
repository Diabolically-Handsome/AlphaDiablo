"""official-v4 launcher 的轻量命令、状态机与防错回归。

本文件不启动 BC、Diablo 引擎或 PPO 训练。
"""

from __future__ import annotations

import json
import hashlib
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import run_v4_combat_recovery as launcher  # noqa: E402
import train_ppo  # noqa: E402


def _flag_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _candidate() -> dict:
    return {
        "candidate_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "preflight_sha256": "c" * 64,
        "expectations_sha256": "d" * 64,
        "training_contract_sha256": "e" * 64,
        "candidate_policy_head_sha256": "f" * 64,
        "root_policy_head_sha256": launcher.V28_POLICY_HEAD_SHA256,
        "implementation_sha256": "1" * 64,
        "bc_v2_demos_sha256": "2" * 64,
        "num_timesteps": launcher.TARGET_STEPS,
    }


def _snapshot() -> dict:
    return {
        "implementation_sha256": "1" * 64,
        "v28_sha256": launcher.V28_SHA256,
        "v28_policy_head_sha256": launcher.V28_POLICY_HEAD_SHA256,
        "manager_sha256": launcher.M29_SHA256,
        "teacher_sha256": launcher.KING_SHA256,
    }


def _bc() -> dict:
    return {
        "v1_demos_sha256": "3" * 64,
        "v1_policy_sha256": "7" * 64,
        "v1_report_sha256": "8" * 64,
        "v2_demos_sha256": "2" * 64,
        "v2_policy_sha256": "9" * 64,
        "v2_report_sha256": "0" * 64,
    }


def _dynamic() -> dict:
    return {
        "bc_aux_liveness_preflight_sha256": "4" * 64,
        "training_contract_sha256": "5" * 64,
        "preflight_path": "/frozen/preflight.json",
    }


def _passing_metrics() -> dict:
    metrics = {
        "scope": "full",
        "mask_mode": "bc-v2-recorded",
        "pairs": 10,
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "tn": 8,
        "true_a12": 1,
        "non_a12": 8,
        "all_non_a12": 9,
        "predicted_a12": 0,
        "predicted_a12_episodes": 0,
        "predicted_a12_margin_min": 0.0,
        "precision_12": 0.0,
        "recall_12": 0.0,
        "fpr_12": 0.0,
        "predicted_share_12": 0.0,
        "high_hp_non_a12": 4,
        "high_hp_false_drinks": 0,
        "high_hp_false_drink_rate": 0.0,
        "eligible_probability_12_min": 0.05,
        "eligible_probability_12_mean": 0.05,
        "eligible_probability_12_max": 0.05,
        "legal_negative_probability_12_mean": 0.0,
        "legal_negative_probability_12_max": 0.0,
        "legal_negative_probability_12_sum": 0.0,
        "predicted_share_13": 0.1,
        "true_share_13": 0.1,
        "a13_reference": "anchor_argmax",
        "a13_reference_share": 0.1,
        "a13_spillover": 0.0,
        "mean_probability_12": 0.005,
        "anchor": {
            "argmax_drift": 0.0,
            "tv_mean": 0.005,
            "kl_anchor_to_policy": 0.005,
            "a12_probability_delta": 0.005,
            "a13_predicted_share": 0.1,
            "critical_action_retention": {
                str(action): {
                    "support": 1,
                    "retained": 1,
                    "retention": 1.0,
                }
                for action in train_ppo._BC_AUX_CRITICAL_ACTIONS
            },
        },
    }
    if set(metrics) != set(train_ppo._BC_AUX_BEHAVIOR_METRIC_KEYS):
        raise AssertionError("测试 metrics 夹具与 producer schema 漂移")
    return metrics


def _valid_liveness_document(
    installation: str = "first-install",
) -> dict:
    if installation not in {"first-install", "preserved-continuation"}:
        raise AssertionError(installation)
    first_install = installation == "first-install"
    plan = {
        "rollout_quantum": launcher.ROLLOUT_QUANTUM,
        "train_calls": launcher.LEG_STEPS // launcher.ROLLOUT_QUANTUM,
        "aux_optimizer_calls": 0,
        "policy_gradient_canary_calls": 1,
        "initial_adapter_calibrations": 1 if first_install else 0,
        "trainable_adapter_parameters": 5,
    }
    metrics = _passing_metrics()
    gate = train_ppo.bc_aux_behavior_gate(
        metrics,
        require_root_anchor=True,
        require_teacher_recall=False,
    )
    circuit = {
        **train_ppo._bc_aux_circuit_spec(),
        "king_support": train_ppo._BC_AUX_CIRCUIT_KING_SUPPORT,
    }
    target = float(circuit["initial_probability"])
    grafted_sha = "7" * 64
    probability_before = 0.05
    probability_after = 0.06
    bias_before = -2.9
    bias_after = -2.8
    policy_gradient_canary = {
        "schema_version": "a12-policy-gradient-canary/1",
        "scope": "bc-v2-nested-validation-positive-only",
        "pairs": 1,
        "heldout_rows_consumed": 0,
        "objective": "negative-mean-log-probability-action12",
        "optimizer_steps": 1,
        "movement_required": True,
        "probability_12_before": probability_before,
        "probability_12_after": probability_after,
        "probability_12_delta": probability_after - probability_before,
        "gate_bias_before": bias_before,
        "gate_bias_after": bias_after,
        "gate_bias_delta": bias_after - bias_before,
        "gate_bias_gradient": -0.95,
        "gradient_norm_before_clip": 1.0,
        "start_policy_head_sha256": grafted_sha,
        "stepped_policy_head_sha256": "8" * 64,
        "state_restored": True,
    }
    calibration = (
        {
            "fit_pairs": 80,
            "validation_pairs": 10,
            "fit_positive_a12": 8,
            "validation_positive_a12": 1,
            "initializer": "exact-contextual-legal-support-mixture",
            "gate_feature_indices": list(circuit["gate_feature_indices"]),
            "gate_parameter_columns":
                list(circuit["gate_parameter_columns"]),
            "target_probability_12": target,
            "fit_positive_probability_min_12": target,
            "fit_positive_probability_12": target,
            "fit_positive_probability_max_12": target,
            "validation_positive_probability_min_12": target,
            "validation_positive_probability_12": target,
            "validation_positive_probability_max_12": target,
            "initial_argmax_lower_bound":
                (1.0 - target) / 14.0 - target,
            "initial_gate_bias": float(circuit["initial_gate_bias"]),
            "gate_coefficients": [0.0, 0.0, 0.0, 0.0],
            "probability_min": float(circuit["probability_min"]),
            "probability_max": float(circuit["probability_max"]),
            "fit_metrics": metrics,
            "validation_metrics": metrics,
            "validation_gate": gate,
            "candidate_policy_head_sha256": grafted_sha,
        }
        if first_install
        else {
            "initializer": "preserved-continuation",
            "gate_coefficients": [0.2, -0.1, 0.3, 0.0],
            "gate_bias": float(circuit["initial_gate_bias"]),
            "fit_pairs_excluded_from_retuning": 80,
            "validation_pairs": 10,
            "validation_metrics": metrics,
            "validation_gate": gate,
            "candidate_policy_head_sha256": grafted_sha,
        }
    )
    return {
        "schema_version": train_ppo._BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION,
        "protocol_version": 4,
        "objective_revision": launcher.BC_AUX_OBJECTIVE_REVISION,
        "inputs": {
            "resume_checkpoint_sha256": launcher.V28_SHA256,
            "demos_sha256": _bc()["v2_demos_sha256"],
            "manager_npz_sha256": launcher.M29_SHA256,
            "implementation_sha256": _snapshot()["implementation_sha256"],
        },
        "config": {
            "bc_aux_lambda": launcher.BC_AUX_LAMBDA,
            "seed": launcher.TRAIN_SEED,
            "device": "cpu",
            "learning_rate": launcher.LEARNING_RATE,
            "distill_beta": launcher.DISTILL_BETA,
            "target_kl": launcher.TARGET_KL,
            "reset_optimizer": first_install,
            "n_steps": launcher.N_STEPS,
            "num_envs": launcher.NUM_ENVS,
            "batch_size": train_ppo._select_batch_size(
                launcher.N_STEPS, launcher.NUM_ENVS
            ),
            "total_steps": launcher.LEG_STEPS,
            "mechanism": launcher.BC_AUX_MODE,
            "circuit": circuit,
            **plan,
        },
        "status": "PASS",
        "simulation":
            "isolated-exact-mixture-with-policy-gradient-canary",
        "installation": installation,
        "evaluation_scope": "bc-v2-nested-validation-only",
        "heldout_rows_consumed": 0,
        "circuit": circuit,
        "optimizer": {
            "class": "torch.optim.adam.Adam",
            "state_entries_at_start": 0 if first_install else 12,
            "learning_rates_at_start": [launcher.LEARNING_RATE],
            "reset_after_topology_change": first_install,
        },
        "policy": {
            "start_head_sha256": (
                launcher.V28_POLICY_HEAD_SHA256
                if first_install else grafted_sha
            ),
            "root_head_sha256": launcher.V28_POLICY_HEAD_SHA256,
            "grafted_head_sha256": grafted_sha,
            "actor_width_before": (
                train_ppo._BC_AUX_CIRCUIT_BASE_WIDTH
                if first_install
                else train_ppo._BC_AUX_CIRCUIT_EXPANDED_WIDTH
            ),
            "actor_width_after": train_ppo._BC_AUX_CIRCUIT_EXPANDED_WIDTH,
        },
        "calls": {
            "planned_train_calls": plan["train_calls"],
            "aux_optimizer_calls": 0,
            "policy_gradient_canary_calls": 1,
            "initial_adapter_calibrations": 1 if first_install else 0,
            "trainable_adapter_parameters": 5,
        },
        "policy_gradient_canary": policy_gradient_canary,
        "calibration": calibration,
        "metrics": metrics,
        "gate": gate,
    }


def _deployable_metrics() -> dict:
    metrics = _passing_metrics()
    metrics.update({
        "scope": "heldout",
        "pairs": 1000,
        "tp": 3,
        "fp": 0,
        "fn": 2,
        "tn": 995,
        "true_a12": 5,
        "non_a12": 995,
        "all_non_a12": 995,
        "predicted_a12": 3,
        "predicted_a12_episodes": 2,
        "predicted_a12_margin_min": 0.01,
        "precision_12": 1.0,
        "recall_12": 0.6,
        "fpr_12": 0.0,
        "predicted_share_12": 0.003,
        "high_hp_non_a12": 500,
        "high_hp_false_drinks": 0,
        "high_hp_false_drink_rate": 0.0,
        "eligible_probability_12_min": 0.4,
        "eligible_probability_12_mean": 0.5,
        "eligible_probability_12_max": 0.6,
        "mean_probability_12": 0.0025,
    })
    return metrics


def _published_receipt() -> dict:
    metrics = _deployable_metrics()
    gate = train_ppo.bc_aux_behavior_gate(
        metrics,
        require_root_anchor=True,
        require_teacher_recall=False,
        require_deployable_a12=False,
    )
    if gate["verdict"] != "PASS":
        raise AssertionError(gate)
    return {
        "schema_version":
            train_ppo._BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION,
        "step": launcher.TARGET_STEPS,
        "demos_sha256": _bc()["v2_demos_sha256"],
        "objective_revision": launcher.BC_AUX_OBJECTIVE_REVISION,
        "evaluation_scope": "original-bc-v2-heldout-episodes",
        "mask_mode": "bc-v2-recorded",
        "anchor": {
            "identity": "bc-aux-root-policy",
            "policy_head_sha256": launcher.V28_POLICY_HEAD_SHA256,
        },
        "candidate_policy_head_sha256": "6" * 64,
        "provenance": {"frozen": True},
        "metrics": metrics,
        "gate": gate,
        "exploration_evidence": {
            "eligible_states": 100,
            "expected_a12_mass": 25.0,
            "requested_a12": 18,
            "sampled_a12": 15,
            "rejected_a12": 3,
            "unexpected_sampled_a12": 0,
            "minimum_expected_a12_mass":
                train_ppo._BC_AUX_MIN_EXPECTED_A12_SAMPLES,
            "minimum_actual_a12_samples":
                train_ppo._BC_AUX_MIN_ACTUAL_A12_SAMPLES,
            "information_status": "INFORMATIVE",
            "reasons": [],
        },
        "publication": "PUBLISHED",
        "model_sha256": "5" * 64,
        "save_error": None,
    }


class FrozenCommandTests(unittest.TestCase):
    def test_r6_isolated_from_all_prior_control_candidate_eval_namespaces(self):
        old_control = launcher.TRAIN / "runs" / "v4-combat-recovery-control"
        old_candidate = launcher.TRAIN / "runs" / "v4-combat-recovery-full"
        r2_control = launcher.TRAIN / "runs" / "v4-combat-recovery-r2-control"
        r2_candidate = launcher.TRAIN / "runs" / "v4-combat-recovery-r2-full"
        r3_control = launcher.TRAIN / "runs" / "v4-combat-recovery-r3-control"
        r3_candidate = launcher.TRAIN / "runs" / "v4-combat-recovery-r3-full"
        r4_control = launcher.TRAIN / "runs" / "v4-combat-recovery-r4-control"
        r4_candidate = launcher.TRAIN / "runs" / "v4-combat-recovery-r4-full"
        r5_control = launcher.TRAIN / "runs" / "v4-combat-recovery-r5-control"
        r5_candidate = launcher.TRAIN / "runs" / "v4-combat-recovery-r5-full"
        self.assertNotIn(
            launcher.CONTROL_DIR,
            {old_control, r2_control, r3_control, r4_control, r5_control},
        )
        self.assertNotIn(
            launcher.CANDIDATE_DIR,
            {
                old_candidate,
                r2_candidate,
                r3_candidate,
                r4_candidate,
                r5_candidate,
            },
        )
        self.assertEqual(
            launcher.CONTROL_DIR.name, "v4-combat-recovery-r6-control")
        self.assertEqual(
            launcher.CANDIDATE_DIR.name,
            "v4-combat-recovery-r6-safe-prefix",
        )
        self.assertEqual(launcher.CAMPAIGN_RECIPE["campaign_revision"], 6)
        self.assertEqual(
            launcher.CAMPAIGN_RECIPE["safe_prefix_replay"][
                "expected_policy_head_sha256"
            ],
            launcher.R6_EXPECTED_POLICY_HEAD_SHA256,
        )
        self.assertTrue(all(
            "official-v4-r6-" in tag
            for tag in launcher.EVAL_TAGS.values()
        ))

    def test_pool_discipline_and_bc_pool_identity_survive_campaign_rollover(self):
        launcher._require_seed_pool_discipline()
        expected = (
            (1, list(range(2000, 2128)),
             "adedee94c012fd4760aa3d086ac5f698e1062f823f0fc3a15af932daea0a81a9"),
            (2, list(range(3000, 3384)),
             "628f6325f0c402b0b5c0b3f033e000a9c7dd04ed385622f99c74eb4a2d5ad13e"),
        )
        for generation, seeds, expected_sha in expected:
            _, pool_sha = train_ppo._bc_final_holdout_marker_identity(
                generation, seeds
            )
            self.assertEqual(pool_sha, expected_sha)
            marker, _, marker_pool_sha = (
                train_ppo._bc_final_holdout_marker_path(
                    launcher.BC_V1_DIR if generation == 1 else launcher.BC_V2_DIR,
                    generation,
                    seeds,
                )
            )
            self.assertEqual(marker_pool_sha, expected_sha)
            self.assertEqual(marker.parent.name, "_bc_final_holdout_registry")

    def test_training_command_is_the_only_exact_recipe(self):
        command = launcher._training_command()
        self.assertEqual(_flag_value(command, "--total-steps"), "249856")
        self.assertEqual(_flag_value(command, "--seed"), "304000")
        self.assertEqual(_flag_value(command, "--resume-from"), str(launcher.V28_ZIP))
        self.assertEqual(_flag_value(command, "--manager-npz"), str(launcher.M29_NPZ))
        self.assertEqual(
            _flag_value(command, "--teacher-override"), str(launcher.KING_SD)
        )
        self.assertEqual(_flag_value(command, "--device"), "cpu")
        self.assertEqual(_flag_value(command, "--target-kl"), "0.02")
        self.assertEqual(_flag_value(command, "--distill-beta"), "0.015625")
        self.assertEqual(_flag_value(command, "--bc-aux-lambda"), "0.0")
        self.assertEqual(
            _flag_value(command, "--dry-curriculum-schedule"),
            launcher.CURRICULUM,
        )
        self.assertIn("--allow-legacy-resume", command)
        self.assertIn("--reset-optimizer", command)
        self.assertIn("--bc-aux-liveness-preflight", command)
        self.assertNotIn("run_v33_content.py", " ".join(command))
        self.assertEqual(launcher.LEG_STEPS, 122 * launcher.ROLLOUT_QUANTUM)
        self.assertEqual(
            launcher.START_STEPS + launcher.LEG_STEPS,
            launcher.TARGET_STEPS,
        )
        self.assertEqual(launcher.TARGET_STEPS, 3_747_840)

    def test_r5_safe_prefix_anchor_is_exactly_replayable(self):
        anchor = launcher._validate_r5_safe_replay_anchor()
        self.assertEqual(anchor["num_timesteps"], 3_747_840)
        self.assertEqual(anchor["ppo_optimizer_steps_completed"], 122 * 80)
        self.assertEqual(
            anchor["policy_head_sha256"],
            launcher.R6_EXPECTED_POLICY_HEAD_SHA256,
        )

    def test_historical_r6_contract_is_superseded_by_rev13(self):
        self.assertEqual(launcher.TRAINING_CONTRACT_REVISION, 12)
        with self.assertRaisesRegex(
            launcher.CampaignError,
            "未命中 official-v4-r6",
        ):
            launcher._expected_training_contract(_snapshot(), _bc())

    def test_prepare_bc_v2_and_all_evals_pin_m29_and_unique_tags(self):
        _, v2 = launcher._prepare_bc_commands()
        self.assertEqual(_flag_value(v2, "--manager-npz"), str(launcher.M29_NPZ))
        for pool in launcher.EVAL_POOLS:
            baseline = launcher._eval_command(pool, "baseline")
            candidate = launcher._eval_command(pool, "candidate")
            self.assertEqual(
                _flag_value(baseline, "--manager-npz"), str(launcher.M29_NPZ)
            )
            self.assertEqual(
                _flag_value(candidate, "--manager-npz"), str(launcher.M29_NPZ)
            )
            self.assertNotEqual(
                _flag_value(baseline, "--tag"),
                _flag_value(candidate, "--tag"),
            )
            self.assertNotIn("--require-published-worker", baseline)
            self.assertIn("--require-published-worker", candidate)
            self.assertEqual(
                _flag_value(candidate, "--publication-expectations"),
                str(launcher.EXPECTATIONS_PATH),
            )

    def test_cli_has_no_training_refire_escape_hatch(self):
        parser = launcher.build_parser()
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["train", "--retry-operational"])

    def test_prepare_bc_does_not_reconsume_valid_v1_when_only_v2_is_stale(self):
        v1 = {
            "v1_demos_sha256": "1" * 64,
            "v1_policy_sha256": "2" * 64,
            "v1_report_sha256": "3" * 64,
        }
        v2 = {
            "v2_demos_sha256": "4" * 64,
            "v2_policy_sha256": "5" * 64,
            "v2_report_sha256": "6" * 64,
        }
        v2_calls = 0

        def v2_validator():
            nonlocal v2_calls
            v2_calls += 1
            if v2_calls == 1:
                raise launcher.CampaignError("stale")
            return v2

        invoked = []
        with (
            mock.patch.object(launcher, "_load_state", return_value=launcher._new_state()),
            mock.patch.object(
                launcher,
                "_base_artifact_snapshot",
                return_value={"implementation_sha256": "7" * 64},
            ),
            mock.patch.object(launcher, "_bc_v1_identity", return_value=v1),
            mock.patch.object(launcher, "_bc_v2_identity", side_effect=v2_validator),
            mock.patch.object(
                launcher, "_bc_identities", return_value={**v1, **v2}
            ),
            mock.patch.object(launcher, "_reject_current_bc_scientific_failure"),
            mock.patch.object(
                launcher,
                "_invoke",
                side_effect=lambda command, _label: invoked.append(command),
            ),
            mock.patch.object(launcher, "_set_phase"),
        ):
            launcher.command_prepare_bc()
        self.assertEqual(len(invoked), 1)
        self.assertIn("--v2", invoked[0])

    def test_bc_terminal_receipt_blocks_only_same_generator_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            report = pathlib.Path(directory) / "bc_report.json"
            implementation = "7" * 64
            generator = hashlib.sha256(
                launcher.BC_V1_POLICY.parents[2]
                .joinpath("bc_worker.py").read_bytes()
            ).hexdigest()
            record = {
                "protocol_version": launcher.PROTOCOL_VERSION,
                "implementation_sha256": implementation,
                "generator_sha256": generator,
                "data_gate": "FAIL",
            }
            report.write_text(json.dumps(record))
            with self.assertRaisesRegex(
                    launcher.CampaignError, "禁止 launcher 自动重采"):
                launcher._reject_current_bc_scientific_failure(
                    report,
                    implementation_sha256=implementation,
                    label="BC-v2",
                )
            record["generator_sha256"] = "0" * 64
            report.write_text(json.dumps(record))
            launcher._reject_current_bc_scientific_failure(
                report,
                implementation_sha256=implementation,
                label="BC-v2",
            )

    def test_train_requires_every_prepared_bc_identity_to_match(self):
        state = launcher._new_state()
        prepared = _bc()
        state["phases"]["prepare_bc"] = {
            "status": "complete",
            "implementation_sha256": _snapshot()["implementation_sha256"],
            **prepared,
        }
        live = dict(prepared)
        live["v1_policy_sha256"] = "a" * 64
        with (
            mock.patch.object(launcher, "_load_state", return_value=state),
            mock.patch.object(
                launcher, "_base_artifact_snapshot", return_value=_snapshot()
            ),
            mock.patch.object(launcher, "_bc_identities", return_value=live),
            mock.patch.object(launcher, "_freeze_expectations") as freeze,
        ):
            with self.assertRaisesRegex(
                launcher.CampaignError, "prepare-bc"
            ):
                launcher.command_train()
        freeze.assert_not_called()

    def test_campaign_rejects_unregistered_bc_v2_oc_threshold(self):
        demos_sha = "d" * 64
        report = {
            "implementation_sha256": "1" * 64,
            "manager_npz_sha256": launcher.M29_SHA256,
            "demos_sha256": demos_sha,
            "preventive_threshold": 0.70,
            "policy_sha256": "2" * 64,
        }
        with (
            mock.patch.object(
                train_ppo,
                "_load_bc_aux_demos_v2",
                return_value=(None, None, None, None, demos_sha),
            ),
            mock.patch.object(
                train_ppo,
                "_implementation_bundle_sha256",
                return_value="1" * 64,
            ),
            mock.patch.object(
                launcher,
                "_stable_read",
                return_value=json.dumps(report).encode(),
            ),
        ):
            with self.assertRaisesRegex(
                launcher.CampaignError, "threshold=0.65"
            ):
                launcher._bc_v2_identity()


class ProvenanceHardeningTests(unittest.TestCase):
    def test_expectations_are_exact_full_17_key_provenance(self):
        document = launcher._expectations_document(
            _snapshot(), _bc(), _dynamic()
        )
        expected = document["expected_provenance"]
        self.assertEqual(set(expected), launcher.EXPECTED_PROVENANCE_KEYS)
        self.assertEqual(len(expected), 17)
        self.assertEqual(
            expected["bc_aux_liveness_preflight_sha256"],
            _dynamic()["bc_aux_liveness_preflight_sha256"],
        )
        self.assertEqual(
            expected["training_contract_sha256"],
            _dynamic()["training_contract_sha256"],
        )

    def test_liveness_requires_exact_mixture_initialization_counts(self):
        document = _valid_liveness_document()
        launcher._validate_liveness_preflight(
            json.dumps(document).encode(),
            snapshot=_snapshot(),
            bc=_bc(),
        )
        del document["calls"]["initial_adapter_calibrations"]
        with self.assertRaisesRegex(
            launcher.CampaignError, "planned/initial"
        ):
            launcher._validate_liveness_preflight(
                json.dumps(document).encode(),
                snapshot=_snapshot(),
                bc=_bc(),
            )

    def test_liveness_requires_policy_gradient_canary_and_one_real_call(self):
        document = _valid_liveness_document()
        del document["policy_gradient_canary"]
        with self.assertRaisesRegex(
            launcher.CampaignError, "顶层 schema"
        ):
            launcher._validate_liveness_preflight(
                json.dumps(document).encode(),
                snapshot=_snapshot(),
                bc=_bc(),
            )

        document = _valid_liveness_document()
        document["calls"]["policy_gradient_canary_calls"] = 0
        with self.assertRaisesRegex(
            launcher.CampaignError, "planned/initial"
        ):
            launcher._validate_liveness_preflight(
                json.dumps(document).encode(),
                snapshot=_snapshot(),
                bc=_bc(),
            )

        document = _valid_liveness_document()
        document["policy_gradient_canary"]["state_restored"] = False
        with self.assertRaisesRegex(
            launcher.CampaignError, "canary schema"
        ):
            launcher._validate_liveness_preflight(
                json.dumps(document).encode(),
                snapshot=_snapshot(),
                bc=_bc(),
            )

    def test_liveness_rejects_boolean_zero_counter_forgery(self):
        for location, key in (
            ("top", "heldout_rows_consumed"),
            ("config", "aux_optimizer_calls"),
            ("calls", "aux_optimizer_calls"),
        ):
            with self.subTest(location=location):
                document = _valid_liveness_document()
                target = (
                    document if location == "top"
                    else document[location]
                )
                target[key] = False
                with self.assertRaisesRegex(
                    launcher.CampaignError, "schema|bool"
                ):
                    launcher._validate_liveness_preflight(
                        json.dumps(document).encode(),
                        snapshot=_snapshot(),
                        bc=_bc(),
                    )

    def test_liveness_rejects_old_fixed_circuit_and_initializer_drift(self):
        document = _valid_liveness_document()
        document["simulation"] = (
            "isolated-structural-static-necessary-condition")
        with self.assertRaisesRegex(
            launcher.CampaignError, "顶层 schema"
        ):
            launcher._validate_liveness_preflight(
                json.dumps(document).encode(),
                snapshot=_snapshot(),
                bc=_bc(),
            )

        document = _valid_liveness_document()
        document["calibration"]["gate_coefficients"][0] = 0.01
        with self.assertRaisesRegex(
            launcher.CampaignError, "initializer"
        ):
            launcher._validate_liveness_preflight(
                json.dumps(document).encode(),
                snapshot=_snapshot(),
                bc=_bc(),
            )

    def test_liveness_accepts_strict_preserved_continuation_branch(self):
        document = _valid_liveness_document("preserved-continuation")
        validated = launcher._validate_liveness_preflight(
            json.dumps(document).encode(),
            snapshot=_snapshot(),
            bc=_bc(),
        )
        self.assertEqual(
            validated["installation"], "preserved-continuation")
        self.assertEqual(
            validated["calls"]["initial_adapter_calibrations"], 0)
        self.assertFalse(
            validated["optimizer"]["reset_after_topology_change"])

        document["calls"]["initial_adapter_calibrations"] = 1
        with self.assertRaisesRegex(
            launcher.CampaignError, "planned/initial"
        ):
            launcher._validate_liveness_preflight(
                json.dumps(document).encode(),
                snapshot=_snapshot(),
                bc=_bc(),
            )

    def test_liveness_safety_gate_has_no_teacher_recall_floor(self):
        document = _valid_liveness_document()
        self.assertEqual(document["metrics"]["recall_12"], 0.0)
        validated = launcher._validate_liveness_preflight(
            json.dumps(document).encode(),
            snapshot=_snapshot(),
            bc=_bc(),
        )
        self.assertIsNone(
            validated["gate"]["thresholds"]["recall_12_min"])
        self.assertFalse(
            validated["gate"]["thresholds"]["teacher_recall_required"])

    def test_publication_leaves_deterministic_a12_deployment_to_ppo(self):
        receipt = _published_receipt()
        validated = launcher._validate_candidate_behavior_receipt(
            receipt,
            expected_model_sha256="5" * 64,
            expected_policy_head_sha256="6" * 64,
            expected_demos_sha256=_bc()["v2_demos_sha256"],
            expected_provenance={"frozen": True},
        )
        self.assertEqual(validated["verdict"], "PASS")
        self.assertFalse(
            validated["thresholds"]["deployable_a12_required"])

        receipt = _published_receipt()
        receipt["metrics"] = _passing_metrics()
        receipt["metrics"]["scope"] = "heldout"
        receipt["gate"] = train_ppo.bc_aux_behavior_gate(
            receipt["metrics"],
            require_root_anchor=True,
            require_teacher_recall=False,
            require_deployable_a12=False,
        )
        validated = launcher._validate_candidate_behavior_receipt(
            receipt,
            expected_model_sha256="5" * 64,
            expected_policy_head_sha256="6" * 64,
            expected_demos_sha256=_bc()["v2_demos_sha256"],
            expected_provenance={"frozen": True},
        )
        self.assertEqual(validated["verdict"], "PASS")
        self.assertEqual(receipt["metrics"]["predicted_a12"], 0)

    def test_publication_rejects_malformed_or_nonclosing_exploration_counts(
            self):
        for field, value in (
            ("minimum_expected_a12_mass", 0.0),
            ("minimum_actual_a12_samples", True),
            ("requested_a12", True),
            ("rejected_a12", 0.5),
            ("requested_a12", 17),
        ):
            with self.subTest(field=field):
                receipt = _published_receipt()
                receipt["exploration_evidence"][field] = value
                with self.assertRaisesRegex(
                    launcher.CampaignError, "20/10"
                ):
                    launcher._validate_candidate_behavior_receipt(
                        receipt,
                        expected_model_sha256="5" * 64,
                        expected_policy_head_sha256="6" * 64,
                        expected_demos_sha256=_bc()["v2_demos_sha256"],
                        expected_provenance={"frozen": True},
                    )

    def test_candidate_receipt_policy_head_string_is_not_trusted(self):
        preflight_payload = b"preflight"
        dynamic = _dynamic()
        dynamic["bc_aux_liveness_preflight_sha256"] = hashlib.sha256(
            preflight_payload
        ).hexdigest()
        expected = launcher._known_expected_provenance(
            _snapshot(), _bc(), dynamic
        )
        receipt = {
            "candidate_policy_head_sha256": "9" * 64,
            "anchor": {
                "identity": "bc-aux-root-policy",
                "policy_head_sha256": launcher.V28_POLICY_HEAD_SHA256,
            },
            "provenance": expected,
        }
        receipt_payload = json.dumps(receipt).encode()

        def stable_read(path, _label):
            path = pathlib.Path(path)
            if path == launcher.CANDIDATE_ZIP:
                return b"candidate"
            if path == launcher.CANDIDATE_RECEIPT:
                return receipt_payload
            if path in {
                launcher.CANDIDATE_PREFLIGHT,
                launcher.PREREG_PREFLIGHT_PATH,
            }:
                return preflight_payload
            raise AssertionError(path)

        with (
            mock.patch.object(launcher, "_base_artifact_snapshot", return_value=_snapshot()),
            mock.patch.object(launcher, "_bc_identities", return_value=_bc()),
            mock.patch.object(
                launcher,
                "_read_expectations",
                return_value=(
                    {
                        "schema_version": launcher.EXPECTATIONS_SCHEMA,
                        "expected_provenance": expected,
                    },
                    "d" * 64,
                ),
            ),
            mock.patch.object(
                launcher,
                "_expected_training_contract",
                return_value=({}, dynamic["training_contract_sha256"]),
            ),
            mock.patch.object(launcher, "_stable_read", side_effect=stable_read),
            mock.patch.object(launcher, "_validate_liveness_preflight"),
            mock.patch.object(
                launcher,
                "_checkpoint_policy_heads",
                return_value={
                    "current_policy_head_sha256": "8" * 64,
                    "root_policy_head_sha256": launcher.V28_POLICY_HEAD_SHA256,
                },
            ),
            mock.patch.object(
                launcher,
                "_checkpoint_data",
                return_value={
                    "num_timesteps": launcher.TARGET_STEPS,
                    "diablogym_contract": {},
                },
            ),
            mock.patch(
                "eval_assembled.capture_published_worker",
                return_value=receipt_payload,
            ),
            mock.patch(
                "eval_assembled.capture_publication_expectations",
                return_value=(expected, {"path": "/x", "sha256": "d" * 64}),
            ),
            mock.patch("eval_assembled.verify_publication_expectations"),
        ):
            with self.assertRaisesRegex(
                launcher.CampaignError, "策略头 SHA"
            ):
                launcher._validate_candidate()


class StateIdentityTests(unittest.TestCase):
    @staticmethod
    def _legacy_state(phases):
        return {
            "schema_version": launcher.LEGACY_STATE_SCHEMA,
            "campaign_recipe_sha256": launcher.CAMPAIGN_RECIPE_SHA256,
            "campaign_recipe": dict(launcher.CAMPAIGN_RECIPE),
            "updated_at_ns": 1,
            "phases": phases,
        }

    def test_pre_scientific_v1_state_migrates_with_full_embedded_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            state_path = base / "status.json"
            legacy = self._legacy_state(
                {"prepare_bc": {"status": "failed", "error": "old"}}
            )
            payload = json.dumps(legacy).encode()
            state_path.write_bytes(payload)
            with mock.patch.multiple(
                launcher,
                STATE_PATH=state_path,
                CANDIDATE_ZIP=base / "candidate.zip",
                CANDIDATE_RECEIPT=base / "receipt.json",
                CANDIDATE_PREFLIGHT=base / "preflight.json",
                EXPECTATIONS_PATH=base / "expectations.json",
                FRESH_LEDGER_PATH=base / "fresh.jsonl",
                FRESH_POOL_OPENED_PATH=base / "opened.json",
                FRESH_CANDIDATE_FIRED_PATH=base / "candidate-fired.json",
                PREREG_PREFLIGHT_PATH=base / "prereg.json",
                CONTROL_DIR=base / "control",
                CANDIDATE_DIR=base / "candidate-dir",
                EVAL_DIR=base / "eval",
            ):
                migrated = launcher._load_state()
            self.assertEqual(migrated["schema_version"], launcher.STATE_SCHEMA)
            self.assertEqual(
                migrated["migration"]["source_state"], legacy
            )
            self.assertEqual(
                migrated["migration"]["source_payload_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(
                migrated["migration"]["source_state_sha256"],
                launcher._canonical_sha256(legacy),
            )

    def test_legacy_state_with_scientific_phase_is_never_auto_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            state_path = base / "status.json"
            state_path.write_text(json.dumps(
                self._legacy_state({"train": {"status": "running"}})
            ))
            with mock.patch.multiple(
                launcher,
                STATE_PATH=state_path,
                CANDIDATE_ZIP=base / "candidate.zip",
                CANDIDATE_RECEIPT=base / "receipt.json",
                EXPECTATIONS_PATH=base / "expectations.json",
                FRESH_LEDGER_PATH=base / "fresh.jsonl",
                FRESH_POOL_OPENED_PATH=base / "opened.json",
            ):
                with self.assertRaisesRegex(
                    launcher.CampaignError, "禁止自动迁移"
                ):
                    launcher._load_state()

    def test_legacy_state_from_another_recipe_is_never_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            state_path = base / "status.json"
            legacy = self._legacy_state({"prepare_bc": {"status": "failed"}})
            legacy["campaign_recipe"] = {
                **legacy["campaign_recipe"],
                "manager": "OTHER",
            }
            legacy["campaign_recipe_sha256"] = launcher._canonical_sha256(
                legacy["campaign_recipe"]
            )
            state_path.write_text(json.dumps(legacy))
            with mock.patch.multiple(
                launcher,
                STATE_PATH=state_path,
                CANDIDATE_ZIP=base / "candidate.zip",
                CANDIDATE_RECEIPT=base / "receipt.json",
                CANDIDATE_PREFLIGHT=base / "preflight.json",
                EXPECTATIONS_PATH=base / "expectations.json",
                FRESH_LEDGER_PATH=base / "fresh.jsonl",
                FRESH_POOL_OPENED_PATH=base / "opened.json",
                CONTROL_DIR=base / "control",
                EVAL_DIR=base / "eval",
            ):
                with self.assertRaisesRegex(
                    launcher.CampaignError, "配方/经理"
                ):
                    launcher._load_state()


class PairedCombatGateTests(unittest.TestCase):
    @staticmethod
    def _documents(
            *, wage_delta=1.0, kill_delta=1, multi=0, max_drinks=1,
            voluntary_drinks=1):
        baseline_rows = []
        candidate_rows = []
        for seed in launcher.EVAL_POOLS["regression"]:
            common = {
                "seed": seed,
                "farm_worker_wage": 10.0,
                "farm_worker_kills": 4,
                "farm_dry_n": 3,
                "farm_fresh_n": 2,
                "farm_dry_worker_wage": 6.0,
                "farm_fresh_worker_wage": 4.0,
                "farm_dry_worker_kills": 2,
                "farm_fresh_worker_kills": 2,
                "ret": 20.0,
                "kills": 7,
                "died": False,
                "farm_voluntary_drinks": 1,
                "farm_reflex_drains": 0,
                "farm_multi_drink_windows": 0,
                "farm_max_voluntary_drinks_per_window": 1,
                "ending_belt_heals": 2,
            }
            baseline_rows.append(dict(common))
            contender = dict(common)
            contender.update(
                farm_worker_wage=common["farm_worker_wage"] + wage_delta,
                farm_worker_kills=common["farm_worker_kills"] + kill_delta,
                farm_dry_worker_wage=(
                    common["farm_dry_worker_wage"] + wage_delta),
                farm_dry_worker_kills=(
                    common["farm_dry_worker_kills"] + kill_delta),
                farm_voluntary_drinks=voluntary_drinks,
                farm_multi_drink_windows=multi,
                farm_max_voluntary_drinks_per_window=max_drinks,
            )
            candidate_rows.append(contender)
        return {"rows": baseline_rows}, {"rows": candidate_rows}

    def test_analysis_includes_potion_diagnostics_and_passes_only_real_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = pathlib.Path(directory)
            with mock.patch.object(launcher, "EVAL_DIR", eval_dir):
                for arm in ("baseline", "candidate"):
                    launcher._eval_archive_path("regression", arm).write_text("{}")
                baseline, candidate = self._documents()
                analysis = launcher._paired_analysis(
                    "regression", baseline, candidate
                )
        self.assertEqual(analysis["verdict"]["status"], "PASS")
        self.assertEqual(
            set(analysis["potion_diagnostics"]),
            {
                "farm_voluntary_drinks",
                "farm_reflex_drains",
                "farm_multi_drink_windows",
                "farm_max_voluntary_drinks_per_window",
                "ending_belt_heals",
            },
        )
        self.assertIn("ending_belt_heals", analysis["summary"])
        self.assertEqual(
            analysis["farm_stratum_diagnostics"],
            [
                "farm_dry_n",
                "farm_fresh_n",
                "farm_dry_worker_wage",
                "farm_fresh_worker_wage",
                "farm_dry_worker_kills",
                "farm_fresh_worker_kills",
            ],
        )
        self.assertEqual(
            analysis["summary"]["farm_dry_worker_wage"]["delta_mean"], 1.0)
        self.assertEqual(
            analysis["summary"]["farm_fresh_worker_wage"]["delta_mean"], 0.0)
        self.assertEqual(
            analysis["summary"]["farm_dry_worker_kills"]["delta_mean"], 1.0)
        self.assertEqual(
            analysis["summary"]["farm_fresh_worker_kills"]["delta_mean"], 0.0)
        self.assertEqual(
            analysis["summary"]["farm_dry_n"]["candidate_mean"], 3.0)
        self.assertEqual(
            analysis["summary"]["farm_fresh_n"]["candidate_mean"], 2.0)
        self.assertTrue(all(
            metric not in analysis["verdict"]["checks"]
            for metric in analysis["farm_stratum_diagnostics"]
        ))

    def test_zero_voluntary_drinks_does_not_override_ppo_outcome_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = pathlib.Path(directory)
            with mock.patch.object(launcher, "EVAL_DIR", eval_dir):
                for arm in ("baseline", "candidate"):
                    launcher._eval_archive_path("regression", arm).write_text("{}")
                baseline, candidate = self._documents(
                    voluntary_drinks=0, max_drinks=0)
                analysis = launcher._paired_analysis(
                    "regression", baseline, candidate
                )
        self.assertEqual(analysis["verdict"]["status"], "PASS")
        self.assertEqual(
            analysis["summary"]["farm_voluntary_drinks"]["candidate_mean"],
            0.0,
        )
        self.assertNotIn(
            "candidate_native_voluntary_drink_observed",
            analysis["verdict"]["checks"],
        )

    def test_no_gain_or_repeat_drink_fails_before_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = pathlib.Path(directory)
            with mock.patch.object(launcher, "EVAL_DIR", eval_dir):
                for arm in ("baseline", "candidate"):
                    launcher._eval_archive_path("regression", arm).write_text("{}")
                baseline, candidate = self._documents(
                    wage_delta=0.0, multi=1, max_drinks=2
                )
                analysis = launcher._paired_analysis(
                    "regression", baseline, candidate
                )
        self.assertEqual(analysis["verdict"]["status"], "FAIL")
        self.assertEqual(
            set(analysis["verdict"]["failed_checks"]),
            {
                "farm_worker_wage_mean_strictly_higher",
                "farm_worker_wage_strict_seed_majority",
                "candidate_multi_drink_windows_zero",
                "candidate_max_one_voluntary_drink_per_window",
            },
        )

    def test_single_outlier_cannot_hide_majority_primary_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = pathlib.Path(directory)
            with mock.patch.object(launcher, "EVAL_DIR", eval_dir):
                for arm in ("baseline", "candidate"):
                    launcher._eval_archive_path("regression", arm).write_text("{}")
                baseline, candidate = self._documents()
                for row in candidate["rows"][:-1]:
                    row["farm_worker_wage"] = 9.0
                    row["farm_worker_kills"] = 3
                candidate["rows"][-1]["farm_worker_wage"] = 100.0
                candidate["rows"][-1]["farm_worker_kills"] = 100
                analysis = launcher._paired_analysis(
                    "regression", baseline, candidate
                )
        self.assertGreater(
            analysis["summary"]["farm_worker_wage"]["delta_mean"], 0
        )
        self.assertEqual(analysis["verdict"]["status"], "FAIL")
        self.assertIn(
            "farm_worker_wage_strict_seed_majority",
            analysis["verdict"]["failed_checks"],
        )
        self.assertIn(
            "farm_worker_kills_strict_seed_majority",
            analysis["verdict"]["failed_checks"],
        )

    def test_fresh_requires_recorded_exact_pass_analysis(self):
        candidate_info = _candidate()
        state = launcher._new_state()
        state["phases"]["eval_regression"] = {
            "status": "complete",
            "candidate_sha256": candidate_info["candidate_sha256"],
            "expectations_sha256": candidate_info["expectations_sha256"],
            "analysis_path": "/wrong.json",
            "analysis_sha256": "0" * 64,
            "verdict": {"status": "PASS"},
        }
        baseline, contender = self._documents()
        exact_analysis = {
            "verdict": {
                "revision": launcher.REGRESSION_GATE_REVISION,
                "status": "PASS",
                "checks": {},
                "failed_checks": [],
            }
        }
        with (
            mock.patch.object(
                launcher,
                "_validate_official_archive",
                side_effect=[(baseline, {}), (contender, {})],
            ),
            mock.patch.object(launcher, "_pair_documents"),
            mock.patch.object(
                launcher,
                "_freeze_paired_analysis",
                return_value=(
                    exact_analysis,
                    pathlib.Path("/exact.json"),
                    "1" * 64,
                ),
            ),
        ):
            with self.assertRaisesRegex(
                launcher.CampaignError, "分析/裁决"
            ):
                launcher._require_regression_complete(state, candidate_info)


class LedgerTests(unittest.TestCase):
    def test_fresh_fsm_and_state_prefix_detect_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = pathlib.Path(directory) / "fresh.jsonl"
            with mock.patch.object(launcher, "FRESH_LEDGER_PATH", ledger):
                launcher._append_fresh_event("BIND", {"candidate": "a"})
                launcher._append_fresh_event(
                    "BASELINE_START",
                    {
                        "command": [],
                        "input_identity_sha256": "1" * 64,
                        "pool_opened_marker_sha256": "2" * 64,
                    },
                )
                baseline_identity = {
                    "path": "/baseline.json",
                    "sha256": "b" * 64,
                    "worker_sha256": "3" * 64,
                    "worker_receipt_sha256": None,
                    "manager_sha256": launcher.M29_SHA256,
                    "protocol_bundle_sha256": "4" * 64,
                }
                launcher._append_fresh_event(
                    "BASELINE_SUCCESS", baseline_identity
                )
                launcher._append_fresh_event(
                    "CANDIDATE_START",
                    {
                        "attempt": 1,
                        "command": [],
                        "input_identity_sha256": "1" * 64,
                        "candidate_fired_marker_sha256": "7" * 64,
                    },
                )
                candidate_identity = {
                    **baseline_identity,
                    "path": "/candidate.json",
                    "sha256": "c" * 64,
                    "worker_sha256": "5" * 64,
                    "worker_receipt_sha256": "6" * 64,
                }
                launcher._append_fresh_event(
                    "CANDIDATE_SUCCESS",
                    {"attempt": 1, "archive": candidate_identity},
                )
                events = launcher._read_fresh_ledger()
                summary = launcher._fresh_summary(events)
                self.assertTrue(summary["baseline_success"])
                self.assertTrue(summary["candidate_success"])
                self.assertEqual(summary["candidate_attempts"], 1)
                self.assertEqual(
                    summary["baseline_success_payload"], baseline_identity
                )
                self.assertEqual(
                    summary["candidate_success_payload"], candidate_identity
                )
                state = launcher._new_state()
                state["phases"]["eval_fresh"] = {
                    "ledger_event_count": len(events),
                    "ledger_head_sha256": events[-1]["event_sha256"],
                }
                with self.assertRaisesRegex(
                    launcher.CampaignError, "截尾|改写"
                ):
                    launcher._verify_fresh_ledger_checkpoint(
                        state, events[:-1]
                    )

    def test_pool_opened_marker_is_create_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "opened.json"
            launcher._exclusive_create_json(path, {"x": 1}, "pool")
            with self.assertRaisesRegex(launcher.CampaignError, "已存在"):
                launcher._exclusive_create_json(path, {"x": 2}, "pool")

    def test_pair_reads_protocol_seed_location(self):
        seeds = list(launcher.EVAL_POOLS["fresh"])
        common = {
            "protocol": {"seeds": seeds},
            "manager": {"sha256": launcher.M29_SHA256},
            "runtime": {"python_protocol": {"sha256": "a" * 64}},
        }
        launcher._pair_documents(
            "fresh",
            {"meta": common},
            {"meta": json.loads(json.dumps(common))},
        )


class FreshRecoveryTests(unittest.TestCase):
    def _paths(self, directory: str):
        base = pathlib.Path(directory)
        return mock.patch.multiple(
            launcher,
            CONTROL_DIR=base / "control",
            STATE_PATH=base / "control" / "status.json",
            FRESH_LEDGER_PATH=base / "control" / "fresh.jsonl",
            FRESH_POOL_OPENED_PATH=base / "control" / "opened.json",
            FRESH_CANDIDATE_FIRED_PATH=base / "control" / "candidate-fired.json",
            EVAL_DIR=base / "eval",
        )

    @staticmethod
    def _archive_validator(pool, arm, _candidate):
        path = launcher._eval_archive_path(pool, arm)
        if not path.is_file():
            raise launcher.CampaignError("missing")
        return (
            {"meta": {"arm": arm}},
            {
                "path": str(path),
                "sha256": arm[0] * 64,
                "worker_sha256": arm[0] * 64,
                "worker_receipt_sha256": None,
                "manager_sha256": launcher.M29_SHA256,
                "protocol_bundle_sha256": "f" * 64,
            },
        )

    def test_candidate_failure_can_never_be_refired_after_partial_output(self):
        with tempfile.TemporaryDirectory() as directory, self._paths(directory):
            launcher._save_state(launcher._new_state())
            invocations: list[str] = []
            candidate_attempts = 0

            def invoke(command, _label):
                nonlocal candidate_attempts
                tag = _flag_value(command, "--tag")
                invocations.append(tag)
                if tag == launcher.EVAL_TAGS[("fresh", "baseline")]:
                    launcher._eval_archive_path("fresh", "baseline").parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    launcher._eval_archive_path("fresh", "baseline").write_text("{}")
                    return
                candidate_attempts += 1
                if candidate_attempts == 1:
                    raise launcher.CommandFailed("operational")
                launcher._eval_archive_path("fresh", "candidate").write_text("{}")

            frozen = {"identity_sha256": "7" * 64, "runtime": {"x": 1}}
            patches = (
                mock.patch.object(launcher, "_validate_candidate", return_value=_candidate()),
                mock.patch.object(launcher, "_require_regression_complete"),
                mock.patch.object(launcher, "_eval_input_identity", return_value=frozen),
                mock.patch.object(
                    launcher,
                    "_validate_official_archive",
                    side_effect=self._archive_validator,
                ),
                mock.patch.object(launcher, "_pair_documents"),
                mock.patch.object(launcher, "_invoke", side_effect=invoke),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with self.assertRaises(launcher.CommandFailed):
                    launcher.command_eval_fresh()
                with self.assertRaisesRegex(
                    launcher.CampaignError, "禁止第二次点火|唯一预注册发次"
                ):
                    launcher.command_eval_fresh()

            self.assertEqual(
                invocations.count(launcher.EVAL_TAGS[("fresh", "baseline")]),
                1,
            )
            self.assertEqual(
                invocations.count(launcher.EVAL_TAGS[("fresh", "candidate")]),
                1,
            )
            summary = launcher._fresh_summary(launcher._read_fresh_ledger())
            self.assertFalse(summary["candidate_success"])
            self.assertEqual(summary["candidate_attempts"], 1)

    def test_failed_baseline_can_never_be_fired_again(self):
        with tempfile.TemporaryDirectory() as directory, self._paths(directory):
            launcher._save_state(launcher._new_state())
            invocations = 0

            def invoke(_command, _label):
                nonlocal invocations
                invocations += 1
                raise launcher.CommandFailed("baseline died")

            frozen = {"identity_sha256": "7" * 64, "runtime": {"x": 1}}
            with (
                mock.patch.object(
                    launcher, "_validate_candidate", return_value=_candidate()
                ),
                mock.patch.object(launcher, "_require_regression_complete"),
                mock.patch.object(
                    launcher, "_eval_input_identity", return_value=frozen
                ),
                mock.patch.object(
                    launcher,
                    "_validate_official_archive",
                    side_effect=self._archive_validator,
                ),
                mock.patch.object(launcher, "_invoke", side_effect=invoke),
            ):
                with self.assertRaises(launcher.CommandFailed):
                    launcher.command_eval_fresh()
                with self.assertRaisesRegex(
                    launcher.CampaignError, "禁止第二次 baseline"
                ):
                    launcher.command_eval_fresh()
            self.assertEqual(invocations, 1)

    def test_pollution_scan_finds_arbitrary_tag_using_fresh_seed(self):
        with tempfile.TemporaryDirectory() as directory, self._paths(directory):
            launcher.EVAL_DIR.mkdir(parents=True)
            (launcher.EVAL_DIR / "unrelated-name.json").write_text(
                json.dumps(
                    {
                        "meta": {
                            "protocol": {
                                "seeds": [12017],
                            }
                        },
                        "rows": [],
                    }
                )
            )
            residue = launcher._fresh_external_residue()
            self.assertTrue(any("12017" in value for value in residue))

    def test_prior_campaign_pool_marker_blocks_new_bind_without_archive(self):
        with tempfile.TemporaryDirectory() as directory, self._paths(directory):
            launcher._save_state(launcher._new_state())
            prior = (
                pathlib.Path(directory)
                / "v4-combat-recovery-r4-control"
            )
            prior.mkdir(parents=True)
            old_marker = prior / "fresh-12000-pool-opened.json"
            old_marker.write_text("{}")
            frozen = {"identity_sha256": "7" * 64, "runtime": {"x": 1}}
            with (
                mock.patch.object(
                    launcher, "_validate_candidate", return_value=_candidate()
                ),
                mock.patch.object(launcher, "_require_regression_complete"),
                mock.patch.object(
                    launcher, "_eval_input_identity", return_value=frozen
                ),
            ):
                with self.assertRaisesRegex(
                    launcher.CampaignError, "BIND 前已有档案/void"
                ):
                    launcher.command_eval_fresh()
            self.assertFalse(launcher.FRESH_LEDGER_PATH.exists())


if __name__ == "__main__":
    unittest.main()
