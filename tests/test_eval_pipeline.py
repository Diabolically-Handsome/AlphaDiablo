"""评估/选模链路的纯函数回归测试（不启动引擎或训练）。"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import dashboard  # noqa: E402
import check_teacher_parity  # noqa: E402
import eval_assembled  # noqa: E402
import eval_contract  # noqa: E402
import evaluate  # noqa: E402
import run_v25_election  # noqa: E402
import run_v28_legs  # noqa: E402
import run_v30_relay  # noqa: E402
import train_ppo  # noqa: E402
from diablogym.controller_wire import DUAL_WORKER_LAYOUT  # noqa: E402


def _hold_output_reservation(path, ready, release):
    with eval_contract.reserve_output(path):
        ready.set()
        release.wait(10)


def _hold_tag_and_board_reservation(tag_path, board_lock, ready, release):
    with eval_contract.reserve_output(tag_path), \
            eval_contract.exclusive_lock(board_lock, "排行榜"):
        ready.set()
        release.wait(10)


def _runtime_content_fixture(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    data = root / "game-data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "spawn.mpq").write_bytes(b"spawn-v1")
    assets = eval_contract.default_assets_dir(root)
    (assets / "txtdata").mkdir(parents=True, exist_ok=True)
    (assets / "ASSETS_VERSION").write_bytes(b"1")
    (assets / "txtdata" / "monsters.tsv").write_bytes(b"monster\n")
    return data, assets


def _valid_v5_archive():
    rows = [
        {"seed": 7, "ret": 1.125, "depth": 1, "died": False, "kills": 2,
         "micro_steps": 2, "terminal_kind": "game_over",
         "farm_r": 1.0, "farm_w": 0.75, "farm_bonus": 0.25,
         "farm_worker_wage": 0.5, "farm_kills": 1,
         "farm_worker_kills": 1, "nonfarm_r": 0.125,
         "nonfarm_kills": 1,
         "farm_dry_n": 1, "farm_fresh_n": 0,
         "farm_dry_worker_wage": 0.5, "farm_fresh_worker_wage": 0.0,
         "farm_dry_worker_kills": 1, "farm_fresh_worker_kills": 0,
         "farm_voluntary_drinks": 1,
         "farm_reflex_drain_attempts": 1, "farm_reflex_drains": 0,
         "farm_multi_drink_windows": 0,
         "farm_max_voluntary_drinks_per_window": 1,
         "ending_belt_heals": 2,
         "farm_n": 1, "farm_tau_mean": 2.0, "farm_tau_sum": 2,
         "farm_descend": 0, "windows": 1, "beats": 2, "overrides": 0,
         "cap": 0, "mode_seq": "F"},
        {"seed": 8, "ret": 3.0, "depth": 3, "died": True, "kills": 4,
         "micro_steps": 5, "terminal_kind": "death",
         "farm_r": 2.0, "farm_w": 1.0, "farm_bonus": 1.0,
         "farm_worker_wage": 0.5, "farm_kills": 3,
         "farm_worker_kills": 2, "nonfarm_r": 1.0,
         "nonfarm_kills": 1,
         "farm_dry_n": 0, "farm_fresh_n": 1,
         "farm_dry_worker_wage": 0.0, "farm_fresh_worker_wage": 0.5,
         "farm_dry_worker_kills": 0, "farm_fresh_worker_kills": 2,
         "farm_voluntary_drinks": 2,
         "farm_reflex_drain_attempts": 2, "farm_reflex_drains": 1,
         "farm_multi_drink_windows": 1,
         "farm_max_voluntary_drinks_per_window": 2,
         "ending_belt_heals": 0,
         "farm_n": 1, "farm_tau_mean": 4.0, "farm_tau_sum": 4,
         "farm_descend": 1, "windows": 2, "beats": 4, "overrides": 1,
         "cap": 1, "mode_seq": "FD†"},
    ]
    source_files = {name: "d" * 64 for name in eval_contract.PROTOCOL_SOURCE_FILES}
    bundle_sha = eval_contract.source_bundle_sha256(source_files)
    worker = {
        "kind": "sb3_checkpoint", "path": "/frozen/worker.zip",
        "sha256": "a" * 64, "num_timesteps": 1234,
        "gate_report_sha256": None,
    }
    manager = {
        "kind": "numpy_policy", "path": "/frozen/manager.npz",
        "sha256": "b" * 64, "num_timesteps": None,
        "gate_report_sha256": None,
    }
    runtime = {
        "bridge": {"path": "/frozen/_diablogym.so", "sha256": "c" * 64},
        "engine": {"path": "/frozen/libdevilutionx.so", "sha256": "e" * 64},
        "content": {
            "game_data": {"path": "/frozen/data/DIABDAT.MPQ",
                          "sha256": "f" * 64},
            "assets": {
                "path": "/frozen/build/engine/devilutionx.app/Contents/Resources",
                "sha256": "1" * 64, "file_count": 2,
            },
        },
        "versions": eval_contract.runtime_versions_identity(),
        "python_protocol": {"sha256": bundle_sha, "files": source_files},
    }
    agg = eval_contract.recompute_agg(rows)
    agg.update({"worker_calls": 2, "worker_action_hist": {"9": 2},
                "worker_divergences": 1,
                "script_divergence_rate": 0.5})
    return {
        "schema_version": 5,
        "meta": eval_contract.make_meta(tag="audit-v4", seeds=[7, 8],
                                        worker=worker, manager=manager, runtime=runtime),
        "agg": agg,
        "rows": rows,
    }


def _published_worker_fixture(
        root: pathlib.Path, *,
        installation: str = "first-install"):
    """最小但全链自洽的正式发布 bundle（无需加载真实 SB3 policy）。"""
    checkpoint = root / "model_final.zip"
    preflight_path = root / "bc_aux_liveness_preflight.json"
    receipt_path = root / eval_contract.PUBLISHED_WORKER_RECEIPT_NAME
    expectations_path = root / "publication-expectations.json"
    implementation_sha = "b" * 64
    manager_sha = "c" * 64
    resume_sha = "d" * 64
    teacher_sha = "e" * 64
    demos_sha = "a" * 64
    circuit = {
        **train_ppo._bc_aux_circuit_spec(),
        "king_support": train_ppo._BC_AUX_CIRCUIT_KING_SUPPORT,
    }
    gate = {
        "verdict": "PASS",
        "reasons": [],
        "thresholds": {
            "root_anchor_required": True,
            "deployable_a12_required": False,
            "deterministic_a12_episode_min": None,
            "deterministic_a12_margin_min": None,
        },
    }
    liveness_metrics = {"fixture": True}
    grafted_sha = "3" * 64
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
        "stepped_policy_head_sha256": "4" * 64,
        "state_restored": True,
    }
    preflight = {
        "schema_version": train_ppo._BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION,
        "protocol_version": eval_contract.PROTOCOL_VERSION,
        "objective_revision": train_ppo._BC_AUX_OBJECTIVE_REVISION,
        "status": "PASS",
        "simulation":
            "isolated-exact-mixture-with-policy-gradient-canary",
        "installation": installation,
        "evaluation_scope": "bc-v2-nested-validation-only",
        "heldout_rows_consumed": 0,
        "circuit": circuit,
        "inputs": {
            "resume_checkpoint_sha256": resume_sha,
            "demos_sha256": demos_sha,
            "manager_npz_sha256": manager_sha,
            "implementation_sha256": implementation_sha,
        },
        "config": {
            "total_steps": 2048,
            "bc_aux_lambda": 0.0,
            "device": "cpu",
            "learning_rate": 3e-4,
            "mechanism":
                "expanded-trainable-a12-contextual-mixture",
            "circuit": circuit,
            "distill_beta": 0.015625,
            "target_kl": 0.02,
            "seed": 304000,
            "reset_optimizer": installation == "first-install",
            "n_steps": 512,
            "num_envs": 4,
            "batch_size": train_ppo._select_batch_size(512, 4),
            "rollout_quantum": 2048,
            "train_calls": 1,
            "aux_optimizer_calls": 0,
            "policy_gradient_canary_calls": 1,
            "initial_adapter_calibrations":
                1 if installation == "first-install" else 0,
            "trainable_adapter_parameters": 5,
        },
        "calls": {
            "planned_train_calls": 1,
            "aux_optimizer_calls": 0,
            "policy_gradient_canary_calls": 1,
            "initial_adapter_calibrations":
                1 if installation == "first-install" else 0,
            "trainable_adapter_parameters": 5,
        },
        "policy_gradient_canary": policy_gradient_canary,
        "optimizer": {
            "class": "torch.optim.adam.Adam",
            "state_entries_at_start":
                0 if installation == "first-install" else 12,
            "learning_rates_at_start": [3e-4],
            "reset_after_topology_change":
                installation == "first-install",
        },
        "policy": {
            "start_head_sha256": (
                "2" * 64 if installation == "first-install"
                else grafted_sha),
            "root_head_sha256": "2" * 64,
            "grafted_head_sha256": grafted_sha,
            "actor_width_before":
                64 if installation == "first-install" else 68,
            "actor_width_after": 68,
        },
        "calibration": {
            "initializer": (
                "exact-contextual-legal-support-mixture"
                if installation == "first-install"
                else "preserved-continuation"),
            "candidate_policy_head_sha256": grafted_sha,
            "validation_metrics": liveness_metrics,
            "validation_gate": gate,
        },
        "metrics": liveness_metrics,
        "gate": gate,
    }
    preflight_path.write_text(json.dumps(preflight, sort_keys=True))
    preflight_sha = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    contract = {
        "contract_revision": train_ppo._CONTRACT_REVISION,
        "artifact_scope": "production",
        "mode": "worker",
        "max_steps": eval_contract.PROTOCOL_MAX_STEPS,
        "gamma": 1.0,
        "observation_shape": [298],
        "action_n": 15,
        "implementation_sha256": implementation_sha,
        "manager_npz_sha256": manager_sha,
        "teacher_sha256": teacher_sha,
        "distill_beta": 0.015625,
        "calib_record_only": False,
        "worker_fast_forward_reward_credit": "none",
        "worker_additional_terminal_death_cost": 0.0,
        # Formal A12 checkpoints consume the raw protocol-v4 gate features.
        "legacy_policy_observation_view": False,
        "worker_policy_observation_view":
            "legacy-v3-a12-overlay",
        "manager_policy_observation_view": "legacy-v3",
        "worker_episode_boundary":
            train_ppo._WORKER_EPISODE_BOUNDARY_V24,
        "worker_window_bootstrap": "next-learning-window",
        "worker_no_progress_timeout":
            dict(train_ppo._WORKER_NO_PROGRESS_TIMEOUT_CONTRACT),
        "gradient_clipping": {"mode": "global", "max_norm": 0.5},
        "actor_migration": "disabled",
        "critic_migration": "disabled",
        "algorithm_recipe": {"target_kl": 0.02},
        "demos_sha256": None,
        "policy_source_roles": {
            "schema": train_ppo._POLICY_SOURCE_ROLES_SCHEMA,
            "initialization": "resume-checkpoint",
            "bc_v1_direct_policy_uses": ["distillation-teacher"],
            "bc_v1_dataset_uses": [],
            "distillation_teacher": "configured-bc-v1-export",
            "worker_action14_policy_sources": [
                "native-reward-bound-on-policy-ppo",
            ],
        },
        "bc_aux": {
            "mode": "expanded-trainable-a12-contextual-mixture",
            "objective_revision": train_ppo._BC_AUX_OBJECTIVE_REVISION,
            "demos_sha256": demos_sha,
            "lambda": 0.0,
            "circuit": train_ppo._bc_aux_circuit_spec(),
            "king_support": train_ppo._BC_AUX_CIRCUIT_KING_SUPPORT,
            "aux_optimizer_calls_per_rollout": 0,
            "initial_calibration":
                "exact-five-percent-contextual-legal-support-mixture",
            "trainable_adapter_parameters": 5,
            "post_step_projection": {
                "gate_parameter_abs_max":
                    train_ppo._bc_aux_circuit_spec()[
                        "gate_parameter_abs_max"],
                "probability_min":
                    train_ppo._bc_aux_circuit_spec()["probability_min"],
                "probability_max":
                    train_ppo._bc_aux_circuit_spec()["probability_max"],
            },
            "liveness_preflight": True,
        },
    }
    provenance = {
        "protocol_version": eval_contract.PROTOCOL_VERSION,
        "implementation_sha256": implementation_sha,
        "manager_npz_sha256": manager_sha,
        "resume_checkpoint_sha256": resume_sha,
        "teacher_sha256": teacher_sha,
        "bc_aux_demos_sha256": demos_sha,
        "bc_aux_liveness_preflight_sha256": preflight_sha,
        "training_contract_sha256":
            train_ppo._canonical_json_sha256(contract),
        "start_steps": 0,
        "target_global_steps": 2048,
        "seed": 304000,
        "optimizer_reset": installation == "first-install",
        "target_kl": 0.02,
        "distill_beta": 0.015625,
        "bc_aux_lambda": 0.0,
        "bc_aux_mode": "expanded-trainable-a12-contextual-mixture",
        "calib_record_only": False,
    }
    checkpoint_data = {
        "num_timesteps": 2048,
        "diablogym_contract": contract,
        "distill_beta": 0.015625,
        "teacher_sha256": teacher_sha,
        "bc_aux_lambda": 0.0,
        "policy_class": {
            ":type:": "<class 'abc.ABCMeta'>",
            ":serialized:": "fixture",
            "__module__": "leashed_ppo",
        },
        "policy_kwargs": {
            "net_arch": {"pi": [68, 68], "vf": [64, 64]},
            "bc_aux_mixture_spec": train_ppo._bc_aux_circuit_spec(),
        },
        "_bc_aux_circuit_spec": train_ppo._bc_aux_circuit_spec(),
        "_bc_aux_eligible_states": 400,
        "_bc_aux_requested_a12": 10,
        "_bc_aux_sampled_a12": 10,
        "_bc_aux_rejected_a12": 0,
        "_bc_aux_unexpected_sampled_a12": 0,
        "_bc_aux_expected_a12_mass": 20.0,
        "_ppo_optimizer_steps_completed": 1,
        "_last_completed_ppo_rollout_steps": 2048,
    }
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps(
            checkpoint_data, sort_keys=True, allow_nan=False))
    model_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = {
        "schema_version": train_ppo._BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION,
        "step": 2048,
        "demos_sha256": demos_sha,
        "objective_revision": train_ppo._BC_AUX_OBJECTIVE_REVISION,
        "evaluation_scope": "original-bc-v2-heldout-episodes",
        "mask_mode": "bc-v2-recorded",
        "anchor": {
            "identity": "bc-aux-root-policy",
            "policy_head_sha256": "2" * 64,
        },
        "candidate_policy_head_sha256": "3" * 64,
        "provenance": provenance,
        "metrics": {"fixture": True},
        "gate": gate,
        "exploration_evidence": {
            "eligible_states": 400,
            "expected_a12_mass": 20.0,
            "requested_a12": 10,
            "sampled_a12": 10,
            "rejected_a12": 0,
            "unexpected_sampled_a12": 0,
            "minimum_expected_a12_mass":
                train_ppo._BC_AUX_MIN_EXPECTED_A12_SAMPLES,
            "minimum_actual_a12_samples":
                train_ppo._BC_AUX_MIN_ACTUAL_A12_SAMPLES,
            "information_status": "INFORMATIVE",
            "reasons": [],
        },
        "publication": "PUBLISHED",
        "model_sha256": model_sha,
        "save_error": None,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    expected = {
        "schema_version": eval_assembled._PUBLICATION_EXPECTATIONS_SCHEMA,
        "expected_provenance": {
            "start_steps": 0,
            "target_global_steps": 2048,
            "resume_checkpoint_sha256": resume_sha,
            "manager_npz_sha256": manager_sha,
            "teacher_sha256": teacher_sha,
            "seed": 304000,
            "optimizer_reset": installation == "first-install",
            "target_kl": 0.02,
            "distill_beta": 0.015625,
            "bc_aux_lambda": 0.0,
            "bc_aux_mode":
                "expanded-trainable-a12-contextual-mixture",
            "calib_record_only": False,
        },
    }
    expectations_path.write_text(json.dumps(expected, sort_keys=True))
    return {
        "checkpoint": checkpoint,
        "receipt": receipt_path,
        "preflight": preflight_path,
        "expectations": expectations_path,
        "implementation_sha": implementation_sha,
        "manager_sha": manager_sha,
        "gate": gate,
        "provenance": provenance,
    }


def _independent_fixture_evidence(bundle: dict) -> dict:
    receipt = json.loads(bundle["receipt"].read_text())
    return {
        "candidate_policy_head_sha256":
            receipt["candidate_policy_head_sha256"],
        "anchor_policy_head_sha256":
            receipt["anchor"]["policy_head_sha256"],
        "metrics": receipt["metrics"],
    }


class EvalPipelineTests(unittest.TestCase):
    def test_assembled_v4_board_is_separate_and_archive_bound(self):
        self.assertEqual(eval_assembled.LB.name, "leaderboard-assembled-v4.md")
        self.assertNotEqual(eval_assembled.LB.name, "leaderboard-hier.md")
        contract = eval_assembled.assembled_board_contract()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = root / "gold.json"
            archive.write_bytes(b"frozen-eval-archive")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            key = evaluate.versioned_row_key("gold", digest)
            visible = f"| {key} | 1 | 1 | 0/32 | 2 | audit |"
            row = evaluate.assembled_leaderboard_row(
                visible, row_key=key, contract=contract,
                archive_path=str(archive), archive_sha256=digest,
                worker_sha256="a" * 64, manager_sha256="b" * 64)
            board = root / "board.md"
            with mock.patch.object(evaluate, "verify_standalone_contract"):
                evaluate.upsert_leaderboard_rows(
                    board, {key: row}, contract=contract,
                    initial_text=eval_assembled.ASSEMBLED_LEADERBOARD_HEADER)
                provenance = evaluate._validate_row_marker(
                    next(line for line in board.read_text().splitlines()
                         if line.startswith(f"| {key} |")), contract)
                self.assertEqual(provenance["kind"], "assembled")
                self.assertEqual(provenance["archive_sha256"], digest)

                archive.write_bytes(b"replacement")
                with self.assertRaisesRegex(RuntimeError, "checkpoint 发生变化"):
                    evaluate.upsert_leaderboard_rows(
                        board, {key: row}, contract=contract,
                        initial_text=eval_assembled.ASSEMBLED_LEADERBOARD_HEADER)

    def test_eval_assembled_import_does_not_preload_native_runtime(self):
        code = f"""
import pathlib, sys
root = pathlib.Path({str(ROOT)!r})
sys.path.insert(0, str(root / 'train'))
sys.path.insert(0, str(root / 'python'))
import eval_assembled
if '_diablogym' in sys.modules or 'diablogym' in sys.modules:
    raise SystemExit('eval_assembled import preloaded native runtime')
"""
        command = [sys.executable]
        if not __debug__:
            command.append("-O")
        completed = subprocess.run(
            [*command, "-c", code], cwd=ROOT, text=True,
            capture_output=True, check=False)
        self.assertEqual(
            completed.returncode, 0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}")

    def test_eval_assembled_main_rejects_a_preloaded_bridge(self):
        with mock.patch.dict(sys.modules, {"_diablogym": object()}), \
                mock.patch.object(
                    sys, "argv", ["eval_assembled.py", "--worker", "script"]):
            with self.assertRaisesRegex(
                    eval_contract.EvalContractError, "未预载.*新进程"):
                eval_assembled.main()

    def test_legacy_sb3_worker_declares_lossless_v3_boundary(self):
        class FakeModel:
            policy = object()

            def __init__(self):
                self.observations = []
                self.masks = []

            def predict(self, obs, **kwargs):
                self.observations.append(obs)
                self.masks.append(kwargs["action_masks"])
                return np.asarray(9), None

        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "legacy.zip"
            checkpoint.write_bytes(b"legacy")
            with mock.patch.object(
                    eval_assembled, "resolve_checkpoint_file",
                    return_value=checkpoint), \
                    mock.patch.object(
                        eval_assembled, "checkpoint_num_timesteps_bytes",
                        return_value=1), \
                    mock.patch(
                        "sb3_contrib.MaskablePPO.load",
                        return_value=model):
                workers, _, _ = eval_assembled.load_worker(
                    str(checkpoint), "a" * 64)

            canonical = np.linspace(
                0.0, 1.0, 298, dtype=np.float32)
            canonical[286] = 3.0 / 8.0
            canonical[296] = 1.0
            canonical[297] = 0.0
            canonical_bytes = canonical.tobytes()
            mask = np.ones(15, dtype=np.bool_)
            self.assertEqual(
                workers[eval_assembled.FARM](canonical, mask), 9)

        callback = workers[eval_assembled.FARM]
        self.assertEqual(
            callback.diablogym_worker_observation_view, "legacy-v3")
        self.assertIs(model.observations[0], canonical)
        self.assertEqual(model.observations[0].tobytes(), canonical_bytes)
        self.assertTrue(mask[12])  # caller-owned live mask is not mutated
        self.assertFalse(model.masks[0][12])
        self.assertTrue(bool(model.masks[0][np.arange(15) != 12].all()))

    def test_rev9_custom_worker_declares_lossless_a12_overlay(self):
        from leashed_ppo import A12MixtureMaskableActorCriticPolicy

        class FakeModel:
            policy = object.__new__(
                A12MixtureMaskableActorCriticPolicy)

            def __init__(self):
                self.observations = []
                self.masks = []

            def predict(self, obs, **kwargs):
                self.observations.append(obs)
                self.masks.append(kwargs["action_masks"])
                return np.asarray(12), None

        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "custom.zip"
            checkpoint.write_bytes(b"custom")
            with mock.patch.object(
                    eval_assembled, "resolve_checkpoint_file",
                    return_value=checkpoint), \
                    mock.patch.object(
                        eval_assembled, "checkpoint_num_timesteps_bytes",
                        return_value=1), \
                    mock.patch(
                        "sb3_contrib.MaskablePPO.load",
                        return_value=model):
                workers, _, _ = eval_assembled.load_worker(
                    str(checkpoint), "a" * 64)

            signed = np.zeros(298, dtype=np.float32)
            signed[286] = 3.0 / 8.0 + 5.0 / 128.0
            signed[296] = 1.0
            signed[297] = -2.25
            signed_bytes = signed.tobytes()
            raw_mask = np.ones(15, dtype=np.bool_)
            action = workers[eval_assembled.FARM](
                signed, raw_mask)

        self.assertEqual(action, 12)
        self.assertEqual(
            workers[eval_assembled.FARM].diablogym_worker_observation_view,
            "legacy-v3-a12-overlay")
        self.assertIs(model.observations[0], signed)
        self.assertEqual(model.observations[0].tobytes(), signed_bytes)
        self.assertEqual(
            model.observations[0][286],
            3.0 / 8.0 + 5.0 / 128.0)
        self.assertEqual(model.observations[0][297], -2.25)
        self.assertIs(model.masks[0], raw_mask)
        self.assertTrue(bool(model.masks[0][12]))

    def test_rev25_and_historical_rev24_asymmetric_workers_load_exact_schema(
            self):
        import gymnasium as gym
        import torch
        from gymnasium import spaces
        from leashed_ppo import (
            ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
            ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
            WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
            AsymmetricWorkerMaskableActorCriticPolicy,
            LeashedMaskablePPO,
            asymmetric_worker_runtime_evidence,
            strict_actor_critic_parameter_partition,
        )
        from diablogym.controller_wire import DUAL_WORKER_LAYOUT_SHA256

        class DualEnv(gym.Env):
            observation_space = spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(DUAL_WORKER_LAYOUT.observation_dim,),
                dtype=np.float32,
            )
            action_space = spaces.Discrete(15)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(
                    DUAL_WORKER_LAYOUT.observation_dim,
                    dtype=np.float32,
                ), {}

            def step(self, action):
                return (
                    np.zeros(
                        DUAL_WORKER_LAYOUT.observation_dim,
                        dtype=np.float32,
                    ),
                    0.0,
                    False,
                    False,
                    {},
                )

            def action_masks(self):
                return np.ones(15, dtype=bool)

        def different_sha256(value):
            prefix = "1" if value.startswith("0") else "0"
            return prefix + value[1:]

        model = LeashedMaskablePPO(
            AsymmetricWorkerMaskableActorCriticPolicy,
            DualEnv(),
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            learning_rate=1e-3,
            seed=37,
            device="cpu",
            verbose=0,
        )
        adapter = model.policy.mlp_extractor.context_adapter
        partition = strict_actor_critic_parameter_partition(
            model.policy, optimizer=model.policy.optimizer)
        with torch.no_grad():
            adapter.output.weight.copy_(torch.eye(
                adapter.output.out_features,
                dtype=adapter.output.weight.dtype,
                device=adapter.output.weight.device,
            ))
            partition["critic"][0].view(-1)[0].add_(0.01)
            model.policy.action_net.bias.zero_()
            model.policy.action_net.bias[7] = 500.0
            model.policy.action_net.bias[12] = 1_000.0
            model.policy.mlp_extractor.enable_actor_context()

        runtime = asymmetric_worker_runtime_evidence(model.policy)
        source_checkpoint_sha256 = "a" * 64
        source_actor_sha256 = "b" * 64
        actor_receipt = {
            "schema": train_ppo._ASYMMETRIC_ACTOR_INIT_SCHEMA,
            "method": train_ppo._ASYMMETRIC_ACTOR_INIT_METHOD,
            "context_architecture":
                train_ppo._ASYMMETRIC_CONTEXT_ARCHITECTURE,
            "controller_layout_schema": DUAL_WORKER_LAYOUT.schema,
            "controller_layout_sha256": DUAL_WORKER_LAYOUT_SHA256,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_actor_sha256": source_actor_sha256,
            "migrated_actor_sha256": different_sha256(
                runtime["policy"]["actor_sha256"]),
            "target_actor_parameter_tensors":
                runtime["policy"]["actor_tensor_count"],
            "target_actor_parameter_count":
                runtime["policy"]["actor_parameter_count"],
            "context_parameter_tensors":
                runtime["context"]["tensor_count"],
            "context_parameter_count":
                runtime["context"]["parameter_count"],
            "context_encoder_sha256": different_sha256(
                runtime["context"]["parameter_groups"]["encoder"]["sha256"]),
            "context_interaction_sha256": different_sha256(
                runtime["context"]["parameter_groups"]["interaction"]["sha256"]),
            "context_output_sha256": different_sha256(
                runtime["context"]["parameter_groups"]["output"]["sha256"]),
        }
        critic_receipt = {
            "schema": train_ppo._ASYMMETRIC_CRITIC_RESET_SCHEMA,
            "method": train_ppo._ASYMMETRIC_CRITIC_RESET_METHOD,
            "critic_architecture":
                train_ppo._ASYMMETRIC_CRITIC_ARCHITECTURE,
            "controller_layout_schema": DUAL_WORKER_LAYOUT.schema,
            "controller_layout_sha256": DUAL_WORKER_LAYOUT_SHA256,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_actor_sha256": source_actor_sha256,
            "actor_parameter_tensors":
                runtime["policy"]["actor_tensor_count"],
            "critic_parameter_tensors":
                runtime["policy"]["critic_tensor_count"],
            "critic_parameter_count":
                runtime["policy"]["critic_parameter_count"],
            "critic_sha256_after": different_sha256(
                runtime["policy"]["critic_sha256"]),
        }
        max_grad_norm = float(
            train_ppo._ALGORITHM_RECIPE["max_grad_norm"])
        clipping = train_ppo._root_context_critic_gradient_clipping(
            max_grad_norm)
        contract_actor_receipt = {
            "method": actor_receipt["method"],
            "context_architecture":
                actor_receipt["context_architecture"],
            "controller_layout_schema":
                actor_receipt["controller_layout_schema"],
            "controller_layout_sha256":
                actor_receipt["controller_layout_sha256"],
            "source_checkpoint_sha256":
                actor_receipt["source_checkpoint_sha256"],
            "source_actor_sha256": actor_receipt["source_actor_sha256"],
            "migrated_actor_sha256":
                actor_receipt["migrated_actor_sha256"],
            "target_actor_parameter_tensors":
                actor_receipt["target_actor_parameter_tensors"],
            "target_actor_parameter_count":
                actor_receipt["target_actor_parameter_count"],
            "context_parameter_tensors":
                actor_receipt["context_parameter_tensors"],
            "context_parameter_count":
                actor_receipt["context_parameter_count"],
            "context_initialization": {
                "hidden": ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
                "output": "exact-zero-disabled-through-critic-warmup",
            },
            "actor_context_excluded_observation_features":
                list(ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES),
        }
        contract_critic_receipt = {
            "method": critic_receipt["method"],
            "critic_architecture": critic_receipt["critic_architecture"],
            "controller_layout_schema":
                critic_receipt["controller_layout_schema"],
            "controller_layout_sha256":
                critic_receipt["controller_layout_sha256"],
            "source_checkpoint_sha256":
                critic_receipt["source_checkpoint_sha256"],
            "source_actor_sha256":
                critic_receipt["source_actor_sha256"],
            "critic_parameter_tensors":
                critic_receipt["critic_parameter_tensors"],
            "critic_parameter_count":
                critic_receipt["critic_parameter_count"],
            "warmup_steps": 8,
            "gradient_clip_mode": clipping["mode"],
            "worker_onpolicy_pg_audit_schema":
                WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout":
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        }
        model.diablogym_contract = {
            "schema_version": 2,
            "contract_revision": train_ppo._CONTRACT_REVISION,
            "mode": "worker",
            "legacy_policy_observation_view": False,
            "worker_policy_observation_view": "dual-v4-asymmetric-v3",
            "worker_episode_boundary":
                train_ppo._WORKER_EPISODE_BOUNDARY_V24,
            "worker_window_bootstrap": "next-learning-window",
            "worker_no_progress_timeout":
                dict(train_ppo._WORKER_NO_PROGRESS_TIMEOUT_CONTRACT),
            "observation_shape": [DUAL_WORKER_LAYOUT.observation_dim],
            "action_n": 15,
            "algorithm_recipe": {"max_grad_norm": max_grad_norm},
            "actor_migration": contract_actor_receipt,
            "critic_migration": contract_critic_receipt,
            "gradient_clipping": clipping,
            "distill_beta": 0.0,
            "distillation": {
                "initial_beta": 0.0,
                "scope": "full-policy-logits",
                "excluded_actions": [],
                "anneal_actor_rollouts": 0,
                "schedule": "constant",
            },
            "worker_action14_logit_bonus": 0.0,
            "drink_sovereignty": False,
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
        }
        model._actor_migration_receipt = actor_receipt
        model._critic_migration_receipt = critic_receipt
        model.distill_beta = 0.0
        model.distill_anneal_actor_rollouts = 0
        model.num_timesteps = 24
        model._last_completed_ppo_rollout_steps = 24
        model._ppo_optimizer_steps_completed = 6
        model._critic_warmup_start_timesteps = 0
        model._critic_warmup_until_timesteps = 8
        model._critic_warmup_expected_rollouts = 2
        model._critic_warmup_rollouts_completed = 2
        model._critic_warmup_optimizer_steps_completed = 2
        model._critic_warmup_completed = True
        model._actor_optimizer_steps_completed = 4
        self.assertTrue(
            train_ppo._asymmetric_worker_deployment_evidence_complete(model))

        observation = np.zeros(
            DUAL_WORKER_LAYOUT.observation_dim, dtype=np.float32)
        raw_mask = np.ones(15, dtype=bool)
        unmasked_action, _ = model.predict(
            observation, action_masks=raw_mask, deterministic=True)
        self.assertEqual(int(unmasked_action), 12)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "current-rev25.zip"
            model.save(checkpoint)
            with mock.patch.object(
                    eval_assembled,
                    "_validate_asymmetric_worker_runtime_state",
                    wraps=eval_assembled
                    ._validate_asymmetric_worker_runtime_state,
            ) as runtime_validation:
                workers, label, identity = eval_assembled.load_worker(
                    str(checkpoint), "a" * 64)
            runtime_validation.assert_called_once_with(
                mock.ANY, contract_revision=25)
            callback = workers[eval_assembled.FARM]
            action = callback(observation, raw_mask)

            # Rev24 is still a valid historical evaluation contract, but its
            # formal PG claim was /8.  It must neither inherit rev25's /9
            # claim nor be rejected merely because the trainer moved on.
            # historical rev23 checkpoint must keep its old boundary string
            # and must not be retroactively required to carry either field.
            current_contract = copy.deepcopy(
                model.diablogym_contract)
            historical_rev24_contract = copy.deepcopy(current_contract)
            historical_rev24_contract["contract_revision"] = 24
            historical_rev24_contract["critic_migration"][
                "worker_onpolicy_pg_audit_schema"
            ] = "diablogym-worker-onpolicy-pg/8"
            model.diablogym_contract = historical_rev24_contract
            historical_rev24_checkpoint = (
                pathlib.Path(directory) / "historical-rev24.zip")
            model.save(historical_rev24_checkpoint)
            _, historical_rev24_label, _ = eval_assembled.load_worker(
                str(historical_rev24_checkpoint), "a" * 64)
            self.assertEqual(
                historical_rev24_label, "historical-rev24")

            forged_rev24 = copy.deepcopy(historical_rev24_contract)
            forged_rev24["critic_migration"][
                "worker_onpolicy_pg_audit_schema"
            ] = "diablogym-worker-onpolicy-pg/9"
            model.diablogym_contract = forged_rev24
            forged_rev24_checkpoint = (
                pathlib.Path(directory) / "forged-rev24-as-rev25.zip")
            model.save(forged_rev24_checkpoint)
            with self.assertRaisesRegex(
                    eval_contract.EvalContractError,
                    "rev24 actor/critic/layout/PG"):
                eval_assembled.load_worker(
                    str(forged_rev24_checkpoint), "a" * 64)

            forged_rev25 = copy.deepcopy(current_contract)
            forged_rev25["critic_migration"][
                "worker_onpolicy_pg_audit_schema"
            ] = "diablogym-worker-onpolicy-pg/8"
            model.diablogym_contract = forged_rev25
            forged_rev25_checkpoint = (
                pathlib.Path(directory) / "forged-rev25-as-rev24.zip")
            model.save(forged_rev25_checkpoint)
            with self.assertRaisesRegex(
                    eval_contract.EvalContractError,
                    "rev25 actor/critic/layout/PG"):
                eval_assembled.load_worker(
                    str(forged_rev25_checkpoint), "a" * 64)

            historical_contract = copy.deepcopy(current_contract)
            historical_contract["contract_revision"] = 23
            historical_contract["worker_episode_boundary"] = (
                "base-game-terminal-only")
            historical_contract.pop("worker_no_progress_timeout")
            historical_contract.pop("policy_source_roles")
            model.diablogym_contract = historical_contract
            historical_checkpoint = (
                pathlib.Path(directory) / "historical-rev23.zip")
            model.save(historical_checkpoint)
            _, historical_label, _ = eval_assembled.load_worker(
                str(historical_checkpoint), "a" * 64)
            self.assertEqual(
                historical_label, "historical-rev23")
            model.diablogym_contract = current_contract

            # A syntactically valid contract bonus is not enough: the live
            # policy must have been reconstructed with the exact same logit
            # transform.  Otherwise formal evaluation would exercise a
            # different action-14 distribution than the registered run.
            model.diablogym_contract[
                "worker_action14_logit_bonus"] = 1.0
            model.diablogym_contract["policy_source_roles"][
                "worker_action14_policy_sources"
            ] = [
                "fixed-logit-prior",
                "native-reward-bound-on-policy-ppo",
            ]
            mismatched_checkpoint = (
                pathlib.Path(directory) / "rev25-bonus-mismatch.zip")
            model.save(mismatched_checkpoint)
            with self.assertRaisesRegex(
                    eval_contract.EvalContractError,
                    "action14 logit bonus"):
                eval_assembled.load_worker(
                    str(mismatched_checkpoint), "a" * 64)

        self.assertEqual(label, "current-rev25")
        self.assertEqual(identity["num_timesteps"], 24)
        self.assertEqual(DUAL_WORKER_LAYOUT.observation_dim, 13_012)
        self.assertEqual(
            callback.diablogym_worker_observation_view,
            "dual-v4-asymmetric-v3")
        self.assertEqual(
            callback.diablogym_worker_action12_mode,
            "permanently-masked")
        self.assertEqual(action, 7)
        self.assertTrue(bool(raw_mask[12]))

    def test_historical_asymmetric_workers_fail_closed_without_wire_builder(self):
        from leashed_ppo import (
            AsymmetricWorkerMaskableActorCriticPolicy,
        )

        class FakeModel:
            def __init__(self, revision):
                self.diablogym_contract = {
                    "contract_revision": revision,
                }
                self.policy = object.__new__(
                    AsymmetricWorkerMaskableActorCriticPolicy)
                self.predict_calls = 0

            def predict(self, obs, **kwargs):
                self.predict_calls += 1
                return np.asarray(9), None

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "dual.zip"
            checkpoint.write_bytes(b"dual")
            for revision in (17, 18, 19, 20, 21):
                model = FakeModel(revision)
                with self.subTest(revision=revision), \
                        mock.patch.object(
                            eval_assembled, "resolve_checkpoint_file",
                            return_value=checkpoint), \
                        mock.patch.object(
                            eval_assembled,
                            "checkpoint_num_timesteps_bytes",
                            return_value=1), \
                        mock.patch(
                            "sb3_contrib.MaskablePPO.load",
                            return_value=model), \
                        self.assertRaisesRegex(
                            eval_contract.EvalContractError,
                            "historical asymmetric Worker wire constructor "
                            "unavailable"):
                    eval_assembled.load_worker(
                        str(checkpoint), "a" * 64)
                self.assertEqual(model.predict_calls, 0)

    def test_asymmetric_historical_dimensions_are_not_reinterpreted(self):
        helper = (
            eval_assembled
            ._asymmetric_worker_observation_dim_for_revision)
        current = DUAL_WORKER_LAYOUT.observation_dim
        self.assertEqual(helper(17, current), 635)
        self.assertEqual(helper(18, current), 635)
        self.assertEqual(helper(19, current), 5_448)
        self.assertEqual(helper(20, current), 9_100)
        self.assertEqual(helper(21, current), 9_100)
        self.assertEqual(helper(22, current), current)
        with self.assertRaises(eval_contract.EvalContractError):
            helper(22, 0)

    def test_rev18_asymmetric_worker_rejects_incomplete_runtime_state(self):
        import torch

        def complete_model():
            context = torch.full((64, 337), 0.125)
            return types.SimpleNamespace(
                num_timesteps=24,
                _last_completed_ppo_rollout_steps=24,
                _ppo_optimizer_steps_completed=6,
                _critic_warmup_start_timesteps=0,
                _critic_warmup_until_timesteps=8,
                _critic_warmup_expected_rollouts=2,
                _critic_warmup_rollouts_completed=2,
                _critic_warmup_optimizer_steps_completed=2,
                _critic_warmup_completed=True,
                _actor_optimizer_steps_completed=4,
                policy=types.SimpleNamespace(
                    mlp_extractor=types.SimpleNamespace(
                        actor_context_enabled=True,
                        context_adapter=types.SimpleNamespace(
                            weight=context)),
                    action_net=types.SimpleNamespace(
                        weight=torch.full((15, 64), 0.25)),
                ),
            )

        mutations = (
            ("warmup incomplete",
             lambda model: setattr(
                 model, "_critic_warmup_completed", False)),
            ("actor never updated",
             lambda model: setattr(
                 model, "_actor_optimizer_steps_completed", 0)),
            ("stale rollout receipt",
             lambda model: setattr(
                 model, "_last_completed_ppo_rollout_steps", 16)),
            ("context disabled",
             lambda model: setattr(
                 model.policy.mlp_extractor,
                 "actor_context_enabled", False)),
            ("context zero",
             lambda model: model.policy.mlp_extractor.context_adapter
             .weight.zero_()),
        )
        for label, mutate in mutations:
            model = complete_model()
            mutate(model)
            with self.subTest(label=label), self.assertRaisesRegex(
                    eval_contract.EvalContractError,
                    "尚未完成可部署 actor 训练|actor context"):
                eval_assembled._validate_asymmetric_worker_runtime_state(
                    model, contract_revision=18)

    def test_npz_worker_uses_strict_worker_callback(self):
        calls = []

        class FakeManager:
            source_sha256 = "f" * 64

            def __init__(self, path):
                calls.append(("init", pathlib.Path(path)))

            def require_worker_contract(self):
                calls.append(("contract",))
                return {"observation_view": "legacy-v3"}

            def worker_callback(self):
                calls.append(("callback",))

                def callback(obs, mask):
                    return 7

                callback.diablogym_worker_observation_view = "legacy-v3"
                return callback

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "worker.npz"
            with mock.patch.object(
                    eval_assembled,
                    "_native_runtime",
                    return_value=(FakeManager, object(), object())):
                workers, _, identity = eval_assembled.load_worker(
                    str(path), "a" * 64)
        self.assertEqual(
            calls,
            [("init", path.resolve()), ("contract",), ("callback",)])
        self.assertEqual(identity["sha256"], "f" * 64)
        self.assertEqual(
            workers[eval_assembled.FARM].diablogym_worker_observation_view,
            "legacy-v3")
        self.assertEqual(
            workers[eval_assembled.FARM](
                np.zeros(298, dtype=np.float32),
                np.ones(15, dtype=bool)),
            7)

    def test_published_worker_requires_exact_campaign_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(pathlib.Path(directory))
            expected, identity = (
                eval_assembled.capture_publication_expectations(
                    bundle["expectations"]))
            with mock.patch.object(
                    train_ppo, "bc_aux_behavior_gate",
                    return_value=bundle["gate"]), \
                    mock.patch.object(
                        eval_assembled,
                        "_recompute_published_worker_evidence",
                        return_value=_independent_fixture_evidence(bundle)):
                payload = eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(),
                hashlib.sha256(bundle["receipt"].read_bytes()).hexdigest())
            eval_assembled.verify_publication_expectations(identity)

            # 内部回执仍自洽也不够：短腿/错终点不得命中正式本案。
            wrong = dict(expected, target_global_steps=32_768)
            with mock.patch.object(
                    train_ppo, "bc_aux_behavior_gate",
                    return_value=bundle["gate"]), \
                    mock.patch.object(
                        eval_assembled,
                        "_recompute_published_worker_evidence",
                        return_value=_independent_fixture_evidence(bundle)), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError, "预注册本案"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=wrong)

    def test_published_worker_rejects_invalid_terminal_death_cost_contract(self):
        for invalid in (-0.25, False, "not-a-number"):
            with self.subTest(invalid=invalid), \
                    tempfile.TemporaryDirectory() as directory:
                bundle = _published_worker_fixture(pathlib.Path(directory))
                expected, _ = (
                    eval_assembled.capture_publication_expectations(
                        bundle["expectations"]))
                with zipfile.ZipFile(bundle["checkpoint"], "r") as archive:
                    checkpoint_data = json.loads(archive.read("data"))
                checkpoint_data["diablogym_contract"][
                    "worker_additional_terminal_death_cost"] = invalid
                with zipfile.ZipFile(bundle["checkpoint"], "w") as archive:
                    archive.writestr(
                        "data",
                        json.dumps(
                            checkpoint_data, sort_keys=True, allow_nan=False))
                receipt = json.loads(bundle["receipt"].read_text())
                receipt["model_sha256"] = hashlib.sha256(
                    bundle["checkpoint"].read_bytes()).hexdigest()
                bundle["receipt"].write_text(
                    json.dumps(receipt, sort_keys=True))
                with mock.patch.object(
                        eval_assembled,
                        "_recompute_published_worker_evidence",
                        return_value=_independent_fixture_evidence(bundle)), \
                        mock.patch.object(
                            train_ppo, "bc_aux_behavior_gate",
                            return_value=bundle["gate"]), \
                        self.assertRaisesRegex(
                            eval_contract.EvalContractError,
                            "终局奖励契约"):
                    eval_assembled.capture_published_worker(
                        bundle["checkpoint"],
                        bundle["checkpoint"].read_bytes(),
                        expected_manager_sha256=bundle["manager_sha"],
                        expected_implementation_sha256=(
                            bundle["implementation_sha"]),
                        expected_provenance=expected)

    def test_published_worker_rejects_receipt_or_preflight_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(pathlib.Path(directory))
            expected, _ = eval_assembled.capture_publication_expectations(
                bundle["expectations"])
            receipt = json.loads(bundle["receipt"].read_text())
            receipt["model_sha256"] = "f" * 64
            bundle["receipt"].write_text(json.dumps(receipt))
            with mock.patch.object(
                    train_ppo, "bc_aux_behavior_gate",
                    return_value=bundle["gate"]), \
                    mock.patch.object(
                        eval_assembled,
                        "_recompute_published_worker_evidence",
                        return_value=_independent_fixture_evidence(bundle)), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "未绑定当前 checkpoint"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

            bundle = _published_worker_fixture(pathlib.Path(directory))
            bundle["preflight"].write_text('{"status":"PASS"}')
            with mock.patch.object(
                    train_ppo, "bc_aux_behavior_gate",
                    return_value=bundle["gate"]), \
                    mock.patch.object(
                        eval_assembled,
                        "_recompute_published_worker_evidence",
                        return_value=_independent_fixture_evidence(bundle)), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "liveness 回执 SHA"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

    def test_published_worker_recomputes_checkpoint_evidence_and_safety_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(pathlib.Path(directory))
            expected, _ = eval_assembled.capture_publication_expectations(
                bundle["expectations"])
            observed = _independent_fixture_evidence(bundle)
            forged = copy.deepcopy(observed)
            forged["metrics"] = {"fixture": False}
            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=forged), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        return_value=bundle["gate"]), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "checkpoint\\+demos 现场重算"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

            calls = []

            def gate_probe(_metrics, **kwargs):
                calls.append(kwargs)
                return bundle["gate"]

            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=observed), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        side_effect=gate_probe):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)
            expected_gate_call = {
                "require_root_anchor": True,
                "require_teacher_recall": False,
                "require_deployable_a12": False,
            }
            self.assertEqual(calls, [expected_gate_call, expected_gate_call])

    def test_published_worker_requires_explicit_non_deployable_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(pathlib.Path(directory))
            expected, _ = eval_assembled.capture_publication_expectations(
                bundle["expectations"])
            receipt = json.loads(bundle["receipt"].read_text())
            receipt["gate"]["thresholds"][
                "deployable_a12_required"] = True
            bundle["receipt"].write_text(json.dumps(receipt, sort_keys=True))
            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=_independent_fixture_evidence(bundle)), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        return_value=bundle["gate"]), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "deterministic a12 非发布门"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

    def test_published_worker_rejects_bool_low_threshold_and_count_nonclosure(self):
        mutations = (
            ("bool count", lambda row: row["exploration_evidence"].update(
                eligible_states=True)),
            ("bool requested", lambda row: row["exploration_evidence"].update(
                requested_a12=True)),
            ("bool rejected", lambda row: row["exploration_evidence"].update(
                rejected_a12=False)),
            ("fractional rejected",
             lambda row: row["exploration_evidence"].update(
                 rejected_a12=0.5)),
            ("request closure", lambda row: row["exploration_evidence"].update(
                requested_a12=11)),
            ("lowered expected threshold",
             lambda row: row["exploration_evidence"].update(
                 minimum_expected_a12_mass=0.0)),
            ("lowered actual threshold",
             lambda row: row["exploration_evidence"].update(
                 minimum_actual_a12_samples=0)),
            ("sample count exceeds eligible",
             lambda row: row["exploration_evidence"].update(
                 eligible_states=9, sampled_a12=10)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), \
                    tempfile.TemporaryDirectory() as directory:
                bundle = _published_worker_fixture(pathlib.Path(directory))
                expected, _ = (
                    eval_assembled.capture_publication_expectations(
                        bundle["expectations"]))
                receipt = json.loads(bundle["receipt"].read_text())
                mutate(receipt)
                bundle["receipt"].write_text(json.dumps(receipt))
                with mock.patch.object(
                        eval_assembled,
                        "_recompute_published_worker_evidence",
                        return_value=_independent_fixture_evidence(bundle)), \
                        mock.patch.object(
                            train_ppo, "bc_aux_behavior_gate",
                            return_value=bundle["gate"]), \
                        self.assertRaisesRegex(
                            eval_contract.EvalContractError,
                            "探索样本证据"):
                    eval_assembled.capture_published_worker(
                        bundle["checkpoint"],
                        bundle["checkpoint"].read_bytes(),
                        expected_manager_sha256=bundle["manager_sha"],
                        expected_implementation_sha256=(
                            bundle["implementation_sha"]),
                        expected_provenance=expected)

    def test_published_worker_binds_request_and_rejection_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(pathlib.Path(directory))
            expected, _ = eval_assembled.capture_publication_expectations(
                bundle["expectations"])
            receipt = json.loads(bundle["receipt"].read_text())
            evidence = receipt["exploration_evidence"]
            evidence["requested_a12"] = 11
            evidence["rejected_a12"] = 1
            # Receipt closure remains internally valid (11 = 10 + 1), but
            # checkpoint runtime state still says 10 requested / 0 rejected.
            bundle["receipt"].write_text(json.dumps(
                receipt, sort_keys=True))
            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=_independent_fixture_evidence(bundle)), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        return_value=bundle["gate"]), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "checkpoint 运行态"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

    def test_published_worker_rejects_missing_or_stale_ppo_update_receipt(self):
        for field, value in (
                ("_ppo_optimizer_steps_completed", None),
                ("_ppo_optimizer_steps_completed", 0),
                ("_last_completed_ppo_rollout_steps", None),
                ("_last_completed_ppo_rollout_steps", 1024)):
            with self.subTest(field=field, value=value), \
                    tempfile.TemporaryDirectory() as directory:
                bundle = _published_worker_fixture(pathlib.Path(directory))
                expected, _ = eval_assembled.capture_publication_expectations(
                    bundle["expectations"])
                with zipfile.ZipFile(bundle["checkpoint"], "r") as archive:
                    checkpoint_data = json.loads(archive.read("data"))
                if value is None:
                    checkpoint_data.pop(field)
                else:
                    checkpoint_data[field] = value
                with zipfile.ZipFile(bundle["checkpoint"], "w") as archive:
                    archive.writestr(
                        "data", json.dumps(
                            checkpoint_data, sort_keys=True,
                            allow_nan=False))
                receipt = json.loads(bundle["receipt"].read_text())
                receipt["model_sha256"] = hashlib.sha256(
                    bundle["checkpoint"].read_bytes()).hexdigest()
                bundle["receipt"].write_text(
                    json.dumps(receipt, sort_keys=True))
                with mock.patch.object(
                        eval_assembled,
                        "_recompute_published_worker_evidence",
                        return_value=_independent_fixture_evidence(bundle)), \
                        mock.patch.object(
                            train_ppo, "bc_aux_behavior_gate",
                            return_value=bundle["gate"]), \
                        self.assertRaisesRegex(
                            eval_contract.EvalContractError,
                            "checkpoint 运行态"):
                    eval_assembled.capture_published_worker(
                        bundle["checkpoint"],
                        bundle["checkpoint"].read_bytes(),
                        expected_manager_sha256=bundle["manager_sha"],
                        expected_implementation_sha256=(
                            bundle["implementation_sha"]),
                        expected_provenance=expected)

    def test_published_worker_requires_exact_checkpoint_expected_mass(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(pathlib.Path(directory))
            expected, _ = eval_assembled.capture_publication_expectations(
                bundle["expectations"])
            with zipfile.ZipFile(bundle["checkpoint"], "r") as archive:
                checkpoint_data = json.loads(archive.read("data"))
            # This differs by less than the former 1e-9 tolerance but crosses
            # the registered minimum.  Receipt and checkpoint are the same
            # evidence source, so any byte-level numeric disagreement fails.
            checkpoint_data["_bc_aux_expected_a12_mass"] = 19.9999999995
            with zipfile.ZipFile(bundle["checkpoint"], "w") as archive:
                archive.writestr("data", json.dumps(
                    checkpoint_data, sort_keys=True, allow_nan=False))
            receipt = json.loads(bundle["receipt"].read_text())
            receipt["model_sha256"] = hashlib.sha256(
                bundle["checkpoint"].read_bytes()).hexdigest()
            bundle["receipt"].write_text(json.dumps(
                receipt, sort_keys=True))
            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=_independent_fixture_evidence(bundle)), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        return_value=bundle["gate"]), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "checkpoint 运行态"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

    def test_published_worker_accepts_preserved_continuation_liveness(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(
                pathlib.Path(directory),
                installation="preserved-continuation")
            expected, _ = eval_assembled.capture_publication_expectations(
                bundle["expectations"])
            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=_independent_fixture_evidence(bundle)), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        return_value=bundle["gate"]):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

            preflight = json.loads(bundle["preflight"].read_text())
            preflight["calls"]["initial_adapter_calibrations"] = 1
            bundle["preflight"].write_text(json.dumps(
                preflight, sort_keys=True))
            receipt = json.loads(bundle["receipt"].read_text())
            receipt["provenance"]["bc_aux_liveness_preflight_sha256"] = (
                hashlib.sha256(bundle["preflight"].read_bytes()).hexdigest())
            bundle["receipt"].write_text(json.dumps(receipt, sort_keys=True))
            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=_independent_fixture_evidence(bundle)), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        return_value=bundle["gate"]), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "liveness 调用账"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

    def test_published_worker_rejects_missing_exact_rev9_policy_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = _published_worker_fixture(pathlib.Path(directory))
            expected, _ = eval_assembled.capture_publication_expectations(
                bundle["expectations"])
            with zipfile.ZipFile(bundle["checkpoint"], "r") as archive:
                checkpoint_data = json.loads(archive.read("data"))
            checkpoint_data["policy_kwargs"]["bc_aux_mixture_spec"] = None
            with zipfile.ZipFile(bundle["checkpoint"], "w") as archive:
                archive.writestr("data", json.dumps(
                    checkpoint_data, sort_keys=True, allow_nan=False))
            receipt = json.loads(bundle["receipt"].read_text())
            receipt["model_sha256"] = hashlib.sha256(
                bundle["checkpoint"].read_bytes()).hexdigest()
            bundle["receipt"].write_text(json.dumps(receipt, sort_keys=True))
            with mock.patch.object(
                    eval_assembled,
                    "_recompute_published_worker_evidence",
                    return_value=_independent_fixture_evidence(bundle)), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_gate",
                        return_value=bundle["gate"]), \
                    self.assertRaisesRegex(
                        eval_contract.EvalContractError,
                        "custom policy/spec"):
                eval_assembled.capture_published_worker(
                    bundle["checkpoint"],
                    bundle["checkpoint"].read_bytes(),
                    expected_manager_sha256=bundle["manager_sha"],
                    expected_implementation_sha256=(
                        bundle["implementation_sha"]),
                    expected_provenance=expected)

    def test_independent_evidence_loads_exact_custom_policy_and_hashes_heads(self):
        import gymnasium as gym
        import numpy as np
        from stable_baselines3.common.vec_env import DummyVecEnv
        from leashed_ppo import LeashedMaskablePPO

        class TinyEnv(gym.Env):
            observation_space = gym.spaces.Box(
                -2.0, 2.0, shape=(298,), dtype=np.float32)
            action_space = gym.spaces.Discrete(15)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(298, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(298, dtype=np.float32),
                    0.0, False, True, {})

            def action_masks(self):
                return np.ones(15, dtype=np.bool_)

        with tempfile.TemporaryDirectory() as directory:
            env = DummyVecEnv([TinyEnv])
            self.addCleanup(env.close)
            model = LeashedMaskablePPO(
                "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
                gamma=1.0, distill_beta=0.0, seed=7,
                device="cpu", verbose=0)
            root_head = train_ppo._persistent_bc_aux_root_anchor(model)
            train_ppo._expand_policy_with_bc_aux_circuit(model)
            train_ppo._reset_policy_optimizer(model, 3e-4)
            checkpoint = pathlib.Path(directory) / "worker.zip"
            model.save(checkpoint)
            payload = checkpoint.read_bytes()
            x = np.zeros((4, 298), dtype=np.float32)
            y = np.asarray([12, 9, 12, 9], dtype=np.int64)
            groups = np.arange(4, dtype=np.int64)
            masks = np.ones((4, 15), dtype=np.bool_)
            expected_metrics = {"independently_recomputed": True}
            with mock.patch.object(
                    eval_assembled,
                    "_canonical_bc_aux_demos_path",
                    return_value=pathlib.Path(directory) / "demos.npz"), \
                    mock.patch.object(
                        train_ppo, "_load_bc_aux_demos_v2",
                        return_value=(
                            x, y, groups, masks, "a" * 64)), \
                    mock.patch.object(
                        train_ppo, "bc_aux_behavior_metrics",
                        return_value=expected_metrics):
                observed = (
                    eval_assembled._recompute_published_worker_evidence(
                        payload,
                        demos_sha256="a" * 64,
                        expected_manager_sha256="b" * 64))
            self.assertEqual(observed["metrics"], expected_metrics)
            self.assertEqual(
                observed["anchor_policy_head_sha256"],
                train_ppo._policy_head_sha256(root_head))
            self.assertEqual(
                observed["candidate_policy_head_sha256"],
                train_ppo._policy_head_sha256(
                    train_ppo._policy_head_snapshot(model.policy)))

    def test_sb3_archive_can_bind_published_receipt_sha(self):
        document = _valid_v5_archive()
        document["meta"]["worker"]["gate_report_sha256"] = "9" * 64
        eval_contract.validate_eval_archive(document)

    def test_eval_assembled_closes_environment_on_manager_failure(self):
        class FailingManager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                raise RuntimeError("injected manager failure")

        class FakeEnv:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.closed = False
                type(self).instances.append(self)

            def reset(self, *, seed):
                return [seed], {}

            def action_masks(self):
                return [True]

            def close(self):
                self.closed = True

        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(FailingManager, FakeEnv, object())):
            with self.assertRaisesRegex(RuntimeError, "manager failure"):
                eval_assembled.evaluate(None, [7], manager_npz="manager.npz",
                                        manager_sha256="a" * 64,
                                        manager_policy_observation_view=(
                                            "legacy-v3"))
        self.assertTrue(FakeEnv.instances[-1].closed)

    def test_eval_assembled_closes_on_worker_failure_and_breaks_env_cycle(self):
        class Manager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                return eval_assembled.FARM

        class WorkerEnv:
            instances = []

            def __init__(self, *_args, workers, **_kwargs):
                self.closed = False
                self._workers = workers
                self.drink_sovereignty = _kwargs.get(
                    "drink_sovereignty")
                self.worker_observation_view = _kwargs.get(
                    "worker_observation_view")
                self.env = types.SimpleNamespace(
                    _raw={}, action_masks=lambda: [True] * 15)
                type(self).instances.append(self)

            def reset(self, *, seed):
                return [seed], {}

            def action_masks(self):
                return [True] * 3

            def step(self, _option):
                wrapped = self._workers[eval_assembled.FARM]
                self.instrumented_contract = (
                    getattr(
                        wrapped,
                        "diablogym_worker_observation_view",
                        None),
                    getattr(
                        wrapped,
                        "diablogym_worker_action12_mode",
                        None),
                )
                wrapped([0.0], [True] * 15)
                raise AssertionError("worker failure should have propagated")

            def close(self):
                self.closed = True

        def failing_worker(_obs, _mask):
            raise RuntimeError("injected worker predict failure")

        failing_worker.diablogym_worker_observation_view = (
            "dual-v4-asymmetric-v3")
        failing_worker.diablogym_worker_action12_mode = "permanently-masked"
        caller_workers = {eval_assembled.FARM: failing_worker}
        fake_package = types.ModuleType("diablogym")
        fake_package.__path__ = []
        fake_options = types.ModuleType("diablogym.options_env")
        fake_options.dispatch = lambda *_args: 0
        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(Manager, WorkerEnv, object())), \
                mock.patch.dict(sys.modules, {
                    "diablogym": fake_package,
                    "diablogym.options_env": fake_options,
                }):
            with self.assertRaisesRegex(RuntimeError, "worker predict failure"):
                eval_assembled.evaluate(
                    caller_workers, [7], manager_npz="manager.npz",
                    manager_sha256="a" * 64,
                    manager_policy_observation_view="legacy-v3")
        instance = WorkerEnv.instances[-1]
        self.assertTrue(instance.closed)
        self.assertEqual(
            instance.instrumented_contract,
            ("dual-v4-asymmetric-v3", "permanently-masked"))
        self.assertEqual(
            instance.worker_observation_view,
            "dual-v4-asymmetric-v3")
        self.assertIs(instance.drink_sovereignty, False)
        self.assertEqual(instance._workers, {})
        self.assertIs(caller_workers[eval_assembled.FARM], failing_worker)

    def test_eval_assembled_closes_environment_after_success(self):
        class Manager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                return eval_assembled.FARM

        class SuccessfulEnv:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.closed = False
                self.env = types.SimpleNamespace(
                    _raw={
                        "dungeon_level": 2, "dead": False,
                        "victory": False, "game_over": True,
                        "belt_heals": 2,
                    },
                    _ep_kills=3, _steps=1)
                type(self).instances.append(self)

            def reset(self, *, seed):
                return [seed], {}

            def action_masks(self):
                return [True] * 3

            def step(self, _option):
                return [0.0], 1.5, True, False, {"option_extra": {
                    "beats": 1, "overrides": 0, "reason": "done",
                    "opt": eval_assembled.FARM, "tau": 1, "mode_seq": "F",
                    "R": 1.5, "W": 1.5, "bonus": 0.0,
                    "worker_wage": 0.0, "kills_delta": 3,
                    "worker_kills": 0,
                    "dry": True,
                    "voluntary_drinks": 1,
                    "drain_attempts": 1, "drains": 0,
                }}

            def close(self):
                self.closed = True

        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(Manager, SuccessfulEnv, object())):
            rows, engage = eval_assembled.evaluate(
                None, [7], manager_npz="manager.npz",
                manager_sha256="a" * 64,
                manager_policy_observation_view="legacy-v3")
        self.assertIsNone(engage)
        self.assertEqual(rows[0]["ret"], 1.5)
        self.assertEqual(rows[0]["farm_r"], 1.5)
        self.assertEqual(rows[0]["farm_w"], 1.5)
        self.assertEqual(rows[0]["farm_kills"], 3)
        self.assertEqual(rows[0]["farm_dry_n"], 1)
        self.assertEqual(rows[0]["farm_fresh_n"], 0)
        self.assertEqual(rows[0]["farm_dry_worker_wage"], 0.0)
        self.assertEqual(rows[0]["farm_fresh_worker_wage"], 0.0)
        self.assertEqual(rows[0]["farm_dry_worker_kills"], 0)
        self.assertEqual(rows[0]["farm_fresh_worker_kills"], 0)
        self.assertEqual(rows[0]["nonfarm_r"], 0.0)
        self.assertEqual(rows[0]["farm_voluntary_drinks"], 1)
        self.assertEqual(rows[0]["farm_reflex_drain_attempts"], 1)
        self.assertEqual(rows[0]["farm_reflex_drains"], 0)
        self.assertEqual(rows[0]["farm_multi_drink_windows"], 0)
        self.assertEqual(rows[0]["farm_max_voluntary_drinks_per_window"], 1)
        self.assertEqual(rows[0]["ending_belt_heals"], 2)
        self.assertEqual(rows[0]["terminal_kind"], "game_over")
        self.assertTrue(SuccessfulEnv.instances[-1].closed)

    def test_eval_assembled_audits_action14_opportunity_request_and_growth(self):
        class Manager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                return eval_assembled.FARM

        class NativeState:
            def __init__(self):
                self._raw = {
                    "dungeon_level": 1,
                    "dead": False,
                    "victory": False,
                    "game_over": False,
                    "belt_heals": 0,
                    "belt_free_slots": 8,
                    "hp": 64,
                    "max_hp": 64,
                    "player_x": 40,
                    "player_y": 40,
                    "monsters": [],
                    "floor_items": [],
                    "progression_targets": [],
                    "gear_combat_utility": 100,
                }
                self._ep_kills = 0
                self._steps = 0
                self.mask = np.ones(15, dtype=bool)

            def controller_action_context(self):
                return self.mask.copy(), None

        class GearAuditEnv:
            def __init__(self, *_args, workers, **_kwargs):
                self._workers = workers
                self.env = NativeState()
                self.seed = None

            def reset(self, *, seed):
                self.seed = seed
                self.env._steps = 0
                self.env._raw["game_over"] = False
                self.env._raw["gear_combat_utility"] = 100
                return [seed], {}

            def action_masks(self):
                return [True, False, False]

            def step(self, _option):
                wrapped = self._workers[eval_assembled.FARM]
                first = np.ones(15, dtype=bool)
                self.env.mask = first
                self.asserted_first = wrapped([0.0], first)
                if self.asserted_first != 14:
                    raise AssertionError(self.asserted_first)
                if self.seed == 7:
                    # The synchronous native receipt says +37, but the later
                    # callback endpoint has fallen below the pre-commit value.
                    # Endpoint net delta must not erase the real success.
                    self.env._raw["gear_combat_utility"] = 63
                    second = np.ones(15, dtype=bool)
                    second[14] = False
                    self.env.mask = second
                    self.asserted_second = wrapped([0.0], second)
                    if self.asserted_second != 9:
                        raise AssertionError(self.asserted_second)
                    requests, successes, delta, steps = 1, 1, 37, 2
                else:
                    # Endpoint utility rises without a native accepted
                    # receipt.  This must remain a failed request.
                    self.env._raw["gear_combat_utility"] = 137
                    requests, successes, delta, steps = 1, 0, 0, 1

                self.env._steps = steps
                self.env._raw["game_over"] = True
                return [0.0], 0.0, True, False, {
                    "option_extra": {
                        "beats": steps,
                        "overrides": 0,
                        "reason": "end",
                        "opt": eval_assembled.FARM,
                        "tau": steps,
                        "mode_seq": "F",
                        "R": 0.0,
                        "W": 0.0,
                        "bonus": 0.0,
                        "worker_wage": 0.0,
                        "kills_delta": 0,
                        "worker_kills": 0,
                        "dry": False,
                        "voluntary_drinks": 0,
                        "drain_attempts": 0,
                        "drains": 0,
                        "worker_action14_requests": requests,
                        "worker_action14_native_successes": successes,
                        "worker_action14_gear_utility_delta": delta,
                    },
                }

            def close(self):
                pass

        actions = iter((14, 9, 14))

        def worker(_obs, _mask):
            return next(actions)

        worker.diablogym_worker_observation_view = (
            "dual-v4-asymmetric-v3")
        worker.diablogym_worker_action12_mode = "permanently-masked"
        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(Manager, GearAuditEnv, object())):
            rows, engage = eval_assembled.evaluate(
                {eval_assembled.FARM: worker},
                [7, 8],
                manager_npz="manager.npz",
                manager_sha256="a" * 64,
                manager_policy_observation_view="legacy-v3",
            )
        self.assertTrue(all(
            row["terminal_kind"] == "game_over" for row in rows))
        self.assertEqual(engage["calls"], 3)
        self.assertEqual(engage["hist"], {14: 2, 9: 1})
        self.assertEqual(engage["action14_mask_opportunities"], 2)
        self.assertEqual(engage["action14_requests"], 2)
        self.assertEqual(engage["action14_native_successes"], 1)
        self.assertEqual(engage["action14_gear_utility_delta"], 37)
        self.assertNotIn("_action14_receipt_requests", engage)

    def test_eval_assembled_partitions_dry_and_fresh_farm_windows(self):
        class Manager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                return eval_assembled.FARM

        class TwoStrataEnv:
            def __init__(self, *_args, **_kwargs):
                self._index = 0
                self.env = types.SimpleNamespace(
                    _raw={
                        "dungeon_level": 2, "dead": False,
                        "victory": False, "game_over": False,
                        "belt_heals": 1,
                    },
                    _ep_kills=0, _steps=0)

            def reset(self, *, seed):
                return [seed], {}

            def action_masks(self):
                return [True] * 3

            def step(self, _option):
                specs = (
                    dict(dry=True, reward=1.0, worker_wage=0.75,
                         kills=1, worker_kills=1, mode_seq="F"),
                    dict(dry=False, reward=2.0, worker_wage=1.25,
                         kills=2, worker_kills=1, mode_seq="FF"),
                )
                spec = specs[self._index]
                self._index += 1
                self.env._steps += 1
                self.env._ep_kills += spec["kills"]
                done = self._index == len(specs)
                if done:
                    self.env._raw["game_over"] = True
                return [0.0], spec["reward"], done, False, {
                    "option_extra": {
                        "beats": 1, "overrides": 0, "reason": "done",
                        "opt": eval_assembled.FARM, "tau": 1,
                        "mode_seq": spec["mode_seq"],
                        "R": spec["reward"], "W": spec["reward"],
                        "bonus": 0.0,
                        "worker_wage": spec["worker_wage"],
                        "kills_delta": spec["kills"],
                        "worker_kills": spec["worker_kills"],
                        "dry": spec["dry"],
                        "voluntary_drinks": 0,
                        "drain_attempts": 0, "drains": 0,
                    },
                }

            def close(self):
                pass

        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(Manager, TwoStrataEnv, object())):
            rows, _ = eval_assembled.evaluate(
                None, [11], manager_npz="manager.npz",
                manager_sha256="a" * 64,
                manager_policy_observation_view="legacy-v3")
        row = rows[0]
        self.assertEqual(row["farm_n"], 2)
        self.assertEqual(row["farm_dry_n"], 1)
        self.assertEqual(row["farm_fresh_n"], 1)
        self.assertEqual(row["farm_dry_worker_wage"], 0.75)
        self.assertEqual(row["farm_fresh_worker_wage"], 1.25)
        self.assertEqual(row["farm_worker_wage"], 2.0)
        self.assertEqual(row["farm_dry_worker_kills"], 1)
        self.assertEqual(row["farm_fresh_worker_kills"], 1)
        self.assertEqual(row["farm_worker_kills"], 2)

    def test_eval_terminal_kind_is_complete_and_fail_closed(self):
        base = {"dead": False, "victory": False, "game_over": False}
        self.assertEqual(eval_assembled.terminal_kind(
            dict(base, dead=True), {}, True, False, 9), "death")
        self.assertEqual(eval_assembled.terminal_kind(
            dict(base, victory=True), {}, True, False, 9), "victory")
        self.assertEqual(eval_assembled.terminal_kind(
            dict(base, game_over=True), {}, True, False, 9), "game_over")
        self.assertEqual(eval_assembled.terminal_kind(
            base, {
                "budget_exhausted": True,
                "decision_idle": True,
                "unsettled_budget_terminal": False,
                "time_limit_bootstrap_safe": True,
            }, False, True, eval_contract.PROTOCOL_MAX_STEPS),
            "time_limit_idle")
        self.assertEqual(eval_assembled.terminal_kind(
            base, {
                "budget_exhausted": True,
                "decision_idle": False,
                "unsettled_budget_terminal": True,
                "time_limit_bootstrap_safe": False,
            },
            True, False, eval_contract.PROTOCOL_MAX_STEPS),
            "time_limit_unsettled")
        with self.assertRaises(eval_contract.EvalContractError):
            eval_assembled.terminal_kind(base, {}, True, False, 9)
        with self.assertRaises(eval_contract.EvalContractError):
            eval_assembled.terminal_kind(
                base, {
                    "budget_exhausted": True,
                    "decision_idle": False,
                    "unsettled_budget_terminal": True,
                    "time_limit_bootstrap_safe": False,
                },
                True, False, eval_contract.PROTOCOL_MAX_STEPS - 1)
        with self.assertRaises(eval_contract.EvalContractError):
            eval_assembled.terminal_kind(
                dict(base, dead=True), {
                    "budget_exhausted": True,
                    "decision_idle": False,
                    "unsettled_budget_terminal": True,
                    "time_limit_bootstrap_safe": False,
                }, True, False, eval_contract.PROTOCOL_MAX_STEPS)
        with self.assertRaises(eval_contract.EvalContractError):
            eval_assembled.terminal_kind(
                base, {
                    "budget_exhausted": True,
                    "time_limit_bootstrap_safe": True,
                }, False, True, eval_contract.PROTOCOL_MAX_STEPS)
        for flag in ("dead", "victory", "game_over"):
            with self.subTest(flag=flag), self.assertRaises(
                    eval_contract.EvalContractError):
                eval_assembled.terminal_kind(
                    dict(base, **{flag: True}), {
                        "budget_exhausted": True,
                        "decision_idle": True,
                        "unsettled_budget_terminal": False,
                        "time_limit_bootstrap_safe": True,
                    }, False, True, eval_contract.PROTOCOL_MAX_STEPS)

    def test_teacher_parity_rejects_temperature_scaled_copy(self):
        import numpy as np
        import torch

        class Teacher(torch.nn.Module):
            def forward(self, obs):
                # 15 维动作头，正比例缩放不改变 argmax，却改变温度/概率。
                return obs[:, :15]

        class Net:
            def __init__(self, scale):
                self.scale = scale

            def worker_logits(self, obs, *, observation_view):
                self.asserted_view = observation_view
                return obs[:15] * self.scale

        obs = np.random.default_rng(0).standard_normal((1000, 20)).astype(np.float32)
        exact = check_teacher_parity.parity_metrics(Teacher(), Net(1.0), obs)
        scaled = check_teacher_parity.parity_metrics(Teacher(), Net(0.5), obs)
        self.assertTrue(check_teacher_parity.parity_passes(exact))
        self.assertEqual(scaled["raw_argmax_mismatch"], 0)
        self.assertFalse(check_teacher_parity.parity_passes(scaled))

    def test_model_kind_comes_from_checkpoint_metadata(self):
        cases = {
            "masked": "sb3_contrib.common.maskable.policies",
            "recurrent": "sb3_contrib.common.recurrent.policies",
            "ppo": "stable_baselines3.common.policies",
        }
        with tempfile.TemporaryDirectory() as directory:
            for expected, module in cases.items():
                path = pathlib.Path(directory) / f"opaque-{expected}.zip"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("data", json.dumps(
                        {"policy_class": {"__module__": module}}))
                # 故意省略 .zip，覆盖 SB3 常见命令行写法。
                self.assertEqual(evaluate.model_kind(str(path.with_suffix(""))), expected)

    def test_seed_parser_and_tag_reject_unsafe_inputs(self):
        self.assertEqual(eval_assembled.parse_seeds("7-9"), [7, 8, 9])
        with self.assertRaises(Exception):
            eval_assembled.parse_seeds("9-7")
        with self.assertRaises(Exception):
            eval_assembled.parse_seeds("0-4294967296")
        with self.assertRaises(Exception):
            eval_assembled.safe_tag("../overwrite")

    def test_default_manager_identity_is_content_pinned(self):
        snapshot = eval_contract.freeze_eval_identity(ROOT, "script")
        self.assertEqual(snapshot["manager"]["sha256"],
                         eval_contract.DEFAULT_MANAGER_SHA256)

    def test_runtime_identity_binds_loaded_engine_binary(self):
        engine_path = eval_contract.engine_binary_path(ROOT)
        runtime = eval_contract.runtime_identity(
            ROOT, eval_contract.bridge_binary_path(ROOT))
        self.assertEqual(runtime["engine"]["path"], str(engine_path))
        self.assertEqual(runtime["engine"]["sha256"],
                         eval_contract.sha256_file(engine_path))
        game_data = runtime["content"]["game_data"]
        self.assertTrue(pathlib.Path(game_data["path"]).is_absolute())
        self.assertEqual(game_data["sha256"],
                         eval_contract.sha256_file(game_data["path"]))
        assets = runtime["content"]["assets"]
        self.assertEqual(assets["path"],
                         str(eval_contract.default_assets_dir(ROOT)))
        self.assertGreater(assets["file_count"], 0)
        self.assertEqual(
            {k: v.split("+", 1)[0]
             for k, v in runtime["versions"]["packages"].items()},
            eval_contract.RUNTIME_PACKAGE_VERSIONS)

        real_version = eval_contract.importlib.metadata.version

        def drifted_version(name):
            return "0.0.0" if name == "numpy" else real_version(name)

        with mock.patch.object(eval_contract.importlib.metadata, "version",
                               side_effect=drifted_version):
            with self.assertRaisesRegex(eval_contract.EvalContractError,
                                        "运行时版本漂移"):
                eval_contract.runtime_versions_identity()

    def test_game_data_priority_and_resources_tree_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            data = root / "data"
            data.mkdir()
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.game_data_identity(data)

            spawn = data / "spawn.mpq"
            spawn.write_bytes(b"spawn")
            identity = eval_contract.game_data_identity(data)
            self.assertEqual(pathlib.Path(identity["path"]).name, "spawn.mpq")
            self.assertEqual(identity["sha256"], hashlib.sha256(b"spawn").hexdigest())

            lower = data / "diabdat.mpq"
            lower.write_bytes(b"full-lower")
            identity = eval_contract.game_data_identity(data)
            self.assertIn(pathlib.Path(identity["path"]).name,
                          {"DIABDAT.MPQ", "diabdat.mpq"})
            self.assertEqual(identity["sha256"],
                             hashlib.sha256(b"full-lower").hexdigest())
            uppercase = data / "DIABDAT.MPQ"
            same_entry = uppercase.exists() and uppercase.samefile(lower)
            if not same_entry:  # case-sensitive filesystem
                uppercase.write_bytes(b"full-upper")
                identity = eval_contract.game_data_identity(data)
                self.assertEqual(pathlib.Path(identity["path"]).name, "DIABDAT.MPQ")
                self.assertEqual(identity["sha256"],
                                 hashlib.sha256(b"full-upper").hexdigest())

            assets = root / "Resources"
            assets.mkdir()
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.assets_tree_identity(assets)
            (assets / "z").mkdir()
            (assets / "z" / "last.bin").write_bytes(b"last")
            (assets / "first.bin").write_bytes(b"first")
            first = eval_contract.assets_tree_identity(assets)
            second = eval_contract.assets_tree_identity(assets)
            self.assertEqual(first, second)
            self.assertEqual(first["file_count"], 2)
            (assets / "z" / "last.bin").write_bytes(b"changed")
            changed = eval_contract.assets_tree_identity(assets)
            self.assertNotEqual(changed["sha256"], first["sha256"])
            (assets / "link.bin").symlink_to(assets / "first.bin")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.assets_tree_identity(assets)

    def test_post_eval_rehash_rejects_engine_binary_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for relative in eval_contract.PROTOCOL_SOURCE_FILES:
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(relative)
            bridge_path = root / "build" / eval_contract.bridge_binary_path(ROOT).name
            bridge_path.parent.mkdir(parents=True, exist_ok=True)
            bridge_path.write_bytes(b"bridge")
            engine_path = (root / "build" / "engine"
                           / eval_contract.engine_binary_path(ROOT).name)
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            engine_path.write_bytes(b"engine-v1")
            data_dir, assets_dir = _runtime_content_fixture(root)
            manager_path = root / "manager.npz"
            manager_path.write_bytes(b"manager")
            runtime = eval_contract.runtime_identity(
                root, bridge_path, engine_path,
                data_dir=data_dir, assets_dir=assets_dir)
            snapshot = {
                "worker": eval_contract.script_worker_identity(
                    runtime["python_protocol"]["sha256"]),
                "manager": eval_contract.file_identity(
                    "numpy_policy", manager_path),
                "runtime": runtime,
            }
            eval_contract.verify_eval_identity(snapshot, root)
            game_path = data_dir / "spawn.mpq"
            game_path.write_bytes(b"spawn-v2")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_eval_identity(snapshot, root)
            game_path.write_bytes(b"spawn-v1")
            eval_contract.verify_eval_identity(snapshot, root)
            asset_path = assets_dir / "ASSETS_VERSION"
            asset_path.write_bytes(b"2")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_eval_identity(snapshot, root)
            asset_path.write_bytes(b"1")
            eval_contract.verify_eval_identity(snapshot, root)
            engine_path.write_bytes(b"engine-v2")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_eval_identity(snapshot, root)
            duplicate_engine = (root / "build" / "engine" / "Debug"
                                / engine_path.name)
            duplicate_engine.parent.mkdir(parents=True, exist_ok=True)
            duplicate_engine.write_bytes(b"ambiguous-engine")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.engine_binary_path(root)

    def test_digest_uses_unrounded_tau_sum(self):
        base = {"ret": 0, "died": False, "depth": 1, "kills": 0,
                "micro_steps": 2, "terminal_kind": "game_over",
                "farm_r": 0.0, "farm_w": 0.0, "farm_bonus": 0.0,
                "farm_worker_wage": 0.0, "farm_kills": 0,
                "farm_worker_kills": 0, "nonfarm_r": 0.0,
                "nonfarm_kills": 0,
                "farm_dry_n": 1, "farm_fresh_n": 0,
                "farm_dry_worker_wage": 0.0,
                "farm_fresh_worker_wage": 0.0,
                "farm_dry_worker_kills": 0,
                "farm_fresh_worker_kills": 0,
                "farm_voluntary_drinks": 0,
                "farm_reflex_drain_attempts": 0,
                "farm_reflex_drains": 0,
                "farm_multi_drink_windows": 0,
                "farm_max_voluntary_drinks_per_window": 0,
                "ending_belt_heals": 0,
                "farm_n": 1, "farm_descend": 0, "overrides": 0,
                "beats": 1, "cap": 0, "windows": 1}
        rows = [dict(base, farm_tau_mean=1.0, farm_tau_sum=1.04),
                dict(base, farm_tau_mean=1.1, farm_tau_sum=1.05)]
        self.assertEqual(eval_assembled.digest(rows)["farm_tau_mean"], 1.045)

    def test_pairing_helpers_reject_duplicate_seed_rows(self):
        rows = [{"seed": seed} for seed in range(7000, 7032)]
        self.assertEqual(len(run_v25_election.by_seed(rows)), 32)
        duplicated = rows[:-1] + [{"seed": 7030}]
        with self.assertRaises(ValueError):
            run_v25_election.by_seed(duplicated)
        with self.assertRaises(ValueError):
            run_v30_relay.by_seed(duplicated)

    def test_probe_reachability_uses_rollout_alignment(self):
        # 450,000 的首个可见 rollout 是 450,560；端点恰落这里仍然可达。
        for helper in (run_v28_legs.reachable_probes, run_v30_relay.reachable_probes):
            self.assertEqual(helper(10_000, 450_560), [260_000, 460_000])
            self.assertEqual(helper(10_000, 450_559), [260_000])
            self.assertEqual(helper(10_000, 249_999), [])

    def test_dashboard_reads_only_requested_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "progress.jsonl"
            path.write_text("".join(f'{{"i":{i}}}\n' for i in range(10_000)))
            self.assertEqual(dashboard.tail_lines(path, 3), [
                '{"i":9997}', '{"i":9998}', '{"i":9999}',
            ])

    def test_eval_v4_binds_expected_model_identity(self):
        document = _valid_v5_archive()
        expected = {
            "expected_tag": "audit-v4", "expected_seeds": [7, 8],
            "expected_worker_sha256": "a" * 64,
            "expected_manager_sha256": "b" * 64,
            "expected_worker_num_timesteps": 1234,
            "expected_engine_sha256": "e" * 64,
            "expected_game_data_path": "/frozen/data/DIABDAT.MPQ",
            "expected_game_data_sha256": "f" * 64,
            "expected_assets_path":
                "/frozen/build/engine/devilutionx.app/Contents/Resources",
            "expected_assets_sha256": "1" * 64,
            "expected_assets_file_count": 2,
        }
        eval_contract.validate_eval_archive(document, **expected)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "archive.json"
            path.write_text(json.dumps(document, allow_nan=False))
            # 落盘/重读后 histogram 的 JSON 字典键会变成字符串，也必须通过。
            eval_contract.read_eval_archive(path, **expected)
            wrong = dict(expected, expected_worker_sha256="f" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                # JSON 本身完整、seed 正确，但来自另一份 worker。
                eval_contract.read_eval_archive(path, **wrong)
            wrong_engine = dict(expected, expected_engine_sha256="f" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path, **wrong_engine)
            wrong_data = dict(expected, expected_game_data_sha256="0" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path, **wrong_data)
            wrong_assets = dict(expected, expected_assets_sha256="0" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path, **wrong_assets)
            for key, value in (
                    ("expected_game_data_path", "/other/data/DIABDAT.MPQ"),
                    ("expected_assets_path", "/other/Contents/Resources"),
                    ("expected_assets_file_count", 3)):
                with self.subTest(expected_field=key):
                    wrong_content = dict(expected, **{key: value})
                    with self.assertRaises(eval_contract.EvalContractError):
                        eval_contract.read_eval_archive(path, **wrong_content)

    def test_expected_identity_carries_full_content_contract(self):
        document = _valid_v5_archive()
        meta = document["meta"]
        expected = eval_contract.expected_eval_identity(
            {"worker": meta["worker"], "manager": meta["manager"],
             "runtime": meta["runtime"]},
            tag=meta["tag"], seeds=meta["protocol"]["seeds"])
        self.assertEqual(expected["expected_game_data_path"],
                         "/frozen/data/DIABDAT.MPQ")
        self.assertEqual(expected["expected_game_data_sha256"], "f" * 64)
        self.assertEqual(expected["expected_assets_sha256"], "1" * 64)
        self.assertEqual(expected["expected_assets_file_count"], 2)

    def test_eval_v5_validates_complete_action14_gear_ledger(self):
        document = _valid_v5_archive()
        document["agg"].update({
            "worker_action_hist": {"9": 1, "14": 1},
            "worker_action14_mask_opportunities": 1,
            "worker_action14_requests": 1,
            "worker_action14_native_successes": 1,
            "worker_action14_gear_utility_delta": 37,
        })
        eval_contract.validate_eval_archive(document)

        partial = copy.deepcopy(document)
        del partial["agg"]["worker_action14_native_successes"]
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "agg 字段"):
            eval_contract.validate_eval_archive(partial)

        histogram_mismatch = copy.deepcopy(document)
        histogram_mismatch["agg"]["worker_action14_requests"] = 0
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "worker_action_hist"):
            eval_contract.validate_eval_archive(histogram_mismatch)

        false_native_success = copy.deepcopy(document)
        false_native_success["agg"][
            "worker_action14_gear_utility_delta"] = 0
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "utility"):
            eval_contract.validate_eval_archive(false_native_success)

    def test_eval_v5_recomputes_agg_and_rejects_nonfinite_rows(self):
        document = _valid_v5_archive()
        self.assertEqual(document["agg"]["ret_mean"], 2.0625)
        self.assertEqual(document["agg"]["farm_worker_kills_mean"], 1.5)
        self.assertEqual(document["agg"]["farm_dry_n_mean"], 0.5)
        self.assertEqual(document["agg"]["farm_fresh_n_mean"], 0.5)
        self.assertEqual(
            document["agg"]["farm_dry_worker_wage_mean"], 0.25)
        self.assertEqual(
            document["agg"]["farm_fresh_worker_wage_mean"], 0.25)
        self.assertEqual(
            document["agg"]["farm_dry_worker_kills_mean"], 0.5)
        self.assertEqual(
            document["agg"]["farm_fresh_worker_kills_mean"], 1.0)
        self.assertEqual(document["agg"]["farm_voluntary_drinks_mean"], 1.5)
        self.assertEqual(
            document["agg"]["farm_reflex_drain_attempts_mean"], 1.5)
        self.assertEqual(document["agg"]["farm_reflex_drains_mean"], 0.5)
        self.assertEqual(document["agg"]["farm_multi_drink_window_rate"], 0.5)
        self.assertEqual(document["agg"]["ending_belt_heals_mean"], 1.0)
        tampered = copy.deepcopy(document)
        tampered["agg"]["ret_mean"] = 1_000_000_000.0
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(tampered)
        tampered_stratum_agg = copy.deepcopy(document)
        tampered_stratum_agg["agg"]["farm_dry_worker_wage_mean"] += 0.5
        with self.assertRaisesRegex(
                eval_contract.EvalContractError,
                "agg.farm_dry_worker_wage_mean"):
            eval_contract.validate_eval_archive(tampered_stratum_agg)

        nonfinite = copy.deepcopy(document)
        nonfinite["rows"][0]["ret"] = float("nan")
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(nonfinite)
        with self.assertRaises(ValueError):
            json.dumps(nonfinite, allow_nan=False)

        impossible_sequence = copy.deepcopy(document)
        impossible_sequence["rows"][0]["mode_seq"] = "R"
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(impossible_sequence)

        naive_timestamp = copy.deepcopy(document)
        naive_timestamp["meta"]["created_at_utc"] = "2026-07-12T12:00:00"
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(naive_timestamp)

        floating_tau = copy.deepcopy(document)
        floating_tau["rows"][0]["farm_tau_sum"] = 2.0
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(floating_tau)

        impossible_worker_calls = copy.deepcopy(document)
        impossible_worker_calls["agg"]["worker_calls"] = 8
        impossible_worker_calls["agg"]["worker_action_hist"] = {"9": 8}
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(impossible_worker_calls)

        # fuse 拒绝会计入 worker callback/overrides，但不会增加已执行 beats；
        # 因而 beats+overrides 才是严格且不误杀的调用上界。
        rejected_proposal = copy.deepcopy(document)
        legal_calls = sum(
            row["beats"] + row["overrides"]
            for row in rejected_proposal["rows"])
        rejected_proposal["agg"].update({
            "worker_calls": legal_calls,
            "worker_action_hist": {"9": legal_calls},
            "worker_divergences": 0,
            "script_divergence_rate": 0.0,
        })
        eval_contract.validate_eval_archive(rejected_proposal)

        broken_return_partition = copy.deepcopy(document)
        broken_return_partition["rows"][0]["nonfarm_r"] += 0.25
        broken_return_partition["agg"] = eval_contract.recompute_agg(
            broken_return_partition["rows"])
        broken_return_partition["agg"].update({
            key: document["agg"][key]
            for key in eval_contract._ENGAGEMENT_KEYS
        })
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "回报.*不守恒"):
            eval_contract.validate_eval_archive(broken_return_partition)

        broken_wage_partition = copy.deepcopy(document)
        broken_wage_partition["rows"][0]["farm_w"] += 0.25
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "R/W/bonus"):
            eval_contract.validate_eval_archive(broken_wage_partition)

        broken_kill_partition = copy.deepcopy(document)
        broken_kill_partition["rows"][0]["farm_worker_kills"] = 2
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "击杀分账"):
            eval_contract.validate_eval_archive(broken_kill_partition)

        # schema-v5 的 FARM dry/fresh 分账必须逐 seed 严格守恒；重算 agg
        # 不能掩盖 row 内部被篡改的窗口、工资或 worker 击杀。
        for case, field, value, message in (
                ("windows", "farm_dry_n", 2, "dry/fresh 窗口"),
                ("wage", "farm_dry_worker_wage", 0.75,
                 "dry/fresh worker 工资"),
                ("kills", "farm_dry_worker_kills", 2,
                 "dry/fresh worker 击杀")):
            broken_stratum = copy.deepcopy(document)
            broken_stratum["rows"][0][field] = value
            broken_stratum["agg"] = eval_contract.recompute_agg(
                broken_stratum["rows"])
            broken_stratum["agg"].update({
                key: document["agg"][key]
                for key in eval_contract._ENGAGEMENT_KEYS
            })
            with self.subTest(stratum_ledger=case), self.assertRaisesRegex(
                    eval_contract.EvalContractError, message):
                eval_contract.validate_eval_archive(broken_stratum)

        empty_stratum = copy.deepcopy(document)
        empty_stratum["rows"][0].update({
            "farm_dry_worker_wage": 0.25,
            "farm_fresh_worker_wage": 0.25,
        })
        empty_stratum["agg"] = eval_contract.recompute_agg(
            empty_stratum["rows"])
        empty_stratum["agg"].update({
            key: document["agg"][key]
            for key in eval_contract._ENGAGEMENT_KEYS
        })
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "fresh 无窗口"):
            eval_contract.validate_eval_archive(empty_stratum)

        broken_terminal = copy.deepcopy(document)
        broken_terminal["rows"][0]["terminal_kind"] = "time_limit_idle"
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(broken_terminal)

        impossible_beats = copy.deepcopy(document)
        impossible_beats["rows"][0]["beats"] = 3
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "action-beat"):
            eval_contract.validate_eval_archive(impossible_beats)

        impossible_multi_drink = copy.deepcopy(document)
        impossible_multi_drink["rows"][0]["farm_multi_drink_windows"] = 1
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "多饮窗口"):
            eval_contract.validate_eval_archive(impossible_multi_drink)

        impossible_drink_count = copy.deepcopy(document)
        impossible_drink_count["rows"][1]["farm_voluntary_drinks"] = 1
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "主动饮"):
            eval_contract.validate_eval_archive(impossible_drink_count)

        impossible_reflex_success = copy.deepcopy(document)
        impossible_reflex_success["rows"][1][
            "farm_reflex_drain_attempts"] = 0
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "反射尝试"):
            eval_contract.validate_eval_archive(impossible_reflex_success)

        # 失败尝试不消耗药水：守恒边界继续只计算真实 drains。
        failed_reflex_attempts = copy.deepcopy(document)
        failed_reflex_attempts["rows"][0][
            "farm_reflex_drain_attempts"] = 2
        failed_reflex_attempts["agg"] = eval_contract.recompute_agg(
            failed_reflex_attempts["rows"])
        failed_reflex_attempts["agg"].update({
            key: document["agg"][key]
            for key in eval_contract._ENGAGEMENT_KEYS
        })
        eval_contract.validate_eval_archive(failed_reflex_attempts)

        # N=1,T=3,M=1,max=2 不可能：唯一窗口总计喝了三瓶，
        # 该窗口本身就必须是 max=3。
        # N=1,T=2,M=0,max=1 也不可能：唯一窗口喝了两瓶，
        # 必然应计为一个多饮窗口且 max=2。
        for case, fields in (
                ("underreported_max", {
                    "farm_voluntary_drinks": 3,
                    "farm_reflex_drain_attempts": 0,
                    "farm_reflex_drains": 0,
                    "farm_multi_drink_windows": 1,
                    "farm_max_voluntary_drinks_per_window": 2,
                }),
                ("hidden_multi_drink_window", {
                    "farm_voluntary_drinks": 2,
                    "farm_reflex_drain_attempts": 0,
                    "farm_reflex_drains": 0,
                    "farm_multi_drink_windows": 0,
                    "farm_max_voluntary_drinks_per_window": 1,
                })):
            impossible_distribution = copy.deepcopy(document)
            impossible_distribution["rows"][1].update(fields)
            impossible_distribution["agg"] = eval_contract.recompute_agg(
                impossible_distribution["rows"])
            impossible_distribution["agg"].update({
                key: document["agg"][key]
                for key in eval_contract._ENGAGEMENT_KEYS
            })
            with self.subTest(drink_distribution=case), self.assertRaisesRegex(
                    eval_contract.EvalContractError, "N/T/M/max"):
                eval_contract.validate_eval_archive(impossible_distribution)

        missing_content = copy.deepcopy(document)
        del missing_content["meta"]["runtime"]["content"]
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(missing_content)

        bad_assets_count = copy.deepcopy(document)
        bad_assets_count["meta"]["runtime"]["content"]["assets"]["file_count"] = 0
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(bad_assets_count)

    def test_protocol_semantic_break_is_version_four(self):
        self.assertEqual(eval_contract.SCHEMA_VERSION, 5)
        self.assertEqual(eval_contract.PROTOCOL_VERSION, 4)
        legacy_schema = _valid_v5_archive()
        legacy_schema["schema_version"] = 4
        with self.assertRaisesRegex(
                eval_contract.EvalContractError, "schema 必须为 v5"):
            eval_contract.validate_eval_archive(legacy_schema)

    def test_same_tag_reservation_is_cross_process_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "same-tag.json"
            context = multiprocessing.get_context("spawn")
            ready, release = context.Event(), context.Event()
            process = context.Process(
                target=_hold_output_reservation, args=(output, ready, release))
            process.start()
            try:
                self.assertTrue(ready.wait(10), "子进程未取得评测 reservation")
                with self.assertRaises(eval_contract.OutputReservationError):
                    with eval_contract.reserve_output(output):
                        pass
            finally:
                release.set()
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            self.assertEqual(process.exitcode, 0)
            with eval_contract.reserve_output(output):
                pass

    def test_different_tags_share_one_leaderboard_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            tag_a, tag_b = root / "tag-a.json", root / "tag-b.json"
            board_lock = root / ".leaderboard.lock"
            context = multiprocessing.get_context("spawn")
            ready, release = context.Event(), context.Event()
            process = context.Process(target=_hold_tag_and_board_reservation,
                                      args=(tag_a, board_lock, ready, release))
            process.start()
            try:
                self.assertTrue(ready.wait(10), "子进程未取得排行榜锁")
                # 不同 tag 自身并不冲突，但排行榜的 read/check/replace 必须冲突。
                with eval_contract.reserve_output(tag_b):
                    with self.assertRaises(eval_contract.OutputReservationError):
                        with eval_contract.exclusive_lock(board_lock, "排行榜"):
                            pass
            finally:
                release.set()
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            self.assertEqual(process.exitcode, 0)

    def test_bc_identity_rechecks_gate_report_after_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            weights, report = root / "policy_sd.pt", root / "bc_report.json"
            weights.write_bytes(b"weights-v1")
            report.write_text('{"data_gate":"PASS"}')
            identity = eval_contract.file_identity(
                "bc_state_dict", weights, gate_report_path=report)
            eval_contract.verify_file_identity(identity)
            report.write_text('{"data_gate":"PASS","changed":true}')
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_file_identity(identity)

    def test_eval_bc_capture_uses_full_training_gate_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            weights, report = root / "policy_sd.pt", root / "bc_report.json"
            weights.write_bytes(b"weights")
            # 这组字段曾足以越过 eval_assembled 的手写子集校验；它没有
            # held-out 指标/逐类召回/demos 绑定，绝不能被命名为 PASS BC。
            report.write_text(json.dumps({
                "data_gate": "PASS",
                "policy_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "protocol_version": eval_contract.PROTOCOL_VERSION,
                "implementation_sha256": "a" * 64,
                "generator_sha256": "b" * 64,
                "manager_npz_sha256": "c" * 64,
            }))
            with self.assertRaisesRegex(ValueError, "字段/schema"):
                eval_assembled.capture_passed_bc(weights)

    def test_probe_reference_must_use_exact_seed_set(self):
        rows = [{"seed": 7, "ret": 1.0, "depth": 1, "died": False}]
        reference = [
            {"seed": 7, "ep_R": 1.0, "depth": 1, "died": False},
            {"seed": 8, "ep_R": 2.0, "depth": 1, "died": False},
        ]
        with self.assertRaises(ValueError):
            eval_assembled.compare_probe_rows(rows, reference)

    def test_legacy_archive_requires_exact_explicit_sha(self):
        legacy = {"agg": {"n": 1}, "rows": [{"seed": 7}]}
        payload = json.dumps(legacy).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy.json"
            path.write_bytes(payload)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path)
            trusted = hashlib.sha256(payload).hexdigest()
            self.assertEqual(eval_contract.read_eval_archive(
                path, trusted_legacy_sha256=trusted), legacy)


if __name__ == "__main__":
    unittest.main()
