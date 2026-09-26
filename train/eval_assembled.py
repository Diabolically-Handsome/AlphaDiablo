"""v23 assembled-agent evaluation: frozen H manager + {script|BC|PPO} FARM worker (docs/prereg/PREREG-v23.md).

Usage:
  H7 baseline:  .venv/bin/python train/eval_assembled.py --worker script --seeds 7000-7031
  G0'' regression: ... --worker script --seeds 7000-7031 --check-probe docs/assets/window_econ_v23_probe.json
  G1 BC replay: ... --worker bc --seeds 7000-7031
  G3 screen:  ... --worker train/runs/<run>/ckpt/model_XXX_steps --seeds 7000-7015
  gold-standard evaluation (once only): ... --worker <winner> --seeds 9000-9031 --board
Protocol: argmax (the manager's numpy forward = the same code reconciled bit for bit in G0'), 3000 micro-steps,
return = the manager's undiscounted ledger. The R4 sentinels (level-change rate/override/cap/tau-bar) are produced as well.
R16 (C3): --worker-decoding {argmax,sample}; the default argmax reproduces old exams bit for bit;
sample = the SB3 worker samples from its training distribution (model.set_random_seed(seed) per episode,
single-threaded torch, bit-reproducible), and the decoding mode is written to meta.protocol.worker_decoding.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import pathlib
import re
import sys
import time
import zipfile
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from eval_contract import (DEFAULT_MANAGER_SHA256, PROTOCOL_MAX_STEPS,
                           PUBLISHED_WORKER_RECEIPT_NAME,
                           R16_ENVIRONMENT_DEFAULTS,
                           SCHEMA_VERSION, UINT32_MAX, EvalContractError,
                           OutputReservationError, PROTOCOL_VERSION,
                           PROTOCOL_SOURCE_FILES,
                           validate_r16_environment, validate_resource_service_config, validate_dive_blocker_recovery,
                           bridge_binary_path, checkpoint_num_timesteps_bytes,
                           exclusive_lock,
                           expected_eval_identity, file_identity, make_meta,
                           loaded_engine_binary_path,
                           read_eval_archive, recompute_agg,
                           reserve_output, resolve_checkpoint_file, runtime_identity,
                           script_worker_identity, strict_json_loads,
                           validate_eval_archive,
                           verify_file_identity)
from evaluate import (assembled_leaderboard_row, ensure_leaderboard_compatible,
                      freeze_standalone_contract, upsert_leaderboard_rows,
                      versioned_row_key)
if TYPE_CHECKING:
    from diablogym import NumpyManager

NPZ = ROOT / "train" / "models" / "v22-h-manager" / "policy.npz"
NPZ_SHA = DEFAULT_MANAGER_SHA256
LB = ROOT / "train" / f"leaderboard-assembled-v{PROTOCOL_VERSION}.md"
OUTDIR = ROOT / "train" / "runs" / "eval-assembled"
LB_LOCK = ROOT / "train" / "runs" / "eval-locks" / "leaderboard-assembled.lock"
FARM = 0
DIVE = 1   # R13 exam new law: the farm-dive-v1 registration also hooks the worker callback onto DIVE windows
_NATIVE_RUNTIME = None
_WORKER_OBSERVATION_VIEWS = frozenset({
    "raw-v4",
    "legacy-v3",
    "legacy-v3-a12-overlay",
    "dual-v4-asymmetric-v3",
})
# Frozen audit identities for dual-Worker contracts that share the current
# controller/runtime topology.  Keeping this map in the evaluator prevents a
# newer validator from retroactively upgrading an older checkpoint's claim.
_DUAL_WORKER_PG_AUDIT_SCHEMA_BY_CONTRACT_REVISION = {
    24: "diablogym-worker-onpolicy-pg/8",
    25: "diablogym-worker-onpolicy-pg/9",
    # A4 (2026-07-27): rev26 = the kl_early_stopped flag, audit schema /10.
    26: "diablogym-worker-onpolicy-pg/10",
}

ASSEMBLED_BOARD_SOURCE_FILES = (*PROTOCOL_SOURCE_FILES, "train/evaluate.py")
ASSEMBLED_BOARD_PROTOCOL = {
    "name": "diablogym.standalone.assembled",
    "seeds": list(range(9000, 9032)),
    "environment": "OptionsEnv",
    "max_steps": 3000,
    "action_selection": "argmax_with_action_masks",
    "manager_forward": "numpy_tanh_mlp",
    "reward": "undiscounted_manager_ledger",
    "worker_kinds": ["script", "bc_state_dict", "numpy_policy", "sb3_checkpoint"],
}
ASSEMBLED_LEADERBOARD_HEADER = (
    f"# Assembled-agent board protocol v{PROTOCOL_VERSION} — 32 fixed seeds\n\n"
    "Protocol: OptionsEnv, 3000 micro-steps, argmax + masks, seeds 9000-9031.\n"
    f"Every row is bound to its immutable schema-v{SCHEMA_VERSION} "
    "evaluation archive.\n\n"
    "| run | ret mean | ret med | died | depth med | notes |\n"
    "|---|---|---|---|---|---|\n"
)


def assembled_board_contract() -> dict:
    return freeze_standalone_contract(
        evaluator=f"standalone-assembled-v{PROTOCOL_VERSION}",
        protocol=ASSEMBLED_BOARD_PROTOCOL,
        source_files=ASSEMBLED_BOARD_SOURCE_FILES)


def _native_runtime(expected_runtime: dict | None = None):
    """Map bridge/engine only after the identity is frozen, and immediately check the actually loaded path."""
    global _NATIVE_RUNTIME
    if _NATIVE_RUNTIME is None:
        from diablogym import NumpyManager, OptionsEnv, bridge
        from diablogym.options_env import DIVE as actual_dive
        from diablogym.options_env import FARM as actual_farm

        if actual_farm != FARM:
            raise EvalContractError(
                f"OptionsEnv FARM index drift: {actual_farm} != {FARM}")
        if actual_dive != DIVE:
            raise EvalContractError(
                f"OptionsEnv DIVE index drift: {actual_dive} != {DIVE}")
        # R16 new-law anchor: the archive identity is encoded as "record only non-defaults", so the CLI/contract defaults
        # must match the OptionsEnv old-law constants bit for bit; any drift fails closed.
        from diablogym.options_env import FARM_SCENE_CAP as actual_scene_cap
        if actual_scene_cap != R16_ENVIRONMENT_DEFAULTS["farm_scene_cap"]:
            raise EvalContractError(
                "OptionsEnv FARM_SCENE_CAP drifted from R16_ENVIRONMENT_DEFAULTS: "
                f"{actual_scene_cap} != "
                f"{R16_ENVIRONMENT_DEFAULTS['farm_scene_cap']}")
        _NATIVE_RUNTIME = (NumpyManager, OptionsEnv, bridge)
    manager_class, env_class, loaded_bridge = _NATIVE_RUNTIME
    if expected_runtime is not None:
        try:
            expected_path = pathlib.Path(
                expected_runtime["bridge"]["path"]).resolve()
            actual_path = pathlib.Path(loaded_bridge.__file__).resolve()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise EvalContractError("frozen bridge runtime identity structure is malformed") from exc
        if actual_path != expected_path:
            raise EvalContractError(
                f"actually loaded bridge path differs from the frozen identity: {actual_path} != {expected_path}")
        loaded_engine_binary_path(expected_runtime["engine"]["path"])
        if runtime_identity(ROOT, actual_path) != expected_runtime:
            raise EvalContractError(
                "bridge, engine, game content or protocol source changed during native import")
    return manager_class, env_class, loaded_bridge


def np_policy_from_sd(source: str | pathlib.Path | bytes,
                      expected_sha256: str | None = None) -> NumpyManager:
    import torch
    payload = (source if isinstance(source, bytes)
               else pathlib.Path(source).read_bytes())
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"BC state_dict SHA mismatch: {actual_sha256} != {expected_sha256}")
    sd = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    required = (
        "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
        "action_net.weight", "action_net.bias",
    )
    if not isinstance(sd, dict) or any(k not in sd for k in required):
        missing = [k for k in required if not isinstance(sd, dict) or k not in sd]
        raise ValueError(f"BC state_dict lacks policy-head keys: {missing}")
    tensors = [sd[k].detach().cpu() for k in required]
    if not all(torch.isfinite(t).all().item() for t in tensors):
        raise ValueError("BC state_dict contains NaN/Inf")
    w0, b0, w1, b1, wa, ba = tensors
    if (w0.ndim != 2 or tuple(w0.shape)[1] != 298 or b0.shape != w0.shape[:1]
            or w1.shape != (w0.shape[0], w0.shape[0]) or b1.shape != w0.shape[:1]
            or wa.shape != (15, w0.shape[0]) or ba.shape != (15,)):
        raise ValueError("BC worker policy shape is malformed (must be 298->hidden->hidden->15)")
    manager_class, _env_class, _bridge = _native_runtime()
    m = manager_class.__new__(manager_class)
    m.w0, m.b0, m.w1, m.b1, m.wa, m.ba = (
        t.numpy().astype(np.float32, copy=False) for t in tensors)
    m.source_sha256 = actual_sha256
    return m


def capture_passed_bc(sd_path: pathlib.Path) -> tuple[bytes, bytes]:
    """Capture policy/report once and validate the gate against those bytes."""
    report = sd_path.with_name("bc_report.json")
    try:
        policy_payload = sd_path.read_bytes()
        report_payload = report.read_bytes()
    except OSError as exc:
        raise ValueError(f"BC gate report missing/unreadable: {report}") from exc
    from train_ppo import _validate_bc_report
    _validate_bc_report(
        sd_path, "data_gate", policy_payload=policy_payload,
        report_payload=report_payload)
    return policy_payload, report_payload


_PUBLICATION_EXPECTATIONS_SCHEMA = "diablogym-publication-expectations/1"
_PUBLISHED_RECEIPT_KEYS = {
    "schema_version", "step", "demos_sha256", "objective_revision",
    "evaluation_scope", "mask_mode", "anchor",
    "candidate_policy_head_sha256", "provenance", "metrics", "gate",
    "exploration_evidence", "publication", "model_sha256", "save_error",
}


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_plain_int(value) -> bool:
    """JSON ``bool`` is an ``int`` subclass; protocol counts must not accept it."""
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _canonical_bc_aux_demos_path(expected_sha256: str) -> pathlib.Path:
    """Resolve the immutable canonical BC-v2 bundle by content identity.

    The recovery launcher has used an isolated ``-r2`` directory while older
    completed bundles live at the original path.  Only these two registered
    locations are eligible; pending files and arbitrary hash-matching files
    elsewhere in the worktree are deliberately ignored.
    """
    if not _is_sha256(expected_sha256):
        raise EvalContractError("Worker publication receipt demos SHA is illegal")
    candidates = (
        ROOT / "train" / "runs" / "bc-worker-v2-r2" / "demos.npz",
        ROOT / "train" / "runs" / "bc-worker-v2" / "demos.npz",
    )
    matches = []
    for path in candidates:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(payload).hexdigest() == expected_sha256:
            matches.append(path.resolve())
    if not matches:
        raise EvalContractError(
            "no canonical BC-v2 demos bound to the Worker publication receipt found")
    return matches[0]


def _recompute_published_worker_evidence(
        checkpoint_payload: bytes, *,
        demos_sha256: str,
        expected_manager_sha256: str) -> dict:
    """Load the exact checkpoint and independently rebuild publication proof."""
    try:
        import torch as th
        from leashed_ppo import (
            A12MixtureMaskableActorCriticPolicy,
            LeashedMaskablePPO,
        )
        from train_ppo import (
            _BC_AUX_CIRCUIT_ACTION,
            _POLICY_HEAD_KEYS,
            _bc_aux_circuit_spec,
            _expand_policy_with_bc_aux_circuit,
            _load_bc_aux_demos_v2,
            _persistent_bc_aux_root_anchor,
            _policy_head_sha256,
            _policy_head_snapshot,
            bc_aux_behavior_metrics,
        )

        model = LeashedMaskablePPO.load(
            io.BytesIO(checkpoint_payload), env=None, device="cpu",
            teacher_path=None, teacher_sha256=None)
    except Exception as exc:
        raise EvalContractError(
            "the official Worker checkpoint cannot be safely loaded with the current LeashedMaskablePPO"
        ) from exc

    spec = _bc_aux_circuit_spec()
    if not (
        type(model.policy) is A12MixtureMaskableActorCriticPolicy
        and model.policy_class is A12MixtureMaskableActorCriticPolicy
        and getattr(model, "_bc_aux_circuit_spec", None) == spec
        and model.policy.bc_aux_mixture_spec == spec
        and isinstance(model.policy_kwargs, dict)
        and model.policy_kwargs.get("bc_aux_mixture_spec") == spec
        and model.policy_kwargs.get("net_arch")
        == {"pi": [68, 68], "vf": [64, 64]}
    ):
        raise EvalContractError(
            "Worker checkpoint is not bound to the exact rev11 custom policy/spec")
    try:
        # Existing-adapter branch performs the topology/class/spec closure
        # checks without changing any parameter.
        if _expand_policy_with_bc_aux_circuit(model) != spec:
            raise ValueError("circuit spec mismatch")
        for parameter, protected in model._bc_aux_circuit_protected_tensors():
            if not bool((parameter.detach()[protected] == 0).all().item()):
                raise ValueError("protected circuit tensor is non-zero")
        action = int(spec["action_index"])
        columns = [int(v) for v in spec["gate_parameter_columns"]]
        coefficients = model.policy.action_net.weight[
            action, columns].detach()
        bias = model.policy.action_net.bias[action].detach()
        limit = float(spec["gate_parameter_abs_max"])
        if not (
            action == _BC_AUX_CIRCUIT_ACTION
            and bool(th.isfinite(coefficients).all().item())
            and bool(th.isfinite(bias).item())
            and bool((coefficients.abs() <= limit).all().item())
            and abs(float(bias.cpu())) <= limit
        ):
            raise ValueError("contextual gate parameters invalid")

        candidate = _policy_head_snapshot(model.policy)
        root_anchor = _persistent_bc_aux_root_anchor(model)
        if set(candidate) != set(_POLICY_HEAD_KEYS) \
                or set(root_anchor) != set(_POLICY_HEAD_KEYS):
            raise ValueError("policy head key set mismatch")
        demos_path = _canonical_bc_aux_demos_path(demos_sha256)
        x, y, episode_id, masks, observed_demos_sha = (
            _load_bc_aux_demos_v2(
                demos_path,
                expected_manager_sha256=expected_manager_sha256))
        if observed_demos_sha != demos_sha256:
            raise ValueError("canonical demos hash mismatch")
        metrics = bc_aux_behavior_metrics(
            candidate, x, y, episode_id, masks,
            anchor_sd=root_anchor, heldout_only=True,
            circuit_spec=spec)
        return {
            "candidate_policy_head_sha256":
                _policy_head_sha256(candidate),
            "anchor_policy_head_sha256":
                _policy_head_sha256(root_anchor),
            "metrics": metrics,
        }
    except EvalContractError:
        raise
    except Exception as exc:
        raise EvalContractError(
            "independent evidence recomputation of the official Worker checkpoint/canonical demos failed"
        ) from exc


def capture_publication_expectations(
        path: str | pathlib.Path) -> tuple[dict, dict]:
    """Freeze this case's pre-registered lineage subset; it must be re-verified by SHA before evaluation ends."""
    expected_path = pathlib.Path(path).resolve()
    try:
        payload = expected_path.read_bytes()
        document = strict_json_loads(payload)
    except (OSError, ValueError) as exc:
        raise EvalContractError(
            f"publication expectations missing/unreadable: {expected_path}") from exc
    if (not isinstance(document, dict)
            or set(document) != {"schema_version", "expected_provenance"}
            or document["schema_version"] != _PUBLICATION_EXPECTATIONS_SCHEMA
            or not isinstance(document["expected_provenance"], dict)
            or not document["expected_provenance"]):
        raise EvalContractError(
            "publication expectations schema/fields are illegal")
    return document["expected_provenance"], {
        "path": str(expected_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_publication_expectations(identity: dict | None) -> None:
    if identity is None:
        return
    path = pathlib.Path(identity["path"])
    if not path.is_file():
        raise EvalContractError(
            f"publication expectations disappeared during evaluation: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != identity["sha256"]:
        raise EvalContractError(
            "publication expectations changed during evaluation: "
            f"{actual} != {identity['sha256']}")


def capture_published_worker(
        checkpoint: pathlib.Path, checkpoint_payload: bytes, *,
        expected_manager_sha256: str,
        expected_implementation_sha256: str,
        expected_provenance: dict) -> bytes:
    """Re-verify the whole chain of the official Worker publication transaction, training contract and liveness.

    The existence of a sibling JSON does not make an official candidate. This function independently reconciles the model SHA,
    exact step count, current implementation, M29, the rev11 contract inside the checkpoint and the PASS preflight,
    and requires its provenance to match this case's expectation subset frozen before launch.
    """
    receipt_path = checkpoint.with_name(PUBLISHED_WORKER_RECEIPT_NAME)
    preflight_path = checkpoint.with_name("bc_aux_liveness_preflight.json")
    try:
        receipt_payload = receipt_path.read_bytes()
        preflight_payload = preflight_path.read_bytes()
        receipt = strict_json_loads(receipt_payload)
        preflight = strict_json_loads(preflight_payload)
        with zipfile.ZipFile(io.BytesIO(checkpoint_payload)) as archive:
            checkpoint_data = strict_json_loads(archive.read("data"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise EvalContractError(
            "official Worker checkpoint/publication receipt/liveness is incomplete") from exc

    from train_ppo import (
        _BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION,
        _BC_AUX_CIRCUIT_KING_SUPPORT,
        _BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION,
        _BC_AUX_MIN_ACTUAL_A12_SAMPLES,
        _BC_AUX_MIN_EXPECTED_A12_SAMPLES,
        _BC_AUX_OBJECTIVE_REVISION,
        _CONTRACT_REVISION,
        _bc_aux_circuit_spec,
        _canonical_json_sha256,
        _validate_publication_provenance,
        bc_aux_behavior_gate,
    )

    def require(condition, message):
        if not condition:
            raise EvalContractError(message)

    model_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    step = checkpoint_num_timesteps_bytes(
        checkpoint_payload, str(checkpoint))
    # A4 (2026-07-27): the contract moved to rev26 (the kl_early_stopped flag); the certified range moves with it.
    require(_BC_AUX_OBJECTIVE_REVISION == 11
            and _CONTRACT_REVISION in {
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26},
            "the evaluator only certifies objective-rev11/contract-rev12-through-26")
    require(isinstance(receipt, dict)
            and set(receipt) == _PUBLISHED_RECEIPT_KEYS,
            "Worker publication receipt fields/schema are not exact")
    require(
        receipt["schema_version"]
        == _BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION,
        "Worker publication receipt schema is outdated")
    require(receipt["publication"] == "PUBLISHED"
            and receipt["save_error"] is None,
            "Worker publication transaction did not complete successfully")
    require(receipt["model_sha256"] == model_sha256,
            "Worker publication receipt is not bound to the current checkpoint bytes")
    require(_is_plain_int(receipt["step"])
            and receipt["step"] == step,
            "Worker publication receipt step count disagrees with the checkpoint")
    require(_is_plain_int(receipt["objective_revision"])
            and receipt["objective_revision"]
            == _BC_AUX_OBJECTIVE_REVISION,
            "Worker publication receipt auxiliary objective revision is outdated")
    require(receipt["evaluation_scope"]
            == "original-bc-v2-heldout-episodes"
            and receipt["mask_mode"] == "bc-v2-recorded",
            "Worker final behavior gate evaluation domain/mask definition is malformed")
    require(_is_sha256(receipt["demos_sha256"])
            and _is_sha256(receipt["candidate_policy_head_sha256"]),
            "Worker publication receipt lacks the demos/policy-head SHA")
    anchor = receipt["anchor"]
    require(isinstance(anchor, dict)
            and set(anchor) == {"identity", "policy_head_sha256"}
            and anchor["identity"] == "bc-aux-root-policy"
            and _is_sha256(anchor["policy_head_sha256"]),
            "Worker publication receipt root policy anchor is malformed")

    try:
        provenance = _validate_publication_provenance(
            receipt["provenance"],
            demos_sha256=receipt["demos_sha256"],
            final_step=step)
    except ValueError as exc:
        raise EvalContractError(
            f"Worker publication lineage is illegal: {exc}") from exc
    require(provenance["implementation_sha256"]
            == expected_implementation_sha256,
            "Worker publication implementation disagrees with the current training/runtime implementation")
    require(provenance["manager_npz_sha256"]
            == expected_manager_sha256,
            "Worker publication manager disagrees with this evaluation's manager")
    allowed_expectation_keys = set(provenance)
    require(set(expected_provenance).issubset(allowed_expectation_keys),
            "publication expectations contain an unknown provenance key")
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    require(not mismatches,
            f"Worker publication lineage does not match this pre-registered case: {mismatches}")

    evidence = receipt["exploration_evidence"]
    require(isinstance(evidence, dict)
            and set(evidence) == {
                "eligible_states", "expected_a12_mass", "requested_a12",
                "sampled_a12", "rejected_a12",
                "unexpected_sampled_a12", "minimum_expected_a12_mass",
                "minimum_actual_a12_samples", "information_status", "reasons",
            },
            "Worker publication exploration-sample evidence fields/schema are not exact")
    eligible_states = evidence.get("eligible_states")
    requested_a12 = evidence.get("requested_a12")
    sampled_a12 = evidence.get("sampled_a12")
    rejected_a12 = evidence.get("rejected_a12")
    unexpected_sampled_a12 = evidence.get("unexpected_sampled_a12")
    expected_a12_mass = evidence.get("expected_a12_mass")
    reported_min_expected = evidence.get("minimum_expected_a12_mass")
    reported_min_actual = evidence.get("minimum_actual_a12_samples")
    require(
        _is_plain_int(eligible_states) and eligible_states >= 0
        and _is_plain_int(requested_a12)
        and 0 <= requested_a12 <= eligible_states
        and _is_plain_int(sampled_a12)
        and 0 <= sampled_a12 <= requested_a12
        and _is_plain_int(rejected_a12)
        and 0 <= rejected_a12 <= requested_a12
        and requested_a12 == sampled_a12 + rejected_a12
        and _is_plain_int(unexpected_sampled_a12)
        and unexpected_sampled_a12 == 0
        and _finite_number(expected_a12_mass)
        and 0.0 <= float(expected_a12_mass) <= float(eligible_states)
        and _finite_number(reported_min_expected)
        and float(reported_min_expected)
        >= float(_BC_AUX_MIN_EXPECTED_A12_SAMPLES)
        and _is_plain_int(reported_min_actual)
        and reported_min_actual >= _BC_AUX_MIN_ACTUAL_A12_SAMPLES
        and float(expected_a12_mass) >= max(
            float(reported_min_expected),
            float(_BC_AUX_MIN_EXPECTED_A12_SAMPLES))
        and sampled_a12 >= max(
            reported_min_actual, _BC_AUX_MIN_ACTUAL_A12_SAMPLES)
        and evidence.get("information_status") == "INFORMATIVE"
        and evidence.get("reasons") == [],
        "Worker publication exploration-sample evidence is insufficient/illegal")

    contract = checkpoint_data.get("diablogym_contract")
    require(isinstance(contract, dict)
            and _is_plain_int(contract.get("contract_revision"))
            and contract.get("contract_revision")
            in {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26},
            "Worker checkpoint is not a supported rev12-rev26 official training contract")
    if contract["contract_revision"] >= 24:
        try:
            from train_ppo import _validate_policy_source_roles
            _validate_policy_source_roles(contract)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise EvalContractError(
                f"rev{contract['contract_revision']} Worker "
                "policy source roles are not closed") from exc
    if contract["contract_revision"] >= 13:
        require(
            contract.get("artifact_scope") == "production"
            and contract.get("worker_fast_forward_reward_credit")
            in {"none", "terminal-death-only"}
            and _finite_number(
                contract.get("worker_additional_terminal_death_cost"))
            and float(contract.get("worker_additional_terminal_death_cost"))
            >= 0.0,
            "rev13 Worker checkpoint lacks the production scope/terminal reward contract",
        )
    if contract["contract_revision"] >= 14:
        require(
            contract.get("legacy_policy_observation_view") is False,
            "rev14+ official A12 Worker must keep the registered gate overlay observation",
        )
    if contract["contract_revision"] >= 16:
        clipping = contract.get("gradient_clipping")
        critic_migration = contract.get("critic_migration")
        expected_boundary = (
            "base-game-terminal-plus-no-progress-timeout-failure"
            if contract["contract_revision"] >= 24
            else "base-game-terminal-only"
        )
        timeout_contract_closed = (
            contract.get("worker_no_progress_timeout") == {
                "boundary": "terminated-no-bootstrap",
                "reward": (
                    "death-ladder-base-plus-"
                    "additional-terminal-death-cost"
                ),
            }
            if contract["contract_revision"] >= 24
            else True
        )
        require(
            contract.get("manager_policy_observation_view") == "legacy-v3"
            and contract.get("worker_episode_boundary")
            == expected_boundary
            and contract.get("worker_window_bootstrap")
            == "next-learning-window"
            and timeout_contract_closed
            and isinstance(clipping, dict)
            and clipping.get("mode")
            in {
                "global",
                "separate-actor-critic-v1",
                "separate-root-context-critic-v2",
            }
            and (
                critic_migration == "disabled"
                or (
                    isinstance(critic_migration, dict)
                    and critic_migration.get("method")
                    in {
                        "sb3-orthogonal-value-only-v1",
                        "structured-layout-v1-orthogonal-value-only-v2",
                    }
                    and critic_migration.get("gradient_clip_mode")
                    in {
                        "separate-actor-critic-v1",
                        "separate-root-context-critic-v2",
                    }
                    and _is_sha256(
                        critic_migration.get("source_checkpoint_sha256"))
                )
            ),
            "rev16 Worker checkpoint lacks the manager-view/termination/bootstrap/clipping contract",
        )
    if contract["contract_revision"] >= 17:
        require(
            contract.get("worker_policy_observation_view")
            == "legacy-v3-a12-overlay"
            and contract.get("actor_migration") == "disabled",
            "rev17+ official A12 Worker must bind the overlay and forbid asymmetric migration",
        )
    require(
        _canonical_json_sha256(contract)
        == provenance["training_contract_sha256"],
        "Worker checkpoint contract disagrees with the publication lineage SHA")
    require(contract.get("mode") == "worker"
            and _is_plain_int(contract.get("max_steps"))
            and contract.get("max_steps") == PROTOCOL_MAX_STEPS
            and _finite_number(contract.get("gamma"))
            and float(contract.get("gamma")) == 1.0
            and contract.get("observation_shape") == [298]
            and _is_plain_int(contract.get("action_n"))
            and contract.get("action_n") == 15,
            "Worker checkpoint environment/space recipe is malformed")
    require(contract.get("implementation_sha256")
            == provenance["implementation_sha256"]
            and contract.get("manager_npz_sha256")
            == provenance["manager_npz_sha256"]
            and contract.get("teacher_sha256")
            == provenance["teacher_sha256"]
            and contract.get("distill_beta")
            == provenance["distill_beta"]
            and contract.get("calib_record_only") is False,
            "Worker checkpoint implementation/manager/teacher/leash contract disagrees with the receipt")
    recipe = contract.get("algorithm_recipe")
    require(isinstance(recipe, dict)
            and recipe.get("target_kl") == provenance["target_kl"],
            "Worker checkpoint target_kl disagrees with the publication lineage")
    aux = contract.get("bc_aux")
    require(isinstance(aux, dict)
            and _is_plain_int(aux.get("objective_revision"))
            and aux.get("objective_revision")
            == _BC_AUX_OBJECTIVE_REVISION
            and aux.get("demos_sha256") == provenance["bc_aux_demos_sha256"]
            and _finite_number(aux.get("lambda"))
            and aux.get("lambda") == provenance["bc_aux_lambda"]
            and aux.get("mode") == provenance["bc_aux_mode"]
            and aux.get("mode")
            == "expanded-trainable-a12-contextual-mixture"
            and aux.get("circuit") == _bc_aux_circuit_spec()
            and aux.get("king_support")
            == _BC_AUX_CIRCUIT_KING_SUPPORT
            and _is_plain_int(aux.get("aux_optimizer_calls_per_rollout"))
            and aux.get("aux_optimizer_calls_per_rollout") == 0
            and _is_plain_int(aux.get("trainable_adapter_parameters"))
            and aux.get("trainable_adapter_parameters") == 5
            and aux.get("initial_calibration")
            == "exact-five-percent-contextual-legal-support-mixture"
            and aux.get("post_step_projection") == {
                "gate_parameter_abs_max":
                    _bc_aux_circuit_spec()["gate_parameter_abs_max"],
                "probability_min":
                    _bc_aux_circuit_spec()["probability_min"],
                "probability_max":
                    _bc_aux_circuit_spec()["probability_max"],
            }
            and aux.get("liveness_preflight") is True,
            "Worker checkpoint bc_aux contract disagrees with the publication lineage")
    policy_class_record = checkpoint_data.get("policy_class")
    policy_kwargs_record = checkpoint_data.get("policy_kwargs")
    require(
        isinstance(policy_class_record, dict)
        and policy_class_record.get("__module__") == "leashed_ppo"
        and isinstance(policy_kwargs_record, dict)
        and policy_kwargs_record.get("bc_aux_mixture_spec")
        == _bc_aux_circuit_spec()
        and policy_kwargs_record.get("net_arch")
        == {"pi": [68, 68], "vf": [64, 64]},
        "Worker checkpoint metadata is not bound to the rev11 custom policy/spec")
    require(_is_plain_int(checkpoint_data.get("num_timesteps"))
            and checkpoint_data.get("num_timesteps") == step
            and checkpoint_data.get("distill_beta")
            == provenance["distill_beta"]
            and checkpoint_data.get("teacher_sha256")
            == provenance["teacher_sha256"]
            and _finite_number(checkpoint_data.get("bc_aux_lambda"))
            and checkpoint_data.get("bc_aux_lambda")
            == provenance["bc_aux_lambda"]
            and checkpoint_data.get("_bc_aux_circuit_spec")
            == _bc_aux_circuit_spec()
            and _is_plain_int(
                checkpoint_data.get("_bc_aux_eligible_states"))
            and checkpoint_data.get("_bc_aux_eligible_states")
            == eligible_states
            and _is_plain_int(
                checkpoint_data.get("_bc_aux_requested_a12"))
            and checkpoint_data.get("_bc_aux_requested_a12")
            == requested_a12
            and _is_plain_int(
                checkpoint_data.get("_bc_aux_sampled_a12"))
            and checkpoint_data.get("_bc_aux_sampled_a12")
            == sampled_a12
            and _is_plain_int(
                checkpoint_data.get("_bc_aux_rejected_a12"))
            and checkpoint_data.get("_bc_aux_rejected_a12")
            == rejected_a12
            and _is_plain_int(checkpoint_data.get(
                "_bc_aux_unexpected_sampled_a12"))
            and checkpoint_data.get("_bc_aux_unexpected_sampled_a12")
            == unexpected_sampled_a12
            and _finite_number(checkpoint_data.get(
                "_bc_aux_expected_a12_mass"))
            and checkpoint_data.get("_bc_aux_expected_a12_mass")
            == expected_a12_mass
            and _is_plain_int(
                checkpoint_data.get("_ppo_optimizer_steps_completed"))
            and checkpoint_data.get("_ppo_optimizer_steps_completed") > 0
            and _is_plain_int(
                checkpoint_data.get("_last_completed_ppo_rollout_steps"))
            and checkpoint_data.get("_last_completed_ppo_rollout_steps")
            == provenance["target_global_steps"],
            "Worker checkpoint runtime state disagrees with the publication lineage")

    require(isinstance(receipt["metrics"], dict),
            "Worker publication receipt metrics is not an object")
    independently_observed = _recompute_published_worker_evidence(
        checkpoint_payload,
        demos_sha256=receipt["demos_sha256"],
        expected_manager_sha256=expected_manager_sha256)
    require(
        independently_observed.get("candidate_policy_head_sha256")
        == receipt["candidate_policy_head_sha256"]
        and independently_observed.get("anchor_policy_head_sha256")
        == anchor["policy_head_sha256"]
        and independently_observed.get("metrics") == receipt["metrics"],
        "Worker publication receipt policy head/root anchor/metrics are not bound to an on-site recomputation from checkpoint+demos")
    recomputed_gate = bc_aux_behavior_gate(
        independently_observed["metrics"], require_root_anchor=True,
        require_teacher_recall=False, require_deployable_a12=False)
    reported_thresholds = (
        receipt["gate"].get("thresholds")
        if isinstance(receipt["gate"], dict) else None)
    require(
        isinstance(reported_thresholds, dict)
        and reported_thresholds.get("deployable_a12_required") is False
        and reported_thresholds.get("deterministic_a12_episode_min") is None
        and reported_thresholds.get("deterministic_a12_margin_min") is None,
        "Worker publication receipt does not explicitly declare deterministic a12 as a non-publication gate")
    require(receipt["gate"] == recomputed_gate
            and recomputed_gate["verdict"] == "PASS",
            "Worker publication safety behavior hard gate is not PASS or disagrees with the on-site metrics")

    require(
        hashlib.sha256(preflight_payload).hexdigest()
        == provenance["bc_aux_liveness_preflight_sha256"],
        "Worker liveness receipt SHA disagrees with the publication lineage")
    require(isinstance(preflight, dict)
            and preflight.get("schema_version")
            == _BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION
            and preflight.get("protocol_version") == PROTOCOL_VERSION
            and _is_plain_int(preflight.get("objective_revision"))
            and preflight.get("objective_revision")
            == _BC_AUX_OBJECTIVE_REVISION
            and preflight.get("status") == "PASS"
            and preflight.get("simulation")
            == "isolated-exact-mixture-with-policy-gradient-canary"
            and preflight.get("installation")
            in {"first-install", "preserved-continuation"}
            and _is_plain_int(preflight.get("heldout_rows_consumed"))
            and preflight.get("heldout_rows_consumed") == 0,
            "Worker liveness receipt status/schema/isolation domain is malformed")
    installation = preflight["installation"]
    expected_calibrations = 1 if installation == "first-install" else 0
    expected_topology_reset = installation == "first-install"
    expected_initializer = (
        "exact-contextual-legal-support-mixture"
        if installation == "first-install"
        else "preserved-continuation")
    exact_circuit = {
        **_bc_aux_circuit_spec(),
        "king_support": _BC_AUX_CIRCUIT_KING_SUPPORT,
    }
    require(preflight.get("circuit") == exact_circuit,
            "Worker liveness top-level circuit spec is not the current exact rev11 spec")
    inputs = preflight.get("inputs")
    require(isinstance(inputs, dict)
            and inputs.get("resume_checkpoint_sha256")
            == provenance["resume_checkpoint_sha256"]
            and inputs.get("demos_sha256")
            == provenance["bc_aux_demos_sha256"]
            and inputs.get("manager_npz_sha256")
            == provenance["manager_npz_sha256"]
            and inputs.get("implementation_sha256")
            == provenance["implementation_sha256"],
            "Worker liveness input lineage is not closed")
    config = preflight.get("config")
    leg_steps = provenance["target_global_steps"] - provenance["start_steps"]
    require(isinstance(config, dict)
            and _is_plain_int(config.get("total_steps"))
            and config.get("total_steps") == leg_steps
            and _finite_number(config.get("bc_aux_lambda"))
            and config.get("bc_aux_lambda") == provenance["bc_aux_lambda"]
            and config.get("mechanism") == provenance["bc_aux_mode"]
            and config.get("mechanism")
            == "expanded-trainable-a12-contextual-mixture"
            and config.get("circuit") == exact_circuit
            and _finite_number(config.get("distill_beta"))
            and config.get("distill_beta") == provenance["distill_beta"]
            and config.get("target_kl") == provenance["target_kl"]
            and (
                config.get("seed") is None
                or (_is_plain_int(config.get("seed"))
                    and 0 <= config.get("seed") < 2**32)
            )
            and config.get("seed") == provenance["seed"]
            and isinstance(config.get("reset_optimizer"), bool)
            and config.get("reset_optimizer")
            is provenance["optimizer_reset"],
            "Worker liveness recipe disagrees with the official leg")
    calls = preflight.get("calls")
    require(isinstance(calls, dict)
            and _is_plain_int(calls.get("aux_optimizer_calls"))
            and calls.get("aux_optimizer_calls") == 0
            and _is_plain_int(
                calls.get("policy_gradient_canary_calls"))
            and calls.get("policy_gradient_canary_calls") == 1
            and _is_plain_int(calls.get("planned_train_calls"))
            and calls.get("planned_train_calls") > 0
            and _is_plain_int(calls.get("initial_adapter_calibrations"))
            and calls.get("initial_adapter_calibrations")
            == expected_calibrations
            and _is_plain_int(calls.get("trainable_adapter_parameters"))
            and calls.get("trainable_adapter_parameters") == 5,
            "Worker structural liveness call ledger is not closed")
    require(
        _is_plain_int(config.get("n_steps"))
        and config.get("n_steps") > 0
        and _is_plain_int(config.get("num_envs"))
        and config.get("num_envs") > 0
        and _is_plain_int(config.get("rollout_quantum"))
        and config.get("rollout_quantum")
        == config.get("n_steps") * config.get("num_envs")
        and _is_plain_int(config.get("train_calls"))
        and config.get("train_calls") == calls.get("planned_train_calls")
        and config.get("total_steps")
        == config.get("rollout_quantum") * config.get("train_calls")
        and _is_plain_int(config.get("aux_optimizer_calls"))
        and config.get("aux_optimizer_calls")
        == calls.get("aux_optimizer_calls") == 0
        and _is_plain_int(config.get("policy_gradient_canary_calls"))
        and config.get("policy_gradient_canary_calls")
        == calls.get("policy_gradient_canary_calls") == 1
        and _is_plain_int(config.get("initial_adapter_calibrations"))
        and config.get("initial_adapter_calibrations")
        == calls.get("initial_adapter_calibrations")
        == expected_calibrations
        and _is_plain_int(config.get("trainable_adapter_parameters"))
        and config.get("trainable_adapter_parameters")
        == calls.get("trainable_adapter_parameters") == 5,
        "Worker liveness rollout/call/parameter counts are not closed")
    canary = preflight.get("policy_gradient_canary")
    require(
        isinstance(canary, dict)
        and canary.get("schema_version")
        == "a12-policy-gradient-canary/1"
        and canary.get("scope")
        == "bc-v2-nested-validation-positive-only"
        and _is_plain_int(canary.get("pairs"))
        and canary.get("pairs") > 0
        and _is_plain_int(canary.get("heldout_rows_consumed"))
        and canary.get("heldout_rows_consumed") == 0
        and canary.get("objective")
        == "negative-mean-log-probability-action12"
        and _is_plain_int(canary.get("optimizer_steps"))
        and canary.get("optimizer_steps") == 1
        and isinstance(canary.get("movement_required"), bool)
        and canary.get("state_restored") is True
        and _is_sha256(canary.get("start_policy_head_sha256"))
        and _is_sha256(canary.get("stepped_policy_head_sha256")),
        "Worker liveness policy-gradient canary schema/isolation ledger is malformed")
    for key in (
            "probability_12_before", "probability_12_after",
            "probability_12_delta", "gate_bias_before", "gate_bias_after",
            "gate_bias_delta", "gate_bias_gradient",
            "gradient_norm_before_clip"):
        require(
            _finite_number(canary.get(key)),
            f"Worker liveness policy-gradient canary {key} is not finite")
    require(
        0.0 < canary["probability_12_before"] < 1.0
        and canary["start_policy_head_sha256"]
        == preflight.get("policy", {}).get("grafted_head_sha256")
        and canary["probability_12_after"]
        >= canary["probability_12_before"]
        and math.isclose(
            canary["probability_12_delta"],
            canary["probability_12_after"]
            - canary["probability_12_before"],
            rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(
            canary["gate_bias_delta"],
            canary["gate_bias_after"] - canary["gate_bias_before"],
            rel_tol=0.0, abs_tol=1e-15)
        and canary["gate_bias_gradient"] < 0.0
        and canary["gradient_norm_before_clip"] > 0.0
        and (
            not canary["movement_required"]
            or (
                canary["probability_12_delta"] > 0.0
                and canary["gate_bias_delta"] > 0.0
                and canary["stepped_policy_head_sha256"]
                != canary["start_policy_head_sha256"]
            )
        ),
        "Worker liveness policy-gradient canary did not prove the optimizer can raise p(a12)")
    optimizer = preflight.get("optimizer")
    require(
        isinstance(optimizer, dict)
        and optimizer.get("reset_after_topology_change")
        is expected_topology_reset,
        "Worker liveness optimizer topology-reset state disagrees with the installed type")
    policy = preflight.get("policy")
    require(
        isinstance(policy, dict)
        and _is_sha256(policy.get("start_head_sha256"))
        and _is_sha256(policy.get("root_head_sha256"))
        and _is_sha256(policy.get("grafted_head_sha256"))
        and (
            installation != "first-install"
            or policy.get("start_head_sha256")
            == policy.get("root_head_sha256")
        )
        and _is_plain_int(policy.get("actor_width_before"))
        and policy.get("actor_width_before")
        == (64 if installation == "first-install" else 68)
        and _is_plain_int(policy.get("actor_width_after"))
        and policy.get("actor_width_after") == 68,
        "Worker liveness policy topology/hash ledger is not closed")
    calibration = preflight.get("calibration")
    require(
        isinstance(calibration, dict)
        and calibration.get("initializer") == expected_initializer
        and calibration.get("candidate_policy_head_sha256")
        == policy.get("grafted_head_sha256"),
        "Worker liveness calibration/continuation identity is not closed")
    liveness_metrics = preflight.get("metrics")
    liveness_gate = preflight.get("gate")
    recomputed_liveness_gate = bc_aux_behavior_gate(
        liveness_metrics, require_root_anchor=True,
        require_teacher_recall=False, require_deployable_a12=False)
    require(
        isinstance(liveness_gate, dict)
        and calibration.get("validation_metrics") == liveness_metrics
        and calibration.get("validation_gate") == liveness_gate
        and liveness_gate == recomputed_liveness_gate
        and recomputed_liveness_gate.get("verdict") == "PASS",
        "Worker liveness behavior gate was not recomputed on site as PASS from its metrics")
    return receipt_payload


def worker_label(spec: str) -> str:
    if spec in {"script", "bc"}:
        return spec
    if pathlib.Path(spec).suffix.lower() == ".npz":
        return pathlib.Path(spec).parent.name
    return pathlib.Path(spec).stem


def _asymmetric_worker_observation_dim_for_revision(
        contract_revision: int, current_observation_dim: int) -> int:
    """Keep historical wire widths immutable; rev22+ tracks current layout."""
    if contract_revision in {17, 18}:
        return 635
    if contract_revision == 19:
        return 5448
    if contract_revision in {20, 21}:
        return 9100
    if contract_revision >= 22:
        if not _is_plain_int(current_observation_dim) \
                or current_observation_dim <= 0:
            raise EvalContractError(
                "rev22+ current asymmetric observation dim is illegal")
        return current_observation_dim
    raise EvalContractError(
        f"unsupported asymmetric Worker contract revision: "
        f"{contract_revision!r}")


def _validate_asymmetric_worker_runtime_state(
        model, *, contract_revision: int) -> None:
    """Reject an asymmetric checkpoint whose actor never left warmup.

    The training contract is installed before the first rollout, so the
    contract and policy class alone do not distinguish a deployable final model
    from a critic-only checkpoint.  Only persisted optimizer/rollout receipts
    plus the live context branch can close that distinction after ``load()``.
    """
    import torch as th

    fields = {
        "num_timesteps": getattr(model, "num_timesteps", None),
        "last_completed_rollout": getattr(
            model, "_last_completed_ppo_rollout_steps", None),
        "ppo_optimizer_steps": getattr(
            model, "_ppo_optimizer_steps_completed", None),
        "warmup_start": getattr(
            model, "_critic_warmup_start_timesteps", None),
        "warmup_until": getattr(
            model, "_critic_warmup_until_timesteps", None),
        "warmup_expected_rollouts": getattr(
            model, "_critic_warmup_expected_rollouts", None),
        "warmup_completed_rollouts": getattr(
            model, "_critic_warmup_rollouts_completed", None),
        "warmup_optimizer_steps": getattr(
            model, "_critic_warmup_optimizer_steps_completed", None),
        "actor_optimizer_steps": getattr(
            model, "_actor_optimizer_steps_completed", None),
    }
    integer_fields = tuple(fields.values())
    state_closed = (
        all(_is_plain_int(value) for value in integer_fields)
        and fields["num_timesteps"] >= 0
        and fields["last_completed_rollout"] == fields["num_timesteps"]
        and fields["ppo_optimizer_steps"] > 0
        and fields["warmup_start"] >= 0
        and fields["warmup_until"] > fields["warmup_start"]
        and fields["num_timesteps"] > fields["warmup_until"]
        and fields["warmup_expected_rollouts"] > 0
        and fields["warmup_completed_rollouts"]
        == fields["warmup_expected_rollouts"]
        and fields["warmup_optimizer_steps"] > 0
        and fields["actor_optimizer_steps"] > 0
        and fields["ppo_optimizer_steps"]
        == fields["warmup_optimizer_steps"]
        + fields["actor_optimizer_steps"]
        and getattr(model, "_critic_warmup_completed", None) is True
    )
    if not state_closed:
        raise EvalContractError(
            "asymmetric Worker checkpoint has not finished deployable actor training, "
            f"or is not at a full PPO update boundary: {fields}")

    if contract_revision >= 22:
        try:
            from train_ppo import (
                _asymmetric_worker_deployment_evidence_complete,
            )
            context_closed = (
                _asymmetric_worker_deployment_evidence_complete(model))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            context_closed = False
        if not context_closed:
            raise EvalContractError(
                "rev22+ asymmetric Worker structured actor/critic "
                "runtime evidence is not closed")
        return

    # rev20/21 historical checkpoints used the old dense projection/fusion
    # topology.  Keep their immutable deployment rule separate from rev22+'s
    # structured helper rather than reinterpreting old artifacts.
    try:
        from leashed_ppo import (
            ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
            ASYMMETRIC_WORKER_LEGACY_DIM,
        )
        extractor = model.policy.mlp_extractor
        adapter = extractor.context_adapter
        projection = adapter.context_projection.weight.detach()
        fusion = adapter.fusion.weight.detach()
        output = adapter.output.weight.detach()
        action_weight = model.policy.action_net.weight.detach()
        action_effect = action_weight @ output
        context_fusion = fusion[:, extractor.latent_dim_pi:]
        structural_context_effect = (
            action_effect @ context_fusion @ projection)
        anneal_rollouts = getattr(
            model, "distill_anneal_actor_rollouts", None)
        anneal_completed = getattr(
            model, "_distill_actor_rollouts_completed", None)
        last_effective_beta = getattr(
            model, "_last_effective_distill_beta", None)
        anneal_closed = (
            (
                float(getattr(model, "distill_beta", 0.0)) == 0.0
                and anneal_rollouts == 0
            )
            or (
                isinstance(anneal_rollouts, int)
                and not isinstance(anneal_rollouts, bool)
                and anneal_rollouts >= 2
                and isinstance(anneal_completed, int)
                and not isinstance(anneal_completed, bool)
                and anneal_completed >= anneal_rollouts
                and isinstance(last_effective_beta, (int, float))
                and math.isfinite(float(last_effective_beta))
                and float(last_effective_beta) == 0.0
            )
        )
        excluded_columns_zero = all(
            int(th.count_nonzero(
                projection[:, index - ASYMMETRIC_WORKER_LEGACY_DIM]
            ).item()) == 0
            for index in ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES
        )
        context_closed = (
            extractor.actor_context_enabled is True
            and projection.ndim == 2
            and fusion.ndim == 2
            and output.ndim == 2
            and action_weight.ndim == 2
            and action_weight.shape[1] == output.shape[0]
            and all(
                bool(th.isfinite(parameter).all().item())
                for parameter in adapter.parameters()
            )
            and bool(th.isfinite(action_effect).all().item())
            and bool(th.isfinite(
                structural_context_effect).all().item())
            and int(th.count_nonzero(projection).item()) > 0
            and int(th.count_nonzero(fusion).item()) > 0
            and int(th.count_nonzero(output).item()) > 0
            and int(th.count_nonzero(action_effect).item()) > 0
            and int(th.count_nonzero(
                structural_context_effect).item()) > 0
            and excluded_columns_zero
            and anneal_closed
        )
    except (AttributeError, RuntimeError, TypeError):
        context_closed = False
    if not context_closed:
        raise EvalContractError(
            "asymmetric Worker checkpoint actor context "
            "is still closed, zero, or cannot produce a finite action effect")


def load_worker(
        spec: str, protocol_bundle_sha256: str, *,
        require_published: bool = False,
        expected_manager_sha256: str | None = None,
        expected_implementation_sha256: str | None = None,
        expected_provenance: dict | None = None,
        worker_decoding: str = "argmax"):
    """Return (workers, label, identity manifest). spec in script | bc | *.npz | SB3 zip.

    worker_decoding (R16/C3): argmax = old law, bit-identical; sample is defined only for SB3
    checkpoints (numpy/BC/script workers have no torch distribution to sample from).
    """
    if worker_decoding not in ("argmax", "sample"):
        raise EvalContractError(
            f"worker_decoding only allows argmax/sample: {worker_decoding!r}")
    if worker_decoding == "sample" and (
            spec in {"script", "bc"}
            or pathlib.Path(spec).suffix.lower() == ".npz"):
        raise EvalContractError(
            "--worker-decoding sample only applies to SB3 checkpoint workers")
    if spec == "script":
        return None, "script", script_worker_identity(protocol_bundle_sha256)
    if spec == "bc":
        sd_path = (ROOT / "train" / "runs" / "bc-worker" / "policy_sd.pt").resolve()
        policy_payload, report_payload = capture_passed_bc(sd_path)
        policy_sha256 = hashlib.sha256(policy_payload).hexdigest()
        identity = {
            "kind": "bc_state_dict", "path": str(sd_path),
            "sha256": policy_sha256, "num_timesteps": None,
            "gate_report_sha256": hashlib.sha256(report_payload).hexdigest(),
        }
        net = np_policy_from_sd(policy_payload, policy_sha256)

        def bc_worker(policy_obs, raw_mask):
            logits = net.forensic_worker_logits(policy_obs)
            policy_mask = np.asarray(raw_mask, dtype=bool)
            if policy_mask.shape != (15,):
                raise EvalContractError(
                    "BC Worker evaluation action mask must be (15,)")
            policy_mask = policy_mask.copy()
            policy_mask[12] = False
            if not policy_mask.any():
                raise EvalContractError(
                    "BC Worker action mask is all False after applying the permanent a12 mask")
            return int(np.argmax(np.where(policy_mask, logits, -np.inf)))

        # OptionsEnv must construct this view from lossless native raw state;
        # the BC state_dict itself contains no observation encoder.
        bc_worker.diablogym_worker_observation_view = "legacy-v3"
        bc_worker.diablogym_worker_action12_mode = "permanently-masked"
        return {FARM: bc_worker}, "bc", identity
    if pathlib.Path(spec).suffix.lower() == ".npz":
        path = pathlib.Path(spec).resolve()
        manager_class, _env_class, _bridge = _native_runtime()
        net = manager_class(path)  # Single read: the same bytes serve both identity and forward pass.
        identity = {
            "kind": "numpy_policy", "path": str(path),
            "sha256": net.source_sha256, "num_timesteps": None,
            "gate_report_sha256": None,
        }
        net.require_worker_contract()
        return ({FARM: net.worker_callback()},
                pathlib.Path(spec).parent.name, identity)
    from sb3_contrib import MaskablePPO
    checkpoint = resolve_checkpoint_file(spec)
    checkpoint_payload = checkpoint.read_bytes()
    receipt_sha256 = None
    receipt_path = checkpoint.with_name(PUBLISHED_WORKER_RECEIPT_NAME)
    if require_published:
        if (expected_manager_sha256 is None
                or expected_implementation_sha256 is None
                or expected_provenance is None):
            raise EvalContractError(
                "official Worker evaluation lacks manager/implementation/pre-registered lineage expectations")
        receipt_payload = capture_published_worker(
            checkpoint, checkpoint_payload,
            expected_manager_sha256=expected_manager_sha256,
            expected_implementation_sha256=expected_implementation_sha256,
            expected_provenance=expected_provenance)
        receipt_sha256 = hashlib.sha256(receipt_payload).hexdigest()
    elif receipt_path.is_file():
        raise EvalContractError(
            "detected an SB3 Worker with a publication receipt; must explicitly pass "
            "--require-published-worker and --publication-expectations")
    identity = {
        "kind": "sb3_checkpoint", "path": str(checkpoint),
        "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
        "num_timesteps": checkpoint_num_timesteps_bytes(
            checkpoint_payload, str(checkpoint)),
        "gate_report_sha256": receipt_sha256,
    }
    from leashed_ppo import (
        ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        A12MixtureMaskableActorCriticPolicy,
        AsymmetricWorkerMaskableActorCriticPolicy,
        asymmetric_worker_runtime_evidence,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )
    if require_published:
        from leashed_ppo import LeashedMaskablePPO
        model = LeashedMaskablePPO.load(
            io.BytesIO(checkpoint_payload), env=None, device="cpu",
            teacher_path=None, teacher_sha256=None)
    else:
        model = MaskablePPO.load(io.BytesIO(checkpoint_payload), device="cpu")

    custom_a12_policy = isinstance(
        model.policy, A12MixtureMaskableActorCriticPolicy)
    asymmetric_policy = isinstance(
        model.policy, AsymmetricWorkerMaskableActorCriticPolicy)
    contract = getattr(model, "diablogym_contract", None)
    contract_revision = (
        contract.get("contract_revision")
        if isinstance(contract, dict) else None)
    if contract_revision in _DUAL_WORKER_PG_AUDIT_SCHEMA_BY_CONTRACT_REVISION:
        try:
            from train_ppo import (
                _WORKER_EPISODE_BOUNDARY_V24,
                _WORKER_NO_PROGRESS_TIMEOUT_CONTRACT,
                _validate_policy_source_roles,
            )
            _validate_policy_source_roles(contract)
            if (
                contract.get("worker_episode_boundary")
                != _WORKER_EPISODE_BOUNDARY_V24
                or contract.get("worker_window_bootstrap")
                != "next-learning-window"
                or contract.get("worker_no_progress_timeout")
                != _WORKER_NO_PROGRESS_TIMEOUT_CONTRACT
            ):
                raise ValueError(
                    f"rev{contract_revision} Worker "
                    "timeout boundary/reward contract drift")
        except (RuntimeError, TypeError, ValueError) as exc:
            raise EvalContractError(
                f"rev{contract_revision} Worker "
                "policy source/timeout roles are not closed") from exc
    if (
        asymmetric_policy
        and contract_revision in {17, 18, 19, 20, 21}
    ):
        # Those revisions used three historical controller wires
        # (635/5,448/9,100 values).  The repository intentionally has only
        # the current rev22+ producer; truncating its vector would cut through
        # entity rows and silently reinterpret field order/scales.  Refuse
        # evaluation until an exact, versioned historical producer exists.
        raise EvalContractError(
            "historical asymmetric Worker wire constructor unavailable "
            f"for contract rev{contract_revision}; refusing to reinterpret "
            "it with the current controller layout"
        )
    worker_policy_view = None
    if contract_revision in {14, 15, 16}:
        legacy_policy_view = contract.get(
            "legacy_policy_observation_view")
        if not isinstance(legacy_policy_view, bool):
            raise EvalContractError(
                f"rev{contract_revision} Worker checkpoint "
                "lacks an explicit policy observation view")
        if custom_a12_policy:
            if legacy_policy_view:
                raise EvalContractError(
                    "A12 custom policy must get legacy-v3-a12-overlay from the environment")
        elif not legacy_policy_view:
            raise EvalContractError(
                f"rev{contract_revision} ordinary Worker must use the full protocol-v3 "
                "policy observation view")
        drink_sovereignty = contract.get("drink_sovereignty")
        if not isinstance(drink_sovereignty, bool):
            raise EvalContractError(
                f"rev{contract_revision} Worker checkpoint "
                "lacks an explicit drink_sovereignty")
        if custom_a12_policy and drink_sovereignty is not True:
            raise EvalContractError(
                f"rev{contract_revision} A12 custom Worker "
                "must enable drink_sovereignty")
        force_action12_mask = (
            not custom_a12_policy
            and drink_sovereignty is False)
        worker_policy_view = (
            "legacy-v3-a12-overlay"
            if custom_a12_policy else "legacy-v3")
    # A4 (2026-07-27): rev26 = rev25 + the kl_early_stopped flag (audit schema /10);
    # loading semantics are the same family as rev25.
    elif contract_revision in {17, 18, 19, 20, 21, 22, 23, 24, 25, 26}:
        legacy_policy_view = contract.get(
            "legacy_policy_observation_view")
        worker_policy_view = contract.get(
            "worker_policy_observation_view")
        expected_view = (
            "legacy-v3-a12-overlay"
            if custom_a12_policy
            else (
                "dual-v4-asymmetric-v3"
                if contract_revision >= 19
                else "dual-v4-asymmetric-v2"
            )
            if asymmetric_policy
            else "legacy-v3"
        )
        expected_legacy_flag = (
            not custom_a12_policy and not asymmetric_policy)
        if (
            not isinstance(legacy_policy_view, bool)
            or legacy_policy_view is not expected_legacy_flag
            or worker_policy_view != expected_view
        ):
            raise EvalContractError(
                f"rev{contract_revision} Worker "
                "policy class/view/legacy flag is not closed")
        actor_migration = contract.get("actor_migration")
        if asymmetric_policy:
            expected_asymmetric_shape = (
                _asymmetric_worker_observation_dim_for_revision(
                    contract_revision,
                    ASYMMETRIC_WORKER_OBSERVATION_DIM,
                )
            )
            rev22_runtime_evidence = None
            if contract_revision >= 22:
                try:
                    rev22_runtime_evidence = (
                        asymmetric_worker_runtime_evidence(model.policy))
                except (
                        AttributeError, RuntimeError,
                        TypeError, ValueError) as exc:
                    raise EvalContractError(
                        "rev22+ asymmetric Worker runtime evidence "
                        "cannot be generated") from exc
            expected_actor_method = (
                "copy-v28-root-plus-zero-structured-centered-context-v3"
                if contract_revision >= 22
                else "copy-v28-root-plus-zero-fused-context-v2"
                if contract_revision >= 20
                else "copy-v28-root-plus-zero-context-v1"
            )
            expected_context_initialization = (
                {
                    "hidden":
                        (
                            "structured-centered-local-uniform-fanin-v2-"
                            "seed-169328862"
                            if contract_revision >= 22
                            else
                            "local-generator-uniform-fanin-v1-seed-169328862"
                        ),
                    "output":
                        "exact-zero-disabled-through-critic-warmup",
                }
                if contract_revision >= 20
                else "zero-disabled-through-critic-warmup"
            )
            if not (
                contract.get("observation_shape")
                == [expected_asymmetric_shape]
                and isinstance(actor_migration, dict)
                and actor_migration.get("method")
                == expected_actor_method
                and _is_sha256(
                    actor_migration.get("source_checkpoint_sha256"))
                and _is_sha256(actor_migration.get("source_actor_sha256"))
                and _is_sha256(
                    actor_migration.get("migrated_actor_sha256"))
                and actor_migration.get("context_initialization")
                == expected_context_initialization
                and (
                    contract_revision < 20
                    or actor_migration.get("context_architecture")
                    == (
                        "layout-v1-shared-blocks-centered-context-"
                        "product-legacy-zero-output-v2"
                        if contract_revision >= 22
                        else (
                            "centered-context-tanh-delta-fusion-"
                            "detached-legacy-zero-output-v1"
                        )
                        if contract_revision >= 21
                        else (
                            "context-tanh-concat-detached-legacy-"
                            "tanh-zero-output-v1"
                        )
                    )
                )
                and actor_migration.get(
                    "actor_context_excluded_observation_features")
                == (
                    list(ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES)
                    if contract_revision >= 22 else [601]
                )
                and (
                    contract_revision < 22
                    or (
                        actor_migration.get("controller_layout_schema")
                        == DUAL_WORKER_LAYOUT.schema
                        and actor_migration.get(
                            "controller_layout_sha256")
                        == DUAL_WORKER_LAYOUT_SHA256
                        and actor_migration.get(
                            "target_actor_parameter_tensors")
                        == rev22_runtime_evidence[
                            "policy"]["actor_tensor_count"]
                        and actor_migration.get(
                            "target_actor_parameter_count")
                        == rev22_runtime_evidence[
                            "policy"]["actor_parameter_count"]
                        and actor_migration.get(
                            "context_parameter_tensors")
                        == rev22_runtime_evidence[
                            "context"]["tensor_count"]
                        and actor_migration.get(
                            "context_parameter_count")
                        == rev22_runtime_evidence[
                            "context"]["parameter_count"]
                    )
                )
            ):
                raise EvalContractError(
                    f"rev{contract_revision} asymmetric Worker "
                    f"lacks the {expected_asymmetric_shape}-dim actor migration contract")
            if contract_revision >= 21:
                distillation = contract.get("distillation")
                beta = contract.get("distill_beta")
                valid_beta = (
                    isinstance(beta, (int, float))
                    and not isinstance(beta, bool)
                    and math.isfinite(float(beta))
                    and float(beta) >= 0.0
                )
                scheduled = (
                    valid_beta and float(beta) > 0.0
                )
                if not (
                    valid_beta
                    and isinstance(distillation, dict)
                    and distillation.get("initial_beta")
                    == beta
                    and (
                        (
                            scheduled
                            and distillation.get("scope")
                            == "legacy-root-logits"
                            and distillation.get(
                                "excluded_actions")
                            == (
                                [12]
                                if contract_revision <= 22
                                else [12, 14]
                            )
                            and isinstance(
                                distillation.get(
                                    "anneal_actor_rollouts"),
                                int,
                            )
                            and distillation[
                                "anneal_actor_rollouts"] >= 2
                            and distillation.get("schedule")
                            == "linear-inclusive-zero-v1"
                        )
                        or (
                            not scheduled
                            and distillation.get("scope")
                            == "full-policy-logits"
                            and distillation.get(
                                "excluded_actions") == []
                            and distillation.get(
                                "anneal_actor_rollouts") == 0
                            and distillation.get("schedule")
                            == "constant"
                        )
                    )
                ):
                    raise EvalContractError(
                        "rev21+ asymmetric Worker distillation "
                        "scope/schedule is not closed")
            if (
                contract_revision
                in _DUAL_WORKER_PG_AUDIT_SCHEMA_BY_CONTRACT_REVISION
            ):
                try:
                    from train_ppo import (
                        _validate_registered_dual_worker_contract,
                    )
                    _validate_registered_dual_worker_contract(
                        contract,
                        expected_contract_revision=contract_revision,
                        expected_worker_onpolicy_pg_audit_schema=(
                            _DUAL_WORKER_PG_AUDIT_SCHEMA_BY_CONTRACT_REVISION[
                                contract_revision
                            ]
                        ),
                        runtime_evidence=rev22_runtime_evidence,
                    )
                except (RuntimeError, TypeError, ValueError) as exc:
                    raise EvalContractError(
                        f"rev{contract_revision} "
                        "actor/critic/layout/PG/clip/source-role "
                        "full contract is not closed") from exc
            if contract_revision >= 23:
                expected_action14_bonus = contract.get(
                    "worker_action14_logit_bonus")
                actual_action14_bonus = getattr(
                    model.policy, "action14_logit_bonus", None)
                if not (
                    _finite_number(expected_action14_bonus)
                    and isinstance(actual_action14_bonus, float)
                    and math.isclose(
                        actual_action14_bonus,
                        float(expected_action14_bonus),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                ):
                    raise EvalContractError(
                        "rev23+ action14 logit bonus checkpoint "
                        "runtime state disagrees with the training contract: "
                        f"policy={actual_action14_bonus!r},"
                        f"contract={expected_action14_bonus!r}")
            _validate_asymmetric_worker_runtime_state(
                model, contract_revision=contract_revision)
        elif actor_migration != "disabled":
            raise EvalContractError(
                f"rev{contract_revision} non-asymmetric Worker "
                "carries an actor migration")
        drink_sovereignty = contract.get("drink_sovereignty")
        if not isinstance(drink_sovereignty, bool):
            raise EvalContractError(
                f"rev{contract_revision} Worker checkpoint "
                "lacks an explicit drink_sovereignty")
        if custom_a12_policy and drink_sovereignty is not True:
            raise EvalContractError(
                f"rev{contract_revision} A12 custom Worker "
                "must enable drink_sovereignty")
        force_action12_mask = (
            not custom_a12_policy
            and drink_sovereignty is False)
    elif contract_revision is None or contract_revision in {7, 8, 9, 10, 11, 12, 13}:
        # Old checkpoints did not persist the view selector.  Preserve their
        # registered class-based compatibility rule.
        legacy_policy_view = not custom_a12_policy
        force_action12_mask = legacy_policy_view
        worker_policy_view = (
            "legacy-v3-a12-overlay"
            if custom_a12_policy else "legacy-v3")
    else:
        raise EvalContractError(
            f"unsupported Worker training contract revision: {contract_revision!r}")

    # R16 (C3): the default argmax passes deterministic=True, bit-identical to the old-law call;
    # sample goes through predict(deterministic=False) = the same masked
    # Categorical sampling as in training (the global torch RNG, seeded per episode by evaluate()).
    deterministic_decoding = worker_decoding == "argmax"

    def w(policy_obs, mask):
        policy_mask = mask
        if force_action12_mask:
            policy_mask = np.asarray(mask, dtype=bool)
            if policy_mask.shape != (15,):
                raise EvalContractError(
                    "legacy Worker evaluation action mask must be (15,)")
            policy_mask = policy_mask.copy()
            # V28/KING were trained with worker-owned action12 permanently
            # masked.  Its latent row is untrained and the rev9 candidate
            # explicitly discards it before installing the mixture; allowing
            # that row only in the baseline would make the paired control a
            # different, undefined policy.
            policy_mask[12] = False
        a, _ = model.predict(
            policy_obs, action_masks=policy_mask,
            deterministic=deterministic_decoding)
        return int(a)
    # Selection is performed by OptionsEnv while native raw entities/items are
    # still available.  The custom A12 adapter receives an exact v3 base plus
    # only its two registered overlay fields; ordinary policies receive exact
    # v3 and no lossy array-level "decoder".
    w.diablogym_worker_observation_view = (
        worker_policy_view)
    w.diablogym_worker_action12_mode = (
        "permanently-masked" if force_action12_mask else "environment-mask")
    w.diablogym_worker_decoding = worker_decoding
    if worker_decoding == "sample":
        # Per-episode seeding hook: the global python/numpy/torch RNGs and action_space are all reset to
        # that episode's seed (SB3 set_random_seed; env=None, so the environment is not touched).
        def _episode_reseed(seed: int) -> None:
            model.set_random_seed(int(seed))
        w.diablogym_worker_episode_reseed = _episode_reseed
    return {FARM: w}, pathlib.Path(spec).stem, identity


def parse_seeds(s: str):
    try:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("seed range must be LO-HI (e.g. 7000-7031)")
    if lo < 0 or hi < lo or hi > UINT32_MAX:
        raise argparse.ArgumentTypeError(
            f"seed range must satisfy 0 <= LO <= HI <= {UINT32_MAX}")
    return list(range(lo, hi + 1))


def safe_tag(tag: str) -> str:
    """An archive tag may only be a file name; path traversal or writing outside OUTDIR is forbidden."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) or tag in {".", ".."}:
        raise argparse.ArgumentTypeError("tag may contain only letters, digits, dots, underscores and hyphens")
    return tag


def terminal_kind(raw: dict, info: dict, done: bool, trunc: bool,
                  micro_steps: int) -> str:
    """Squash the Gym boundary into mutually exclusive, auditable terminal categories.

    The order deliberately lets a real death/victory/game_over take precedence over budget exhaustion on the same beat. An idle time limit can safely
    bootstrap; a budget boundary with an unresolved animation is a fail-closed termination.
    Unknown boundaries are never archived; they cannot pose as an ordinary survival truncation.
    """
    if bool(done) == bool(trunc):
        raise EvalContractError(
            f"terminal terminated/truncated must be mutually exclusive: done={done}, trunc={trunc}")
    unsettled = info.get("unsettled_budget_terminal")
    bootstrap_safe = info.get("time_limit_bootstrap_safe")
    decision_idle = info.get("decision_idle")

    def native_terminal(kind: str) -> str:
        if not done or trunc:
            raise EvalContractError(f"{kind} terminal must be terminated, not truncated")
        if unsettled is True or bootstrap_safe is True:
            raise EvalContractError(f"{kind} conflicts with the time-limit terminal flag")
        return kind

    if raw.get("dead"):
        return native_terminal("death")
    if raw.get("victory"):
        return native_terminal("victory")
    if raw.get("game_over"):
        return native_terminal("game_over")
    if unsettled is True:
        if (not done or trunc or micro_steps != PROTOCOL_MAX_STEPS
                or info.get("budget_exhausted") is not True
                or bootstrap_safe is not False
                or decision_idle is not False):
            raise EvalContractError("unsettled budget terminal ledger is inconsistent")
        return "time_limit_unsettled"
    if trunc:
        if (info.get("budget_exhausted") is not True
                or info.get("time_limit_bootstrap_safe") is not True
                or unsettled is not False
                or decision_idle is not True
                or micro_steps != PROTOCOL_MAX_STEPS):
            raise EvalContractError("idle time-limit truncation ledger is inconsistent")
        return "time_limit_idle"
    raise EvalContractError(
        "cannot classify a non-death terminal: neither victory/game_over nor a certified time-limit boundary")


def _action14_option_receipt(extra: dict) -> tuple[int, int, int]:
    """Read worker-only synchronous native gear receipts from one FARM window."""
    keys = (
        "worker_action14_requests",
        "worker_action14_native_successes",
        "worker_action14_gear_utility_delta",
    )
    if not isinstance(extra, dict) or any(key not in extra for key in keys):
        raise EvalContractError(
            "option_extra lacks the action14 native causal receipt split")
    requests, successes, utility_delta = (
        extra[key] for key in keys)
    if (
        not all(
            _is_plain_int(value)
            for value in (requests, successes, utility_delta)
        )
        or not 0 <= successes <= requests
        or utility_delta < 0
        or (successes == 0) != (utility_delta == 0)
    ):
        raise EvalContractError(
            "option_extra action14 native causal receipts do not balance")
    return requests, successes, utility_delta


def evaluate(
        workers, seeds, manager_npz=None, manager_sha256=None,
        manager_policy_observation_view=None, reward_economy="v1",
        worker_window_registration="farm-only", worker_decoding="argmax",
        r16_environment=None):
    if manager_policy_observation_view is None:
        if manager_npz is not None:
            raise EvalContractError(
                "a custom manager_npz must explicitly declare "
                "manager_policy_observation_view")
        manager_policy_observation_view = "legacy-v3"
    if manager_policy_observation_view not in {"legacy-v3", "raw-v4"}:
        raise EvalContractError(
            "manager_policy_observation_view only allows legacy-v3/raw-v4")
    manager_class, env_class, _bridge = _native_runtime()
    manager_path = pathlib.Path(manager_npz) if manager_npz else NPZ
    expected_manager_sha256 = manager_sha256 or (NPZ_SHA if not manager_npz else None)
    mgr = manager_class(
        str(manager_path), expected_sha256=expected_manager_sha256)
    # evaluate() is also called repeatedly in one process by tests/forensic scripts. Do not wrap the caller's
    # workers dict in place, or a second call would stack instrumentation and even close over an old env.
    if worker_window_registration not in ("farm-only", "farm-dive-v1"):
        raise EvalContractError(
            "worker_window_registration must be farm-only/farm-dive-v1, "
            f"got {worker_window_registration!r}")
    active_workers = dict(workers) if workers else None
    if worker_window_registration == "farm-dive-v1":
        # R13 exam new law: the worker callback is also hooked onto DIVE windows (control handed over on the exam side). The script
        # worker has no callback to hook, so farm-dive-v1 is meaningless for it; fail closed.
        if not active_workers:
            raise EvalContractError(
                "farm-dive-v1 registration needs a worker callback (not applicable to the script worker)")
        active_workers[DIVE] = active_workers[FARM]
    drink_sovereignty = True
    worker_observation_view = "raw-v4"
    if active_workers:
        if FARM not in active_workers:
            raise EvalContractError("Worker mapping lacks a FARM callback")
        worker_observation_view = getattr(
            active_workers[FARM],
            "diablogym_worker_observation_view",
            None,
        )
        if worker_observation_view not in _WORKER_OBSERVATION_VIEWS:
            raise EvalContractError(
                "Worker callback lacks a valid observation deployment contract: "
                f"{worker_observation_view!r}"
            )
        action12_mode = getattr(
            active_workers[FARM],
            "diablogym_worker_action12_mode",
            None,
        )
        if action12_mode == "permanently-masked":
            drink_sovereignty = False
        elif action12_mode == "environment-mask":
            drink_sovereignty = True
        else:
            raise EvalContractError(
                "Worker callback lacks a valid action12 deployment contract: "
                f"{action12_mode!r}")
    # R16 (C3): the decoding mode must match the one used at load time (fail closed); sample needs a per-episode seeding
    # hook. Under the default argmax this block only reads attributes and touches no runtime state.
    if worker_decoding not in ("argmax", "sample"):
        raise EvalContractError(
            f"worker_decoding only allows argmax/sample: {worker_decoding!r}")
    declared_decoding = (
        getattr(active_workers[FARM], "diablogym_worker_decoding", "argmax")
        if active_workers else "argmax")
    if declared_decoding != worker_decoding:
        raise EvalContractError(
            "Worker callback decoding mode disagrees with the evaluation request: "
            f"callback={declared_decoding!r},evaluate={worker_decoding!r}")
    episode_reseed = None
    if worker_decoding == "sample":
        episode_reseed = getattr(
            active_workers[FARM], "diablogym_worker_episode_reseed", None)
        if not callable(episode_reseed):
            raise EvalContractError(
                "sample-decoding Worker callback lacks a per-episode seeding hook")
        import torch
        # Single-threaded forward pass: removes the perturbation of multithreaded reduction order on the last bits of the logits, so the sampling stream
        # is bit-reproducible across processes under the same seed.
        torch.set_num_threads(1)
    # The frozen M29 manager was trained on the complete protocol-v3 base and
    # legacy clock extras.  OptionsEnv defaults to raw-v4 for newly trained
    # managers, so deployment must bind the old view explicitly.
    env_kwargs = dict(
        max_steps=PROTOCOL_MAX_STEPS,
        workers=active_workers,
        drink_sovereignty=drink_sovereignty,
        worker_observation_view=worker_observation_view,
        manager_observation_view=manager_policy_observation_view,
        reward_economy=reward_economy,
    )
    if worker_window_registration == "farm-dive-v1":
        # With the flag off not even the keyword appears, so the constructor call is bit-for-bit isomorphic to the old law
        env_kwargs["dive_live_sovereignty"] = True
    if r16_environment:
        # R16 new-law anchor: pass only non-default switches through to the constructor (farm_scene_cap /
        # reset_layer_clock_on_window are consumed by name by OptionsEnv, the rest go via
        # **env_kwargs to DiabloGymEnv); with all defaults env_kwargs has not a single extra key.
        # validate_r16_environment also rules out explicit default values and unknown keys (fail closed).
        env_kwargs.update(validate_r16_environment(dict(r16_environment)))
    env = env_class(**env_kwargs)
    engage = None
    if active_workers:
        # Engagement evidence (follow-up to the 2026-07-10 forensic review): call count/action histogram/divergence rate from the script,
        # so the evaluation file carries its own evidence that "the worker really is driving", and measures how far PPO drifted from the teacher.
        # action14 additionally gets the full opportunity -> request -> native utility gain chain; the old official archives had an action
        # histogram but could not tell "a gear-swap target never appeared" from "there was a target but the policy always ignored it",
        # nor prove that a request really equipped the item.
        from diablogym.options_env import dispatch
        inner = active_workers[FARM]
        engage = {
            "calls": 0,
            "hist": Counter(),
            "diverge": 0,
            "action14_mask_opportunities": 0,
            "action14_requests": 0,
            "action14_native_successes": 0,
            "action14_gear_utility_delta": 0,
            "_action14_receipt_requests": 0,
        }

        def instrumented(obs, mask, _inner=inner):
            live_mask = np.asarray(mask, dtype=bool)
            if live_mask.shape != (15,):
                raise EvalContractError(
                    "Worker engagement action mask must be (15,)")
            a = int(_inner(obs, mask))
            engage["calls"] += 1
            engage["hist"][int(a)] += 1
            if bool(live_mask[14]):
                engage["action14_mask_opportunities"] += 1
            if a == 14:
                if not bool(live_mask[14]):
                    raise EvalContractError(
                        "environment mask was False when the Worker requested action14")
                engage["action14_requests"] += 1
            script_mask, nearest = env.env.controller_action_context()
            script_mask = np.asarray(script_mask, dtype=bool)
            # R13: the divergence reference takes the canonical script for the current window mode (under farm-only registration
            # instrumented is only called in FARM windows, mode is always farm, and the old definition stays
            # bit-identical).
            _win = getattr(env, "_win", None) or {}
            s = dispatch(
                ("farm", "dive", "resupply")[int(_win.get("opt", FARM))],
                env.env._raw,
                bool(script_mask[14]),
                action_mask=script_mask,
                nearest_engageable_distance=nearest,
            )
            if a != s:
                engage["diverge"] += 1
            return a

        # OptionsEnv selects the Worker representation by reading attributes on
        # the callable at the lossless raw-state boundary.  Instrumentation is
        # semantically transparent only if it preserves both deployment tags;
        # otherwise the wrapper silently falls back to raw-v4.
        for attribute in (
                "diablogym_worker_observation_view",
                "diablogym_worker_action12_mode"):
            if hasattr(inner, attribute):
                setattr(instrumented, attribute, getattr(inner, attribute))
            # R13: every registration key pointing to the same inner (FARM, or DIVE under farm-dive-v1)
            # is replaced by the same instrumented: the engage counters are shared, and the action14 native
            # receipt reconciliation (receipt_requests==requests) closes naturally under dual registration.
            # Under farm-only registration this loop replaces exactly the FARM key; the old definition stays bit-identical.
        for _key in [k for k, v in active_workers.items() if v is inner]:
            active_workers[_key] = instrumented   # env holds the same dict reference, so replacing in the copy takes effect
    rows = []
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            if episode_reseed is not None:
                # At episode start seed the worker sampling RNG deterministically with the episode seed (the FARM/DIVE dual
                # registration points to the same callback and shares one stream).
                episode_reseed(seed)
            done = trunc = False
            R = 0.0
            farm = Counter()
            farm_dry = Counter()
            farm_fresh = Counter()
            nonfarm = Counter()
            allw = Counter()
            seq = ""
            final_info = {}
            while not (done or trunc):
                opt = mgr.choose(obs, env.action_masks())
                obs, r, done, trunc, info = env.step(opt)
                final_info = info
                R += float(r)
                oe = info["option_extra"]
                if engage is not None:
                    requests, successes, utility_delta = (
                        _action14_option_receipt(oe))
                    engage["_action14_receipt_requests"] += requests
                    engage["action14_native_successes"] += successes
                    engage[
                        "action14_gear_utility_delta"
                    ] += utility_delta
                allw["n"] += 1
                allw["beats"] += oe["beats"]
                allw["overrides"] += oe["overrides"]
                allw["cap"] += oe["reason"] == "cap"
                if oe["opt"] == FARM:
                    dry = oe["dry"]
                    if not isinstance(dry, bool):
                        raise EvalContractError(
                            "option_extra.dry must be an exact bool")
                    stratum = farm_dry if dry else farm_fresh
                    farm["n"] += 1
                    stratum["n"] += 1
                    farm["tau"] += oe["tau"]
                    farm["descend"] += oe["reason"] == "descend"
                    farm["r"] += float(oe["R"])
                    farm["w"] += float(oe["W"])
                    farm["bonus"] += float(oe["bonus"])
                    stratum["worker_wage"] += float(oe["worker_wage"])
                    farm["kills"] += int(oe["kills_delta"])
                    stratum["worker_kills"] += int(oe["worker_kills"])
                    voluntary = int(oe["voluntary_drinks"])
                    farm["voluntary_drinks"] += voluntary
                    farm["reflex_drain_attempts"] += int(
                        oe["drain_attempts"])
                    farm["reflex_drains"] += int(oe["drains"])
                    farm["multi_drink_windows"] += voluntary > 1
                    farm["max_voluntary_drinks"] = max(
                        farm["max_voluntary_drinks"], voluntary)
                else:
                    nonfarm["r"] += float(oe["R"])
                    nonfarm["kills"] += int(oe["kills_delta"])
                seq = oe["mode_seq"]
            raw = env.env._raw
            micro_steps = int(env.env._steps)
            ending = terminal_kind(
                raw, final_info, bool(done), bool(trunc), micro_steps)
            rows.append({
                # The archive keeps full Python floats; only terminal display/aggregate display rounds.
                "seed": seed, "ret": float(R),
                "depth": raw["dungeon_level"],
                "died": bool(raw.get("dead")), "kills": env.env._ep_kills,
                "micro_steps": micro_steps, "terminal_kind": ending,
                "farm_r": float(farm["r"]),
                "farm_w": float(farm["w"]),
                "farm_bonus": float(farm["bonus"]),
                # The totals are produced directly from the dry/fresh two-layer split, so in the official archive the
                # conservation relations hold exactly on JSON floats instead of relying on an approximate tolerance.
                "farm_worker_wage": float(
                    farm_dry["worker_wage"] + farm_fresh["worker_wage"]),
                "farm_kills": int(farm["kills"]),
                "farm_worker_kills": int(
                    farm_dry["worker_kills"] + farm_fresh["worker_kills"]),
                "nonfarm_r": float(nonfarm["r"]),
                "nonfarm_kills": int(nonfarm["kills"]),
                "farm_dry_n": int(farm_dry["n"]),
                "farm_fresh_n": int(farm_fresh["n"]),
                "farm_dry_worker_wage": float(farm_dry["worker_wage"]),
                "farm_fresh_worker_wage": float(farm_fresh["worker_wage"]),
                "farm_dry_worker_kills": int(farm_dry["worker_kills"]),
                "farm_fresh_worker_kills": int(farm_fresh["worker_kills"]),
                "farm_voluntary_drinks": int(farm["voluntary_drinks"]),
                "farm_reflex_drain_attempts":
                    int(farm["reflex_drain_attempts"]),
                "farm_reflex_drains": int(farm["reflex_drains"]),
                "farm_multi_drink_windows":
                    int(farm["multi_drink_windows"]),
                "farm_max_voluntary_drinks_per_window":
                    int(farm["max_voluntary_drinks"]),
                "ending_belt_heals":
                    int(raw.get("belt_heals", 0)),
                "farm_n": farm["n"],
                "farm_tau_mean": (
                    farm["tau"] / max(1, farm["n"])),
                # Numerators and means keep full precision; formatting happens only on the console/board.
                "farm_tau_sum": farm["tau"],
                "farm_descend": farm["descend"], "windows": allw["n"],
                "beats": allw["beats"], "overrides": allw["overrides"],
                "cap": allw["cap"], "mode_seq": seq,
            })
            print(f"  seed {seed}: ret {R:.1f} depth {raw['dungeon_level']} "
                  f"died {bool(raw.get('dead'))} "
                  f"farmτ̄ {rows[-1]['farm_tau_mean']}", flush=True)
    finally:
        try:
            env.close()
        finally:
            # OptionsEnv holds active_workers, while the instrumentation closure captures
            # env; clear the evaluation-only copy explicitly to break the env->dict->fn->env cycle. The caller's
            # original workers were never modified.
            if active_workers is not None:
                active_workers.clear()
    if engage is not None:
        if (
            engage["_action14_receipt_requests"]
            != engage["action14_requests"]
        ):
            raise EvalContractError(
                "action14 callback request count and Options native receipt request count do not close: "
                f"{engage['action14_requests']} != "
                f"{engage['_action14_receipt_requests']}")
        engage.pop("_action14_receipt_requests")
    return rows, engage


def digest(rows):
    return recompute_agg(rows)


def compare_probe_rows(rows, reference_rows):
    """G0'' reconciliation; both sides' seeds must be exactly the same set, so a subset cannot fake a PASS."""
    current = {row["seed"]: row for row in rows}
    reference = {row["seed"]: row for row in reference_rows}
    if len(current) != len(rows):
        raise ValueError("G0'' current evaluation contains a duplicate seed")
    if len(reference) != len(reference_rows):
        raise ValueError("G0'' reference archive contains a duplicate seed")
    if set(current) != set(reference):
        missing = sorted(set(current) - set(reference))
        extra = sorted(set(reference) - set(current))
        raise ValueError(f"G0'' seed sets are not exactly equal: reference lacks {missing}, reference has extra {extra}")
    bad = []
    for seed in sorted(current):
        row, expected = current[seed], reference[seed]
        if (abs(row["ret"] - expected["ep_R"]) > 0.6
                or row["depth"] != expected["depth"]
                or row["died"] != expected["died"]):
            bad.append((seed, row["ret"], expected["ep_R"],
                        row["depth"], expected["depth"]))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True, help="script | bc | *.npz | ckpt path")
    ap.add_argument("--manager-npz", default=None,
                    help="v25: manager npz (default v22-h; the regression definition for old archives is unchanged)")
    ap.add_argument(
        "--manager-policy-observation-view",
        choices=["legacy-v3", "raw-v4"],
        default=None,
        help="Manager policy input semantics; the frozen default v22-h/M29 is legacy-v3. "
             "Any custom --manager-npz must declare it explicitly; guessing from 303 dims is forbidden",
    )
    ap.add_argument("--seeds", type=parse_seeds, default=parse_seeds("7000-7031"))
    ap.add_argument("--board", action="store_true",
                    help=f"write {LB.name} after the gold-standard evaluation (the old leaderboard-hier.md is read-only)")
    ap.add_argument("--check-probe", default=None,
                    help="G0'': per-seed regression against the pre-v23 probe archive (ret +/-0.6/depth/died all equal)")
    ap.add_argument(
        "--require-published-worker", action="store_true",
        help="An SB3 candidate must re-verify the PUBLISHED behavior gate, the rev11 checkpoint contract, "
             "PASS liveness and the pre-registered lineage")
    ap.add_argument(
        "--publication-expectations", default=None,
        help="publication expectations JSON frozen before launch; must be given together with "
             "--require-published-worker")
    ap.add_argument("--tag", type=safe_tag, default=None)
    ap.add_argument("--reward-economy", default="v1",
                    choices=("v1", "v2", "v3", "v3b", "v4"),
                    help="R10 reward-economy act (v1 = historical, bit-identical; v2 = depth economy; "
                         "v4 = same source as R16 v2 + idle_counts_micro_beats + "
                         "idle_reset_on_kill). Changes only the return bookkeeping, not the action/mask/"
                         "observation protocol")
    # R16 new-law anchor: environment-side/classroom-side switches. All defaults = old law, bit-identical (not even the env_kwargs
    # keywords appear); any non-default value enters meta.protocol.r16_environment.
    ap.add_argument("--explore-global-hunt", action="store_true",
                    help="DiabloGymEnv explore_global_hunt (the main R16 lever: when no monster is visible"
                         " in the window, BFS across the whole map toward the nearest living monster)")
    ap.add_argument("--explore-global-fallback", action="store_true",
                    help="DiabloGymEnv explore_global_fallback(C5)")
    ap.add_argument("--progress-far-tiles", type=int,
                    default=R16_ENVIRONMENT_DEFAULTS["progress_far_tiles"],
                    help="DiabloGymEnv progress_far_tiles (C6; default 0 = old law)")
    ap.add_argument("--farm-scene-cap", type=int,
                    default=R16_ENVIRONMENT_DEFAULTS["farm_scene_cap"],
                    help="OptionsEnv farm_scene_cap (default 1800 = "
                         "the options_env.FARM_SCENE_CAP old law)")
    ap.add_argument("--reset-layer-clock-on-window", action="store_true",
                    help="OptionsEnv reset_layer_clock_on_window (opening a FARM window"
                         " resets the no-progress clock)")
    ap.add_argument("--resource-protocol", default="off",
                    choices=("off", "l2-town-v1"))
    ap.add_argument("--resource-purchase-mode", default="full",
                    choices=("none", "heal", "potions", "armor", "full"))
    ap.add_argument("--resource-service-policy", default="legacy-v1",
                    choices=("legacy-v1", "sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6"),
                    help="Explicit service identity; sustain-v2/v3/v4 require l2-town-v1/full")
    ap.add_argument("--resource-readiness-law", default="veto-v1",
                    choices=("veto-v1", "coach-v03"),
                    help="R17.1 ruling 3 readiness law; coach-v03 (six native "
                         "conditions, health excluded) requires l2-town-v1")
    ap.add_argument("--resource-retreat", default="off",
                    choices=("off", "retreat-v1"),
                    help="R18-B return-to-town law; retreat-v1 requires "
                         "l2-town-v1 under coach-v03")
    ap.add_argument("--resource-portal", default="off",
                    choices=("off", "portal-v1"),
                    help="R18-B5 Scroll of Town Portal vehicle for the retreat; "
                         "portal-v1 requires l2-town-v1 under coach-v03 and "
                         "--resource-retreat retreat-v1")
    ap.add_argument("--resource-sweep", default="off",
                    choices=("off", "sweep-v1"),
                    help="R18-B6 chest/barrel sweep on main L1; sweep-v1 requires "
                         "l2-town-v1 with --resource-service-policy sustain-loot-v1")
    ap.add_argument("--resource-identify", default="off",
                    choices=("off", "cain-v1"),
                    help="R18-B6 Cain identify leg of the loot town trip; cain-v1 "
                         "requires l2-town-v1 with sustain-loot-v1")
    ap.add_argument("--resource-weapon-upgrade", default="off",
                    choices=("off", "smith-v1"),
                    help="R18-B6 Griswold weapon upgrade leg; smith-v1 requires "
                         "l2-town-v1/full with sustain-loot-v1")
    ap.add_argument("--hunt-scope", default="all",
                    choices=("all", "l1-only"),
                    help="R18-B5 scope of the a10 global hunt; l1-only masks it "
                         "on main L2+")
    ap.add_argument("--dive-blocker-recovery", default="off", choices=("off", "adjacent-v1"),
                    help="Explicit bounded a11 blocker combat; requires resource protocol")
    ap.add_argument("--worker-window-registration", default="farm-only",
                    choices=("farm-only", "farm-dive-v1"),
                    help="R13 exam new law (default farm-only reproduces old exams bit for bit): "
                         "farm-dive-v1 also hooks the worker callback onto DIVE windows and activates the in-window "
                         "control handover: the worker drives DIVE itself on the exam for the first time")
    ap.add_argument("--worker-decoding", default="argmax",
                    choices=("argmax", "sample"),
                    help="R16 (C3) worker decoding (default argmax reproduces old exams bit for bit): "
                         "sample = SB3 worker predict(deterministic=False), "
                         "each episode seeds the torch/numpy RNGs with its seed, single-threaded torch; "
                         "written to meta.protocol.worker_decoding")
    args = ap.parse_args()
    if (args.worker_window_registration == "farm-dive-v1"
            and args.worker == "script"):
        ap.error("--worker-window-registration farm-dive-v1 "
                 "does not apply to the script worker (no callback to hook)")
    if args.worker_decoding == "sample":
        if (args.worker in {"script", "bc"}
                or pathlib.Path(args.worker).suffix.lower() == ".npz"):
            ap.error("--worker-decoding sample only applies to SB3 checkpoint workers")
        if args.board:
            ap.error("--board gold-standard protocol is argmax; refusing to put sample decoding on the board")
    if args.progress_far_tiles < 0:
        ap.error("--progress-far-tiles must be a non-negative integer (0 = old law)")
    if args.farm_scene_cap <= 0:
        ap.error("--farm-scene-cap must be a positive integer (1800 = old law)")
    try:
        validate_resource_service_config(args.resource_protocol,
                                         args.resource_purchase_mode,
                                         args.resource_service_policy)
        validate_dive_blocker_recovery(args.resource_protocol, args.dive_blocker_recovery)
    except ValueError as exc:
        ap.error(str(exc))
    if (args.resource_readiness_law != "veto-v1"
            and args.resource_protocol != "l2-town-v1"):
        ap.error("--resource-readiness-law coach-v03 requires "
                 "--resource-protocol l2-town-v1")
    if (args.resource_retreat != "off"
            and (args.resource_protocol != "l2-town-v1"
                 or args.resource_readiness_law != "coach-v03")):
        ap.error("--resource-retreat retreat-v1 requires "
                 "--resource-protocol l2-town-v1 and "
                 "--resource-readiness-law coach-v03")
    # R18-B5 (2026-09-07): the portal is the retreat's carrier; besides the classroom it also needs that retreat law.
    if (args.resource_portal != "off"
            and (args.resource_protocol != "l2-town-v1"
                 or args.resource_readiness_law != "coach-v03")):
        ap.error("--resource-portal portal-v1 requires "
                 "--resource-protocol l2-town-v1 and "
                 "--resource-readiness-law coach-v03")
    if args.resource_portal != "off" and args.resource_retreat != "retreat-v1":
        ap.error("--resource-portal portal-v1 requires "
                 "--resource-retreat retreat-v1")
    # R18-B5 review fix (2026-09-07): l1-only only acts on the gate of the a10 whole-map monster search,
    # so with the search off it is a no-op; never mint an archive identity that testifies for a law that did not run.
    if args.hunt_scope != "all" and not args.explore_global_hunt:
        ap.error("--hunt-scope l1-only requires --explore-global-hunt")
    # R18-B6 (2026-09-07): the chest sweep/identify/weapon-upgrade laws all live in the loot economy's town
    # trips; these three ap.error calls copy the wording of the three validators of the OptionsEnv constructor.
    # Review fix (same day): "verbatim" holds only for the first two. The vocabularies of validate_sweep_protocol /
    # validate_identify_protocol are ("off","sweep-v1") /
    # ("off","cain-v1"), verbatim the same as the choices= above; but
    # resource_weapon_upgrade.RESOURCE_WEAPON_UPGRADES is
    # ("off","dry-v1","smith-v1"), and both this file's choices= and eval_contract's
    # validate_r16_environment are **deliberately** one value narrower. The reason is the same as on the training side:
    # dry-v1 sends no command, takes no micro-step and makes no native call, so its row data are bit-identical to the control arm;
    # minting an archive identity for it would falsely testify for an environment bit-identical to the default world,
    # exactly the hole the hunt_scope l1-only review fix closed. So this narrowing was carried
    # into the deployment side on purpose, not by oversight; examining a dry-v1 counterfactual arm first needs its own
    # identity decision (B6-REPORT.md open question 2).
    # Note the choices= of args.resource_weapon_upgrade already stops dry-v1 at the
    # parsing layer above, so the three checks below only ever see off or a law name.
    for flag, law, value in (("--resource-sweep", "sweep-v1", args.resource_sweep),
                             ("--resource-identify", "cain-v1", args.resource_identify),
                             ("--resource-weapon-upgrade", "smith-v1",
                              args.resource_weapon_upgrade)):
        if value == "off":
            continue
        if args.resource_protocol != "l2-town-v1":
            ap.error(f"{flag} {law} requires --resource-protocol l2-town-v1")
        if args.resource_service_policy != "sustain-loot-v1":
            ap.error(f"{flag} {law} requires --resource-service-policy sustain-loot-v1")
    if (args.resource_weapon_upgrade != "off"
            and args.resource_purchase_mode != "full"):
        ap.error("--resource-weapon-upgrade smith-v1 requires "
                 "--resource-purchase-mode full")
    requested_r16 = {
        "explore_global_hunt": bool(args.explore_global_hunt),
        "hunt_scope": args.hunt_scope,
        "explore_global_fallback": bool(args.explore_global_fallback),
        "progress_far_tiles": int(args.progress_far_tiles),
        "farm_scene_cap": int(args.farm_scene_cap),
        "reset_layer_clock_on_window": bool(args.reset_layer_clock_on_window),
        "resource_protocol": args.resource_protocol,
        "resource_purchase_mode": args.resource_purchase_mode,
        "resource_service_policy": args.resource_service_policy,
        "resource_readiness_law": args.resource_readiness_law,
        "resource_retreat": args.resource_retreat,
        "resource_portal": args.resource_portal,
        # R18-B6 (2026-09-07): the three loot-trip laws enter the archive identity; the default off is
        # filtered out by "record only non-defaults", so old exams stay byte-identical.
        "resource_sweep": args.resource_sweep,
        "resource_identify": args.resource_identify,
        "resource_weapon_upgrade": args.resource_weapon_upgrade,
        "dive_blocker_recovery": args.dive_blocker_recovery,
    }
    # The identity records only non-default keys; all defaults => empty dict => env_kwargs/meta byte-identical.
    r16_environment = {
        key: value for key, value in requested_r16.items()
        if value != R16_ENVIRONMENT_DEFAULTS[key]}
    if r16_environment and args.board:
        ap.error("--board gold-standard protocol is the old-law environment; refusing to put R16 environment switches on the board")
    if bool(args.require_published_worker) != bool(
            args.publication_expectations):
        ap.error("--require-published-worker and --publication-expectations "
                 "must be given together")
    if (args.require_published_worker
            and (args.worker in {"script", "bc"}
                 or pathlib.Path(args.worker).suffix.lower() == ".npz")):
        ap.error("--require-published-worker only applies to SB3 checkpoints")
    if args.manager_npz and args.manager_policy_observation_view is None:
        ap.error("a custom --manager-npz must be given together with"
                 " --manager-policy-observation-view")
    manager_policy_observation_view = (
        args.manager_policy_observation_view or "legacy-v3")

    # The CLI must be a fresh process that has not mapped the native extension yet; otherwise the on-disk SHA cannot prove the byte
    # identity of the mapped pages. All official drivers start this entry point as a subprocess.
    if "_diablogym" in sys.modules:
        raise EvalContractError(
            "eval_assembled must run in a fresh process that has not preloaded the diablogym bridge")

    seeds = args.seeds
    seed_label = f"{seeds[0]}-{seeds[-1]}"
    label = worker_label(args.worker)
    auto_tag = f"{label}-{seed_label}"
    if args.worker_decoding != "argmax":
        auto_tag += f"-{args.worker_decoding}"
    tag = args.tag or safe_tag(auto_tag)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / f"{tag}.json"
    if args.board and seeds != list(range(9000, 9032)):
        ap.error("--board only allows the gold-standard seeds 9000-9031; refusing to pollute the leaderboard with the probe pool")
    board_contract = assembled_board_contract() if args.board else None
    if board_contract is not None:
        ensure_leaderboard_compatible(
            LB, board_contract, initial_text=ASSEMBLED_LEADERBOARD_HEADER)
    board_guard = (exclusive_lock(OUTDIR / ".gold-evaluation.lock", "gold-standard evaluation launch")
                   if args.board else contextlib.nullcontext())
    try:
        # Held from before launch until the atomic commit and the read-back re-verification finish; concurrent processes with the same tag
        # cannot both pass the exists check and then os.replace each other.
        with reserve_output(out_path), board_guard:
            bridge_path = bridge_binary_path(ROOT)
            runtime = runtime_identity(ROOT, bridge_path)
            if (board_contract is not None
                    and board_contract["runtime"] != runtime):
                raise EvalContractError("assembled-agent board contract disagrees with the frozen evaluation runtime")
            _native_runtime(runtime)
            manager_path = pathlib.Path(args.manager_npz).resolve() if args.manager_npz else NPZ
            manager_id = file_identity("numpy_policy", manager_path)
            if not args.manager_npz and manager_id["sha256"] != NPZ_SHA:
                raise EvalContractError(
                    f"default manager npz sha drift: {manager_id['sha256']} != {NPZ_SHA}")
            expected_provenance = None
            expectations_identity = None
            if args.publication_expectations:
                (expected_provenance,
                 expectations_identity) = capture_publication_expectations(
                    args.publication_expectations)
            training_implementation_sha256 = None
            if args.require_published_worker:
                from train_ppo import _implementation_bundle_sha256
                training_implementation_sha256 = (
                    _implementation_bundle_sha256())
            workers, loaded_label, worker_id = load_worker(
                args.worker, runtime["python_protocol"]["sha256"],
                require_published=args.require_published_worker,
                expected_manager_sha256=manager_id["sha256"],
                expected_implementation_sha256=(
                    training_implementation_sha256),
                expected_provenance=expected_provenance,
                worker_decoding=args.worker_decoding)
            if loaded_label != label:
                raise EvalContractError("worker label derivation disagrees with the load result")
            if args.worker_decoding != "argmax":
                print(f"  worker decoding: {args.worker_decoding}"
                      " (set_random_seed(seed) per episode, single-threaded torch)")
            if r16_environment:
                print(f"  R16 environment switches (non-default): {r16_environment}")

            # Use the same file already resolved in the identity manifest, so a custom symlink cannot change its target between
            # the hash and the actual NumpyManager.load.
            rows, engage = evaluate(
                workers, seeds, manager_npz=str(manager_path),
                manager_sha256=manager_id["sha256"],
                manager_policy_observation_view=(
                    manager_policy_observation_view),
                reward_economy=args.reward_economy,
                worker_window_registration=(
                    args.worker_window_registration),
                worker_decoding=args.worker_decoding,
                r16_environment=r16_environment or None)
            agg = digest(rows)
            if engage:
                agg["worker_calls"] = engage["calls"]
                agg["worker_action_hist"] = dict(sorted(engage["hist"].items()))
                agg["worker_divergences"] = engage["diverge"]
                agg["script_divergence_rate"] = (
                    engage["diverge"] / max(1, engage["calls"]))
                agg["worker_action14_mask_opportunities"] = (
                    engage["action14_mask_opportunities"])
                agg["worker_action14_requests"] = (
                    engage["action14_requests"])
                agg["worker_action14_native_successes"] = (
                    engage["action14_native_successes"])
                agg["worker_action14_gear_utility_delta"] = (
                    engage["action14_gear_utility_delta"])
                print(f"  engagement: worker calls {engage['calls']}, "
                      f"action histogram {agg['worker_action_hist']}, "
                      f"divergence rate from script {agg['script_divergence_rate']:.4f}")
                print(
                    "  gear-swap ledger: "
                    f"opportunities {agg['worker_action14_mask_opportunities']}, "
                    f"requests {agg['worker_action14_requests']}, "
                    f"native successes {agg['worker_action14_native_successes']}, "
                    "utility gain "
                    f"{agg['worker_action14_gear_utility_delta']}"
                )
            print(f"{tag}: ret {agg['ret_mean']:.1f} "
                  f"(med {agg['ret_median']:.2f}) "
                  f"died {agg['died']}/{agg['n']} depth_med {agg['depth_median']} | "
                  f"R4: level-change rate {agg['farm_descend_rate']:.4f} "
                  f"override {agg['override_rate']:.4f} "
                  f"cap {agg['cap_rate']:.4f} "
                  f"farmτ̄ {agg['farm_tau_mean']:.1f}")
            print(
                "  potion ledger: "
                f"active drinks/episode {agg['farm_voluntary_drinks_mean']:.3f}, "
                "reflex attempts/successes per episode "
                f"{agg['farm_reflex_drain_attempts_mean']:.3f}/"
                f"{agg['farm_reflex_drains_mean']:.3f}, "
                f"multi-drink window rate {agg['farm_multi_drink_window_rate']:.6f}, "
                "max active drinks in one window "
                f"{max(row['farm_max_voluntary_drinks_per_window'] for row in rows)}, "
                f"mean ending belt heals {agg['ending_belt_heals_mean']:.3f}"
            )

            if args.check_probe:
                probe_doc = strict_json_loads(
                    pathlib.Path(args.check_probe).read_text(encoding="utf-8"))
                try:
                    ref_rows = probe_doc["argmax_episodes"]
                except (KeyError, TypeError) as exc:
                    raise ValueError("G0'' reference archive lacks argmax_episodes") from exc
                bad = compare_probe_rows(rows, ref_rows)
                print(f"G0'' regression: {len(rows) - len(bad)}/{len(rows)} match"
                      + (f"; mismatches {bad}" if bad else " -- PASS"))
                if bad:
                    raise SystemExit(1)

            # If file inputs or runtime code are replaced during a long evaluation, this archive must be voided.
            verify_file_identity(worker_id)
            verify_file_identity(manager_id)
            verify_publication_expectations(expectations_identity)
            if runtime_identity(ROOT, bridge_path) != runtime:
                raise EvalContractError("bridge, engine or protocol source changed during evaluation")

            document = {
                "schema_version": SCHEMA_VERSION,
                "meta": make_meta(tag=tag, seeds=seeds, worker=worker_id,
                                  manager=manager_id, runtime=runtime,
                                  worker_decoding=args.worker_decoding,
                                  r16_environment=r16_environment or None),
                "agg": agg,
                "rows": rows,
            }
            expected = expected_eval_identity(
                {"worker": worker_id, "manager": manager_id, "runtime": runtime},
                tag=tag, seeds=seeds)
            if args.worker_decoding != "argmax":
                # Under the default argmax the expected dict is unchanged key by key; the read-back re-verification of a sample archive
                # must check the meta.protocol.worker_decoding declaration.
                expected["expected_worker_decoding"] = args.worker_decoding
            if r16_environment:
                # Likewise: the R16 new-law anchor's read-back re-verification must check the r16_environment declaration.
                expected["expected_r16_environment"] = dict(r16_environment)
            validate_eval_archive(document, **expected)
            payload = json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False)

            # Temp file in the same directory + fsync + os.replace: SIGKILL leaves no half-written official JSON.
            tmp = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, out_path)
            finally:
                tmp.unlink(missing_ok=True)
            try:
                loaded = read_eval_archive(out_path, **expected)
                verify_file_identity(worker_id)
                verify_file_identity(manager_id)
                verify_publication_expectations(expectations_identity)
                if runtime_identity(ROOT, bridge_path) != runtime:
                    raise EvalContractError(
                        "bridge, engine or protocol source changed while committing the archive")
            except Exception:
                out_path.rename(out_path.with_suffix(f".{time.time_ns()}.void"))
                raise
            agg = loaded["agg"]
            print(f"saved and re-verified {out_path}")

            if args.board:
                archive_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
                row_key = versioned_row_key(tag, archive_sha)
                visible = (
                    f"| {row_key} | {agg['ret_mean']:.1f} | "
                    f"{agg['ret_median']:.2f} | "
                    f"{agg['died']}/{agg['n']} | {agg['depth_median']} | "
                    f"worker={worker_id['kind']}; L3+ {agg['l3']}; "
                    f"kills {agg['kills_mean']:.1f}; "
                    f"level-change rate {agg['farm_descend_rate']:.4f}; "
                    f"override {agg['override_rate']:.4f} |")
                row = assembled_leaderboard_row(
                    visible, row_key=row_key, contract=board_contract,
                    archive_path=str(out_path), archive_sha256=archive_sha,
                    worker_sha256=worker_id["sha256"],
                    manager_sha256=manager_id["sha256"])
                upsert_leaderboard_rows(
                    LB, {row_key: row}, contract=board_contract,
                    initial_text=ASSEMBLED_LEADERBOARD_HEADER,
                    lock_path=LB_LOCK)
                print(f"wrote {LB.name}")
    except OutputReservationError as exc:
        ap.error(f"{exc} (wait for the concurrent evaluation to finish, or archive the old result per the restart protocol)")


if __name__ == "__main__":
    main()
