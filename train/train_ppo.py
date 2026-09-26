"""DiabloGym v1 training: PPO learns to clear dungeon level 1.

Usage (from the repo root):
  .venv/bin/python train/train_ppo.py --total-steps 2998272 --num-envs 4
  (metrics are written to runs/<run>/progress.jsonl + status.json and read live by dashboard.py)
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import functools
import hashlib
import io
import json
import math
import os
import pathlib
import random
import shutil
import sys
import time
import types
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from eval_contract import (PROTOCOL_VERSION, RUNTIME_PACKAGE_VERSIONS,
                           resource_service_recipe, validate_resource_service_config,
                           dive_blocker_recovery_recipe, validate_dive_blocker_recovery,
                           worker_prefix_recipe, EARNED_DIVE_SUFFIX_SCOPE)
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


_POLICY_HEAD_KEYS = (
    "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
    "action_net.weight", "action_net.bias",
)
_RUN_ARTIFACTS = (
    "progress.jsonl", "status.json", "status.tmp.json", "sentinel.jsonl", "calib.jsonl",
    # Launch audit fix A: the three new gauge files are append-only ("a" mode); if they are not listed, a rerun under the same name
    # accumulates leftovers, and the course leg's second end-of-leg full-table review falsely rules CASE_HALT_G0 (same cause as the G0-2a incident).
    "dry_curriculum.jsonl", "distill_ce_probe.jsonl", "drywin_metrics.jsonl",
    "bc_aux_monitor.jsonl", "bc_aux_behavior_receipt.json",
    "bc_aux_liveness_preflight.json",
    "model_final.zip", "model_candidate.zip", "model_development.zip",
    "policy.npz", "policy_sd.pt", "tb", "ckpt",
)
_ARTIFACT_SCOPE_RESULTS = {
    "development": ("model_development.zip", "DEVELOPMENT_ONLY"),
    "candidate": ("model_candidate.zip", "PRODUCTION_CANDIDATE"),
    "production": ("model_final.zip", "PUBLISHED"),
}
_GEAR_PRESENT_INDEX = 293  # base obs zero-based; "dimension 294" in the docs
# Single source of truth for the training contract revision number. rev7 moved distill_beta / teacher_sha256 /
# calib_record_only, previously written only into the easily lost run config, into the checkpoint contract,
# and registers whether an active bc_aux carries liveness. Otherwise a leg with beta=0 or record-only (no verdict) could
# masquerade as the same formal target.
# rev10 replaced the harmful fixed argmax circuit with an exact 5% mixture inside the policy distribution;
# rev11 fixed its fatal misalignment of "random exploration exists, deterministic deployment never reachable": four stable
# raw combat features + a bias form a 5-parameter contextual gate, still trained in the same exact mixture
# distribution, and before release it must prove that deterministic a12 is reachable across several held-out episodes.
# rev12 withdrew that last deterministic action quota: whether to drink under argmax is decided autonomously by PPO's combat
# return; release still requires natively certified exploration samples, low false drinking, low drift and resource safety,
# and actual combat strength is left to the independent paired efficacy gate.
# rev13 made the WorkerWindowEnv cross-window terminal credit and additional death cost a non-omittable part of the
# training contract. The old implementation wrote real deaths during frozen manager/script time only to the audit ledger,
# and PPO received 0; meanwhile adding a risk cost to deaths inside FARM and during fast-forward easily double-penalised.
# Both knobs must be itemised by the environment per transition; the defaults keep rev12 semantics bit for bit.
# rev14 explicitly binds whether the Worker policy sees raw protocol-v4 or legacy protocol-v3
# observations, but the so-called legacy then only decoded 286/297; v4 had already lost information in the earlier monster/map/item
# columns, so the old actor was still OOD. rev15 requires the environment to rebuild the full v3 base at the lossless raw
# boundary. The ordinary actor receives exact legacy-v3; the A12 custom actor receives
# legacy-v3-a12-overlay (only the reversible packed belt/latch is overlaid back onto 286/297),
# and its inherited actor/value also decode internally to exactly the full v3.
# rev16 closes the optimization target itself: real base-game termination and
# cross-window bootstrap are named, manager view is explicit, and main-PPO
# gradient clipping records whether actor/critic are independently bounded.
# rev18 binds the asymmetric 635-wide v2 wire contract.  v2 distinguishes an
# armed fuse at counter zero and excludes the training-only p_skip field from
# the actor context.  rev19 replaces that still-aliased wire with v3's bounded
# controller snapshot: combat candidates/ledger, potion order, exact scene
# state, and the radius-12 map consumed by a9/a10/a13/a14 are visible and
# execution uses the same frozen snapshot.  Rev20 replaces the affine context
# bias with a zero-output nonlinear context×legacy fusion.  Migrated logits
# remain bitwise V28 at ignition while conditional combat decisions become
# representable.  Rev21 binds root-only KING supervision, actor-rollout beta
# annealing, centered context fusion and context-specific reward/PPO evidence.
# Rev22 replaces both flat 9,100-wide branches with the frozen-layout shared
# block encoders, binds their layout identity/capacity, and clips legacy root,
# actor context, and critic independently.
# Rev23 removes a14 from legacy KING/V28 distillation support.  The historical
# teacher never executed a14, so supervising that logit actively opposes the
# new on-policy whole-loadout gear objective.
# Rev24 records the causal role of every policy source.  In particular, merely
# binding BC-v1 demos for PASS validation/read-only dry-anchor instrumentation
# is no longer allowed to masquerade as BC initialization or an optimizer
# objective; a14's fixed prior and native-reward PPO path are named separately.
# It also distinguishes a progressful TimeLimit truncation from the
# no-progress budget failure, which is terminal, non-bootstrap, and charged
# the current depth's death-equivalent base cost plus the configured risk cost.
# Rev25 upgrades the formal asymmetric Worker policy-gradient audit to
# diablogym-worker-onpolicy-pg/9.  That receipt seals the collection actor,
# independently closes GAE/log-probability inputs, and proves the optimizer's
# realised actor/root/context movement.  Rev24 remains an immutable historical
# evaluation contract bound to its original /8 audit; it is not silently
# reinterpreted as /9 and is not accepted for current-training continuation.
_CONTRACT_REVISION = 26
_REGISTERED_DUAL_WORKER_PG_AUDIT_SCHEMAS = types.MappingProxyType({
    24: "diablogym-worker-onpolicy-pg/8",
    25: "diablogym-worker-onpolicy-pg/9",
    # A4 (2026-07-27): KL early-stop exemption flag added to the receipt, audit schema /10, contract bumped to rev26.
    26: "diablogym-worker-onpolicy-pg/10",
})
_POLICY_SOURCE_ROLES_SCHEMA = "diablogym-policy-source-roles/1"
_WORKER_EPISODE_BOUNDARY_V24 = (
    "base-game-terminal-plus-no-progress-timeout-failure"
)
_WORKER_NO_PROGRESS_TIMEOUT_CONTRACT = {
    "boundary": "terminated-no-bootstrap",
    "reward": (
        "death-ladder-base-plus-additional-terminal-death-cost"
    ),
}
# R16 amendment wiring constants: the CLI default means "not enabled" (the contract always writes None).
# farm_scene_cap defaults to a mirror of diablogym.options_env.FARM_SCENE_CAP (=1800);
# no_progress_timeout_credit defaults to death-equivalent = the old rule (a stall timeout is booked
# as death-equivalent), zero = the new R16 booking definition (semantics implemented on the worker_env side).
_FARM_SCENE_CAP_DEFAULT = 1800
_WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT = "death-equivalent"
_WORKER_NO_PROGRESS_TIMEOUT_CREDITS = ("death-equivalent", "zero")
_WORKER_VIEW_LEGACY_V3 = "legacy-v3"
_WORKER_VIEW_A12_OVERLAY = "legacy-v3-a12-overlay"
_WORKER_VIEW_DUAL_V4_ASYMMETRIC = "dual-v4-asymmetric-v3"
# R13 classroom reform: v5 = v4 bit-for-bit prefix + a 12-dim window-mode append block (see options_env).
_WORKER_VIEW_DUAL_V5_WINDOW_MODE = "dual-v5-window-mode-v1"
# dual-family views share the asymmetric policy class and the dual continuation-classification machinery.
_WORKER_DUAL_VIEWS = frozenset({
    _WORKER_VIEW_DUAL_V4_ASYMMETRIC,
    _WORKER_VIEW_DUAL_V5_WINDOW_MODE,
})
_DUAL_LEGACY_ACTOR_MIGRATION = "legacy-actor-migration"
_DUAL_ENV_RESTART_CONTINUATION = (
    "parameter-continuation-with-environment-restart-v1"
)
_RESUME_LINEAGE_SCHEMA = "diablogym-resume-lineage/1"
_ASYMMETRIC_ACTOR_INIT_SCHEMA = (
    "diablogym-asymmetric-worker-actor-init/5"
)
_ASYMMETRIC_ACTOR_INIT_METHOD = (
    "copy-v28-root-plus-zero-structured-centered-context-v3"
)
_ASYMMETRIC_CONTEXT_ARCHITECTURE = (
    "layout-v1-shared-blocks-centered-context-product-legacy-zero-output-v2"
)
_ASYMMETRIC_CRITIC_RESET_SCHEMA = (
    "diablogym-worker-critic-reset/3"
)
_ASYMMETRIC_CRITIC_RESET_METHOD = (
    "structured-layout-v1-orthogonal-value-only-v2"
)
_ASYMMETRIC_CRITIC_ARCHITECTURE = (
    "layout-v1-independent-shared-blocks-centered-value-v2"
)
_ASYMMETRIC_CANONICAL_EVIDENCE_SCHEMA = (
    "diablogym-asymmetric-worker-canonical-evidence/1"
)
_RUNTIME_VERSIONS = dict(RUNTIME_PACKAGE_VERSIONS)
_ALGORITHM_RECIPE = {
    "gae_lambda": 0.95,
    "n_epochs": 10,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "target_kl": None,
    "use_sde": False,
    "sde_sample_freq": -1,
}
_SCHEDULE_PROBES = (1.0, 0.5, 0.0)
_BC_REPORT_SCHEMA_VERSION = 1
_EXPORT_MANIFEST_SCHEMA_VERSION = 1
# Every predecessor registry remains burned once collection opens it, even if
# no PASS artifact survives.  The immediately preceding 2100000/2101000 pools
# were consumed and therefore cannot service a new R7 prepare-bc.  Register
# fresh, disjoint active pools without deleting any old range from the
# ordinary-training exclusion table.
_WORKER_BC_DEMO_SEEDS = tuple(range(2_142_000, 2_142_128))
_BURNED_BC_EPISODES = frozenset(
    (*range(100, 484), *range(1000, 1384),
     *range(2000, 2128), *range(3000, 3384),
     *range(2_100_000, 2_100_128),
     *range(2_101_000, 2_101_384)))
_WORKER_BC_FORBIDDEN_ACTIONS = (11, 12)
# a14 is deliberately sparse, so the generic >=300-sample class rule used to
# omit it while aggregate top-1 still passed.  It is now a mandatory recall
# class with explicit demonstration breadth.
_WORKER_BC_REQUIRED_RECALL_ACTIONS = (14,)
_WORKER_BC_MIN_ACTION14_LABELS = 64
_WORKER_BC_MIN_ACTION14_EPISODES = 16
# E3 4B: forbidden-action assertion conditioned on generation (shared source of truth with design E2) -- v1 forbids (11,12) unchanged (line above),
# v2 forbids 11, allows 12 (guard surface not weakened).
_WORKER_BC_V2_FORBIDDEN_ACTIONS = (11,)
# E3 4B: frozen main-case λ_bc constant (registered discretion in D7, same order of magnitude as the distillation anchor β=0.015625);
# it is the documentation/test anchor for the value passed explicitly on the L-full CLI, not the --bc-aux-lambda default (default 0.0 = absent).
_BC_AUX_MAIN_LAMBDA = 0.015625
# v33 fixed auxiliary objective. The old objective kept only the 199 positives with y==12 and replayed them in every PPO
# minibatch, which measurably pushed the policy to "drink almost always whenever m[12] is legal". rev2 pins down
# the data surface, loss surface and consumption frequency together and writes them into training_contract, forbidding old rev5 checkpoints from
# silently carrying positive-only semantics into a new leg.
# rev3 further restricts the auxiliary bank to BC-v2 training episodes; the original held-out
# episodes serve only the final release gate, forbidding positives/negatives from leaking into the optimizer and then posing as independent acceptance.
# rev4 moved the anchor of the negative KL/rollout monitor from "each leg's starting point" to the first aux root policy,
# persisted with the checkpoint; this forbids continuations from re-anchoring per leg and accumulating drift.
# rev5 split the once-per-rollout auxiliary update out of the joint clipping of the first PPO minibatch,
# running it as a standalone optimizer step after the PPO epochs; production and the liveness preflight share the same
# zero_grad/backward/clip/step atomic path, so a pure-aux preflight cannot systematically overestimate learnability.
# rev6 proved the structural path can be installed losslessly, but also produced a decisive online counter-proof: freezing the teacher's
# hp-band target into argmax does not add combat strength, it only moves the successful brainstem reflex earlier;
# the 16 episodes' deaths were completely unchanged and returns fell. rev7 therefore demoted the teacher to a training-time exploration prior:
# the actor is widened losslessly to 68, and the new column carries only one dedicated probability scalar. rev8 fixed rev7's last
# tail hole: a single raw-logit weight cannot achieve both "5% on average" and "never the argmax"
# across heterogeneous states. The policy distribution layer now defines exactly
# π'(12)=ε, π'(a≠12)=(1−ε)π_non12(a), shared by rollout and evaluate_actions;
# with ε=.05, top non12 >= .95/14 > .05, so it is never the argmax in any state by construction. But rev8
# capped ε at 0.25, while the real deterministic switching threshold of V28 eligible states is 0.327..0.499;
# training could sample a12, yet the formal deterministic eval would never drink once. rev9 learns ε(s) from four stable
# raw features (HP, monster density, nearest-monster distance, belt economy) + bias, probability cap
# 0.95, still exactly 0 outside the hard predicate; the five parameters do not backpropagate into the old actor. rev10 no longer mixes
# a teacher-style deterministic a12 quota into release: PPO may learn or refuse the action, and the final
# efficacy is decided only by the paired combat-strength gate.
_BC_AUX_OBJECTIVE_REVISION = 11
_BC_AUX_CIRCUIT_SCHEMA = "a12-onpolicy-contextual-mixture-adapter/1"
_BC_AUX_CIRCUIT_BASE_WIDTH = 64
_BC_AUX_CIRCUIT_EXPANDED_WIDTH = 68
_BC_AUX_CIRCUIT_ACTION = 12
_BC_AUX_CIRCUIT_GATE_FEATURE_INDICES = (0, 8, 9, 286)
_BC_AUX_CIRCUIT_GATE_PARAMETER_COLUMNS = (64, 65, 66, 67)
_BC_AUX_CIRCUIT_INITIAL_PROBABILITY = 0.05
_BC_AUX_CIRCUIT_PROBABILITY_MIN = 0.001
_BC_AUX_CIRCUIT_PROBABILITY_MAX = 0.95
_BC_AUX_CIRCUIT_INITIAL_GATE_BIAS = math.log(
    (_BC_AUX_CIRCUIT_INITIAL_PROBABILITY
     - _BC_AUX_CIRCUIT_PROBABILITY_MIN)
    / (_BC_AUX_CIRCUIT_PROBABILITY_MAX
       - _BC_AUX_CIRCUIT_INITIAL_PROBABILITY)
)
_BC_AUX_CIRCUIT_GATE_PARAMETER_ABS_MAX = 8.0
_BC_AUX_CIRCUIT_KING_SUPPORT = "legal-non12-non14-renormalized"
_BC_AUX_MIN_DETERMINISTIC_A12_EPISODES = 2
_BC_AUX_MIN_DETERMINISTIC_A12_MARGIN = 1e-4
_BC_AUX_MIN_EXPECTED_A12_SAMPLES = 20.0
_BC_AUX_MIN_ACTUAL_A12_SAMPLES = 10
_BC_AUX_NEGATIVE_RATIO = 8
_BC_AUX_MIN_NEGATIVE_RATIO = 3
# Non-12 rows with feature297<0 and m12=False prove that the visible "already drank this window" latch really closed the
# worker-owned drink key. The rev8 mixture gives p12=0 by construction outside the predicate, so these rows
# serve as three-domain interface/mask evidence and no longer pose as legal negatives fit for the action-12 BCE.
_BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO = 1
_BC_AUX_UPDATE_EVERY = 1
_BC_AUX_POSITIVE_FRACTION = 0.25
_BC_AUX_POSITIVE_TARGET = 0.65
_BC_AUX_NEGATIVE_TARGET = 0.01
_BC_AUX_ANCHOR_KL_COEF = 0.25
_BC_V2_DEMOS_SCHEMA_VERSION = "bc-worker-v2-demos/5"
_BC_V2_REPORT_SCHEMA_VERSION = "bc-worker-v2/7"
_BC_V2_TEACHER_GENERATION = 2
_BC_V2_PREVENTIVE_THRESHOLDS = (0.65,)
# Collection itself inspects whole-pool integrity, so the producer creates an
# immutable one-shot marker before the first reset.  The previous 2101000 pool
# has already been opened and stays burned; the active v2 registry below is
# disjoint from every predecessor, active v1, and the R7 eval bank.
_BC_V2_COLLECTION_EPISODES = tuple(range(2_143_000, 2_143_384))
if (
    set(_WORKER_BC_DEMO_SEEDS) & _BURNED_BC_EPISODES
    or set(_BC_V2_COLLECTION_EPISODES) & _BURNED_BC_EPISODES
    or set(_WORKER_BC_DEMO_SEEDS) & set(_BC_V2_COLLECTION_EPISODES)
):
    raise RuntimeError("active BC fresh seed registries overlap with already-viewed pools/each other")
_BC_V2_N12_MIN = 122
# Shared source of truth for the BC-v2 producer/consumer. bc_worker imports from this module in reverse, and the training entry point
# recomputes the three-domain receipt directly with the same constants, so no duplicate schema/split can drift silently.
_A12_CALIBRATION_SCHEMA_VERSION = "a12-teacher-boundary/3"
_A12_CALIBRATION_HP_FEATURE = 0
_A12_CALIBRATION_DRINK_LATCH_FEATURE = 297
# A 0.60 fit target left only ten percentage points above the 0.50 independent
# gate.  With roughly fifty a12 examples per episode-level validation split,
# ordinary sampling variation made that margin unreliable.  The fresh rev13
# pool freezes a 0.75 fit-only target before it is opened; final heldout remains
# unavailable to calibration.
_A12_CALIBRATION_TRAIN_RECALL_TARGET = 0.75
_A12_CALIBRATION_PREDICATE = (
    "visible-hp-band-live-m12-and-no-prior-window-drink")
_A12_VISIBLE_HP_BOUNDARY_EPS = 1e-6
_BC_FINAL_SPLIT_SEED = 23
_BC_FINAL_HOLDOUT_POOL_SCHEMA = "bc-final-holdout-pool/1"
_BC_FINAL_HOLDOUT_MARKER_SCHEMA = "bc-final-holdout-consumption/2"
_BC_SELECTION_VALIDATION_FRACTION = 0.10
_BC_SELECTION_SPLIT_SEED = 2301
# Behaviour hard gates shared by BC-v2 / smoke. All thresholds are computed on the original held-out distribution, not on
# the 1:8 enriched training bank; denominators and the real per-sample masks go into the receipt together.
_A12_PRECISION_MIN = 0.05
_A12_RECALL_MIN = 0.50
# The BC-v2 teacher itself is stricter than the later PPO release gate; bc_worker imports from here,
# so its dedicated 0.5 gate cannot again be overridden implicitly by the generic ">=300-class recall>=0.85".
_BC_V2_TEACHER_RECALL_MIN = 0.50
_A12_FPR_MAX = 0.002
_A12_PREDICTED_SHARE_MIN = 0.00005
_A12_PREDICTED_SHARE_MAX = 0.01
_A12_HIGH_HP_FALSE_DRINK_MAX = 0.001
_A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX = 1e-4
_A12_LEGAL_NEGATIVE_PROBABILITY_MAX = 1e-3
_A13_SPILLOVER_MAX = 0.02
# The final release judges drift against "the root policy at the first bc_aux mount", not each leg's starting point. Otherwise each leg
# could lose 15% and, after several continuations, accumulate enough to wreck combat/exploration while every leg PASSes.
_BC_AUX_ROOT_ARGMAX_DRIFT_MAX = 0.20
_BC_AUX_ROOT_TV_MAX = 0.15
_BC_AUX_ROOT_KL_MAX = 0.25
_BC_AUX_CRITICAL_RETENTION_MIN = 0.50
_BC_AUX_CRITICAL_ACTIONS = (9, 10, 13)
_BC_AUX_BEHAVIOR_METRIC_KEYS = frozenset({
    "scope", "mask_mode", "pairs", "tp", "fp", "fn", "tn",
    "true_a12", "non_a12", "all_non_a12", "predicted_a12",
    "predicted_a12_episodes", "predicted_a12_margin_min",
    "precision_12", "recall_12", "fpr_12", "predicted_share_12",
    "high_hp_non_a12", "high_hp_false_drinks",
    "high_hp_false_drink_rate", "predicted_share_13", "true_share_13",
    "eligible_probability_12_min", "eligible_probability_12_mean",
    "eligible_probability_12_max",
    "legal_negative_probability_12_mean",
    "legal_negative_probability_12_max",
    "legal_negative_probability_12_sum",
    "a13_reference", "a13_reference_share", "a13_spillover",
    "mean_probability_12", "anchor",
})
_BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION = "bc-aux-behavior/8"
_BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION = "bc-aux-liveness-preflight/4"
_BC_V2_PASS_KEYS = frozenset({
    "schema_version", "teacher_generation", "preventive_threshold",
    "pairs", "held_out_top1", "held_out_pairs", "held_out_episodes",
    "class_recalls", "class_weighted_retry",
    "n12", "n12_gate_min", "n12_by_episode",
    "recall_12", "recall_12_denominator", "recall_12_gate_min",
    "class_share_12", "class_share_13", "belt_economy",
    "a12_behavior", "a12_behavior_gate", "a12_calibration",
    "collection_episodes", "class_weights",
    "data_gate", "protocol_version", "implementation_sha256",
    "generator_sha256", "manager_npz_sha256", "policy_sha256",
    "demos_sha256", "final_pool_sha256",
    "final_holdout_marker_sha256",
})
# Compatibility symbol for the historical v3 driver; the current training path never treats it as truth. Protocol v4
# changed the Worker/mask/window semantics, BC-v1 must be re-collected, and the probe identity is now bound dynamically to the
# demos_sha256 of the strict PASS receipt of the current protocol+implementation.
_BC_V1_DEMOS_SHA256 = (
    "3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363")
# Dedicated rng seed of the E5 probe (following the DryAnchorSentinel rng(26) precedent, same shape as its twin;
# the read-only probe owns its stream and never touches the training RNG).
_E5_PROBE_RNG_SEED = 26
# Per-group sampling cap for E5 probe demo states (following the DryAnchorSentinel fixed-2000 precedent).
_E5_PROBE_GROUP_CAP = 2000
_BC_REPLAY_SEEDS = tuple(range(7000, 7032))
_BC_REPLAY_CACHE: dict[tuple[str, str, str, str], dict] = {}
_BC_PASS_KEYS = {
    "data_gate": {
        "schema_version", "pairs", "held_out_top1", "held_out_pairs",
        "held_out_episodes", "class_recalls", "class_weighted_retry",
        "data_gate", "protocol_version", "implementation_sha256",
        "generator_sha256", "manager_npz_sha256", "policy_sha256",
        "demos_sha256", "final_pool_sha256",
        "final_holdout_marker_sha256",
    },
    "hypothesis": {
        "schema_version", "pairs", "teacher_demo_mean", "bc_replay_7000",
        "teacher_7000", "ratio", "hypothesis", "protocol_version",
        "implementation_sha256", "generator_sha256", "policy_sha256",
    },
    "memoryless_hypothesis": {
        "schema_version", "pairs", "teacher_mean_demo",
        "bc_replay_mean_7000s", "teacher_replay_mean_7000s", "ratio",
        "memoryless_hypothesis", "protocol_version",
        "implementation_sha256", "generator_sha256", "policy_sha256",
    },
}


def _require(condition: bool, message: str) -> None:
    """Training contracts cannot use assert: `python -O` strips asserts entirely."""
    if not condition:
        raise ValueError(message)


def _masked_action_or_first_legal(
        requested, mask, *, n_actions: int, label: str) -> int:
    """Return ``requested`` when legal, otherwise the first legal action.

    Manager teachers intentionally remain simple heuristics and may propose an
    option that a newer environment contract has forced closed.  Falling back
    to a hard-coded action merely repeats the same violation when that action
    is also masked.  Validate the complete mask and derive the deterministic
    fallback from the mask itself; an all-false mask is an environment contract
    failure and must stop the run.
    """
    import numpy as np

    valid = np.asarray(mask, dtype=bool)
    _require(
        valid.shape == (n_actions,),
        f"{label} action mask shape invalid: {valid.shape} != {(n_actions,)}",
    )
    legal = np.flatnonzero(valid)
    _require(len(legal) > 0, f"{label} action mask is all False")
    if (
        isinstance(requested, (int, np.integer))
        and not isinstance(requested, (bool, np.bool_))
    ):
        candidate = int(requested)
        if 0 <= candidate < n_actions and bool(valid[candidate]):
            return candidate
    return int(legal[0])


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite_number(value, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _requested_drink_sovereignty(args) -> bool | None:
    """Return the CLI's tri-state action-12 request.

    New command lines use ``drink_sovereignty=None`` for automatic binding to
    a strict Worker NPZ contract.  Older unit fixtures still expose only the
    historical negative flag, so retain that input spelling without letting it
    leak back into the production parser.
    """
    if hasattr(args, "drink_sovereignty"):
        requested = getattr(args, "drink_sovereignty")
        _require(
            requested is None or isinstance(requested, bool),
            "drink_sovereignty request must be a bool or None",
        )
        return requested
    legacy_disabled = getattr(args, "no_drink_sovereignty", False)
    _require(
        isinstance(legacy_disabled, bool),
        "no_drink_sovereignty compatibility field must be a bool",
    )
    return not legacy_disabled


def _effective_drink_sovereignty(args) -> bool:
    """Return the already-resolved action-12 semantics for contracts/config."""
    if hasattr(args, "resolved_drink_sovereignty"):
        resolved = getattr(args, "resolved_drink_sovereignty")
        _require(
            isinstance(resolved, bool),
            "resolved_drink_sovereignty must be a bool",
        )
        return resolved
    requested = _requested_drink_sovereignty(args)
    # Worker training and OptionsEnv without a tagged Worker historically use
    # environment-managed action 12.  ``None`` is only materially different
    # while assembling a strict Worker NPZ, where the metadata is authoritative.
    return True if requested is None else requested


def _read_worker_zip_contract(
        path: str | pathlib.Path, *, expected_sha256: str | None = None) -> dict:
    """R9: read the diablogym_contract embedded in an SB3 zip (no torch import, no policy built).

    The deployment contract of a certified release artifact is the single source of truth for action12/observation view; the main process parses and
    locks it from the same byte string before any VecEnv subprocess loads.
    """
    try:
        payload = pathlib.Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"worker zip unreadable: {path}: {exc}") from exc
    if expected_sha256 is not None:
        actual = hashlib.sha256(payload).hexdigest()
        _require(actual == expected_sha256,
                 f"worker zip SHA256 drift: {actual} != {expected_sha256}")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            data = json.loads(archive.read("data"))
    except (KeyError, TypeError, ValueError, zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"worker zip is not a parseable SB3 checkpoint: {path}: {exc}") from exc
    contract = data.get("diablogym_contract")
    _require(isinstance(contract, dict),
             "--worker-zip must carry a diablogym_contract (training contract of a certified release artifact)")
    _require(contract.get("contract_revision") == 26,
             "--worker-zip training contract contract_revision must be 26, "
             f"got {contract.get('contract_revision')!r}")
    view = contract.get("worker_policy_observation_view")
    _require(isinstance(view, str) and bool(view),
             "--worker-zip contract is missing worker_policy_observation_view")
    _require(isinstance(contract.get("drink_sovereignty"), bool),
             "--worker-zip contract is missing bool drink_sovereignty")
    return contract


def _resolve_training_drink_sovereignty(
        args, *, worker_npz_sha256: str | None,
        worker_zip_sha256: str | None = None) -> bool:
    """Resolve a tri-state CLI request against immutable Worker metadata."""
    requested = _requested_drink_sovereignty(args)
    worker_zip = getattr(args, "worker_zip", None)
    if getattr(args, "options", False) and worker_zip:
        # R9: zip worker branch -- the rev26 certified contract is the single source of truth for action12
        # (certified release artifact drink_sovereignty=False -> always masked).
        _require(
            _is_sha256(worker_zip_sha256),
            "Options Worker (zip) action12 resolution must be bound to the captured zip SHA256",
        )
        contract = _read_worker_zip_contract(
            worker_zip, expected_sha256=worker_zip_sha256)
        derived = bool(contract["drink_sovereignty"])
        _require(
            requested is None or requested == derived,
            "command-line drink_sovereignty conflicts with the --worker-zip contract: "
            f"requested={requested},contract={derived}",
        )
        return derived
    worker_npz = getattr(args, "worker_npz", None)
    if getattr(args, "options", False) and worker_npz:
        from diablogym import NumpyManager
        from diablogym.worker_env import (
            WORKER_ACTION12_ENVIRONMENT_MASK,
            WORKER_ACTION12_PERMANENTLY_MASKED,
        )

        _require(
            _is_sha256(worker_npz_sha256),
            "Options Worker action12 resolution must be bound to the captured NPZ SHA256",
        )
        net = NumpyManager(
            worker_npz, expected_sha256=worker_npz_sha256)
        net.require_io_shape(298, 15, "Options worker")
        mode = net.worker_action12_mode
        _require(
            mode in {
                WORKER_ACTION12_ENVIRONMENT_MASK,
                WORKER_ACTION12_PERMANENTLY_MASKED,
            },
            f"Worker NPZ action12 mode invalid: {mode!r}",
        )
        derived = mode == WORKER_ACTION12_ENVIRONMENT_MASK
        _require(
            requested is None or requested == derived,
            "command-line drink_sovereignty conflicts with the Worker NPZ action12 contract"
            f": requested={requested},contract={mode!r}",
        )
        return derived
    return True if requested is None else requested


def _bc_final_holdout_pool_spec(generation: int, seeds) -> dict:
    """Canonical one-shot final-pool identity shared by producer/consumer."""
    _require(
        _is_plain_int(generation) and generation in (1, 2),
        f"BC final heldout generation invalid: {generation!r}")
    raw_seeds = list(seeds)
    _require(
        bool(raw_seeds)
        and all(_is_plain_int(seed) for seed in raw_seeds)
        and len(raw_seeds) == len(set(raw_seeds)),
        "BC final heldout seed registry must be a non-empty table of unique integers")
    normalized = [int(seed) for seed in raw_seeds]
    return {
        # Pool identity must not change when the marker file format evolves.
        # Otherwise a schema bump could make the same episodes look fresh.
        "schema_version": _BC_FINAL_HOLDOUT_POOL_SCHEMA,
        "teacher_generation": generation,
        "episode_seeds": normalized,
        "outer_split_seed": _BC_FINAL_SPLIT_SEED,
        "outer_holdout_fraction": 0.10,
        "split_unit": "whole-episode",
    }


def _bc_final_holdout_marker_identity(generation: int, seeds) -> tuple[dict, str]:
    spec = _bc_final_holdout_pool_spec(generation, seeds)
    payload = json.dumps(
        spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return spec, hashlib.sha256(payload).hexdigest()


def _bc_final_holdout_marker_path(
        out_dir: str | pathlib.Path, generation: int, seeds
) -> tuple[pathlib.Path, dict, str]:
    """Return the stable registry path for a one-shot BC pool.

    The registry is a sibling of the BC artifact directories, not a child of
    either bundle.  Renaming/archiving ``bc-worker[-v2]`` therefore cannot
    make the same seed pool look unused again.
    """
    spec, pool_sha256 = _bc_final_holdout_marker_identity(
        generation, seeds)
    registry = pathlib.Path(out_dir).parent / "_bc_final_holdout_registry"
    return registry / f"{pool_sha256}.json", spec, pool_sha256


def _assert_bc_final_holdout_pool_disjoint(
        out_dir: str | pathlib.Path, generation: int, seeds) -> None:
    """Reject any exact *or partial* reuse of a previously opened BC pool.

    The marker filename binds the exact pool, but exact-hash lookup alone does
    not prevent a later campaign from registering an overlapping superset or
    shifted range.  Every marker is therefore treated as an append-only global
    episode registry across both teacher generations.  A malformed historical
    marker is fail-closed: ignoring it could silently relabel consumed episodes
    as fresh.
    """
    marker, spec, _ = _bc_final_holdout_marker_path(
        out_dir, generation, seeds)
    requested = set(spec["episode_seeds"])
    registry = marker.parent
    if not registry.exists():
        return
    from eval_contract import EvalContractError, strict_json_loads

    _require(registry.is_dir(),
             f"BC final heldout registry is not a directory: {registry}")
    _require(not registry.is_symlink(),
             f"BC final heldout registry does not allow symbolic links: {registry}")
    entries = sorted(registry.iterdir())
    unexpected = [
        entry for entry in entries
        if entry.name != ".registry.lock" and entry.suffix != ".json"
    ]
    _require(
        not unexpected,
        "BC final heldout registry contains unknown leftovers; cannot prove it is unconsumed: "
        f"{unexpected}",
    )
    for lock in (entry for entry in entries
                 if entry.name == ".registry.lock"):
        _require(lock.is_file() and not lock.is_symlink(),
                 f"BC final heldout registry lock is not a regular file: {lock}")
    marker_keys = {
        "schema_version",
        "teacher_generation",
        "episode_seeds",
        "outer_split_seed",
        "outer_holdout_fraction",
        "split_unit",
        "pool_sha256",
        "marker_schema_version",
        "started_at_ns",
        "provenance",
        "consumption_stage",
    }
    for existing in (entry for entry in entries if entry.suffix == ".json"):
        _require(existing.is_file() and not existing.is_symlink(),
                 f"BC final heldout registry marker is not a regular file: {existing}")
        try:
            record = strict_json_loads(existing.read_bytes())
        except (OSError, EvalContractError) as exc:
            raise ValueError(
                f"BC final heldout registry old marker unparseable: {existing}"
            ) from exc
        _require(
            isinstance(record, dict)
            and set(record) == marker_keys
            and _is_plain_int(record.get("teacher_generation"))
            and record["teacher_generation"] in (1, 2)
            and isinstance(record.get("episode_seeds"), list)
            and bool(record["episode_seeds"])
            and all(_is_plain_int(seed)
                    for seed in record["episode_seeds"])
            and len(record["episode_seeds"])
            == len(set(record["episode_seeds"])),
            f"BC final heldout registry old marker pool identity invalid: {existing}",
        )
        existing_spec, existing_pool_sha256 = (
            _bc_final_holdout_marker_identity(
                record["teacher_generation"],
                record["episode_seeds"],
            )
        )
        _require(
            all(record.get(key) == value
                for key, value in existing_spec.items())
            and record.get("pool_sha256") == existing_pool_sha256
            and existing.stem == existing_pool_sha256
            and record.get("marker_schema_version")
            == _BC_FINAL_HOLDOUT_MARKER_SCHEMA
            and _is_plain_int(record.get("started_at_ns"))
            and record["started_at_ns"] > 0
            and isinstance(record.get("provenance"), dict)
            and bool(record["provenance"])
            and record.get("consumption_stage")
            == "before_pool_collection",
            f"BC final heldout registry old marker full identity invalid: {existing}",
        )
        overlap = requested.intersection(record["episode_seeds"])
        _require(
            not overlap,
            "BC final heldout episodes partially/fully overlap a consumed registry: "
            f"marker={existing},overlap={sorted(overlap)[:16]},"
            f"overlap_n={len(overlap)}",
        )


def _validate_bc_final_holdout_marker(
        out_dir: str | pathlib.Path, generation: int, seeds,
        expected_report: dict) -> dict:
    """Require the immutable pre-collection marker bound by a PASS report."""
    from eval_contract import EvalContractError, strict_json_loads

    marker, spec, pool_sha256 = _bc_final_holdout_marker_path(
        out_dir, generation, seeds)
    try:
        marker_payload = marker.read_bytes()
        record = strict_json_loads(marker_payload)
    except (OSError, EvalContractError) as exc:
        raise ValueError(
            f"BC final heldout one-shot marker missing/unreadable: {marker}") from exc
    expected_keys = {
        *spec,
        "pool_sha256",
        "marker_schema_version",
        "started_at_ns",
        "provenance",
        "consumption_stage",
    }
    _require(
        isinstance(record, dict) and set(record) == expected_keys,
        "BC final heldout marker fields/schema not exact")
    _require(
        all(record[key] == value for key, value in spec.items())
        and record["pool_sha256"] == pool_sha256
        and record["marker_schema_version"]
        == _BC_FINAL_HOLDOUT_MARKER_SCHEMA
        and _is_plain_int(record["started_at_ns"])
        and record["started_at_ns"] > 0
        and record["consumption_stage"] == "before_pool_collection",
        "BC final heldout marker pool/time/consumption-stage identity does not close")
    _require(
        isinstance(expected_report, dict)
        and expected_report.get("final_pool_sha256") == pool_sha256
        and _is_sha256(
            expected_report.get("final_holdout_marker_sha256"))
        and expected_report["final_holdout_marker_sha256"]
        == hashlib.sha256(marker_payload).hexdigest(),
        "BC PASS report not exactly bound to the final pool/marker bytes")
    provenance = record["provenance"]
    _require(isinstance(provenance, dict), "BC final marker provenance is not an object")
    provenance_keys = {
        "schema_version",
        "protocol_version",
        "implementation_sha256",
        "generator_sha256",
        "manager_npz_sha256",
    }
    if generation == 2:
        provenance_keys |= {"teacher_generation", "preventive_threshold"}
    _require(
        set(provenance) == provenance_keys
        and all(
            provenance.get(key) == expected_report.get(key)
            for key in provenance_keys
        ),
        "BC final heldout marker provenance disagrees with the PASS report")
    return record


def _checkpoint_path(path: str | pathlib.Path) -> pathlib.Path:
    """Tolerate an omitted `.zip` on the command line, following SB3 rules."""
    p = pathlib.Path(path)
    return p if p.exists() or p.suffix == ".zip" else pathlib.Path(f"{p}.zip")


def _capture_file_sha256(path: str | pathlib.Path, label: str) -> str:
    p = pathlib.Path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} unreadable: {p}: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


_IMPLEMENTATION_SOURCE_FILES = (
    "train/train_ppo.py",
    "train/migrate_resource_candidate.py",
    "train/migrate_completion_candidate.py",
    # R18-B3b (2026-09-07): the sustain-loot-v1 warm-start schema/2.
    "train/migrate_loot_candidate.py",
    "train/prefix_worker.py",
    "train/training_diagnostics.py",
    "train/rollout_diagnostics.py",
    "train/analyze_first_update.py",
    "train/leashed_ppo.py",
    "train/models.py",
    "train/eval_contract.py",
    "python/diablogym/__init__.py",
    "python/diablogym/controller_wire.py",
    "python/diablogym/env.py",
    "python/diablogym/nav.py",
    "python/diablogym/options_env.py",
    "python/diablogym/worker_env.py",
    "python/diablogym/completion_clock.py",
    "python/diablogym/resource_protocol.py",
    "python/diablogym/resource_navigation.py",
    "python/diablogym/resource_sustain.py",
    "python/diablogym/resource_gold_memory.py",
    "python/diablogym/resource_sustain_gold.py",
    "python/diablogym/resource_sustain_armor.py",
    "python/diablogym/resource_sustain_completion.py",
    # R18-B6 (2026-09-07): every law module that exists in the tree is bound.
    # R18-B5 flagged the gap: the retreat/portal/aggro/engagement/sweep/
    # identify/weapon-upgrade laws all decide what the training window does,
    # yet none of them were hashed, so a resume could silently ride a changed
    # law while the bundle digest still matched.  The three sustain modules
    # below were bound by eval_contract.PROTOCOL_SOURCE_FILES but not here,
    # which is the same gap seen from the other side.  boss_avoidance.py is
    # deliberately absent: R18-J is not merged, so the file does not exist and
    # a non-existent path would fail closed at _implementation_bundle_sha256.
    "python/diablogym/resource_sustain_combat.py",
    "python/diablogym/resource_sustain_combinations.py",
    "python/diablogym/resource_sustain_loot.py",
    "python/diablogym/resource_sustain_protection.py",
    "python/diablogym/resource_retreat.py",
    "python/diablogym/resource_portal.py",
    "python/diablogym/resource_sweep.py",
    "python/diablogym/resource_identify.py",
    "python/diablogym/resource_weapon_upgrade.py",
    "python/diablogym/aggro_cap.py",
    "python/diablogym/engagement.py",
)


def _implementation_bundle_sha256() -> str:
    """Bind resume to code, native binaries and the actual game content."""
    import sysconfig

    from eval_contract import (content_identity, engine_binary_path,
                               runtime_versions_identity)

    root = pathlib.Path(__file__).resolve().parents[1]
    rel_paths = [
        pathlib.Path(relative)
        for relative in _IMPLEMENTATION_SOURCE_FILES
    ]
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    _require(bool(suffix), "this Python has no EXT_SUFFIX; cannot bind the native bridge")
    rel_paths.append(pathlib.Path("build") / f"_diablogym{suffix}")

    digest = hashlib.sha256()
    for rel in rel_paths:
        p = root / rel
        try:
            payload = p.read_bytes()
        except OSError as exc:
            raise ValueError(f"implementation binding file unreadable: {p}: {exc}") from exc
        name = rel.as_posix().encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    # The bridge dynamically links the engine, so hashing only the extension
    # is insufficient. Locate the one actual engine binary fail-closed rather
    # than silently omitting it on another build layout.
    engine = engine_binary_path(root)
    try:
        engine_payload = engine.read_bytes()
    except OSError as exc:
        raise ValueError(f"implementation binding engine unreadable: {engine}: {exc}") from exc
    engine_label = b"native-engine"
    digest.update(len(engine_label).to_bytes(4, "big"))
    digest.update(engine_label)
    digest.update(len(engine_payload).to_bytes(8, "big"))
    digest.update(engine_payload)

    # MPQ and Resources change world generation, collision, monsters and
    # rewards without changing source or binary bytes. Bind content rather
    # than host-specific absolute paths so an identical relocation is safe.
    content = content_identity(root)
    content_contract = {
        "game_data_sha256": content["game_data"]["sha256"],
        "assets_sha256": content["assets"]["sha256"],
        "assets_file_count": content["assets"]["file_count"],
    }
    encoded_content = json.dumps(
        content_contract, sort_keys=True, separators=(",", ":")).encode("ascii")
    content_label = b"runtime-content-v1"
    digest.update(len(content_label).to_bytes(4, "big"))
    digest.update(content_label)
    digest.update(len(encoded_content).to_bytes(8, "big"))
    digest.update(encoded_content)

    # BC generators call this helper too.  Binding only source/native/content
    # would let a policy trained under a different NumPy/Torch/SB3 numerical
    # stack present the same implementation identity later.
    encoded_versions = json.dumps(
        runtime_versions_identity(), sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    versions_label = b"runtime-versions-v1"
    digest.update(len(versions_label).to_bytes(4, "big"))
    digest.update(versions_label)
    digest.update(len(encoded_versions).to_bytes(8, "big"))
    digest.update(encoded_versions)
    return digest.hexdigest()


def _validate_runtime_versions() -> None:
    from importlib.metadata import version

    actual = {name: version(name) for name in _RUNTIME_VERSIONS}
    # Cross-platform note (2026-07-27 WSL2 port): Linux wheels carry a local version segment (2.12.1+cpu);
    # the threshold compares the public version segment; the identity record still stores the full local version (same as the eval_contract revision).
    mismatches = {name: (actual[name], expected)
                  for name, expected in _RUNTIME_VERSIONS.items()
                  if actual[name] != expected
                  and actual[name].split("+", 1)[0] != expected}
    _require(not mismatches,
             f"training runtime version drift (upgrades must redo the numeric regression): {mismatches}")


def _validate_model_recipe(model, expected_target_kl=None) -> None:
    """Reject a foreign/resumed PPO whose hidden defaults changed."""
    clip_samples = tuple(float(model.clip_range(progress))
                         for progress in _SCHEDULE_PROBES)
    clip_vf_samples = (None if model.clip_range_vf is None else
                       tuple(float(model.clip_range_vf(progress))
                             for progress in _SCHEDULE_PROBES))
    actual = {
        "gae_lambda": float(model.gae_lambda),
        "n_epochs": int(model.n_epochs),
        # A schedule can equal the registered constant at progress=1 while
        # silently annealing later.  Sample the beginning, midpoint and end.
        "clip_range": clip_samples,
        "clip_range_vf": clip_vf_samples,
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "normalize_advantage": bool(model.normalize_advantage),
        "target_kl": model.target_kl,
        "use_sde": bool(getattr(model, "use_sde", False)),
        "sde_sample_freq": int(getattr(model, "sde_sample_freq", -1)),
    }
    expected_recipe = dict(_ALGORITHM_RECIPE)
    expected_recipe["target_kl"] = expected_target_kl
    expected_recipe["clip_range"] = (
        _ALGORITHM_RECIPE["clip_range"],) * len(_SCHEDULE_PROBES)
    if _ALGORITHM_RECIPE["clip_range_vf"] is not None:
        expected_recipe["clip_range_vf"] = (
            _ALGORITHM_RECIPE["clip_range_vf"],) * len(_SCHEDULE_PROBES)
    differences = {
        key: (actual[key], expected)
        for key, expected in expected_recipe.items()
        if (actual[key] != expected if not isinstance(expected, float)
            else not math.isclose(actual[key], expected, rel_tol=0, abs_tol=1e-12))
    }
    _require(not differences,
             f"PPO implicit algorithm recipe drift (foreign/resume checkpoint): {differences}")


def _worker_policy_observation_view(args) -> str | None:
    if not getattr(args, "worker", False):
        return None
    explicit = getattr(args, "worker_policy_observation_view", None)
    legacy_flag = bool(getattr(
        args, "legacy_worker_policy_observation_view", False))
    if explicit is not None:
        return str(explicit)
    if _bc_aux_structural_active(args):
        return _WORKER_VIEW_A12_OVERLAY
    if legacy_flag:
        return _WORKER_VIEW_LEGACY_V3
    return None


def _root_context_critic_gradient_clipping(max_grad_norm: float) -> dict:
    """Return the one registered clipping recipe from one finite bound."""
    value = float(max_grad_norm)
    _require(
        math.isfinite(value) and value > 0.0,
        "root/context/critic gradient clipping requires a finite positive max_grad_norm",
    )
    return {
        "mode": "separate-root-context-critic-v2",
        "root_max_norm": value / math.sqrt(2.0),
        "context_max_norm": value / math.sqrt(2.0),
        "combined_actor_max_norm": value,
        "critic_max_norm": value,
        "optimizer": "single",
        "trainable_shared_parameters": "forbidden",
    }


def _new_current_asymmetric_worker_policy():
    """Construct the registered policy topology without an environment."""
    import numpy as np
    import torch
    from gymnasium import spaces
    from leashed_ppo import (
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        AsymmetricWorkerMaskableActorCriticPolicy,
    )

    return AsymmetricWorkerMaskableActorCriticPolicy(
        spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ASYMMETRIC_WORKER_OBSERVATION_DIM,),
            dtype=np.float32,
        ),
        spaces.Discrete(15),
        lambda _progress_remaining: 3e-4,
        net_arch={"pi": [64, 64], "vf": [64, 64]},
        activation_fn=torch.nn.Tanh,
    )


@functools.lru_cache(maxsize=1)
def _cached_current_asymmetric_worker_runtime_evidence() -> dict:
    """Measure current capacity/layout without consuming caller RNG streams."""
    import numpy as np
    import torch
    from leashed_ppo import asymmetric_worker_runtime_evidence

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    try:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        policy = _new_current_asymmetric_worker_policy()
        return asymmetric_worker_runtime_evidence(policy)
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)


def _current_asymmetric_worker_runtime_evidence() -> dict:
    """Return an isolated copy so callers cannot corrupt the cached spec."""
    return copy.deepcopy(
        _cached_current_asymmetric_worker_runtime_evidence())


def _validate_registered_dual_worker_contract(
        contract: dict, *, expected_contract_revision: int,
        expected_worker_onpolicy_pg_audit_schema: str,
        runtime_evidence: dict | None = None) -> dict:
    """Validate a registered rev24+ dual-Worker architecture/optimizer ABI.

    Capacity is measured from the current policy topology instead of copied
    into another list of numeric constants.  The caller must supply the exact
    contract/audit pair so historical rev24 (/8) can be evaluated without
    either upgrading its claim or weakening current rev25 (/9).
    """
    from leashed_ppo import (
        ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
        ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        ASYMMETRIC_WORKER_FROZEN_ACTOR_PARAMETER_COUNT,
        ASYMMETRIC_WORKER_FROZEN_ACTOR_TENSOR_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CONTEXT_PARAMETER_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CONTEXT_TENSOR_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CRITIC_PARAMETER_COUNT,
        ASYMMETRIC_WORKER_FROZEN_CRITIC_TENSOR_COUNT,
        ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA,
        WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    _require(
        _REGISTERED_DUAL_WORKER_PG_AUDIT_SCHEMAS.get(
            expected_contract_revision)
        == expected_worker_onpolicy_pg_audit_schema,
        "dual-v4 contract revision/PG audit schema not registered or mismatched",
    )
    _validate_policy_source_roles(contract)
    evidence = (
        _current_asymmetric_worker_runtime_evidence()
        if runtime_evidence is None else runtime_evidence
    )
    _require(
        isinstance(evidence, dict)
        and evidence.get("schema")
        == ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA
        and isinstance(evidence.get("layout"), dict)
        and evidence["layout"].get("schema") == DUAL_WORKER_LAYOUT.schema
        and evidence["layout"].get("sha256")
        == DUAL_WORKER_LAYOUT_SHA256
        and evidence["layout"].get("observation_dim")
        == DUAL_WORKER_LAYOUT.observation_dim
        and isinstance(evidence.get("policy"), dict)
        and isinstance(evidence.get("context"), dict),
        "current asymmetric Worker runtime evidence invalid",
    )
    policy = evidence["policy"]
    context = evidence["context"]
    _require(
        context.get("parameter_count")
        == ASYMMETRIC_WORKER_FROZEN_CONTEXT_PARAMETER_COUNT
        and context.get("tensor_count")
        == ASYMMETRIC_WORKER_FROZEN_CONTEXT_TENSOR_COUNT
        and policy.get("actor_parameter_count")
        == ASYMMETRIC_WORKER_FROZEN_ACTOR_PARAMETER_COUNT
        and policy.get("actor_tensor_count")
        == ASYMMETRIC_WORKER_FROZEN_ACTOR_TENSOR_COUNT
        and policy.get("critic_parameter_count")
        == ASYMMETRIC_WORKER_FROZEN_CRITIC_PARAMETER_COUNT
        and policy.get("critic_tensor_count")
        == ASYMMETRIC_WORKER_FROZEN_CRITIC_TENSOR_COUNT,
        "current asymmetric Worker capacity/tensor topology drift",
    )
    actor_migration = (
        contract.get("actor_migration")
        if isinstance(contract, dict) else None)
    critic_migration = (
        contract.get("critic_migration")
        if isinstance(contract, dict) else None)
    algorithm_recipe = (
        contract.get("algorithm_recipe")
        if isinstance(contract, dict) else None)
    max_grad_norm = (
        algorithm_recipe.get("max_grad_norm")
        if isinstance(algorithm_recipe, dict) else None)
    action14_logit_bonus = (
        contract.get("worker_action14_logit_bonus")
        if isinstance(contract, dict) else None)
    _require(
        isinstance(max_grad_norm, (int, float))
        and not isinstance(max_grad_norm, bool)
        and math.isclose(
            float(max_grad_norm),
            float(_ALGORITHM_RECIPE["max_grad_norm"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "dual-v4 contract max_grad_norm not bound to the current recipe",
    )
    expected_clipping = _root_context_critic_gradient_clipping(
        float(max_grad_norm))
    _require(
        _is_plain_int(expected_contract_revision)
        and expected_contract_revision >= 24
        and isinstance(expected_worker_onpolicy_pg_audit_schema, str)
        and bool(expected_worker_onpolicy_pg_audit_schema)
        and isinstance(contract, dict)
        and contract.get("schema_version") == 2
        and contract.get("contract_revision") == expected_contract_revision
        and contract.get("mode") == "worker"
        and contract.get("worker_policy_observation_view")
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
        and contract.get("worker_episode_boundary")
        == _WORKER_EPISODE_BOUNDARY_V24
        and contract.get("worker_window_bootstrap")
        == "next-learning-window"
        and contract.get("worker_no_progress_timeout")
        == _WORKER_NO_PROGRESS_TIMEOUT_CONTRACT
        and isinstance(action14_logit_bonus, (int, float))
        and not isinstance(action14_logit_bonus, bool)
        and math.isfinite(float(action14_logit_bonus))
        and 0.0 <= float(action14_logit_bonus) <= 10.0
        and contract.get("observation_shape")
        == [DUAL_WORKER_LAYOUT.observation_dim]
        and contract.get("action_n") == 15
        and isinstance(actor_migration, dict)
        and actor_migration.get("method")
        == _ASYMMETRIC_ACTOR_INIT_METHOD
        and actor_migration.get("context_architecture")
        == _ASYMMETRIC_CONTEXT_ARCHITECTURE
        and actor_migration.get("controller_layout_schema")
        == DUAL_WORKER_LAYOUT.schema
        and actor_migration.get("controller_layout_sha256")
        == DUAL_WORKER_LAYOUT_SHA256
        and actor_migration.get("target_actor_parameter_tensors")
        == policy.get("actor_tensor_count")
        and actor_migration.get("target_actor_parameter_count")
        == policy.get("actor_parameter_count")
        and actor_migration.get("context_parameter_tensors")
        == context.get("tensor_count")
        and actor_migration.get("context_parameter_count")
        == context.get("parameter_count")
        and actor_migration.get("context_initialization") == {
            "hidden": ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
            "output": "exact-zero-disabled-through-critic-warmup",
        }
        and actor_migration.get(
            "actor_context_excluded_observation_features")
        == list(ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES)
        and _is_sha256(actor_migration.get(
            "source_checkpoint_sha256"))
        and _is_sha256(actor_migration.get("source_actor_sha256"))
        and _is_sha256(actor_migration.get("migrated_actor_sha256"))
        and isinstance(critic_migration, dict)
        and critic_migration.get("method")
        == _ASYMMETRIC_CRITIC_RESET_METHOD
        and critic_migration.get("critic_architecture")
        == _ASYMMETRIC_CRITIC_ARCHITECTURE
        and critic_migration.get("controller_layout_schema")
        == DUAL_WORKER_LAYOUT.schema
        and critic_migration.get("controller_layout_sha256")
        == DUAL_WORKER_LAYOUT_SHA256
        and critic_migration.get("critic_parameter_tensors")
        == policy.get("critic_tensor_count")
        and critic_migration.get("critic_parameter_count")
        == policy.get("critic_parameter_count")
        and critic_migration.get("source_checkpoint_sha256")
        == actor_migration.get("source_checkpoint_sha256")
        and critic_migration.get("source_actor_sha256")
        == actor_migration.get("source_actor_sha256")
        and _is_plain_int(critic_migration.get("warmup_steps"))
        and critic_migration["warmup_steps"] > 0
        and critic_migration.get("gradient_clip_mode")
        == expected_clipping["mode"]
        and critic_migration.get("worker_onpolicy_pg_audit_schema")
        == expected_worker_onpolicy_pg_audit_schema
        and critic_migration.get(
            "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout")
        == WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT
        and contract.get("gradient_clipping") == expected_clipping,
        f"checkpoint is not a fully registered rev{expected_contract_revision} "
        "dual-v4 Worker",
    )
    return evidence


def _validate_current_dual_worker_contract(
        contract: dict, *, runtime_evidence: dict | None = None) -> dict:
    """Validate the exact current dual-Worker training/resume contract."""
    from leashed_ppo import WORKER_ONPOLICY_PG_AUDIT_SCHEMA

    return _validate_registered_dual_worker_contract(
        contract,
        expected_contract_revision=_CONTRACT_REVISION,
        expected_worker_onpolicy_pg_audit_schema=(
            WORKER_ONPOLICY_PG_AUDIT_SCHEMA),
        runtime_evidence=runtime_evidence,
    )


def _classify_dual_worker_resume(args, resume_data: dict) -> str | None:
    """Classify V28 migration versus an explicit environment-restart resume.

    SB3 checkpoints do not contain the native Diablo world, wrapper/controller
    state, VecEnv observations, or every RNG stream.  A dual checkpoint can
    therefore preserve parameters, Adam and counters, but it cannot continue
    the same trajectory.  Keep that weaker operation explicit in both the
    classifier name and the required CLI acknowledgement.
    """
    if (_worker_policy_observation_view(args)
            not in _WORKER_DUAL_VIEWS):
        return None
    _require(isinstance(resume_data, dict),
             "dual-v4 resume checkpoint data is not an object")
    saved = resume_data.get("diablogym_contract")
    reset_critic = bool(getattr(args, "reset_worker_critic", False))
    allow_legacy = bool(getattr(args, "allow_legacy_resume", False))
    if saved is None:
        _require(
            reset_critic and allow_legacy,
            "launching dual-v4 from a 298-dim old Worker requires both explicit "
            "--reset-worker-critic and --allow-legacy-resume",
        )
        return _DUAL_LEGACY_ACTOR_MIGRATION

    _require(isinstance(saved, dict),
             "dual-v4 checkpoint training_contract is not an object")
    _require(
        not reset_critic and not allow_legacy,
        "a checkpoint that already has a dual-v4 training_contract must continue parameters/optimizer; "
        "repeated reset critic or disguised legacy migration is forbidden",
    )
    # Validate the complete architecture/optimizer ABI before classifying the
    # weaker environment-restart continuation operation.
    _validate_current_dual_worker_contract(saved)
    _require(
        bool(getattr(args, "allow_environment_restart_resume", False)),
        "dual-v4 checkpoint holds no native world/wrapper/RNG state; parameter and Adam "
        "continuation restarts from new environment trajectories. Must pass "
        "--allow-environment-restart-resume",
    )
    return _DUAL_ENV_RESTART_CONTINUATION


def _build_resume_lineage(
        resume_data: dict, *, parent_sha256: str, operation: str,
        seed: int | None, optimizer_reset: bool,
        critic_reset: bool) -> dict:
    """Build the checkpoint-persisted receipt for a non-exact resume."""
    _require(
        isinstance(resume_data, dict)
        and _is_sha256(parent_sha256)
        and operation in {
            _DUAL_LEGACY_ACTOR_MIGRATION,
            _DUAL_ENV_RESTART_CONTINUATION,
        },
        "resume lineage input invalid",
    )
    parent_steps = resume_data.get("num_timesteps")
    _require(
        _is_plain_int(parent_steps) and parent_steps >= 0,
        "resume lineage parent num_timesteps invalid",
    )
    _require(
        seed is None
        or (_is_plain_int(seed) and 0 <= seed < 2**32),
        "resume lineage seed invalid",
    )
    previous = resume_data.get("_resume_lineage")
    if previous is None:
        generation = 1
    else:
        _require(
            isinstance(previous, dict)
            and previous.get("schema") == _RESUME_LINEAGE_SCHEMA
            and _is_plain_int(previous.get("generation"))
            and previous["generation"] >= 1,
            "resume checkpoint existing lineage invalid",
        )
        generation = previous["generation"] + 1
    return {
        "schema": _RESUME_LINEAGE_SCHEMA,
        "generation": generation,
        "operation": operation,
        "immediate_parent_sha256": parent_sha256,
        "immediate_parent_num_timesteps": parent_steps,
        "environment_state_mode":
            "reinitialized-no-native-or-wrapper-snapshot",
        "rng_state_mode": (
            "reseeded-from-explicit-cli-seed"
            if seed is not None else
            "runtime-new-streams-without-exact-restoration"
        ),
        "requested_seed": seed,
        "policy_parameter_state": (
            "v28-root-transplanted-context-canonical-init"
            if operation == _DUAL_LEGACY_ACTOR_MIGRATION
            else "checkpoint-preserved"
        ),
        "optimizer_state": "reset" if optimizer_reset else "preserved",
        "critic_state": "reset" if critic_reset else "preserved",
        "exact_trajectory_continuation": False,
    }


def _validate_worker_policy_observation_binding(args, model) -> None:
    """Close the CLI view selector against the policy class actually loaded."""
    if not args.worker:
        return
    from leashed_ppo import (
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        A12MixtureMaskableActorCriticPolicy,
        AsymmetricWorkerMaskableActorCriticPolicy,
    )

    expected_spec = _bc_aux_circuit_spec()
    custom = isinstance(
        model.policy, A12MixtureMaskableActorCriticPolicy)
    asymmetric = isinstance(
        model.policy, AsymmetricWorkerMaskableActorCriticPolicy)
    model_spec = getattr(model, "_bc_aux_circuit_spec", None)
    policy_spec = getattr(model.policy, "bc_aux_mixture_spec", None)
    if custom:
        _require(
            model_spec == expected_spec
            and policy_spec == expected_spec
            and getattr(model, "policy_class", None)
            is A12MixtureMaskableActorCriticPolicy,
            "actual A12 custom policy class/spec not closed",
        )
    else:
        _require(
            model_spec is None and policy_spec is None
            and getattr(model, "policy_class", None)
            is not A12MixtureMaskableActorCriticPolicy,
            "ordinary Worker carries a residual A12 class/spec",
        )
    expected_custom = _bc_aux_structural_active(args)
    _require(
        custom is expected_custom,
        "CLI structural A12 recipe disagrees with the actually loaded policy class",
    )
    legacy_view = bool(getattr(
        args, "legacy_worker_policy_observation_view", False))
    selected_view = _worker_policy_observation_view(args)
    _require(
        (
            custom
            and not legacy_view
            and selected_view == _WORKER_VIEW_A12_OVERLAY
        )
        or (
            asymmetric
            and not legacy_view
            and selected_view in _WORKER_DUAL_VIEWS
        )
        or (
            not custom
            and not asymmetric
            and selected_view == _WORKER_VIEW_LEGACY_V3
        ),
        "actual Worker policy class disagrees with the environment observation view",
    )
    _require(
        asymmetric
        is (selected_view in _WORKER_DUAL_VIEWS),
        "dual-family observation disagrees with the actual asymmetric policy class",
    )
    expected_action14_bonus = float(getattr(
        args, "worker_action14_logit_bonus", 0.0))
    actual_action14_bonus = getattr(
        model.policy, "action14_logit_bonus", None)
    _require(
        (
            asymmetric
            and isinstance(actual_action14_bonus, float)
            and math.isclose(
                actual_action14_bonus,
                expected_action14_bonus,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        or (
            not asymmetric
            and expected_action14_bonus == 0.0
            and actual_action14_bonus is None
        ),
        "CLI action14 logit prior disagrees with the actual Worker policy",
    )
    # R13: a11 cold-start prior binding (checked after the post-load double write, live == CLI).
    expected_action11_bonus = float(getattr(
        args, "worker_dive_action11_logit_bonus", 0.0))
    actual_action11_bonus = getattr(
        model.policy, "action11_logit_bonus", None)
    _require(
        (
            asymmetric
            and isinstance(actual_action11_bonus, float)
            and math.isclose(
                actual_action11_bonus, expected_action11_bonus,
                rel_tol=0, abs_tol=0)
        )
        or (
            not asymmetric
            and expected_action11_bonus == 0.0
        ),
        "CLI action11 logit prior disagrees with the actual Worker policy",
    )
    # R16: a13 potion-pickup prior binding (checked after the post-load double write, live == CLI; same as a11).
    expected_action13_bonus = float(getattr(
        args, "worker_potion_action13_logit_bonus", 0.0))
    actual_action13_bonus = getattr(
        model.policy, "action13_logit_bonus", None)
    _require(
        (
            asymmetric
            and isinstance(actual_action13_bonus, float)
            and math.isclose(
                actual_action13_bonus, expected_action13_bonus,
                rel_tol=0, abs_tol=0)
        )
        or (
            not asymmetric
            and expected_action13_bonus == 0.0
        ),
        "CLI action13 logit prior disagrees with the actual Worker policy",
    )
    actor_migration = getattr(model, "_actor_migration_receipt", None)
    _require(
        (asymmetric and isinstance(actor_migration, dict))
        or (not asymmetric and actor_migration is None),
        "asymmetric policy class disagrees with the actor migration receipt",
    )
    expected_observation_dim = (
        ASYMMETRIC_WORKER_OBSERVATION_DIM if asymmetric else 298)
    _require(
        tuple(model.observation_space.shape) == (expected_observation_dim,),
        "Worker policy class disagrees with the actual observation space shape: "
        f"{model.observation_space.shape} != ({expected_observation_dim},)",
    )


def _check_finite_tree(value, label: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        _require(torch.isfinite(value).all().item(), f"{label} contains NaN/Inf")
    elif isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite scalar")


def _validate_checkpoint_bytes(payload: bytes, label: str,
                               require_leashed: bool = False) -> dict:
    """Validate one immutable checkpoint byte string."""
    import torch

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            bad_member = archive.testzip()
            _require(bad_member is None, f"checkpoint CRC failed: {bad_member}")
            member_names = archive.namelist()
            _require(len(member_names) == len(set(member_names)),
                     f"checkpoint contains duplicate ZIP members: {label}")
            names = set(member_names)
            _require({"data", "policy.pth", "policy.optimizer.pth"} <= names,
                     f"checkpoint is missing key members: {label}")
            data = json.loads(archive.read("data"))
            saved_sb3 = archive.read("_stable_baselines3_version").decode().strip()
            _require(saved_sb3 == _RUNTIME_VERSIONS["stable-baselines3"],
                     f"checkpoint SB3 version {saved_sb3} does not match the runtime recipe")
            for name in sorted(n for n in names if n.endswith(".pth")):
                state = torch.load(io.BytesIO(archive.read(name)), map_location="cpu",
                                   weights_only=True)
                _check_finite_tree(state, name)
    except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile,
            json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("checkpoint"):
            raise
        raise ValueError(f"checkpoint unreadable/unsafe: {label}: {exc}") from exc
    _require(isinstance(data, dict), f"checkpoint data is not an object: {label}")
    try:
        steps = data["num_timesteps"]
    except KeyError as exc:
        raise ValueError("checkpoint num_timesteps missing/invalid") from exc
    _require(_is_plain_int(steps) and steps >= 0,
             "checkpoint num_timesteps must be a non-negative plain integer")
    if require_leashed:
        _require("distill_beta" in data,
                 "resume checkpoint is not LeashedMaskablePPO (missing distill_beta marker)")
    return data


def _validate_checkpoint_file(path: str | pathlib.Path,
                              require_leashed: bool = False) -> dict:
    """CRC + metadata + finite policy/Adam, read from the path exactly once."""
    p = _checkpoint_path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint unreadable/unsafe: {p}: {exc}") from exc
    return _validate_checkpoint_bytes(payload, str(p), require_leashed)


def _validate_leashed_metadata(data: dict) -> dict:
    _require("distill_beta" in data,
             "resume checkpoint is not LeashedMaskablePPO (missing distill_beta marker)")
    try:
        beta = float(data["distill_beta"])
    except (TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint distill_beta marker invalid") from exc
    _require(math.isfinite(beta) and beta >= 0,
             "resume checkpoint distill_beta must be a finite non-negative number")
    return data


def _validate_resumable_leashed_boundary(data: dict) -> dict:
    """Reject a contracted checkpoint whose latest rollout was not trained.

    SB3 increments ``num_timesteps`` during collection, before ``train()``
    consumes that buffer.  Saving in that interval creates a particularly
    dangerous checkpoint: a later run starts its budget/curriculum after the
    collected samples although the parameters never received their gradient.
    Registered legacy migration sources have no project contract and are
    validated by their own frozen SHA/topology gates; every contracted
    Leashed continuation must carry the stronger optimizer receipt.
    """
    contract = data.get("diablogym_contract")
    if contract is None:
        return data
    _require(isinstance(contract, dict),
             "resume training_contract is not an object")
    steps = data.get("num_timesteps")
    completed = data.get("_last_completed_ppo_rollout_steps")
    optimizer_steps = data.get("_ppo_optimizer_steps_completed")
    _require(
        _is_plain_int(completed) and completed == steps,
        "resume checkpoint is not a rollout boundary already consumed by the optimizer: "
        f"completed={completed!r},num_timesteps={steps!r}",
    )
    _require(
        _is_plain_int(optimizer_steps) and optimizer_steps > 0,
        "resume checkpoint is missing the completed PPO optimizer step receipt: "
        f"{optimizer_steps!r}",
    )
    n_steps = contract.get("n_steps")
    num_envs = contract.get("num_envs")
    _require(
        _is_plain_int(n_steps) and n_steps > 0
        and _is_plain_int(num_envs) and num_envs > 0,
        "resume training_contract is missing valid n_steps/num_envs",
    )
    quantum = n_steps * num_envs
    _require(
        steps % quantum == 0,
        "resume checkpoint num_timesteps not aligned to the training rollout quantum: "
        f"{steps} % {quantum}",
    )
    # These lists belong to the buffer that is about to be optimized.  Normal
    # Leashed saves deliberately exclude them; reject forged/foreign payloads
    # that try to carry an unconsumed stream across an environment restart.
    for key in (
        "_worker_onpolicy_pg_pending_receipts",
        "_bc_aux_pending_action_receipts",
    ):
        _require(
            key not in data or data[key] == [],
            f"resume checkpoint carries an unconsumed pending receipt: {key}",
        )
    return data


def _validate_leashed_checkpoint(path: str | pathlib.Path) -> dict:
    """Tell Leashed checkpoints apart by their saved metadata; an ordinary MaskablePPO may not pose as a resume source."""
    return _validate_resumable_leashed_boundary(
        _validate_leashed_metadata(
            _validate_checkpoint_file(path, require_leashed=True)))


def _capture_leashed_checkpoint(path: str | pathlib.Path) -> tuple[bytes, dict, str]:
    """Capture, validate and hash the exact bytes later passed to SB3.load()."""
    p = _checkpoint_path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"resume checkpoint unreadable: {p}: {exc}") from exc
    data = _validate_resumable_leashed_boundary(
        _validate_leashed_metadata(
            _validate_checkpoint_bytes(
                payload, str(p), require_leashed=True)))
    return payload, data, hashlib.sha256(payload).hexdigest()


def _select_batch_size(n_steps: int, n_envs: int, cap: int = 256) -> int:
    """Keep the 256 recipe; lower it only when the tail minibatch is exactly 1, to avoid std=NaN."""
    rollout_size = n_steps * n_envs
    _require(rollout_size > 1, "n_steps * num_envs must be greater than 1")
    for size in range(cap, 1, -1):
        if rollout_size <= size or rollout_size % size != 1:
            return size
    return rollout_size


def _reset_policy_optimizer(model, learning_rate: float) -> None:
    """Rebuild the policy optimizer, clearing all Adam moments carried over by a continuation.

    Weight tensors do not round-trip through state_dict; only the optimizer object and lr schedule are replaced.
    The caller must run this after load completes and before training starts, explicitly authorised by ``--reset-optimizer``.
    The helper is independently testable, preventing fake resets that "only zero the step while exp_avg remains".
    """
    import torch

    _require(math.isfinite(learning_rate) and learning_rate > 0,
             "reset optimizer learning rate must be a finite positive number")
    policy = model.policy
    optimizer_class = getattr(policy, "optimizer_class", None)
    optimizer_kwargs = dict(getattr(policy, "optimizer_kwargs", {}) or {})
    _require(optimizer_class is not None, "policy is missing optimizer_class; cannot rebuild safely")
    before = {name: value.detach().clone()
              for name, value in policy.state_dict().items()}
    model.learning_rate = float(learning_rate)
    model._setup_lr_schedule()
    optimizer_kwargs.pop("lr", None)
    policy.optimizer = optimizer_class(
        policy.parameters(), lr=float(model.lr_schedule(1.0)),
        **optimizer_kwargs)
    _require(not policy.optimizer.state,
             "reset optimizer still holds old state/Adam moments")
    for name, value in policy.state_dict().items():
        _require(torch.equal(before[name], value.detach()),
                 f"reset optimizer unexpectedly changed policy weights: {name}")


def _stable_named_tensor_sha256(state: dict, keys) -> str:
    """Match leashed_ppo's versioned named-parameter digest."""
    import numpy as np
    import torch

    keys = tuple(keys)
    _require(
        isinstance(state, dict) and all(key in state for key in keys),
        "policy state_dict is missing digest tensors",
    )
    digest = hashlib.sha256()
    for key in sorted(keys):
        tensor = state[key]
        _require(
            isinstance(tensor, torch.Tensor)
            and bool(torch.isfinite(tensor).all().item()),
            f"policy state_dict tensor invalid: {key}",
        )
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _checkpoint_policy_state(checkpoint_payload: bytes) -> dict:
    import torch

    try:
        with zipfile.ZipFile(io.BytesIO(checkpoint_payload)) as archive:
            state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=True,
            )
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("checkpoint policy.pth cannot be parsed") from exc
    _require(isinstance(state, dict), "checkpoint policy.pth is not a state_dict")
    return state


def _initialize_asymmetric_worker_actor(
        model, *, source_checkpoint_payload: bytes,
        source_checkpoint_sha256: str) -> dict:
    """Copy V28's six actor tensors into the current asymmetric policy."""
    import numpy as np
    import torch
    from sb3_contrib.common.maskable.policies import (
        MaskableActorCriticPolicy,
    )
    from stable_baselines3.common.torch_layers import (
        FlattenExtractor,
        MlpExtractor,
    )
    from leashed_ppo import (
        ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
        ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        ASYMMETRIC_WORKER_LEGACY_DIM,
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        AsymmetricWorkerMaskableActorCriticPolicy,
        LeashedMaskablePPO,
        actor_parameter_sha256,
        strict_actor_critic_parameter_partition,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    _require(
        isinstance(model.policy, AsymmetricWorkerMaskableActorCriticPolicy),
        "dual-v4 actor migration requires an asymmetric policy class")
    _require(str(getattr(model, "device", "")) == "cpu",
             "dual-v4 actor bitwise migration is currently certified on CPU only")
    _require(_is_sha256(source_checkpoint_sha256),
             "dual-v4 actor migration is missing the source checkpoint SHA")
    _require(
        hashlib.sha256(source_checkpoint_payload).hexdigest()
        == source_checkpoint_sha256,
        "dual-v4 actor migration source payload/SHA mismatch",
    )
    source = _checkpoint_policy_state(source_checkpoint_payload)
    critic_keys = (
        "mlp_extractor.value_net.0.weight",
        "mlp_extractor.value_net.0.bias",
        "mlp_extractor.value_net.2.weight",
        "mlp_extractor.value_net.2.bias",
        "value_net.weight",
        "value_net.bias",
    )
    expected_source_keys = set((*_POLICY_HEAD_KEYS, *critic_keys))
    _require(
        set(source) == expected_source_keys,
        "dual-v4 actor migration source policy state_dict fields not exact: "
        f"missing={sorted(expected_source_keys - set(source))},"
        f"extra={sorted(set(source) - expected_source_keys)}",
    )
    # ``BaseAlgorithm.load()`` reconstructs the source model and calls its
    # saved ``set_random_seed``.  This forensic load must not replace the
    # already-created target run's Python/NumPy/Torch streams with V28's fixed
    # seed: those streams drive action sampling and PPO minibatch permutations,
    # and collapsing them would invalidate independent training replications.
    # The migration is CPU-only above, so the complete Torch stream in scope is
    # the CPU generator captured here.
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    try:
        source_model = LeashedMaskablePPO.load(
            io.BytesIO(source_checkpoint_payload),
            env=None,
            device="cpu",
            teacher_path=None,
            teacher_sha256=None,
        )
    except Exception as exc:
        raise ValueError(
            "dual-v4 actor migration source model cannot be loaded as the real policy"
        ) from exc
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
    source_policy = source_model.policy
    _require(
        type(source_model) is LeashedMaskablePPO
        and type(source_policy) is MaskableActorCriticPolicy
        and getattr(source_model, "policy_class", None)
        is MaskableActorCriticPolicy
        and type(source_policy.features_extractor) is FlattenExtractor
        and bool(source_policy.share_features_extractor)
        and source_policy.activation_fn is torch.nn.Tanh
        and source_policy.net_arch == {
            "pi": [64, 64], "vf": [64, 64]}
        and type(source_policy.mlp_extractor) is MlpExtractor
        and tuple(source_model.observation_space.shape) == (298,)
        and getattr(source_model.action_space, "n", None) == 15,
        "dual-v4 actor migration source is not the registered "
        "plain Flatten/Tanh 298→64→64→15 policy",
    )
    expected_layers = (
        torch.nn.Linear,
        torch.nn.Tanh,
        torch.nn.Linear,
        torch.nn.Tanh,
    )
    _require(
        tuple(type(layer) for layer in source_policy.mlp_extractor.policy_net)
        == expected_layers
        and tuple(
            type(layer)
            for layer in source_policy.mlp_extractor.value_net
        ) == expected_layers,
        "dual-v4 actor migration source MLP layer order not exact",
    )
    loaded_source_state = source_policy.state_dict()
    _require(
        set(loaded_source_state) == expected_source_keys
        and all(
            torch.equal(loaded_source_state[key].detach().cpu(), source[key])
            for key in expected_source_keys
        ),
        "dual-v4 actor migration source actual policy disagrees with policy.pth",
    )
    source_actor_sha256 = _stable_named_tensor_sha256(
        source, _POLICY_HEAD_KEYS)
    source_critic_sha256 = _stable_named_tensor_sha256(
        source, critic_keys)
    target = model.policy.state_dict()
    with torch.no_grad():
        for key in _POLICY_HEAD_KEYS:
            _require(
                key in target
                and target[key].shape == source[key].shape
                and target[key].dtype == source[key].dtype,
                f"dual-v4 actor tensor cannot be transplanted bit for bit: {key}",
            )
            target[key].copy_(source[key])
    extractor = model.policy.mlp_extractor
    adapter = extractor.context_adapter
    parameter_groups = adapter.named_parameter_groups()
    _require(
        tuple(parameter_groups) == ("encoder", "interaction", "output"),
        "dual-v4 actor context semantic parameter groups drifted",
    )
    hidden_parameters = (
        *parameter_groups["encoder"],
        *parameter_groups["interaction"],
    )
    output_parameters = parameter_groups["output"]
    all_context_parameters = tuple(adapter.parameters())
    context_parameter_count = sum(
        int(parameter.numel()) for parameter in all_context_parameters)
    hidden_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in hidden_parameters
    )
    output_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in output_parameters
    )
    encoder_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in parameter_groups["encoder"]
    )
    interaction_nonzero = sum(
        int(torch.count_nonzero(parameter).item())
        for parameter in parameter_groups["interaction"]
    )
    _require(
        not extractor.actor_context_enabled
        and len(all_context_parameters) > 0
        and context_parameter_count > 0
        and encoder_nonzero > 0
        and interaction_nonzero > 0
        and hidden_nonzero == encoder_nonzero + interaction_nonzero
        and output_nonzero == 0
        and all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in (*hidden_parameters, *output_parameters)
        ),
        "dual-v4 actor context must start closed, hidden canonical nonzero, "
        "output canonical zero",
    )
    for key in _POLICY_HEAD_KEYS:
        _require(torch.equal(target[key], source[key]),
                 f"dual-v4 actor not bit-identical after transplant: {key}")
    # Tensor equality alone does not prove semantic equality if activation or
    # preprocessing topology drifted.  Run both graphs on a deterministic
    # nontrivial batch and require exact raw-logit identity before signing the
    # migration receipt.
    probe_rows = 4
    probe_legacy = torch.linspace(
        -1.0,
        1.0,
        steps=probe_rows * 298,
        dtype=source[_POLICY_HEAD_KEYS[0]].dtype,
    ).reshape(probe_rows, 298)
    with torch.no_grad():
        source_features = source_policy.extract_features(probe_legacy)
        source_hidden = source_policy.mlp_extractor.forward_actor(
            source_features)
        source_logits = source_policy.action_net(source_hidden)
        probe_dual = torch.zeros(
            (probe_rows, ASYMMETRIC_WORKER_OBSERVATION_DIM),
            dtype=probe_legacy.dtype,
            device=model.device,
        )
        probe_dual[:, :298] = probe_legacy.to(model.device)
        target_features = model.policy.extract_features(probe_dual)
        target_latent = model.policy.mlp_extractor.forward_actor(
            target_features)
        target_logits = model.policy.action_net(target_latent).cpu()
        # p_skip is a rollout sampler control, not deployable game state.
        # Prove at the adapter's nonzero pre-output (rather than at its
        # zero-initialized output) that the semantic exclusion is structural.
        exclusion_probe = torch.linspace(
            -0.75,
            0.75,
            steps=probe_rows * ASYMMETRIC_WORKER_OBSERVATION_DIM,
            dtype=probe_legacy.dtype,
            device=model.device,
        ).reshape(probe_rows, ASYMMETRIC_WORKER_OBSERVATION_DIM)
        exclusion_probe[:, :298] = probe_legacy.to(model.device)
        exclusion_zero = exclusion_probe.clone()
        exclusion_one = exclusion_probe.clone()
        exclusion_index = DUAL_WORKER_LAYOUT.p_skip_semantic_index
        exclusion_zero[:, exclusion_index] = 0.0
        exclusion_one[:, exclusion_index] = 1.0
        exclusion_legacy = extractor.policy_net(
            exclusion_probe[:, :298])
        excluded_preoutput_equal = torch.equal(
            adapter.preoutput(exclusion_zero, exclusion_legacy),
            adapter.preoutput(exclusion_one, exclusion_legacy),
        )
    _require(
        torch.equal(source_logits, target_logits)
        and excluded_preoutput_equal,
        "dual-v4 actor transplant: logits or the p_skip structural exclusion are not bit-identical",
    )
    probe_array = source_logits.contiguous().numpy()
    probe_digest = hashlib.sha256()
    probe_digest.update(str(probe_array.dtype).encode("ascii"))
    probe_digest.update(
        np.asarray(probe_array.shape, dtype=np.int64).tobytes())
    probe_digest.update(probe_array.tobytes())
    partition = strict_actor_critic_parameter_partition(
        model.policy, optimizer=model.policy.optimizer)
    context_names = {
        id(parameter):
            f"mlp_extractor.context_adapter.{name}"
        for name, parameter in adapter.named_parameters()
    }
    context_keys = tuple(context_names[id(parameter)]
                         for parameter in all_context_parameters)
    context_encoder_keys = tuple(
        context_names[id(parameter)]
        for parameter in parameter_groups["encoder"])
    context_interaction_keys = tuple(
        context_names[id(parameter)]
        for parameter in parameter_groups["interaction"])
    context_output_keys = tuple(
        context_names[id(parameter)]
        for parameter in parameter_groups["output"])
    receipt = {
        "schema": _ASYMMETRIC_ACTOR_INIT_SCHEMA,
        "method": _ASYMMETRIC_ACTOR_INIT_METHOD,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_actor_sha256": source_actor_sha256,
        "source_critic_sha256": source_critic_sha256,
        "source_policy_class": "MaskableActorCriticPolicy",
        "source_policy_observation_shape": [298],
        "migrated_actor_sha256": actor_parameter_sha256(
            model.policy, optimizer=model.policy.optimizer),
        "source_actor_parameter_tensors": len(_POLICY_HEAD_KEYS),
        "target_actor_parameter_tensors": len(partition["actor"]),
        "target_actor_parameter_count": sum(
            int(parameter.numel()) for parameter in partition["actor"]),
        "context_parameter_tensors": len(all_context_parameters),
        "context_parameter_count": context_parameter_count,
        "context_enabled": extractor.actor_context_enabled,
        "context_architecture": _ASYMMETRIC_CONTEXT_ARCHITECTURE,
        "context_initializer": ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        "controller_layout_schema": DUAL_WORKER_LAYOUT.schema,
        "controller_layout_sha256": DUAL_WORKER_LAYOUT_SHA256,
        "context_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_keys),
        "context_encoder_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_encoder_keys),
        "context_interaction_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_interaction_keys),
        "context_output_sha256": _stable_named_tensor_sha256(
            model.policy.state_dict(), context_output_keys),
        "context_hidden_nonzero": hidden_nonzero,
        "context_output_nonzero": output_nonzero,
        "context_excluded_preoutput_bitwise_equal":
            bool(excluded_preoutput_equal),
        "actor_context_excluded_observation_features": list(
            ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES),
        "bitwise_probe_rows": probe_rows,
        "bitwise_probe_sha256": probe_digest.hexdigest(),
    }
    model._actor_migration_receipt = dict(receipt)
    return receipt


def _reset_worker_critic(
        model, *, training_seed: int,
        source_checkpoint_sha256: str,
        source_actor_sha256: str | None = None,
        source_critic_sha256: str | None = None) -> dict:
    """Reinitialize only the value branch for the full-game Worker target.

    V28's critic learned on FARM-window pseudo terminals.  The current Worker
    bootstraps across those windows and terminates only with the underlying
    game, so retaining that value function is a target mismatch rather than a
    continuation.  Preserve the migrated actor bit-for-bit, initialize the
    independent structured critic with SB3's native orthogonal gains under an
    isolated deterministic RNG stream, and return a layout/capacity receipt.
    """
    import torch
    from leashed_ppo import (
        actor_parameter_sha256,
        critic_parameter_sha256,
        strict_actor_critic_parameter_partition,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    _require(
        isinstance(training_seed, int) and not isinstance(training_seed, bool)
        and 0 <= training_seed < 2**32,
        "fresh Worker critic requires an explicit uint32 training seed",
    )
    _require(_is_sha256(source_checkpoint_sha256),
             "fresh Worker critic is missing a trusted source checkpoint SHA256")
    _require(str(getattr(model, "device", "")) == "cpu",
             "fresh Worker critic migration currently allows CPU only, to keep initialisation reproducible")
    policy = model.policy
    partition = strict_actor_critic_parameter_partition(
        policy, optimizer=policy.optimizer)
    actor_before = actor_parameter_sha256(
        policy, optimizer=policy.optimizer)
    critic_before = critic_parameter_sha256(
        policy, optimizer=policy.optimizer)
    if source_actor_sha256 is None:
        source_actor_sha256 = actor_before
    if source_critic_sha256 is None:
        source_critic_sha256 = critic_before
    _require(_is_sha256(source_actor_sha256)
             and _is_sha256(source_critic_sha256),
             "fresh critic is missing the source actor/critic SHA")
    rng_before = torch.random.get_rng_state().clone()
    init_seed = int.from_bytes(
        hashlib.sha256(
            b"diablogym/full-game-worker-critic-v1\0"
            + str(training_seed).encode("ascii")
        ).digest()[:8],
        byteorder="big",
    ) & ((1 << 63) - 1)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(init_seed)
        policy.mlp_extractor.value_net.apply(
            functools.partial(policy.init_weights, gain=math.sqrt(2.0)))
        policy.value_net.apply(
            functools.partial(policy.init_weights, gain=1.0))
    _require(torch.equal(torch.random.get_rng_state(), rng_before),
             "fresh critic initialisation polluted the global Torch RNG")
    actor_after = actor_parameter_sha256(
        policy, optimizer=policy.optimizer)
    critic_after = critic_parameter_sha256(
        policy, optimizer=policy.optimizer)
    _require(actor_after == actor_before,
             "fresh critic initialisation rewrote the frozen V28 actor")
    _require(critic_after != critic_before,
             "fresh critic digest unchanged after initialisation")
    _require(all(
        bool(parameter.detach().isfinite().all().item())
        for parameter in partition["critic"]
    ), "fresh critic initialisation produced NaN/Inf")
    critic_parameter_count = sum(
        int(parameter.numel()) for parameter in partition["critic"])
    _require(
        len(partition["actor"]) > 0
        and len(partition["critic"]) > 0
        and critic_parameter_count > 0,
        "fresh structured critic capacity/tensor count drifted",
    )
    receipt = {
        "schema": _ASYMMETRIC_CRITIC_RESET_SCHEMA,
        "method": _ASYMMETRIC_CRITIC_RESET_METHOD,
        "critic_architecture": _ASYMMETRIC_CRITIC_ARCHITECTURE,
        "controller_layout_schema": DUAL_WORKER_LAYOUT.schema,
        "controller_layout_sha256": DUAL_WORKER_LAYOUT_SHA256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_actor_sha256": source_actor_sha256,
        "source_critic_sha256": source_critic_sha256,
        "training_seed": training_seed,
        "init_seed": init_seed,
        "actor_sha256_before": actor_before,
        "actor_sha256_after": actor_after,
        "critic_sha256_before": critic_before,
        "critic_sha256_after": critic_after,
        "actor_parameter_tensors": len(partition["actor"]),
        "critic_parameter_tensors": len(partition["critic"]),
        "critic_parameter_count": critic_parameter_count,
    }
    model._critic_reset_receipt = dict(receipt)
    return receipt


def _canonical_asymmetric_worker_migration_evidence(
        *, source_checkpoint_payload: bytes,
        source_checkpoint_sha256: str,
        training_seed: int) -> dict:
    """Reconstruct the canonical actor migration and critic reset receipts.

    R7 consumes this independently reconstructed evidence rather than trusting
    the initial sub-hashes embedded by the artifact under inspection.  The
    reconstruction uses no environment rollout and restores every global RNG
    stream touched by policy/source-model construction.
    """
    import numpy as np
    import torch
    from leashed_ppo import asymmetric_worker_runtime_evidence

    _require(
        isinstance(source_checkpoint_payload, bytes)
        and _is_sha256(source_checkpoint_sha256)
        and hashlib.sha256(source_checkpoint_payload).hexdigest()
        == source_checkpoint_sha256,
        "canonical asymmetric evidence source payload/SHA mismatch",
    )
    _require(
        _is_plain_int(training_seed) and 0 <= training_seed < 2**32,
        "canonical asymmetric evidence requires a uint32 training_seed",
    )
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state().clone()
    try:
        # Reproduce SB3's target-policy construction seed so even the
        # pre-reset critic digest is independently reconstructible.  The
        # migrated actor and reset critic that survive this helper remain
        # defined by their dedicated canonical paths.
        random.seed(training_seed)
        np.random.seed(training_seed)
        torch.manual_seed(training_seed)
        target = types.SimpleNamespace(
            policy=_new_current_asymmetric_worker_policy())
        target.device = target.policy.device
        actor_receipt = _initialize_asymmetric_worker_actor(
            target,
            source_checkpoint_payload=source_checkpoint_payload,
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
        critic_receipt = _reset_worker_critic(
            target,
            training_seed=training_seed,
            source_checkpoint_sha256=source_checkpoint_sha256,
            source_actor_sha256=actor_receipt["source_actor_sha256"],
            source_critic_sha256=actor_receipt["source_critic_sha256"],
        )
        runtime = asymmetric_worker_runtime_evidence(target.policy)
        _require(
            runtime["policy"]["actor_sha256"]
            == actor_receipt["migrated_actor_sha256"]
            and runtime["policy"]["critic_sha256"]
            == critic_receipt["critic_sha256_after"]
            and runtime["policy"]["actor_tensor_count"]
            == actor_receipt["target_actor_parameter_tensors"]
            and runtime["policy"]["actor_parameter_count"]
            == actor_receipt["target_actor_parameter_count"]
            and runtime["context"]["tensor_count"]
            == actor_receipt["context_parameter_tensors"]
            and runtime["context"]["parameter_count"]
            == actor_receipt["context_parameter_count"]
            and runtime["policy"]["critic_tensor_count"]
            == critic_receipt["critic_parameter_tensors"]
            and runtime["policy"]["critic_parameter_count"]
            == critic_receipt["critic_parameter_count"],
            "canonical asymmetric migration/reset/runtime evidence not closed",
        )
        return {
            "schema": _ASYMMETRIC_CANONICAL_EVIDENCE_SCHEMA,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "training_seed": training_seed,
            "actor_migration": copy.deepcopy(actor_receipt),
            "critic_reset": copy.deepcopy(critic_receipt),
            "runtime": copy.deepcopy(runtime),
        }
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)


def _policy_source_roles(args, demos_sha256: str | None) -> dict:
    """Describe which frozen artifacts can actually change policy parameters."""
    # Parameter provenance only: the independent warm-start receipt explicitly
    # denies optimizer/counter/trajectory continuation.
    resume = bool(getattr(args, "resume_from", None)
                  or getattr(args, "resource_warm_start", None))
    bc_init = bool(getattr(args, "bc_init", None))
    beta = float(getattr(args, "distill_beta", 0.0))
    teacher_override = bool(getattr(args, "teacher_override", None))
    worker = bool(getattr(args, "worker", False))
    dual = (
        worker
        and _worker_policy_observation_view(args)
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    )
    action14_bonus = float(getattr(
        args, "worker_action14_logit_bonus", 0.0))

    direct_bc_uses = []
    if bc_init:
        direct_bc_uses.append("policy-initialization")
    if worker and beta > 0.0 and not teacher_override:
        direct_bc_uses.append("distillation-teacher")

    if beta == 0.0:
        distillation_teacher = "disabled"
    elif teacher_override:
        distillation_teacher = "teacher-override"
    elif worker:
        distillation_teacher = "configured-bc-v1-export"
    else:
        distillation_teacher = "configured-teacher"

    action14_sources = []
    if worker:
        if bc_init:
            action14_sources.append("bc-v1-policy-initialization")
        if dual and action14_bonus != 0.0:
            action14_sources.append("fixed-logit-prior")
        action14_sources.append("native-reward-bound-on-policy-ppo")

    roles = {
        "schema": _POLICY_SOURCE_ROLES_SCHEMA,
        "initialization": (
            "resume-checkpoint"
            if resume else
            "bc-v1-policy-initialization"
            if bc_init else
            "random-initialization"
        ),
        "bc_v1_direct_policy_uses": direct_bc_uses,
        "bc_v1_dataset_uses": (
            ["pass-gate", "read-only-dry-anchor-instrumentation"]
            if demos_sha256 is not None else []
        ),
        "distillation_teacher": distillation_teacher,
        "worker_action14_policy_sources": action14_sources,
    }
    return roles


def _validate_policy_source_roles(contract: dict) -> dict:
    """Fail closed when provenance labels claim a policy path that is absent."""
    _require(isinstance(contract, dict),
             "policy source roles require a training contract object")
    roles = contract.get("policy_source_roles")
    _require(
        isinstance(roles, dict)
        and set(roles) == {
            "schema",
            "initialization",
            "bc_v1_direct_policy_uses",
            "bc_v1_dataset_uses",
            "distillation_teacher",
            "worker_action14_policy_sources",
        }
        and roles.get("schema") == _POLICY_SOURCE_ROLES_SCHEMA,
        "policy source roles schema/fields not exact",
    )
    initialization = roles["initialization"]
    direct = roles["bc_v1_direct_policy_uses"]
    dataset = roles["bc_v1_dataset_uses"]
    teacher = roles["distillation_teacher"]
    action14 = roles["worker_action14_policy_sources"]
    _require(
        initialization in {
            "resume-checkpoint",
            "bc-v1-policy-initialization",
            "random-initialization",
        }
        and isinstance(direct, list)
        and len(direct) == len(set(direct))
        and set(direct).issubset({
            "policy-initialization", "distillation-teacher"})
        and (
            ("policy-initialization" in direct)
            is (initialization == "bc-v1-policy-initialization")
        )
        and (
            contract.get("demos_sha256") is None
            or _is_sha256(contract.get("demos_sha256"))
        )
        and dataset == (
            ["pass-gate", "read-only-dry-anchor-instrumentation"]
            if contract.get("demos_sha256") is not None else []
        ),
        "BC-v1 policy/data role disagrees with the initialisation or demos binding",
    )
    beta = contract.get("distill_beta")
    _require(
        isinstance(beta, (int, float))
        and not isinstance(beta, bool)
        and math.isfinite(float(beta))
        and float(beta) >= 0.0,
        "policy source roles are missing a finite non-negative distill_beta",
    )
    if float(beta) == 0.0:
        _require(
            teacher == "disabled"
            and "distillation-teacher" not in direct,
            "β=0 but a distillation policy path is claimed",
        )
    else:
        _require(
            teacher in {
                "teacher-override",
                "configured-bc-v1-export",
                "configured-teacher",
            }
            and (
                ("distillation-teacher" in direct)
                is (teacher == "configured-bc-v1-export")
            ),
            "distillation teacher does not close with the BC-v1 direct role",
        )
        _require(
            _is_sha256(contract.get("teacher_sha256")),
            "distillation enabled but no valid teacher_sha256",
        )

    mode = contract.get("mode")
    bonus = contract.get("worker_action14_logit_bonus")
    _require(
        isinstance(action14, list)
        and len(action14) == len(set(action14)),
        "worker a14 policy sources must be a list without duplicates",
    )
    expected_action14 = []
    if mode == "worker":
        if initialization == "bc-v1-policy-initialization":
            expected_action14.append("bc-v1-policy-initialization")
        if (
            contract.get("worker_policy_observation_view")
            == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
            and isinstance(bonus, (int, float))
            and not isinstance(bonus, bool)
            and math.isfinite(float(bonus))
            and float(bonus) != 0.0
        ):
            expected_action14.append("fixed-logit-prior")
        expected_action14.append(
            "native-reward-bound-on-policy-ppo")
    _require(
        action14 == expected_action14,
        "worker a14 policy sources disagree with the real initialisation/prior/PPO path",
    )
    if (
        mode == "worker"
        and contract.get("worker_policy_observation_view")
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    ):
        _require(
            initialization == "resume-checkpoint"
            and isinstance(contract.get("actor_migration"), dict)
            and contract["actor_migration"].get("method")
            == _ASYMMETRIC_ACTOR_INIT_METHOD,
            "dual-v4 actor migration must be registered truthfully as resume-checkpoint",
        )
    return roles


def _resource_service_recipe_for(protocol, mode, service_policy,
                                 worker_time_protocol="legacy"):
    """R18-B3 (2026-09-07): the training-side recipe must state the completion clock actually running.

    Only the sustain-loot-v1 recipe carries a clock; the other service policies take the original call (byte-for-byte unchanged).
    sustain-loot-v1 with the legacy clock fails closed here -- the loot economy has no old-clock version.
    """
    if service_policy != "sustain-loot-v1":
        return resource_service_recipe(protocol, mode, service_policy)
    return resource_service_recipe(protocol, mode, service_policy,
                                   time_protocol=worker_time_protocol)


def _training_contract(args, model, batch_size: int,
                       manager_npz_sha256: str | None = None,
                       worker_npz_sha256: str | None = None,
                       worker_zip_sha256: str | None = None,
                       demos_sha256: str | None = None,
                       implementation_sha256: str | None = None,
                       bc_aux_demos_sha256: str | None = None) -> dict:
    mode = "worker" if args.worker else "options" if args.options else \
        "flat_clock" if args.flat_clock else "flat"
    manager_policy_view = (
        "legacy-v3" if args.worker
        else getattr(args, "manager_policy_observation_view", "raw-v4")
        if args.options
        else None
    )
    gradient_clip_mode = getattr(args, "gradient_clip_mode", "global")
    gradient_clipping = (
        _root_context_critic_gradient_clipping(model.max_grad_norm)
        if gradient_clip_mode == "separate-root-context-critic-v2"
        else
        {
            "mode": "separate-actor-critic-v1",
            "actor_max_norm": float(model.max_grad_norm),
            "critic_max_norm": float(model.max_grad_norm),
            "optimizer": "single",
            "trainable_shared_parameters": "forbidden",
        }
        if gradient_clip_mode == "separate-actor-critic-v1"
        else {
            "mode": "global",
            "max_norm": float(model.max_grad_norm),
        }
    )
    critic_receipt = getattr(
        model, "_critic_migration_receipt", None)
    actor_receipt = getattr(model, "_actor_migration_receipt", None)
    asymmetric_runtime = None
    if critic_receipt is not None or actor_receipt is not None:
        from leashed_ppo import asymmetric_worker_runtime_evidence
        asymmetric_runtime = asymmetric_worker_runtime_evidence(
            model.policy)
    if critic_receipt is None:
        critic_migration = "disabled"
    else:
        from leashed_ppo import (
            WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        )
        from diablogym.controller_wire import (
            DUAL_WORKER_LAYOUT,
            DUAL_WORKER_LAYOUT_SHA256,
        )

        _require(
            isinstance(critic_receipt, dict)
            and critic_receipt.get("schema")
            == _ASYMMETRIC_CRITIC_RESET_SCHEMA
            and critic_receipt.get("method")
            == _ASYMMETRIC_CRITIC_RESET_METHOD
            and critic_receipt.get("critic_architecture")
            == _ASYMMETRIC_CRITIC_ARCHITECTURE
            and critic_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and critic_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and asymmetric_runtime is not None
            and critic_receipt.get("actor_parameter_tensors")
            == asymmetric_runtime["policy"]["actor_tensor_count"]
            and critic_receipt.get("critic_parameter_tensors")
            == asymmetric_runtime["policy"]["critic_tensor_count"]
            and critic_receipt.get("critic_parameter_count")
            == asymmetric_runtime["policy"]["critic_parameter_count"]
            and _is_sha256(
                critic_receipt.get("source_checkpoint_sha256"))
            and isinstance(critic_receipt.get("warmup_steps"), int)
            and critic_receipt["warmup_steps"] > 0
            and critic_receipt.get("gradient_clip_mode")
            in {
                "separate-actor-critic-v1",
                "separate-root-context-critic-v2",
            }
            and _is_sha256(critic_receipt.get("source_actor_sha256"))
            and critic_receipt.get(
                "worker_onpolicy_pg_audit_schema")
            == WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            "model critic migration receipt invalid; refusing to write the training contract",
        )
        critic_migration = {
            "method": critic_receipt["method"],
            "critic_architecture":
                critic_receipt["critic_architecture"],
            "controller_layout_schema":
                critic_receipt["controller_layout_schema"],
            "controller_layout_sha256":
                critic_receipt["controller_layout_sha256"],
            "critic_parameter_tensors":
                critic_receipt["critic_parameter_tensors"],
            "critic_parameter_count":
                critic_receipt["critic_parameter_count"],
            "source_checkpoint_sha256":
                critic_receipt["source_checkpoint_sha256"],
            "warmup_steps": critic_receipt["warmup_steps"],
            "gradient_clip_mode":
                critic_receipt["gradient_clip_mode"],
            "source_actor_sha256":
                critic_receipt["source_actor_sha256"],
            "worker_onpolicy_pg_audit_schema":
                critic_receipt[
                    "worker_onpolicy_pg_audit_schema"],
            "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout":
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        }
    if actor_receipt is None:
        actor_migration = "disabled"
    else:
        from leashed_ppo import (
            ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES,
            ASYMMETRIC_WORKER_CONTEXT_INITIALIZER,
        )
        from diablogym.controller_wire import (
            DUAL_WORKER_LAYOUT,
            DUAL_WORKER_LAYOUT_SHA256,
        )
        _require(
            isinstance(actor_receipt, dict)
            and actor_receipt.get("schema")
            == _ASYMMETRIC_ACTOR_INIT_SCHEMA
            and actor_receipt.get("method")
            == _ASYMMETRIC_ACTOR_INIT_METHOD
            and _is_sha256(
                actor_receipt.get("source_checkpoint_sha256"))
            and _is_sha256(actor_receipt.get("source_actor_sha256"))
            and _is_sha256(actor_receipt.get("migrated_actor_sha256"))
            and _is_sha256(actor_receipt.get("context_sha256"))
            and _is_sha256(
                actor_receipt.get("context_encoder_sha256"))
            and _is_sha256(
                actor_receipt.get("context_interaction_sha256"))
            and _is_sha256(
                actor_receipt.get("context_output_sha256"))
            and actor_receipt.get("source_policy_class")
            == "MaskableActorCriticPolicy"
            and actor_receipt.get("source_policy_observation_shape")
            == [298]
            and actor_receipt.get("source_actor_parameter_tensors") == 6
            and asymmetric_runtime is not None
            and actor_receipt.get("target_actor_parameter_tensors")
            == asymmetric_runtime["policy"]["actor_tensor_count"]
            and actor_receipt.get("target_actor_parameter_count")
            == asymmetric_runtime["policy"]["actor_parameter_count"]
            and actor_receipt.get("context_parameter_tensors")
            == asymmetric_runtime["context"]["tensor_count"]
            and actor_receipt.get("context_parameter_count")
            == asymmetric_runtime["context"]["parameter_count"]
            and actor_receipt.get("context_enabled") is False
            and actor_receipt.get("context_architecture")
            == _ASYMMETRIC_CONTEXT_ARCHITECTURE
            and actor_receipt.get("context_initializer")
            == ASYMMETRIC_WORKER_CONTEXT_INITIALIZER
            and actor_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and actor_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and _is_plain_int(
                actor_receipt.get("context_hidden_nonzero"))
            and actor_receipt["context_hidden_nonzero"] > 0
            and actor_receipt.get("context_output_nonzero") == 0
            and actor_receipt.get(
                "context_excluded_preoutput_bitwise_equal") is True
            and actor_receipt.get(
                "actor_context_excluded_observation_features")
            == list(ASYMMETRIC_WORKER_ACTOR_EXCLUDED_FEATURES)
            and actor_receipt.get("bitwise_probe_rows") == 4
            and _is_sha256(
                actor_receipt.get("bitwise_probe_sha256")),
            "model asymmetric actor migration receipt invalid; refusing to write the training contract",
        )
        actor_migration = {
            "method": actor_receipt["method"],
            "source_checkpoint_sha256":
                actor_receipt["source_checkpoint_sha256"],
            "source_actor_sha256":
                actor_receipt["source_actor_sha256"],
            "migrated_actor_sha256":
                actor_receipt["migrated_actor_sha256"],
            "context_architecture":
                actor_receipt["context_architecture"],
            "controller_layout_schema":
                actor_receipt["controller_layout_schema"],
            "controller_layout_sha256":
                actor_receipt["controller_layout_sha256"],
            "target_actor_parameter_tensors":
                actor_receipt["target_actor_parameter_tensors"],
            "target_actor_parameter_count":
                actor_receipt["target_actor_parameter_count"],
            "context_parameter_tensors":
                actor_receipt["context_parameter_tensors"],
            "context_parameter_count":
                actor_receipt["context_parameter_count"],
            "context_initialization": {
                "hidden": actor_receipt["context_initializer"],
                "output": "exact-zero-disabled-through-critic-warmup",
            },
            "actor_context_excluded_observation_features":
                actor_receipt[
                    "actor_context_excluded_observation_features"],
        }
    action_count = getattr(model.action_space, "n", None)
    from leashed_ppo import LEGACY_DISTILLATION_EXCLUDED_ACTIONS
    contract = {
        "schema_version": 2,
        "contract_revision": _CONTRACT_REVISION,   # v32: +drink_sovereignty (4C environment semantics enter the contract)
        "implementation_sha256": implementation_sha256,
        # R12: the economy rule enters the contract (v1 = historical default; later continuations can see the wage-system identity)
        "reward_economy": getattr(args, "reward_economy", "v1"),
        # R12 amendment 2 (docs/rounds/r12-LAUNCH-RECORD-20260830.md): the scripted-coach identity enters the contract (None = npz manager)
        "manager_heuristic": getattr(args, "manager_heuristic", None),
        "mode": mode,
        "arch": args.arch,
        "max_steps": args.max_steps,
        "num_envs": args.num_envs,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "gamma": args.gamma,
        "learning_rate": args.lr,
        "ent_coef": args.ent_coef,
        "distill_beta": float(args.distill_beta),
        "distillation": {
            "initial_beta": float(args.distill_beta),
            "scope": (
                "legacy-root-logits"
                if (
                    args.worker
                    and _worker_policy_observation_view(args)
                    == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
                    and float(args.distill_beta) > 0.0
                )
                else "full-policy-logits"
            ),
            "excluded_actions": (
                list(LEGACY_DISTILLATION_EXCLUDED_ACTIONS)
                if float(args.distill_beta) > 0.0 else []
            ),
            "anneal_actor_rollouts": int(getattr(
                args, "distill_anneal_actor_rollouts", 0)),
            "schedule": (
                "linear-inclusive-zero-v1"
                if int(getattr(
                    args,
                    "distill_anneal_actor_rollouts",
                    0)) > 0
                else "constant"
            ),
        },
        "teacher_sha256": getattr(model, "teacher_sha256", None),
        "calib_record_only": bool(args.calib_record_only),
        "device": str(model.device),
        "skip_dry": bool(args.skip_dry),
        "drink_sovereignty": _effective_drink_sovereignty(args),
        "worker_fast_forward_reward_credit": getattr(
            args, "worker_fast_forward_reward_credit", "none"),
        "worker_additional_terminal_death_cost":
            float(getattr(
                args, "worker_additional_terminal_death_cost", 0.0)),
        # R13: the classroom scope enters the contract. Not enabled (farm-only) always writes None; old checkpoint
        # contracts .get None too -> resume equality compatible, revision 26 unchanged
        # (worker_zip_sha256 precedent); enabled drift goes through the whitelist (v0.3 revision).
        "worker_learning_window_scope": (
            getattr(args, "worker_learning_window_scope", "farm-only")
            if (getattr(args, "worker", False)
                and getattr(
                    args, "worker_learning_window_scope", "farm-only")
                != "farm-only")
            else None),
        # R13 cold-start lever, same booking: 0.0 (not enabled) always writes None.
        "worker_dive_action11_logit_bonus": (
            float(getattr(
                args, "worker_dive_action11_logit_bonus", 0.0))
            if (getattr(args, "worker", False)
                and float(getattr(
                    args, "worker_dive_action11_logit_bonus", 0.0))
                > 0.0)
            else None),
        # R16 potion-pickup prior, same booking: 0.0 (not enabled) always writes None.
        "worker_potion_action13_logit_bonus": (
            float(getattr(
                args, "worker_potion_action13_logit_bonus", 0.0))
            if (getattr(args, "worker", False)
                and float(getattr(
                    args, "worker_potion_action13_logit_bonus", 0.0))
                > 0.0)
            else None),
        # R16 amendment: one None-off key each for the new environment/classroom/wage-side parameters -- CLI defaults
        # (False/0/1800/death-equivalent) always write None, old checkpoint contracts .get
        # None too -> resume equality compatible, revision 26 unchanged (worker_zip_sha256
        # precedent); enabled drift goes through the _ENVIRONMENT_RESTART_ALLOWED_DRIFT whitelist.
        "explore_global_fallback": (
            True
            if bool(getattr(args, "explore_global_fallback", False))
            else None),
        # R16 C5 main lever (later addition, same booking as fallback): False (not enabled) always writes None.
        "explore_global_hunt": (
            True
            if bool(getattr(args, "explore_global_hunt", False))
            else None),
        # R18-B5 (2026-09-07): scope of the a10 global monster hunt (all = frozen behaviour, always written as
        # None; only l1-only is a literal), same booking as explore_global_hunt.
        "hunt_scope": (
            getattr(args, "hunt_scope", "all")
            if getattr(args, "hunt_scope", "all") != "all" else None),
        "progress_far_tiles": (
            int(getattr(args, "progress_far_tiles", 0))
            if int(getattr(args, "progress_far_tiles", 0)) > 0
            else None),
        "farm_scene_cap": (
            int(getattr(args, "farm_scene_cap", _FARM_SCENE_CAP_DEFAULT))
            if int(getattr(args, "farm_scene_cap", _FARM_SCENE_CAP_DEFAULT))
            != _FARM_SCENE_CAP_DEFAULT
            else None),
        "reset_layer_clock_on_window": (
            True
            if bool(getattr(args, "reset_layer_clock_on_window", False))
            else None),
        "resource_protocol": (
            getattr(args, "resource_protocol", "off")
            if getattr(args, "resource_protocol", "off") != "off" else None),
        "resource_purchase_mode": (
            getattr(args, "resource_purchase_mode", "full")
            if getattr(args, "resource_protocol", "off") != "off" else None),
        "resource_service_policy": (
            getattr(args, "resource_service_policy", "legacy-v1")
            if getattr(args, "resource_service_policy", "legacy-v1") != "legacy-v1" else None),
        # R18-B3: the loot recipe carries a clock; non-loot service-policy calls stay byte-for-byte unchanged.
        "resource_service_recipe": _resource_service_recipe_for(
            getattr(args, "resource_protocol", "off"),
            getattr(args, "resource_purchase_mode", "full"),
            getattr(args, "resource_service_policy", "legacy-v1"),
            getattr(args, "worker_time_protocol", "legacy")),
        # R17.1 ruling 3: veto-v1 (the legacy law) always writes None; only
        # coach-v03 is literal, so old checkpoint contracts stay equal.
        "resource_readiness_law": (
            getattr(args, "resource_readiness_law", "veto-v1")
            if getattr(args, "resource_readiness_law", "veto-v1") != "veto-v1" else None),
        # R18-B: off (the legacy default) always writes None; only retreat-v1
        # is literal, so old checkpoint contracts stay equal.
        "resource_retreat": (
            getattr(args, "resource_retreat", "off")
            if getattr(args, "resource_retreat", "off") != "off" else None),
        # R18-B5 (2026-09-07): off (the legacy default) always writes None;
        # only portal-v1 is literal, so old checkpoint contracts stay equal.
        "resource_portal": (
            getattr(args, "resource_portal", "off")
            if getattr(args, "resource_portal", "off") != "off" else None),
        # R18-B6 (2026-09-07): off (the legacy default) always writes None; only
        # the versioned law is literal, so old checkpoint contracts stay equal.
        "resource_sweep": (
            getattr(args, "resource_sweep", "off")
            if getattr(args, "resource_sweep", "off") != "off" else None),
        "resource_identify": (
            getattr(args, "resource_identify", "off")
            if getattr(args, "resource_identify", "off") != "off" else None),
        "resource_weapon_upgrade": (
            getattr(args, "resource_weapon_upgrade", "off")
            if getattr(args, "resource_weapon_upgrade", "off") != "off" else None),
        "worker_hp_loss_price": (
            float(getattr(args, "worker_hp_loss_price", 0.0))
            if (getattr(args, "worker", False)
                and float(getattr(args, "worker_hp_loss_price", 0.0))
                > 0.0)
            else None),
        "worker_potion_pickup_bonus": (
            float(getattr(args, "worker_potion_pickup_bonus", 0.0))
            if (getattr(args, "worker", False)
                and float(getattr(
                    args, "worker_potion_pickup_bonus", 0.0)) > 0.0)
            else None),
        "worker_no_progress_timeout_credit": (
            str(getattr(
                args, "worker_no_progress_timeout_credit",
                _WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT))
            if (getattr(args, "worker", False)
                and getattr(
                    args, "worker_no_progress_timeout_credit",
                    _WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT)
                != _WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT)
            else None),
        # R13.2 plan A, same booking: 0.0 (not enabled) always writes None.
        "worker_depth_shaping_unit": (
            float(getattr(args, "worker_depth_shaping_unit", 0.0))
            if (getattr(args, "worker", False)
                and float(getattr(
                    args, "worker_depth_shaping_unit", 0.0)) > 0.0)
            else None),
        # R14 plan B, same booking: 0.0 (not enabled) always writes None.
        "worker_descend_bonus_fraction": (
            float(getattr(
                args, "worker_descend_bonus_fraction", 0.0))
            if (getattr(args, "worker", False)
                and float(getattr(
                    args, "worker_descend_bonus_fraction", 0.0)) > 0.0)
            else None),
        # R14.2 plan D, same booking: 0.0 (not enabled) always writes None.
        "worker_descend_escrow_fraction": (
            float(getattr(
                args, "worker_descend_escrow_fraction", 0.0))
            if (getattr(args, "worker", False)
                and float(getattr(
                    args, "worker_descend_escrow_fraction", 0.0)) > 0.0)
            else None),
        # R15 amendment 2: False (not enabled) always writes None.
        "worker_descend_escrow_readiness_gate": (
            True
            if (getattr(args, "worker", False)
                and bool(getattr(
                    args, "worker_descend_escrow_readiness_gate",
                    False)))
            else None),
        # R17.0 amendment 1: v1 (old ruler, not enabled) always writes None; only v2 writes a literal.
        "worker_descend_escrow_readiness_table": (
            "v2"
            if (getattr(args, "worker", False)
                and str(getattr(
                    args, "worker_descend_escrow_readiness_table",
                    "v1")) == "v2")
            else None),
        # R14.3 plan E: power=1.0 (the old linear rule) always writes None.
        "worker_descend_escrow_power": (
            float(getattr(
                args, "worker_descend_escrow_power", 1.0))
            if (getattr(args, "worker", False)
                and float(getattr(
                    args, "worker_descend_escrow_power", 1.0)) != 1.0)
            else None),
        "legacy_policy_observation_view": (
            _worker_policy_observation_view(args)
            == _WORKER_VIEW_LEGACY_V3),
        "worker_policy_observation_view":
            _worker_policy_observation_view(args),
        "worker_action14_logit_bonus": float(getattr(
            args, "worker_action14_logit_bonus", 0.0)),
        "manager_policy_observation_view": manager_policy_view,
        "worker_episode_boundary": (
            _WORKER_EPISODE_BOUNDARY_V24 if args.worker else None),
        "worker_window_bootstrap": (
            "next-learning-window" if args.worker else None),
        "worker_no_progress_timeout": (
            dict(_WORKER_NO_PROGRESS_TIMEOUT_CONTRACT)
            if args.worker else None
        ),
        "gradient_clipping": gradient_clipping,
        "actor_migration": actor_migration,
        "critic_migration": critic_migration,
        "artifact_scope": getattr(args, "artifact_scope", "production"),
        # E4 rev5 dual keys (review round 7, unified across the three legs): disabled or the live payload; the skip_dry key
        # keeps the CLI flag literal, unaffected by these two keys (rev3 correction; contract and receipt are isomorphic).
        "dry_curriculum": _contract_dry_curriculum(args),
        "bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),
        "manager_npz_sha256": manager_npz_sha256,
        "worker_npz_sha256": worker_npz_sha256,
        # R9: identity of the zip worker assembly point (certified release artifact); always None when no zip worker is mounted,
        # old checkpoint contracts .get None too -> resume equality compatible, revision unchanged.
        "worker_zip_sha256": worker_zip_sha256,
        "demos_sha256": demos_sha256,
        "policy_source_roles":
            _policy_source_roles(args, demos_sha256),
        # Gymnasium exposes Discrete.n (and on some versions shape entries) as
        # NumPy integer scalars.  Normalize before this contract is embedded in
        # status.json/SB3 data; stdlib json intentionally cannot encode np.int64.
        "observation_shape": [int(value) for value in model.observation_space.shape],
        "action_n": None if action_count is None else int(action_count),
        "runtime_versions": dict(_RUNTIME_VERSIONS),
        "algorithm_recipe": {
            **_ALGORITHM_RECIPE,
            "target_kl": getattr(args, "target_kl", None),
        },
    }
    prefix_identity = _worker_prefix_identity(args)
    if prefix_identity is not None:
        contract["worker_prefix"] = prefix_identity
    time_identity = _worker_time_identity(
        getattr(args, "worker_time_protocol", "legacy"))
    if time_identity is not None:
        _validate_worker_time_args(args)
        contract["worker_time_protocol"] = time_identity["protocol"]
        contract["worker_time_recipe"] = time_identity
    recovery = getattr(args, "dive_blocker_recovery", "off")
    validate_dive_blocker_recovery(getattr(args, "resource_protocol", "off"), recovery)
    if recovery != "off":
        contract["dive_blocker_recovery"] = recovery
    _validate_policy_source_roles(contract)
    if (
        args.worker
        and _worker_policy_observation_view(args)
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    ):
        _validate_current_dual_worker_contract(
            contract, runtime_evidence=asymmetric_runtime)
    return contract


# R12 rule change (2026-08-30, implementing the design decision to take route A): whitelist exemption for
# environment-restart continuation. Re-education = the same worker keeps training in "the world after sanctioned changes"; drift in the
# following fields is itself the rule change named in the pre-registration and is not a contract violation; outside the whitelist
# (view/action space/geometry/algorithm recipe etc.) strict equality still holds.
_ENVIRONMENT_RESTART_ALLOWED_DRIFT = frozenset({
    "demos_sha256", "distill_beta", "distillation", "dry_curriculum",
    "implementation_sha256", "policy_source_roles", "teacher_sha256",
    "worker_additional_terminal_death_cost", "reward_economy",
    # R13 rule change (2026-08-30, implementing the classroom-reform design decision): the classroom scope is
    # itself the rule change named in the pre-registration (None->farm-dive-v1), same as reward_economy.
    "worker_learning_window_scope",
    # R13 cold-start lever (pre-authorised as decision point 5, enabled if G0 acceptance falls short):
    # None->bonus drift is a rule change named in the pre-registration.
    "worker_dive_action11_logit_bonus",
    # R13.2 plan A (design decision, 2026-08-31): potential-based depth shaping,
    # None->unit drift is a rule change named in the pre-registration.
    "worker_depth_shaping_unit",
    # R14 plan B (design decision, 2026-08-31): bounded descend-bonus refund,
    # None->fraction drift is a rule change named in the pre-registration.
    "worker_descend_bonus_fraction",
    # R14.2 plan D (design decision: if plan B fails, go straight to plan D): escrow payout,
    # None->fraction drift is a rule change named in the pre-registration.
    "worker_descend_escrow_fraction",
    # R14.3 plan E (design decision: find a usable payout curve): convex-curve pricing,
    # None->power drift is a rule change named in the pre-registration.
    "worker_descend_escrow_power",
    # R15 amendment 2: readiness-conditioned escrow; None->True drift is a rule change named in the pre-registration.
    "worker_descend_escrow_readiness_gate",
    # R17.0 amendment 1 (panel correction 2.1): escrow readiness-gate ruler selector,
    # None->v2 drift is a rule change named in the pre-registration.
    "worker_descend_escrow_readiness_table",
    # R17.1 decision 3: readiness law (six-condition coach law, HP excluded); None->coach-v03
    # drift is a rule change named in the pre-registration (same as readiness_table).
    "resource_readiness_law",
    # R18-B: retreat law (retreat-v1 return to town); None->retreat-v1 drift is a rule change named
    # in the pre-registration (same as readiness_law).
    "resource_retreat",
    # R18-B5 (2026-09-07): portal law (portal-v1 retreat vehicle); None->portal-v1
    # drift is a rule change named in the pre-registration (same as retreat).
    "resource_portal",
    # R18-B6 (2026-09-07): chest-sweep law (sweep-v1), identify leg (cain-v1),
    # weapon-upgrade leg (smith-v1); None->literal drift is a rule change named in the pre-registration
    # (same as retreat/portal); all three only hold in the loot economy.
    "resource_sweep",
    "resource_identify",
    "resource_weapon_upgrade",
    # R16 amendment (2026-09-01, named in the pre-registration): a13 potion-pickup cold-start prior,
    # None->bonus drift is a rule change named in the pre-registration (same as a11).
    "worker_potion_action13_logit_bonus",
    # R16 amendment: environment-side (explore global fallback / full-map monster hunt / far-cell progress count) and classroom-side
    # (farm-window scene cap / level-clock reset inside the window) semantic changes; None->value drift is a rule change named
    # in the pre-registration. explore_global_hunt = R16 C5 main lever (8-seed probe:
    # fallback 375->375 kills, almost never triggers; hunt 375->670 kills), same as fallback.
    "explore_global_fallback",
    "explore_global_hunt",
    # R18-B5 (2026-09-07): a10 hunt scope; None->l1-only drift is a rule change named
    # in the pre-registration (same as explore_global_hunt).
    "hunt_scope",
    "progress_far_tiles",
    "farm_scene_cap",
    "reset_layer_clock_on_window",
    # R16 amendment: wage side (HP-loss pricing / potion-pickup bonus / stall-timeout booking definition);
    # None->value drift is a rule change named in the pre-registration (same as
    # worker_additional_terminal_death_cost).
    "worker_hp_loss_price",
    "worker_potion_pickup_bonus",
    "worker_no_progress_timeout_credit",
    # R16 amendment: the training episode length (max_steps 3000->>=6000) and opening potion autonomy
    # (drink_sovereignty) are both rule-change drift named in the pre-registration; outside the whitelist
    # view/action space/geometry/algorithm recipe still require strict equality.
    "max_steps",
    "drink_sovereignty",
})


def _validate_worker_time_resume_identity(saved: dict | None, current: dict) -> None:
    """Physical time contracts never inherit broad environment-drift waivers."""
    identities = []
    for label, identity in (("saved", saved), ("current", current)):
        record = identity if isinstance(identity, dict) else {}
        protocol = record.get("worker_time_protocol")
        expected = _worker_time_identity("legacy" if protocol is None else protocol)
        actual = record.get("worker_time_recipe")
        if expected is None:
            _require(actual is None, f"{label} legacy time protocol cannot carry a recipe")
        else:
            _require(isinstance(identity, dict), f"{label} completion time contract is missing")
            _require(isinstance(actual, dict) and actual == expected
                     and all(type(actual[key]) is type(value)
                             for key, value in expected.items()),
                     f"{label} worker_time_recipe is not the exact {protocol} recipe")
        identities.append(expected)
    _require(identities[0] == identities[1],
             "worker time protocol changes require separately identified initialization; ordinary resume is forbidden")


def _validate_worker_prefix_resume_identity(saved: dict | None, current: dict) -> None:
    """Prefix scope/caps/source are never covered by broad drift allowances."""
    previous = saved if isinstance(saved, dict) else {}
    scopes = (previous.get("worker_learning_window_scope"),
              current.get("worker_learning_window_scope"))
    if EARNED_DIVE_SUFFIX_SCOPE not in scopes:
        _require(previous.get("worker_prefix") is None and current.get("worker_prefix") is None,
                 "worker_prefix metadata requires earned-dive-suffix-v1")
        return
    _require(isinstance(saved, dict), "earned suffix resume requires an exact contract")
    for label, identity in (("saved", previous), ("current", current)):
        if identity.get("worker_learning_window_scope") == EARNED_DIVE_SUFFIX_SCOPE:
            recipe = identity.get("worker_prefix")
            _require(isinstance(recipe, dict), f"{label} earned suffix lacks worker_prefix")
            expected = worker_prefix_recipe(EARNED_DIVE_SUFFIX_SCOPE,
                recipe.get("source_sha256"), recipe.get("max_attempts"), recipe.get("max_microsteps"))
            _require(recipe == expected, f"{label} worker_prefix recipe is not exact")
    _require(scopes[0] == scopes[1] and previous.get("worker_prefix") == current.get("worker_prefix"),
             "earned suffix scope/source/budget changes require separately identified initialization; ordinary resume is forbidden")


def _validate_resource_resume_identity(saved: dict | None, current: dict) -> None:
    """Native armor/preservation versions cannot bypass exact resume identity."""
    saved_policy = saved.get("resource_service_policy") if isinstance(saved, dict) else None
    protected_policies = ("sustain-v5", "sustain-v6", "sustain-loot-v1")
    if not any(policy in protected_policies
               for policy in (saved_policy, current.get("resource_service_policy"))):
        return
    _require(isinstance(saved, dict),
             "sustain-v5/sustain-v6 resume requires an exact resource_service contract; legacy bypass is unavailable")
    for label, identity in (("saved", saved), ("current", current)):
        if identity.get("resource_service_policy") in protected_policies:
            expected = _resource_service_recipe_for(identity.get("resource_protocol"),
                identity.get("resource_purchase_mode"), identity["resource_service_policy"],
                identity.get("worker_time_protocol") or "legacy")
            _require(identity.get("resource_service_recipe") == expected,
                     f"{label} resource_service_recipe has mismatched native armor/preservation scope/version")
    # R18-B: the retreat law joins the identity key group; with the flag off both sides are None, old continuations unchanged bit for bit.
    # R18-B5 (2026-09-07): the portal law is also a resource-side rule (portal-v1 only holds under
    # l2-town-v1/coach-v03/retreat-v1) and joins the identity key group too;
    # hunt_scope is not in this list -- it is a DiabloGymEnv-level exploration switch, not part of the
    # resource_service identity; its drift is governed by _ENVIRONMENT_RESTART_ALLOWED_DRIFT.
    # R18-B6 (2026-09-07): the sweep/identify/weapon-upgrade laws are all versions of the sustain-loot-v1
    # trip itself (validate_sweep_protocol / validate_identify_protocol /
    # validate_weapon_upgrade all require l2-town-v1 + sustain-loot-v1); changing the law changes the
    # world, so they sit in the identity key group with retreat/portal; hunt_scope is still not in this list.
    keys = ("resource_protocol", "resource_purchase_mode",
            "resource_service_policy", "resource_service_recipe",
            "resource_retreat", "resource_portal",
            "resource_sweep", "resource_identify", "resource_weapon_upgrade")
    _require(all(saved.get(key) == current.get(key) for key in keys),
             "resource_service version change requires a separately identified initialization; ordinary resume is forbidden")


def _validate_resume_contract(saved: dict | None, current: dict,
                              allow_manager_change: bool = False,
                              allow_legacy_resume: bool = False,
                              allow_optimizer_reset: bool = False,
                              allow_target_kl_change: bool = False,
                               allow_environment_restart: bool = False) -> None:
    _validate_worker_time_resume_identity(saved, current)
    _validate_worker_prefix_resume_identity(saved, current)
    _validate_resource_resume_identity(saved, current)
    if saved is None:
        _require(allow_legacy_resume,
                 "resume checkpoint has no training_contract; cannot prove the original training environment/resources; "
                 "if a one-off migration is really needed, pass --allow-legacy-resume explicitly")
        print("   [legacy migration] a contract-less checkpoint was explicitly allowed; "
              f"this leg will write a contract_revision {_CONTRACT_REVISION} contract")
        return
    _require(isinstance(saved, dict), "checkpoint training_contract is not an object")
    allowed = ({"manager_npz_sha256", "manager_heuristic"}
               if allow_manager_change else set())
    if allow_environment_restart:
        allowed |= _ENVIRONMENT_RESTART_ALLOWED_DRIFT
        drifted = sorted(
            k for k in _ENVIRONMENT_RESTART_ALLOWED_DRIFT
            if saved.get(k) != current.get(k))
        if drifted:
            print("   [environment-restart resume] whitelist-exempt drifted fields: "
                  + ", ".join(drifted))
    if allow_optimizer_reset:
        # reset explicitly cuts off the old Adam moments; the learning rate and the reset receipt therefore belong to this leg's
        # new optimizer identity, and the previous leg's contract must not block this explicit migration. The reset itself is
        # an event of this leg; it goes only into config/receipt, not into the persisted resume equality.
        allowed.add("learning_rate")
    saved_recipe = saved.get("algorithm_recipe")
    current_recipe = current.get("algorithm_recipe")
    if allow_target_kl_change and isinstance(saved_recipe, dict) \
            and isinstance(current_recipe, dict):
        saved_without = {**saved_recipe, "target_kl": current_recipe.get("target_kl")}
        if saved_without == current_recipe:
            allowed.add("algorithm_recipe")
    differences = {key: (saved.get(key), current.get(key))
                   for key in sorted(set(saved) | set(current))
                   if key not in allowed and saved.get(key) != current.get(key)}
    _require(not differences, f"resume training/environment contract drift: {differences}")


# ---- E1 5A dry-window course (docs/prereg/PREREG-v33-content-case.md E1, rev3 certified correction) ----

# Leg-relative anchoring constants: the leg start is always = the end point of the throne zip, 3,497,984 (P3 relaunches always start from the throne zip);
# global-step anchoring is disabled -- p-table index = (num_timesteps − 3,497,984) / 2048.
_DRY_CURRICULUM_LEG_START = 3_497_984
# Main table (fixed at approval; review round 2 annex ruling): the first 147×2048=301,056 steps linear 1.0->0.5,
# the next 97×2048=198,656 steps hold 0.5; 147+97=244 quanta, exactly the leg length 499,712.
_DRY_CURRICULUM_MAIN_TABLE = "linear:1.0:0.5:147,hold:0.5:97"


def _dry_window_mechanism_active(args) -> bool:
    """E1 blast-radius predicate (rev3 correction, the four gates unified): dry-window mechanism present = skip_dry or schedule."""
    return bool(args.skip_dry) or bool(args.dry_curriculum_schedule)


def _mount_dry_anchor_sentinel(args) -> bool:
    """E1 four gates: dry_cb mount gate (original :2101-2104 predicate): worker and dry-window mechanism present."""
    return bool(args.worker) and _dry_window_mechanism_active(args)


def _precheck_dry_window_demos(args) -> None:
    """E1 four gates: demos/BC preflight gate (original :499-505): predicate rewritten to mechanism-present, assertion unchanged."""
    if not _dry_window_mechanism_active(args):
        return
    demos = pathlib.Path(__file__).resolve().parent / "runs" / "bc-worker" / "demos.npz"
    _require(demos.is_file(),
             f"demo set required by the dry-window mechanism (--skip-dry/--dry-curriculum-schedule) does not exist: {demos}")
    # v4: the probe set is bound to the current strict PASS receipt; old v3 protocol/implementation
    # must be refused even if its bytes still equal the historical constant.
    _assert_bc_v1_demos_frozen(demos)
    policy = demos.with_name("policy_sd.pt")
    _require(policy.is_file(),
             f"BC weights required by the dry-window mechanism (--skip-dry/--dry-curriculum-schedule) do not exist: {policy}")
    report = _validate_bc_report(policy, "data_gate")
    _load_dry_anchor_demos(demos, report.get("demos_sha256"))


def _capture_dry_window_demos_sha256(args) -> str | None:
    """E1 four gates: demos_sha256 capture gate (original :1766-1771): predicate rewritten, path and assertion unchanged."""
    if not _dry_window_mechanism_active(args):
        return None
    demos = pathlib.Path(__file__).resolve().parent / "runs" / "bc-worker" / "demos.npz"
    # v4: capture from the current strict PASS receipt; the historical v3 frozen constant is not trusted.
    _assert_bc_v1_demos_frozen(demos)
    report = _validate_bc_report(demos.with_name("policy_sd.pt"), "data_gate")
    _, _, demos_sha256 = _load_dry_anchor_demos(demos, report.get("demos_sha256"))
    return demos_sha256


def _parse_dry_curriculum_schedule(spec: str) -> tuple[float, ...]:
    """Parse --dry-curriculum-schedule into the full per-rollout "index->p" table.

    Syntax (comma-separated segments, colon-separated fields):
      linear:<p0>:<p1>:<n> -- n (>=2) rollout endpoints, linear p0->p1 inclusive,
                              item k = p0 + (p1−p0)·k/(n−1), k=0..n−1;
      hold:<p>:<n>         -- n (>=1) rollouts at constant p.
    Every p must be a finite number in [0, 1]. Main table = linear:1.0:0.5:147,hold:0.5:97.
    """
    _require(isinstance(spec, str) and bool(spec.strip()),
             "--dry-curriculum-schedule must not be empty")
    table: list[float] = []
    for raw_segment in spec.split(","):
        segment = raw_segment.strip()
        fields = segment.split(":")
        if fields[0] == "linear":
            _require(len(fields) == 4,
                     f"--dry-curriculum-schedule segment format must be linear:<p0>:<p1>:<n>: {segment!r}")
            try:
                p0, p1, n = float(fields[1]), float(fields[2]), int(fields[3])
            except ValueError as exc:
                raise ValueError(
                    f"--dry-curriculum-schedule segment values unparseable: {segment!r}") from exc
            _require(n >= 2, f"linear segment needs n>=2 (use hold for a single point): {segment!r}")
            values = [p0 + (p1 - p0) * k / (n - 1) for k in range(n)]
        elif fields[0] == "hold":
            _require(len(fields) == 3,
                     f"--dry-curriculum-schedule segment format must be hold:<p>:<n>: {segment!r}")
            try:
                p, n = float(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"--dry-curriculum-schedule segment values unparseable: {segment!r}") from exc
            _require(n >= 1, f"hold segment needs n>=1: {segment!r}")
            values = [p] * n
        else:
            raise ValueError(
                f"--dry-curriculum-schedule unknown segment type (only linear/hold allowed): {segment!r}")
        _require(all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values),
                 f"--dry-curriculum-schedule p values must lie in [0, 1]: {segment!r}")
        table.extend(values)
    return tuple(table)


# ---- R9 deep-start curriculum (PREREG-R9, curriculum arm prologue) ----

# Cap on reseeded redraws when the episode ends (death/truncation) during the prologue; past the cap the scripted play-through is abandoned,
# and a new seed hands over from an ordinary start (telemetry resamples is recorded truthfully in info).
_DEEP_START_MAX_RESAMPLES = 8


def _parse_deep_start_curriculum(spec: str) -> dict:
    """Parse --deep-start-curriculum 'p=0.5,target=2,cap=8' into a dict.

    p in [0,1] is the per-episode independent trigger probability (deterministic RNG derived from the episode seed, never touching the global RNG);
    target >= 2 is the target dungeon_level of the prologue play-through (the start is always level 1);
    cap >= 1 caps the number of play-through windows. All three keys are required; unknown/duplicate keys are not allowed.
    """
    _require(isinstance(spec, str) and bool(spec.strip()),
             "--deep-start-curriculum must not be empty")
    fields: dict[str, float | int] = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        key, sep, value = item.partition("=")
        _require(bool(sep),
                 f"--deep-start-curriculum items must be key=value: {item!r}")
        key = key.strip()
        _require(key in ("p", "target", "cap"),
                 f"--deep-start-curriculum unknown key (only p/target/cap allowed): {key!r}")
        _require(key not in fields, f"--deep-start-curriculum duplicate key: {key!r}")
        try:
            fields[key] = float(value) if key == "p" else int(value)
        except ValueError as exc:
            raise ValueError(
                f"--deep-start-curriculum value unparseable: {item!r}") from exc
    _require(set(fields) == {"p", "target", "cap"},
             "--deep-start-curriculum must give all three keys p/target/cap")
    _require(math.isfinite(fields["p"]) and 0.0 <= fields["p"] <= 1.0,
             f"--deep-start-curriculum p must lie in [0, 1]: {fields['p']!r}")
    _require(fields["target"] >= 2,
             f"--deep-start-curriculum target must be >=2 (the start is always level 1): {fields['target']!r}")
    _require(fields["cap"] >= 1,
             f"--deep-start-curriculum cap must be >=1: {fields['cap']!r}")
    return fields


def _prologue_dungeon_level(env) -> int:
    """Current dungeon_level of an OptionsEnv (testable with an isomorphic stub env; missing raw counts as level 1)."""
    raw = getattr(getattr(env, "env", None), "_raw", None)
    if isinstance(raw, dict):
        return int(raw.get("dungeon_level", 1))
    return 1


def _play_deep_start_prologue(env, obs, info, *, spec: dict, form: str):
    """One prologue play-through; returns None at episode end (death/truncation) so the caller redraws.

    dive form: repeatedly choose DIVE (FARM if masked) until dungeon_level >= target or windows >= cap;
    exhausted form: FARM until drained and DIVE legal (or FARM masked = forced handover), then hand over.
    The mask is never all False (OptionsEnv contract: FARM is the fallback when DIVE is illegal), so the chosen action is always legal.
    """
    from diablogym.options_env import DIVE, FARM

    target = int(spec["target"])
    cap = int(spec["cap"])
    windows = 0
    while windows < cap:
        masks = env.action_masks()
        if form == "dive":
            if _prologue_dungeon_level(env) >= target:
                break
            action = DIVE if bool(masks[DIVE]) else FARM
        else:  # exhausted
            if ((bool(getattr(env, "exhausted", False)) and bool(masks[DIVE]))
                    or not bool(masks[FARM])):
                break
            action = FARM
        obs, _reward, terminated, truncated, info = env.step(action)
        windows += 1
        if terminated or truncated:
            return None
        if form == "dive" and _prologue_dungeon_level(env) >= target:
            break
    return obs, info, windows


def _deep_start_prologue(env, obs, info, seed, *, spec: dict, form: str,
                         resample_seed, reset):
    """Core of the R9 deep-start curriculum prologue (module level, covered by stub-env unit tests).

    The trigger decision is independent per episode: a deterministic RNG derived from the environment episode seed (``random.Random(seed
    ^ 0xD1CE)``), never touching the global RNG -- the same seed always decides the same. An episode end during the prologue
    (death/truncation) redraws with a new seed through the ``resample_seed``/``reset`` closures, capped at
    ``_DEEP_START_MAX_RESAMPLES``; past the cap the play-through is abandoned and a new seed hands over from an ordinary start.
    Returns (obs, info, seed, telemetry); the four telemetry keys
    {prologue_triggered, start_dlvl, prologue_windows, resamples} are merged by the caller
    into the info returned by reset.
    """
    probability = float(spec["p"])
    triggered = random.Random(seed ^ 0xD1CE).random() < probability
    if not triggered:
        return obs, info, seed, {
            "prologue_triggered": False,
            "start_dlvl": _prologue_dungeon_level(env),
            "prologue_windows": 0,
            "resamples": 0,
        }
    resamples = 0
    while True:
        outcome = _play_deep_start_prologue(env, obs, info, spec=spec, form=form)
        if outcome is not None:
            obs, info, windows = outcome
            break
        if resamples >= _DEEP_START_MAX_RESAMPLES:
            seed = resample_seed()
            obs, info = reset(seed)
            windows = 0
            break
        resamples += 1
        seed = resample_seed()
        obs, info = reset(seed)
    return obs, info, seed, {
        "prologue_triggered": True,
        "start_dlvl": _prologue_dungeon_level(env),
        "prologue_windows": windows,
        "resamples": resamples,
    }


def _resolve_dry_curriculum_start(
        schedule_table,
        *,
        start_steps: int,
        rollout_quantum: int,
        total_steps: int,
        leg_start: int = _DRY_CURRICULUM_LEG_START,
) -> tuple[int, float]:
    """Resolve the probability that must be live before SB3's first reset.

    ``BaseAlgorithm._setup_learn()`` resets a newly attached VecEnv before
    callbacks receive ``on_training_start``.  A native continuation therefore
    cannot initialize every environment with table[0] and repair it later:
    that reset may already select/skip the first FARM window under the wrong
    probability.  Bind the initial environment value to the immutable
    checkpoint step and reject a table that cannot cover the remaining run.
    """
    table = tuple(float(value) for value in schedule_table)
    _require(table and all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in table
    ), "dry-curriculum start resolution needs a non-empty table of finite probabilities in [0,1]")
    for label, value in (
            ("start_steps", start_steps),
            ("rollout_quantum", rollout_quantum),
            ("total_steps", total_steps),
            ("leg_start", leg_start)):
        _require(
            _is_plain_int(value),
            f"dry-curriculum {label} must be a plain integer",
        )
    _require(rollout_quantum > 0 and total_steps > 0,
             "dry-curriculum rollout quantum/training steps must be positive")
    _require(total_steps % rollout_quantum == 0,
             "dry-curriculum training steps must close on the rollout quantum")
    offset = start_steps - leg_start
    _require(
        offset >= 0 and offset % rollout_quantum == 0,
        "dry-curriculum checkpoint start is not on a leg-relative rollout boundary: "
        f"start={start_steps},leg_start={leg_start},quantum={rollout_quantum}",
    )
    index = offset // rollout_quantum
    rollout_count = total_steps // rollout_quantum
    _require(
        index < len(table)
        and index + rollout_count <= len(table),
        "dry-curriculum table not long enough to cover the continuation: "
        f"start_index={index},rollouts={rollout_count},table={len(table)}",
    )
    return int(index), float(table[index])


# ---- E3 4B auxiliary demo pathway (PREREG-v33-content-case E3; the two flags do not force each other, zero-intrusion clause) ----


def _bc_aux_active(args) -> bool:
    """Auxiliary path is active only with demos and an explicit mechanism.

    ``--bc-aux-graft`` is the rev9 circuit used by the rev10 objective.  The
    λ predicate is retained here only so old contracts fail with a precise
    migration error in ``_validate_args`` rather than silently becoming
    zero-intrusion.
    """
    return bool(getattr(args, "bc_aux_demos", None)) and (
        float(getattr(args, "bc_aux_lambda", 0.0)) > 0
        or bool(getattr(args, "bc_aux_graft", False)))


def _bc_aux_structural_active(args) -> bool:
    return bool(getattr(args, "bc_aux_graft", False)
                and getattr(args, "bc_aux_demos", None))


def _parse_bc_aux_demos_v2(path: str | pathlib.Path):
    """E3 4B: dedicated validator for the bc-worker-v2 demo set (mirror assertions written separately per generation).

    The v1 surface (_BC_REPORT_SCHEMA_VERSION=1/_validate_bc_report/_load_dry_anchor_demos/
    canonical bc-worker path) is untouched; this validator stands alone, and the v2 demos schema follows design
    E2's shared source of truth = the v1 keys (X/Y/episode_id) + a per-sample masks array (captured live from
    env.action_masks() at collection, the single on-manifold source of truth; inferring from obs is forbidden).
    Generation-conditioned forbidden-action mirror: v2 forbids 11, allows 12. The v1 dry-state dual-channel saturation assertion
    belongs to the dry-anchor probe only and is not mirrored (implementation note). Returns
    (X, Y, episode_id, masks, sha256); episode_id is kept for the real held-out behaviour gate,
    and passing off whole-pool training states as independent validation states is forbidden.
    """
    import numpy as np

    p = pathlib.Path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"4B v2 demo set unreadable: {p}: {exc}") from exc
    sha256 = hashlib.sha256(payload).hexdigest()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            required = {
                "X", "Y", "episode_id", "masks",
                "schema_version", "protocol_version",
                "implementation_sha256", "generator_sha256",
                "manager_npz_sha256", "teacher_generation",
                "preventive_threshold",
            }
            _require(set(data.files) == required,
                     "4B v2 demos.npz schema/provenance fields do not match exactly: "
                     f"missing={sorted(required - set(data.files))},"
                     f"extra={sorted(set(data.files) - required)}")
            x, y = data["X"].copy(), data["Y"].copy()
            episode_id, masks = data["episode_id"].copy(), data["masks"].copy()
            meta = {}
            for key in required - {"X", "Y", "episode_id", "masks"}:
                value = np.asarray(data[key])
                _require(value.shape == (),
                         f"4B v2 provenance {key} must be a 0-d scalar")
                meta[key] = value.item()
    except (OSError, ValueError) as exc:
        raise ValueError(f"4B v2 demo set unreadable: {p}: {exc}") from exc
    _require(meta["schema_version"] == _BC_V2_DEMOS_SCHEMA_VERSION,
             f"4B v2 demos schema outdated: {meta['schema_version']!r}")
    _require(isinstance(meta["protocol_version"], (int, np.integer))
             and int(meta["protocol_version"]) == PROTOCOL_VERSION,
             "4B v2 demos evaluation/environment protocol outdated: "
             f"{meta['protocol_version']!r} != {PROTOCOL_VERSION}")
    _require(meta["implementation_sha256"] == _implementation_bundle_sha256(),
             "4B v2 demos implementation_sha256 disagrees with the current training implementation")
    generator = pathlib.Path(__file__).with_name("bc_worker.py")
    _require(meta["generator_sha256"]
             == hashlib.sha256(generator.read_bytes()).hexdigest(),
             "4B v2 demos generator_sha256 drift: train/bc_worker.py")
    _require(_is_sha256(meta["manager_npz_sha256"]),
             "4B v2 demos manager_npz_sha256 invalid")
    _require(isinstance(meta["teacher_generation"], (int, np.integer))
             and int(meta["teacher_generation"]) == _BC_V2_TEACHER_GENERATION,
             f"4B v2 demos teacher_generation is not 2: {meta['teacher_generation']!r}")
    _require(isinstance(meta["preventive_threshold"], (int, float,
                                                        np.integer, np.floating))
             and float(meta["preventive_threshold"])
             in _BC_V2_PREVENTIVE_THRESHOLDS,
             "4B v2 demos preventive_threshold not registered: "
             f"{meta['preventive_threshold']!r}")
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.ndim == 1 and len(x) == len(y),
             f"4B v2 array shapes invalid: X={x.shape},Y={y.shape}")
    _require(x.dtype == np.float32 and np.issubdtype(y.dtype, np.integer),
             f"4B v2 dtype invalid: X={x.dtype},Y={y.dtype}")
    _require(episode_id.ndim == 1 and len(episode_id) == len(x)
             and np.issubdtype(episode_id.dtype, np.integer)
             and len(np.unique(episode_id)) >= 2,
             "4B v2 episode_id shape/type/independent episode count invalid")
    _require(masks.ndim == 2 and masks.shape == (len(x), 15)
             and masks.dtype == np.bool_,
             f"4B v2 masks shape/dtype invalid: {getattr(masks, 'shape', None)},"
             f"{getattr(masks, 'dtype', None)}")
    _require(bool(((y >= 0) & (y < 15)).all()), "4B v2 labels out of range")
    _require(not np.isin(y, _WORKER_BC_V2_FORBIDDEN_ACTIONS).any(),
             "4B v2 demo set contains generation-forbidden action 11 (v2 forbids 11, allows 12; guard surface not weakened)")
    # Discretionary hardening note: every per-sample label must be a legal bit of its own mask (a given for really executed on-manifold steps;
    # a masked label would make the auxiliary CE take the -1e8 entry); for class 12 pairs this implies the design's m[12]=True assertion.
    _require(bool(masks[np.arange(len(y)), y].all()),
             "4B v2 has demo pairs whose label is forbidden by their own mask (on-manifold broken, fail-loud)")
    latch = x[:, _A12_CALIBRATION_DRINK_LATCH_FEATURE]
    _require(bool(np.isfinite(latch).all())
             and bool((((0.0 <= latch) & (latch <= 1.0))
                       | ((-2.0 <= latch) & (latch <= -1.0))).all()),
             "4B v2 feature297 active-drink bit outside the not-drunk [0,1] / drunk [-2,-1] encoding domains")
    hp = x[:, _A12_CALIBRATION_HP_FEATURE]
    threshold = float(meta["preventive_threshold"])
    visible_target = (
        (hp >= 0.5 - _A12_VISIBLE_HP_BOUNDARY_EPS)
        & (hp < threshold - _A12_VISIBLE_HP_BOUNDARY_EPS)
        & masks[:, 12]
        & (latch >= 0.0)
    )
    _require(np.array_equal(y == 12, visible_target),
             "4B v2 labels differ from the live visible TeacherV2 predicate"
             " (hp/m12/active-drink bit); refusing hidden state or obs/raw misalignment")
    return x, y, episode_id, masks, sha256, meta


def _load_bc_aux_demos_v2(
        path: str | pathlib.Path, *,
        expected_manager_sha256: str):
    """Read and validate a committed BC-v2 PASS bundle.

    The metadata embedded in demos can only prove a "self-reported identity"; the generator may write data
    before training/calibration fails. The sibling PASS report is the commit marker of the three-file set and must bind the current
    demos/policy bytes, the fixed 384 episodes and the n12 data gate; RUNNING/FAIL/missing files
    are always refused entry to the PPO optimizer.
    """
    from eval_contract import EvalContractError, strict_json_loads
    import numpy as np

    p = pathlib.Path(path)
    _require(_is_sha256(expected_manager_sha256),
             "4B v2 loader is missing this training run's manager_npz_sha256")
    x, y, episode_id, masks, demos_sha256, meta = (
        _parse_bc_aux_demos_v2(p))
    _require(meta["manager_npz_sha256"] == expected_manager_sha256,
             "4B v2 demos manager distribution disagrees with this training run's --manager-npz: "
             f"{meta['manager_npz_sha256']} != {expected_manager_sha256}")
    report_path = p.with_name("bc_report_v2.json")
    policy_path = p.with_name("policy_sd.pt")
    try:
        report = strict_json_loads(report_path.read_bytes())
    except (OSError, EvalContractError) as exc:
        raise ValueError(
            f"4B v2 PASS receipt missing/unreadable: {report_path}") from exc
    _require(isinstance(report, dict), "4B v2 PASS receipt must be a JSON object")
    _require(set(report) == set(_BC_V2_PASS_KEYS),
             "4B v2 PASS receipt fields/schema not exact: "
             f"missing={sorted(set(_BC_V2_PASS_KEYS) - set(report))},"
             f"extra={sorted(set(report) - set(_BC_V2_PASS_KEYS))}")
    _require(report["schema_version"] == _BC_V2_REPORT_SCHEMA_VERSION
             and report["data_gate"] == "PASS",
             "4B v2 sibling report did not pass the current schema/data_gate")
    for key in (
            "protocol_version", "implementation_sha256", "generator_sha256",
            "manager_npz_sha256", "teacher_generation",
            "preventive_threshold"):
        _require(report[key] == meta[key],
                 f"4B v2 report/demos provenance mismatch: {key}")
    _validate_bc_final_holdout_marker(
        p.parent, 2, _BC_V2_COLLECTION_EPISODES, report)
    _require(report["demos_sha256"] == demos_sha256,
             "4B v2 PASS receipt not bound to the live demos bytes")
    _require(report["pairs"] == len(y)
             and report["collection_episodes"]
             == len(_BC_V2_COLLECTION_EPISODES),
             "4B v2 PASS receipt pairs/collection_episodes do not close")
    episodes = np.unique(episode_id)
    _require(np.array_equal(
        episodes, np.asarray(_BC_V2_COLLECTION_EPISODES, dtype=episodes.dtype)),
        "4B v2 demos must exactly cover the current fixed episodes "
        "2103000..2103383")
    _bc_v2_post_drink_coverage(x, y, episode_id, masks)
    n12 = int((y == 12).sum())
    _require(report["n12"] == n12
             and report["n12_gate_min"] == _BC_V2_N12_MIN
             and n12 >= _BC_V2_N12_MIN,
             "4B v2 n12 data gate does not close with the live labels")
    heldout = _bc_v2_holdout_indices(episode_id)
    heldout_episodes = sorted(
        int(v) for v in np.unique(episode_id[heldout]))
    _require(report["held_out_pairs"] == len(heldout)
             and report["held_out_episodes"] == heldout_episodes,
             "4B v2 held-out episode split does not close with the live data")
    gate = report["a12_behavior_gate"]
    _require(isinstance(gate, dict) and gate.get("verdict") == "PASS",
             "4B v2 sibling report's a12 behaviour gate did not PASS")
    try:
        policy_payload = policy_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"4B v2 PASS receipt bound weights missing/unreadable: {policy_path}") from exc
    _require(hashlib.sha256(policy_payload).hexdigest()
             == report["policy_sha256"],
             "4B v2 PASS receipt not bound to the live policy_sd.pt bytes")

    # A hash only proves "these bytes are named by the receipt"; it cannot prove they are an executable policy, still less
    # that the receipt's held-out behaviour readings were produced by these weights. Deserialise with weights_only
    # only, and pin the standard BC-v2 six-tensor shapes/finiteness one by one.
    import torch as th
    try:
        policy_sd = th.load(
            io.BytesIO(policy_payload), map_location="cpu",
            weights_only=True)
    except Exception as exc:
        raise ValueError(
            "4B v2 policy_sd.pt is not safely parseable weights-only weights"
        ) from exc
    expected_shapes = {
        "mlp_extractor.policy_net.0.weight": (64, x.shape[1]),
        "mlp_extractor.policy_net.0.bias": (64,),
        "mlp_extractor.policy_net.2.weight": (64, 64),
        "mlp_extractor.policy_net.2.bias": (64,),
        "action_net.weight": (15, 64),
        "action_net.bias": (15,),
    }
    _require(isinstance(policy_sd, dict)
             and set(policy_sd) == set(_POLICY_HEAD_KEYS),
             "4B v2 policy_sd.pt policy-head six-tensor key set not exact")
    for key, shape in expected_shapes.items():
        tensor = policy_sd[key]
        _require(isinstance(tensor, th.Tensor)
                 and tensor.dtype == th.float32
                 and tuple(tensor.shape) == shape
                 and bool(th.isfinite(tensor).all()),
                 f"4B v2 policy_sd.pt tensor shape/finiteness invalid: "
                 f"{key}={getattr(tensor, 'shape', None)}/"
                 f"{getattr(tensor, 'dtype', None)}")

    _validate_bc_v2_calibration_receipt(
        report, policy_sd, x, y, episode_id, masks)

    # Neither the hash nor the a12 safety gate proves the other combat classes are still there. Recompute top1 and every
    # per-class recall with enough coverage on the live held-out/masks; a12 goes through the dedicated 0.5 + safety gate below,
    # the other classes still strictly require 0.85.
    with th.no_grad():
        heldout_logits = _policy_logits_from_sb3_state_dict(
            policy_sd, x[heldout])
        heldout_mask = th.as_tensor(masks[heldout], dtype=th.bool)
        heldout_pred = th.where(
            heldout_mask, heldout_logits,
            th.full_like(heldout_logits, -1e8)
        ).argmax(dim=-1).cpu().numpy()
    heldout_y = y[heldout]
    observed_top1 = float((heldout_pred == heldout_y).mean())
    _require(
        isinstance(report["held_out_top1"], (int, float))
        and not isinstance(report["held_out_top1"], bool)
        and math.isclose(
            float(report["held_out_top1"]), observed_top1,
            rel_tol=0.0, abs_tol=1e-15)
        and observed_top1 >= 0.95,
        "4B v2 held_out_top1 disagrees with the live-weights recomputation or missed the gate")
    full_counts = np.bincount(y, minlength=15)
    gated_actions = np.flatnonzero(full_counts >= 300)
    reported_recalls = report["class_recalls"]
    _require(
        isinstance(reported_recalls, dict)
        and set(reported_recalls)
        == {str(int(action)) for action in gated_actions},
        "4B v2 class_recalls class set disagrees with the live demos")
    for action in gated_actions:
        selected = heldout_y == action
        recall = (
            float((heldout_pred[selected] == action).mean())
            if selected.any() else 0.0)
        reported = reported_recalls[str(int(action))]
        _require(
            isinstance(reported, (int, float))
            and not isinstance(reported, bool)
            and math.isclose(
                float(reported), recall,
                rel_tol=0.0, abs_tol=1e-15),
            "4B v2 class_recalls disagree with the live-weights recomputation:"
            f" action={int(action)}")
        if int(action) != 12:
            _require(
                recall >= 0.85,
                "4B v2 non-a12 per-class recall below 0.85:"
                f" action={int(action)}, recall={recall}")

    observed_behavior = bc_aux_behavior_metrics(
        policy_sd, x, y, episode_id, masks, heldout_only=True)
    reported_behavior = report["a12_behavior"]

    def _metrics_match(expected, observed) -> bool:
        # Counts/strings/None in JSON must match exactly; floats tolerate last-digit differences between the CPU forward and a JSON
        # round trip, but non-finite values are refused.
        if isinstance(expected, dict) or isinstance(observed, dict):
            return (isinstance(expected, dict)
                    and isinstance(observed, dict)
                    and set(expected) == set(observed)
                    and all(_metrics_match(expected[key], observed[key])
                            for key in expected))
        if (isinstance(expected, (int, float))
                and not isinstance(expected, bool)
                and isinstance(observed, (int, float))
                and not isinstance(observed, bool)):
            if isinstance(expected, int) and isinstance(observed, int):
                return expected == observed
            return (math.isfinite(float(expected))
                    and math.isfinite(float(observed))
                    and math.isclose(float(expected), float(observed),
                                     rel_tol=1e-7, abs_tol=1e-9))
        return type(expected) is type(observed) and expected == observed

    _require(_metrics_match(reported_behavior, observed_behavior),
             "4B v2 receipt a12_behavior disagrees with the live-weights recomputation")
    # The BC-v2 teacher recall is handled by the separate, same-source 0.5 gate below; since rev8 the generic safety gate
    # uses the same schema as the deployment-side exact mixture and must no longer imply
    # the old target "a12 must become the argmax", otherwise producer/loader/evaluator
    # would compute different gates for the same receipt.
    observed_gate = bc_aux_behavior_gate(
        observed_behavior, require_teacher_recall=False)
    _require(observed_gate["verdict"] == "PASS",
             "4B v2 live-weights recomputed a12 behaviour gate did not PASS: "
             f"{observed_gate['reasons']}")
    _require(
        observed_behavior["recall_12"] >= _BC_V2_TEACHER_RECALL_MIN
        and report["recall_12_denominator"]
        == observed_behavior["true_a12"]
        and math.isclose(
            float(report["recall_12"]),
            round(float(observed_behavior["recall_12"]), 4),
            rel_tol=0.0, abs_tol=1e-12),
        "4B v2 live weights failed the teacher-only a12 recall 0.5 gate"
        " or the legacy recall field has a different source")
    _require(report["a12_behavior_gate"] == observed_gate,
             "4B v2 receipt a12_behavior_gate disagrees with the live recomputation")
    return x, y, episode_id, masks, demos_sha256


def _filter_bc_aux_demo_pairs(x, y, masks):
    """Build the legacy calibration bank: all positives + hard negatives with a real m[12].

    The old implementation ``keep = y == 12`` repeatedly maximised π(12) without any counter-examples;
    a 102,400-step smoke run measurably collapsed to 91% "drink whenever there is a potion". The fixed version consumes only the live
    collection masks, caps negatives at 8:1, and covers both:
      * clearly non-trigger states with hp>=0.65 (the bulk of the old model's false positives);
      * non-a9 actions, so the negative side does not degenerate into "learn a9 everywhere".
    Post-drink rows with feature297<0 must now also have m12=False; they only prove the visible latch/mask
    interface closes and cannot go into a legacy BCE bank that requires m12=True.
    Selection is a deterministic equally spaced subsample and consumes neither the training nor the auxiliary RNG.
    """
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    masks = np.asarray(masks)
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.shape == (len(x),)
             and masks.shape == (len(x), 15),
             "a12 calibration bank input shape invalid")
    positive = np.flatnonzero(y == 12)
    _require(len(positive) > 0,
             "--bc-aux-demos has no class-12 demo pairs (fail-loud)")
    _require(bool(masks[positive, 12].all()),
             "class-12 demo pairs with m[12]=False: collection surface on-manifold broken")
    negative = np.flatnonzero((y != 12) & masks[:, 12])
    _require(len(negative) > 0,
             "--bc-aux-demos has no non-12 hard negative with m[12]=True; "
             "refusing to fall back to the positive-only objective")
    _require(len(negative) >= _BC_AUX_MIN_NEGATIVE_RATIO * len(positive),
             "a12 calibration bank has too few legal hard negatives: "
             f"{len(negative)} < {_BC_AUX_MIN_NEGATIVE_RATIO}×"
             f"{len(positive)}; suspect insufficient coverage of the active-drink visible bit/safe negatives")
    post_drink_closed = np.flatnonzero(
        (y != 12)
        & ~masks[:, 12]
        & (x[:, _A12_CALIBRATION_DRINK_LATCH_FEATURE] < 0.0))
    _require(
        len(post_drink_closed)
        >= _BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO * len(positive),
        "a12 demos lack evidence that the visible post-drink latch closes: "
        f"{len(post_drink_closed)} < "
        f"{_BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO}×{len(positive)}; "
        "cannot prove that repeat drinking in the same window is closed by visible state + mask")

    limit = min(len(negative), _BC_AUX_NEGATIVE_RATIO * len(positive))
    chosen: list[int] = []
    seen: set[int] = set()

    def take(pool, quota):
        pool = np.asarray([int(i) for i in pool if int(i) not in seen],
                          dtype=np.int64)
        count = min(int(quota), len(pool), limit - len(chosen))
        if count <= 0:
            return
        # floor(k*N/n) is strictly increasing for n<=N, so there are no duplicates and coverage is even across the original sequence.
        pick = pool[np.floor(
            np.arange(count, dtype=np.float64) * len(pool) / count
        ).astype(np.int64)]
        for raw in pick:
            index = int(raw)
            if index not in seen:
                chosen.append(index)
                seen.add(index)

    hp = x[:, 0]
    take(negative[hp[negative] >= 0.65], max(1, limit // 2))
    take(negative[(hp[negative] >= 0.5) & (hp[negative] < 0.65)],
         max(1, limit // 4))
    take(negative[y[negative] != 9], max(1, limit // 8))
    take(negative, limit - len(chosen))
    _require(len(chosen) == limit,
             f"a12 hard-negative selection count invalid: {len(chosen)} != {limit}")
    keep = np.concatenate([positive, np.asarray(chosen, dtype=np.int64)])
    return x[keep], y[keep], masks[keep]


def _bc_v2_holdout_indices(episode_id):
    """Mirror the rng(23)/10% whole-episode split of bc_worker.split_by_episode.

    train_ppo cannot import bc_worker in reverse (the latter already imports this module), so a small closed
    pure function is kept here, and tests align it array by array.
    """
    import numpy as np

    groups = np.asarray(episode_id)
    _require(groups.ndim == 1 and len(groups) > 0
             and np.issubdtype(groups.dtype, np.integer),
             "BC-v2 held-out episode_id invalid")
    episodes = np.unique(groups)
    _require(len(episodes) >= 2, "BC-v2 held-out needs at least 2 independent episodes")
    order = np.random.default_rng(_BC_FINAL_SPLIT_SEED).permutation(episodes)
    held = order[:max(1, int(round(len(order) * 0.1)))]
    indices = np.flatnonzero(np.isin(groups, held))
    _require(0 < len(indices) < len(groups),
             "BC-v2 episode split produced an empty training set or empty held-out")
    return indices


def _bc_v2_training_indices(episode_id):
    """Training row indices mutually exclusive with and complementary to the fixed held-out episodes."""
    import numpy as np

    groups = np.asarray(episode_id)
    heldout = _bc_v2_holdout_indices(groups)
    selected = np.ones(len(groups), dtype=np.bool_)
    selected[heldout] = False
    training = np.flatnonzero(selected)
    _require(len(training) + len(heldout) == len(groups)
             and not np.intersect1d(training, heldout).size,
             "BC-v2 training/held-out row split not mutually exclusive and complete")
    train_episodes = np.unique(groups[training])
    heldout_episodes = np.unique(groups[heldout])
    _require(not np.intersect1d(train_episodes, heldout_episodes).size,
             "BC-v2 training/held-out episode leak")
    return training


def _bc_v2_fit_validation_indices(episode_id):
    """Mirror bc_worker's nested rng(2301)/10% whole-episode candidate-selection split."""
    import numpy as np

    groups = np.asarray(episode_id)
    training = _bc_v2_training_indices(groups)
    training_episodes = np.unique(groups[training])
    _require(len(training_episodes) >= 2,
             "BC-v2 nested split needs at least two training episodes")
    order = np.random.default_rng(
        _BC_SELECTION_SPLIT_SEED).permutation(training_episodes)
    n_validation = max(
        1, int(round(
            len(order) * _BC_SELECTION_VALIDATION_FRACTION)))
    n_validation = min(n_validation, len(order) - 1)
    validation_episodes = order[:n_validation]
    training_mask = np.zeros(len(groups), dtype=np.bool_)
    training_mask[training] = True
    validation_mask = np.isin(groups, validation_episodes)
    fit = np.flatnonzero(training_mask & ~validation_mask)
    validation = np.flatnonzero(training_mask & validation_mask)
    heldout = _bc_v2_holdout_indices(groups)
    _require(len(fit) > 0 and len(validation) > 0
             and len(fit) + len(validation) + len(heldout) == len(groups)
             and not np.intersect1d(fit, validation).size
             and not np.intersect1d(fit, heldout).size
             and not np.intersect1d(validation, heldout).size,
             "BC-v2 nested fit/validation/final three domains not mutually exclusive and complete")
    return fit, validation


def _bc_v2_post_drink_coverage(
        x, y, episode_id, masks, *,
        scopes: tuple[str, ...] = ("fit", "validation", "final")) -> dict:
    """Prove, on the domains the caller is allowed to read, that the visible post-drink latch and the m12 mask share a source.

    Before the candidate is frozen the producer may only pass ``("fit", "validation")``. The default three domains
    are only for re-verification by the consumer of an already committed one-shot bundle; so the coverage diagnosis itself can no longer peek at
    the final heldout.
    """
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)
    groups = np.asarray(episode_id)
    masks = np.asarray(masks)
    _require(
        x.ndim == 2 and x.shape[1] == 298
        and y.shape == groups.shape == (len(x),)
        and masks.shape == (len(x), 15)
        and masks.dtype == np.bool_,
        "BC-v2 post-drink coverage input shape/dtype invalid",
    )
    fit, validation = _bc_v2_fit_validation_indices(groups)
    heldout = _bc_v2_holdout_indices(groups)
    allowed = {
        "fit": fit,
        "validation": validation,
        "final": heldout,
    }
    _require(
        isinstance(scopes, tuple)
        and bool(scopes)
        and len(scopes) == len(set(scopes))
        and all(scope in allowed for scope in scopes),
        f"BC-v2 post-drink coverage scopes invalid: {scopes!r}",
    )
    result = {}
    for scope in scopes:
        indices = allowed[scope]
        positive = int((y[indices] == 12).sum())
        post_drink_negative = int((
            (y[indices] != 12)
            & ~masks[indices, 12]
            & (x[indices, _A12_CALIBRATION_DRINK_LATCH_FEATURE] < 0.0)
        ).sum())
        _require(
            positive > 0
            and post_drink_negative
            >= _BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO * positive,
            f"BC-v2 {scope} domain lacks visible post-drink latch coverage: "
            f"positive={positive},post_drink_negative="
            f"{post_drink_negative}, required>="
            f"{_BC_AUX_MIN_POST_DRINK_NEGATIVE_RATIO}:1",
        )
        result[scope] = {
            "positive_a12": positive,
            "post_drink_masked_negatives": post_drink_negative,
        }
    return result


def _build_bc_aux_training_bank(x, y, episode_id, masks):
    """Build the optimisation bank only from the fixed training episodes; held-out never enters the graph."""
    training = _bc_v2_training_indices(episode_id)
    return _filter_bc_aux_demo_pairs(
        x[training], y[training], masks[training])


def _policy_logits_from_sb3_state_dict(
        policy_sd, obs, *, legacy_scene_clock: bool = False,
        action_masks=None, circuit_spec=None,
        return_raw_actor_logits: bool = False):
    """Offline forward pass on the six SB3 policy-head tensors; shared by BC-v2/E5/G0.

    ``legacy_scene_clock`` is a parameter name kept for existing callers; when true it actually applies
    the full protocol-v3 worker compatibility view (286 restores the heals main scale, 297 restores the
    exhausted bit). The current student's contextual gate must still read the raw signed
    observation and must not be converted by default in this generic forward pass.
    """
    import numpy as np
    import torch as th

    required = (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight", "action_net.bias",
    )
    _require(isinstance(return_raw_actor_logits, bool),
             "return_raw_actor_logits must be a bool")
    _require(isinstance(policy_sd, dict)
             and all(key in policy_sd for key in required),
             "the six SB3 policy-head tensors required by the offline behaviour gate are incomplete")
    raw_x = th.as_tensor(np.asarray(obs, dtype=np.float32))
    x = raw_x
    if legacy_scene_clock:
        from leashed_ppo import _legacy_worker_observation_view

        x = _legacy_worker_observation_view(x)
    w0, b0, w1, b1, wa, ba = (policy_sd[key] for key in required)
    tensors = tuple(th.as_tensor(t, dtype=th.float32)
                    for t in (w0, b0, w1, b1, wa, ba))
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.ndim == 2 and w0.shape[1] == x.shape[1]
             and wa.ndim == 2 and wa.shape[0] == 15,
             "offline behaviour gate policy-head shape invalid")
    with th.no_grad():
        h = th.tanh(x @ w0.T + b0)
        h = th.tanh(h @ w1.T + b1)
        logits = h @ wa.T + ba
        raw_actor_logits = logits
        expanded = int(w0.shape[0]) == _BC_AUX_CIRCUIT_EXPANDED_WIDTH
        _require(
            (expanded and circuit_spec == _bc_aux_circuit_spec())
            or (not expanded and circuit_spec is None),
            "the offline behaviour gate must not guess adapter semantics from the 68 width alone; "
            "the current contextual circuit spec must be bound explicitly",
        )
        if expanded:
            _require(action_masks is not None,
                     "contextual mixture offline forward must provide per-sample masks")
            from leashed_ppo import (
                _a12_mixture_logits,
                _legacy_worker_observation_view,
            )
            # The live adapter feeds the complete protocol-v3 view to the old
            # actor while eligibility reads the untouched signed row.
            legacy_x = _legacy_worker_observation_view(raw_x)
            h = th.tanh(legacy_x @ w0.T + b0)
            h = th.tanh(h @ w1.T + b1)
            raw_logits = h @ wa.T + ba
            raw_actor_logits = raw_logits
            spec = circuit_spec
            feature_indices = tuple(
                int(value) for value in spec["gate_feature_indices"])
            parameter_columns = tuple(
                int(value) for value in spec["gate_parameter_columns"])
            gate_logits = (
                raw_x[:, feature_indices]
                @ wa[
                    int(spec["action_index"]),
                    list(parameter_columns),
                ]
                + ba[int(spec["action_index"])]
            )
            logits = _a12_mixture_logits(
                raw_logits,
                raw_x,
                action_masks,
                gate_logits,
                action=int(spec["action_index"]),
                hp_low=float(spec["hp_low"]),
                hp_high=float(spec["hp_high"]),
                boundary_epsilon=float(spec["boundary_epsilon"]),
                probability_min=float(spec["probability_min"]),
                probability_max=float(spec["probability_max"]),
            )
    return raw_actor_logits if return_raw_actor_logits else logits


def _validate_bc_v2_calibration_receipt(
        report, policy_sd, x, y, episode_id, masks) -> None:
    """Recompute the calibration fit receipt and three-domain boundaries from the canonical demos + final six tensors."""
    import numpy as np
    import torch as th

    calibration = report.get("a12_calibration")
    expected_keys = {
        "schema_version", "fit_scope", "fit_pairs", "fit_episodes",
        "validation_pairs_excluded", "validation_episodes_excluded",
        "final_heldout_pairs_excluded",
        "final_heldout_episodes_excluded", "hp_low", "hp_high",
        "hp_feature", "drink_latch_feature", "predicate", "bias_12",
        "target_recall_12", "fit_metrics",
    }
    _require(isinstance(calibration, dict)
             and set(calibration) == expected_keys,
             "4B v2 a12_calibration fields/schema not exact")
    _require(
        calibration["schema_version"] == _A12_CALIBRATION_SCHEMA_VERSION
        and calibration["fit_scope"] == "nested-fit-episodes-only"
        and calibration["hp_low"] == 0.5
        and calibration["hp_high"] == report["preventive_threshold"]
        and calibration["hp_feature"] == _A12_CALIBRATION_HP_FEATURE
        and calibration["drink_latch_feature"]
        == _A12_CALIBRATION_DRINK_LATCH_FEATURE
        and calibration["predicate"] == _A12_CALIBRATION_PREDICATE
        and calibration["target_recall_12"]
        == _A12_CALIBRATION_TRAIN_RECALL_TARGET,
        "4B v2 a12_calibration identity/visible predicate mismatch")

    fit, validation = _bc_v2_fit_validation_indices(episode_id)
    final_heldout = _bc_v2_holdout_indices(episode_id)
    expected_counts = {
        "fit_pairs": len(fit),
        "fit_episodes": len(np.unique(np.asarray(episode_id)[fit])),
        "validation_pairs_excluded": len(validation),
        "validation_episodes_excluded":
            len(np.unique(np.asarray(episode_id)[validation])),
        "final_heldout_pairs_excluded": len(final_heldout),
        "final_heldout_episodes_excluded":
            len(np.unique(np.asarray(episode_id)[final_heldout])),
    }
    for key, expected in expected_counts.items():
        value = calibration[key]
        _require(_is_plain_int(value) and value > 0 and value == expected,
                 "4B v2 a12_calibration three-domain deterministic split inconsistent: "
                 f"{key}={value!r} != {expected}")

    bias = _finite_number(
        calibration["bias_12"], "4B v2 a12_calibration bias_12")
    policy_bias = float(
        th.as_tensor(policy_sd["action_net.bias"])[12].detach().cpu())
    # The producer reads the final float32 parameters back before writing JSON; JSON numbers round-trip that value losslessly,
    # so exact equality is required here. An approximate tolerance would reopen the "receipt is not the deployed weights" gap.
    _require(bias == policy_bias,
        "4B v2 a12_calibration bias_12 not bound to the final policy")

    fx = np.asarray(x)[fit]
    fy = np.asarray(y)[fit]
    fm = np.asarray(masks)[fit]
    logits = _policy_logits_from_sb3_state_dict(policy_sd, fx)
    masked = th.where(
        th.as_tensor(fm, dtype=th.bool), logits,
        th.full_like(logits, -1e8))
    pred = masked.argmax(dim=-1).cpu().numpy()
    probabilities = th.softmax(masked, dim=-1)[:, 12].cpu().numpy()
    pred12 = pred == 12
    true12 = fy == 12
    negative = ~true12
    legal12 = fm[:, 12]
    legal_negative = negative & legal12
    high_hp = (
        negative & legal12
        & (fx[:, _A12_CALIBRATION_HP_FEATURE] >= 0.65))
    tp = int((pred12 & true12).sum())
    fp = int((pred12 & negative).sum())
    high_hp_fp = int((pred12 & high_hp).sum())
    legal_negative_probabilities = probabilities[legal_negative]
    legal_negative_probabilities_f64 = legal_negative_probabilities.astype(
        np.float64, copy=False)
    legal_negative_probability_mean = (
        float(legal_negative_probabilities_f64.mean())
        if len(legal_negative_probabilities) else 0.0)
    legal_negative_probability_max = (
        float(legal_negative_probabilities.max())
        if len(legal_negative_probabilities) else 0.0)
    pred13_share = float((pred == 13).mean())
    true13_share = float((fy == 13).mean())
    observed = {
        "tp": tp,
        "fp": fp,
        "precision_12": tp / max(1, tp + fp),
        "recall_12": tp / max(1, int(true12.sum())),
        "fpr_12": fp / max(1, int(legal_negative.sum())),
        "predicted_share_12": float(pred12.mean()),
        "high_hp_false_drink_rate":
            high_hp_fp / max(1, int(high_hp.sum())),
        "legal_negative_probability_12_mean":
            legal_negative_probability_mean,
        "legal_negative_probability_12_max":
            legal_negative_probability_max,
        "a13_spillover": max(0.0, pred13_share - true13_share),
    }
    metrics = calibration["fit_metrics"]
    _require(isinstance(metrics, dict)
             and set(metrics) == set(observed),
             "4B v2 a12_calibration fit_metrics fields not exact")
    for key, expected in observed.items():
        value = metrics[key]
        if key in {"tp", "fp"}:
            _require(_is_plain_int(value) and value >= 0
                     and value == expected,
                     "4B v2 a12_calibration fit counts not bound to the live policy: "
                     f"{key}={value!r} != {expected}")
        else:
            numeric = _finite_number(
                value, f"4B v2 a12_calibration fit_metrics.{key}")
            _require(0.0 <= numeric <= 1.0
                     and math.isclose(
                         numeric, expected, rel_tol=0.0, abs_tol=1e-15),
                     "4B v2 a12_calibration fit metrics not bound to the live policy: "
                     f"{key}={numeric!r} != {expected!r}")
    _require(
        observed["recall_12"] >= _A12_CALIBRATION_TRAIN_RECALL_TARGET
        and observed["precision_12"] >= max(0.10, _A12_PRECISION_MIN)
        and observed["fpr_12"] <= _A12_FPR_MAX * 0.5
        and _A12_PREDICTED_SHARE_MIN
        <= observed["predicted_share_12"]
        <= _A12_PREDICTED_SHARE_MAX
        and int(high_hp.sum()) > 0
        and observed["high_hp_false_drink_rate"]
        <= _A12_HIGH_HP_FALSE_DRINK_MAX * 0.5
        and observed["legal_negative_probability_12_mean"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
        and observed["legal_negative_probability_12_max"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX
        and observed["a13_spillover"] <= _A13_SPILLOVER_MAX,
        "4B v2 a12_calibration live fit safety gate not passed")


def bc_aux_behavior_metrics(
        policy_sd, x, y, episode_id, masks, *,
        anchor_sd=None, heldout_only: bool = True, circuit_spec=None) -> dict:
    """E5/a12 behaviour surface: held-out classification and starting-point drift on the real v2 masks.

    This is not a restatement of the training loss. precision/FPR/predicted share/high-HP false drinks catch the collapse
    "recall is high but it drinks everywhere"; a13 spillover and anchor TV/KL catch the side effect of crowding out
    the original resupply/engagement distribution in non-trigger states.
    """
    import numpy as np
    import torch as th

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    episode_id = np.asarray(episode_id)
    masks = np.asarray(masks)
    _require(x.ndim == 2 and y.shape == (len(x),)
             and episode_id.shape == (len(x),)
             and masks.shape == (len(x), 15)
             and masks.dtype == np.bool_,
             "E5 BC-v2 behaviour gate input shape/dtype invalid")
    indices = (_bc_v2_holdout_indices(episode_id) if heldout_only
               else np.arange(len(x), dtype=np.int64))
    hx, hy, hm = x[indices], y[indices], masks[indices]
    logits = _policy_logits_from_sb3_state_dict(
        policy_sd, hx, action_masks=hm, circuit_spec=circuit_spec)
    mask_t = th.as_tensor(hm, dtype=th.bool)
    masked = th.where(mask_t, logits, th.full_like(logits, -1e8))
    probs = th.softmax(masked, dim=-1)
    pred = probs.argmax(dim=-1).cpu().numpy()

    true12 = hy == 12
    pred12 = pred == 12
    all_negative = ~true12
    # The actionable FPR denominator can only be live non-12 states with m12=True. Mixing in structurally impossible m12=False states
    # such as an exhausted quota or an empty belt would, as the mask strengthens, artificially
    # dilute the FPR toward zero, hiding exactly the collapse "drink in every state where it can".
    legal_negative = all_negative & hm[:, 12]
    tp = int((pred12 & true12).sum())
    fp = int((pred12 & legal_negative).sum())
    fn = int((~pred12 & true12).sum())
    tn = int((~pred12 & legal_negative).sum())
    positive_denom = int(true12.sum())
    all_negative_denom = int(all_negative.sum())
    negative_denom = int(legal_negative.sum())
    predicted_positive = int(pred12.sum())
    high_hp_negative = legal_negative & (hx[:, 0] >= 0.65)
    high_hp_denom = int(high_hp_negative.sum())
    high_hp_fp = int((pred12 & high_hp_negative).sum())
    legal_negative_p12 = probs[:, 12].cpu().numpy()[legal_negative]
    eligible_p12 = probs[:, 12].cpu().numpy()[true12]
    # NumPy preserves float32 for mean/sum by default.  On a constant vector
    # its accumulated mean can round *above* the elementwise maximum, which
    # made the fail-closed order check reject a mathematically valid policy.
    # Accumulate audit scalars in float64 while preserving deployed float32
    # probabilities for min/max and argmax.
    legal_negative_p12_f64 = legal_negative_p12.astype(
        np.float64, copy=False)
    eligible_p12_f64 = eligible_p12.astype(np.float64, copy=False)
    all_p12_f64 = probs[:, 12].cpu().numpy().astype(
        np.float64, copy=False)
    non12_support = mask_t.clone()
    non12_support[:, _BC_AUX_CIRCUIT_ACTION] = False
    top_non12_probability = th.where(
        non12_support, probs, th.zeros_like(probs)
    ).max(dim=-1).values.cpu().numpy()
    a12_margins = (
        probs[:, _BC_AUX_CIRCUIT_ACTION].cpu().numpy()
        - top_non12_probability
    )
    predicted_episode_count = int(np.unique(
        episode_id[indices][pred12]).size)
    predicted_margin_min = (
        float(a12_margins[pred12].min()) if predicted_positive else 0.0)
    n = len(indices)

    pred13_share = float((pred == 13).mean()) if n else 0.0
    true13_share = float((hy == 13).mean()) if n else 0.0
    anchor = None
    if anchor_sd is not None:
        # The root anchor comes from protocol-v3 worker semantics. Only the anchor input is converted; the current
        # candidate contextual gate above still consumes the packed belt and the signed latch bit.
        anchor_logits = _policy_logits_from_sb3_state_dict(
            anchor_sd, hx, legacy_scene_clock=True)
        if circuit_spec == _bc_aux_circuit_spec():
            from leashed_ppo import (
                _legacy_distillation_masks,
                _masked_log_softmax_from_raw,
            )

            root_support = _legacy_distillation_masks(mask_t)
            anchor_logp = _masked_log_softmax_from_raw(
                anchor_logits, root_support)
            anchor_probs = anchor_logp.exp()
            current_root_logits = _policy_logits_from_sb3_state_dict(
                policy_sd,
                hx,
                action_masks=hm,
                circuit_spec=circuit_spec,
                return_raw_actor_logits=True,
            )
            current_root_logp = _masked_log_softmax_from_raw(
                current_root_logits, root_support)
            current_root_probs = current_root_logp.exp()
            current_root_pred = (
                current_root_probs.argmax(dim=-1).cpu().numpy())
        else:
            anchor_support = mask_t.clone()
            anchor_support[:, _BC_AUX_CIRCUIT_ACTION] = False
            anchor_masked = th.where(
                anchor_support, anchor_logits,
                th.full_like(anchor_logits, -1e8))
            anchor_probs = th.softmax(anchor_masked, dim=-1)
            anchor_logp = th.log(
                anchor_probs.clamp_min(th.finfo(anchor_probs.dtype).eps))
            current_root_probs = probs
            current_root_logp = th.log(
                probs.clamp_min(th.finfo(probs.dtype).eps))
            current_root_pred = pred
        anchor_pred = anchor_probs.argmax(dim=-1).cpu().numpy()
        critical_retention = {}
        for action in _BC_AUX_CRITICAL_ACTIONS:
            selected = anchor_pred == action
            support = int(selected.sum())
            retained = int(
                (selected & (current_root_pred == action)).sum())
            critical_retention[str(action)] = {
                "support": support,
                "retained": retained,
                "retention": (retained / support if support else None),
            }
        anchor = {
            "argmax_drift": float(
                (anchor_pred != current_root_pred).mean()),
            "tv_mean": float(
                (0.5 * (current_root_probs - anchor_probs)
                 .abs().sum(dim=-1)).mean()),
            "kl_anchor_to_policy": float(
                (anchor_probs * (
                    anchor_logp
                    - current_root_logp
                )).sum(dim=-1).mean().clamp_min(0.0)),
            "a12_probability_delta": float(
                (probs[:, 12] - anchor_probs[:, 12]).mean()),
            "a13_predicted_share": float((anchor_pred == 13).mean()),
            "critical_action_retention": critical_retention,
        }
        a13_reference = anchor["a13_predicted_share"]
        a13_reference_name = "anchor_argmax"
    else:
        a13_reference = true13_share
        a13_reference_name = "heldout_label"

    return {
        "scope": "heldout" if heldout_only else "full",
        "mask_mode": "bc-v2-recorded",
        "pairs": int(n),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "true_a12": positive_denom,
        "non_a12": negative_denom,
        "all_non_a12": all_negative_denom,
        "predicted_a12": predicted_positive,
        "predicted_a12_episodes": predicted_episode_count,
        "predicted_a12_margin_min": predicted_margin_min,
        "precision_12": (tp / predicted_positive
                         if predicted_positive else 0.0),
        "recall_12": tp / positive_denom if positive_denom else 0.0,
        "fpr_12": fp / negative_denom if negative_denom else 1.0,
        "predicted_share_12": predicted_positive / n if n else 0.0,
        "high_hp_non_a12": high_hp_denom,
        "high_hp_false_drinks": high_hp_fp,
        "high_hp_false_drink_rate": (
            high_hp_fp / high_hp_denom if high_hp_denom else 1.0),
        "eligible_probability_12_min": (
            float(eligible_p12.min()) if positive_denom else 0.0),
        "eligible_probability_12_mean": (
            float(eligible_p12_f64.mean()) if positive_denom else 0.0),
        "eligible_probability_12_max": (
            float(eligible_p12.max()) if positive_denom else 0.0),
        "legal_negative_probability_12_mean": (
            float(legal_negative_p12_f64.mean())
            if negative_denom else 1.0),
        "legal_negative_probability_12_max": (
            float(legal_negative_p12.max())
            if negative_denom else 1.0),
        "legal_negative_probability_12_sum": (
            float(legal_negative_p12_f64.sum())
            if negative_denom else float("inf")),
        "predicted_share_13": pred13_share,
        "true_share_13": true13_share,
        "a13_reference": a13_reference_name,
        "a13_reference_share": a13_reference,
        "a13_spillover": max(0.0, pred13_share - a13_reference),
        "mean_probability_12": float(all_p12_f64.mean()) if n else 0.0,
        "anchor": anchor,
    }


def bc_aux_behavior_gate(
        metrics: dict, *, require_root_anchor: bool = False,
        require_teacher_recall: bool = True,
        require_deployable_a12: bool = False) -> dict:
    """Production hard gate; returns structured reasons, and callers must not consume a bare PASS string."""
    reasons = []
    if not isinstance(metrics, dict) \
            or set(metrics) != set(_BC_AUX_BEHAVIOR_METRIC_KEYS):
        reasons.append("metric_schema_mismatch")
        metrics = metrics if isinstance(metrics, dict) else {}

    def plain_int(key):
        value = metrics.get(key)
        if not _is_plain_int(value) or value < 0:
            reasons.append(f"{key}_invalid")
            return None
        return int(value)

    def finite_number(key):
        value = metrics.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            reasons.append(f"{key}_invalid")
            return None
        return float(value)

    def at_least(key, threshold):
        value = finite_number(key)
        if value is None or value < threshold:
            reasons.append(f"{key}<{threshold}")

    def at_most(key, threshold):
        value = finite_number(key)
        if value is None or value > threshold:
            reasons.append(f"{key}>{threshold}")

    if metrics.get("scope") not in {"heldout", "full"}:
        reasons.append("scope_invalid")
    if metrics.get("mask_mode") != "bc-v2-recorded":
        reasons.append("mask_mode_invalid")
    count_keys = (
        "pairs", "tp", "fp", "fn", "tn", "true_a12", "non_a12",
        "all_non_a12", "predicted_a12", "predicted_a12_episodes",
        "high_hp_non_a12", "high_hp_false_drinks",
    )
    counts = {key: plain_int(key) for key in count_keys}
    if all(value is not None for value in counts.values()):
        if not (
            counts["pairs"] > 0
            and counts["tp"] + counts["fn"] == counts["true_a12"]
            and counts["fp"] + counts["tn"] == counts["non_a12"]
            and counts["true_a12"] + counts["all_non_a12"]
            == counts["pairs"]
            and counts["non_a12"] <= counts["all_non_a12"]
            and counts["predicted_a12"] == counts["tp"] + counts["fp"]
            and counts["predicted_a12_episodes"]
            <= counts["predicted_a12"]
            and counts["high_hp_false_drinks"]
            <= counts["high_hp_non_a12"]
            and (
                (counts["predicted_a12"] == 0
                 and counts["predicted_a12_episodes"] == 0)
                or (
                    counts["predicted_a12"] > 0
                    and 1 <= counts["predicted_a12_episodes"]
                    <= counts["predicted_a12"]
                )
            )
        ):
            reasons.append("count_closure_invalid")

    unit_interval_keys = (
        "precision_12", "recall_12", "fpr_12", "predicted_share_12",
        "high_hp_false_drink_rate", "predicted_share_13",
        "true_share_13", "eligible_probability_12_min",
        "eligible_probability_12_mean", "eligible_probability_12_max",
        "legal_negative_probability_12_mean",
        "legal_negative_probability_12_max", "a13_reference_share",
        "a13_spillover", "mean_probability_12",
    )
    numbers = {
        key: finite_number(key)
        for key in (
            *unit_interval_keys,
            "predicted_a12_margin_min",
            "legal_negative_probability_12_sum",
        )
    }
    for key in unit_interval_keys:
        value = numbers[key]
        if value is not None and not 0.0 <= value <= 1.0:
            reasons.append(f"{key}_out_of_range")
    margin = numbers["predicted_a12_margin_min"]
    if margin is not None and not -1.0 <= margin <= 1.0:
        reasons.append("predicted_a12_margin_min_out_of_range")
    if (
        margin is not None
        and counts.get("predicted_a12") is not None
        and (
            (counts["predicted_a12"] == 0 and margin != 0.0)
            or (counts["predicted_a12"] > 0 and margin < 0.0)
        )
    ):
        reasons.append("predicted_a12_margin_count_mismatch")
    probability_sum = numbers["legal_negative_probability_12_sum"]
    if probability_sum is not None and probability_sum < 0.0:
        reasons.append("legal_negative_probability_12_sum_out_of_range")
    if all(numbers.get(key) is not None for key in (
            "eligible_probability_12_min",
            "eligible_probability_12_mean",
            "eligible_probability_12_max")) and not (
        numbers["eligible_probability_12_min"]
        <= numbers["eligible_probability_12_mean"]
        <= numbers["eligible_probability_12_max"]
    ):
        reasons.append("eligible_probability_order_invalid")
    if all(numbers.get(key) is not None for key in (
            "legal_negative_probability_12_mean",
            "legal_negative_probability_12_max",
            "legal_negative_probability_12_sum")) and not (
        numbers["legal_negative_probability_12_mean"]
        <= numbers["legal_negative_probability_12_max"] + 1e-12
        and (
            counts.get("non_a12") is None
            or math.isclose(
                numbers["legal_negative_probability_12_sum"],
                numbers["legal_negative_probability_12_mean"]
                * counts["non_a12"],
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
        )
    ):
        reasons.append("legal_negative_probability_closure_invalid")
    if all(counts.get(key) is not None for key in (
            "pairs", "tp", "fp", "true_a12", "non_a12",
            "predicted_a12", "high_hp_non_a12",
            "high_hp_false_drinks")):
        expected_rates = {
            "precision_12": (
                counts["tp"] / counts["predicted_a12"]
                if counts["predicted_a12"] else 0.0),
            "recall_12": (
                counts["tp"] / counts["true_a12"]
                if counts["true_a12"] else 0.0),
            "fpr_12": (
                counts["fp"] / counts["non_a12"]
                if counts["non_a12"] else 1.0),
            "predicted_share_12":
                counts["predicted_a12"] / counts["pairs"],
            "high_hp_false_drink_rate": (
                counts["high_hp_false_drinks"]
                / counts["high_hp_non_a12"]
                if counts["high_hp_non_a12"] else 1.0),
        }
        for key, expected in expected_rates.items():
            value = numbers.get(key)
            if value is not None and not math.isclose(
                    value, expected, rel_tol=0.0, abs_tol=1e-12):
                reasons.append(f"{key}_count_mismatch")
    if all(
        counts.get(key) is not None
        for key in ("pairs", "true_a12", "non_a12")
    ) and all(
        numbers.get(key) is not None
        for key in (
            "eligible_probability_12_mean",
            "legal_negative_probability_12_sum",
            "mean_probability_12",
        )
    ) and counts["pairs"] > 0:
        # The recorded mask makes p12 exactly zero on the remaining
        # ``all_non_a12 - non_a12`` rows.  Bind the global probability mean
        # to the two non-zero domains instead of accepting an unrelated
        # self-reported scalar.
        expected_mean_probability = (
            numbers["eligible_probability_12_mean"]
            * counts["true_a12"]
            + numbers["legal_negative_probability_12_sum"]
        ) / counts["pairs"]
        if not math.isclose(
                numbers["mean_probability_12"],
                expected_mean_probability,
                rel_tol=1e-5, abs_tol=1e-7):
            reasons.append("mean_probability_12_closure_invalid")

    reference = metrics.get("a13_reference")
    reference_share = numbers.get("a13_reference_share")
    predicted_share_13 = numbers.get("predicted_share_13")
    spillover = numbers.get("a13_spillover")
    anchor = metrics.get("anchor")
    if anchor is None:
        if reference != "heldout_label":
            reasons.append("a13_reference_invalid")
        true_share = numbers.get("true_share_13")
        if (
            reference_share is not None
            and true_share is not None
            and not math.isclose(
                reference_share, true_share,
                rel_tol=0.0, abs_tol=1e-12)
        ):
            reasons.append("a13_reference_share_mismatch")
    else:
        if reference != "anchor_argmax":
            reasons.append("a13_reference_invalid")
        expected_anchor_keys = {
            "argmax_drift", "tv_mean", "kl_anchor_to_policy",
            "a12_probability_delta", "a13_predicted_share",
            "critical_action_retention",
        }
        if not isinstance(anchor, dict) or set(anchor) != expected_anchor_keys:
            reasons.append("root_anchor_schema_mismatch")
        else:
            anchor_a13 = anchor.get("a13_predicted_share")
            if (
                not isinstance(anchor_a13, (int, float))
                or isinstance(anchor_a13, bool)
                or not math.isfinite(float(anchor_a13))
                or not 0.0 <= float(anchor_a13) <= 1.0
            ):
                reasons.append("root_anchor.a13_predicted_share_invalid")
            elif (
                reference_share is not None
                and not math.isclose(
                    reference_share, float(anchor_a13),
                    rel_tol=0.0, abs_tol=1e-12)
            ):
                reasons.append("a13_reference_share_mismatch")
            delta = anchor.get("a12_probability_delta")
            if (
                not isinstance(delta, (int, float))
                or isinstance(delta, bool)
                or not math.isfinite(float(delta))
                or not -1.0 <= float(delta) <= 1.0
            ):
                reasons.append("root_anchor.a12_probability_delta_invalid")
    if (
        predicted_share_13 is not None
        and reference_share is not None
        and spillover is not None
        and not math.isclose(
            spillover,
            max(0.0, predicted_share_13 - reference_share),
            rel_tol=0.0, abs_tol=1e-12)
    ):
        reasons.append("a13_spillover_closure_invalid")

    at_least("true_a12", 1)
    at_least("non_a12", 1)
    at_least("all_non_a12", 1)
    at_least("high_hp_non_a12", 1)
    if require_teacher_recall:
        at_least("precision_12", _A12_PRECISION_MIN)
        at_least("recall_12", _A12_RECALL_MIN)
        at_least("predicted_share_12", _A12_PREDICTED_SHARE_MIN)
    elif isinstance(metrics.get("predicted_a12"), int) \
            and not isinstance(metrics.get("predicted_a12"), bool) \
            and metrics["predicted_a12"] > 0:
        # A learned argmax is allowed, but if it exists it must still be
        # concentrated on teacher-eligible states rather than global spill.
        at_least("precision_12", _A12_PRECISION_MIN)
    if require_deployable_a12:
        at_least(
            "predicted_a12_episodes",
            _BC_AUX_MIN_DETERMINISTIC_A12_EPISODES)
        at_least(
            "predicted_a12_margin_min",
            _BC_AUX_MIN_DETERMINISTIC_A12_MARGIN)
    at_most("fpr_12", _A12_FPR_MAX)
    at_most("predicted_share_12", _A12_PREDICTED_SHARE_MAX)
    at_most("high_hp_false_drink_rate", _A12_HIGH_HP_FALSE_DRINK_MAX)
    at_most(
        "legal_negative_probability_12_mean",
        _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX)
    at_most(
        "legal_negative_probability_12_max",
        _A12_LEGAL_NEGATIVE_PROBABILITY_MAX)
    at_most("a13_spillover", _A13_SPILLOVER_MAX)
    if require_root_anchor:
        if not isinstance(anchor, dict):
            reasons.append("root_anchor_missing")
        else:
            for key, threshold in (
                ("argmax_drift", _BC_AUX_ROOT_ARGMAX_DRIFT_MAX),
                ("tv_mean", _BC_AUX_ROOT_TV_MAX),
                ("kl_anchor_to_policy", _BC_AUX_ROOT_KL_MAX),
            ):
                value = anchor.get(key)
                if (not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                        or float(value) > threshold):
                    reasons.append(f"root_anchor.{key}>{threshold}")
            retention = anchor.get("critical_action_retention")
            if not isinstance(retention, dict):
                reasons.append("root_anchor.critical_action_retention_missing")
            else:
                for action in _BC_AUX_CRITICAL_ACTIONS:
                    row = retention.get(str(action))
                    # If an action has zero coverage in the root policy's held-out argmax there are no rows to keep,
                    # so skip it; with coverage keep at least half, so 9/10/13 cannot
                    # be wiped out as a whole class by a12 calibration or PPO updates.
                    if not isinstance(row, dict):
                        reasons.append(
                            f"root_anchor.action_{action}_retention_missing")
                        continue
                    support = row.get("support")
                    retained = row.get("retained")
                    value = row.get("retention")
                    if (not _is_plain_int(support) or support < 0
                            or not _is_plain_int(retained)
                            or not 0 <= retained <= support):
                        reasons.append(
                            f"root_anchor.action_{action}_support_invalid")
                    elif support > 0 and (
                            not isinstance(value, (int, float))
                            or isinstance(value, bool)
                            or not math.isfinite(float(value))
                            or not math.isclose(
                                float(value), retained / support,
                                rel_tol=0.0, abs_tol=1e-12)
                            or float(value)
                            < _BC_AUX_CRITICAL_RETENTION_MIN):
                        reasons.append(
                            f"root_anchor.action_{action}_retention"
                            f"<{_BC_AUX_CRITICAL_RETENTION_MIN}")
                    elif support == 0 and value is not None:
                        reasons.append(
                            f"root_anchor.action_{action}_zero_support_retention")
    # Keep structured receipts deterministic even when several closure checks
    # discover the same bad scalar through different paths.
    reasons = list(dict.fromkeys(reasons))
    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "thresholds": {
            "precision_12_min": _A12_PRECISION_MIN,
            "recall_12_min": (
                _A12_RECALL_MIN if require_teacher_recall else None),
            "teacher_recall_required": bool(require_teacher_recall),
            "deployable_a12_required": bool(require_deployable_a12),
            "deterministic_a12_episode_min": (
                _BC_AUX_MIN_DETERMINISTIC_A12_EPISODES
                if require_deployable_a12 else None),
            "deterministic_a12_margin_min": (
                _BC_AUX_MIN_DETERMINISTIC_A12_MARGIN
                if require_deployable_a12 else None),
            "fpr_12_max": _A12_FPR_MAX,
            "predicted_share_12_min": (
                _A12_PREDICTED_SHARE_MIN
                if require_teacher_recall else None),
            "predicted_share_12_max": _A12_PREDICTED_SHARE_MAX,
            "high_hp_false_drink_rate_max":
                _A12_HIGH_HP_FALSE_DRINK_MAX,
            "legal_negative_probability_12_mean_max":
                _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX,
            "legal_negative_probability_12_max":
                _A12_LEGAL_NEGATIVE_PROBABILITY_MAX,
            "a13_spillover_max": _A13_SPILLOVER_MAX,
            "root_anchor_required": bool(require_root_anchor),
            "root_argmax_drift_max": _BC_AUX_ROOT_ARGMAX_DRIFT_MAX,
            "root_tv_mean_max": _BC_AUX_ROOT_TV_MAX,
            "root_kl_anchor_to_policy_max": _BC_AUX_ROOT_KL_MAX,
            "root_critical_action_retention_min":
                _BC_AUX_CRITICAL_RETENTION_MIN,
        },
    }


def _policy_head_snapshot(policy) -> dict:
    """Freeze the current six SB3 policy-head tensors; used for the starting anchor and the final release behaviour gate."""
    state = policy.state_dict()
    _require(all(key in state for key in _POLICY_HEAD_KEYS),
             "policy state_dict is missing the six tensors required by the behaviour gate")
    return {
        key: state[key].detach().cpu().clone()
        for key in _POLICY_HEAD_KEYS
    }


def _policy_head_sha256(state: dict) -> str:
    """Content digest of the six tensors, stable across torch.save versions."""
    import numpy as np

    _require(isinstance(state, dict)
             and set(state) == set(_POLICY_HEAD_KEYS),
             "policy-head digest input key set not exact")
    digest = hashlib.sha256()
    for key in _POLICY_HEAD_KEYS:
        tensor = state[key].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _persistent_bc_aux_root_anchor(model) -> dict:
    """The first aux leg freezes the root anchor; later checkpoint continuations inherit the same anchor."""
    current = _policy_head_snapshot(model.policy)
    root = getattr(model, "bc_aux_root_anchor_sd", None)
    if root is None:
        root = current
    _require(isinstance(root, dict)
             and set(root) == set(_POLICY_HEAD_KEYS),
             "bc_aux root policy anchor fields in the checkpoint invalid")
    frozen = {}
    for key in _POLICY_HEAD_KEYS:
        value = root[key]
        _require(hasattr(value, "detach")
                 and bool(value.detach().isfinite().all().item()),
                 f"bc_aux root policy anchor tensor invalid: {key}")
        frozen[key] = value.detach().cpu().clone()
    # rev9 widens only the live actor.  Its persistent root intentionally
    # remains the exact 64-wide V28 head and therefore must not be forced to
    # match the candidate's 68-wide shapes.
    w0, b0, w1, b1, wa, ba = (
        frozen[key] for key in _POLICY_HEAD_KEYS)
    # The root anchor is self-describing.  Do not require a fully constructed
    # SB3 model here: publication tests and offline gate tooling deliberately
    # use a minimal policy carrier with no observation/action spaces.
    obs_dim = int(w0.shape[1]) if w0.ndim == 2 else -1
    action_dim = int(wa.shape[0]) if wa.ndim == 2 else -1
    _require(
        w0.ndim == 2 and w0.shape[1] == obs_dim
        and w0.shape[0] == _BC_AUX_CIRCUIT_BASE_WIDTH
        and b0.shape == (w0.shape[0],)
        and w1.shape == (w0.shape[0], w0.shape[0])
        and b1.shape == (w0.shape[0],)
        and wa.shape == (action_dim, w0.shape[0])
        and ba.shape == (action_dim,),
        "bc_aux root policy anchor must be a 298->64->64->15 pre-adapter actor; "
        "when an existing adapter lacks a root, silently re-anchoring from the 68-wide current head is forbidden",
    )
    current_w0 = current["mlp_extractor.policy_net.0.weight"]
    current_wa = current["action_net.weight"]
    _require(
        current_w0.ndim == 2 and int(current_w0.shape[1]) == obs_dim
        and current_wa.ndim == 2 and int(current_wa.shape[0]) == action_dim,
        "bc_aux current policy and root anchor input/action dimensions disagree",
    )
    model.bc_aux_root_anchor_sd = {
        key: value.clone() for key, value in frozen.items()
    }
    return frozen


def _bc_aux_circuit_spec() -> dict:
    """Canonical identity of the rev9 contextual exact-mixture adapter."""
    return {
        "schema_version": _BC_AUX_CIRCUIT_SCHEMA,
        "base_width": _BC_AUX_CIRCUIT_BASE_WIDTH,
        "expanded_width": _BC_AUX_CIRCUIT_EXPANDED_WIDTH,
        "action_index": _BC_AUX_CIRCUIT_ACTION,
        "gate_feature_indices":
            list(_BC_AUX_CIRCUIT_GATE_FEATURE_INDICES),
        "gate_parameter_columns":
            list(_BC_AUX_CIRCUIT_GATE_PARAMETER_COLUMNS),
        "hp_low": 0.5,
        "hp_high": 0.65,
        "boundary_epsilon": _A12_VISIBLE_HP_BOUNDARY_EPS,
        "initial_probability": _BC_AUX_CIRCUIT_INITIAL_PROBABILITY,
        "initial_gate_bias": _BC_AUX_CIRCUIT_INITIAL_GATE_BIAS,
        "probability_min": _BC_AUX_CIRCUIT_PROBABILITY_MIN,
        "probability_max": _BC_AUX_CIRCUIT_PROBABILITY_MAX,
        "gate_parameter_abs_max":
            _BC_AUX_CIRCUIT_GATE_PARAMETER_ABS_MAX,
    }


def _expand_policy_with_bc_aux_circuit(model) -> dict:
    """Losslessly widen V28 and install the exact distribution adapter.

    The value branch is untouched.  The old 64x64 actor block and all fourteen
    non-a12 output rows are copied bitwise; every new/cross block stays zero.
    Four action-head cells store coefficients over stable raw features and
    action12's bias stores their intercept.  The policy distribution maps that
    five-parameter score to ε(s); no a12 gradient enters the legacy actor.
    The caller must reset the optimizer after this topology/class change.
    """
    import torch as th

    policy = model.policy
    net = policy.mlp_extractor.policy_net
    _require(len(net) >= 4
             and isinstance(net[0], th.nn.Linear)
             and isinstance(net[2], th.nn.Linear)
             and isinstance(policy.action_net, th.nn.Linear),
             "a12 circuit only supports a standard two-layer MlpPolicy actor")
    expected_spec = _bc_aux_circuit_spec()
    existing_spec = getattr(model, "_bc_aux_circuit_spec", None)
    if existing_spec is not None:
        from leashed_ppo import A12MixtureMaskableActorCriticPolicy
        _require(existing_spec == expected_spec,
                 "checkpoint a12 adapter spec disagrees with rev9")
        _require(
            isinstance(policy, A12MixtureMaskableActorCriticPolicy)
            and model.policy_class is A12MixtureMaskableActorCriticPolicy
            and policy.bc_aux_mixture_spec == expected_spec
            and model.policy_kwargs.get("bc_aux_mixture_spec")
            == expected_spec
            and
            net[0].weight.shape
            == (_BC_AUX_CIRCUIT_EXPANDED_WIDTH,
                int(model.observation_space.shape[0]))
            and net[2].weight.shape
            == (_BC_AUX_CIRCUIT_EXPANDED_WIDTH,
                _BC_AUX_CIRCUIT_EXPANDED_WIDTH)
            and policy.action_net.weight.shape
            == (int(model.action_space.n),
                _BC_AUX_CIRCUIT_EXPANDED_WIDTH),
            "checkpoint a12 circuit topology disagrees with the spec")
        return expected_spec

    old0, old1, olda = net[0], net[2], policy.action_net
    base = _BC_AUX_CIRCUIT_BASE_WIDTH
    width = _BC_AUX_CIRCUIT_EXPANDED_WIDTH
    obs_dim = int(model.observation_space.shape[0])
    action_dim = int(model.action_space.n)
    _require(
        old0.weight.shape == (base, obs_dim)
        and old0.bias.shape == (base,)
        and old1.weight.shape == (base, base)
        and old1.bias.shape == (base,)
        and olda.weight.shape == (action_dim, base)
        and olda.bias.shape == (action_dim,)
        and action_dim > _BC_AUX_CIRCUIT_ACTION
        and obs_dim > _A12_CALIBRATION_DRINK_LATCH_FEATURE,
        "a12 circuit may only migrate from the frozen 298->64->64->15 V28 actor")
    root = _persistent_bc_aux_root_anchor(model)
    _require(_policy_head_sha256(root)
             == _policy_head_sha256(_policy_head_snapshot(policy)),
             "before the first a12 circuit migration the root must equal the current V28 actor bit for bit")

    device, dtype = old0.weight.device, old0.weight.dtype
    new0 = th.nn.Linear(
        obs_dim, width, bias=True, device=device, dtype=dtype)
    new1 = th.nn.Linear(
        width, width, bias=True, device=device, dtype=dtype)
    newa = th.nn.Linear(
        width, action_dim, bias=True, device=device, dtype=dtype)
    with th.no_grad():
        new0.weight.zero_()
        new0.bias.zero_()
        new1.weight.zero_()
        new1.bias.zero_()
        newa.weight.zero_()
        newa.bias.copy_(olda.bias)
        new0.weight[:base].copy_(old0.weight)
        new0.bias[:base].copy_(old0.bias)
        new1.weight[:base, :base].copy_(old1.weight)
        new1.bias[:base].copy_(old1.bias)
        newa.weight[:, :base].copy_(olda.weight)

        # a12 was permanently masked during V28/KING training.  Its old row is
        # untrained and is ignored by the contextual gate; zero it so no
        # alternate latent path can masquerade as a registered raw feature.
        newa.weight[_BC_AUX_CIRCUIT_ACTION].zero_()
        # Zero coefficients + this affine-sigmoid intercept produce exactly
        # 5% in every eligible state.  The four coefficients occupy the
        # otherwise-unused expanded columns and are consumed directly from raw
        # observations by the custom distribution, not via zero latent units.
        newa.bias[_BC_AUX_CIRCUIT_ACTION] = (
            _BC_AUX_CIRCUIT_INITIAL_GATE_BIAS)

    new0.train(old0.training)
    new1.train(old1.training)
    newa.train(olda.training)
    net[0] = new0
    net[2] = new1
    policy.action_net = newa
    policy.mlp_extractor.latent_dim_pi = width
    net_arch = {"pi": [width, width], "vf": [64, 64]}
    policy.net_arch = net_arch
    model.policy_kwargs = dict(getattr(model, "policy_kwargs", {}) or {})
    model.policy_kwargs["net_arch"] = net_arch
    model.policy_kwargs["bc_aux_mixture_spec"] = expected_spec
    from leashed_ppo import A12MixtureMaskableActorCriticPolicy
    # Both are pure-Python nn.Module classes with identical storage layout;
    # changing the behavior class preserves the live tensors just copied.
    policy.__class__ = A12MixtureMaskableActorCriticPolicy
    policy.bc_aux_mixture_spec = expected_spec
    model.policy_class = A12MixtureMaskableActorCriticPolicy
    model._bc_aux_circuit_spec = expected_spec
    return expected_spec


def _calibrate_bc_aux_adapter_weight(
        model, x, y, episode_id, masks) -> dict:
    """Initialize the exact on-policy mixture and verify nested safety.

    No data-dependent parameter fitting remains: ε is the preregistered 5% in
    every eligible state.  Fit/validation data are consumed only to prove that
    the visible predicate, masks, old-policy argmax and distribution identity
    agree.  The final held-out split is not read here.
    """
    import numpy as np
    import torch as th

    spec = _bc_aux_circuit_spec()
    _require(getattr(model, "_bc_aux_circuit_spec", None) == spec,
             "a12 adapter spec missing/drifted before calibration")
    from leashed_ppo import A12MixtureMaskableActorCriticPolicy
    _require(
        isinstance(model.policy, A12MixtureMaskableActorCriticPolicy)
        and model.policy.bc_aux_mixture_spec == spec
        and model.policy_class is A12MixtureMaskableActorCriticPolicy,
        "a12 mixture policy class/spec not installed atomically")
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(episode_id)
    masks = np.asarray(masks)
    _require(
        x.ndim == 2 and x.shape[1] == 298
        and y.shape == groups.shape == (len(x),)
        and masks.shape == (len(x), 15)
        and masks.dtype == np.bool_,
        "a12 adapter calibration input shape/dtype invalid")
    fit, validation = _bc_v2_fit_validation_indices(groups)
    visible_eligible = (
        (x[:, 0] >= spec["hp_low"] - spec["boundary_epsilon"])
        & (x[:, 0] < spec["hp_high"] - spec["boundary_epsilon"])
        & (x[:, _A12_CALIBRATION_DRINK_LATCH_FEATURE] >= 0.0)
        & masks[:, _BC_AUX_CIRCUIT_ACTION]
    )
    _require(
        np.array_equal(y == _BC_AUX_CIRCUIT_ACTION, visible_eligible),
        "a12 mixture data labels differ from the visible hp/latch/m12 predicate; "
        "refusing to initialise from old-semantics demos")
    positive_fit = fit[y[fit] == _BC_AUX_CIRCUIT_ACTION]
    _require(len(positive_fit) >= 2
             and bool(masks[positive_fit, _BC_AUX_CIRCUIT_ACTION].all()),
             "a12 adapter nested-fit positives insufficient/invalid")
    action = _BC_AUX_CIRCUIT_ACTION
    parameter_columns = tuple(
        int(value) for value in spec["gate_parameter_columns"])
    action_bias = model.policy.action_net.bias
    adapter_weight = model.policy.action_net.weight
    _require(
        bool((adapter_weight[
            action, list(parameter_columns)] == 0).all().item())
        and math.isclose(
            float(action_bias[action].detach().cpu()),
            float(spec["initial_gate_bias"]),
            rel_tol=0.0,
            abs_tol=2e-7,
        ),
        "a12 contextual gate initial coefficients/bias have drifted")

    device = model.device

    def positive_probability_summary(indices) -> dict:
        with th.no_grad():
            dist = model.policy.get_distribution(
                th.as_tensor(x[indices], device=device),
                action_masks=th.as_tensor(masks[indices], device=device))
            probabilities = dist.distribution.logits[:, action].exp()
            return {
                "min": float(probabilities.min().cpu()),
                "mean": float(probabilities.mean().cpu()),
                "max": float(probabilities.max().cpu()),
                "predicted_a12": int(
                    (dist.distribution.logits.argmax(dim=-1) == action)
                    .sum().cpu()),
            }

    target_probability = float(spec["initial_probability"])

    fit_positive_summary = positive_probability_summary(positive_fit)
    fit_positive_probability = fit_positive_summary["mean"]
    positive_validation = validation[
        y[validation] == _BC_AUX_CIRCUIT_ACTION]
    _require(len(positive_validation) >= 1,
             "a12 adapter nested-validation has no positives")
    validation_positive_summary = positive_probability_summary(
        positive_validation)
    validation_positive_probability = validation_positive_summary["mean"]
    candidate = _policy_head_snapshot(model.policy)
    fit_metrics = bc_aux_behavior_metrics(
        candidate, x[fit], y[fit], groups[fit], masks[fit],
        anchor_sd=model.bc_aux_root_anchor_sd, heldout_only=False,
        circuit_spec=spec)

    _require(
        math.isclose(
            fit_positive_probability, target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            fit_positive_summary["min"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            fit_positive_summary["max"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and fit_positive_summary["predicted_a12"] == 0
        and fit_metrics["predicted_a12"] == 0
        and fit_metrics["anchor"]["argmax_drift"] == 0.0
        and fit_metrics["fpr_12"] == 0.0
        and fit_metrics["high_hp_false_drink_rate"]
        == 0.0
        and fit_metrics["legal_negative_probability_12_mean"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MEAN_MAX
        and fit_metrics["legal_negative_probability_12_max"]
        <= _A12_LEGAL_NEGATIVE_PROBABILITY_MAX,
        "a12 adapter nested-fit initial exploration/safety gate not passed")
    validation_metrics = bc_aux_behavior_metrics(
        candidate, x[validation], y[validation],
        groups[validation], masks[validation],
        anchor_sd=model.bc_aux_root_anchor_sd, heldout_only=False,
        circuit_spec=spec)
    validation_gate = bc_aux_behavior_gate(
        validation_metrics, require_root_anchor=True,
        require_teacher_recall=False)
    _require(validation_gate["verdict"] == "PASS",
             "a12 adapter nested-validation safety gate not passed: "
             f"{validation_gate['reasons']}")
    _require(
        math.isclose(
            validation_positive_probability, target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            validation_positive_summary["min"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and math.isclose(
            validation_positive_summary["max"], target_probability,
            rel_tol=0.0, abs_tol=5e-7)
        and validation_positive_summary["predicted_a12"] == 0
        and validation_metrics["predicted_a12"] == 0
        and validation_metrics["anchor"]["argmax_drift"] == 0.0,
        "a12 adapter nested-validation exploration probability/argmax invalid")
    return {
        "fit_pairs": int(len(fit)),
        "validation_pairs": int(len(validation)),
        "fit_positive_a12": int(len(positive_fit)),
        "validation_positive_a12": int(len(positive_validation)),
        "initializer": "exact-contextual-legal-support-mixture",
        "gate_feature_indices": list(spec["gate_feature_indices"]),
        "gate_parameter_columns":
            list(spec["gate_parameter_columns"]),
        "target_probability_12": target_probability,
        "fit_positive_probability_min_12":
            fit_positive_summary["min"],
        "fit_positive_probability_12": fit_positive_probability,
        "fit_positive_probability_max_12":
            fit_positive_summary["max"],
        "validation_positive_probability_min_12":
            validation_positive_summary["min"],
        "validation_positive_probability_12":
            validation_positive_probability,
        "validation_positive_probability_max_12":
            validation_positive_summary["max"],
        "initial_argmax_lower_bound": (
            (1.0 - target_probability) / 14.0
            - target_probability),
        "initial_gate_bias": float(
            action_bias[action].detach().cpu()),
        "gate_coefficients": [
            float(value)
            for value in adapter_weight[
                action, list(parameter_columns)].detach().cpu()
        ],
        "probability_min": float(spec["probability_min"]),
        "probability_max": float(spec["probability_max"]),
        "fit_metrics": fit_metrics,
        "validation_metrics": validation_metrics,
        "validation_gate": validation_gate,
        "candidate_policy_head_sha256":
            _policy_head_sha256(candidate),
    }


def _bc_aux_liveness_call_plan(total_steps: int, n_steps: int,
                               num_envs: int) -> dict:
    """Convert the production leg budget into ``train()``/aux call counts, so a hand-written 244 cannot drift."""
    quantum = int(n_steps) * int(num_envs)
    _require(quantum > 0 and int(total_steps) > 0
             and int(total_steps) % quantum == 0,
             "bc_aux liveness budget must divide by the rollout quantum")
    train_calls = int(total_steps) // quantum
    _require(_BC_AUX_UPDATE_EVERY >= 1,
             "bc_aux update_every must be a positive integer")
    aux_calls = len(range(0, train_calls, _BC_AUX_UPDATE_EVERY))
    return {
        "rollout_quantum": quantum,
        "train_calls": train_calls,
        "aux_optimizer_calls": aux_calls,
        "update_every": _BC_AUX_UPDATE_EVERY,
    }


def _simulate_bc_aux_liveness(
        model, *, bank, x, y, episode_id, masks,
        bc_aux_lambda: float, seed: int | None,
        call_plan: dict) -> dict:
    """Run the production number of aux calls on an isolated clone, gating only on training episodes.

    This is a necessary-condition probe before the environment launches: it keeps the real policy, Adam
    moments, lr, batch size and persistent root from the checkpoint; the only thing removed is the PPO/distillation
    gradient. The probe never carries simulated weights back into the production model and never reads held-out rows.
    """
    import numpy as np
    from leashed_ppo import derive_bc_aux_rng

    _require(math.isfinite(bc_aux_lambda) and bc_aux_lambda > 0,
             "bc_aux liveness λ must be a finite positive number")
    _require(isinstance(call_plan, dict)
             and call_plan.get("update_every") == _BC_AUX_UPDATE_EVERY,
             "bc_aux liveness call plan disagrees with the current objective")
    train_calls = int(call_plan.get("train_calls", -1))
    expected_aux_calls = int(call_plan.get("aux_optimizer_calls", -1))
    _require(train_calls > 0 and expected_aux_calls > 0,
             "bc_aux liveness call plan is empty")

    bx, by, bm = bank
    root = _persistent_bc_aux_root_anchor(model)
    start = _policy_head_snapshot(model.policy)
    model.bc_aux_lambda = float(bc_aux_lambda)
    model.mount_bc_aux_demos(
        bx, by, bm, rng=derive_bc_aux_rng(seed))
    optimizer = model.policy.optimizer
    optimizer_state_entries = len(optimizer.state)
    optimizer_lrs = [
        float(group["lr"]) for group in optimizer.param_groups]
    _require(all(math.isfinite(value) and value > 0
                 for value in optimizer_lrs),
             "bc_aux liveness optimizer lr invalid")

    applied = 0
    last_loss = None
    for call_index in range(train_calls):
        # Mirrors the call count and due check of LeashedMaskablePPO.train(); with the current rev5
        # update_every=1, production 499,712/2,048 therefore gives exactly 244.
        model._bc_aux_train_calls += 1
        if call_index % _BC_AUX_UPDATE_EVERY != 0:
            continue
        aux_loss = model._apply_bc_aux_step()
        applied += 1
        last_loss = float(aux_loss.detach().cpu())
    _require(applied == expected_aux_calls,
             "bc_aux liveness actual aux call count disagrees with the production plan: "
             f"{applied} != {expected_aux_calls}")

    groups = np.asarray(episode_id)
    training = _bc_v2_training_indices(groups)
    heldout = _bc_v2_holdout_indices(groups)
    train_episodes = np.unique(groups[training])
    heldout_episodes = np.unique(groups[heldout])
    _require(not np.intersect1d(
        train_episodes, heldout_episodes).size,
        "bc_aux liveness training/heldout episode leak")
    candidate = _policy_head_snapshot(model.policy)
    metrics = bc_aux_behavior_metrics(
        candidate, np.asarray(x)[training], np.asarray(y)[training],
        groups[training], np.asarray(masks)[training],
        anchor_sd=root, heldout_only=False)
    gate = bc_aux_behavior_gate(
        metrics, require_root_anchor=True)
    episode_digest = hashlib.sha256(
        np.asarray(train_episodes, dtype=np.int64).tobytes()).hexdigest()
    return {
        "status": "PASS" if gate["verdict"] == "PASS" else "FAIL",
        "simulation": "isolated-aux-only-necessary-condition",
        "evaluation_scope": "bc-v2-training-episodes-only",
        "heldout_rows_consumed": 0,
        "split": {
            "training_pairs": int(len(training)),
            "training_episodes": int(len(train_episodes)),
            "training_episode_ids_sha256": episode_digest,
            "heldout_pairs_excluded": int(len(heldout)),
            "heldout_episodes_excluded": int(len(heldout_episodes)),
            "episode_disjoint": True,
        },
        "bank": {
            "pairs": int(len(by)),
            "true_a12": int((np.asarray(by) == 12).sum()),
            "hard_negatives": int((np.asarray(by) != 12).sum()),
        },
        "optimizer": {
            "class": (f"{type(optimizer).__module__}."
                      f"{type(optimizer).__qualname__}"),
            "state_entries_at_start": int(optimizer_state_entries),
            "learning_rates_at_start": optimizer_lrs,
            "max_grad_norm": float(model.max_grad_norm),
        },
        "policy": {
            "start_head_sha256": _policy_head_sha256(start),
            "root_head_sha256": _policy_head_sha256(root),
            "simulated_end_head_sha256": _policy_head_sha256(candidate),
        },
        "calls": {**call_plan, "actual_aux_optimizer_calls": applied},
        "last_unscaled_aux_loss": last_loss,
        "metrics": metrics,
        "gate": gate,
    }


def _run_bc_aux_policy_gradient_canary(
        model, *, x, y, episode_id, masks, spec: dict) -> dict:
    """Prove the real distribution/optimizer path can raise eligible p(a12).

    The canary uses nested-validation positives only, applies one genuine
    optimizer step through ``get_distribution().log_prob()``, and then restores
    both policy and optimizer bit-for-bit.  It is a wiring/learnability
    necessary condition, not evidence that the live environment will assign a
    positive advantage to drinking.
    """
    import numpy as np
    import torch as th

    groups = np.asarray(episode_id)
    labels = np.asarray(y)
    observations = np.asarray(x, dtype=np.float32)
    action_masks = np.asarray(masks)
    _, validation = _bc_v2_fit_validation_indices(groups)
    action = int(spec["action_index"])
    positive = validation[labels[validation] == action]
    _require(
        len(positive) > 0
        and bool(action_masks[positive, action].all()),
        "a12 policy-gradient canary lacks legal nested-validation positives")

    # Keep the probe compact and deterministic while spanning more than one
    # episode whenever the validation split permits it.
    positive = positive[:min(256, len(positive))]
    obs_t = th.as_tensor(observations[positive], device=model.device)
    masks_t = th.as_tensor(
        action_masks[positive], dtype=th.bool, device=model.device)
    actions_t = th.full(
        (len(positive),), action, dtype=th.long, device=model.device)
    policy_state = {
        key: value.detach().clone()
        for key, value in model.policy.state_dict().items()
    }
    optimizer = model.policy.optimizer
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    was_training = bool(model.policy.training)
    start_head = _policy_head_snapshot(model.policy)
    gate_bias_parameter = model.policy.action_net.bias
    restored = False
    try:
        model.policy.set_training_mode(True)
        with th.no_grad():
            before_distribution = model.policy.get_distribution(
                obs_t, action_masks=masks_t)
            before_probability = float(
                before_distribution.distribution.logits[:, action]
                .exp().mean().cpu())
            bias_before = float(
                gate_bias_parameter[action].detach().cpu())

        optimizer.zero_grad()
        distribution = model.policy.get_distribution(
            obs_t, action_masks=masks_t)
        loss = -distribution.log_prob(actions_t).mean()
        _require(bool(th.isfinite(loss).item()),
                 "a12 policy-gradient canary loss not finite")
        loss.backward()
        gradient = gate_bias_parameter.grad
        _require(
            gradient is not None
            and gradient.shape == gate_bias_parameter.shape
            and bool(th.isfinite(gradient).all().item()),
            "a12 policy-gradient canary gate bias gradient missing/not finite")
        bias_gradient = float(gradient[action].detach().cpu())
        _require(
            bias_gradient < 0.0,
            "a12 policy-gradient canary: a favourable advantage produced no gradient that raises the gate")
        circuit_snapshot = model._protect_bc_aux_circuit_before_step()
        gradient_norm = float(th.nn.utils.clip_grad_norm_(
            model.policy.parameters(), model.max_grad_norm).detach().cpu())
        _require(math.isfinite(gradient_norm) and gradient_norm > 0.0,
                 "a12 policy-gradient canary gradient norm invalid")
        optimizer.step()
        model._project_bc_aux_adapter_weight()
        model._assert_bc_aux_circuit_unchanged(circuit_snapshot)

        with th.no_grad():
            after_distribution = model.policy.get_distribution(
                obs_t, action_masks=masks_t)
            after_probability = float(
                after_distribution.distribution.logits[:, action]
                .exp().mean().cpu())
            bias_after = float(
                gate_bias_parameter[action].detach().cpu())
        probability_delta = after_probability - before_probability
        bias_delta = bias_after - bias_before
        movement_required = (
            before_probability
            < float(spec["probability_max"]) - 1e-6)
        _require(
            after_probability >= before_probability
            and (not movement_required or probability_delta > 0.0)
            and (not movement_required or bias_delta > 0.0),
            "a12 policy-gradient canary optimizer step did not raise eligible p(a12)")
        end_head = _policy_head_snapshot(model.policy)
        return {
            "schema_version": "a12-policy-gradient-canary/1",
            "scope": "bc-v2-nested-validation-positive-only",
            "pairs": int(len(positive)),
            "heldout_rows_consumed": 0,
            "objective": "negative-mean-log-probability-action12",
            "optimizer_steps": 1,
            "movement_required": movement_required,
            "probability_12_before": before_probability,
            "probability_12_after": after_probability,
            "probability_12_delta": probability_delta,
            "gate_bias_before": bias_before,
            "gate_bias_after": bias_after,
            "gate_bias_delta": bias_delta,
            "gate_bias_gradient": bias_gradient,
            "gradient_norm_before_clip": gradient_norm,
            "start_policy_head_sha256": _policy_head_sha256(start_head),
            "stepped_policy_head_sha256": _policy_head_sha256(end_head),
            "state_restored": True,
        }
    finally:
        model.policy.load_state_dict(policy_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        model.policy.set_training_mode(was_training)
        restored = all(
            th.equal(model.policy.state_dict()[key], value)
            for key, value in policy_state.items())
        if not restored:
            raise RuntimeError(
                "a12 policy-gradient canary did not restore the isolated policy state")


def _simulate_bc_aux_circuit_liveness(
        model, *, x, y, episode_id, masks,
        learning_rate: float, call_plan: dict) -> dict:
    """Install/gate the exact rev9 contextual on-policy mixture adapter."""
    start = _policy_head_snapshot(model.policy)
    root = _persistent_bc_aux_root_anchor(model)
    existing = getattr(model, "_bc_aux_circuit_spec", None) is not None
    if not existing:
        _require(_policy_head_sha256(start) == _policy_head_sha256(root),
                 "rev9 first-install starting point is not the unmodified V28 root")
    spec = _expand_policy_with_bc_aux_circuit(model)
    if existing:
        _require(not callable(model.learning_rate)
                 and math.isclose(
                     float(model.learning_rate), learning_rate,
                     rel_tol=0.0, abs_tol=1e-12),
                 "rev9 continuation learning rate drift")
        frozen_values = []
        for parameter, protected in (
                model._bc_aux_circuit_protected_tensors()):
            frozen_values.append(parameter.detach()[protected])
            state = model.policy.optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                moment = state.get(key)
                if moment is not None:
                    _require(
                        moment.shape == parameter.shape
                        and bool((moment[protected] == 0).all().item()),
                        f"rev9 continuation protected Adam {key} nonzero/wrong shape")
        _require(
            frozen_values
            and all(bool((value == 0).all().item())
                    for value in frozen_values),
            "rev9 continuation fixed adapter tensors are no longer canonical zero")
        action = int(spec["action_index"])
        columns = [
            int(value) for value in spec["gate_parameter_columns"]]
        gate_coefficients = model.policy.action_net.weight[
            action, columns].detach()
        gate_bias = model.policy.action_net.bias[action].detach()
        limit = float(spec["gate_parameter_abs_max"])
        _require(
            bool(gate_coefficients.isfinite().all().item())
            and bool(gate_bias.isfinite().item())
            and bool((gate_coefficients.abs() <= limit).all().item())
            and abs(float(gate_bias.cpu())) <= limit,
            "rev9 continuation contextual gate not finite/out of range")
        fit, validation = _bc_v2_fit_validation_indices(episode_id)
        candidate = _policy_head_snapshot(model.policy)
        metrics = bc_aux_behavior_metrics(
            candidate,
            x[validation], y[validation], episode_id[validation],
            masks[validation],
            anchor_sd=model.bc_aux_root_anchor_sd,
            heldout_only=False,
            circuit_spec=spec,
        )
        gate = bc_aux_behavior_gate(
            metrics, require_root_anchor=True,
            require_teacher_recall=False)
        calibration = {
            "initializer": "preserved-continuation",
            "gate_coefficients": [
                float(value) for value in gate_coefficients.cpu()],
            "gate_bias": float(gate_bias.cpu()),
            "fit_pairs_excluded_from_retuning": int(len(fit)),
            "validation_pairs": int(len(validation)),
            "validation_metrics": metrics,
            "validation_gate": gate,
            "candidate_policy_head_sha256":
                _policy_head_sha256(candidate),
        }
    else:
        _reset_policy_optimizer(model, learning_rate)
        calibration = _calibrate_bc_aux_adapter_weight(
            model, x, y, episode_id, masks)
    policy_gradient_canary = _run_bc_aux_policy_gradient_canary(
        model, x=x, y=y, episode_id=episode_id, masks=masks, spec=spec)
    candidate = _policy_head_snapshot(model.policy)
    # Calibration already gates nested validation.  This duplicate explicit
    # verdict keeps the preflight receipt easy to verify without executing
    # code from an untrusted JSON field.
    gate = calibration["validation_gate"]
    return {
        "status": "PASS" if gate["verdict"] == "PASS" else "FAIL",
        "simulation":
            "isolated-exact-mixture-with-policy-gradient-canary",
        "installation": (
            "preserved-continuation" if existing else "first-install"),
        "evaluation_scope": "bc-v2-nested-validation-only",
        "heldout_rows_consumed": 0,
        "circuit": {
            **spec,
            "king_support": _BC_AUX_CIRCUIT_KING_SUPPORT,
        },
        "optimizer": {
            "class": (
                f"{type(model.policy.optimizer).__module__}."
                f"{type(model.policy.optimizer).__qualname__}"),
            "state_entries_at_start": int(
                len(model.policy.optimizer.state)),
            "learning_rates_at_start": [
                float(group["lr"])
                for group in model.policy.optimizer.param_groups
            ],
            "reset_after_topology_change": not existing,
        },
        "policy": {
            "start_head_sha256": _policy_head_sha256(start),
            "root_head_sha256": _policy_head_sha256(root),
            "grafted_head_sha256": _policy_head_sha256(candidate),
            "actor_width_before": (
                _BC_AUX_CIRCUIT_EXPANDED_WIDTH if existing
                else _BC_AUX_CIRCUIT_BASE_WIDTH),
            "actor_width_after": _BC_AUX_CIRCUIT_EXPANDED_WIDTH,
        },
        "calls": {
            "planned_train_calls": int(call_plan["train_calls"]),
            "aux_optimizer_calls": 0,
            "policy_gradient_canary_calls": 1,
            "initial_adapter_calibrations": 0 if existing else 1,
            "trainable_adapter_parameters": 5,
        },
        "policy_gradient_canary": policy_gradient_canary,
        "calibration": calibration,
        "metrics": calibration["validation_metrics"],
        "gate": gate,
    }


def _write_bc_aux_behavior_receipt(path: pathlib.Path, record: dict) -> None:
    """FAIL also writes its receipt atomically; the canonical model is published only after PASS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            allow_nan=False).encode("utf-8")
        with open(tmp, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_PUBLICATION_PROVENANCE_KEYS = frozenset({
    "protocol_version", "implementation_sha256", "manager_npz_sha256",
    "resume_checkpoint_sha256", "teacher_sha256",
    "bc_aux_demos_sha256", "bc_aux_liveness_preflight_sha256",
    "training_contract_sha256", "start_steps", "target_global_steps",
    "seed", "optimizer_reset", "target_kl", "distill_beta",
    "bc_aux_lambda", "bc_aux_mode", "calib_record_only",
})


def _validate_publication_provenance(
        provenance: dict, *, demos_sha256: str, final_step: int) -> dict:
    """Strictly freeze the training lineage of the final model, for re-verification at the evaluation entry point."""
    _require(isinstance(provenance, dict)
             and set(provenance) == _PUBLICATION_PROVENANCE_KEYS,
             "bc_aux final release lineage fields incomplete")
    _require(provenance["protocol_version"] == PROTOCOL_VERSION,
             "bc_aux final release lineage protocol mismatch")
    for key in (
            "implementation_sha256", "manager_npz_sha256",
            "resume_checkpoint_sha256", "teacher_sha256",
            "bc_aux_demos_sha256", "bc_aux_liveness_preflight_sha256",
            "training_contract_sha256"):
        _require(_is_sha256(provenance[key]),
                 f"bc_aux final release lineage {key} invalid")
    _require(provenance["bc_aux_demos_sha256"] == demos_sha256,
             "bc_aux final release lineage demos disagree with the behaviour gate")
    start = provenance["start_steps"]
    target = provenance["target_global_steps"]
    _require(_is_plain_int(start) and _is_plain_int(target)
             and 0 <= start < target == int(final_step),
             "bc_aux final release lineage step counts do not close")
    seed = provenance["seed"]
    _require(seed is None or (_is_plain_int(seed) and 0 <= seed < 2**32),
             "bc_aux final release lineage seed invalid")
    _require(isinstance(provenance["optimizer_reset"], bool),
             "bc_aux final release lineage optimizer_reset must be a bool")
    target_kl = provenance["target_kl"]
    _require(target_kl is None or (
        isinstance(target_kl, (int, float))
        and not isinstance(target_kl, bool)
        and math.isfinite(float(target_kl))
        and float(target_kl) > 0),
        "bc_aux final release lineage target_kl invalid")
    value = provenance["distill_beta"]
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0,
        "bc_aux final release lineage distill_beta must be a finite positive number")
    _require(
        provenance["bc_aux_mode"]
        == "expanded-trainable-a12-contextual-mixture"
        and isinstance(provenance["bc_aux_lambda"], (int, float))
        and not isinstance(provenance["bc_aux_lambda"], bool)
        and float(provenance["bc_aux_lambda"]) == 0.0,
        "bc_aux final release lineage must bind the rev10 contextual mixture/λ=0")
    _require(provenance["calib_record_only"] is False,
             "bc_aux formal release forbids calib_record_only to bypass the hard gate")
    return dict(provenance)


def _run_bc_aux_liveness_preflight(
        *, run_dir: pathlib.Path, args,
        resume_checkpoint_bytes: bytes,
        resume_checkpoint_sha256: str,
        bank, x, y, episode_id, masks,
        demos_sha256: str, manager_npz_sha256: str,
        implementation_sha256: str, batch_size: int) -> tuple[dict, str]:
    """Run the aux necessary-condition preflight on an isolated checkpoint clone before the environment launches.

    FAIL/ERROR only publishes the sibling receipt, actively removes the canonical model_final and raises;
    a PASS clone is also discarded at once, and preflight weights or RNG state never leak into real training.
    """
    import numpy as np
    import torch as th
    from leashed_ppo import LeashedMaskablePPO

    for label, value in (
            ("resume_checkpoint_sha256", resume_checkpoint_sha256),
            ("demos_sha256", demos_sha256),
            ("manager_npz_sha256", manager_npz_sha256),
            ("implementation_sha256", implementation_sha256)):
        _require(_is_sha256(value),
                 f"bc_aux liveness input {label} invalid")
    _require(isinstance(resume_checkpoint_bytes, bytes)
             and len(resume_checkpoint_bytes) > 0,
             "bc_aux liveness is missing the frozen resume checkpoint bytes")
    structural = _bc_aux_structural_active(args)
    call_plan = (
        {
            "rollout_quantum": int(args.n_steps * args.num_envs),
            "train_calls": int(
                args.total_steps // (args.n_steps * args.num_envs)),
            "aux_optimizer_calls": 0,
            "policy_gradient_canary_calls": 1,
            "initial_adapter_calibrations": 1,
            "trainable_adapter_parameters": 5,
        }
        if structural else _bc_aux_liveness_call_plan(
            args.total_steps, args.n_steps, args.num_envs)
    )
    receipt_path = pathlib.Path(run_dir) / "bc_aux_liveness_preflight.json"
    base = {
        "schema_version": _BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "objective_revision": _BC_AUX_OBJECTIVE_REVISION,
        "inputs": {
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "demos_sha256": demos_sha256,
            "manager_npz_sha256": manager_npz_sha256,
            "implementation_sha256": implementation_sha256,
        },
        "config": {
            "bc_aux_lambda": float(args.bc_aux_lambda),
            "seed": args.seed,
            "device": str(args.device),
            "learning_rate": float(args.lr),
            "distill_beta": float(args.distill_beta),
            "target_kl": args.target_kl,
            "reset_optimizer": bool(args.reset_optimizer),
            "n_steps": int(args.n_steps),
            "num_envs": int(args.num_envs),
            "batch_size": int(batch_size),
            "total_steps": int(args.total_steps),
            "mechanism": (
                "expanded-trainable-a12-contextual-mixture"
                if structural else "legacy-gradient-aux"),
            **({
                "circuit": {
                    **_bc_aux_circuit_spec(),
                    "king_support":
                        _BC_AUX_CIRCUIT_KING_SUPPORT,
                },
            } if structural else {
                "positive_fraction": _BC_AUX_POSITIVE_FRACTION,
                "positive_target": _BC_AUX_POSITIVE_TARGET,
                "negative_target": _BC_AUX_NEGATIVE_TARGET,
                "anchor_kl_coef": _BC_AUX_ANCHOR_KL_COEF,
            }),
            **call_plan,
        },
    }

    # SB3 load(seed=...) replays the global Python/NumPy/Torch seeds. The preflight is an isolated probe
    # and must restore every stream before the real model/env is built, so it cannot change the subsequent production trajectory.
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = th.get_rng_state().clone()
    cuda_states = (
        th.cuda.get_rng_state_all()
        if th.cuda.is_available() else None)
    record = None
    failure = None
    try:
        load_kw = {
            "teacher_path": None,
            "teacher_sha256": None,
        }
        if args.seed is not None:
            load_kw["seed"] = args.seed
        model = LeashedMaskablePPO.load(
            io.BytesIO(resume_checkpoint_bytes), env=None,
            device=args.device, **load_kw)
        if structural:
            # Whether the isolated clone is a first install or a genuine
            # continuation is checkpoint state, not a launcher assumption.
            # Keep the preregistered config and the executed ``calls`` receipt
            # identical so a preserved adapter cannot be falsely described as
            # having been recalibrated.
            initial_calibrations = int(
                getattr(model, "_bc_aux_circuit_spec", None) is None)
            call_plan["initial_adapter_calibrations"] = (
                initial_calibrations)
            base["config"]["initial_adapter_calibrations"] = (
                initial_calibrations)
        saved_lr = model.learning_rate
        if structural:
            # Topology changes inside the exact structural simulation; its
            # helper resets the optimizer immediately afterwards.
            pass
        elif args.reset_optimizer:
            _reset_policy_optimizer(model, args.lr)
        else:
            _require(not callable(saved_lr)
                     and math.isclose(float(saved_lr), args.lr,
                                      rel_tol=0, abs_tol=1e-12),
                     "bc_aux liveness resume learning rate disagrees with the production CLI: "
                     f"{saved_lr} != {args.lr}")
            _require(all(math.isclose(
                float(group["lr"]), args.lr,
                rel_tol=0, abs_tol=1e-12)
                for group in model.policy.optimizer.param_groups),
                "bc_aux liveness checkpoint optimizer current lr disagrees with production")
        model.target_kl = args.target_kl
        _require(math.isclose(float(model.ent_coef), args.ent_coef,
                              rel_tol=0, abs_tol=1e-12)
                 and math.isclose(float(model.gamma), args.gamma,
                                  rel_tol=0, abs_tol=1e-12)
                 and model.n_steps == args.n_steps
                 and model.batch_size == batch_size,
                 "bc_aux liveness clone disagrees with the frozen production recipe")
        _validate_model_recipe(
            model, expected_target_kl=args.target_kl)
        result = (
            _simulate_bc_aux_circuit_liveness(
                model, x=x, y=y, episode_id=episode_id,
                masks=masks, learning_rate=args.lr,
                call_plan=call_plan)
            if structural else
            _simulate_bc_aux_liveness(
                model, bank=bank, x=x, y=y, episode_id=episode_id,
                masks=masks, bc_aux_lambda=args.bc_aux_lambda,
                seed=args.seed, call_plan=call_plan)
        )
        record = {**base, **result}
        del model
    except Exception as exc:
        failure = exc
        record = {
            **base,
            "status": "ERROR",
            "simulation": (
                "isolated-exact-mixture-with-policy-gradient-canary"
                if structural else
                "isolated-aux-only-necessary-condition"),
            "evaluation_scope": (
                "bc-v2-nested-validation-only"
                if structural else "bc-v2-training-episodes-only"),
            "heldout_rows_consumed": 0,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        th.set_rng_state(torch_state)
        if cuda_states is not None:
            th.cuda.set_rng_state_all(cuda_states)

    _write_bc_aux_behavior_receipt(receipt_path, record)
    receipt_sha256 = hashlib.sha256(
        receipt_path.read_bytes()).hexdigest()
    if record["status"] != "PASS":
        (pathlib.Path(run_dir) / "model_final.zip").unlink(
            missing_ok=True)
        if failure is not None:
            raise ValueError(
                "bc_aux liveness preflight ERROR; environment not launched, training refused; "
                f"see {receipt_path}") from failure
        _require(
            False,
            "bc_aux liveness preflight FAIL; environment not launched, training refused: "
            f"{record['gate']['reasons']}; see {receipt_path}")
    return record, receipt_sha256


def _publish_model_final_with_bc_aux_gate(
        model, destination: str | pathlib.Path, *,
        x, y, episode_id, masks, demos_sha256: str,
        anchor_sd: dict, publication_provenance: dict) -> dict:
    """Publish only after the adapter's held-out *safety* gate passes.

    Teacher recall and deterministic a12 deployment are intentionally not
    objectives in rev10: PPO decides from combat return whether a12 belongs
    in its greedy policy.  This transaction gates spill, drift and resource
    safety, records whether PPO actually sampled enough native-certified a12
    transitions, and leaves efficacy to paired native evaluation.
    """
    _require(_is_sha256(demos_sha256),
             "bc_aux final behaviour gate is missing the demos_sha256 binding")
    _require(isinstance(anchor_sd, dict)
             and set(anchor_sd) == set(_POLICY_HEAD_KEYS),
             "bc_aux final behaviour gate is missing the mounted starting policy anchor")
    final = pathlib.Path(destination)
    if final.suffix.lower() != ".zip":
        final = pathlib.Path(f"{final}.zip")
    current_sd = _policy_head_snapshot(model.policy)
    metrics = bc_aux_behavior_metrics(
        current_sd, x, y, episode_id, masks,
        anchor_sd=anchor_sd, heldout_only=True,
        circuit_spec=_bc_aux_circuit_spec())
    gate = bc_aux_behavior_gate(
        metrics, require_root_anchor=True,
        require_teacher_recall=False,
        require_deployable_a12=False)
    provenance = _validate_publication_provenance(
        publication_provenance, demos_sha256=demos_sha256,
        final_step=int(model.num_timesteps))
    eligible_states = getattr(model, "_bc_aux_eligible_states", 0)
    requested_a12 = getattr(model, "_bc_aux_requested_a12", 0)
    sampled_a12 = getattr(model, "_bc_aux_sampled_a12", 0)
    rejected_a12 = getattr(model, "_bc_aux_rejected_a12", 0)
    unexpected_sampled_a12 = getattr(
        model, "_bc_aux_unexpected_sampled_a12", 0)
    for name, value in (
        ("eligible_states", eligible_states),
        ("requested_a12", requested_a12),
        ("sampled_a12", sampled_a12),
        ("rejected_a12", rejected_a12),
        ("unexpected_sampled_a12", unexpected_sampled_a12),
    ):
        _require(
            _is_plain_int(value) and value >= 0,
            f"bc_aux rollout count {name} is not a plain non-negative integer",
        )
    _require(
        requested_a12 == sampled_a12 + rejected_a12
        and requested_a12 <= eligible_states,
        "bc_aux rollout requested/executed/refused counts do not close",
    )
    expected_a12_mass = getattr(
        model, "_bc_aux_expected_a12_mass", 0.0)
    _require(
        isinstance(expected_a12_mass, (int, float))
        and not isinstance(expected_a12_mass, bool)
        and math.isfinite(float(expected_a12_mass))
        and 0.0 <= float(expected_a12_mass) <= float(eligible_states),
        "bc_aux rollout expected_a12_mass not finite or outside the eligible closure",
    )
    exploration_reasons = []
    if not math.isfinite(expected_a12_mass) \
            or expected_a12_mass < _BC_AUX_MIN_EXPECTED_A12_SAMPLES:
        exploration_reasons.append(
            f"expected_a12_mass<{_BC_AUX_MIN_EXPECTED_A12_SAMPLES}")
    if sampled_a12 < _BC_AUX_MIN_ACTUAL_A12_SAMPLES:
        exploration_reasons.append(
            f"sampled_a12<{_BC_AUX_MIN_ACTUAL_A12_SAMPLES}")
    if unexpected_sampled_a12 != 0:
        exploration_reasons.append("unexpected_sampled_a12!=0")
    exploration_status = (
        "INFORMATIVE" if not exploration_reasons
        else "INSUFFICIENT_OR_INVALID")
    record = {
        "schema_version": _BC_AUX_BEHAVIOR_RECEIPT_SCHEMA_VERSION,
        "step": int(model.num_timesteps),
        "demos_sha256": demos_sha256,
        "objective_revision": _BC_AUX_OBJECTIVE_REVISION,
        "evaluation_scope": "original-bc-v2-heldout-episodes",
        "mask_mode": "bc-v2-recorded",
        "anchor": {
            "identity": "bc-aux-root-policy",
            "policy_head_sha256": _policy_head_sha256(anchor_sd),
        },
        "candidate_policy_head_sha256": _policy_head_sha256(current_sd),
        "provenance": provenance,
        "metrics": metrics,
        "gate": gate,
        "exploration_evidence": {
            "eligible_states": eligible_states,
            "expected_a12_mass": float(expected_a12_mass),
            "requested_a12": requested_a12,
            "sampled_a12": sampled_a12,
            "rejected_a12": rejected_a12,
            "unexpected_sampled_a12": unexpected_sampled_a12,
            "minimum_expected_a12_mass":
                _BC_AUX_MIN_EXPECTED_A12_SAMPLES,
            "minimum_actual_a12_samples":
                _BC_AUX_MIN_ACTUAL_A12_SAMPLES,
            "information_status": exploration_status,
            "reasons": exploration_reasons,
        },
        "publication": (
            "GATE_PASS_PENDING"
            if gate["verdict"] == "PASS" and not exploration_reasons
            else "REFUSED"),
        "model_sha256": None,
        "save_error": None,
    }
    receipt = final.parent / "bc_aux_behavior_receipt.json"
    if gate["verdict"] != "PASS" or exploration_reasons:
        _write_bc_aux_behavior_receipt(receipt, record)
        _require(False,
                 "bc_aux final held-out/exploration-evidence hard gate FAIL; refusing to publish"
                 f" model_final: {gate['reasons'] + exploration_reasons}; "
                 f"see {receipt}")

    # Never write gate PASS as "published" first: if disk-full/CRC/finite checks fail, that would
    # leave a success receipt with no model. Only after the canonical zip succeeds and is re-hashed is
    # PUBLISHED+model_sha256 written atomically. A failure receipt is explicitly SAVE_FAILED.
    try:
        _atomic_save_model(model, final)
        model_sha256 = hashlib.sha256(final.read_bytes()).hexdigest()
        _require(_is_sha256(model_sha256),
                 "model_final SHA256 computation failed after publishing")
        record["model_sha256"] = model_sha256
        record["publication"] = "PUBLISHED"
        _write_bc_aux_behavior_receipt(receipt, record)
    except Exception as exc:
        # If the model was written but the final receipt could not be committed, rather withdraw the unbacked canonical; the checkpoint
        # recovery copies stay in ckpt/, so failing closed here never loses training progress.
        final.unlink(missing_ok=True)
        record["publication"] = "SAVE_FAILED"
        record["model_sha256"] = None
        record["save_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            _write_bc_aux_behavior_receipt(receipt, record)
        except Exception:
            pass
        raise
    return record


# ---- E4 contract rev5 dual keys + E6 probe set pinned (PREREG-v33-content-case E4/E6) ----


def _contract_dry_curriculum(args):
    """E4 rev5 key: course 5 absent = "disabled"; present = {"schedule": CLI literal text}.

    The payload is the CLI literal string of --dry-curriculum-schedule (same source as the D3 per-leg extras column,
    L-cur/L-full carry the main table; L-base has no schedule -> disabled). The schedule literal is
    the source of truth for the full table's semantics (expanded deterministically by _parse_dry_curriculum_schedule), and resume reconciliation
    is carried by string equality.
    """
    return ({"schedule": str(args.dry_curriculum_schedule)}
            if args.dry_curriculum_schedule else "disabled")


def _contract_bc_aux(args, bc_aux_demos_sha256):
    """rev12 key: bind the contextual mixture and its immutable data.

    The rejected rev5 gradient mechanism is deliberately not representable in
    a new production contract.
    """
    if not _bc_aux_active(args):
        return "disabled"
    _require(_bc_aux_structural_active(args)
             and float(args.bc_aux_lambda) == 0.0,
             "rev12 contract only accepts a structural graft")
    _require(_is_sha256(bc_aux_demos_sha256),
             "4B present but the bc-worker-v2 demo set sha256 is missing; refusing to write the rev5 contract")
    return {"mode": "expanded-trainable-a12-contextual-mixture",
            "lambda": float(args.bc_aux_lambda),
            "demos_sha256": bc_aux_demos_sha256,
            "objective_revision": _BC_AUX_OBJECTIVE_REVISION,
            "circuit": {
                **_bc_aux_circuit_spec(),
            },
            "king_support": _BC_AUX_CIRCUIT_KING_SUPPORT,
            "aux_optimizer_calls_per_rollout": 0,
            "initial_calibration":
                "exact-five-percent-contextual-legal-support-mixture",
            "trainable_adapter_parameters": 5,
            "post_step_projection": {
                "gate_parameter_abs_max":
                    _BC_AUX_CIRCUIT_GATE_PARAMETER_ABS_MAX,
                "probability_min":
                    _BC_AUX_CIRCUIT_PROBABILITY_MIN,
                "probability_max":
                    _BC_AUX_CIRCUIT_PROBABILITY_MAX,
            },
            "liveness_preflight": bool(args.bc_aux_liveness_preflight)}


def _assert_bc_v1_demos_frozen(path: str | pathlib.Path) -> str:
    """Compatibility function name: bind the BC-v1 demos to the current strict PASS receipt.

    The historical constant only serves import compatibility for the protocol-v3 driver and cannot prove v4 semantics. Here we first
    check the protocol, implementation, generator and replay evidence of policy/report,
    then require the report demos_sha256 to equal the live bytes; so old reports naturally fail closed,
    and newly re-collected v4 artifacts are not permanently blocked by the old 3bf8....
    """
    demos = pathlib.Path(path)
    actual = _capture_file_sha256(demos, "BC-v1 demos (bound to the current PASS receipt)")
    policy = demos.with_name("policy_sd.pt")
    _require(policy.is_file(), f"BC-v1 current weights missing/unreadable: {policy}")
    report = _validate_bc_report(policy, "data_gate")
    expected = report.get("demos_sha256")
    _require(_is_sha256(expected),
             "BC-v1 current PASS receipt is missing demos_sha256")
    _require(actual == expected,
             "BC-v1 demos bytes drifted from the current PASS receipt: "
             f"{actual} != {expected}")
    return actual


def _worker_time_identity(protocol="legacy") -> dict | None:
    if protocol == "legacy":
        return None
    from diablogym.completion_clock import COMPLETION_PROTOCOLS, COMPLETION_RECIPES
    _require(isinstance(protocol, str) and protocol in COMPLETION_PROTOCOLS,
             "worker_time_protocol must be legacy/" + "/".join(COMPLETION_PROTOCOLS))
    return COMPLETION_RECIPES[protocol].as_dict()


def _validate_worker_time_config(protocol="legacy", *, worker,
        worker_learning_window_scope, resource_protocol, resource_purchase_mode,
        resource_service_policy, max_steps, farm_scene_cap) -> dict | None:
    recipe = _worker_time_identity(protocol)
    if recipe is None:
        return None
    _require(worker and worker_learning_window_scope == EARNED_DIVE_SUFFIX_SCOPE,
             "completion-l2 protocols require an earned-dive-suffix-v1 Worker")
    # R18-B3 (2026-09-07): sustain-loot-v1 stands alongside sustain-v6; other service policies are still refused.
    _require(resource_protocol == "l2-town-v1" and resource_purchase_mode == "full"
             and resource_service_policy in ("sustain-v6", "sustain-loot-v1"),
             "completion-l2 protocols require l2-town-v1/full/sustain-v6 or sustain-loot-v1")
    _require(max_steps == recipe["actor_denominator"]
             and farm_scene_cap == recipe["farm_microsteps"],
             "completion-l2 protocols require max_steps=6000 (observation only) and farm_scene_cap=3600")
    return recipe


def _validate_worker_time_args(args) -> None:
    # R18-B3 re-review (2026-09-07): loot exists only under the completion clock, and this rule has nothing to do with
    # the learning-window scope. Before, loot + legacy under farm-only / farm-dive-v1 passed the whole parameter validation layer
    # and only died when OptionsEnv was constructed (python/diablogym/options_env.py:486) -- by then
    # the run directory and config were on disk and the native engine had started. The entry point now fails closed first, with the same wording.
    if getattr(args, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
        from diablogym.completion_clock import COMPLETION_PROTOCOLS
        _require(getattr(args, "worker_time_protocol", "legacy") in COMPLETION_PROTOCOLS,
                 "sustain-loot-v1 requires an explicit completion-l2 time protocol")
    _validate_worker_time_config(getattr(args, "worker_time_protocol", "legacy"),
        worker=getattr(args, "worker", False),
        worker_learning_window_scope=getattr(args, "worker_learning_window_scope", "farm-only"),
        resource_protocol=getattr(args, "resource_protocol", "off"),
        resource_purchase_mode=getattr(args, "resource_purchase_mode", "full"),
        resource_service_policy=getattr(args, "resource_service_policy", "legacy-v1"),
        max_steps=getattr(args, "max_steps", None),
        farm_scene_cap=getattr(args, "farm_scene_cap", None))


def _worker_prefix_identity(args) -> dict | None:
    scope = getattr(args, "worker_learning_window_scope", "farm-only")
    path = getattr(args, "worker_prefix_model", None)
    attempts = getattr(args, "worker_prefix_max_attempts", None)
    microsteps = getattr(args, "worker_prefix_max_microsteps", None)
    if scope != EARNED_DIVE_SUFFIX_SCOPE:
        # Reject stray flags before opening any model in the legacy scopes.
        return worker_prefix_recipe(scope, "specified" if path is not None else None,
                                    attempts, microsteps)
    _require(path is not None, "earned-dive-suffix-v1 requires --worker-prefix-model")
    return worker_prefix_recipe(scope, _capture_file_sha256(path, "worker_prefix_model"),
                                attempts, microsteps)


def _validate_worker_prefix_args(args) -> None:
    if _worker_prefix_identity(args) is None:
        return
    _require(args.worker and args.algo == "mppo" and str(args.device) == "cpu",
             "earned-dive-suffix-v1 requires CPU Worker MaskablePPO")
    _require(args.seed is not None and args.max_steps == 6000,
             "earned-dive-suffix-v1 requires an explicit seed and max_steps=6000")
    # R18-B3 (2026-09-07): the service policy changes from a single pinned value to set membership (sustain-v6 or
    # sustain-loot-v1), but **stays in its original position in `fixed` (third)**: the R18-B3 re-review pointed out
    # that moving it before the loop would change the first error that existing invalid callers get. Apart from the service policy itself
    # (whose rule did change), the wording and check order of every pinned value are still unchanged.
    fixed = {
        "resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
        "resource_service_policy": ("sustain-v6", "sustain-loot-v1"),
        "dive_blocker_recovery": "adjacent-v1",
        "worker_policy_observation_view": "dual-v4-asymmetric-v3",
        "reward_economy": "v4", "worker_action14_logit_bonus": 2.5,
        "worker_dive_action11_logit_bonus": 2.0, "worker_potion_action13_logit_bonus": 2.0,
        "worker_hp_loss_price": 0.1, "worker_potion_pickup_bonus": 2.0,
        "worker_no_progress_timeout_credit": "zero",
        "worker_descend_escrow_fraction": 0.5, "worker_descend_escrow_power": 1.6,
        "worker_descend_escrow_readiness_gate": True,
        "farm_scene_cap": 3600, "reset_layer_clock_on_window": True,
    }
    for key, expected in fixed.items():
        # tuple = set membership (since R18-B3 only resource_service_policy);
        # single-value keys keep the original equality comparison and wording.
        if isinstance(expected, tuple):
            _require(getattr(args, key, None) in expected,
                     f"earned-dive-suffix-v1 requires {key} " + " or ".join(expected))
        else:
            _require(getattr(args, key, None) == expected,
                     f"earned-dive-suffix-v1 requires {key}={expected!r}")
    # R18-B3 (2026-09-07): the loot economy has no old-clock version; wording identical to worker_env /
    # options_env.
    from diablogym.completion_clock import COMPLETION_PROTOCOLS
    _require(getattr(args, "resource_service_policy", None) != "sustain-loot-v1"
             or getattr(args, "worker_time_protocol", "legacy") in COMPLETION_PROTOCOLS,
             "sustain-loot-v1 requires an explicit completion-l2 time protocol")
    _require(not any(getattr(args, key, None) for key in (
        "skip_dry", "dry_curriculum_schedule", "deep_start_curriculum", "deep",
        "death_ladder", "resource_calibration", "teacher_override", "bc_init",
        "freeze_policy_steps", "worker_zip", "worker_npz")),
        "earned-dive-suffix-v1 forbids alternate prefix/curriculum/teacher overrides")
    _require(_requested_drink_sovereignty(args) is not False,
             "earned-dive-suffix-v1 requires environment-mask drink sovereignty")
    _require(args.distill_beta == 0 and not _bc_aux_active(args),
             "earned-dive-suffix-v1 excludes teacher/BC training")


def _validate_resource_warm_start_args(args) -> None:
    if not getattr(args, "resource_warm_start", None):
        return
    _require(args.worker and args.algo == "mppo" and str(args.device) == "cpu",
             "--resource-warm-start requires CPU Worker MaskablePPO")
    _require(args.seed is not None,
             "--resource-warm-start requires an explicit seed")
    _require(not any(getattr(args, key, None) for key in (
        "resume_from", "bc_init", "teacher_override", "reset_optimizer",
        "reset_worker_critic", "allow_legacy_resume", "allow_manager_change",
        "allow_environment_restart_resume", "freeze_policy_steps",
        "deep_start_curriculum", "dry_curriculum_schedule", "skip_dry")),
        "resource warm-start cannot combine with resume/migration/reset/curriculum overrides")
    _require(args.distill_beta == 0 and not _bc_aux_active(args),
             "resource warm-start preserves the parent disabled teacher/BC regime")
    # R18-B3b (2026-09-07): sustain-loot-v1 joins the list. It is mintable only
    # through migrate_loot_candidate (schema/2), whose target contract pins the
    # completion clock, the earned-suffix prefix and the readiness/retreat laws;
    # the sustain-v2..v6 arms are unchanged and still go through schema/1.
    _require(getattr(args, "resource_protocol", "off") == "l2-town-v1"
             and getattr(args, "resource_purchase_mode", "full") == "full"
             and getattr(args, "resource_service_policy", "legacy-v1") in ("sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"),
             "resource warm-start requires an explicit sustain-v2/sustain-v3/sustain-v4/sustain-v5/sustain-v6/sustain-loot-v1 environment")
    if getattr(args, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
        from diablogym.completion_clock import COMPLETION_PROTOCOLS
        _require(getattr(args, "worker_time_protocol", "legacy") in COMPLETION_PROTOCOLS
                 and getattr(args, "worker_learning_window_scope", "farm-only")
                     == EARNED_DIVE_SUFFIX_SCOPE
                 and getattr(args, "dive_blocker_recovery", "off") == "adjacent-v1",
                 "sustain-loot-v1 warm start requires an explicit completion-l2 time "
                 "protocol, earned-dive-suffix-v1 and dive_blocker_recovery adjacent-v1")
    _require(pathlib.Path(args.resource_warm_start).is_file(),
             "resource warm-start manifest does not exist")


def _validate_first_update_diagnostic_args(args) -> None:
    mode = getattr(args, "diagnostic_rollout", "disabled")
    _require(mode in ("disabled", "first-update-v1"),
             "unknown diagnostic rollout mode")
    if mode == "disabled":
        return
    _require(args.worker and args.algo == "mppo" and str(args.device) == "cpu"
             and getattr(args, "artifact_scope", "production") == "candidate",
             "first-update-v1 requires a CPU Worker MaskablePPO candidate")
    _require(bool(getattr(args, "resource_warm_start", None))
             and not getattr(args, "resume_from", None),
             "first-update-v1 requires a fresh resource warm-start")
    _require(args.total_steps == args.n_steps * args.num_envs,
             "first-update-v1 records exactly one complete rollout")
    _require(not getattr(args, "calib_probes", "")
             and not getattr(args, "calib_record_only", False),
             "first-update-v1 excludes calibration interruptions")


def _first_update_diagnostic_callback(args, run_dir, implementation_sha256):
    if getattr(args, "diagnostic_rollout", "disabled") == "disabled":
        return None
    _validate_first_update_diagnostic_args(args)
    from rollout_diagnostics import FirstUpdateDiagnosticCallback
    return FirstUpdateDiagnosticCallback(run_dir, implementation_sha256)


def _validate_args(args) -> None:
    _validate_worker_time_args(args)
    _validate_worker_prefix_args(args)
    _validate_resource_warm_start_args(args)
    _validate_first_update_diagnostic_args(args)
    _require(args.total_steps > 0, "--total-steps must be > 0")
    _require(args.num_envs > 0, "--num-envs must be > 0")
    _require(args.n_steps > 0, "--n-steps must be > 0")
    rollout_quantum = args.n_steps * args.num_envs
    _require(args.total_steps % rollout_quantum == 0,
             "--total-steps must be divisible by n_steps * num_envs, "
             f"otherwise SB3 silently over-samples upward (current quantum {rollout_quantum})")
    _require(args.max_steps > 0, "--max-steps must be > 0")
    _require(math.isfinite(args.lr) and args.lr > 0, "--lr must be a finite positive number")
    _require(math.isfinite(args.gamma) and 0 <= args.gamma <= 1,
             "--gamma must lie in [0, 1]")
    _require(math.isfinite(args.ent_coef) and args.ent_coef >= 0,
             "--ent-coef must be a finite non-negative number")
    _require(math.isfinite(args.distill_beta) and args.distill_beta >= 0,
             "--distill-beta must be a finite non-negative number")
    action14_logit_bonus = float(getattr(
        args, "worker_action14_logit_bonus", 0.0))
    _require(
        math.isfinite(action14_logit_bonus)
        and 0.0 <= action14_logit_bonus <= 10.0,
        "--worker-action14-logit-bonus must be a finite number in [0,10]",
    )
    _require(
        action14_logit_bonus == 0.0
        or (
            args.worker
            and args.algo == "mppo"
            and _worker_policy_observation_view(args)
            == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
        ),
        "--worker-action14-logit-bonus nonzero only applies to "
        "dual-v4 asymmetric Worker MaskablePPO",
    )
    action11_logit_bonus = float(getattr(
        args, "worker_dive_action11_logit_bonus", 0.0))
    _require(
        math.isfinite(action11_logit_bonus)
        and 0.0 <= action11_logit_bonus <= 10.0,
        "--worker-dive-action11-logit-bonus must be a finite number in [0,10]",
    )
    _require(
        action11_logit_bonus == 0.0
        or (
            args.worker
            and args.algo == "mppo"
            and getattr(
                args, "worker_learning_window_scope", "farm-only")
            in ("farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE)
        ),
        "--worker-dive-action11-logit-bonus nonzero only applies to "
        "Worker MaskablePPO in the live-DIVE classroom (R13 cold-start lever)",
    )
    action13_logit_bonus = float(getattr(
        args, "worker_potion_action13_logit_bonus", 0.0))
    _require(
        math.isfinite(action13_logit_bonus)
        and 0.0 <= action13_logit_bonus <= 10.0,
        "--worker-potion-action13-logit-bonus must be a finite number in [0,10]",
    )
    # R16: the prior hangs on the asymmetric policy class (policy_kwargs are only assembled under
    # the dual-v4 view), so nonzero is limited to dual-v4 Worker MaskablePPO (same as a14).
    _require(
        action13_logit_bonus == 0.0
        or (
            args.worker
            and args.algo == "mppo"
            and _worker_policy_observation_view(args)
            == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
        ),
        "--worker-potion-action13-logit-bonus nonzero only applies to "
        "dual-v4 asymmetric Worker MaskablePPO (R16 potion-pickup prior)",
    )
    depth_shaping_unit = float(getattr(
        args, "worker_depth_shaping_unit", 0.0))
    _require(
        math.isfinite(depth_shaping_unit)
        and 0.0 <= depth_shaping_unit <= 100.0,
        "--worker-depth-shaping-unit must be a finite number in [0,100]",
    )
    _require(
        depth_shaping_unit == 0.0
        or (
            args.worker
            and args.algo == "mppo"
            and getattr(
                args, "worker_learning_window_scope", "farm-only")
            in ("farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE)
        ),
        "--worker-depth-shaping-unit nonzero only applies to "
        "Worker MaskablePPO in the live-DIVE classroom (R13.2 plan A)",
    )
    descend_bonus_fraction = float(getattr(
        args, "worker_descend_bonus_fraction", 0.0))
    _require(
        math.isfinite(descend_bonus_fraction)
        and 0.0 <= descend_bonus_fraction <= 1.0,
        "--worker-descend-bonus-fraction must lie in [0,1]",
    )
    _require(
        descend_bonus_fraction == 0.0
        or (
            args.worker
            and args.algo == "mppo"
            and getattr(
                args, "worker_learning_window_scope", "farm-only")
            in ("farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE)
        ),
        "--worker-descend-bonus-fraction nonzero only applies to "
        "Worker MaskablePPO in the live-DIVE classroom (R14 plan B)",
    )
    descend_escrow_fraction = float(getattr(
        args, "worker_descend_escrow_fraction", 0.0))
    _require(
        math.isfinite(descend_escrow_fraction)
        and 0.0 <= descend_escrow_fraction <= 1.0,
        "--worker-descend-escrow-fraction must lie in [0,1]",
    )
    _require(
        descend_escrow_fraction == 0.0
        or (
            args.worker
            and args.algo == "mppo"
            and getattr(
                args, "worker_learning_window_scope", "farm-only")
            in ("farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE)
        ),
        "--worker-descend-escrow-fraction nonzero only applies to "
        "Worker MaskablePPO in the live-DIVE classroom (R14.2 plan D)",
    )
    _require(
        descend_escrow_fraction == 0.0
        or descend_bonus_fraction == 0.0,
        "plan B (bonus) and plan D (escrow) are mutually exclusive and must not be enabled together",
    )
    descend_escrow_power = float(getattr(
        args, "worker_descend_escrow_power", 1.0))
    _require(
        math.isfinite(descend_escrow_power)
        and 1.0 <= descend_escrow_power <= 3.0,
        "--worker-descend-escrow-power must lie in [1,3]",
    )
    _require(
        descend_escrow_power == 1.0
        or descend_escrow_fraction > 0.0,
        "--worker-descend-escrow-power != 1 only takes effect together with escrow (R14.3 plan E)",
    )
    _require(
        not getattr(args, "worker_descend_escrow_readiness_gate", False)
        or descend_escrow_fraction > 0.0,
        "--worker-descend-escrow-readiness-gate only takes effect together with escrow"
        " (R15 amendment 2)",
    )
    descend_escrow_readiness_table = str(getattr(
        args, "worker_descend_escrow_readiness_table", "v1"))
    _require(
        descend_escrow_readiness_table in ("v1", "v2"),
        "--worker-descend-escrow-readiness-table must be v1/v2",
    )
    _require(
        descend_escrow_readiness_table == "v1"
        or bool(getattr(
            args, "worker_descend_escrow_readiness_gate", False)),
        "--worker-descend-escrow-readiness-table v2 only takes effect together with the readiness gate"
        " (--worker-descend-escrow-readiness-gate)"
        " (R17.0 amendment 1)",
    )
    distill_anneal_actor_rollouts = int(getattr(
        args, "distill_anneal_actor_rollouts", 0))
    _require(
        distill_anneal_actor_rollouts == 0
        or distill_anneal_actor_rollouts >= 2,
        "--distill-anneal-actor-rollouts must be 0 or >=2",
    )
    _require(
        distill_anneal_actor_rollouts == 0
        or (
            args.distill_beta > 0
            and args.worker
            and args.algo == "mppo"
        ),
        "--distill-anneal-actor-rollouts only applies to "
        "Worker MaskablePPO with β>0",
    )
    _require(args.target_kl is None
             or (math.isfinite(args.target_kl) and args.target_kl > 0),
             "--target-kl, when given, must be a finite positive number")
    _require(not args.reset_optimizer or bool(args.resume_from),
             "--reset-optimizer only applies to a --resume-from continuation")
    reset_worker_critic = bool(getattr(
        args, "reset_worker_critic", False))
    critic_warmup_steps = int(getattr(
        args, "critic_warmup_steps", 0))
    gradient_clip_mode = getattr(
        args, "gradient_clip_mode", "global")
    _require(
        gradient_clip_mode in {
            "global",
            "separate-actor-critic-v1",
            "separate-root-context-critic-v2",
        },
        "--gradient-clip-mode is not a registered mode",
    )
    _require(critic_warmup_steps >= 0,
             "--critic-warmup-steps must not be negative")
    if reset_worker_critic:
        _require(
            args.worker and args.algo == "mppo" and bool(args.resume_from),
            "--reset-worker-critic only applies to "
            "--worker --algo mppo --resume-from",
        )
        _require(args.reset_optimizer,
                 "--reset-worker-critic requires --reset-optimizer as well")
        _require(args.seed is not None,
                 "--reset-worker-critic requires an explicit --seed")
        _require(str(args.device) == "cpu",
                 "--reset-worker-critic currently only allows --device cpu")
        _require(
            critic_warmup_steps > 0
            and critic_warmup_steps % rollout_quantum == 0
            and critic_warmup_steps < args.total_steps,
            "--critic-warmup-steps must be positive, divide the rollout quantum and be less than"
            " --total-steps, so the actor really updates after warmup",
        )
        _require(
            gradient_clip_mode in {
                "separate-actor-critic-v1",
                "separate-root-context-critic-v2",
            },
            "--reset-worker-critic must use"
            " --gradient-clip-mode grouped clipping",
        )
    else:
        _require(
            critic_warmup_steps == 0,
            "--critic-warmup-steps is only used with --reset-worker-critic",
        )
    _require(
        gradient_clip_mode == "global"
        or (args.worker and args.algo == "mppo"
            and bool(args.resume_from or getattr(args, "resource_warm_start", None))),
        "grouped clipping only applies to Worker MaskablePPO continuation/resource warm-start",
    )
    _require(args.freeze_policy_steps >= 0, "--freeze-policy-steps must not be negative")
    _require(args.ckpt_every_steps > 0, "--ckpt-every-steps must be > 0")
    _require(args.sentinel_every > 0, "--sentinel-every must be > 0")
    _require(args.dry_anchor_every > 0, "--dry-anchor-every must be > 0")
    # E5 gauge knobs (record-only, no verdict; 0 = absent = zero intrusion, G0-2a prerequisite)
    _require(args.distill_ce_probe_every >= 0, "--distill-ce-probe-every must not be negative")
    _require(args.drywin_metrics_every >= 0, "--drywin-metrics-every must not be negative")
    _require(
        getattr(args, "worker_fast_forward_reward_credit", "none") in {
            "none", "terminal-death-only"
        },
        "--worker-fast-forward-reward-credit only allows "
        "none/terminal-death-only",
    )
    _require(
        math.isfinite(getattr(
            args, "worker_additional_terminal_death_cost", 0.0))
        and getattr(args, "worker_additional_terminal_death_cost", 0.0)
        >= 0.0,
        "--worker-additional-terminal-death-cost must be a finite non-negative number",
    )
    _require(
        args.worker
        or (
            getattr(
                args, "worker_fast_forward_reward_credit", "none") == "none"
            and getattr(
                args, "worker_additional_terminal_death_cost", 0.0) == 0.0
        ),
        "Worker reward-contract knobs only apply to --worker",
    )
    # R16 amendment: new wage-side knobs (explicit WorkerWindowEnv parameters) -- range + --worker only.
    worker_hp_loss_price = float(getattr(
        args, "worker_hp_loss_price", 0.0))
    _require(
        math.isfinite(worker_hp_loss_price) and worker_hp_loss_price >= 0.0,
        "--worker-hp-loss-price must be a finite non-negative number",
    )
    worker_potion_pickup_bonus = float(getattr(
        args, "worker_potion_pickup_bonus", 0.0))
    _require(
        math.isfinite(worker_potion_pickup_bonus)
        and worker_potion_pickup_bonus >= 0.0,
        "--worker-potion-pickup-bonus must be a finite non-negative number",
    )
    worker_no_progress_timeout_credit = getattr(
        args, "worker_no_progress_timeout_credit",
        _WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT)
    _require(
        worker_no_progress_timeout_credit
        in set(_WORKER_NO_PROGRESS_TIMEOUT_CREDITS),
        "--worker-no-progress-timeout-credit only allows "
        + "/".join(_WORKER_NO_PROGRESS_TIMEOUT_CREDITS),
    )
    _require(
        args.worker
        or (
            worker_hp_loss_price == 0.0
            and worker_potion_pickup_bonus == 0.0
            and worker_no_progress_timeout_credit
            == _WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT
        ),
        "R16 wage-side knobs (--worker-hp-loss-price/--worker-potion-pickup-bonus/"
        "--worker-no-progress-timeout-credit) only apply to --worker",
    )
    # R16 amendment: new environment-side/classroom-side parameters -- range + classroom parameters only for modes that hold an OptionsEnv
    # (--worker/--options); DiabloGymEnv-level parameters (explore_global_
    # fallback/explore_global_hunt/progress_far_tiles) are allowed in every mode -- all three
    # make_env paths construct DiabloGymEnv directly, and bool switches have no range to check.
    resource_protocol = getattr(args, "resource_protocol", "off")
    resource_purchase_mode = getattr(args, "resource_purchase_mode", "full")
    resource_service_policy = getattr(args, "resource_service_policy", "legacy-v1")
    validate_resource_service_config(resource_protocol, resource_purchase_mode,
                                     resource_service_policy)
    validate_dive_blocker_recovery(resource_protocol, getattr(args, "dive_blocker_recovery", "off"))
    _require(resource_protocol in ("off", "l2-town-v1"),
             "--resource-protocol must be off/l2-town-v1")
    _require(resource_purchase_mode in ("none", "heal", "potions", "armor", "full"),
             "--resource-purchase-mode must be none/heal/potions/armor/full")
    _require(resource_protocol != "off" or resource_purchase_mode == "full",
             "--resource-purchase-mode requires --resource-protocol l2-town-v1")
    _require(resource_protocol == "off" or args.worker or args.options,
             "--resource-protocol requires --worker/--options")
    resource_readiness_law = getattr(args, "resource_readiness_law", "veto-v1")
    _require(resource_readiness_law in ("veto-v1", "coach-v03"),
             "--resource-readiness-law must be veto-v1/coach-v03")
    _require(resource_readiness_law == "veto-v1" or resource_protocol == "l2-town-v1",
             "--resource-readiness-law coach-v03 requires --resource-protocol l2-town-v1")
    # R18-B: retreat-v1 only holds in the l2-town-v1/coach-v03 classroom (same on the engine side).
    resource_retreat = getattr(args, "resource_retreat", "off")
    _require(resource_retreat in ("off", "retreat-v1"),
             "--resource-retreat must be off/retreat-v1")
    _require(resource_retreat == "off" or (resource_protocol == "l2-town-v1"
                                           and resource_readiness_law == "coach-v03"),
             "--resource-retreat retreat-v1 requires --resource-protocol l2-town-v1 "
             "and --resource-readiness-law coach-v03")
    _require(resource_retreat == "off" or args.worker or args.options,
             "--resource-retreat requires --worker/--options")
    # R18-B5 (2026-09-07): portal-v1 is the retreat vehicle; besides l2-town-v1/coach-v03
    # it also needs retreat-v1 first (same as validate_portal_protocol on the engine side).
    resource_portal = getattr(args, "resource_portal", "off")
    _require(resource_portal in ("off", "portal-v1"),
             "--resource-portal must be off/portal-v1")
    _require(resource_portal == "off" or (resource_protocol == "l2-town-v1"
                                          and resource_readiness_law == "coach-v03"),
             "--resource-portal portal-v1 requires --resource-protocol l2-town-v1 "
             "and --resource-readiness-law coach-v03")
    _require(resource_portal == "off" or resource_retreat == "retreat-v1",
             "--resource-portal portal-v1 requires --resource-retreat retreat-v1")
    _require(resource_portal == "off" or args.worker or args.options,
             "--resource-portal requires --worker/--options")
    # R18-B6 (2026-09-07): sweep-v1 / cain-v1 / smith-v1 all live inside the trips of the loot economy
    # (same as validate_sweep_protocol / validate_identify_protocol /
    # validate_weapon_upgrade on the deployment side): l2-town-v1 + sustain-loot-v1; smith-v1
    # also needs the full purchase mode. All three only pass through the **env_kwargs of
    # WorkerWindowEnv/OptionsEnv, so they are still limited to --worker/--options.
    # (resource_protocol / resource_purchase_mode / resource_service_policy were already read above in
    # this function; the same values are reused here instead of reading args again.)
    resource_sweep = getattr(args, "resource_sweep", "off")
    _require(resource_sweep in ("off", "sweep-v1"),
             "--resource-sweep must be off/sweep-v1")
    _require(resource_sweep == "off" or resource_protocol == "l2-town-v1",
             "--resource-sweep sweep-v1 requires --resource-protocol l2-town-v1")
    _require(resource_sweep == "off" or resource_service_policy == "sustain-loot-v1",
             "--resource-sweep sweep-v1 requires --resource-service-policy "
             "sustain-loot-v1 (the loot economy collects what the sweep drops)")
    _require(resource_sweep == "off" or args.worker or args.options,
             "--resource-sweep requires --worker/--options")
    resource_identify = getattr(args, "resource_identify", "off")
    _require(resource_identify in ("off", "cain-v1"),
             "--resource-identify must be off/cain-v1")
    _require(resource_identify == "off" or resource_protocol == "l2-town-v1",
             "--resource-identify cain-v1 requires --resource-protocol l2-town-v1")
    _require(resource_identify == "off" or resource_service_policy == "sustain-loot-v1",
             "--resource-identify cain-v1 requires --resource-service-policy "
             "sustain-loot-v1 (the identify leg is part of that town trip)")
    _require(resource_identify == "off" or args.worker or args.options,
             "--resource-identify requires --worker/--options")
    # The training-side vocabulary is one value narrower than the deployment side, on purpose: RESOURCE_WEAPON_UPGRADES
    # also has dry-v1, the "observe only, never act" twin of smith-v1 (resource_weapon_
    # upgrade.py:129-137); it sends no commands, takes no micro-steps and makes no native calls, and its row data
    # is bit-identical to the control arm. Training with it would mint a lying identity for a control-arm run
    # -- the same reason as the hunt_scope review fix, so the training side fails closed.
    resource_weapon_upgrade = getattr(args, "resource_weapon_upgrade", "off")
    _require(resource_weapon_upgrade in ("off", "smith-v1"),
             "--resource-weapon-upgrade must be off/smith-v1")
    _require(resource_weapon_upgrade == "off" or resource_protocol == "l2-town-v1",
             "--resource-weapon-upgrade smith-v1 requires --resource-protocol l2-town-v1")
    _require(resource_weapon_upgrade == "off"
             or resource_service_policy == "sustain-loot-v1",
             "--resource-weapon-upgrade smith-v1 requires --resource-service-policy "
             "sustain-loot-v1 (the smith leg rides the loot itinerary)")
    _require(resource_weapon_upgrade == "off" or resource_purchase_mode == "full",
             "--resource-weapon-upgrade smith-v1 requires --resource-purchase-mode full")
    _require(resource_weapon_upgrade == "off" or args.worker or args.options,
             "--resource-weapon-upgrade requires --worker/--options")
    # R18-B5 (2026-09-07): hunt_scope is a DiabloGymEnv-level switch (the env.py constructor
    # does not require a resource protocol, so we do not over-tighten here), but on the training side it only passes through the **env_kwargs of
    # WorkerWindowEnv/OptionsEnv, so it is still limited to --worker/--options.
    hunt_scope = getattr(args, "hunt_scope", "all")
    _require(hunt_scope in ("all", "l1-only"),
             "--hunt-scope must be all/l1-only")
    _require(hunt_scope == "all" or args.worker or args.options,
             "--hunt-scope l1-only requires --worker/--options")
    # Review fix (same day): the only consumer of l1-only is the gate of the a10 full-map monster search
    # (`_explore_global_hunt and self._hunt_allowed_here(raw)` in env.py),
    # which is a guaranteed no-op when hunting is off. The contract and archive identity would still record this law, so the identity
    # would vouch for a law that never ran, and a pure bookkeeping difference would get later continuations ruled as environment drift.
    # Same reason as every other item in this section: silently ineffective is much worse than an early error.
    _require(hunt_scope == "all" or bool(getattr(args, "explore_global_hunt", False)),
             "--hunt-scope l1-only requires --explore-global-hunt")
    progress_far_tiles = int(getattr(args, "progress_far_tiles", 0))
    _require(
        progress_far_tiles >= 0,
        "--progress-far-tiles must not be negative",
    )
    farm_scene_cap = int(getattr(
        args, "farm_scene_cap", _FARM_SCENE_CAP_DEFAULT))
    _require(
        farm_scene_cap > 0,
        "--farm-scene-cap must be > 0",
    )
    _require(
        args.worker or args.options
        or (
            farm_scene_cap == _FARM_SCENE_CAP_DEFAULT
            and not bool(getattr(
                args, "reset_layer_clock_on_window", False))
        ),
        "--farm-scene-cap/--reset-layer-clock-on-window are OptionsEnv classroom"
        " parameters and only apply to --worker/--options",
    )
    _require(
        getattr(args, "worker_learning_window_scope", "farm-only") in {
            "farm-only", "farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE
        },
        "--worker-learning-window-scope only allows farm-only/farm-dive-v1/earned-dive-suffix-v1",
    )
    _require(
        args.worker
        or getattr(
            args, "worker_learning_window_scope", "farm-only")
        == "farm-only",
        "--worker-learning-window-scope only applies to --worker",
    )
    legacy_policy_view = bool(getattr(
        args, "legacy_worker_policy_observation_view", False))
    explicit_worker_view = getattr(
        args, "worker_policy_observation_view", None)
    _require(
        explicit_worker_view in {
            None,
            _WORKER_VIEW_LEGACY_V3,
            _WORKER_VIEW_DUAL_V4_ASYMMETRIC,
            _WORKER_VIEW_DUAL_V5_WINDOW_MODE,
        },
        "--worker-policy-observation-view only allows "
        "legacy-v3/dual-v4-asymmetric-v3/dual-v5-window-mode-v1",
    )
    # R13 cross gate (v0.3 revision): the live-DIVE classroom requires the dual-v4 view -- after the autonomy
    # handover the worker mask features (617-632) naturally carry the window-mode signal (m[11] can be 1 only
    # inside live DIVE windows), so the v4 observation semantics reflect the new rules faithfully, with no aliasing blind spot;
    # an explicit mode one-hot (v5) is kept in reserve for R13.2.
    _require(
        getattr(args, "worker_learning_window_scope", "farm-only")
        == "farm-only"
        or explicit_worker_view == _WORKER_VIEW_DUAL_V4_ASYMMETRIC,
        "live-DIVE worker scope must be paired with "
        "--worker-policy-observation-view dual-v4-asymmetric-v3",
    )
    _require(
        args.worker or not legacy_policy_view,
        "--legacy-worker-policy-observation-view only applies to --worker",
    )
    _require(
        args.worker or explicit_worker_view is None,
        "--worker-policy-observation-view only applies to --worker",
    )
    _require(
        not (legacy_policy_view and explicit_worker_view is not None),
        "the old legacy Worker flag and an explicit worker observation view are mutually exclusive",
    )
    manager_policy_view = getattr(
        args, "manager_policy_observation_view", "raw-v4")
    _require(
        manager_policy_view in {"raw-v4", "legacy-v3"},
        "--manager-policy-observation-view only allows raw-v4/legacy-v3",
    )
    _require(
        args.options or manager_policy_view == "raw-v4",
        "--manager-policy-observation-view=legacy-v3 only applies to --options; "
        "the frozen M29 view of --worker is bound by WorkerWindowEnv",
    )
    if args.worker:
        if _bc_aux_structural_active(args):
            _require(
                not legacy_policy_view
                and explicit_worker_view is None,
                "A12 custom policy must read legacy-v3-a12-overlay; "
                "the old actor/value are decoded to the full v3 inside the policy",
            )
            _require(
                _requested_drink_sovereignty(args) is not False,
                "A12 custom policy must enable drink sovereignty; "
                "--no-drink-sovereignty would close m[12] permanently and cut off online learning",
            )
        elif explicit_worker_view == _WORKER_VIEW_DUAL_V4_ASYMMETRIC:
            _require(
                bool(args.resume_from or getattr(args, "resource_warm_start", None)),
                "dual-v4-asymmetric-v3 must launch from a registered Worker checkpoint, "
                "an explicit environment-restart parameter continuation or an independent resource warm-start",
            )
        elif explicit_worker_view == _WORKER_VIEW_DUAL_V5_WINDOW_MODE:
            _require(
                bool(args.resume_from),
                "dual-v5-window-mode-v1 must continue from a v4->v5 migrated enrolment zip"
                " (zero-padded widening; the D0 gate keeps argmax identical)",
            )
        else:
            _require(
                legacy_policy_view
                or explicit_worker_view == _WORKER_VIEW_LEGACY_V3,
                "an ordinary Worker policy must explicitly carry the"
                " legacy-v3 observation view, "
                "so training and deployment use the same protocol-v3 actor/value input",
            )
    _require(
        getattr(args, "artifact_scope", "production")
        in _ARTIFACT_SCOPE_RESULTS,
        "--artifact-scope only allows development/candidate/production",
    )
    _require(args.distill_ce_probe_every == 0
             or (args.worker and args.algo == "mppo"),
             "--distill-ce-probe-every only applies to --worker --algo mppo"
             " (the probe needs a Leashed teacher)")
    _require(args.drywin_metrics_every == 0 or args.worker,
             "--drywin-metrics-every only applies to --worker")
    if args.run_name is not None:
        _require(bool(args.run_name) and pathlib.Path(args.run_name).name == args.run_name
                 and args.run_name not in (".", ".."),
                 "--run-name must be a single directory name without path separators")
    if args.seed is not None:
        _require(0 <= args.seed < 2**32, "--seed must lie in [0, 2**32)")
        _require(args.seed + args.num_envs - 1 < 2**32,
                 "--seed + num_envs - 1 must be less than 2**32")

    modes = int(args.worker) + int(args.options) + int(args.flat_clock)
    _require(modes <= 1, "--worker/--options/--flat-clock are mutually exclusive")
    # E1 two-flag mutual-exclusion assertion (following engineering B1) + the four gates' exclusion/mode gate: predicate = skip_dry or schedule.
    _require(not (args.skip_dry and args.dry_curriculum_schedule),
             "--skip-dry and --dry-curriculum-schedule are mutually exclusive")
    _require(not _dry_window_mechanism_active(args) or args.worker,
             "--skip-dry/--dry-curriculum-schedule can only be used with --worker")
    if args.dry_curriculum_schedule:
        curriculum_table = _parse_dry_curriculum_schedule(args.dry_curriculum_schedule)
        _require(len(curriculum_table) * rollout_quantum >= args.total_steps,
                 f"--dry-curriculum-schedule p table has {len(curriculum_table)} entries,"
                 f" not enough to cover this leg's {args.total_steps // rollout_quantum} rollouts"
                 " (leg-relative anchoring forbids out-of-range clamping)")
    _require(not args.worker_npz or args.options, "--worker-npz can only be used with --options")
    # R9: zip worker assembly point (certified release artifact) -- exclusion/mode gates shaped like npz.
    _require(not args.worker_zip or args.options, "--worker-zip can only be used with --options")
    _require(not (args.worker_zip and args.worker_npz),
             "--worker-zip and --worker-npz are mutually exclusive (one assembly point mounts one worker)")
    _require(not args.worker_zip_sha256 or args.worker_zip,
             "--worker-zip-sha256 must be used with --worker-zip")
    _require(args.worker_zip_sha256 is None or _is_sha256(args.worker_zip_sha256),
             f"--worker-zip-sha256 must be 64 lowercase hex digits: {args.worker_zip_sha256!r}")
    # R9 curriculum knobs: --options only; the form flag must not appear without the curriculum flag.
    _require(not args.deep_start_curriculum or args.options,
             "--deep-start-curriculum can only be used with --options")
    _require(args.deep_start_form is None or bool(args.deep_start_curriculum),
             "--deep-start-form must be used with --deep-start-curriculum")
    if args.deep_start_curriculum:
        _parse_deep_start_curriculum(args.deep_start_curriculum)
    _require(not args.teacher_override or (args.resume_from and args.worker),
             "--teacher-override can only be used with a worker-side --resume-from")
    _require(not args.allow_manager_change or (args.resume_from and args.worker),
             "--allow-manager-change only allows an explicit manager change on worker resume")
    _require(not args.allow_legacy_resume or args.resume_from,
             "--allow-legacy-resume can only be used with --resume-from")
    allow_environment_restart_resume = bool(getattr(
        args, "allow_environment_restart_resume", False))
    _require(
        not allow_environment_restart_resume
        or (args.worker and args.algo == "mppo" and bool(args.resume_from)),
        "--allow-environment-restart-resume only applies to "
        "--worker --algo mppo --resume-from",
    )
    _require(
        not (args.worker and args.allow_legacy_resume)
        or reset_worker_critic,
        "a one-off migration of an old Worker checkpoint must also pass"
        " --reset-worker-critic; clearing the optimizer alone cannot fix the window-termination critic",
    )
    _require(args.freeze_policy_steps == 0 or args.bc_init,
             "--freeze-policy-steps > 0 requires --bc-init")
    _require(args.bc_init or args.init_source == "bc",
             "--init-source checkpoint must be used with --bc-init")
    _require(args.distill_beta == 0 or (args.worker and args.algo == "mppo"),
             "--distill-beta > 0 only applies to --worker --algo mppo")
    _require(not (args.calib_probes or args.calib_record_only)
             or (args.worker and args.algo == "mppo"),
             "G-CAL parameters only apply to --worker --algo mppo")
    # E3 4B: the two flags do not force each other (either alone raises no error); present = λ_bc>0 and demos given.
    _require(math.isfinite(args.bc_aux_lambda) and args.bc_aux_lambda >= 0,
             "--bc-aux-lambda must be a finite non-negative number")
    _require(not args.bc_aux_graft or bool(args.bc_aux_demos),
             "--bc-aux-graft requires --bc-aux-demos as well")
    _require(
        not _bc_aux_structural_active(args)
        or math.isclose(float(args.bc_aux_lambda), 0.0,
                        rel_tol=0.0, abs_tol=0.0),
        "rev10 contextual mixture adapter requires --bc-aux-lambda=0; "
        "reviving the unreachable gradient auxiliary tug-of-war is forbidden")
    _require(
        not (args.bc_aux_lambda > 0 and bool(args.bc_aux_demos)),
        "rev5 gradient bc_aux was rejected by measurement; use"
        " --bc-aux-graft --bc-aux-lambda 0")
    _require(not _bc_aux_active(args) or (args.worker and args.algo == "mppo"),
             "the 4B auxiliary pathway "
             "only applies to --worker --algo mppo")   # same gate shape as --distill-beta (discretionary note)
    _require(not args.bc_aux_liveness_preflight
             or (_bc_aux_active(args) and bool(args.resume_from)),
             "--bc-aux-liveness-preflight only applies to a --resume-from"
             " production leg with bc_aux present")
    _require(
        getattr(args, "artifact_scope", "production") == "production"
        or not _bc_aux_active(args),
        "non-production artifacts must not consume the bc_aux final-heldout release gate; "
        "turn off bc_aux or use the independent nested-validation development path",
    )
    if _bc_aux_active(args):
        _require(bool(args.resume_from),
                 "a formal leg with bc_aux present must resume from a frozen worker checkpoint")
        _require(args.bc_aux_liveness_preflight,
                 "a formal leg with bc_aux present must carry --bc-aux-liveness-preflight")
        _require(args.distill_beta > 0,
                 "a formal leg with bc_aux present must keep a nonzero KING/BC distillation leash")
        _require(_bc_aux_structural_active(args),
                 "formal bc_aux only accepts the rev10 contextual mixture adapter")
        resume_metadata = _validate_leashed_checkpoint(args.resume_from)
        saved_adapter = resume_metadata.get("_bc_aux_circuit_spec")
        if saved_adapter is None:
            _require(args.reset_optimizer,
                     "the first a12 actor widening requires --reset-optimizer; "
                     "binding old Adam moments to the new topology is forbidden")
        else:
            _require(saved_adapter == _bc_aux_circuit_spec(),
                     "continuation checkpoint a12 mixture spec drift")
            _require(not args.reset_optimizer,
                     "an existing a12 mixture continuation forbids --reset-optimizer; "
                     "the ε PPO has learned, the Adam moments and the exploration counts must be kept")
        _require(not args.calib_record_only,
                 "with bc_aux present, --calib-record-only may not bypass the G-CAL hard gate")
        _require(pathlib.Path(args.bc_aux_demos).is_file(),
                 f"4B v2 demo set (bc-worker-v2) does not exist: {args.bc_aux_demos}")
        # dedicated v2 validator + positive/negative calibration bank fail-loud (mirrors the _precheck precedent,
        # load then discard; zero intrusion when absent -- not even the file's existence is checked)
        expected_manager_sha256 = _capture_file_sha256(
            args.manager_npz, "manager_npz")
        _x, _y, _, _masks, _ = _load_bc_aux_demos_v2(
            args.bc_aux_demos,
            expected_manager_sha256=expected_manager_sha256)
    _require(args.arch != "attn" or not (args.worker or args.options or args.flat_clock),
             "EntityAttention only supports the 295-dim flat observation")

    if args.resume_from:
        _require((args.worker or args.options) and args.algo == "mppo",
                 "--resume-from only supports worker/options mppo checkpoints")
        _require(not args.bc_init and args.freeze_policy_steps == 0,
                 "--resume-from must not be used with --bc-init/--freeze-policy-steps")
        _require(_checkpoint_path(args.resume_from).is_file(),
                 f"resume checkpoint does not exist: {args.resume_from}")
        if args.worker:
            resume_metadata = _validate_leashed_checkpoint(args.resume_from)
            contracted = resume_metadata.get("diablogym_contract") is not None
            _require(
                contracted is allow_environment_restart_resume,
                "a Worker checkpoint with a training_contract can only continue parameters/Adam with an explicit "
                "--allow-environment-restart-resume; "
                "an old launch checkpoint must go through the separate legacy migration gate",
            )
        else:
            # v31 manager continuation entry: the generic checkpoint gate (CRC/key members/steps/finite weights);
            # distill_beta is a marker specific to Leashed workers, so manager checkpoints do not assert it.
            _validate_checkpoint_file(args.resume_from)
    if args.bc_init:
        _require(pathlib.Path(args.bc_init).is_file(), f"BC weights do not exist: {args.bc_init}")
        if args.init_source == "bc":
            gate = ("data_gate" if args.worker else "hypothesis" if args.options
                    else "memoryless_hypothesis")
            _validate_bc_report(pathlib.Path(args.bc_init), gate)
        else:
            _validate_export_manifest(pathlib.Path(args.bc_init))
    if args.teacher_override:
        _require(pathlib.Path(args.teacher_override).is_file(),
                 f"teacher override file does not exist: {args.teacher_override}")
        _validate_export_manifest(pathlib.Path(args.teacher_override))
    if args.worker:
        # R16 amendment: episode length 3000 (old rule) or >=6000 (new rule; audit C6/C11: a 1800-step scene
        # budget + a 3000-step episode length drove the worker downstairs; the whitelist already allows max_steps drift).
        _require(args.algo == "mppo" and args.gamma == 1.0
                 and (args.max_steps == 3000 or args.max_steps >= 6000),
                 "PREREG-v23/R16: --worker requires --algo mppo --gamma 1.0 "
                 "--max-steps 3000 or >=6000")
        if getattr(args, "manager_heuristic", None):
            # --manager-npz has an argparse default path; in scripted-coach mode it is cleared,
            # so the single source of truth is the heuristic (if both are passed explicitly this line wins, and the contract records only the heuristic).
            args.manager_npz = None
        else:
            _require(pathlib.Path(args.manager_npz).is_file(),
                     f"manager npz does not exist: {args.manager_npz}")
        if args.distill_beta > 0 and not args.resume_from:
            _require(pathlib.Path(args.teacher_sd).is_file(),
                     f"teacher state_dict does not exist: {args.teacher_sd}")
            _validate_bc_report(pathlib.Path(args.teacher_sd), "data_gate")
        # E1 four gates: demos/BC preflight gate: skip_dry or schedule (predicate inside the helper, assertion unchanged)
        _precheck_dry_window_demos(args)
    if args.options:
        _require(args.algo == "mppo" and args.gamma == 1.0
                 and (args.max_steps == 3000 or args.max_steps >= 6000),
                 "PREREG-v25/R16: --options requires --algo mppo --gamma 1.0 "
                 "--max-steps 3000 or >=6000")
        if args.worker_npz:
            _require(args.n_steps == 64 and args.seed is not None,
                     "PREREG-v25 D2: the re-election requires --n-steps 64 and an explicit --seed")
            _require(pathlib.Path(args.worker_npz).is_file(),
                     f"worker npz does not exist: {args.worker_npz}")
        if args.worker_zip:
            _require(args.n_steps == 64 and args.seed is not None,
                     "PREREG-R9: a zip worker re-election requires --n-steps 64 and an explicit --seed")
            _require(pathlib.Path(args.worker_zip).is_file(),
                     f"worker zip does not exist: {args.worker_zip}")
    if args.seed is not None:
        from diablogym.worker_env import is_reserved_train_seed

        _require(not any(
            is_reserved_train_seed(args.seed + rank)
            for rank in range(args.num_envs)),
                 "seed discipline: --seed + actual env rank collides with a registered BC/evaluation pool")

    try:
        probes = [int(x) for x in args.calib_probes.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("--calib-probes must be comma-separated integers") from exc
    _require(all(p >= 0 for p in probes), "--calib-probes must not contain negative numbers")
    _select_batch_size(args.n_steps, args.num_envs)


def _prepare_run_dir(run_dir: pathlib.Path, resume_from: str | None,
                     protected_inputs=()) -> None:
    """On a same-name rerun, preserve old artifacts while keeping progress/tb/ckpt out of the new attempt."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_root = run_dir.resolve()
    protected = [pathlib.Path(p).resolve() for p in protected_inputs if p]
    in_output_tree = [p for p in protected
                      if p == run_root or p.is_relative_to(run_root)]
    _require(not in_output_tree,
             "training inputs cannot live inside this run's output directory; copy them to train/models "
             f"or a separate inputs directory first: {in_output_tree}")
    existing = [run_dir / name for name in _RUN_ARTIFACTS if (run_dir / name).exists()]
    if not existing:
        return
    if resume_from:
        source = _checkpoint_path(resume_from).resolve()
        _require(all(p.resolve() != source
                     and not (p.is_dir() and source.is_relative_to(p.resolve()))
                     for p in existing),
                 "cannot resume in place from the model_final of the same run_dir; use another run-name")
    archive = run_dir / "_attempts" / f"pre-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
    archive.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))
    print(f"   archived old artifacts of the same-name run: {archive}")


class _RunLock:
    """Process-level exclusive lock; the kernel releases the flock automatically on crash/SIGKILL."""

    def __init__(self, run_dir: pathlib.Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / ".run.lock"
        self._file = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.seek(0)
            owner = self._file.read().strip() or "unknown"
            self._file.close()
            raise RuntimeError(f"run_dir is being used by another training process: {run_dir} ({owner})") from exc
        self._file.seek(0)
        self._file.truncate()
        self._file.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if getattr(self, "_file", None) is None or self._file.closed:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()

    def __del__(self):
        self.close()


class _TrainingResources:
    """Own the run lock and VecEnv from acquisition through every failure path."""

    def __init__(self):
        self.run_lock = None
        self.vec_env = None

    @staticmethod
    def _reap_failed_vec_env(processes, remotes) -> dict:
        """Bounded recovery of this VecEnv's handles after normal close failed."""
        import time

        result = {"terminate": [], "kill": [], "remaining": [], "errors": []}

        def attempt(label, operation):
            try:
                return operation()
            except Exception as exc:
                result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
                return None

        # Never recv again: a dead worker may have left waiting=True and a pipe
        # with no reply. Close only the endpoints owned by this VecEnv.
        for remote in remotes:
            attempt("remote.close", remote.close)
        for operation in ("terminate", "kill"):
            for process in processes:
                if attempt("process.is_alive", process.is_alive):
                    def signal_owned(process=process, operation=operation):
                        getattr(process, operation)()
                        result[operation].append(process.pid)
                    attempt(f"process.{operation}", signal_owned)
            # One deadline for all workers, not a fresh allowance per worker.
            deadline = time.monotonic() + 2.0
            for process in processes:
                attempt("process.join", lambda process=process: process.join(
                    timeout=max(0.0, deadline - time.monotonic())))
        for process in processes:
            if attempt("process.is_alive", process.is_alive):
                result["remaining"].append(process.pid)
        return result

    @staticmethod
    def _close_vec_env(vec_env) -> None:
        import signal
        import threading

        # Preserve the exact owned handles even if close fails partway through.
        # No global child enumeration or process-group signalling is permitted.
        processes = tuple(getattr(vec_env, "processes", ()))
        remotes = tuple(getattr(vec_env, "remotes", ()))
        close_error = None
        # SIGALRM is process-global and can only be installed from the main
        # thread.  Keep embedding/tests safe by falling back to an ordinary
        # close outside it, and restore any host handler/timer afterwards.
        armed = threading.current_thread() is threading.main_thread()
        previous_handler = previous_alarm = None

        def _close_timeout(*_):
            raise TimeoutError("vec_env.close() timed out (worker probably dead)")

        if armed:
            previous_handler = signal.getsignal(signal.SIGALRM)
            previous_alarm = signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, _close_timeout)
                signal.alarm(20)
            except Exception:
                # Do not leave the caller's timer cancelled if signal setup is
                # unavailable in an unusual embedding environment.
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_alarm:
                    signal.alarm(previous_alarm)
                armed = False
        try:
            vec_env.close()
        except Exception as exc:
            close_error = exc
        finally:
            if armed:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_alarm:
                    signal.alarm(previous_alarm)
        if close_error is not None:
            print(f"vec_env.close error: {type(close_error).__name__}: {close_error}")
            if processes:
                recovery = _TrainingResources._reap_failed_vec_env(processes, remotes)
                print(f"vec_env.close own-process reaping: {json.dumps(recovery, sort_keys=True)}")

    def close(self) -> None:
        # Clear ownership before calling user/library cleanup so a recursive
        # or repeated close remains idempotent even when close itself raises.
        vec_env, self.vec_env = self.vec_env, None
        run_lock, self.run_lock = self.run_lock, None
        try:
            if vec_env is not None:
                self._close_vec_env(vec_env)
        finally:
            if run_lock is not None:
                run_lock.close()


def _atomic_save_model(model, destination: str | pathlib.Path) -> pathlib.Path:
    """Write a unique temporary zip first, fully check CRC/finite, then atomically publish the canonical."""
    final = pathlib.Path(destination)
    if final.suffix.lower() != ".zip":
        final = pathlib.Path(f"{final}.zip")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(
        f".{final.stem}.{os.getpid()}.{time.time_ns()}.tmp.zip")
    try:
        model.save(str(tmp))
        # SB3 close() only hands the bytes to the kernel; force them to disk before validation/atomic replace,
        # so a power loss cannot leave a checkpoint whose name is published but whose data was never persisted.
        with open(tmp, "rb") as stream:
            os.fsync(stream.fileno())
        _validate_checkpoint_file(tmp, require_leashed=hasattr(model, "distill_beta"))
        os.replace(tmp, final)
        try:
            directory_fd = os.open(final.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some non-POSIX/network file systems do not support directory fsync; the file itself
            # is already persisted, and the atomicity of replace is unaffected.
            pass
    finally:
        tmp.unlink(missing_ok=True)
    return final


def _validate_worker_bc_evidence(rec: dict, demos_payload: bytes,
                                 policy_payload: bytes) -> None:
    """Recompute the worker holdout gate from its immutable evidence bytes."""
    import numpy as np
    import torch

    try:
        with np.load(io.BytesIO(demos_payload), allow_pickle=False) as archive:
            _require(set(archive.files) == {"X", "Y", "episode_id"},
                     "BC worker demos.npz fields must be exactly X/Y/episode_id")
            x = archive["X"]
            y = archive["Y"]
            episode_id = archive["episode_id"]
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("BC worker"):
            raise
        raise ValueError("BC worker demos.npz unparseable") from exc

    pairs = rec["pairs"]
    _require(x.ndim == 2 and x.shape == (pairs, 298)
             and x.dtype == np.float32,
             f"BC worker X shape/dtype disagrees with the report: {x.shape}/{x.dtype}")
    _require(y.shape == (pairs,) and y.dtype == np.int64,
             f"BC worker Y shape/dtype disagrees with the report: {y.shape}/{y.dtype}")
    _require(episode_id.shape == (pairs,) and episode_id.dtype == np.int64,
             "BC worker episode_id shape/dtype disagrees with the report")
    _require(np.isfinite(x).all(), "BC worker demos X contains NaN/Inf")
    from diablogym.worker_env import legacy_worker_policy_observation_view

    _require(
        np.array_equal(x, legacy_worker_policy_observation_view(x)),
        "BC worker demos X is not the canonical protocol-v3 policy view; "
        "suspect raw packed-belt/signed-latch demos used for a legacy actor",
    )
    _require(bool(((0 <= y) & (y < 15)).all()),
             "BC worker demos Y contains out-of-range actions")
    _require(not np.isin(y, _WORKER_BC_FORBIDDEN_ACTIONS).any(),
             "BC worker demos Y contains forbidden actions 11/12 (11 always masked; 12 never collected after the teacher's drain)")
    _require(bool((episode_id >= 0).all()),
             "BC worker demos episode_id must not be negative")
    episodes = np.unique(episode_id)
    expected_demo_seeds = np.asarray(_WORKER_BC_DEMO_SEEDS, dtype=np.int64)
    _require(np.array_equal(episodes, expected_demo_seeds),
             "BC worker demos must exactly cover the fixed demo seeds "
             "2142000..2142127, at least one pair each")
    action14 = y == 14
    action14_episodes = int(np.unique(episode_id[action14]).size)
    _require(
        int(action14.sum()) >= _WORKER_BC_MIN_ACTION14_LABELS
        and action14_episodes >= _WORKER_BC_MIN_ACTION14_EPISODES,
        "BC worker demos a14 strict-upgrade coverage insufficient: "
        f"labels={int(action14.sum())},episodes={action14_episodes}",
    )
    order = np.random.default_rng(_BC_FINAL_SPLIT_SEED).permutation(episodes)
    n_holdout = max(1, int(round(len(order) * 0.1)))
    expected_episodes = np.sort(order[:n_holdout])
    reported_episodes = np.asarray(rec["held_out_episodes"], dtype=np.int64)
    _require(np.array_equal(reported_episodes, expected_episodes),
             "BC worker held_out_episodes disagree with the deterministic whole-episode split")
    holdout_indices = np.flatnonzero(np.isin(episode_id, expected_episodes))
    _require(len(holdout_indices) == rec["held_out_pairs"],
             "BC worker held_out_pairs disagree with the demos episode_id recomputation")
    _require(0 < len(holdout_indices) < pairs,
             "BC worker deterministic whole-episode split produced an empty training set or empty held-out")

    try:
        state = torch.load(io.BytesIO(policy_payload), map_location="cpu",
                           weights_only=True)
    except Exception as exc:
        raise ValueError("BC worker policy state_dict unparseable") from exc
    required = _POLICY_HEAD_KEYS
    _require(isinstance(state, dict) and set(state) == set(required),
             "BC worker policy state_dict fields must exactly match the policy head")
    tensors = [state[key] for key in required]
    _require(all(isinstance(value, torch.Tensor) for value in tensors),
             "BC worker policy state_dict contains non-Tensor values")
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.shape == (64, 298) and b0.shape == (64,)
             and w1.shape == (64, 64) and b1.shape == (64,)
             and wa.shape == (15, 64) and ba.shape == (15,),
             "BC worker policy must be 298->64->64->15")
    _require(all(value.dtype == torch.float32 for value in tensors),
             "BC worker policy head dtype must be float32")
    _require(all(torch.isfinite(value).all().item() for value in tensors),
             "BC worker policy head contains NaN/Inf")

    # Mirror bc_worker.train_bc exactly: one CPU batch over X[holdout].  Using
    # different chunk sizes can select a different GEMM kernel and flip an
    # argmax at a near-tie even though the policy bytes are identical.
    with torch.no_grad():
        obs = torch.from_numpy(np.ascontiguousarray(x[holdout_indices]))
        hidden = torch.tanh(torch.nn.functional.linear(obs, w0, b0))
        hidden = torch.tanh(torch.nn.functional.linear(hidden, w1, b1))
        logits = torch.nn.functional.linear(hidden, wa, ba)
        pred = logits.argmax(1).cpu().numpy()
    heldout_y = y[holdout_indices]
    top1 = round(float((pred == heldout_y).mean()), 4)
    _require(rec["held_out_top1"] == top1,
             "BC worker held_out_top1 disagrees with the demos/policy recomputation: "
             f"{rec['held_out_top1']} != {top1}")

    full_counts = np.bincount(y, minlength=15)
    gated_classes = np.asarray(sorted({
        *map(int, np.flatnonzero(full_counts >= 300)),
        *_WORKER_BC_REQUIRED_RECALL_ACTIONS,
    }), dtype=np.int64)
    reported_recalls = {int(key): value
                        for key, value in rec["class_recalls"].items()}
    _require(set(reported_recalls) == set(map(int, gated_classes)),
             "BC worker class_recalls class set disagrees with the full demos counts")
    for class_id in gated_classes:
        mask = heldout_y == class_id
        recall = (float((pred[mask] == class_id).mean())
                  if mask.any() else 0.0)
        _require(math.isclose(
                     float(reported_recalls[int(class_id)]), recall,
                     rel_tol=0.0, abs_tol=1e-15),
                 "BC worker class_recalls disagree with the demos/policy recomputation: "
                 f"class={class_id}, {reported_recalls[int(class_id)]} != {recall}")


def _recompute_replay_bc_evidence(required_gate: str,
                                  policy_payload: bytes) -> dict:
    """Execute the deterministic BC demo/replay pools from frozen weights.

    Aggregate JSON is not evidence: a random policy plus edited means and SHA
    fields used to pass.  These gates are infrequent, pre-training operations,
    so correctness wins over the roughly minute-scale deterministic replay.
    """
    import numpy as np
    import torch

    dimensions = {
        "hypothesis": (303, 3),
        "memoryless_hypothesis": (296, 15),
    }
    _require(required_gate in dimensions, f"unknown replay BC gate: {required_gate}")
    obs_dim, action_dim = dimensions[required_gate]
    try:
        state = torch.load(io.BytesIO(policy_payload), map_location="cpu",
                           weights_only=True)
    except Exception as exc:
        raise ValueError("BC replay policy state_dict unparseable") from exc
    _require(isinstance(state, dict) and set(state) == set(_POLICY_HEAD_KEYS),
             "BC replay policy state_dict fields must exactly match the policy head")
    tensors = [state[key] for key in _POLICY_HEAD_KEYS]
    _require(all(isinstance(value, torch.Tensor) for value in tensors),
             "BC replay policy state_dict contains non-Tensor values")
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.shape == (64, obs_dim) and b0.shape == (64,)
             and w1.shape == (64, 64) and b1.shape == (64,)
             and wa.shape == (action_dim, 64) and ba.shape == (action_dim,),
             f"BC replay policy must be {obs_dim}->64->64->{action_dim}")
    _require(all(value.dtype == torch.float32 for value in tensors),
             "BC replay policy head dtype must be float32")
    _require(all(torch.isfinite(value).all().item() for value in tensors),
             "BC replay policy head contains NaN/Inf")

    def policy_action(obs, mask) -> int:
        vector = np.asarray(obs, dtype=np.float32)
        _require(vector.shape == (obs_dim,),
                 f"BC replay observation dimension invalid: {vector.shape} != {(obs_dim,)}")
        valid = np.asarray(mask, dtype=bool)
        _require(valid.shape == (action_dim,) and bool(valid.any()),
                 "BC replay action mask dimension invalid or all False")
        with torch.no_grad():
            x = torch.from_numpy(vector)
            hidden = torch.tanh(torch.nn.functional.linear(x, w0, b0))
            hidden = torch.tanh(torch.nn.functional.linear(hidden, w1, b1))
            logits = torch.nn.functional.linear(hidden, wa, ba)
            logits = logits.masked_fill(~torch.from_numpy(valid), -torch.inf)
            return int(logits.argmax().item())

    if required_gate == "hypothesis":
        from diablogym import OptionsEnv
        from diablogym.options_env import DIVE, FARM

        env = OptionsEnv(max_steps=3000)

        def teacher_action(manager_env, _obs, _mask) -> int:
            raw = manager_env.env._raw
            return (DIVE if (manager_env.exhausted
                             or raw["char_level"] >= raw["dungeon_level"] + 2)
                    else FARM)

        def rollout(chooser, seed: int) -> tuple[float, int]:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            total = 0.0
            pairs = 0
            while not (done or trunc):
                mask = env.action_masks()
                action = _masked_action_or_first_legal(
                    chooser(env, obs, mask),
                    mask,
                    n_actions=3,
                    label="BC manager replay",
                )
                obs, reward, done, trunc, _ = env.step(action)
                total += float(reward)
                pairs += 1
            return total, pairs

        try:
            demo = [rollout(teacher_action, seed)
                    for seed in _WORKER_BC_DEMO_SEEDS]
            replay = [rollout(
                lambda _env, obs, mask: policy_action(obs, mask), seed)[0]
                for seed in _BC_REPLAY_SEEDS]
            teacher_replay = [rollout(teacher_action, seed)[0]
                              for seed in _BC_REPLAY_SEEDS]
        finally:
            env.close()
        teacher_demo_mean = sum(value for value, _ in demo) / len(demo)
        bc_mean = sum(replay) / len(replay)
        teacher_mean = sum(teacher_replay) / len(teacher_replay)
        _require(teacher_mean > 0, "BC manager recomputed teacher replay not positive")
        return {
            "pairs": sum(count for _, count in demo),
            "teacher_demo_mean": teacher_demo_mean,
            "bc_replay_7000": bc_mean,
            "teacher_7000": teacher_mean,
            "ratio": bc_mean / teacher_mean,
        }

    from diablogym import DiabloGymEnv, StagnationClockWrapper
    from diablogym.options_env import KILL_PATIENCE, dispatch

    env = StagnationClockWrapper(DiabloGymEnv(
        ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
        include_raw=False, descend_ladder=True, death_ladder=True))

    def teacher_action(flat_env, _obs) -> int:
        raw = flat_env.env._raw
        if flat_env._clock >= KILL_PATIENCE:
            return 11
        mode = ("dive" if raw["char_level"] >= raw["dungeon_level"] + 2
                else "farm")
        masks, nearest = flat_env.env.controller_action_context()
        return dispatch(
            mode, raw, bool(masks[14]), action_mask=masks,
            nearest_engageable_distance=nearest)

    def rollout(chooser, seed: int) -> tuple[float, int]:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        total = 0.0
        pairs = 0
        while not (done or trunc):
            action = int(chooser(env, obs))
            obs, reward, done, trunc, _ = env.step(action)
            total += float(reward)
            pairs += 1
        return total, pairs

    try:
        demo = [rollout(teacher_action, seed)
                for seed in _WORKER_BC_DEMO_SEEDS]
        replay = [rollout(
            lambda flat_env, obs: policy_action(
                obs, flat_env.env.action_masks()), seed)[0]
            for seed in _BC_REPLAY_SEEDS]
        teacher_replay = [rollout(teacher_action, seed)[0]
                          for seed in _BC_REPLAY_SEEDS]
    finally:
        env.close()
    teacher_demo_mean = sum(value for value, _ in demo) / len(demo)
    bc_mean = sum(replay) / len(replay)
    teacher_mean = sum(teacher_replay) / len(teacher_replay)
    _require(teacher_mean > 0, "BC flat recomputed teacher replay not positive")
    return {
        "pairs": sum(count for _, count in demo),
        "teacher_mean_demo": teacher_demo_mean,
        "bc_replay_mean_7000s": bc_mean,
        "teacher_replay_mean_7000s": teacher_mean,
        "ratio": bc_mean / teacher_mean,
    }


def _validate_bc_report(p: pathlib.Path, required_gate: str,
                        expected_implementation_sha256: str | None = None,
                        *, policy_payload: bytes | None = None,
                        report_payload: bytes | None = None,
                        verify_replay: bool = True) -> dict:
    """Validate the BC gate and bind the weights, generator and the full training runtime.

    If the caller has already frozen the policy/report bytes for TOCTOU safety, it must pass them via payload;
    the validator will not read the path again. Training loading and assembled evaluation therefore share exactly the same
    schema/metric/provenance gates instead of each maintaining a slowly drifting subset.
    """
    from eval_contract import EvalContractError, PROTOCOL_VERSION, strict_json_loads

    _require(required_gate in {"data_gate", "hypothesis", "memoryless_hypothesis"},
             f"unknown BC gate: {required_gate}")
    report = p.with_name("bc_report.json")
    try:
        frozen_report = (report.read_bytes() if report_payload is None
                         else report_payload)
        rec = strict_json_loads(frozen_report)
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"BC gate report missing/unreadable: {report}") from exc
    _require(isinstance(rec, dict), f"BC gate report must be a JSON object: {report}")
    _require(set(rec) == _BC_PASS_KEYS[required_gate],
             f"BC gate report fields/schema mismatch: {report}")
    _require(_is_plain_int(rec.get("schema_version"))
             and rec["schema_version"] == _BC_REPORT_SCHEMA_VERSION,
             f"BC gate report schema outdated: {rec.get('schema_version')!r}")
    _require(rec[required_gate] == "PASS",
             f"refusing to load BC weights that did not pass the {required_gate} gate: {rec[required_gate]!r}")
    expected_sha = rec.get("policy_sha256")
    _require(_is_sha256(expected_sha),
             f"BC gate report is missing its policy_sha256 binding: {report}")
    try:
        frozen_policy = p.read_bytes() if policy_payload is None else policy_payload
    except OSError as exc:
        raise ValueError(f"BC weights missing/unreadable: {p}") from exc
    actual_sha = hashlib.sha256(frozen_policy).hexdigest()
    _require(actual_sha == expected_sha,
             f"BC weights SHA does not match the gate report: {actual_sha} != {expected_sha}")

    # A policy hash proves which bytes were loaded, but not which world made
    # their demonstrations.  In particular, pre-v3 worker demos may contain
    # trajectories that returned to town.  Bind every PASS report to the
    # current environment/native/content bundle and the exact BC generator.
    _require(rec.get("protocol_version") == PROTOCOL_VERSION,
             f"BC report protocol outdated: {rec.get('protocol_version')!r} != {PROTOCOL_VERSION}")
    expected_impl = (expected_implementation_sha256
                     if expected_implementation_sha256 is not None
                     else _implementation_bundle_sha256())
    _require(rec.get("implementation_sha256") == expected_impl,
             "BC report implementation/engine/game-content identity disagrees with the current runtime")
    root = pathlib.Path(__file__).resolve().parents[1]
    generator_name = {
        "data_gate": "bc_worker.py",
        "hypothesis": "bc_manager.py",
        "memoryless_hypothesis": "bc_flat.py",
    }[required_gate]
    generator_sha = hashlib.sha256(
        (root / "train" / generator_name).read_bytes()).hexdigest()
    _require(rec.get("generator_sha256") == generator_sha,
             f"BC report generator has drifted: train/{generator_name}")
    if required_gate == "data_gate":
        _validate_bc_final_holdout_marker(
            p.parent, 1, _WORKER_BC_DEMO_SEEDS, rec)
        pairs = rec["pairs"]
        held_out_pairs = rec["held_out_pairs"]
        top1 = _finite_number(rec["held_out_top1"], "BC held_out_top1")
        _require(_is_plain_int(pairs) and pairs > 0
                 and _is_plain_int(held_out_pairs) and 0 < held_out_pairs < pairs,
                 "BC worker report sample count invalid")
        # A2 (approved 2026-07-27): the quality line is record-only, no verdict; here only the reading range is constrained;
        # bitwise recomputation consistency is guaranteed by _validate_worker_bc_evidence.
        _require(top1 >= 0.0 and top1 <= 1.0,
                 "BC worker held-out top-1 reading out of range")
        episodes = rec["held_out_episodes"]
        _require(isinstance(episodes, list) and episodes
                 and all(_is_plain_int(value) and value >= 0 for value in episodes)
                 and episodes == sorted(set(episodes)),
                 "BC worker held_out_episodes not canonical")
        _require(isinstance(rec["class_weighted_retry"], bool),
                 "BC worker class_weighted_retry must be a bool")
        recalls = rec["class_recalls"]
        _require(isinstance(recalls, dict), "BC worker class_recalls must be an object")
        for raw_class, raw_recall in recalls.items():
            try:
                class_id = int(raw_class)
            except (TypeError, ValueError) as exc:
                raise ValueError("BC worker class_recalls keys must be action numbers") from exc
            _require(str(class_id) == str(raw_class) and 0 <= class_id < 15,
                     f"BC worker class_recalls key invalid: {raw_class!r}")
            recall = _finite_number(raw_recall,
                                    f"BC worker class_recalls[{raw_class!r}]")
            # A2: as above, recalls are record-only readings, no verdict.
            _require(0.0 <= recall <= 1.0,
                     "BC worker class_recalls reading out of range")
        _require(_is_sha256(rec.get("demos_sha256")),
                 "BC worker report is missing demos_sha256")
        demos_path = p.with_name("demos.npz")
        try:
            demos_payload = demos_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"BC worker demo set missing/unreadable: {demos_path}") from exc
        actual_demos_sha = hashlib.sha256(demos_payload).hexdigest()
        _require(actual_demos_sha == rec["demos_sha256"],
                 "BC worker demos.npz SHA does not match the gate report: "
                 f"{actual_demos_sha} != {rec['demos_sha256']}")
        _validate_worker_bc_evidence(rec, demos_payload, frozen_policy)
        manager = root / "train" / "models" / "v22-h-manager" / "policy.npz"
        manager_sha = hashlib.sha256(manager.read_bytes()).hexdigest()
        _require(rec.get("manager_npz_sha256") == manager_sha,
                 "BC worker report not bound to the current frozen manager NPZ")
    else:
        pairs = rec["pairs"]
        ratio = _finite_number(rec["ratio"], "BC replay ratio")
        _require(_is_plain_int(pairs) and pairs > 0, "BC report sample count invalid")
        if required_gate == "hypothesis":
            demo_key, bc_key, teacher_key = (
                "teacher_demo_mean", "bc_replay_7000", "teacher_7000")
        else:
            demo_key, bc_key, teacher_key = (
                "teacher_mean_demo", "bc_replay_mean_7000s",
                "teacher_replay_mean_7000s")
        _finite_number(rec[demo_key], f"BC {demo_key}")
        bc_replay = _finite_number(rec[bc_key], f"BC {bc_key}")
        teacher_replay = _finite_number(rec[teacher_key], f"BC {teacher_key}")
        _require(teacher_replay > 0,
                 f"BC {teacher_key} must be positive for the replay ratio to be defined")
        recomputed_ratio = bc_replay / teacher_replay
        _require(math.isclose(ratio, recomputed_ratio,
                              rel_tol=1e-12, abs_tol=1e-12),
                 "BC replay ratio disagrees with the BC/teacher metrics in the report: "
                 f"{ratio} != {recomputed_ratio}")
        _require(ratio >= 0.85, "BC report PASS disagrees with the replay ratio")
        if verify_replay:
            cache_key = (required_gate, actual_sha, expected_impl, generator_sha)
            evidence = _BC_REPLAY_CACHE.get(cache_key)
            cache_miss = evidence is None
            if evidence is None:
                print(f"   BC {required_gate}: replaying fixed demo/replay seeds to re-check the report evidence")
                evidence = _recompute_replay_bc_evidence(
                    required_gate, frozen_policy)
            for key, actual_value in evidence.items():
                reported_value = rec[key]
                if key == "pairs":
                    matches = (_is_plain_int(reported_value)
                               and reported_value == actual_value)
                else:
                    matches = (isinstance(reported_value, (int, float))
                               and not isinstance(reported_value, bool)
                               and math.isclose(float(reported_value),
                                                float(actual_value),
                                                rel_tol=1e-12,
                                                abs_tol=1e-9))
                _require(matches,
                         "BC replay report disagrees with the frozen policy/current runtime recomputation: "
                         f"{key}={reported_value!r} != {actual_value!r}")
            if cache_miss:
                _BC_REPLAY_CACHE[cache_key] = dict(evidence)
    return rec


def _dry_anchor_partition(x):
    """Return complementary (pre-dry, fresh) row masks under worker v4 dual-channel semantics.

    The observation of a worker transition happens before the action executes; the action that actually pushes the 140/1800 count to
    the cap closes the window immediately, so a legal demo pool usually sees only cap-1 and never
    exactly 1.0. The negative domain of col297 also carries the visible drink latch, so its scene clock must be decoded first
    rather than mistaking negatives for fresh. The thresholds are taken directly from the single-step frontier of the environment's two caps, so the probe
    cannot silently get zero coverage because of a terminal observation that can never occur.
    """
    import numpy as np
    from diablogym.options_env import FARM_SCENE_CAP, KILL_PATIENCE

    values = np.asarray(x)
    _require(values.ndim == 2 and values.shape[1] == 298,
             f"dry-anchor grouping input shape invalid: {values.shape}")
    short_threshold = np.float32(
        (KILL_PATIENCE - 1) / KILL_PATIENCE)
    scene_threshold = np.float32(
        (FARM_SCENE_CAP - 1) / FARM_SCENE_CAP)
    encoded_scene = values[:, 297]
    scene_clock = np.where(
        encoded_scene < 0.0, -encoded_scene - 1.0, encoded_scene)
    dry = ((values[:, 296] >= short_threshold)
           | (scene_clock >= scene_threshold))
    return dry, ~dry


def _load_dry_anchor_demos(path: str | pathlib.Path,
                           expected_sha256: str | None) -> tuple[object, object, str]:
    """Hash and parse the same demos.npz bytes, then enforce the BC binding."""
    import numpy as np

    p = pathlib.Path(path)
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             f"BC gate report is missing its demos_sha256 binding: {p.with_name('bc_report.json')}")
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"dry-anchor demo set unreadable: {p}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(actual_sha256 == expected_sha256,
             f"dry-anchor demos SHA mismatch: {actual_sha256} != {expected_sha256}")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            _require(all(key in data for key in ("X", "Y", "episode_id")),
                     "dry-anchor demos.npz is missing X/Y/episode_id")
            x, y = data["X"].copy(), data["Y"].copy()
            episode_id = data["episode_id"].copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"dry-anchor demo set unreadable: {p}: {exc}") from exc
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.ndim == 1 and len(x) == len(y),
             f"dry-anchor array shapes invalid: X={x.shape},Y={y.shape}")
    _require(x.dtype == np.float32 and np.issubdtype(y.dtype, np.integer),
             f"dry-anchor dtype invalid: X={x.dtype},Y={y.dtype}")
    _require(episode_id.ndim == 1 and len(episode_id) == len(x)
             and np.issubdtype(episode_id.dtype, np.integer)
             and len(np.unique(episode_id)) >= 2,
             "dry-anchor episode_id shape/type/independent episode count invalid")
    dry, _ = _dry_anchor_partition(x)
    _require(bool(dry.any()),
             "dry-anchor demo set has no dual-channel cap-1 frontier states "
             "(col296 short clock / col297 signed farm-scene clock)")
    return x, y, actual_sha256


def _export_manifest_path(p: pathlib.Path) -> pathlib.Path:
    return p.with_name(f"{p.name}.manifest.json")


def _validate_export_manifest(p: pathlib.Path) -> dict:
    import torch

    from eval_contract import EvalContractError, strict_json_loads

    manifest_path = _export_manifest_path(p)
    try:
        rec = strict_json_loads(manifest_path.read_bytes())
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"checkpoint export manifest missing/unreadable: {manifest_path}") from exc
    _require(isinstance(rec, dict), f"checkpoint export manifest must be a JSON object: {manifest_path}")
    _require(set(rec) == {
        "schema_version", "artifact_type", "artifact_sha256",
        "source_checkpoint", "source_checkpoint_sha256", "tensor_count"},
        f"checkpoint export manifest fields invalid: {manifest_path}")
    _require(_is_plain_int(rec["schema_version"])
             and rec["schema_version"] == _EXPORT_MANIFEST_SCHEMA_VERSION,
             f"checkpoint export manifest schema invalid: {rec['schema_version']!r}")
    _require(rec.get("artifact_type") == "checkpoint_policy_state",
             f"export manifest artifact_type invalid: {rec.get('artifact_type')!r}")
    expected = rec.get("artifact_sha256")
    try:
        artifact_payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint export unreadable: {p}: {exc}") from exc
    actual = hashlib.sha256(artifact_payload).hexdigest()
    _require(_is_sha256(expected) and expected == actual,
             f"checkpoint export SHA does not match the manifest: {actual} != {expected!r}")
    source_sha = rec.get("source_checkpoint_sha256")
    _require(_is_sha256(source_sha),
             "checkpoint export manifest is missing source_checkpoint_sha256")
    source_checkpoint = rec.get("source_checkpoint")
    _require(isinstance(source_checkpoint, str) and source_checkpoint
             and pathlib.Path(source_checkpoint).is_absolute(),
             "checkpoint export manifest source_checkpoint must be an absolute path")
    source_path = pathlib.Path(source_checkpoint)
    _require(str(source_path.resolve()) == source_checkpoint,
             "checkpoint export manifest source_checkpoint must be a normalised absolute path")
    _require(source_path.resolve() != p.resolve(),
             "a checkpoint export cannot declare itself as its source checkpoint")
    try:
        source_payload = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"source checkpoint declared by the checkpoint export manifest unreadable: {source_path}") from exc
    actual_source_sha = hashlib.sha256(source_payload).hexdigest()
    _require(actual_source_sha == source_sha,
             "checkpoint export manifest source checkpoint SHA mismatch: "
             f"{actual_source_sha} != {source_sha}")
    _validate_checkpoint_bytes(source_payload, str(source_path))

    tensor_count = rec.get("tensor_count")
    _require(_is_plain_int(tensor_count) and tensor_count > 0,
             "checkpoint export manifest tensor_count must be a positive integer")
    try:
        artifact_state = torch.load(
            io.BytesIO(artifact_payload), map_location="cpu", weights_only=True)
        with zipfile.ZipFile(io.BytesIO(source_payload)) as source_archive:
            source_state = torch.load(
                io.BytesIO(source_archive.read("policy.pth")),
                map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint export or source policy state_dict unparseable") from exc
    _require(isinstance(artifact_state, dict) and isinstance(source_state, dict),
             "checkpoint export and source policy must both be state_dicts")
    _require(tensor_count == len(artifact_state) == len(source_state),
             "checkpoint export manifest tensor_count disagrees with the export/source policy")
    _require(set(artifact_state) == set(source_state),
             "checkpoint export fields disagree with the source checkpoint policy")
    for key in source_state:
        source_value = source_state[key]
        artifact_value = artifact_state[key]
        _require(isinstance(source_value, torch.Tensor)
                 and isinstance(artifact_value, torch.Tensor),
                 f"checkpoint policy field is not a Tensor: {key}")
        _require(torch.isfinite(artifact_value).all().item(),
                 f"checkpoint export contains NaN/Inf: {key}")
        _require(artifact_value.shape == source_value.shape
                 and artifact_value.dtype == source_value.dtype
                 and torch.equal(artifact_value, source_value),
                 f"checkpoint export tensor disagrees with the source checkpoint policy: {key}")
    return rec


def _load_bc_state_dict(path: str, policy, required_gate: str,
                        source_kind: str = "bc") -> dict:
    """Check the BC gate and the key tensors; forbid a silent start with "0 keys matched but strict=False"."""
    import torch

    p = pathlib.Path(path)
    manifest = None
    if source_kind == "bc":
        expected_sha256 = _validate_bc_report(p, required_gate)["policy_sha256"]
    elif source_kind == "checkpoint":
        manifest = _validate_export_manifest(p)
        expected_sha256 = manifest["artifact_sha256"]
    else:
        raise ValueError(f"unknown init source kind: {source_kind}")

    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"--bc-init weights unreadable: {p}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(actual_sha256 == expected_sha256,
             f"BC/init weights drifted after the gate check: "
             f"{actual_sha256} != {expected_sha256}")
    # Integrity check and torch deserialization consume the same bytes.
    sd = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    _require(isinstance(sd, dict), "--bc-init must be a policy state_dict")
    bad_types = [k for k, value in sd.items() if not isinstance(value, torch.Tensor)]
    _require(not bad_types, f"BC state_dict contains non-Tensor values: {bad_types}")
    nonfinite = [k for k, value in sd.items()
                 if not torch.isfinite(value).all().item()]
    _require(not nonfinite, f"BC state_dict contains NaN/Inf: {nonfinite}")
    target = policy.state_dict()
    if source_kind == "checkpoint":
        missing_all = sorted(set(target) - set(sd))
        unexpected_all = sorted(set(sd) - set(target))
        mismatched_all = sorted(
            key for key in set(target) & set(sd)
            if target[key].shape != sd[key].shape)
        dtype_mismatched = sorted(
            key for key in set(target) & set(sd)
            if target[key].dtype != sd[key].dtype)
        _require(not missing_all and not unexpected_all and not mismatched_all
                 and not dtype_mismatched,
                 "checkpoint full policy state_dict does not exactly match the target: "
                 f"missing={missing_all}, unexpected={unexpected_all}, "
                 f"shape={mismatched_all}, dtype={dtype_mismatched}")
        _require(manifest is not None and manifest["tensor_count"] == len(sd),
                 "checkpoint export manifest tensor_count disagrees with the state_dict")
    missing = [k for k in _POLICY_HEAD_KEYS if k not in sd]
    mismatched = [k for k in _POLICY_HEAD_KEYS if k in sd and
                  (k not in target or target[k].shape != sd[k].shape)]
    _require(not missing, f"BC state_dict is missing policy-head keys: {missing}")
    _require(not mismatched, f"BC state_dict policy-head shape mismatch: {mismatched}")
    return sd


def make_env(max_steps: int = 1500, deep: bool = False, death_ladder: bool = False,
             options: bool = False, flat_clock: bool = False,
             worker: bool = False, manager_npz: str | None = None,
             worker_npz: str | None = None, skip_dry: float | bool = False,
             drink_sovereignty: bool | None = None,
             reward_economy: str = "v1",
             manager_heuristic: str | None = None,
             legacy_policy_observation_view: bool = False,
             worker_policy_observation_view: str | None = None,
             manager_policy_observation_view: str = "raw-v4",
             worker_fast_forward_reward_credit: str = "none",
             worker_additional_terminal_death_cost: float = 0.0,
             worker_learning_window_scope: str = "farm-only",
             worker_depth_shaping_unit: float = 0.0,
             worker_descend_bonus_fraction: float = 0.0,
             worker_descend_escrow_fraction: float = 0.0,
             worker_descend_escrow_power: float = 1.0,
             worker_descend_escrow_readiness_gate: bool = False,
             worker_descend_escrow_readiness_table: str = "v1",
             worker_hp_loss_price: float = 0.0,
             worker_potion_pickup_bonus: float = 0.0,
             worker_no_progress_timeout_credit: str = (
                 _WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT),
             explore_global_fallback: bool = False,
             explore_global_hunt: bool = False,
             progress_far_tiles: int = 0,
             farm_scene_cap: int = _FARM_SCENE_CAP_DEFAULT,
             reset_layer_clock_on_window: bool = False,
             resource_protocol: str = "off",
             resource_purchase_mode: str = "full",
             resource_service_policy: str = "legacy-v1",
             resource_readiness_law: str = "veto-v1",
             resource_retreat: str = "off",
             resource_portal: str = "off",
             resource_sweep: str = "off",
             resource_identify: str = "off",
             resource_weapon_upgrade: str = "off",
             hunt_scope: str = "all",
             dive_blocker_recovery: str = "off",
             manager_npz_sha256: str | None = None,
             worker_npz_sha256: str | None = None,
             worker_zip: str | None = None,
             worker_zip_sha256: str | None = None,
             deep_start_curriculum: dict | None = None,
             deep_start_form: str = "dive",
             implementation_sha256: str | None = None,
             worker_prefix_model: str | None = None,
             worker_prefix_sha256: str | None = None,
             worker_prefix_max_attempts: int | None = None,
             worker_prefix_max_microsteps: int | None = None,
             worker_time_protocol: str = "legacy"):
    time_identity = _validate_worker_time_config(worker_time_protocol,
        worker=worker, worker_learning_window_scope=worker_learning_window_scope,
        resource_protocol=resource_protocol, resource_purchase_mode=resource_purchase_mode,
        resource_service_policy=resource_service_policy, max_steps=max_steps,
        farm_scene_cap=farm_scene_cap)
    if implementation_sha256 is not None:
        actual_implementation = _implementation_bundle_sha256()
        _require(actual_implementation == implementation_sha256,
                 "training implementation bundle drifted before VecEnv creation: "
                 f"{actual_implementation} != {implementation_sha256}")
    _require(deep_start_curriculum is None or options,
             "deep_start_curriculum only applies to --options manager training")
    _require(deep_start_form in ("dive", "exhausted"),
             f"deep_start_form must be dive/exhausted: {deep_start_form!r}")
    validate_resource_service_config(resource_protocol, resource_purchase_mode,
                                     resource_service_policy)
    _require(resource_readiness_law in ("veto-v1", "coach-v03"),
             "resource_readiness_law must be veto-v1/coach-v03")
    _require(resource_readiness_law == "veto-v1" or resource_protocol == "l2-town-v1",
             "resource_readiness_law coach-v03 requires resource_protocol l2-town-v1")
    # R18-B: retreat-v1 only holds in the l2-town-v1/coach-v03 classroom (same on the engine side).
    _require(resource_retreat in ("off", "retreat-v1"),
             "resource_retreat must be off/retreat-v1")
    _require(resource_retreat == "off" or (resource_protocol == "l2-town-v1"
                                           and resource_readiness_law == "coach-v03"),
             "resource_retreat retreat-v1 requires l2-town-v1 under coach-v03")
    _require(resource_retreat == "off" or worker or options,
             "resource_retreat requires WorkerWindowEnv/OptionsEnv")
    # R18-B5: portal-v1 is the retreat vehicle (same as validate_portal_protocol on the engine side).
    _require(resource_portal in ("off", "portal-v1"),
             "resource_portal must be off/portal-v1")
    _require(resource_portal == "off" or (resource_protocol == "l2-town-v1"
                                          and resource_readiness_law == "coach-v03"),
             "resource_portal portal-v1 requires l2-town-v1 under coach-v03")
    _require(resource_portal == "off" or resource_retreat == "retreat-v1",
             "resource_portal portal-v1 requires resource_retreat retreat-v1")
    _require(resource_portal == "off" or worker or options,
             "resource_portal requires WorkerWindowEnv/OptionsEnv")
    # R18-B6: the sweep/identify/weapon-upgrade laws copy the deployment-side validator wording verbatim.
    _require(resource_sweep in ("off", "sweep-v1"),
             "resource_sweep must be off/sweep-v1")
    _require(resource_sweep == "off" or resource_protocol == "l2-town-v1",
             "resource_sweep sweep-v1 requires l2-town-v1")
    _require(resource_sweep == "off" or resource_service_policy == "sustain-loot-v1",
             "resource_sweep sweep-v1 requires sustain-loot-v1")
    _require(resource_sweep == "off" or worker or options,
             "resource_sweep requires WorkerWindowEnv/OptionsEnv")
    _require(resource_identify in ("off", "cain-v1"),
             "resource_identify must be off/cain-v1")
    _require(resource_identify == "off" or resource_protocol == "l2-town-v1",
             "resource_identify cain-v1 requires l2-town-v1")
    _require(resource_identify == "off" or resource_service_policy == "sustain-loot-v1",
             "resource_identify cain-v1 requires sustain-loot-v1")
    _require(resource_identify == "off" or worker or options,
             "resource_identify requires WorkerWindowEnv/OptionsEnv")
    # dry-v1 is the observation twin; the training side does not accept it (same reason as _validate_args).
    _require(resource_weapon_upgrade in ("off", "smith-v1"),
             "resource_weapon_upgrade must be off/smith-v1")
    _require(resource_weapon_upgrade == "off" or resource_protocol == "l2-town-v1",
             "resource_weapon_upgrade smith-v1 requires l2-town-v1")
    _require(resource_weapon_upgrade == "off"
             or resource_service_policy == "sustain-loot-v1",
             "resource_weapon_upgrade smith-v1 requires sustain-loot-v1")
    _require(resource_weapon_upgrade == "off" or resource_purchase_mode == "full",
             "resource_weapon_upgrade smith-v1 requires the full purchase mode")
    _require(resource_weapon_upgrade == "off" or worker or options,
             "resource_weapon_upgrade requires WorkerWindowEnv/OptionsEnv")
    # R18-B5: hunt_scope only passes through the **env_kwargs of WorkerWindowEnv/OptionsEnv;
    # review fix: it is a no-op when hunting is off, so refuse to mint a lying identity (same as _validate_args).
    _require(hunt_scope in ("all", "l1-only"),
             "hunt_scope must be all/l1-only")
    _require(hunt_scope == "all" or worker or options,
             "hunt_scope l1-only requires WorkerWindowEnv/OptionsEnv")
    _require(hunt_scope == "all" or explore_global_hunt,
             "hunt_scope l1-only requires explore_global_hunt")
    validate_dive_blocker_recovery(resource_protocol, dive_blocker_recovery)
    _require(resource_protocol in ("off", "l2-town-v1"),
             "resource_protocol must be off/l2-town-v1")
    _require(resource_purchase_mode in ("none", "heal", "potions", "armor", "full"),
             "invalid resource_purchase_mode")
    _require(resource_protocol != "off" or resource_purchase_mode == "full",
             "resource_purchase_mode requires resource_protocol")
    _require(resource_protocol == "off" or worker or options,
             "resource_protocol requires WorkerWindowEnv/OptionsEnv")
    resource_kwargs = ({"resource_protocol": resource_protocol,
                        "resource_purchase_mode": resource_purchase_mode}
                       if resource_protocol != "off" else {})
    if resource_service_policy != "legacy-v1":
        resource_kwargs["resource_service_policy"] = resource_service_policy
    # R17.1 ruling 3: the default law adds no keyword, so old calls stay identical.
    if resource_readiness_law != "veto-v1":
        resource_kwargs["resource_readiness_law"] = resource_readiness_law
    # R18-B: off adds no keyword, so old calls stay identical.
    if resource_retreat != "off":
        resource_kwargs["resource_retreat"] = resource_retreat
    # R18-B5: off/all add no keyword, so old calls stay identical.
    if resource_portal != "off":
        resource_kwargs["resource_portal"] = resource_portal
    # R18-B6: off adds no keyword, so old calls stay identical.
    if resource_sweep != "off":
        resource_kwargs["resource_sweep"] = resource_sweep
    if resource_identify != "off":
        resource_kwargs["resource_identify"] = resource_identify
    if resource_weapon_upgrade != "off":
        resource_kwargs["resource_weapon_upgrade"] = resource_weapon_upgrade
    if hunt_scope != "all":
        resource_kwargs["hunt_scope"] = hunt_scope
    if dive_blocker_recovery != "off":
        resource_kwargs["dive_blocker_recovery"] = dive_blocker_recovery
    if time_identity is not None:
        resource_kwargs["worker_time_protocol"] = time_identity["protocol"]
    prefix_spec = worker_prefix_recipe(worker_learning_window_scope, worker_prefix_sha256,
                                       worker_prefix_max_attempts, worker_prefix_max_microsteps)
    _require((prefix_spec is not None) == (worker_prefix_model is not None),
             "worker prefix model and earned scope must be supplied together")
    _require(prefix_spec is None or worker, "earned prefix requires WorkerWindowEnv")
    from diablogym import DiabloGymEnv

    def with_seed_discipline(env):
        """Per-episode seed wrapper shared by flat/policy-brain modes; the worker has its own cross-window episode state and is not wrapped."""
        import gymnasium as gym
        import numpy as np
        from diablogym.worker_env import (
            is_reserved_train_seed,
            sample_train_seed,
        )

        # R9 curriculum arm: the prologue is present only in options manager training; the wrapping order
        # Monitor(_SeedDiscipline(OptionsEnv)) guarantees the play-through steps fall naturally outside Monitor
        # bookkeeping, and the prologue reward never enters the manager's return ledger (PREREG-R9 §4).
        prologue_spec = (
            dict(deep_start_curriculum)
            if (options and deep_start_curriculum) else None)

        class _SeedDiscipline(gym.Wrapper):
            def __init__(self, wrapped):
                super().__init__(wrapped)
                self._seed_rng = np.random.default_rng()

            def reset(self, *, seed=None, options=None):
                if seed is not None:
                    if is_reserved_train_seed(seed):
                        raise ValueError(f"training reset refuses reserved seed {seed}")
                    self._seed_rng = np.random.default_rng(seed)
                else:
                    seed = sample_train_seed(self._seed_rng)
                obs, info = self.env.reset(seed=seed, options=options)
                if prologue_spec is not None:
                    # R9 curriculum arm: the prologue core is module-level (covered by stub-env unit tests);
                    # redrawn seeds go through the existing seed-discipline sampler, which refuses every registered evaluation pool.
                    obs, info, seed, telemetry = _deep_start_prologue(
                        self.env, obs, info, seed,
                        spec=prologue_spec, form=deep_start_form,
                        resample_seed=lambda: sample_train_seed(self._seed_rng),
                        reset=lambda new_seed: self.env.reset(
                            seed=new_seed, options=options),
                    )
                    info = dict(info)
                    info.update(telemetry)
                info["episode_seed"] = seed
                return obs, info

            def action_masks(self):
                return self.env.action_masks()

        return Monitor(_SeedDiscipline(env))

    if worker:
        # v4 window line: episode = one whole underlying game; natural FARM window boundaries are
        # nonterminal and are reported explicitly through info["farm_window_end"].
        # (rng_seed=None -> each subprocess has its own entropy source; the seed sampler refuses every registered evaluation pool)
        # v26: with skip_dry=True, dry-level revisit windows are played by the script and stay out of the learning distribution (the "oasis" remedy)
        from diablogym import WorkerWindowEnv
        if prefix_spec is not None:
            from prefix_worker import load_prefix_worker
            resource_kwargs.update(
                prefix_worker=load_prefix_worker(worker_prefix_model, expected_sha256=worker_prefix_sha256),
                prefix_worker_sha256=worker_prefix_sha256,
                prefix_max_attempts=worker_prefix_max_attempts,
                prefix_max_microsteps=worker_prefix_max_microsteps)
        worker_drink_sovereignty = (
            True if drink_sovereignty is None else drink_sovereignty)
        return Monitor(WorkerWindowEnv(manager_npz=manager_npz, max_steps=max_steps,
                                       **resource_kwargs,
                                       skip_dry=skip_dry,
                                       drink_sovereignty=worker_drink_sovereignty,
                                       legacy_policy_observation_view=(
                                           legacy_policy_observation_view),
                                       policy_observation_view=(
                                           worker_policy_observation_view),
                                       fast_forward_reward_credit=(
                                           worker_fast_forward_reward_credit),
                                       additional_terminal_death_cost=(
                                           worker_additional_terminal_death_cost),
                                       learning_window_scope=(
                                           worker_learning_window_scope),
                                       depth_shaping_unit=(
                                           worker_depth_shaping_unit),
                                       worker_descend_bonus_fraction=(
                                           worker_descend_bonus_fraction),
                                       worker_descend_escrow_fraction=(
                                           worker_descend_escrow_fraction),
                                       worker_descend_escrow_power=(
                                           worker_descend_escrow_power),
                                       worker_descend_escrow_readiness_gate=(
                                           worker_descend_escrow_readiness_gate),
                                       worker_descend_escrow_readiness_table=(
                                           worker_descend_escrow_readiness_table),
                                       # R16 amendment: explicit wage-side parameters (consumed by name by
                                       # WorkerWindowEnv, not in env_kwargs)
                                       worker_hp_loss_price=worker_hp_loss_price,
                                       worker_potion_pickup_bonus=(
                                           worker_potion_pickup_bonus),
                                       worker_no_progress_timeout_credit=(
                                           worker_no_progress_timeout_credit),
                                       seed_scope="train",
                                       manager_sha256=manager_npz_sha256,
                                       manager_heuristic=manager_heuristic,
                                       # R12: the economy rule passes through env_kwargs to
                                       # OptionsEnv->DiabloGymEnv (v1 default unchanged)
                                       reward_economy=reward_economy,
                                       # R16 amendment: classroom-side parameters are consumed by name by
                                       # OptionsEnv through env_kwargs; environment-side parameters pass on to
                                       # DiabloGymEnv, consumed by name (default = old rule)
                                       farm_scene_cap=farm_scene_cap,
                                       reset_layer_clock_on_window=(
                                           reset_layer_clock_on_window),
                                       explore_global_fallback=(
                                           explore_global_fallback),
                                       explore_global_hunt=explore_global_hunt,
                                       progress_far_tiles=progress_far_tiles))
    if options:
        # v22: policy brain/operator brain -- OptionsEnv brings its own deep+death_ladder defaults
        # v25: when worker_npz is non-empty, mount the npz worker (NumpyManager is constructed inside this function --
        # spawned subprocesses need no torch, PREREG-v25 D1 clause), and apply the thin seed-discipline wrapper
        # R9: when worker_zip is non-empty, mount the certified SB3 zip worker (the dual-v4 asymmetric policy
        # has no npz export; the frozen MaskablePPO forward is loaded per subprocess, with a byte-identity check before loading;
        # wiring and dual labels follow the case file train/runs/efix-g0-evidence/probe_r9_dive.py:31-61)
        from diablogym import NumpyManager, OptionsEnv

        if worker_zip:
            _require(not worker_npz, "--worker-zip and --worker-npz are mutually exclusive")
            import numpy as np
            import leashed_ppo  # noqa: F401  registers the custom policy class (needed for load deserialisation)
            from sb3_contrib import MaskablePPO

            zip_payload = pathlib.Path(worker_zip).read_bytes()
            if worker_zip_sha256 is not None:
                actual_zip_sha = hashlib.sha256(zip_payload).hexdigest()
                _require(actual_zip_sha == worker_zip_sha256,
                         "worker zip drifted before VecEnv creation: "
                         f"{actual_zip_sha} != {worker_zip_sha256}")
            model = MaskablePPO.load(io.BytesIO(zip_payload), device="cpu")
            zip_contract = getattr(model, "diablogym_contract", None)
            _require(isinstance(zip_contract, dict),
                     "--worker-zip must carry a diablogym_contract (training contract of a certified release artifact)")
            _require(zip_contract.get("contract_revision") == 26,
                     "--worker-zip training contract contract_revision must be 26, "
                     f"got {zip_contract.get('contract_revision')!r}")
            zip_view = zip_contract["worker_policy_observation_view"]
            zip_sovereignty = bool(zip_contract["drink_sovereignty"])

            def zip_worker(policy_obs, mask):
                policy_mask = mask
                if not zip_sovereignty:
                    # a worker without potion autonomy defensively always masks a12 (same as probe_r9_dive).
                    policy_mask = np.asarray(mask, dtype=bool).copy()
                    policy_mask[12] = False
                action, _ = model.predict(
                    policy_obs, action_masks=policy_mask, deterministic=True)
                return int(action)

            zip_worker.diablogym_worker_observation_view = zip_view
            zip_worker.diablogym_worker_action12_mode = (
                "environment-mask" if zip_sovereignty
                else "permanently-masked")
            env = OptionsEnv(max_steps=max_steps,
                             **resource_kwargs,
                             # None -> self-bound from the worker's dual labels (single source of truth);
                             # OptionsEnv rejects conflicts between labels or between a label and an explicit value.
                             drink_sovereignty=None,
                             manager_observation_view=(
                                 manager_policy_observation_view),
                             worker_observation_view=zip_view,
                             workers={0: zip_worker},
                             reward_economy=reward_economy,
                             farm_scene_cap=farm_scene_cap,
                             reset_layer_clock_on_window=(
                                 reset_layer_clock_on_window),
                             explore_global_fallback=explore_global_fallback,
                             explore_global_hunt=explore_global_hunt,
                             progress_far_tiles=progress_far_tiles)
            _require(bool(env.drink_sovereignty) == zip_sovereignty,
                     "OptionsEnv drink_sovereignty not self-bound per the zip worker contract")
        elif worker_npz:
            # Clause summary: the worker enters subprocesses as npz + numpy forward (no pickled network, no SB3
            # model load, no per-step torch forward). The torch module itself still enters the subprocess through
            # the top-level train_ppo import (same as the v23 precedent); a "no torch" assertion cannot be implemented, and the pre-registration was corrected accordingly.
            net = NumpyManager(worker_npz, expected_sha256=worker_npz_sha256)
            net.require_io_shape(298, 15, "Options worker")
            net.require_worker_contract()
            env = OptionsEnv(max_steps=max_steps,
                             **resource_kwargs,
                             drink_sovereignty=drink_sovereignty,
                             manager_observation_view=(
                                 manager_policy_observation_view),
                             worker_observation_view=(
                                 net.worker_observation_view),
                             workers={0: net.worker_callback()},
                             reward_economy=reward_economy,
                             farm_scene_cap=farm_scene_cap,
                             reset_layer_clock_on_window=(
                                 reset_layer_clock_on_window),
                             explore_global_fallback=explore_global_fallback,
                             explore_global_hunt=explore_global_hunt,
                             progress_far_tiles=progress_far_tiles)
        else:
            env = OptionsEnv(max_steps=max_steps,
                             **resource_kwargs,
                             drink_sovereignty=drink_sovereignty,
                             manager_observation_view=(
                                 manager_policy_observation_view),
                             reward_economy=reward_economy,
                             farm_scene_cap=farm_scene_cap,
                             reset_layer_clock_on_window=(
                                 reset_layer_clock_on_window),
                             explore_global_fallback=explore_global_fallback,
                             explore_global_hunt=explore_global_hunt,
                             progress_far_tiles=progress_far_tiles)

        # With or without an npz worker, manager training must follow the same seed discipline.
        return with_seed_discipline(env)
    if flat_clock:
        # v22 devil arm F: 296-dim flat (stall clock is the same gauge as the policy brain's)
        from diablogym import StagnationClockWrapper
        return with_seed_discipline(StagnationClockWrapper(DiabloGymEnv(
            ticks_per_step=4, max_steps=max_steps, start_in_dungeon=True,
            include_raw=False, descend_ladder=True, death_ladder=True,
            explore_global_fallback=explore_global_fallback,
            explore_global_hunt=explore_global_hunt,
            progress_far_tiles=progress_far_tiles)))
    env = DiabloGymEnv(
        ticks_per_step=4,      # each decision = 0.2 s of game time
        max_steps=max_steps,   # 1500 = champion (v6) recipe; 3000 = v10 long-episode experiment + v17 deep water.
                               # the 32-seed leaderboard evaluation is fixed at 1500 steps (comparability); the deep-water chapter has its own table
        start_in_dungeon=True, # skip town, start at the dungeon level 1 entrance
        include_raw=False,     # training does not pass the big raw dict (less multi-process IPC)
        descend_ladder=deep,   # v17: descend bonus grows with depth (8×N), giving "going down alive" a future
        death_ladder=death_ladder,  # v18: dying on level N costs 8×N -- "arriving alive" must beat "touching depth"
        # R16 amendment: environment-side parameters pass through (default False/False/0 = old rule, bit for bit)
        explore_global_fallback=explore_global_fallback,
        explore_global_hunt=explore_global_hunt,
        progress_far_tiles=progress_far_tiles,
    )
    return with_seed_discipline(env)


def _is_publishable_rollout_boundary(
        model, *, rollout_buffer_may_be_reset: bool = False) -> bool:
    """Require collection plus an explicit optimizer-consumption receipt.

    ``MaskablePPO.collect_rollouts()`` resets its buffer immediately before
    ``callback.on_rollout_start()``.  Periodic checkpointing runs at that hook
    because the preceding ``train()`` has just returned, so callers may
    explicitly rely on the persisted Leashed receipt after that reset.  Final
    publication keeps the stricter default and still requires ``buffer.full``.
    """
    rollout_full = bool(getattr(getattr(model, "rollout_buffer", None),
                                "full", False))
    if ((not rollout_full and not rollout_buffer_may_be_reset)
            or bool(getattr(model, "_calib_tripped", False))):
        return False
    if not hasattr(model, "_last_completed_ppo_rollout_steps"):
        # Non-Leashed legacy algorithms do not yet emit the stronger receipt.
        return True
    completed_at = getattr(
        model, "_last_completed_ppo_rollout_steps", None)
    optimizer_steps = getattr(
        model, "_ppo_optimizer_steps_completed", None)
    return (
        _is_plain_int(completed_at)
        and completed_at == int(getattr(model, "num_timesteps", -1))
        and _is_plain_int(optimizer_steps)
        and optimizer_steps > 0
    )


def _asymmetric_worker_deployment_evidence_complete(model) -> bool:
    """Close the structured actor/critic deployment evidence without key lore."""
    from leashed_ppo import (
        ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA,
        AsymmetricWorkerMaskableActorCriticPolicy,
        asymmetric_worker_runtime_evidence,
    )
    from diablogym.controller_wire import (
        DUAL_WORKER_LAYOUT,
        DUAL_WORKER_LAYOUT_SHA256,
    )

    if not isinstance(
            getattr(model, "policy", None),
            AsymmetricWorkerMaskableActorCriticPolicy):
        return True
    try:
        evidence = asymmetric_worker_runtime_evidence(model.policy)
        layout = evidence["layout"]
        policy = evidence["policy"]
        context = evidence["context"]
        groups = context["parameter_groups"]
        probes = evidence["probes"]
        focused_effects = probes["actor_focused_effects"]
        actor_receipt = getattr(
            model, "_actor_migration_receipt", None)
        critic_receipt = getattr(
            model, "_critic_migration_receipt", None)
        anneal_rollouts = getattr(
            model, "distill_anneal_actor_rollouts", None)
        anneal_completed = getattr(
            model, "_distill_actor_rollouts_completed", None)
        last_effective_beta = getattr(
            model, "_last_effective_distill_beta", None)
        distillation_complete = (
            (
                float(getattr(model, "distill_beta", 0.0)) == 0.0
                and anneal_rollouts == 0
            )
            or (
                _is_plain_int(anneal_rollouts)
                and anneal_rollouts >= 2
                and _is_plain_int(anneal_completed)
                and anneal_completed >= anneal_rollouts
                and isinstance(last_effective_beta, (int, float))
                and not isinstance(last_effective_beta, bool)
                and math.isfinite(float(last_effective_beta))
                and float(last_effective_beta) == 0.0
            )
        )
        action_probe_keys = (
            (
                "forced_context_action_effect",
                "forced_context_action_logit_changed_elements",
                "forced_context_action_logit_max_abs_delta",
            ),
            (
                "nonzero_context_action_effect",
                "nonzero_context_action_logit_changed_elements",
                "nonzero_context_action_logit_max_abs_delta",
            ),
        )
        action_effects_closed = all(
            probes.get(enabled_key) is True
            and _is_plain_int(probes.get(count_key))
            and probes[count_key] > 0
            and isinstance(probes.get(delta_key), (int, float))
            and not isinstance(probes[delta_key], bool)
            and math.isfinite(float(probes[delta_key]))
            and float(probes[delta_key]) > 0.0
            for enabled_key, count_key, delta_key in action_probe_keys
        )
        focused_effects_closed = (
            isinstance(focused_effects, dict)
            and set(focused_effects) == {
                "current_v4_base",
                "wrapper_scalars",
                "controller_combat",
            }
            and all(
                isinstance(record, dict)
                and record.get("registered_actor_input") is True
                and _is_plain_int(record.get("feature_index"))
                and record.get("preoutput_effect") is True
                and _is_plain_int(
                    record.get("preoutput_changed_elements"))
                and record["preoutput_changed_elements"] > 0
                and isinstance(
                    record.get("preoutput_max_abs_delta"),
                    (int, float),
                )
                and not isinstance(
                    record["preoutput_max_abs_delta"], bool)
                and math.isfinite(float(
                    record["preoutput_max_abs_delta"]))
                and float(record["preoutput_max_abs_delta"]) > 0.0
                and record.get("context_action_effect") is True
                and _is_plain_int(record.get(
                    "context_action_logit_changed_elements"))
                and record[
                    "context_action_logit_changed_elements"] > 0
                and isinstance(record.get(
                    "context_action_logit_max_abs_delta"), (int, float))
                and not isinstance(
                    record["context_action_logit_max_abs_delta"], bool)
                and math.isfinite(float(
                    record["context_action_logit_max_abs_delta"]))
                and float(
                    record["context_action_logit_max_abs_delta"]) > 0.0
                for record in focused_effects.values()
            )
        )
        group_receipt_keys = {
            "encoder": "context_encoder_sha256",
            "interaction": "context_interaction_sha256",
            "output": "context_output_sha256",
        }
        groups_closed = (
            isinstance(groups, dict)
            and set(groups) == set(group_receipt_keys)
            and all(
                isinstance(groups[name], dict)
                and _is_plain_int(groups[name].get("tensor_count"))
                and groups[name]["tensor_count"] > 0
                and _is_plain_int(groups[name].get("parameter_count"))
                and groups[name]["parameter_count"] > 0
                and _is_plain_int(groups[name].get("nonzero_count"))
                and groups[name]["nonzero_count"] > 0
                and _is_sha256(groups[name].get("sha256"))
                for name in group_receipt_keys
            )
        )
        actor_receipt_closed = (
            isinstance(actor_receipt, dict)
            and actor_receipt.get("schema")
            == _ASYMMETRIC_ACTOR_INIT_SCHEMA
            and actor_receipt.get("method")
            == _ASYMMETRIC_ACTOR_INIT_METHOD
            and actor_receipt.get("context_architecture")
            == _ASYMMETRIC_CONTEXT_ARCHITECTURE
            and actor_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and actor_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and actor_receipt.get("target_actor_parameter_tensors")
            == policy["actor_tensor_count"]
            and actor_receipt.get("target_actor_parameter_count")
            == policy["actor_parameter_count"]
            and actor_receipt.get("context_parameter_tensors")
            == context["tensor_count"]
            and actor_receipt.get("context_parameter_count")
            == context["parameter_count"]
            and _is_sha256(
                actor_receipt.get("migrated_actor_sha256"))
            and policy["actor_sha256"]
            != actor_receipt["migrated_actor_sha256"]
            and groups_closed
            and all(
                _is_sha256(actor_receipt.get(receipt_key))
                and groups[name]["sha256"]
                != actor_receipt[receipt_key]
                for name, receipt_key in group_receipt_keys.items()
            )
        )
        critic_receipt_closed = (
            isinstance(critic_receipt, dict)
            and critic_receipt.get("schema")
            == _ASYMMETRIC_CRITIC_RESET_SCHEMA
            and critic_receipt.get("method")
            == _ASYMMETRIC_CRITIC_RESET_METHOD
            and critic_receipt.get("critic_architecture")
            == _ASYMMETRIC_CRITIC_ARCHITECTURE
            and critic_receipt.get("controller_layout_schema")
            == DUAL_WORKER_LAYOUT.schema
            and critic_receipt.get("controller_layout_sha256")
            == DUAL_WORKER_LAYOUT_SHA256
            and critic_receipt.get("actor_parameter_tensors")
            == policy["actor_tensor_count"]
            and critic_receipt.get("critic_parameter_tensors")
            == policy["critic_tensor_count"]
            and critic_receipt.get("critic_parameter_count")
            == policy["critic_parameter_count"]
            and _is_sha256(
                critic_receipt.get("critic_sha256_after"))
            and policy["critic_sha256"]
            != critic_receipt["critic_sha256_after"]
            and critic_receipt.get("source_checkpoint_sha256")
            == actor_receipt.get("source_checkpoint_sha256")
            and critic_receipt.get("source_actor_sha256")
            == actor_receipt.get("source_actor_sha256")
        )
        return bool(
            evidence.get("schema")
            == ASYMMETRIC_WORKER_RUNTIME_EVIDENCE_SCHEMA
            and layout.get("schema") == DUAL_WORKER_LAYOUT.schema
            and layout.get("sha256") == DUAL_WORKER_LAYOUT_SHA256
            and layout.get("observation_dim")
            == DUAL_WORKER_LAYOUT.observation_dim
            and context.get("enabled") is True
            and _is_sha256(policy.get("actor_sha256"))
            and _is_sha256(policy.get("critic_sha256"))
            and probes.get("p_skip_preoutput_invariant") is True
            and probes.get("p_skip_preoutput_max_abs_delta") == 0.0
            and probes.get("p_skip_preoutput_zero_sha256")
            == probes.get("p_skip_preoutput_one_sha256")
            and probes.get("p_skip_action_logits_invariant") is True
            and probes.get("p_skip_action_logits_max_abs_delta") == 0.0
            and probes.get("p_skip_action_logits_zero_sha256")
            == probes.get("p_skip_action_logits_one_sha256")
            and action_effects_closed
            and focused_effects_closed
            and actor_receipt_closed
            and critic_receipt_closed
            and distillation_complete
        )
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False


def _training_completion_report(model, target_global_steps: int) -> dict:
    """Evaluate the existing publication predicates once and name each result.

    When a callback returns ``False``, SB3's ``learn`` returns normally; checking only a full
    buffer would mistake any earlier rollout boundary for a complete leg. Strict equality also blocks
    a silent overshoot.
    """
    migration_start = getattr(
        model, "_critic_warmup_start_timesteps", None)
    inherited_resource = getattr(model, "_resource_warm_start_receipt", None)
    inherited_resource_valid = False
    if inherited_resource is not None:
        try:
            model._assert_critic_migration_contract()
            inherited_resource_valid = True
        except (AttributeError, RuntimeError, TypeError, ValueError):
            inherited_resource_valid = False
    worker_pg_complete = True
    if migration_start is not None or inherited_resource is not None:
        try:
            from leashed_ppo import worker_onpolicy_pg_audit_complete
            worker_pg_complete = worker_onpolicy_pg_audit_complete(model)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            worker_pg_complete = False
    migration_complete = (
        migration_start is None
        or (
            bool(getattr(model, "_critic_warmup_completed", False))
            and int(getattr(
                model, "_critic_warmup_rollouts_completed", -1))
            == int(getattr(
                model, "_critic_warmup_expected_rollouts", -2))
            and int(getattr(
                model, "_critic_warmup_optimizer_steps_completed", 0)) > 0
            and int(getattr(
                model, "_actor_optimizer_steps_completed", 0)) > 0
            and int(getattr(model, "num_timesteps", -1))
            > int(getattr(
                model, "_critic_warmup_until_timesteps", 2**63 - 1))
        )
    )
    if inherited_resource is not None:
        migration_complete = (inherited_resource_valid
            and int(getattr(model, "_actor_optimizer_steps_completed", 0)) > 0)
    asymmetric_complete = (
        _asymmetric_worker_deployment_evidence_complete(model))
    checks = {
        "rollout_boundary": _is_publishable_rollout_boundary(model),
        "exact_target": (int(getattr(model, "num_timesteps", -1))
                         == int(target_global_steps)),
        "migration": bool(migration_complete),
        "worker_pg": bool(worker_pg_complete),
        "asymmetric": bool(asymmetric_complete),
    }
    counters = {
        "num_timesteps": int(getattr(model, "num_timesteps", -1)),
        "target_global_steps": int(target_global_steps),
        "last_completed_ppo_rollout_steps": getattr(
            model, "_last_completed_ppo_rollout_steps", None),
        "ppo_optimizer_steps_completed": getattr(
            model, "_ppo_optimizer_steps_completed", None),
        "actor_optimizer_steps_completed": getattr(
            model, "_actor_optimizer_steps_completed", None),
        "worker_onpolicy_pg_joint_rollouts": getattr(
            model, "_worker_onpolicy_pg_joint_rollouts", None),
        "worker_onpolicy_pg_qualifying_rollouts": getattr(
            model, "_worker_onpolicy_pg_qualifying_rollouts", None),
    }
    return {
        "schema": "diablogym-training-completion/1",
        "publication_eligible": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "counters": counters,
    }


def _is_exact_training_completion(model, target_global_steps: int) -> bool:
    return _training_completion_report(model, target_global_steps)[
        "publication_eligible"]


class _TrainingCompletionRejected(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        checks = report["checks"]
        prefix = ("training updates completed, but final release acceptance failed"
                  if checks["rollout_boundary"] and checks["exact_target"]
                  else "training did not stop exactly at the frozen target of completed updates")
        super().__init__(f"{prefix}: failed_checks={report['failed_checks']}, "
                         f"counters={report['counters']}")


def _require_exact_training_completion(model, target_global_steps: int) -> dict:
    """Promote SB3's "normal early return" to a process failure, so the scheduler cannot report a false success."""
    report = _training_completion_report(model, target_global_steps)
    if report["publication_eligible"]:
        return report
    raise _TrainingCompletionRejected(report)


def _retain_refused_training_diagnostic(
        model, run_dir: pathlib.Path, report: dict,
        expected_implementation: str) -> dict:
    """Retain an isolated diagnostic only after a normal, consumed end boundary."""
    checks = report["checks"]
    if (not all(checks[name] for name in (
            "rollout_boundary", "exact_target", "migration"))
            or getattr(model, "_resource_warm_start_receipt", None) is None):
        return {"status": "NOT_ELIGIBLE", "reason": "no exact inherited completed boundary"}
    _require(_implementation_bundle_sha256() == expected_implementation,
             "implementation/engine/game content drifted before the diagnostic save")
    from leashed_ppo import validate_worker_onpolicy_pg_receipt
    from training_diagnostics import archive_refused_training
    return archive_refused_training(
        model, run_dir, report, expected_implementation=expected_implementation,
        checkpoint_validator=_validate_checkpoint_bytes,
        receipt_validator=validate_worker_onpolicy_pg_receipt)


class AtomicRolloutCheckpointCallback(BaseCallback):
    """Save only at the boundary where the previous rollout has completed its update, ruling out "steps recorded, gradient not consumed"."""

    def __init__(self, run_dir: pathlib.Path, every_steps: int = 250_000,
                 implementation_sha256: str | None = None):
        super().__init__()
        self.run_dir = run_dir
        self.every_steps = every_steps
        self.implementation_sha256 = implementation_sha256
        self.period = None
        self.next_at = None
        self.last_saved = None

    def _on_training_start(self) -> None:
        quantum = int(self.model.n_steps * self.model.get_env().num_envs)
        self.period = max(quantum, (self.every_steps // quantum) * quantum)
        self.next_at = int(self.num_timesteps + self.period)

    def _save_due(self, *, rollout_buffer_may_be_reset: bool = False) -> None:
        if self.next_at is None or self.num_timesteps < self.next_at:
            return
        # ``_on_rollout_start`` normally follows a completed ``train()``, but a
        # calibration trip or any other zero-update/stale-receipt path can leave
        # num_timesteps one rollout ahead of the weights.  Reuse the same strong
        # predicate as final publication instead of inferring completion merely
        # from callback timing.
        if not _is_publishable_rollout_boundary(
                self.model,
                rollout_buffer_may_be_reset=rollout_buffer_may_be_reset):
            if bool(getattr(self.model, "_calib_tripped", False)):
                print("   [G-CAL] current rollout has not completed its update; refusing to publish a checkpoint")
            else:
                print("   current rollout lacks a complete PPO update receipt; refusing to publish a checkpoint")
            return
        step = int(self.num_timesteps)
        if self.last_saved != step:
            if self.implementation_sha256 is not None:
                actual = _implementation_bundle_sha256()
                _require(actual == self.implementation_sha256,
                         "implementation/engine/game content drifted during training; refusing to publish a checkpoint: "
                         f"{actual} != {self.implementation_sha256}")
            path = self.run_dir / "ckpt" / f"model_{step}_steps.zip"
            _atomic_save_model(self.model, path)
            self.last_saved = step
            print(f"   rollout-boundary checkpoint: {path}")
        while self.next_at <= step:
            self.next_at += self.period

    def _on_rollout_start(self) -> None:
        # The first call has num_timesteps = the start; later calls happen after the previous rollout's train().
        self._save_due(rollout_buffer_may_be_reset=True)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # A normal finish never reaches the next _on_rollout_start; only buffer.full means
        # the last batch really went through train(). A half rollout cut short by a callback must not pose as a checkpoint.
        if _is_publishable_rollout_boundary(self.model):
            self._save_due()


class WorkerSentinelCallback(BaseCallback):
    """v23 sentinel (PREREG appendix A/C): every 500k steps, aggregate subprocess WorkerWindowEnv.stats
    (dry/fresh level window mix, termination reason spectrum, fallback reroll count) + cumulative action shares -> sentinel.jsonl.
    The collapse verdict itself uses the 2M/4M checkpoint assembled replay (appendix C); this only serves telemetry and post-mortems."""

    def __init__(self, run_dir: pathlib.Path, every: int = 500_000):
        super().__init__()
        self.run_dir = run_dir
        self.every = every
        self.next_at = every
        self.action_counts = None
        self._last_emit_step = None

    def _on_training_start(self) -> None:
        # v24 fix: the global step of a resume leg does not start at 0 -- align to the next 500k boundary to prevent an empty burst
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        import numpy as np
        # v24 G-CAL: a calibration probe that sets its flag ends the leg (the driver decides on recalibration, pre-registered clause)
        if getattr(self.model, "_calib_tripped", False):
            print("   [G-CAL] production calibration hard gate triggered -- ending the leg, handing over to the driver")
            return False
        acts = self.locals.get("actions")
        if acts is not None:
            if self.action_counts is None:
                self.action_counts = np.zeros(15, dtype=np.int64)
            for a in np.asarray(acts).ravel():
                self.action_counts[int(a)] += 1
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _emit(self, final: bool) -> None:
        import numpy as np

        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        per_env = self.model.get_env().get_attr("stats")   # passed through Monitor.__getattr__
        count_keys = (
            "windows", "dry", "fresh", "ff_windows", "ff_dry",
            "ff_terminals", "episodes", "reseeds",
            "interrupted_resets", "manual_ff_calls",
            "direct_terminal_deaths",
            "transition_ff_terminal_deaths",
            "reset_ff_terminal_deaths",
            "manual_ff_terminal_deaths",
            "direct_no_progress_timeouts",
            "transition_ff_no_progress_timeouts",
            "reset_ff_no_progress_timeouts",
            "manual_ff_no_progress_timeouts",
        )
        reward_keys = (
            "transition_ff_reward", "reset_ff_reward", "manual_ff_reward",
            "direct_existing_terminal_death_reward",
            "direct_additional_terminal_death_reward",
            "transition_ff_terminal_death_reward",
            "transition_ff_additional_terminal_death_reward",
            "credited_ff_terminal_death_reward",
            "reset_ff_terminal_death_reward",
            "reset_ff_additional_terminal_death_reward",
            "additional_terminal_death_reward",
            "direct_no_progress_timeout_failure_reward",
            "transition_ff_no_progress_timeout_failure_reward",
            "reset_ff_no_progress_timeout_failure_reward",
            "manual_ff_no_progress_timeout_failure_reward",
            "credited_no_progress_timeout_failure_reward",
        )
        env = self.model.get_env()
        credit_modes = env.get_attr("fast_forward_reward_credit")
        configured_costs = env.get_attr("additional_terminal_death_cost")
        _require(
            isinstance(per_env, (list, tuple)) and len(per_env) > 0
            and len(credit_modes) == len(per_env)
            and len(configured_costs) == len(per_env),
            "Worker sentinel sub-environment stats/reward config counts do not close",
        )
        _require(
            all(isinstance(mode, str)
                and mode in {"none", "terminal-death-only"}
                for mode in credit_modes)
            and len(set(credit_modes)) == 1,
            "Worker sentinel sub-environment fast-forward reward credit drift",
        )
        normalized_costs = [
            _finite_number(value, "Worker sentinel additional death cost")
            for value in configured_costs
        ]
        _require(
            all(value >= 0.0 for value in normalized_costs)
            and all(value == normalized_costs[0]
                    for value in normalized_costs),
            "Worker sentinel sub-environment additional death cost drift/invalid",
        )
        agg = {key: 0 for key in count_keys}
        agg.update({key: 0.0 for key in reward_keys})
        reasons = {}
        ff_reasons = {}
        for env_index, stats in enumerate(per_env):
            _require(
                isinstance(stats, dict)
                and all(key in stats for key in count_keys)
                and all(key in stats for key in reward_keys)
                and isinstance(stats.get("reasons"), dict)
                and isinstance(stats.get("ff_reasons"), dict),
                f"Worker sentinel env[{env_index}] stats fields incomplete",
            )
            for key in count_keys:
                value = stats[key]
                _require(
                    _is_plain_int(value) and value >= 0,
                    f"Worker sentinel env[{env_index}].{key} "
                    "must be a non-negative plain integer",
                )
                agg[key] += value
            for key in reward_keys:
                agg[key] += _finite_number(
                    stats[key],
                    f"Worker sentinel env[{env_index}].{key}",
                )
            for key, value in stats["reasons"].items():
                _require(
                    isinstance(key, str) and bool(key)
                    and _is_plain_int(value) and value >= 0,
                    f"Worker sentinel env[{env_index}] reasons invalid",
                )
                reasons[key] = reasons.get(key, 0) + value
            for key, value in stats["ff_reasons"].items():
                _require(
                    isinstance(key, str) and bool(key)
                    and _is_plain_int(value) and value >= 0,
                    f"Worker sentinel env[{env_index}] ff_reasons invalid",
                )
                ff_reasons[key] = ff_reasons.get(key, 0) + value
        for key in reward_keys:
            agg[key] = round(agg[key], 6)
        top1 = int(self.action_counts.argmax()) if self.action_counts is not None else -1
        share = (float(self.action_counts[top1] / max(1, self.action_counts.sum()))
                 if top1 >= 0 else 0.0)
        line = {"sentinel": "v23", "step": step, **agg,
                "dry_share": round(agg["dry"] / max(1, agg["dry"] + agg["fresh"]), 4),
                "reasons": reasons, "ff_reasons": ff_reasons,
                "fast_forward_reward_credit_mode": credit_modes[0],
                "configured_additional_terminal_death_cost":
                    normalized_costs[0],
                "top1_action": top1, "top1_share": round(share, 4),
                # v24 leash readings (reconciled against gate_ledger in the double ledger)
                "beta_initial":
                    getattr(self.model, "distill_beta", None),
                "beta": getattr(
                    self.model,
                    "_last_effective_distill_beta",
                    None),
                "distill_actor_rollouts_completed": getattr(
                    self.model,
                    "_distill_actor_rollouts_completed",
                    None),
                "distill_ce": getattr(self.model, "_last_distill_ce", None),
                "teacher_entropy": getattr(
                    self.model, "_last_teacher_entropy", None),
                "distill_kl": getattr(
                    self.model, "_last_distill_kl", None),
                "distill_tv": getattr(
                    self.model, "_last_distill_tv", None),
                "teacher_diverge": getattr(self.model, "_last_diverge", None)}
        if final:
            line["final"] = True
        with open(self.run_dir / "sentinel.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._last_emit_step = step
        print(f"   [sentinel] {line}")

    def _on_training_end(self) -> None:
        # The leg length is usually 499,712 (<500k); writing only by interval would give a complete leg zero sentinel records.
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class PrefixAuditCallback(BaseCallback):
    """Persist actual per-env prefix ledgers separately from learner transitions."""
    def __init__(self, run_dir):
        super().__init__()
        self.path = pathlib.Path(run_dir) / "worker_prefix_audit.jsonl"

    # R18-B8 (2026-09-08): the per-env prefix ledger carries the WHOLE attempt history with
    # full state dumps per DIVE opening, so writing it at every rollout boundary grows the
    # audit file quadratically (6.2 GB by 254k learner steps; the R18-B arm died of
    # ENOSPC). Rollout boundaries now write a compact receipt (counters + the
    # latest attempt without state dumps); the complete ledger is written once at
    # training_end. Nothing in training reads this file.
    _ROLLOUT_ATTEMPT_KEEP = 2

    @classmethod
    def compact_ledger(cls, per_env):
        compact = []
        for env in per_env:
            if not isinstance(env, dict):
                compact.append(env)
                continue
            row = {k: v for k, v in env.items() if k != "attempts"}
            attempts = env.get("attempts") or []
            row["attempts_count"] = len(attempts)
            kept = []
            for attempt in attempts[-cls._ROLLOUT_ATTEMPT_KEEP:]:
                if not isinstance(attempt, dict):
                    kept.append(attempt)
                    continue
                slim = {k: v for k, v in attempt.items() if k not in ("dive_openings", "windows")}
                openings = attempt.get("dive_openings") or []
                slim["dive_openings_count"] = len(openings)
                slim["dive_openings"] = [{k: v for k, v in o.items() if k != "state"}
                                         for o in openings[-3:] if isinstance(o, dict)]
                kept.append(slim)
            row["attempts"] = kept
            compact.append(row)
        return compact

    def _emit(self, stage):
        per_env = self.training_env.env_method("get_prefix_ledger")
        if stage != "training_end":
            per_env = self.compact_ledger(per_env)
        receipt = {"version": "earned-dive-prefix-audit-v1", "stage": stage,
                   "learner_steps": int(self.num_timesteps),
                   "prefix_training_contract": "excluded",
                   "ledger_form": "complete" if stage == "training_end" else "compact-v1",
                   "per_env": per_env}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")

    def _on_rollout_start(self):
        self._emit("rollout_start")

    def _on_rollout_end(self):
        self._emit("rollout_end")

    def _on_training_end(self):
        self._emit("training_end")

    def _on_step(self):
        return True


class R13DiveAuditCallback(BaseCallback):
    """R13 classroom-reform audit (pure read + IO, record-only, no verdict): per-window metering and whether the classroom happens.

    Mounted only under the live-DIVE worker scope. Reads, per env, the
    dive_live_*/farm_live_steps counters of WorkerWindowEnv.stats (keys specific to the flag-on branch,
    .get tolerates absence) and writes the aggregate to r13_dive_audit.jsonl. A new standalone class that never touches
    the frozen emission surface of WorkerSentinelCallback (R7 schema/get_attr whitelist).
    """

    _KEYS = (
        "dive_live_windows", "dive_live_steps", "farm_live_steps",
        "dive_live_descends", "dive_live_stalls", "dive_live_deaths",
        "dive_live_a11_requests", "dive_live_a11_executed",
        # R17.0 amendment 1: number of descends that passed the readiness gate (counted under both rulers)
        "descend_escrow_ready_vested_count",
    )
    # R13.2: shaping accounts (float, conservation audit: credited+refunded = policy net gain);
    # R14 plan B: descend_bonus_kept mirrors each env's retained amount (absolute value, added directly)
    _FLOAT_KEYS = ("depth_shaping_credited", "depth_shaping_refunded",
                   "descend_bonus_kept",
                   "descend_escrow_vested", "descend_escrow_forfeited",
                   "descend_escrow_unready_denied",
                   # R16 amendment: HP-loss pricing / potion-pickup booking (worker_env stats; default 0.0)
                   "hp_loss_charged", "potion_pickup_credited")

    def __init__(self, run_dir: pathlib.Path, every: int = 63_488):
        super().__init__()
        self.run_dir = run_dir
        self.every = every
        self.next_at = every

    def _emit_line(self, final: bool) -> None:
        per_env = self.training_env.get_attr("stats")
        agg = {key: 0 for key in self._KEYS}
        agg.update({key: 0.0 for key in self._FLOAT_KEYS})
        for stats in per_env:
            for key in self._KEYS:
                agg[key] += int(stats.get(key, 0))
            for key in self._FLOAT_KEYS:
                agg[key] = round(
                    agg[key] + float(stats.get(key, 0.0)), 3)
        # R17.0 gauge: per-pass gate telemetry (worker_env stats descend_escrow_gate_log,
        # empty by default with the flag off) -- each emission rewrites r17_descend_gate.jsonl in full, with the env index on each row.
        gate_rows = []
        for env_index, stats in enumerate(per_env):
            for row in (stats.get("descend_escrow_gate_log") or ()):
                gate_rows.append({"env": int(env_index), **row})
        if gate_rows:
            with open(self.run_dir / "r17_descend_gate.jsonl", "w") as f:
                for row in gate_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        live = agg["dive_live_steps"] + agg["farm_live_steps"]
        line = {
            "r13_dive_audit": "v1", "step": int(self.num_timesteps),
            **agg,
            "descend_escrow_gate_rows": len(gate_rows),
            "dive_share": (
                agg["dive_live_steps"] / live if live else 0.0),
            "descends_per_10_windows": (
                10.0 * agg["dive_live_descends"]
                / agg["dive_live_windows"]
                if agg["dive_live_windows"] else 0.0),
            "final": bool(final),
        }
        with open(self.run_dir / "r13_dive_audit.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"   [R13 classroom] {line}", flush=True)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.next_at = (
                (self.num_timesteps // self.every) + 1) * self.every
            self._emit_line(False)
        return True

    def _on_training_end(self) -> None:
        self._emit_line(True)


class DryAnchorSentinel(BaseCallback):
    """v26 dry-level anchor sentinel (record-only, no verdict): a fixed draw of 2000 dual-channel cap-1 frontier teacher samples
    from demos.npz; every 500k steps it measures the student-argmax mismatch rate against the teacher labels -- under skip_dry, dry-level behaviour
    runs unanchored, and this gauge is its only observer."""

    def __init__(self, run_dir: pathlib.Path, demos_npz: str,
                 expected_sha256: str, every: int = 500_000):
        super().__init__()
        import numpy as np
        self.run_dir = run_dir
        self.every = every
        self.next_at = every
        self._last_emit_step = None
        X, Y, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, expected_sha256)
        m, _ = _dry_anchor_partition(X)
        idx = np.random.default_rng(26).choice(
            np.flatnonzero(m), size=min(2000, int(m.sum())), replace=False)
        self.X, self.Y = X[idx], Y[idx]
        if (not np.isfinite(self.X).all() or np.any(self.Y < 0)
                or np.any(self.Y >= 15)):
            raise ValueError("dry-anchor samples contain non-finite observations or out-of-range labels")

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _emit(self, final: bool) -> None:
        import numpy as np
        import torch as th

        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        with th.no_grad():
            raw_obs = th.as_tensor(self.X, device=self.model.device)
            # The dry-level anchor is record-only telemetry: the mask keeps the old v28-v30 definition (11/12 always masked)
            # to keep the cross-leg mismatch curve on the same scale -- after v32 autonomy the deployment mask includes 12, but the teacher
            # labels (demos) have no 12, and autonomy behaviour is measured separately by the evaluation a12 gauge (PREREG-v32
            # definition note); changing this definition would break the historical telemetry series, so it is recorded as is and not changed.
            masks = th.ones((len(self.X), 15), dtype=th.bool,
                            device=self.model.device)
            masks[:, 11] = masks[:, 12] = False
            masks[:, 14] = raw_obs[:, _GEAR_PRESENT_INDEX] > 0.5
            policy_obs = _probe_policy_observation_view(
                self.model, raw_obs)
            dist = self.model.policy.get_distribution(
                policy_obs, action_masks=masks)
            pred = dist.distribution.logits.argmax(-1).cpu().numpy()
        mis = float((pred != self.Y).mean())
        line = {"sentinel": "dry-anchor", "step": step,
                "mismatch": round(mis, 4), "n": int(len(self.Y))}
        if final:
            line["final"] = True
        with open(self.run_dir / "sentinel.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._last_emit_step = step
        print(f"   [dry-anchor] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryCurriculumCallback(BaseCallback):
    """E1 5A: register the next p_skip before collecting each rollout.

    - the p table is anchored to the leg-relative rollout index (num_timesteps − 3,497,984)/2048,
      global-step anchoring is disabled; before-leg/misaligned/out-of-range all raise (no clamping -- this closes off the adversarial
      "clamp out-of-range to the table tail, constant 0.5" construction);
    - the schedule_table attribute exposes the full "index->p" table (for the driver to write DRY_CURRICULUM_TABLE);
    - the first value is aligned directly at training start; after that rollout-start goes through
      ``schedule_skip_dry_p(next_p, n_steps)`` to register an independent countdown for each Worker env.
      After the n-th Worker action completes, the environment atomically commits next_p before building next_obs, crossing a window or
      a VecEnv auto-reset;
    - the rollout-tail callback only verifies the commit receipt and feature 601, and no longer switches for the first time after the fact;
      the per-rollout ledger and the values read back from env must still equal the registered table exactly; any mismatch raises.
    """

    def __init__(self, schedule_table, run_dir: pathlib.Path | None = None,
                 leg_start: int = _DRY_CURRICULUM_LEG_START):
        super().__init__()
        table = tuple(float(p) for p in schedule_table)
        _require(len(table) > 0, "dry-curriculum p table must not be empty")
        _require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in table),
                 "dry-curriculum p table must lie entirely in [0, 1]")
        self.schedule_table = table
        self.leg_start = int(leg_start)
        self.run_dir = pathlib.Path(run_dir) if run_dir is not None else None
        self.pushed: list[dict] = []   # per-rollout ledger of what was actually pushed (index->p)
        self.quantum = None
        self._active_index = None
        self._active_p = None
        self._boundary_preapplied_index = None
        self._scheduled_next_index = None
        self._scheduled_next_p = None

    def _on_training_start(self) -> None:
        self.quantum = int(self.model.n_steps * self.model.get_env().num_envs)
        index = self._rollout_index()
        self._activate(
            index,
            getattr(self.model, "_last_obs", None),
            boundary_preapply=False,
        )

    def _rollout_index(self) -> int:
        offset = int(self.num_timesteps) - self.leg_start
        _require(offset >= 0,
                 f"dry-curriculum leg-relative anchoring meaningless: num_timesteps={self.num_timesteps} "
                 f"is before the leg start {self.leg_start} (global-step anchoring disabled; the leg always relaunches from the throne zip)")
        _require(offset % self.quantum == 0,
                 f"dry-curriculum rollout boundary misaligned: in-leg offset {offset} "
                 f"is not a multiple of the quantum {self.quantum}")
        index = offset // self.quantum
        _require(index < len(self.schedule_table),
                 f"dry-curriculum leg-relative rollout index {index} out of range"
                 f" (p table length {len(self.schedule_table)}, no clamping)")
        return index

    def _refresh_dual_observation(
            self, observation, p: float, label: str) -> bool:
        """Refresh the cached Markov state when only curriculum p changes."""
        if observation is None:
            return False
        import numpy as np
        from diablogym.options_env import (
            DUAL_WORKER_OBSERVATION_DIM,
            DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
        )

        _require(
            isinstance(observation, np.ndarray)
            and observation.ndim == 2
            and observation.shape[0]
            == int(self.model.get_env().num_envs)
            and observation.shape[1] in {
                298, DUAL_WORKER_OBSERVATION_DIM},
            f"dry-curriculum {label} cached observation shape invalid: "
            f"{getattr(observation, 'shape', None)!r}",
        )
        if observation.shape[1] != DUAL_WORKER_OBSERVATION_DIM:
            return False
        _require(
            bool(observation.flags.writeable)
            and np.issubdtype(observation.dtype, np.floating)
            and bool(np.isfinite(observation).all()),
            f"dry-curriculum {label} dual cached observation "
            "must be writable, floating-point and finite",
        )
        observation[
            :, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE] = p
        _require(
            bool(np.all(
                observation[
                    :, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]
                == np.asarray(p, dtype=observation.dtype)
            )),
            f"dry-curriculum {label} dual p_skip cache refresh failed",
        )
        return True

    def _verify_active_probability(self, index: int, p: float) -> None:
        env = self.model.get_env()
        # rev3 E1-2 identity assertion: the actually present p (read back per env) == the matching registered entry; any mismatch raises.
        for rank, actual in enumerate(env.get_attr("skip_dry")):
            _require(float(actual) == p,
                     f"dry-curriculum identity assertion mismatch: env[{rank}] present p={actual} "
                     f"!= registered[{index}]={p}")

    def _schedule_states(self) -> list[dict]:
        states = self.model.get_env().env_method(
            "skip_dry_schedule_state")
        _require(
            isinstance(states, (list, tuple))
            and len(states) == int(self.model.get_env().num_envs)
            and all(isinstance(state, dict) for state in states),
            "dry-curriculum per-env schedule state shape invalid",
        )
        return list(states)

    def _verify_schedule_states(
            self, *, current_p: float,
            pending_p: float | None, remaining: int) -> None:
        expected_keys = {
            "current_probability",
            "pending_probability",
            "remaining_env_steps",
        }
        for rank, state in enumerate(self._schedule_states()):
            _require(
                set(state) == expected_keys
                and float(state["current_probability"]) == current_p
                and (
                    state["pending_probability"] is None
                    if pending_p is None
                    else float(state["pending_probability"]) == pending_p
                )
                and state["remaining_env_steps"] == remaining,
                "dry-curriculum per-env schedule state mismatch: "
                f"env[{rank}]={state!r},"
                f"expected=({current_p},{pending_p},{remaining})",
            )

    def _register_next_probability(self, index: int, p: float) -> None:
        _require(
            self._scheduled_next_index is None
            and self._scheduled_next_p is None,
            "dry-curriculum previous rollout countdown not yet closed",
        )
        target = getattr(self.model, "_total_timesteps", None)
        _require(
            _is_plain_int(target)
            and int(target) >= int(self.num_timesteps) + int(self.quantum),
            "dry-curriculum is missing a valid global training end point",
        )
        has_next_rollout = (
            int(self.num_timesteps) + int(self.quantum) < int(target)
        )
        self._verify_schedule_states(
            current_p=p, pending_p=None, remaining=0)
        if not has_next_rollout:
            return
        next_index = int(index) + 1
        _require(
            next_index < len(self.schedule_table),
            "dry-curriculum next rollout exceeds the registered table: "
            f"{next_index} >= {len(self.schedule_table)}",
        )
        next_p = float(self.schedule_table[next_index])
        remaining = int(self.model.n_steps)
        receipts = self.model.get_env().env_method(
            "schedule_skip_dry_p", next_p, remaining)
        _require(
            isinstance(receipts, (list, tuple))
            and len(receipts) == int(self.model.get_env().num_envs),
            "dry-curriculum countdown registration receipt shape invalid",
        )
        self._verify_schedule_states(
            current_p=p, pending_p=next_p, remaining=remaining)
        self._scheduled_next_index = next_index
        self._scheduled_next_p = next_p

    def _verify_dual_observation_probability(
            self, observation, p: float, label: str) -> bool:
        if observation is None:
            return False
        import numpy as np
        from diablogym.options_env import (
            DUAL_WORKER_OBSERVATION_DIM,
            DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
        )

        _require(
            isinstance(observation, np.ndarray)
            and observation.ndim == 2
            and observation.shape[0]
            == int(self.model.get_env().num_envs)
            and observation.shape[1] in {
                298, DUAL_WORKER_OBSERVATION_DIM},
            f"dry-curriculum {label} cached observation shape invalid: "
            f"{getattr(observation, 'shape', None)!r}",
        )
        if observation.shape[1] != DUAL_WORKER_OBSERVATION_DIM:
            return False
        expected = np.asarray(p, dtype=observation.dtype)
        _require(
            bool(np.all(
                observation[
                    :, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]
                == expected
            )),
            f"dry-curriculum {label} dual p_skip not yet switched atomically by the environment",
        )
        return True

    def _activate(
            self, index: int, observation,
            *, boundary_preapply: bool) -> bool:
        p = self.schedule_table[index]
        env = self.model.get_env()
        env.env_method("set_skip_dry_p", p)   # passed through Monitor __getattr__
        self._verify_active_probability(index, p)
        refreshed = self._refresh_dual_observation(
            observation, p,
            "rollout-tail" if boundary_preapply else "rollout-start",
        )
        self._active_index = int(index)
        self._active_p = float(p)
        self._boundary_preapplied_index = (
            int(index) if boundary_preapply else None)
        return refreshed

    def _on_rollout_start(self) -> None:
        index = self._rollout_index()
        p = self.schedule_table[index]
        preapplied = self._boundary_preapplied_index == index
        _require(
            self._active_index == index and self._active_p == p,
            "dry-curriculum rollout boundary not switched atomically by the in-environment countdown: "
            f"active=({self._active_index},{self._active_p}),"
            f"expected=({index},{p})",
        )
        self._verify_active_probability(index, p)
        refreshed = self._refresh_dual_observation(
            getattr(self.model, "_last_obs", None),
            p,
            "rollout-start",
        )
        entry = {"rollout_index": int(index), "p": float(p),
                 "num_timesteps": int(self.num_timesteps),
                 "boundary_preapplied": bool(preapplied),
                 "cached_dual_observation_refreshed": bool(refreshed)}
        self.pushed.append(entry)
        if self.run_dir is not None:
            with open(self.run_dir / "dry_curriculum.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        self._boundary_preapplied_index = None
        self._register_next_probability(index, p)

    def _on_step(self) -> bool:
        # WorkerWindowEnv owns the actual boundary switch: every environment
        # counts successful Worker actions and commits the pending probability
        # immediately after action N, before it constructs new_obs or returns a
        # terminal that VecEnv will auto-reset.  This callback only verifies
        # that atomic commit and refreshes the exact ndarray later used for the
        # rollout-tail value and next rollout cache.
        n_steps = self.locals.get("n_steps")
        n_rollout_steps = self.locals.get("n_rollout_steps")
        if not (
            _is_plain_int(n_steps)
            and _is_plain_int(n_rollout_steps)
            and n_steps == n_rollout_steps - 1
        ):
            return True
        if self._scheduled_next_index is None:
            _require(
                self._scheduled_next_p is None,
                "dry-curriculum next index/p partially registered",
            )
            self._verify_schedule_states(
                current_p=float(self._active_p),
                pending_p=None,
                remaining=0,
            )
            return True

        next_index = self._rollout_index()
        next_p = float(self.schedule_table[next_index])
        _require(
            self._active_index is not None
            and next_index == self._active_index + 1
            and next_index == self._scheduled_next_index
            and next_p == self._scheduled_next_p,
            "dry-curriculum rollout-tail index/probability not consecutive: "
            f"active={self._active_index},"
            f"scheduled=({self._scheduled_next_index},"
            f"{self._scheduled_next_p}),next=({next_index},{next_p})",
        )
        self._verify_active_probability(next_index, next_p)
        self._verify_schedule_states(
            current_p=next_p, pending_p=None, remaining=0)
        new_obs = self.locals.get("new_obs")
        self._verify_dual_observation_probability(
            new_obs, next_p, "rollout-tail")
        self._refresh_dual_observation(
            new_obs, next_p, "rollout-tail")
        self._active_index = next_index
        self._active_p = next_p
        self._boundary_preapplied_index = next_index
        self._scheduled_next_index = None
        self._scheduled_next_p = None
        return True


# ---- New E5 gauges (PREREG-v33-content-case E5; all record-only, no verdict, zero intrusion by default, included in the W-G0
# proof scope; the probe demo set is bound to the strict-PASS BC-v1 receipt of the current protocol) ----


def _probe_legacy_masks(obs):
    """Old-definition mask shared by the E5 probes (verbatim from the DryAnchorSentinel v28-v30 definition: 11/12 always masked,
    14 follows the gear bit) -- the demo set is from the v1 generation with no per-sample masks, and the full deployment mask cannot be fully
    rebuilt from obs (E2 note: inferring it would be a second source of truth, forbidden); take the old cross-leg same-scale definition and note
    mask_mode on each output row, record-only, no verdict (implementation discretion, listed separately in the handover notes)."""
    import torch as th

    masks = th.ones((len(obs), 15), dtype=th.bool, device=obs.device)
    masks[:, 11] = masks[:, 12] = False
    masks[:, 14] = obs[:, _GEAR_PRESENT_INDEX] > 0.5
    return masks


def _probe_policy_observation_view(model, raw_obs):
    """Feed probe rows through the same policy-view boundary as deployment.

    A12 custom policies need the untouched signed latch for their contextual
    gate and decode only the inherited actor/value path internally.  Every
    ordinary Worker policy consumes the canonical protocol-v3 view.
    """
    from leashed_ppo import (
        A12MixtureMaskableActorCriticPolicy,
        ASYMMETRIC_WORKER_LEGACY_DIM,
        ASYMMETRIC_WORKER_OBSERVATION_DIM,
        AsymmetricWorkerMaskableActorCriticPolicy,
        _legacy_worker_observation_view,
    )

    if isinstance(model.policy, A12MixtureMaskableActorCriticPolicy):
        return raw_obs
    if isinstance(model.policy, AsymmetricWorkerMaskableActorCriticPolicy):
        if raw_obs.shape[-1] == ASYMMETRIC_WORKER_OBSERVATION_DIM:
            return raw_obs
        _require(
            raw_obs.shape[-1] == ASYMMETRIC_WORKER_LEGACY_DIM,
            "asymmetric Worker offline probe only accepts 298-dim or the current dual-dim observations",
        )
        # Historical dry-anchor rows contain no trustworthy v4/controller
        # context.  Preserve their exact v3 root and explicitly mark every
        # appended field unknown-as-zero; never reinterpret old columns as
        # current state merely to satisfy the wider policy shape.
        padded = raw_obs.new_zeros(
            (*raw_obs.shape[:-1], ASYMMETRIC_WORKER_OBSERVATION_DIM))
        padded[..., :ASYMMETRIC_WORKER_LEGACY_DIM] = (
            _legacy_worker_observation_view(raw_obs))
        return padded
    return _legacy_worker_observation_view(raw_obs)


class DistillCeProbe(BaseCallback):
    """E5-1 dry/fresh split distill_ce offline probe (record-only, no verdict; twin of DryAnchorSentinel).

    On the fixed demo-state set (bound to the current-protocol BC-v1 PASS receipt), split into dry/fresh groups by whether the col296 short clock or
    the decoded col297 farm_scene_fraction reaches the cap-1 frontier, and
    compute the teacher-student distill CE (formula mirrors the leash block of leashed_ppo train():
    ce = mean of −Σ t_probs·logp_all; teacher and student are fed the same old-definition mask); dedicated rng
    (following the dry-anchor rng(26) precedent), zero contact with the training path (pure read + IO, never touching the training RNG/gradients/
    env streams). The in-training buffer split was abolished to protect the G0-2a zero-intrusion proof surface (engineering M2);
    this is its registered replacement; the course-2 calibration data supply obligation (rewritten in review round 12) is also met here.
    Output run_dir/distill_ce_probe.jsonl.
    """

    def __init__(self, run_dir: pathlib.Path, demos_npz: str, every: int):
        super().__init__()
        import numpy as np

        self.run_dir = run_dir
        self.every = int(every)
        _require(self.every > 0, "distill-ce probe interval must be > 0")
        self.next_at = self.every
        self._last_emit_step = None
        expected_sha = _assert_bc_v1_demos_frozen(demos_npz)
        X, _, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, expected_sha)
        dry, fresh = _dry_anchor_partition(X)
        dry_rows = np.flatnonzero(dry)
        fresh_rows = np.flatnonzero(fresh)
        _require(len(dry_rows) > 0 and len(fresh_rows) > 0,
                 "distill-ce probe needs both the dry and fresh demo-state groups non-empty (fail-loud)")
        rng = np.random.default_rng(_E5_PROBE_RNG_SEED)
        dry_idx = rng.choice(dry_rows,
                             size=min(_E5_PROBE_GROUP_CAP, len(dry_rows)),
                             replace=False)
        fresh_idx = rng.choice(fresh_rows,
                               size=min(_E5_PROBE_GROUP_CAP, len(fresh_rows)),
                               replace=False)
        self.X_dry, self.X_fresh = X[dry_idx], X[fresh_idx]
        if not (np.isfinite(self.X_dry).all() and np.isfinite(self.X_fresh).all()):
            raise ValueError("distill-ce probe samples contain non-finite observations")

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _group_metrics(self, x) -> dict:
        import torch as th

        from leashed_ppo import (
            _legacy_distillation_masks,
            _masked_log_softmax_from_raw,
        )

        with th.no_grad():
            raw_obs = th.as_tensor(x, device=self.model.device)
            masks = _probe_legacy_masks(raw_obs)
            policy_obs = _probe_policy_observation_view(
                self.model, raw_obs)
            distill_masks = _legacy_distillation_masks(
                masks)
            t_probs = self.model._teacher_probs(
                policy_obs, distill_masks)
            student_logits = self.model._student_distillation_logits(
                policy_obs)
            logp_all = _masked_log_softmax_from_raw(
                student_logits, distill_masks)
            teacher_logp = th.where(
                t_probs > 0.0,
                th.log(t_probs),
                th.zeros_like(t_probs),
            )
            ce = -(t_probs * logp_all).sum(dim=-1).mean()
            entropy = -(
                t_probs * teacher_logp).sum(dim=-1).mean()
            kl = (
                t_probs * (teacher_logp - logp_all)
            ).sum(dim=-1).mean()
            tv = 0.5 * th.abs(
                t_probs - logp_all.exp()).sum(dim=-1).mean()
            return {
                "ce": float(ce),
                "teacher_entropy": float(entropy),
                "kl": float(kl),
                "tv": float(tv),
            }

    def _group_ce(self, x) -> float:
        """Compatibility accessor for the historical CE-only probe."""
        return self._group_metrics(x)["ce"]

    def _emit(self, final: bool) -> None:
        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        _require(getattr(self.model, "teacher", None) is not None,
                 "distill-ce probe needs the teacher present (Leashed teacher; fail-loud)")
        dry = self._group_metrics(self.X_dry)
        fresh = self._group_metrics(self.X_fresh)
        line = {"probe": "distill-ce", "step": step,
                "dry_ce": round(dry["ce"], 6),
                "dry_teacher_entropy":
                    round(dry["teacher_entropy"], 6),
                "dry_kl": round(dry["kl"], 6),
                "dry_tv": round(dry["tv"], 6),
                "dry_n": int(len(self.X_dry)),
                "fresh_ce": round(fresh["ce"], 6),
                "fresh_teacher_entropy":
                    round(fresh["teacher_entropy"], 6),
                "fresh_kl": round(fresh["kl"], 6),
                "fresh_tv": round(fresh["tv"], 6),
                "fresh_n": int(len(self.X_fresh)),
                "beta_initial":
                    getattr(self.model, "distill_beta", None),
                "beta": getattr(
                    self.model,
                    "_last_effective_distill_beta",
                    None),
                "distill_actor_rollouts_completed": getattr(
                    self.model,
                    "_distill_actor_rollouts_completed",
                    None),
                "mask_mode": "legacy-root-exclude-a12-a14",
                "demos_sha16": self.demos_sha256[:16]}
        if final:
            line["final"] = True
        with open(self.run_dir / "distill_ce_probe.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._last_emit_step = step
        print(f"   [dry/fresh distill probe] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryWindowMetricsCallback(BaseCallback):
    """E5-2 dry-window behaviour gauge (record-only, no verdict; the starting point for closing audit gap i, with the baseline first built in this case).

    Two reading surfaces, both hung on existing sampling surfaces with zero new training-side contact:
    (1) dry-state action distribution -- on the fixed dry-state demo set (bound to the current-protocol BC-v1 PASS receipt,
       the same sampling surface as dry-anchor), the student policy's distribution entropy and argmax histogram (old-definition mask,
       with a mask_mode note);
    (2) window economy -- the end-of-window option_extra in the SB3 rollout infos stream (learning windows; fast-forward windows
       do not pass through this stream), aggregating wage W and width (τ̄/depth=dlvl_end) by dry/fresh group; reset at every emit
       interval (interval-local means); a group with n=0 records n:0 and mean null rather than vanishing (fail-closed).
    Output run_dir/drywin_metrics.jsonl (process-side raw material for the ledger entry DRYWIN_METRICS).
    """

    _WINDOW_KEYS = ("n", "wage_sum", "tau_sum", "depth_sum")

    def __init__(self, run_dir: pathlib.Path, demos_npz: str, every: int):
        super().__init__()
        import numpy as np

        self.run_dir = run_dir
        self.every = int(every)
        _require(self.every > 0, "drywin gauge interval must be > 0")
        self.next_at = self.every
        self._last_emit_step = None
        expected_sha = _assert_bc_v1_demos_frozen(demos_npz)
        X, _, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, expected_sha)
        dry, _ = _dry_anchor_partition(X)
        dry_rows = np.flatnonzero(dry)
        _require(len(dry_rows) > 0, "drywin gauge needs a non-empty dry-state demo set (fail-loud)")
        rng = np.random.default_rng(_E5_PROBE_RNG_SEED)
        idx = rng.choice(dry_rows,
                         size=min(_E5_PROBE_GROUP_CAP, len(dry_rows)),
                         replace=False)
        self.X_dry = X[idx]
        if not np.isfinite(self.X_dry).all():
            raise ValueError("drywin gauge dry-state samples contain non-finite observations")
        self._acc = self._fresh_acc()

    @classmethod
    def _fresh_acc(cls) -> dict:
        return {group: dict.fromkeys(cls._WINDOW_KEYS, 0)
                for group in ("dry", "fresh")}

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            extra = info.get("option_extra") if isinstance(info, dict) else None
            if extra is None:
                continue
            acc = self._acc["dry" if extra.get("dry") else "fresh"]
            acc["n"] += 1
            acc["wage_sum"] += float(extra["W"])
            acc["tau_sum"] += float(extra["tau"])
            acc["depth_sum"] += float(extra["dlvl_end"])
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _dry_state_readout(self) -> tuple[float, list[int]]:
        import numpy as np
        import torch as th

        with th.no_grad():
            raw_obs = th.as_tensor(self.X_dry, device=self.model.device)
            masks = _probe_legacy_masks(raw_obs)
            policy_obs = _probe_policy_observation_view(
                self.model, raw_obs)
            dist = self.model.policy.get_distribution(
                policy_obs, action_masks=masks)
            entropy = float(dist.distribution.entropy().mean())
            pred = dist.distribution.logits.argmax(-1).cpu().numpy()
        hist = np.bincount(pred, minlength=15)
        return entropy, [int(count) for count in hist]

    @staticmethod
    def _window_summary(acc: dict) -> dict:
        n = acc["n"]
        mean = (lambda total: round(total / n, 4) if n else None)
        return {"n": int(n), "wage_mean": mean(acc["wage_sum"]),
                "tau_mean": mean(acc["tau_sum"]),
                "depth_mean": mean(acc["depth_sum"])}

    def _emit(self, final: bool) -> None:
        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        entropy, hist = self._dry_state_readout()
        line = {"metrics": "drywin", "step": step,
                "dry_state_entropy": round(entropy, 6),
                "dry_state_n": int(len(self.X_dry)),
                "dry_state_argmax_hist": hist,
                "windows": {group: self._window_summary(self._acc[group])
                            for group in ("dry", "fresh")},
                "mask_mode": "dry-anchor-legacy",
                "demos_sha16": self.demos_sha256[:16]}
        if final:
            line["final"] = True
        with open(self.run_dir / "drywin_metrics.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._acc = self._fresh_acc()
        self._last_emit_step = step
        print(f"   [dry-window behaviour] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


# E5-3 canary a12/episode mid-run gauge (for the offline checkpoint series; a standalone statistics function + recorder,
# not hung on a training callback -- zero contact with the training path, RC.11 recorded point by point, record-only, no verdict).
_A12_CANARY_SCHEMA_VERSION = "a12-canary/1"
_A12_CANARY_STATS_KEYS = frozenset({
    "episodes", "a12_total", "a12_per_episode", "episodes_with_a12", "a12_max"})


def a12_canary_stats(a12_counts) -> dict:
    """E5-3 statistics: per-episode a12 actual-drink count sequence -> a12/episode reading (called by the driver after extracting
    per-episode values from the checkpoint evaluation archive). Empty sequences/negatives/non-integers fail loudly (a12/episode over zero episodes is undefined,
    and silently recording 0 as if measured is forbidden)."""
    counts = list(a12_counts)
    _require(len(counts) > 0, "a12 canary statistics need >=1 episode (empty sequence fails loudly)")
    _require(all(_is_plain_int(count) for count in counts),
             "a12 per-episode counts must all be integers")
    _require(all(count >= 0 for count in counts), "a12 per-episode counts must not be negative")
    total = sum(counts)
    return {"episodes": len(counts),
            "a12_total": int(total),
            "a12_per_episode": round(total / len(counts), 6),
            "episodes_with_a12": sum(1 for count in counts if count > 0),
            "a12_max": int(max(counts))}


def record_a12_canary(out_path: str | pathlib.Path, *, checkpoint_step: int,
                      manager: str, stats: dict, tag: str | None = None) -> dict:
    """E5-3 recorder: write one jsonl line with the a12 canary reading (process-side raw material for the ledger entry A12_CANARY;
    closed schema, exact key-set assertion, fail-loud). Returns the written line."""
    _require(_is_plain_int(checkpoint_step) and checkpoint_step >= 0,
             "a12 canary checkpoint_step must be a non-negative integer")
    _require(isinstance(manager, str) and bool(manager),
             "a12 canary manager must be a non-empty string")
    _require(isinstance(stats, dict) and set(stats) == set(_A12_CANARY_STATS_KEYS),
             f"a12 canary stats key set must equal exactly {sorted(_A12_CANARY_STATS_KEYS)}")
    line = {"canary": "a12", "schema_version": _A12_CANARY_SCHEMA_VERSION,
            "checkpoint_step": int(checkpoint_step), "manager": manager}
    if tag is not None:
        _require(isinstance(tag, str) and bool(tag),
                 "a12 canary tag, when given, must be a non-empty string")
        line["tag"] = tag
    line.update(stats)
    with open(out_path, "a") as f:
        f.write(json.dumps(line) + "\n")
    return line


class EpisodeJsonlCallback(BaseCallback):
    """Write each episode's results to progress.jsonl; refresh status.json periodically (polled by the dashboard)."""

    def __init__(self, run_dir: pathlib.Path, config: dict):
        super().__init__()
        self.run_dir = run_dir
        self.config = config
        self.ep_count = 0
        self.t0 = time.time()
        self._progress = open(run_dir / "progress.jsonl", "a", buffering=1)
        self._last_status = 0.0
        self._steps0 = 0

    def _on_training_start(self) -> None:
        # v24 fix: sps counts this leg's increment (otherwise a resume leg reads tens of times too high and the slowdown gate goes blind)
        self._steps0 = self.num_timesteps
        self.t0 = time.time()

    def _write_status(self, now: float, training_ended: bool = False) -> None:
        elapsed = now - self.t0
        target_steps = self.config.get("target_global_steps",
                                       self.config["total_steps"])
        rollout_full = bool(getattr(
            getattr(getattr(self, "model", None), "rollout_buffer", None),
            "full", False))
        status = {
            "run": self.run_dir.name,
            "total_steps": int(self.num_timesteps),
            "target_steps": target_steps,
            "start_steps": self.config.get("start_steps", 0),
            "leg_steps": int(self.num_timesteps
                             - self.config.get("start_steps", 0)),
            "leg_target_steps": self.config["total_steps"],
            "episodes": self.ep_count,
            "sps": round((self.num_timesteps - self._steps0) / max(1e-9, elapsed)),
            "elapsed_sec": round(elapsed),
            "updated_at": now,
            "training_ended": training_ended,
            "rollout_full": rollout_full,
            "target_reached": int(self.num_timesteps) >= int(target_steps),
            "config": self.config,
        }
        # dashboard polling must never read half a JSON file.
        tmp = self.run_dir / "status.tmp.json"
        tmp.write_text(json.dumps(status, ensure_ascii=False))
        tmp.replace(self.run_dir / "status.json")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is None:
                continue
            self.ep_count += 1
            extra = info.get("episode_extra", {})
            line = {
                "ep": self.ep_count,
                "t": round(time.time() - self.t0, 1),
                "reward": round(float(ep["r"]), 3),
                "len": int(ep["l"]),
                **extra,
            }
            self._progress.write(json.dumps(line, ensure_ascii=False) + "\n")

        now = time.time()
        if now - self._last_status > 1.0:
            self._last_status = now
            self._write_status(now)
        return True

    def _on_training_end(self) -> None:
        # Short runs or early stops can finish inside the 1s refresh window; without a forced flush
        # the driver would misjudge a complete leg as a few steps short.
        self._write_status(time.time(), training_ended=True)
        self.close()

    def close(self) -> None:
        """Close the per-episode log file idempotently, also on error paths."""
        if not self._progress.closed:
            self._progress.close()


def _record_run_publication_status(
        run_dir: pathlib.Path, state: str, *,
        model_sha256: str | None = None,
        detail: str | None = None,
        completion_report: dict | None = None,
        diagnostic_artifact: dict | None = None,
        learn_returned: bool | None = None) -> None:
    """Atomically disambiguate "training ended" from "model published".

    ``EpisodeJsonlCallback`` necessarily finishes before the final behavior and
    provenance gates run.  Without a second, terminal status write, a scheduler
    that only watches ``status.json`` can mistake a refused publication for a
    successful run even though the process exits non-zero.
    """
    allowed = {
        "PUBLISHED", "PRODUCTION_CANDIDATE", "DEVELOPMENT_ONLY",
        "TRAINING_ERROR", "PUBLICATION_REFUSED",
    }
    _require(state in allowed, f"unknown release state: {state}")
    if state in {"PUBLISHED", "PRODUCTION_CANDIDATE", "DEVELOPMENT_ONLY"}:
        _require(_is_sha256(model_sha256),
                 f"{state} state must bind the matching model SHA-256")
    else:
        _require(model_sha256 is None,
                 f"{state} state must not register a published model SHA-256")

    status_path = run_dir / "status.json"
    try:
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read the final status.json: {exc}") from exc
    _require(isinstance(status, dict), "status.json top level must be an object")
    status.update({
        "training_ended": True,
        "publication_status": state,
        "model_published": state == "PUBLISHED",
        "model_production_candidate": state == "PRODUCTION_CANDIDATE",
        "model_development_only": state == "DEVELOPMENT_ONLY",
        "model_sha256": model_sha256,
        "publication_detail": detail,
        "updated_at": time.time(),
    })
    if completion_report is not None:
        status["training_completion"] = completion_report
    if diagnostic_artifact is not None:
        status["training_diagnostic"] = diagnostic_artifact
    if learn_returned is not None:
        status["learn_returned_normally"] = bool(learn_returned)
    tmp = run_dir / "status.tmp.json"
    tmp.write_text(json.dumps(status, ensure_ascii=False))
    tmp.replace(status_path)


def _main(resources: _TrainingResources):
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=1_998_848,
                    help="number of new samples; must be divisible by n_steps×num_envs, so SB3 cannot silently over-sample")
    ap.add_argument(
        "--artifact-scope",
        choices=tuple(_ARTIFACT_SCOPE_RESULTS),
        default="production",
        help="development only produces model_development.zip/DEVELOPMENT_ONLY, "
             "candidate only produces model_candidate.zip/PRODUCTION_CANDIDATE; "
             "neither may read final-heldout or pose as a release artifact; only production produces"
             " model_final.zip/PUBLISHED",
    )
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--diagnostic-rollout", choices=("disabled", "first-update-v1"),
                    default="disabled",
                    help="Optional single-update evidence capture; fresh resource candidate only; no publication override")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--device", default="cpu", help="cpu / mps (cpu is usually faster for a small MLP)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=512, help="sampling steps per env per round")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "rppo", "mppo"],
                    help="rppo = RecurrentPPO/LSTM (plan B: learned memory instead of a hand-written macro state machine); "
                         "mppo = MaskablePPO (v16: invalid-action masking, env.action_masks)")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "attn"],
                    help="attn = entity-attention perception (v9: AlphaStar-style entity encoder + map CNN)")
    ap.add_argument("--max-steps", type=int, default=1500,
                    help="episode step cap; 1500 = champion (v6) recipe, 3000 = v10 long-episode experiment")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (SB3 global seed + environment reset seed; multi-process sampling timing still adds a little nondeterminism, so only approximate reproduction is guaranteed)")
    ap.add_argument("--deep", action="store_true",
                    help="v17 deep water: descend bonus grows with depth (N->N+1 pays 8×N); use with --max-steps 3000")
    ap.add_argument("--death-ladder", action="store_true",
                    help="v18: death cost priced by depth (dying on level N costs 8×N, replacing the constant -2)")
    ap.add_argument("--options", action="store_true",
                    help="v22: policy brain/operator brain (OptionsEnv, Discrete(3); requires --algo mppo --gamma 1.0)")
    ap.add_argument("--flat-clock", action="store_true",
                    help="v22 devil arm: 296-dim flat (stall clock in the observation), use with --bc-init")
    ap.add_argument("--worker", action="store_true",
                    help="v23: on-policy training of the FARM operator brain (WorkerWindowEnv, Discrete(15), 11/12 masked; "
                         "requires --algo mppo --gamma 1.0, see docs/prereg/PREREG-v23.md)")
    ap.add_argument("--manager-npz",
                    default=str(pathlib.Path(__file__).resolve().parent
                                / "models" / "v22-h-manager" / "policy.npz"),
                    help="frozen manager weights npz (produced by export_manager_npz.py)")
    ap.add_argument("--worker-npz", default=None,
                    help="v25: mount an npz worker during manager training (OptionsEnv workers assembly point)")
    ap.add_argument("--worker-zip", default=None,
                    help="R9: mount an SB3 zip worker during manager training (certified release artifact, the MaskablePPO"
                         " frozen forward is loaded inside subprocesses; mutually exclusive with --worker-npz, --options only)")
    ap.add_argument("--worker-zip-sha256", default=None,
                    help="R9: expected SHA-256 of --worker-zip (64 lowercase hex digits); "
                         "if given, the zip byte identity is checked before loading")
    ap.add_argument("--resource-protocol", default="off",
                    choices=("off", "l2-town-v1"),
                    help="Versioned native readiness gate and L1 town service")
    ap.add_argument("--resource-purchase-mode", default="full",
                    choices=("none", "heal", "potions", "armor", "full"),
                    help="Resource ablation arm; requires resource protocol")
    ap.add_argument("--resource-service-policy", default="legacy-v1",
                    choices=("legacy-v1", "sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"),
                    help="Explicit service identity; sustain-v2/v3/v4/v5/v6 and sustain-loot-v1 "
                         "require l2-town-v1/full, and sustain-loot-v1 additionally requires a "
                         "completion-l2 --worker-time-protocol (R18-B3)")
    ap.add_argument("--resource-readiness-law", default="veto-v1",
                    choices=("veto-v1", "coach-v03"),
                    help="R17.1 ruling 3 readiness law; coach-v03 (six native "
                         "conditions, health excluded) requires l2-town-v1")
    ap.add_argument("--resource-retreat", default="off",
                    choices=("off", "retreat-v1"),
                    help="R18-B return-to-town law; retreat-v1 requires "
                         "l2-town-v1 under coach-v03 and --worker/--options")
    ap.add_argument("--resource-portal", default="off",
                    choices=("off", "portal-v1"),
                    help="R18-B5 Scroll of Town Portal vehicle for the retreat; "
                         "portal-v1 requires l2-town-v1 under coach-v03, "
                         "--resource-retreat retreat-v1 and --worker/--options")
    ap.add_argument("--resource-sweep", default="off",
                    choices=("off", "sweep-v1"),
                    help="R18-B6 chest/barrel sweep on main L1; sweep-v1 requires "
                         "l2-town-v1 with --resource-service-policy sustain-loot-v1 "
                         "and --worker/--options")
    ap.add_argument("--resource-identify", default="off",
                    choices=("off", "cain-v1"),
                    help="R18-B6 Cain identify leg of the loot town trip; cain-v1 "
                         "requires l2-town-v1 with sustain-loot-v1 and "
                         "--worker/--options")
    ap.add_argument("--resource-weapon-upgrade", default="off",
                    choices=("off", "smith-v1"),
                    help="R18-B6 Griswold weapon upgrade leg; smith-v1 requires "
                         "l2-town-v1/full with sustain-loot-v1 and "
                         "--worker/--options (dry-v1 is deployment-only: it is "
                         "an observation twin and would mint a training identity "
                         "for a bit-identical control arm)")
    ap.add_argument("--hunt-scope", default="all",
                    choices=("all", "l1-only"),
                    help="R18-B5 scope of the a10 global hunt; l1-only masks it "
                         "on main L2+ and requires --worker/--options")
    ap.add_argument("--dive-blocker-recovery", default="off", choices=("off", "adjacent-v1"),
                    help="Explicit bounded a11 blocker combat; requires resource protocol")
    ap.add_argument("--manager-heuristic", default=None,
                    choices=("level-margin-1", "readiness-v1",
                             "readiness-v2", "readiness-v3"),
                    help="R12 amendment 2: scripted coach in worker mode (dive when drained or one level higher, "
                         "otherwise farm; the canonical rule in the repo). Mutually exclusive with --manager-npz")
    ap.add_argument("--reward-economy", default="v1",
                    choices=("v1", "v2", "v3", "v3b", "v4"),
                    help="R10: reward economy rule. v1 = historical constants unchanged bit for bit (default); "
                         "v2 = depth economy (A higher descend bonus / B farm income depth multiplier / "
                         "C decreasing death penalty / D anti-idling / E1 equipment repricing, "
                         "all booked to the manager; approved 2026-08-27); v4 = R16 amendment "
                         "economy (semantics defined on the env.py side)")
    # R16 amendment: new environment-side (DiabloGymEnv)/classroom-side (OptionsEnv) parameters -- passed through
    # make_env explicitly (the worker path goes WorkerWindowEnv **env_kwargs ->
    # OptionsEnv -> DiabloGymEnv, each __init__ consuming named parameters). Defaults
    # reproduce the old rule bit for bit; semantics are implemented in env.py/options_env.py.
    ap.add_argument("--explore-global-fallback", action="store_true",
                    help="R16 (DiabloGymEnv): global fallback for the explore macro -- when local candidates run out, "
                         "fall back to unexplored targets on the whole map; default off = old rule")
    ap.add_argument("--explore-global-hunt", action="store_true",
                    help="R16 C5 main lever (DiabloGymEnv, later addition): when no monster is visible in the 25×25 window, a10"
                         " uses privileged information (radius-112 full-map BFS) to advance toward the nearest"
                         " living monster; in an 8-seed probe fallback 375->375 kills almost never triggered, "
                         "hunt 375->670 kills; default off = old rule")
    ap.add_argument("--progress-far-tiles", type=int, default=0,
                    help="R16 (DiabloGymEnv): tile-count threshold of the far-cell progress criterion; "
                         "0 = off (old rule)")
    ap.add_argument("--farm-scene-cap", type=int,
                    default=_FARM_SCENE_CAP_DEFAULT,
                    help="R16 (OptionsEnv): FARM window scene step cap; default 1800 "
                         "mirrors options_env.FARM_SCENE_CAP (old rule); only "
                         "--worker/--options")
    ap.add_argument("--reset-layer-clock-on-window", action="store_true",
                    help="R16 (OptionsEnv): reset the level clock at the start of each window; "
                         "default off = old rule; --worker/--options only")
    ap.add_argument("--deep-start-curriculum", default=None,
                    help="R9 curriculum arm: deep-start prologue, format 'p=0.5,target=2,cap=8'"
                         " (p = per-episode independent trigger probability [deterministic RNG derived from the episode seed], target = target"
                         " dungeon_level, cap = cap on play-through windows; death/truncation redraws capped at 8; "
                         "the prologue completes inside _SeedDiscipline.reset and stays out of the Monitor ledger; "
                         "--options only)")
    ap.add_argument("--deep-start-form", choices=["dive", "exhausted"], default=None,
                    help="R9 curriculum prologue form (default dive): dive = repeatedly choose DIVE (FARM if"
                         " masked) until dungeon_level>=target; exhausted = FARM until drained and DIVE"
                         " is legal, then hand over; only with --deep-start-curriculum")
    ap.add_argument(
        "--manager-policy-observation-view",
        choices=["raw-v4", "legacy-v3"],
        default="raw-v4",
        help="Options manager policy input contract: new managers default to raw-v4; choose legacy-v3 explicitly only when continuing/reproducing"
             " the old M29. The frozen M29 inside WorkerWindow "
             "does not read this flag; the environment fixes it to legacy-v3",
    )
    ap.add_argument("--skip-dry", action="store_true",
                    help="v26 oasis: dry-level revisit windows are played by the script, the worker only takes lessons in fresh-level windows")
    ap.add_argument("--dry-curriculum-schedule", default=None,
                    help="E1 5A dry-window course annealing table: comma-separated segments, "
                         "'linear:<p0>:<p1>:<n>' (n>=2, endpoints inclusive, linear) or 'hold:<p>:<n>'; "
                         "p in [0,1] is the probability that the script plays a dry-level revisit window; the table is indexed by the leg-relative rollout index"
                         " (num_timesteps−3497984)/2048 and pushed per rollout before collection; "
                         "mutually exclusive with --skip-dry, --worker only. Main table = "
                         "linear:1.0:0.5:147,hold:0.5:97")
    action12_group = ap.add_mutually_exclusive_group()
    action12_group.add_argument(
        "--drink-sovereignty",
        dest="drink_sovereignty",
        action="store_const",
        const=True,
        help="explicitly ask the environment to manage Worker action12; when a strict Worker NPZ is mounted it "
             "must match its metadata",
    )
    action12_group.add_argument(
        "--no-drink-sovereignty",
        dest="drink_sovereignty",
        action="store_const",
        const=False,
        help="turn off the worker's potion autonomy (m[12] always masked again); when a strict Worker NPZ is mounted it "
             "must match its metadata",
    )
    ap.set_defaults(drink_sovereignty=None)
    ap.add_argument(
        "--legacy-worker-policy-observation-view",
        action="store_true",
        help="required for rev15 ordinary Worker actor/value: at the environment boundary rebuild feature286/297 "
             "together with the monster/map/item channels into the full protocol-v3; "
             "an A12 custom policy disables this flag and receives legacy-v3-a12-overlay, "
             "decoding the old network input internally",
    )
    ap.add_argument(
        "--worker-policy-observation-view",
        choices=[
            _WORKER_VIEW_LEGACY_V3,
            _WORKER_VIEW_DUAL_V4_ASYMMETRIC,
            # v5 (dual-v5-window-mode-v1) is kept in reserve for R13.2: the environment-side observation is
            # implemented, but the frozen layout/parameter count of the asymmetric policy has not been widened yet,
            # so the CLI does not allow it, preventing a 13012/13024 topology-mismatch time bomb.
        ],
        default=None,
        help="explicit Worker input contract. dual-v4-asymmetric-v3 keeps the first 298 columns"
             " as the V28/KING root input and appends the full v4/controller/mask context; "
             "it can be launched once from an old Worker with a fresh critic; later checkpoints "
             "only allow an explicit environment-restart parameter continuation",
    )
    ap.add_argument(
        "--worker-fast-forward-reward-credit",
        choices=["none", "terminal-death-only"],
        default="none",
        help="rev13 Worker reward contract: default none keeps the old semantics; "
             "terminal-death-only passes to PPO only the real underlying death component that happened within the same transition during the frozen manager/script"
             " phase, without collecting its positive returns",
    )
    ap.add_argument(
        "--worker-learning-window-scope",
        choices=["farm-only", "farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE],
        default="farm-only",
        help="R13 classroom reform: default farm-only reproduces the old rule bit for bit (non-FARM windows are fast-forwarded"
             " by the script); farm-dive-v1 makes DIVE windows first-class live learning windows and hands over"
             " a11/stepping autonomy inside the window; earned-dive-suffix-v1 takes over from the first ready DIVE after a fixed R16 completion prefix"
             " (both require the dual-v4-asymmetric-v3 view)",
    )
    ap.add_argument("--worker-prefix-model", default=None,
                    help="earned suffix only: exact registered R16 sampled prefix model")
    ap.add_argument("--worker-prefix-max-attempts", type=int, default=None,
                    help="explicit per-env lifetime prefix attempt cap")
    ap.add_argument("--worker-prefix-max-microsteps", type=int, default=None,
                     help="explicit per-env lifetime formal prefix microstep cap")
    ap.add_argument("--worker-time-protocol",
                    choices=["legacy", "completion-l2-v1", "completion-l2-r18c"],
                    default="legacy",
                    help="explicit physical arrival/followup and service deadlines; max_steps remains the original observation denominator")
    ap.add_argument(
        "--worker-dive-action11-logit-bonus",
        type=float,
        default=0.0,
        help="R13 cold-start lever (enabled under pre-authorisation if G0 classroom-occurrence acceptance falls short): "
             "applies an on-policy logit prior to legal a11 inside live DIVE windows, "
             "same mechanism as the a14 bonus; a11 is always masked under the old rules, so the prior is automatically inert",
    )
    ap.add_argument(
        "--worker-potion-action13-logit-bonus",
        type=float,
        default=0.0,
        help="R16 potion-pickup cold-start prior ([0,10], default 0 = off): applies an on-policy logit prior only on rows where the environment mask marks a13"
             " (pick up potion) legal, same mechanism as a11/a14"
             ", gradients flow; under the old always-masked rules the prior is automatically inert; "
             "dual-v4 Worker MaskablePPO only",
    )
    ap.add_argument(
        "--worker-depth-shaping-unit",
        type=float,
        default=0.0,
        help="R13.2 plan A (approved): potential-based depth shaping φ=unit×(dungeon_level−1), "
             "F=unit×Δd per transition + true-terminal refund (φ(absorbing)=0); "
             "enters only the worker policy_reward, never the manager ledger/exam; the wage-stripping rule is untouched",
    )
    ap.add_argument(
        "--worker-descend-bonus-fraction",
        type=float,
        default=0.0,
        help="R14 plan B (design decision, 2026-08-31): share of the descend bonus the worker keeps inside live DIVE windows"
             " ([0,1], default 0 = the old rule of stripping it all from the wage); an amendment trial -- tests"
             " whether a bounded refund revives the speedrunning behaviour; the identity R≡W+actual stripped amount still holds",
    )
    ap.add_argument(
        "--worker-descend-escrow-fraction",
        type=float,
        default=0.0,
        help="R14.2 plan D (fallback design decision): escrowed descend fee -- a descend fee for a new deepest level"
             " ×fraction goes into escrow and vests into policy_reward at the next non-death window close; "
             "death forfeits all of it (worker-side transplant of the R11 certified valve); mutually exclusive with plan B",
    )
    ap.add_argument(
        "--worker-descend-escrow-readiness-gate",
        action="store_true",
        help="R15 amendment 2 (wage-side enforcement): escrow books only descends with readiness>=1"
             " -- the mask law can force a handover, but a descend below the bar earns nothing; "
             "only with escrow_fraction>0",
    )
    ap.add_argument(
        "--worker-descend-escrow-readiness-table",
        choices=("v1", "v2"),
        default="v1",
        help="R17.0 amendment 1 (panel correction 2.1): ruler of the escrow readiness gate -- v1="
             "readiness_power_ratio (v0.1 table, old rule, default bit for bit); v2="
             "readiness_power_ratio_v2 (v0.2 table, same ruler as the readiness-v3 coach, "
             "fixes the ruler artefact of R16 vested 0/denied 6441); only with"
             " --worker-descend-escrow-readiness-gate",
    )
    ap.add_argument(
        "--worker-descend-escrow-power",
        type=float,
        default=1.0,
        help="R14.3 plan E (design decision): convex escrow wage curve, each level priced at d^power"
             " ([1,3], default 1.0 = the old linear rule). Measured risk grows geometrically (h≈0.23×"
             "1.53^(d−1)), so convexity must keep up with risk convexity (fit suggests 1.6); only takes effect with"
             " escrow_fraction>0",
    )
    ap.add_argument(
        "--worker-additional-terminal-death-cost",
        type=float,
        default=0.0,
        help="rev13 Worker survival risk cost; counted exactly once per real death, "
             "same inside FARM and at fast-forward terminals; default 0 keeps the old semantics",
    )
    # R16 amendment: new wage-side knobs (explicit WorkerWindowEnv parameters, not in env_kwargs).
    ap.add_argument(
        "--worker-hp-loss-price",
        type=float,
        default=0.0,
        help="R16 (WorkerWindowEnv): worker HP-loss pricing (policy_reward deduction per HP point "
             "lost, >=0); default 0 = old rule; --worker only",
    )
    ap.add_argument(
        "--worker-potion-pickup-bonus",
        type=float,
        default=0.0,
        help="R16 (WorkerWindowEnv): worker potion-pickup bonus (a13 successfully into the belt) (>=0); "
             "default 0 = old rule; --worker only",
    )
    ap.add_argument(
        "--worker-no-progress-timeout-credit",
        choices=list(_WORKER_NO_PROGRESS_TIMEOUT_CREDITS),
        default=_WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT,
        help="R16 (WorkerWindowEnv): stall-timeout booking definition. death-equivalent = old rule"
             " (a timeout is booked as death-equivalent); zero = no death penalty for timeouts; --worker only",
    )
    ap.add_argument(
        "--worker-action14-logit-bonus",
        type=float,
        default=0.0,
        help="trainable equipment logit prior applied only when the exact a14 mask is true; "
             "0 = off, dual-v4 Worker only",
    )
    ap.add_argument("--ent-coef", type=float, default=0.02,
                    help="entropy coefficient (v22 devil-arm fine-tuning uses 0.005 to prevent BC drift)")
    ap.add_argument("--bc-init", default=None,
                    help="behaviour-cloning warm start: path of the policy-head state_dict to load")
    ap.add_argument("--init-source", choices=["bc", "checkpoint"], default="bc",
                    help="source type of --bc-init; checkpoint requires an export_manager_sd manifest")
    ap.add_argument("--freeze-policy-steps", type=int, default=0,
                    help="steps after the BC warm start during which the policy head is frozen and only the value head trains")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="discount factor. 0.99 has a half-life of 69 steps (definition from the old 1500-step chapter); "
                         "v20 deep water uses 0.997; --options (v22) should be 1.0")
    ap.add_argument("--distill-beta", type=float, default=0.0,
                    help="v24 leash coefficient β (CE to the frozen BC teacher; 0 = pure v23 recipe, G-KL-B proves bit-for-bit equivalence)")
    ap.add_argument(
        "--distill-anneal-actor-rollouts",
        type=int,
        default=0,
        help="counting only complete rollouts after the actor unfreezes, anneal β linearly from its initial value"
             " to 0; 0 = constant, nonzero must be >=2",
    )
    ap.add_argument("--teacher-sd",
                    default=str(pathlib.Path(__file__).resolve().parent
                                / "runs" / "bc-worker" / "policy_sd.pt"),
                    help="v24 teacher state_dict (SB3 key names)")
    ap.add_argument("--bc-aux-lambda", type=float, default=0.0,
                    help="E3 4B: auxiliary calibration coefficient λ_bc (a12 positives + legal non-a12"
                         " hard negatives + persistent root policy KL; consumes only training"
                         " episodes; main-case frozen constant 0.015625, D7). "
                         "Present only together with --bc-aux-demos; the two flags do not force each other, "
                         "if either is absent -> zero intrusion (no loading, no sampling, not in the loss graph)")
    ap.add_argument("--bc-aux-demos", default=None,
                    help="E3 4B: path of the bc-worker-v2 demo set demos.npz (v2 schema="
                         "X/Y/episode_id + per-sample masks, dedicated validator; the v1 canonical"
                         " path runs/bc-worker is untouched)")
    ap.add_argument(
        "--bc-aux-graft", action="store_true",
        help="rev10: widen the V28 actor losslessly to 68 and install inside the policy distribution"
             " a contextual a12 mixture initialised at exactly 5%% per eligible state; "
             "rollout/log-prob share a source, and the five stable raw-feature gate parameters are decided by PPO"
             " autonomously from combat return; the formal path requires"
             " --bc-aux-lambda=0")
    ap.add_argument(
        "--bc-aux-liveness-preflight", action="store_true",
        help="required gate for a formal leg with aux present: before the environment launches, on an isolated clone of the resume worker/Adam, "
             "run the same number of aux calls as this leg; only training episodes judge the safe-learnability gate, "
             "and FAIL refuses training")
    ap.add_argument("--resource-warm-start", default=None,
                    help="Exact zero-step resource initialization manifest; independent of --resume-from")
    ap.add_argument("--resume-from", default=None,
                    help="v24 split-leg continuation: path of the previous leg's model_final.zip (must not be used with --bc-init/--freeze)")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="continuation safety knob: rebuild the optimizer after load, clearing all "
                         "Adam step/exp_avg/exp_avg_sq; off by default for compatibility with old recipes. "
                         "When on, --lr may be lowered explicitly; policy weights stay unchanged bit for bit")
    ap.add_argument(
        "--reset-worker-critic",
        action="store_true",
        help="rebuild the Worker value MLP/head of the old window-termination target with SB3's native"
             " orthogonal initialisation; the actor is kept bit for bit. Requires reset-optimizer, "
             "an explicit seed, critic warmup and grouped clipping",
    )
    ap.add_argument(
        "--critic-warmup-steps",
        type=int,
        default=0,
        help="number of critic-only samples for the fresh critic; must divide the full rollout "
             "quantum and be less than total-steps. The actor starts joint updates only after warmup",
    )
    ap.add_argument(
        "--gradient-clip-mode",
        choices=[
            "global",
            "separate-actor-critic-v1",
            "separate-root-context-critic-v2",
        ],
        default="global",
        help="main PPO gradient clipping; v2 clips actor root/context each to"
             " max_grad_norm/sqrt(2), and the critic separately to max_grad_norm, "
             "still using a single Adam",
    )
    ap.add_argument("--target-kl", type=float, default=None,
                    help="PPO approximate-KL early-stop threshold; default None keeps the old recipe. "
                         "For continuations, enable it explicitly together with a lower --lr and --reset-optimizer "
                         "(e.g. 0.02); the value must be a finite positive number")
    ap.add_argument("--calib-probes", default="",
                    help="v24 G-CAL probe global steps (comma-separated; pass 300000,600000 only on leg 1)")
    ap.add_argument("--calib-record-only", action="store_true",
                    help="v28: G-CAL record-only, no verdict -- the tripped bit is still written to calib.jsonl, the flag is not armed"
                         " (the continuation starts at 41.5%% divergence, where the 20%% threshold is meaningless; review-panel fix)")
    ap.add_argument("--teacher-override", default=None,
                    help="v30 anchor follows the throne: on resume, overwrite the teacher_path carried in the zip with this sd"
                         " (injected via load kwargs so _setup_model builds it right the first time; only effective on the resume branch)")
    ap.add_argument("--allow-manager-change", action="store_true",
                    help="explicitly allow a worker resume to change manager_npz; the contract forbids it by default")
    ap.add_argument("--allow-legacy-resume", action="store_true",
                    help="one-off migration of an old checkpoint without a training_contract; refused by default")
    ap.add_argument(
        "--allow-environment-restart-resume",
        action="store_true",
        help="explicitly acknowledge that a Worker checkpoint with a contract only continues policy/Adam/global counters, "
             "and does not restore the native world, wrapper/controller or full RNG/trajectory state; "
             "without this flag it fails closed",
    )
    # B1-E0 gauge knobs (a closed enumeration of three, PREREG-B1; all pure read + IO, never touching RNG/gradients/env streams/
    # masks/contract fields; defaults inherit the formerly hard-coded constants verbatim, zero drift in default behaviour, pinned by live W-G0 runs)
    ap.add_argument("--ckpt-every-steps", type=int, default=250_000,
                    help="B1-E0: expose AtomicRolloutCheckpointCallback.every_steps"
                         " (global steps; quantum alignment and refusing half-updated ckpts are guaranteed by the callback's original logic)")
    ap.add_argument("--sentinel-every", type=int, default=500_000,
                    help="B1-E0: WorkerSentinelCallback aggregation interval (global steps, pure read + IO)")
    ap.add_argument("--dry-anchor-every", type=int, default=500_000,
                    help="B1-E0: DryAnchorSentinel interval (global steps; dry states are judged by col296 "
                         "exhausted or col297 farm_scene_fraction saturation; "
                         "own rng(26), never touching the training RNG)")
    # E5 gauge knobs (PREREG-v33-content-case E5, a closed enumeration of two; all pure read + IO, record-only, no verdict,
    # default 0 = absent = code path equivalent to HEAD, G0-2a prerequisite; the probe demo set is pinned to
    # the BC-v1 demos bytes, E6)
    ap.add_argument("--distill-ce-probe-every", type=int, default=0,
                    help="E5-1: dry/fresh split distill_ce offline probe interval (global steps; "
                         "0 = absent; the fixed demo-state set is grouped by col296 exhausted or "
                         "col297 farm_scene_fraction saturation, dedicated rng, "
                         "zero contact with the training path; writes distill_ce_probe.jsonl)")
    ap.add_argument("--drywin-metrics-every", type=int, default=0,
                    help="E5-2: dry-window behaviour gauge interval (global steps; 0 = absent; dry-state action"
                         " distribution entropy/a distribution + dry/fresh window wage and width τ̄/depth, record-only to "
                         "drywin_metrics.jsonl)")
    invocation_argv = list(sys.argv[1:])
    args = ap.parse_args()

    try:
        _validate_runtime_versions()
        _validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))

    run_name = args.run_name or (
        time.strftime("ppo-l1-%m%d-%H%M%S")
        + f"-{os.getpid()}-{time.time_ns() % 1_000_000_000:09d}")
    run_dir = pathlib.Path(__file__).resolve().parent / "runs" / run_name
    try:
        run_lock = _RunLock(run_dir)
    except RuntimeError as exc:
        ap.error(str(exc))
    resources.run_lock = run_lock
    protected_inputs = [args.bc_init, args.teacher_override,
                        getattr(args, "resource_warm_start", None)]
    if getattr(args, "worker_learning_window_scope", "farm-only") == EARNED_DIVE_SUFFIX_SCOPE:
        protected_inputs.append(args.worker_prefix_model)
    if args.worker:
        protected_inputs.append(args.manager_npz)
        if args.distill_beta > 0 and not args.resume_from:
            protected_inputs.append(args.teacher_sd)
    if args.options and args.worker_npz:
        protected_inputs.append(args.worker_npz)
    if args.options and args.worker_zip:
        protected_inputs.append(args.worker_zip)
    if _bc_aux_active(args):
        protected_inputs.append(args.bc_aux_demos)   # E3: the v2 demo set is protected too
    _prepare_run_dir(run_dir, args.resume_from, protected_inputs)

    # Capture all externally supplied brains before any VecEnv/subprocess can
    # load them.  Children receive these exact expectations and parse from a
    # single read, so an atomic path replacement is fail-loud rather than a
    # silent mixed-policy run.
    manager_npz_sha256 = (
        _capture_file_sha256(args.manager_npz, "manager_npz")
        if args.worker and not getattr(args, "manager_heuristic", None)
        else None)
    worker_npz_sha256 = (_capture_file_sha256(args.worker_npz, "worker_npz")
                         if args.worker_npz else None)
    worker_zip_sha256 = (_capture_file_sha256(args.worker_zip, "worker_zip")
                         if args.worker_zip else None)
    if args.worker_zip_sha256 is not None:
        # R9: the zip identity pinned on the command line is checked before any subprocess loads it (fail-loud).
        _require(worker_zip_sha256 == args.worker_zip_sha256,
                 "--worker-zip-sha256 disagrees with the actual zip bytes: "
                 f"{worker_zip_sha256} != {args.worker_zip_sha256}")
    args.resolved_drink_sovereignty = (
        _resolve_training_drink_sovereignty(
            args, worker_npz_sha256=worker_npz_sha256,
            worker_zip_sha256=worker_zip_sha256))
    implementation_sha256 = _implementation_bundle_sha256()
    worker_prefix_identity = _worker_prefix_identity(args)

    fresh_teacher_sha256 = None
    if args.worker and args.distill_beta > 0 and not args.resume_from:
        fresh_teacher_sha256 = _validate_bc_report(
            pathlib.Path(args.teacher_sd), "data_gate")["policy_sha256"]

    teacher_override_sha256 = None
    if args.teacher_override:
        teacher_override_sha256 = _validate_export_manifest(
            pathlib.Path(args.teacher_override))["artifact_sha256"]

    # E1 four gates: demos_sha256 capture gate: skip_dry or schedule (predicate inside the helper, assertion unchanged)
    demos_sha256 = _capture_dry_window_demos_sha256(args)

    # E1 5A: the course table is parsed once here, shared by the env initial value and the course callback (already validated by _validate_args).
    dry_curriculum_table = (
        _parse_dry_curriculum_schedule(args.dry_curriculum_schedule)
        if args.dry_curriculum_schedule else None)

    # R9: the deep-start curriculum is parsed once (already validated by _validate_args); the form defaults to dive.
    deep_start_curriculum = (
        _parse_deep_start_curriculum(args.deep_start_curriculum)
        if args.deep_start_curriculum else None)
    deep_start_form = args.deep_start_form or "dive"

    # E3 4B: when present, load the v2 demo set; the auxiliary optimisation bank may only consume the fixed training
    # episodes, and the original held-out episodes are left whole for the final release hard gate. Then, inside the training split,
    # the class-12/hard-negative filter runs. Absent -> zero intrusion, nothing loaded (literal design).
    bc_aux_bank = None
    bc_aux_fit = None
    bc_aux_validation = None
    bc_aux_demos_sha256 = None
    _aux_x = _aux_y = _aux_episode_id = _aux_masks = None
    if _bc_aux_active(args):
        (_aux_x, _aux_y, _aux_episode_id, _aux_masks,
         bc_aux_demos_sha256) = (
            _load_bc_aux_demos_v2(
                args.bc_aux_demos,
                expected_manager_sha256=manager_npz_sha256))
        if _bc_aux_structural_active(args):
            _fit, _validation = _bc_v2_fit_validation_indices(
                _aux_episode_id)
            bc_aux_fit = (
                _aux_x[_fit], _aux_y[_fit], _aux_masks[_fit])
            # Keep a non-None marker for the shared active-path setup below;
            # structural code never consumes this as a replay minibatch.
            bc_aux_bank = bc_aux_fit
            bc_aux_validation = (
                _aux_x[_validation], _aux_y[_validation],
                _aux_masks[_validation])
        else:
            bc_aux_bank = _build_bc_aux_training_bank(
                _aux_x, _aux_y, _aux_episode_id, _aux_masks)
    elif args.bc_aux_lambda > 0:
        # Literal design: without --bc-aux-demos there is zero intrusion (the two flags do not force each other);
        # print it honestly so a misconfiguration is not silent (implementation discretion note).
        print("   [4b] --bc-aux-lambda>0 but no --bc-aux-demos given: "
              "the auxiliary pathway is absent under the E3 zero-intrusion clause")

    resource_warm_start_payload = None
    resource_warm_start_manifest = None
    if args.resource_warm_start:
        from migrate_resource_candidate import capture_initialization
        (resource_warm_start_payload, _resource_warm_start_data,
         resource_warm_start_manifest) = capture_initialization(
            args.resource_warm_start, implementation_sha256)

    resume_checkpoint_bytes = None
    resume_data = None
    resume_checkpoint_sha256 = None
    if args.resume_from and args.worker:
        (resume_checkpoint_bytes, resume_data,
         resume_checkpoint_sha256) = _capture_leashed_checkpoint(args.resume_from)
    elif args.resume_from:
        # v31 manager continuation entry: generic capture (byte freeze + generic gate + sha); class fidelity is left to the loading section
        _resume_path = _checkpoint_path(args.resume_from)
        try:
            resume_checkpoint_bytes = _resume_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"resume checkpoint unreadable: {_resume_path}: {exc}") from exc
        resume_data = _validate_checkpoint_bytes(
            resume_checkpoint_bytes, str(_resume_path), False)
        resume_checkpoint_sha256 = hashlib.sha256(
            resume_checkpoint_bytes).hexdigest()

    if args.resume_from:
        # Check the captured checkpoint before any VecEnv/native reset. The
        # full contract equality remains enforced again after model loading.
        _validate_worker_prefix_resume_identity(resume_data.get("diablogym_contract"), {
            "worker_learning_window_scope": args.worker_learning_window_scope,
            **({"worker_prefix": worker_prefix_identity} if worker_prefix_identity is not None else {}),
        })
        _validate_resource_resume_identity(resume_data.get("diablogym_contract"), {
            "resource_protocol": (args.resource_protocol if args.resource_protocol != "off" else None),
            "resource_purchase_mode": (args.resource_purchase_mode if args.resource_protocol != "off" else None),
            "resource_service_policy": (args.resource_service_policy if args.resource_service_policy != "legacy-v1" else None),
            "resource_service_recipe": _resource_service_recipe_for(
                args.resource_protocol, args.resource_purchase_mode, args.resource_service_policy,
                getattr(args, "worker_time_protocol", "legacy")),
            # R18-B3: the clock is carried only to recompute the expected loot recipe; it does not join the
            # resource identity key group (the clock has its own _validate_worker_time_resume_identity).
            "worker_time_protocol": (getattr(args, "worker_time_protocol", "legacy")
                                     if getattr(args, "worker_time_protocol", "legacy") != "legacy"
                                     else None),
            # R18-B: same None-off form as _training_contract.
            "resource_retreat": (getattr(args, "resource_retreat", "off")
                                 if getattr(args, "resource_retreat", "off") != "off" else None),
            # R18-B5: same None-off form as _training_contract.
            "resource_portal": (getattr(args, "resource_portal", "off")
                                if getattr(args, "resource_portal", "off") != "off" else None),
            # R18-B6: same None-off form as _training_contract.
            "resource_sweep": (getattr(args, "resource_sweep", "off")
                               if getattr(args, "resource_sweep", "off") != "off" else None),
            "resource_identify": (getattr(args, "resource_identify", "off")
                                  if getattr(args, "resource_identify", "off") != "off" else None),
            "resource_weapon_upgrade": (
                getattr(args, "resource_weapon_upgrade", "off")
                if getattr(args, "resource_weapon_upgrade", "off") != "off" else None),
        })

    dry_curriculum_start_index = None
    dry_curriculum_start_probability = None
    if dry_curriculum_table:
        dry_curriculum_start_index, dry_curriculum_start_probability = (
            _resolve_dry_curriculum_start(
                dry_curriculum_table,
                start_steps=(
                    int(resume_data["num_timesteps"])
                    if resume_data is not None else 0
                ),
                rollout_quantum=int(args.n_steps * args.num_envs),
                total_steps=int(args.total_steps),
            )
        )

    batch_size = _select_batch_size(args.n_steps, args.num_envs)
    bc_aux_preflight = None
    bc_aux_preflight_sha256 = None
    if args.bc_aux_liveness_preflight:
        # Production aux is a continuation recipe; only the resume zip carries both the real starting
        # worker and the Adam moments. Without it an isomorphic liveness sandbox cannot run before the environment launches,
        # so fail closed rather than let a random new model pose as the preflight.
        _require(resume_checkpoint_bytes is not None
                 and resume_checkpoint_sha256 is not None,
                 "training with bc_aux present requires --resume-from, "
                 "so the real worker/optimizer liveness preflight can run before the environment launches")
        (bc_aux_preflight,
         bc_aux_preflight_sha256) = _run_bc_aux_liveness_preflight(
            run_dir=run_dir, args=args,
            resume_checkpoint_bytes=resume_checkpoint_bytes,
            resume_checkpoint_sha256=resume_checkpoint_sha256,
            bank=bc_aux_bank,
            x=_aux_x, y=_aux_y, episode_id=_aux_episode_id,
            masks=_aux_masks,
            demos_sha256=bc_aux_demos_sha256,
            manager_npz_sha256=manager_npz_sha256,
            implementation_sha256=implementation_sha256,
            batch_size=batch_size)

    hierarchical = args.worker or args.options or args.flat_clock
    effective_deep = True if hierarchical else args.deep
    effective_death_ladder = True if hierarchical else args.death_ladder
    config = {
        # Audit receipt: keep exactly the argv this process gave argparse; not part of the training contract.
        "invocation_argv": invocation_argv,
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "device": args.device,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "algo": ({"rppo": "RecurrentPPO/MlpLstmPolicy",
                  "mppo": "MaskablePPO/MlpPolicy(gear-key mask)"}.get(args.algo, "PPO/MlpPolicy")
                 + ("+EntityAttention" if args.arch == "attn" else "")),
        "goal": ("deep water: depth-scaled bonus, dive down alive (L3/L4)" if effective_deep
                 else "dungeon level 1: kill monsters for XP, find the stairs to level 2"),
        "deep": effective_deep,
        "death_ladder": effective_death_ladder,
        "gamma": args.gamma,
        "options": args.options,      # v22: when True, the Monitor ep_len definition = number of policy-brain decisions
        "flat_clock": args.flat_clock,
        "worker": args.worker,        # v4: when True, ep = one whole underlying game; FARM window boundaries are nonterminal
        "skip_dry": args.skip_dry,
        "drink_sovereignty":
            _effective_drink_sovereignty(args),   # resolved action12 contract
        "legacy_policy_observation_view":
            args.legacy_worker_policy_observation_view,
        "worker_policy_observation_view":
            _worker_policy_observation_view(args),
        "worker_action14_logit_bonus": float(getattr(
            args, "worker_action14_logit_bonus", 0.0)),
        "manager_policy_observation_view": (
            "legacy-v3" if args.worker
            else args.manager_policy_observation_view
            if args.options else None),
        "worker_fast_forward_reward_credit":
            args.worker_fast_forward_reward_credit,
        "worker_additional_terminal_death_cost":
            float(args.worker_additional_terminal_death_cost),
        "artifact_scope": args.artifact_scope,
        "diagnostic_rollout": args.diagnostic_rollout,
        # E4 rev5 dual keys (keys added isomorphically to the contract and the config receipt; the skip_dry key is still the CLI flag
        # literal, and whether the mechanism is present is carried by these two keys, rev3 correction)
        "dry_curriculum": _contract_dry_curriculum(args),
        "dry_curriculum_start_index": dry_curriculum_start_index,
        "dry_curriculum_start_probability":
            dry_curriculum_start_probability,
        "bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),
        "bc_aux_liveness_preflight": (
            {
                "status": bc_aux_preflight["status"],
                "receipt_sha256": bc_aux_preflight_sha256,
                "schema_version":
                    _BC_AUX_LIVENESS_PREFLIGHT_SCHEMA_VERSION,
            } if bc_aux_preflight is not None else "disabled"),
        "calib_probes_requested": args.calib_probes,

        "bc_init": args.bc_init,
        "init_source": args.init_source,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl,
        "reset_optimizer": args.reset_optimizer,
        "reset_worker_critic": args.reset_worker_critic,
        "critic_warmup_steps": args.critic_warmup_steps,
        "gradient_clip_mode": args.gradient_clip_mode,
        "freeze_policy_steps": args.freeze_policy_steps,
        "distill_beta": args.distill_beta,    # v24 leash
        "distill_anneal_actor_rollouts": int(getattr(
            args, "distill_anneal_actor_rollouts", 0)),
        "resume_from": args.resume_from,
        "resource_warm_start": args.resource_warm_start,
        "resource_warm_start_receipt": (resource_warm_start_manifest["receipt"]
            if resource_warm_start_manifest else None),
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "worker_npz": args.worker_npz,        # v25 re-election: manager training mounts an npz worker
        # R9 re-election: manager training mounts a certified SB3 zip worker (mutually exclusive with the npz assembly point)
        "worker_zip": args.worker_zip,
        "worker_zip_sha16": (worker_zip_sha256[:16]
                             if worker_zip_sha256 else None),
        # R9 curriculum arm receipt (CLI literals + parsed effective values; not in training_contract,
        # the full CLI is already recorded by invocation_argv)
        "deep_start_curriculum": args.deep_start_curriculum,
        "deep_start_curriculum_resolved": deep_start_curriculum,
        "deep_start_form": (deep_start_form
                            if deep_start_curriculum else None),
        # v30 relay: self-evidence chain -- which incumbent this leg ran under and which anchor it is tied to, receipted on the process side (review-panel minor finding)
        "manager_npz": args.manager_npz,
        "manager_npz_sha16": (manager_npz_sha256[:16]
                               if manager_npz_sha256 else None),
        "teacher_override": args.teacher_override,
        "teacher_override_sha16": (teacher_override_sha256[:16]
                                    if teacher_override_sha256 else None),
        "demos_sha16": demos_sha256[:16] if demos_sha256 else None,
        "implementation_sha16": implementation_sha256[:16],
        "allow_manager_change": args.allow_manager_change,
        "allow_legacy_resume": args.allow_legacy_resume,
        "allow_environment_restart_resume":
            args.allow_environment_restart_resume,
        # B1-E0 gauge knob receipt (read-only telemetry, not in training_contract, contract untouched)
        "ckpt_every_steps": args.ckpt_every_steps,
        "sentinel_every": args.sentinel_every,
        "dry_anchor_every": args.dry_anchor_every,
        # E5 gauge knob receipt (as above: read-only telemetry, not in training_contract)
        "distill_ce_probe_every": args.distill_ce_probe_every,
        "drywin_metrics_every": args.drywin_metrics_every,
    }
    prefix_env_kwargs = {}
    time_identity = _worker_time_identity(args.worker_time_protocol)
    if time_identity is not None:
        config["worker_time_protocol"] = time_identity["protocol"]
        config["worker_time_recipe"] = time_identity
        prefix_env_kwargs["worker_time_protocol"] = time_identity["protocol"]
    if worker_prefix_identity is not None:
        config["worker_prefix"] = worker_prefix_identity
        config["worker_prefix_model"] = str(args.worker_prefix_model)
        prefix_env_kwargs.update({
            "worker_prefix_model": args.worker_prefix_model,
            "worker_prefix_sha256": worker_prefix_identity["source_sha256"],
            "worker_prefix_max_attempts": worker_prefix_identity["max_attempts"],
            "worker_prefix_max_microsteps": worker_prefix_identity["max_microsteps"],
        })
    if args.dive_blocker_recovery != "off":
        config["dive_blocker_recovery_recipe"] = dive_blocker_recovery_recipe(args.dive_blocker_recovery)
    print(f"== DiabloGym PPO training == run={run_name}")
    print(f"   {config}")

    env_fn = functools.partial(
        make_env, **prefix_env_kwargs,
        max_steps=args.max_steps,
        deep=args.deep,
        death_ladder=args.death_ladder,
        options=args.options,
        flat_clock=args.flat_clock,
        worker=args.worker,
        manager_npz=args.manager_npz,
        worker_npz=args.worker_npz,
        # SB3 _setup_learn's env.reset() happens before callback training_start.
        # A first migration starts from the table head; a native continuation must start from the entry matching the checkpoint,
        # and must not pick a window by table[0] first and then only change p in the observation.
        skip_dry=(dry_curriculum_start_probability
                  if dry_curriculum_table else args.skip_dry),
        drink_sovereignty=_effective_drink_sovereignty(args),
        legacy_policy_observation_view=(
            args.legacy_worker_policy_observation_view),
        worker_policy_observation_view=(
            _worker_policy_observation_view(args)),
        manager_policy_observation_view=(
            args.manager_policy_observation_view),
        worker_fast_forward_reward_credit=(
            args.worker_fast_forward_reward_credit),
        worker_additional_terminal_death_cost=(
            args.worker_additional_terminal_death_cost),
        worker_learning_window_scope=getattr(
            args, "worker_learning_window_scope", "farm-only"),
        worker_depth_shaping_unit=float(getattr(
            args, "worker_depth_shaping_unit", 0.0)),
        worker_descend_bonus_fraction=float(getattr(
            args, "worker_descend_bonus_fraction", 0.0)),
        worker_descend_escrow_fraction=float(getattr(
            args, "worker_descend_escrow_fraction", 0.0)),
        worker_descend_escrow_power=float(getattr(
            args, "worker_descend_escrow_power", 1.0)),
        worker_descend_escrow_readiness_gate=bool(getattr(
            args, "worker_descend_escrow_readiness_gate", False)),
        worker_descend_escrow_readiness_table=str(getattr(
            args, "worker_descend_escrow_readiness_table", "v1")),
        # R16 amendment: explicit wage-side parameters + pass-through environment/classroom parameters (already validated by _validate_args)
        worker_hp_loss_price=float(getattr(
            args, "worker_hp_loss_price", 0.0)),
        worker_potion_pickup_bonus=float(getattr(
            args, "worker_potion_pickup_bonus", 0.0)),
        worker_no_progress_timeout_credit=getattr(
            args, "worker_no_progress_timeout_credit",
            _WORKER_NO_PROGRESS_TIMEOUT_CREDIT_DEFAULT),
        explore_global_fallback=bool(getattr(
            args, "explore_global_fallback", False)),
        explore_global_hunt=bool(getattr(
            args, "explore_global_hunt", False)),
        progress_far_tiles=int(getattr(args, "progress_far_tiles", 0)),
        farm_scene_cap=int(getattr(
            args, "farm_scene_cap", _FARM_SCENE_CAP_DEFAULT)),
        reset_layer_clock_on_window=bool(getattr(
            args, "reset_layer_clock_on_window", False)),
        resource_protocol=getattr(args, "resource_protocol", "off"),
        resource_purchase_mode=getattr(args, "resource_purchase_mode", "full"),
        resource_service_policy=getattr(args, "resource_service_policy", "legacy-v1"),
        resource_readiness_law=getattr(args, "resource_readiness_law", "veto-v1"),
        resource_retreat=getattr(args, "resource_retreat", "off"),
        # R18-B5 (2026-09-07): portal vehicle and hunt scope pass through (defaults off/all do not change
        # any make_env construction call).
        resource_portal=getattr(args, "resource_portal", "off"),
        # R18-B6 (2026-09-07): sweep/identify/weapon upgrade pass through (default off does not change
        # any make_env construction call).
        resource_sweep=getattr(args, "resource_sweep", "off"),
        resource_identify=getattr(args, "resource_identify", "off"),
        resource_weapon_upgrade=getattr(args, "resource_weapon_upgrade", "off"),
        hunt_scope=getattr(args, "hunt_scope", "all"),
        dive_blocker_recovery=getattr(args, "dive_blocker_recovery", "off"),
        manager_npz_sha256=manager_npz_sha256,
        worker_npz_sha256=worker_npz_sha256,
        worker_zip=args.worker_zip,
        worker_zip_sha256=worker_zip_sha256,
        deep_start_curriculum=deep_start_curriculum,
        deep_start_form=deep_start_form,
        implementation_sha256=implementation_sha256,
        reward_economy=args.reward_economy,
        manager_heuristic=args.manager_heuristic,
    )
    if args.num_envs == 1:
        vec_env = DummyVecEnv([env_fn])
    else:
        vec_env = SubprocVecEnv([env_fn] * args.num_envs, start_method="spawn")
    resources.vec_env = vec_env

    common = dict(
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,  # default 0.02 (the first run's 0.01 once collapsed into wall-hugging); v22 devil arm 0.005
        gae_lambda=_ALGORITHM_RECIPE["gae_lambda"],
        n_epochs=_ALGORITHM_RECIPE["n_epochs"],
        clip_range=_ALGORITHM_RECIPE["clip_range"],
        clip_range_vf=_ALGORITHM_RECIPE["clip_range_vf"],
        vf_coef=_ALGORITHM_RECIPE["vf_coef"],
        max_grad_norm=_ALGORITHM_RECIPE["max_grad_norm"],
        normalize_advantage=_ALGORITHM_RECIPE["normalize_advantage"],
        target_kl=args.target_kl,
        device=args.device,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
    )
    bc_aux_circuit_calibration = None
    actor_migration_receipt = None
    critic_migration_receipt = None
    dual_resume_kind = None
    resume_lineage = None
    policy_kwargs = {}
    if (
        args.worker
        and _worker_policy_observation_view(args)
        == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
    ):
        policy_kwargs["action14_logit_bonus"] = float(
            args.worker_action14_logit_bonus)
        if float(getattr(
                args, "worker_dive_action11_logit_bonus", 0.0)) > 0.0:
            policy_kwargs["action11_logit_bonus"] = float(
                args.worker_dive_action11_logit_bonus)
        # R16: the a13 potion-pickup prior follows the a11 precedent -- only nonzero values enter policy_kwargs (the default path
        # is unchanged bit for bit).
        if float(getattr(
                args, "worker_potion_action13_logit_bonus", 0.0)) > 0.0:
            policy_kwargs["action13_logit_bonus"] = float(
                args.worker_potion_action13_logit_bonus)
    if args.arch == "attn":
        from models import EntityAttentionExtractor
        policy_kwargs = dict(
            features_extractor_class=EntityAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128], vf=[128]),
        )
    if args.algo == "rppo":
        model = RecurrentPPO(
            "MlpLstmPolicy", vec_env,
            n_steps=args.n_steps, batch_size=batch_size,
            policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, **policy_kwargs),
            use_sde=_ALGORITHM_RECIPE["use_sde"],
            sde_sample_freq=_ALGORITHM_RECIPE["sde_sample_freq"],
            **common,
        )
    elif args.algo == "mppo":
        # v16: masked sampling and masked updates are both handled by MaskablePPO; the mask itself comes from
        # env.action_masks() (collected via VecEnv.env_method). Note this replaces the whole algorithm
        # implementation, so it is the prime suspect when a run misbehaves (recorded in the honest ledger).
        # v24: the worker path always uses LeashedMaskablePPO (with β=0, G-KL-B proves bit-for-bit equivalence with the original)
        calib = [int(x) for x in args.calib_probes.split(",") if x.strip()]
        if args.resource_warm_start:
            from migrate_resource_candidate import load_initialization
            _require(resource_warm_start_payload is not None
                     and resource_warm_start_manifest is not None,
                     "resource warm-start capture is missing")
            model = load_initialization(resource_warm_start_payload,
                resource_warm_start_manifest, env=vec_env, seed=args.seed)
            model.tensorboard_log = str(run_dir / "tb")
            actor_migration_receipt = dict(model._actor_migration_receipt)
            critic_migration_receipt = dict(model._critic_migration_receipt)
            print("   [resource warm-start] preserved parent weights; new-world steps=0, empty Adam")
        elif args.resume_from and args.options:
            # v31 manager continuation entry: class fidelity (continue with whatever class was saved; M29 is a plain MaskablePPO;
            # no teacher/β involved, no G-KL-B obligation); seal assertions unchanged from v24.
            from sb3_contrib import MaskablePPO
            _require(resume_checkpoint_bytes is not None and resume_data is not None,
                     "resume checkpoint capture state missing")
            _load_kw = {"seed": args.seed} if args.seed is not None else {}
            model = MaskablePPO.load(io.BytesIO(resume_checkpoint_bytes), env=vec_env,
                                     device=args.device, **_load_kw)
            model.tensorboard_log = str(run_dir / "tb")
            saved_lr = model.learning_rate
            if args.reset_optimizer:
                _reset_policy_optimizer(model, args.lr)
            else:
                _require(not callable(saved_lr)
                         and math.isclose(float(saved_lr), args.lr,
                                          rel_tol=0, abs_tol=1e-12),
                         f"resume learning rate mismatch: checkpoint={saved_lr}, CLI={args.lr}; "
                         "to lower lr safely, pass --reset-optimizer explicitly")
            model.target_kl = args.target_kl
            _require(math.isclose(float(model.ent_coef), args.ent_coef,
                                  rel_tol=0, abs_tol=1e-12)
                     and math.isclose(float(model.gamma), args.gamma,
                                      rel_tol=0, abs_tol=1e-12)
                     and model.n_steps == args.n_steps
                     and model.batch_size == batch_size,
                     "PREREG-v24 seal-5: resume leg hyperparameters do not match the frozen recipe")
            _require(model.target_kl == args.target_kl,
                     "resume target_kl explicit override failed")
            if args.seed is not None:
                model.set_random_seed(args.seed)
                model.seed = args.seed  # set_random_seed does not update the persisted attribute; keeps the zip from carrying the old seed forward
            print(f"   [v31] options resume @ {model.num_timesteps} steps (manager continuation entry)")
        elif args.resume_from:
            from leashed_ppo import (
                AsymmetricWorkerMaskableActorCriticPolicy,
                LeashedMaskablePPO,
            )
            _require(resume_checkpoint_bytes is not None and resume_data is not None,
                     "resume checkpoint capture state missing")
            _load_kw = {"seed": args.seed} if args.seed is not None else {}
            if args.distill_beta == 0:
                # β=0 consumes no teacher; an absolute path in an old zip must not invalidate a usable checkpoint after a move.
                _load_kw.update(teacher_path=None, teacher_sha256=None)
            elif args.teacher_override:
                # v30 anchor follows the throne: kwargs take effect after the zip data and before _setup_model,
                # so the teacher is built right the first time (a post-load rebuild is second best; dropped after a major review-panel finding)
                _load_kw["teacher_path"] = args.teacher_override
                _load_kw["teacher_sha256"] = teacher_override_sha256
            else:
                saved_teacher = resume_data.get("teacher_path")
                _require(isinstance(saved_teacher, str) and saved_teacher,
                         "β>0 resume checkpoint has no teacher_path; pass --teacher-override explicitly")
                teacher_report = _validate_bc_report(
                    pathlib.Path(saved_teacher), "data_gate")
                current_teacher_sha = teacher_report["policy_sha256"]
                saved_teacher_sha = resume_data.get("teacher_sha256")
                if saved_teacher_sha is not None:
                    _require(saved_teacher_sha == current_teacher_sha,
                             "checkpoint teacher SHA disagrees with the current BC report; "
                             "changing the anchor requires an explicit --teacher-override")
                    _load_kw["teacher_sha256"] = saved_teacher_sha
                else:
                    # Old checkpoints have no SHA field: only a single TOFU migration with the current PASS + bound report is allowed.
                    _load_kw["teacher_sha256"] = current_teacher_sha
                _load_kw["teacher_path"] = saved_teacher
            dual_actor_migration = (
                _worker_policy_observation_view(args)
                == _WORKER_VIEW_DUAL_V4_ASYMMETRIC
            )
            dual_resume_kind = (
                _classify_dual_worker_resume(args, resume_data)
                if dual_actor_migration else None
            )
            if dual_resume_kind == _DUAL_LEGACY_ACTOR_MIGRATION:
                # A 298-wide V28 checkpoint cannot be loaded against a
                # wider environment.  Construct the registered asymmetric
                # topology from scratch, then transplant only the six actor
                # tensors.  Critic, optimizer and all algorithm-side counters
                # deliberately start fresh while the absolute lineage step is
                # inherited from the immutable source checkpoint.
                source_steps = resume_data.get("num_timesteps")
                _require(
                    _is_plain_int(source_steps) and source_steps >= 0,
                    "dual-v4 actor migration source num_timesteps invalid",
                )
                model = LeashedMaskablePPO(
                    AsymmetricWorkerMaskableActorCriticPolicy,
                    vec_env,
                    n_steps=args.n_steps,
                    batch_size=batch_size,
                    policy_kwargs=policy_kwargs or None,
                    distill_beta=args.distill_beta,
                    distill_anneal_actor_rollouts=int(getattr(
                        args, "distill_anneal_actor_rollouts", 0)),
                    teacher_path=_load_kw.get("teacher_path"),
                    teacher_sha256=_load_kw.get("teacher_sha256"),
                    calib_probes=calib,
                    calib_out=(
                        str(run_dir / "calib.jsonl") if calib else None),
                    **common,
                )
                model.num_timesteps = int(source_steps)
                actor_migration_receipt = (
                    _initialize_asymmetric_worker_actor(
                        model,
                        source_checkpoint_payload=resume_checkpoint_bytes,
                        source_checkpoint_sha256=resume_checkpoint_sha256,
                    )
                )
            else:
                _require(
                    not dual_actor_migration
                    or dual_resume_kind == _DUAL_ENV_RESTART_CONTINUATION,
                    "dual-v4 resume classification not closed",
                )
                model = LeashedMaskablePPO.load(
                    io.BytesIO(resume_checkpoint_bytes),
                    env=vec_env,
                    device=args.device,
                    **_load_kw,
                )
                if dual_resume_kind == _DUAL_ENV_RESTART_CONTINUATION:
                    actor_migration_receipt = getattr(
                        model, "_actor_migration_receipt", None)
                    critic_migration_receipt = getattr(
                        model, "_critic_migration_receipt", None)
                    _require(
                        isinstance(actor_migration_receipt, dict)
                        and isinstance(critic_migration_receipt, dict),
                        "dual-v4 continuation checkpoint "
                        "is missing complete migration receipts",
                    )
                _a11_bonus = float(getattr(
                    args, "worker_dive_action11_logit_bonus", 0.0))
                if _a11_bonus > 0.0:
                    # R13 cold-start lever: old zips lack this kwarg, so write it twice explicitly after load --
                    # the live attribute (effective in this leg's forward) and policy_kwargs (baked into the output
                    # zip at save, so exam loading uses the same prior and the lever does not evaporate in the exam room).
                    _require(
                        hasattr(model.policy, "action11_logit_bonus"),
                        "policy is missing action11_logit_bonus (leashed version too old)")
                    model.policy.action11_logit_bonus = _a11_bonus
                    model.policy_kwargs["action11_logit_bonus"] = _a11_bonus
                _a13_bonus = float(getattr(
                    args, "worker_potion_action13_logit_bonus", 0.0))
                if _a13_bonus > 0.0:
                    # R16 potion-pickup prior: old zips lack this kwarg, so write it twice explicitly after load (a11
                    # precedent) -- the live attribute (effective in this leg's forward) and policy_kwargs
                    # (baked into the output zip at save, so exam loading uses the same prior).
                    _require(
                        hasattr(model.policy, "action13_logit_bonus"),
                        "policy is missing action13_logit_bonus (leashed version too old)")
                    model.policy.action13_logit_bonus = _a13_bonus
                    model.policy_kwargs["action13_logit_bonus"] = _a13_bonus
            if getattr(model, "teacher_path", None) and not args.teacher_override:
                _validate_bc_report(pathlib.Path(model.teacher_path), "data_gate")
            # PREREG-v24 D4: explicit β override (load writes __dict__ directly without checks, so no silent carry-over);
            # same for the tb path (otherwise the curves of legs 2-8 all land in leg 1's directory); knob seal assertions.
            _require(hasattr(model, "distill_beta"),
                     "LeashedMaskablePPO.load left the distill_beta internal attribute missing")
            model.distill_beta = args.distill_beta
            model.distill_anneal_actor_rollouts = int(getattr(
                args, "distill_anneal_actor_rollouts", 0))
            _require(
                isinstance(
                    getattr(
                        model,
                        "_distill_actor_rollouts_completed",
                        None),
                    int,
                )
                and not isinstance(
                    model._distill_actor_rollouts_completed, bool)
                and model._distill_actor_rollouts_completed >= 0,
                "resume checkpoint distill actor rollout count invalid",
            )
            model.calib_probes, model.calib_out = calib, (
                str(run_dir / "calib.jsonl") if calib else None)
            model.calib_record_only = args.calib_record_only
            model.tensorboard_log = str(run_dir / "tb")
            bc_aux_adapter_existing = False
            if _bc_aux_structural_active(args):
                bc_aux_adapter_existing = (
                    getattr(model, "_bc_aux_circuit_spec", None) is not None)
                _expand_policy_with_bc_aux_circuit(model)
            saved_lr = model.learning_rate
            if args.reset_worker_critic:
                _require(
                    resume_data.get("diablogym_contract") is None,
                    "a Worker that already has a full-game training_contract "
                    "must not rebuild the critic again",
                )
                _require(
                    not bc_aux_adapter_existing
                    and not _bc_aux_structural_active(args),
                    "fresh critic migration must not rewrite the topology together with the A12 circuit",
                )
                source_actor_sha256 = (
                    actor_migration_receipt["source_actor_sha256"]
                    if actor_migration_receipt is not None else None
                )
                source_critic_sha256 = (
                    actor_migration_receipt["source_critic_sha256"]
                    if actor_migration_receipt is not None else None
                )
                critic_migration_receipt = _reset_worker_critic(
                    model,
                    training_seed=args.seed,
                    source_checkpoint_sha256=resume_checkpoint_sha256,
                    source_actor_sha256=source_actor_sha256,
                    source_critic_sha256=source_critic_sha256,
                )
            if args.reset_optimizer:
                _require(not bc_aux_adapter_existing,
                         "an existing a12 mixture must not clear the optimizer")
                _reset_policy_optimizer(model, args.lr)
            else:
                _require(not callable(saved_lr)
                         and math.isclose(float(saved_lr), args.lr,
                                          rel_tol=0, abs_tol=1e-12),
                         f"resume learning rate mismatch: checkpoint={saved_lr}, CLI={args.lr}; "
                         "to lower lr safely, pass --reset-optimizer explicitly")
            if args.reset_worker_critic:
                migration = model.configure_critic_migration(
                    gradient_clip_mode=args.gradient_clip_mode,
                    critic_warmup_steps=args.critic_warmup_steps,
                )
                critic_migration_receipt.update({
                    **migration,
                    "optimizer_reset": True,
                })
                if actor_migration_receipt is not None:
                    _require(
                        critic_migration_receipt["source_actor_sha256"]
                        == actor_migration_receipt["source_actor_sha256"]
                        and critic_migration_receipt[
                            "source_critic_sha256"]
                        == actor_migration_receipt["source_critic_sha256"]
                        and critic_migration_receipt[
                            "actor_sha256_before"]
                        == actor_migration_receipt[
                            "migrated_actor_sha256"]
                        and critic_migration_receipt[
                            "actor_sha256_after"]
                        == actor_migration_receipt[
                            "migrated_actor_sha256"]
                        and critic_migration_receipt["actor_sha256"]
                        == actor_migration_receipt[
                            "migrated_actor_sha256"],
                        "asymmetric actor and critic migration receipts not closed",
                    )
                model._critic_migration_receipt = dict(
                    critic_migration_receipt)
            model.target_kl = args.target_kl
            _require(math.isclose(float(model.ent_coef), args.ent_coef,
                                  rel_tol=0, abs_tol=1e-12)
                     and math.isclose(float(model.gamma), args.gamma,
                                      rel_tol=0, abs_tol=1e-12)
                     and model.n_steps == args.n_steps
                     and model.batch_size == batch_size,
                     "PREREG-v24 seal-5: resume leg hyperparameters do not match the frozen recipe")
            _require(model.target_kl == args.target_kl,
                     "resume target_kl explicit override failed")
            if args.teacher_override:
                # v30 identity-chain assertion (review-panel blocker: the file that passed the gate and the file training consumed must be the same)
                _require(model.teacher_path == args.teacher_override, "teacher override did not take effect")
                _require(model.teacher[0].in_features == 298
                         and model.teacher[-1].out_features == 15,
                         "self-anchored teacher shape invalid (must be the 298->15 worker net)")
            if args.distill_beta > 0:
                _require(model.teacher is not None, "β>0 but the teacher was not rebuilt from teacher_path")
            if _bc_aux_structural_active(args):
                if bc_aux_adapter_existing:
                    columns = list(
                        _BC_AUX_CIRCUIT_GATE_PARAMETER_COLUMNS)
                    bc_aux_circuit_calibration = {
                        "initializer": "preserved-continuation",
                        "gate_coefficients": [
                            float(value) for value in
                            model.policy.action_net.weight[
                                _BC_AUX_CIRCUIT_ACTION,
                                columns,
                            ].detach().cpu()
                        ],
                        "gate_bias": float(
                            model.policy.action_net.bias[
                                _BC_AUX_CIRCUIT_ACTION].detach().cpu()),
                        "candidate_policy_head_sha256":
                            _policy_head_sha256(
                                _policy_head_snapshot(model.policy)),
                    }
                else:
                    bc_aux_circuit_calibration = (
                        _calibrate_bc_aux_adapter_weight(
                            model, _aux_x, _aux_y,
                            _aux_episode_id, _aux_masks))
                _require(
                    bc_aux_preflight is not None
                    and bc_aux_circuit_calibration[
                        "candidate_policy_head_sha256"]
                    == bc_aux_preflight["policy"][
                        "grafted_head_sha256"],
                    "formal a12 circuit is not bit-identical to the isolated preflight weights")
            if args.seed is not None:
                model.set_random_seed(args.seed)
                model.seed = args.seed  # set_random_seed does not update the persisted attribute; keeps the zip from writing leg 1's seed
            print(f"   [v24] resume @ {model.num_timesteps} steps, β={model.distill_beta}")
        elif args.worker:
            from leashed_ppo import LeashedMaskablePPO
            model = LeashedMaskablePPO(
                "MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=batch_size,
                policy_kwargs=policy_kwargs or None,
                distill_beta=args.distill_beta,
                distill_anneal_actor_rollouts=int(getattr(
                    args, "distill_anneal_actor_rollouts", 0)),
                teacher_path=args.teacher_sd if args.distill_beta > 0 else None,
                teacher_sha256=fresh_teacher_sha256,
                calib_probes=calib,
                calib_out=str(run_dir / "calib.jsonl") if calib else None,
                **common)
            model.calib_record_only = args.calib_record_only
        else:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO("MlpPolicy", vec_env, n_steps=args.n_steps,
                                batch_size=batch_size,
                                policy_kwargs=policy_kwargs or None, **common)
    else:
        model = PPO("MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=batch_size,
                    policy_kwargs=policy_kwargs or None,
                    use_sde=_ALGORITHM_RECIPE["use_sde"],
                    sde_sample_freq=_ALGORITHM_RECIPE["sde_sample_freq"],
                    **common)

    if dual_resume_kind is not None:
        _require(
            resume_data is not None
            and resume_checkpoint_sha256 is not None,
            "dual Worker resume lineage is missing the parent payload identity",
        )
        resume_lineage = _build_resume_lineage(
            resume_data,
            parent_sha256=resume_checkpoint_sha256,
            operation=dual_resume_kind,
            seed=args.seed,
            optimizer_reset=bool(args.reset_optimizer),
            critic_reset=bool(args.reset_worker_critic),
        )
        model._resume_lineage = dict(resume_lineage)
    config["resume_lineage"] = (
        dict(resume_lineage) if resume_lineage is not None else None)

    if args.worker and args.algo == "mppo":
        actual_clip_mode = getattr(model, "gradient_clip_mode", "global")
        _require(
            actual_clip_mode == args.gradient_clip_mode,
            "CLI gradient clip mode disagrees with the actual Leashed model: "
            f"{args.gradient_clip_mode!r} != {actual_clip_mode!r}",
        )
    config["critic_migration_receipt"] = (
        dict(critic_migration_receipt)
        if critic_migration_receipt is not None else None)
    config["actor_migration_receipt"] = (
        dict(actor_migration_receipt)
        if actor_migration_receipt is not None else None)

    # E3 4B: only λ is overridden explicitly here; the demo bank must be mounted after BC init/continuation
    # are fully in place, so that "the real training starting policy" is frozen as the anchor. The old order wrongly anchored
    # the random initialisation under a fresh --bc-init.
    if bc_aux_bank is not None:
        _require(hasattr(model, "bc_aux_lambda"),
                 "the 4B auxiliary pathway requires LeashedMaskablePPO (--worker --algo mppo)")
        model.bc_aux_lambda = args.bc_aux_lambda
    elif hasattr(model, "bc_aux_lambda"):
        model.bc_aux_lambda = 0.0

    _validate_model_recipe(model, expected_target_kl=args.target_kl)
    _validate_worker_policy_observation_binding(args, model)
    current_contract = _training_contract(
        args, model, batch_size,
        manager_npz_sha256=manager_npz_sha256,
        worker_npz_sha256=worker_npz_sha256,
        worker_zip_sha256=worker_zip_sha256,
        demos_sha256=demos_sha256,
        implementation_sha256=implementation_sha256,
        bc_aux_demos_sha256=bc_aux_demos_sha256,   # E4 rev5: 4B presence payload
    )
    if args.resume_from:
        _validate_resume_contract(
            getattr(model, "diablogym_contract", None), current_contract,
            allow_manager_change=args.allow_manager_change,
            allow_legacy_resume=args.allow_legacy_resume,
            allow_optimizer_reset=args.reset_optimizer,
            allow_target_kl_change=args.target_kl is not None,
            allow_environment_restart=args.allow_environment_restart_resume)
    if args.resource_warm_start:
        # No drift allowances apply here. Missing historical optional keys and
        # explicit None retain the existing strict-resume equivalence only.
        _validate_resume_contract(resource_warm_start_manifest["target_contract"],
                                  current_contract)
        from migrate_resource_candidate import validate_inherited_receipt
        validate_inherited_receipt(model._resource_warm_start_receipt, current_contract)
    model.diablogym_contract = current_contract
    config["training_contract"] = current_contract
    config["teacher_sha256"] = getattr(model, "teacher_sha256", None)

    # status total_steps is the SB3 global step; target_steps must use the same definition.
    config["start_steps"] = int(model.num_timesteps)
    target_global_steps = int(model.num_timesteps + args.total_steps)
    config["target_global_steps"] = target_global_steps
    publication_provenance = None
    if bc_aux_bank is not None:
        publication_provenance = {
            "protocol_version": PROTOCOL_VERSION,
            "implementation_sha256": implementation_sha256,
            "manager_npz_sha256": manager_npz_sha256,
            "resume_checkpoint_sha256": resume_checkpoint_sha256,
            "teacher_sha256": getattr(model, "teacher_sha256", None),
            "bc_aux_demos_sha256": bc_aux_demos_sha256,
            "bc_aux_liveness_preflight_sha256":
                bc_aux_preflight_sha256,
            "training_contract_sha256":
                _canonical_json_sha256(current_contract),
            "start_steps": int(model.num_timesteps),
            "target_global_steps": target_global_steps,
            "seed": args.seed,
            "optimizer_reset": bool(args.reset_optimizer),
            "target_kl": args.target_kl,
            "distill_beta": float(args.distill_beta),
            "bc_aux_lambda": float(args.bc_aux_lambda),
            "bc_aux_mode":
                "expanded-trainable-a12-contextual-mixture",
            "calib_record_only": bool(args.calib_record_only),
        }
        # Validate the full lineage before any real rollout; at final release validate once more with the actual end point,
        # so the training path cannot tamper with fields midway.
        _validate_publication_provenance(
            publication_provenance,
            demos_sha256=bc_aux_demos_sha256,
            final_step=target_global_steps)
        config["publication_provenance"] = publication_provenance

    if args.bc_init:
        # v22 devil arm: BC warm-start policy head; during the freeze only the value head trains (classic pitfall: the first PPO update
        # of a new value head destroys the BC policy, so freeze first to withstand it)
        gate = ("data_gate" if args.worker else "hypothesis" if args.options
                else "memoryless_hypothesis")
        sd = _load_bc_state_dict(args.bc_init, model.policy, gate, args.init_source)
        missing, unexpected = model.policy.load_state_dict(
            sd, strict=args.init_source == "checkpoint")
        _require(not unexpected, f"BC state_dict contains unknown keys: {unexpected}")
        _require(all(k not in missing for k in _POLICY_HEAD_KEYS),
                 f"BC policy head not fully loaded: {missing}")
        print(f"   BC warm start: loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        if args.freeze_policy_steps > 0:
            from stable_baselines3.common.callbacks import BaseCallback

            pi_params = (list(model.policy.mlp_extractor.policy_net.parameters())
                         + list(model.policy.action_net.parameters()))
            if getattr(model.policy, "share_features_extractor", False):
                pi_params += list(model.policy.features_extractor.parameters())
            # If the shared feature extractor were still updated by the value loss, the BC policy output would drift during the "freeze"
            # even with requires_grad=False on the head. Deduplicate and freeze them together.
            pi_params = list({id(p): p for p in pi_params}.values())
            for p in pi_params:
                p.requires_grad = False

            class _Unfreeze(BaseCallback):
                def __init__(self, when):
                    super().__init__()
                    self.when, self.done_ = when, False

                def _on_rollout_start(self):
                    # PPO updates only after the rollout is collected. Unfreezing in the _on_step that crosses the
                    # threshold would let the whole batch (including pre-threshold samples) update the
                    # policy. Unfreezing only at the start of the next rollout guarantees that the first
                    # freeze_policy_steps samples train only the value head.
                    if not self.done_ and self.num_timesteps >= self.when:
                        for p in pi_params:
                            p.requires_grad = True
                        self.done_ = True
                        print(f"   policy head unfrozen @ {self.num_timesteps}")

                def _on_step(self):
                    return True

            unfreeze_cb = _Unfreeze(args.freeze_policy_steps)
        else:
            unfreeze_cb = None
    else:
        unfreeze_cb = None

    bc_aux_anchor_sd = None
    if bc_aux_bank is not None:
        from leashed_ppo import derive_bc_aux_rng

        # By now resume/BC-init are both done. The first aux leg establishes the persistent root;
        # continuations reuse the same root anchor from the checkpoint. The root anchor must be restored/established before mount,
        # so neither the negative KL nor the rollout monitor re-anchors per leg. The original held-out
        # X/Y/episode/masks are kept in memory separately for the final release gate.
        bc_aux_anchor_sd = _persistent_bc_aux_root_anchor(model)
        if _bc_aux_structural_active(args):
            _require(bc_aux_fit is not None
                     and bc_aux_validation is not None,
                     "a12 circuit fit/validation not constructed")
            model.mount_bc_aux_circuit_fit(*bc_aux_fit)
            model.mount_bc_aux_circuit_validation(
                *bc_aux_validation)
            initial_monitor = model._bc_aux_rollout_monitor()
            _require(
                initial_monitor is not None
                and not initial_monitor["tripped"],
                "a12 circuit pre-launch online gate did not PASS")
        else:
            model.mount_bc_aux_demos(
                *bc_aux_bank, rng=derive_bc_aux_rng(args.seed))
        model.bc_aux_monitor_out = str(run_dir / "bc_aux_monitor.jsonl")
        # The very first real rollout reads the joint PPO/value/entropy/KING/aux gradient
        # and the fixed bank; we cannot wait until +49,152 to discover a directional conflict. This probe only peeks at
        # the demo stream; the real standalone rev5 aux step still consumes exactly one batch after the epochs.
        first_real_probe = int(model.num_timesteps) + int(
            model.n_steps * model.get_env().num_envs)
        model.calib_probes = sorted(set(
            list(getattr(model, "calib_probes", ())) + [first_real_probe]))
        model.calib_out = str(run_dir / "calib.jsonl")
        config["calib_probes_effective"] = list(model.calib_probes)
        print(f"   [4b] auxiliary demo pathway present: λ_bc={model.bc_aux_lambda},"
              f" bank n={len(bc_aux_bank[1])}"
              f"(a12={int((bc_aux_bank[1] == 12).sum())},"
              f" hard-neg={int((bc_aux_bank[1] != 12).sum())}),"
              f" objective_rev={_BC_AUX_OBJECTIVE_REVISION},"
              f" mode={'circuit' if _bc_aux_structural_active(args) else 'legacy'},"
              f" demos_sha16={bc_aux_demos_sha256[:16]},"
              " anchor=first-aux-root(persistent across continuations))")

    # Save an atomic checkpoint every ~250k samples with completed updates; a 499,712-step leg is protected at least at its midpoint.
    # B1-E0: the three intervals now come from CLI knobs (defaults inherit the old constants verbatim, zero drift in default behaviour).
    ckpt = AtomicRolloutCheckpointCallback(
        run_dir, every_steps=args.ckpt_every_steps,
        implementation_sha256=implementation_sha256)
    sentinel_cb = (WorkerSentinelCallback(run_dir, every=args.sentinel_every)
                   if args.worker else None)
    # R13: classroom audit gauge (pure read + IO, independent of the frozen sentinel emission surface), mounted only with the flag on
    r13_dive_audit_cb = (
        R13DiveAuditCallback(run_dir, every=args.sentinel_every)
        if (args.worker
            and getattr(args, "worker_learning_window_scope", "farm-only")
            in ("farm-dive-v1", EARNED_DIVE_SUFFIX_SCOPE))
        else None)
    prefix_audit_cb = (PrefixAuditCallback(run_dir)
                       if worker_prefix_identity is not None else None)
    # E1 four gates: dry_cb mount gate: worker and (skip_dry or schedule) (predicate inside the helper)
    dry_cb = (DryAnchorSentinel(run_dir, str(pathlib.Path(__file__).resolve().parent
                                             / "runs" / "bc-worker" / "demos.npz"),
                                  demos_sha256, every=args.dry_anchor_every)
              if _mount_dry_anchor_sentinel(args) else None)
    # E1 5A course callback (schedule is --worker only, already asserted by _validate_args)
    curriculum_cb = (DryCurriculumCallback(dry_curriculum_table, run_dir=run_dir)
                     if (args.worker and dry_curriculum_table) else None)
    # E5 gauge mounting (record-only, no verdict; knob 0 = not mounted = callback list equivalent to HEAD, G0-2a
    # prerequisite; the probe demo set is always pinned to the canonical BC-v1 demos bytes, asserted at construction, E6)
    _probe_demos = str(pathlib.Path(__file__).resolve().parent
                       / "runs" / "bc-worker" / "demos.npz")
    distill_ce_cb = (DistillCeProbe(run_dir, _probe_demos,
                                    every=args.distill_ce_probe_every)
                     if (args.worker and args.distill_ce_probe_every > 0)
                     else None)
    if distill_ce_cb is not None:
        _require(getattr(model, "teacher", None) is not None,
                 "--distill-ce-probe-every>0 requires a Leashed teacher present"
                 " (β>0 or teacher_path; fails loudly before launch)")
    drywin_cb = (DryWindowMetricsCallback(run_dir, _probe_demos,
                                          every=args.drywin_metrics_every)
                 if (args.worker and args.drywin_metrics_every > 0) else None)
    # Construct the only callback that holds a file handle last; the setup after it has no I/O that can fail.
    callback = EpisodeJsonlCallback(run_dir, config)
    learn_returned = False
    learn_completed = False
    completion_report = None
    diagnostic_artifact = None
    try:
        first_update_cb = _first_update_diagnostic_callback(
            args, run_dir, implementation_sha256)
        # E1 callback order pinned: the course callback is first in the list -- rollout-start first registers the next boundary's
        # per-env countdown, and rollout-tail verifies the in-environment atomic commit before the other callbacks.
        cbs = (([curriculum_cb] if curriculum_cb else [])
               + [callback, ckpt] + ([unfreeze_cb] if unfreeze_cb else [])
               + ([sentinel_cb] if sentinel_cb else [])
               + ([r13_dive_audit_cb] if r13_dive_audit_cb else [])
               + ([prefix_audit_cb] if prefix_audit_cb else [])
               + ([dry_cb] if dry_cb else [])
               # E5 gauges at the end of the list (pure read + IO; when absent these two entries are empty and the list equals HEAD)
               + ([distill_ce_cb] if distill_ce_cb else [])
               + ([drywin_cb] if drywin_cb else [])
               + ([first_update_cb] if first_update_cb else []))
        # v24: resume legs use reset_num_timesteps=False (False means train N more steps with continuous global steps
        # -> ckpt file names globally unique, β schedule and budget accounting unbroken; audit BLOCKER 2)
        model.learn(total_timesteps=args.total_steps, callback=cbs,
                    reset_num_timesteps=not args.resume_from)
        learn_returned = True
        # learn() also returns normally when collect_rollouts is cut short by a callback; in addition
        # G-CAL can refuse the whole update at the first minibatch of a full buffer. Neither may
        # publish weights with num_timesteps recorded but gradients not consumed as the formal end point.
        try:
            completion_report = _require_exact_training_completion(
                model, target_global_steps)
        except _TrainingCompletionRejected as exc:
            completion_report = exc.report
            if args.artifact_scope == "candidate":
                try:
                    diagnostic_artifact = _retain_refused_training_diagnostic(
                        model, run_dir, completion_report, implementation_sha256)
                except Exception as diagnostic_exc:
                    # Retention must not replace the original refusal or skip cleanup.
                    diagnostic_artifact = {
                        "status": "FAILED",
                        "error": f"{type(diagnostic_exc).__name__}: {diagnostic_exc}",
                    }
                    print(f"diagnostic artifact save failed: {diagnostic_exc}")
            raise
        learn_completed = True
    finally:
        active_exception = sys.exc_info()[1]
        active_error = active_exception is not None
        callback.close()
        save_error = None
        model_saved = False
        output_name, successful_publication_state = (
            _ARTIFACT_SCOPE_RESULTS[args.artifact_scope])
        output_path = run_dir / output_name
        if not active_error and learn_completed:
            try:
                _require(
                    int(model.num_timesteps) == target_global_steps,
                    "final release step count does not equal the frozen target: "
                    f"{int(model.num_timesteps)} != {target_global_steps}")
                final_implementation = _implementation_bundle_sha256()
                _require(final_implementation == implementation_sha256,
                         "implementation/engine/game content drifted during training; "
                         f"refusing to produce {output_name}: "
                         f"{final_implementation} != {implementation_sha256}")
                if args.artifact_scope != "production":
                    _require(
                        bc_aux_bank is None,
                        "non-production artifacts must not enter the bc_aux "
                        "final-heldout release gate",
                    )
                    _atomic_save_model(model, output_path)
                elif bc_aux_bank is not None:
                    _require(
                        all(value is not None for value in (
                            _aux_x, _aux_y, _aux_episode_id, _aux_masks,
                            bc_aux_demos_sha256, bc_aux_anchor_sd)),
                        "bc_aux final release gate evidence/starting anchor missing")
                    _publish_model_final_with_bc_aux_gate(
                        model, run_dir / "model_final.zip",
                        x=_aux_x, y=_aux_y,
                        episode_id=_aux_episode_id, masks=_aux_masks,
                        demos_sha256=bc_aux_demos_sha256,
                        anchor_sd=bc_aux_anchor_sd,
                        publication_provenance=publication_provenance)
                else:
                    _atomic_save_model(model, output_path)
                model_saved = True
            except Exception as exc:
                # close must run; the original implementation skipped subprocess cleanup when save failed.
                save_error = exc
                print(f"model save failed: {exc}")
        else:
            # After an error or a half-rollout early stop, num_timesteps may already include samples not yet updated;
            # such weights must not pose as the formal end point. The latest rollout-boundary ckpt can still be restored.
            if (completion_report is not None
                    and completion_report["checks"]["rollout_boundary"]
                    and completion_report["checks"]["exact_target"]):
                print(f"training updates completed, release acceptance failed; refusing to produce {output_name}: "
                      f"{completion_report['failed_checks']}")
            else:
                print(f"training did not stop on a complete update boundary; refusing to produce {output_name}")
        if model_saved:
            publication_state = successful_publication_state
            publication_sha256 = _capture_file_sha256(
                output_path, f"frozen {output_name}")
            publication_detail = None
        elif active_error:
            publication_state = "TRAINING_ERROR"
            publication_sha256 = None
            publication_detail = (
                f"{type(active_exception).__name__}: {active_exception}")
        else:
            publication_state = "PUBLICATION_REFUSED"
            publication_sha256 = None
            publication_detail = (
                f"{type(save_error).__name__}: {save_error}"
                if save_error is not None
                else "training did not complete at a publishable update boundary")
        status_error = None
        try:
            _record_run_publication_status(
                run_dir, publication_state,
                model_sha256=publication_sha256,
                detail=publication_detail,
                completion_report=completion_report,
                diagnostic_artifact=diagnostic_artifact,
                learn_returned=learn_returned)
        except Exception as status_exc:
            # Keep the error first and finish reclaiming resources. If training/saving itself had no earlier error,
            # the terminal status is part of the formal artifact transaction, and a failure must make the subprocess
            # exit non-zero; otherwise R7 would see a produced model that never gets its completion receipt.
            status_error = status_exc
            print(f"final status write failed: {status_exc}")
        # When a worker process crashes, close may block on a broken pipe; the resource owner
        # does the timeout cleanup centrally and restores the host's SIGALRM handler/timer.
        resources.close()
        if model_saved:
            print(f"model saved: {output_path}")
        if save_error is not None and not active_error:
            raise save_error
        if status_error is not None and not active_error:
            raise RuntimeError(
                "model/training terminal state could not be written transactionally to status.json"
            ) from status_error


def main():
    resources = _TrainingResources()
    try:
        return _main(resources)
    finally:
        # Covers every pre-learn failure too: model/load, contract, BC init,
        # and callback construction.  The normal learn-finally path is
        # idempotent and clears these handles before returning here.
        resources.close()


if __name__ == "__main__":
    main()
