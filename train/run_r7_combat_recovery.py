"""R7 survival-aligned, multi-seed combat-recovery campaign.

R6 proved that a single 32-pair development pool can select a policy whose
local combat efficiency rises while its fresh-pool survival and total combat
fall.  R7 changes both the training objective and the scientific protocol:

* frozen manager/script terminal deaths are credited back without leaking
  their positive rewards;
* two pre-registered death-risk costs compete as recipes, never as hand-picked
  checkpoints;
* each recipe is repeated under three fixed development training RNG seeds;
* development models are ``DEVELOPMENT_ONLY`` artifacts and cannot consume a
  final heldout publication gate;
* an independent production RNG retrains a non-published candidate from V28;
* final evidence is a one-shot 256-pair evaluation with simultaneous one-sided
  confidence bounds, exact sign tests, and paired death noninferiority;
* only a final PASS atomically promotes that candidate into an independent
  ``model_final.zip`` plus publication receipt.

The old R6 launcher and all of its burned pools remain immutable evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(TRAIN))

import eval_contract  # noqa: E402
import r7_statistics  # noqa: E402
import train_ppo  # noqa: E402
from leashed_ppo import (  # noqa: E402
    ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM,
    ASYMMETRIC_WORKER_CONTEXT_INTERACTION_DIM,
    WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
    WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
    validate_worker_onpolicy_pg_receipt,
)
from diablogym.options_env import DUAL_WORKER_OBSERVATION_DIM  # noqa: E402
from diablogym.controller_wire import (  # noqa: E402
    DUAL_WORKER_LAYOUT,
    DUAL_WORKER_LAYOUT_SHA256,
)
from diablogym.worker_env import (  # noqa: E402
    BC_RESERVED_SEED_RANGES,
    EVAL_RESERVED_SEED_RANGES,
    is_reserved_train_seed,
)


class CampaignError(RuntimeError):
    """R7 protocol or artifact identity is invalid."""


STATE_SCHEMA = "diablogym-r7-campaign-state/1"
TRAINING_RECEIPT_SCHEMA = "diablogym-r7-training-artifact/9"
DRY_CURRICULUM_LEDGER_SCHEMA = "diablogym-r7-dry-curriculum-ledger/1"
TRAINING_FIRED_SCHEMA = "diablogym-r7-training-fired/1"
DEVELOPMENT_DECISION_SCHEMA = "diablogym-r7-development-decision/1"
FINAL_OPENED_SCHEMA = "diablogym-r7-final-pool-opened/1"
FINAL_FIRED_SCHEMA = "diablogym-r7-final-candidate-fired/1"
EVAL_FIRED_SCHEMA = "diablogym-r7-eval-fired/1"
EVAL_ATTESTATION_SCHEMA = "diablogym-r7-eval-attestation/1"
FINAL_REGISTRY_SCHEMA = "diablogym-r7-final-pool-registry/1"
PUBLICATION_RECEIPT_SCHEMA = "diablogym-r7-publication/1"
CAMPAIGN_REVISION = 21

CONTROL_DIR = TRAIN / "runs" / "r7-combat-recovery-control"
STATE_PATH = CONTROL_DIR / "status.json"
LOCK_PATH = CONTROL_DIR / ".campaign.lock"
DEVELOPMENT_DECISION_PATH = CONTROL_DIR / "development-decision.json"
FINAL_OPENED_PATH = CONTROL_DIR / "final-pool-opened.json"
FINAL_FIRED_PATH = CONTROL_DIR / "final-candidate-fired.json"
FINAL_ANALYSIS_PATH = CONTROL_DIR / "analysis-final.json"
EVAL_INPUT_DIR = CONTROL_DIR / "eval-inputs"
EVAL_ATTESTATION_DIR = CONTROL_DIR / "eval-attestations"
TRAINING_FIRED_DIR = CONTROL_DIR / "training-fired"

EVAL_DIR = TRAIN / "runs" / "eval-assembled"
FINAL_REGISTRY_DIR = TRAIN / "runs" / "_r7_final_pool_registry"
FINAL_REGISTRY_LOCK = FINAL_REGISTRY_DIR / ".registry.lock"
PUBLISHED_DIR = TRAIN / "runs" / "r7-combat-recovery-published"
PUBLISHED_MODEL_PATH = PUBLISHED_DIR / "model_final.zip"
PUBLISHED_RECEIPT_PATH = PUBLISHED_DIR / "r7_publication_receipt.json"
V28_ZIP = TRAIN / "models" / "v28-worker-leg1" / "model_final.zip"
V28_SHA256 = "2f7bc9dd810956c3feeb330575c9a03ddff0b476333ac429a411935985b04f42"
V28_ACTOR_PARAMETER_SHA256 = (
    "4b205c7cd0358014b36ddf592374a82512e2816741de262fb0d0dd840a662ff8"
)
V28_CRITIC_PARAMETER_SHA256 = (
    "ace80a8a752375b5ceaf2ab25021c5da555e2a5dc1a4032a6dcef3be11f08a43"
)
M29_NPZ = TRAIN / "models" / "v29-manager-mfresh" / "policy.npz"
M29_SHA256 = "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6"
KING_SD = TRAIN / "runs" / "v32" / "king_anchor_sd.pt"
KING_SHA256 = "009aaad29d2653cde3f4e8ed2fafd8861a0f1f572a140c64118df9e3fa3df35d"
FULL_GAME_DATA_MODE = "full-diablo"
FULL_GAME_DATA_BASENAMES = ("DIABDAT.MPQ", "diabdat.mpq")
FULL_GAME_DATA_SHA256 = (
    "63fb47d9c76484024c7640d90ab6b7ec5e13f567a7e1a917b6c03a6631d3f2b0"
)
BC_V1_DIR = TRAIN / "runs" / "bc-worker"
BC_V1_POLICY = BC_V1_DIR / "policy_sd.pt"
BC_V1_DEMOS = BC_V1_DIR / "demos.npz"
BC_V1_REPORT = BC_V1_DIR / "bc_report.json"

START_STEPS = 3_497_984
CRITIC_WARMUP_STEPS = 16_384
ACTOR_LEG_STEPS = 249_856
LEG_STEPS = CRITIC_WARMUP_STEPS + ACTOR_LEG_STEPS
TARGET_STEPS = START_STEPS + LEG_STEPS
N_STEPS = 512
NUM_ENVS = 4
ROLLOUT_QUANTUM = N_STEPS * NUM_ENVS
TRAIN_CALLS = LEG_STEPS // ROLLOUT_QUANTUM
CRITIC_WARMUP_CALLS = CRITIC_WARMUP_STEPS // ROLLOUT_QUANTUM
ACTOR_TRAIN_CALLS = ACTOR_LEG_STEPS // ROLLOUT_QUANTUM
LEARNING_RATE = 1e-4
ENT_COEF = 0.005
TARGET_KL = 0.01
DISTILL_BETA = 0.015625
DISTILL_ANNEAL_ACTOR_ROLLOUTS = 61
ACTION14_LOGIT_BONUS = 2.5
CHECKPOINT_EVERY_STEPS = 31 * ROLLOUT_QUANTUM
WORKER_POLICY_OBSERVATION_VIEW = "dual-v4-asymmetric-v3"
# The fresh full-game critic receives eight critic-only rollouts at the
# original p_skip=1.0 start.  The actor then reaches deployment p_skip=0 after
# 92 rollouts and holds there for 30 more.  Hiding p_skip from the actor avoids
# a direct shortcut, but it cannot repair a state-distribution gap by itself;
# the zero tail is therefore part of the frozen recipe.
CURRICULUM = (
    "hold:1.0:8,"
    "linear:1.0:0.0:92,"
    "hold:0.0:30"
)
FAST_FORWARD_CREDIT = "terminal-death-only"
TERMINAL_DEATH_EVIDENCE_SCHEMA = (
    "diablogym-r7-terminal-death-reward-evidence/3"
)

_WORKER_SENTINEL_COUNT_KEYS = (
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
_WORKER_SENTINEL_REWARD_KEYS = (
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
_WORKER_FINAL_SENTINEL_KEYS = frozenset({
    "sentinel", "step", *_WORKER_SENTINEL_COUNT_KEYS,
    *_WORKER_SENTINEL_REWARD_KEYS,
    "dry_share", "reasons", "ff_reasons",
    "fast_forward_reward_credit_mode",
    "configured_additional_terminal_death_cost",
    "top1_action", "top1_share", "beta_initial", "beta",
    "distill_actor_rollouts_completed", "distill_ce",
    "teacher_entropy", "distill_kl", "distill_tv",
    "teacher_diverge", "final",
})

BC_V1_RANGE = (2_106_000, 2_106_128)
BC_V2_RANGE = (2_103_000, 2_103_384)
EVAL_BANK_RANGE = (2_110_000, 2_130_000)
DEV_POOLS = {
    "dev-a": tuple(range(2_110_000, 2_110_128)),
    "dev-b": tuple(range(2_111_000, 2_111_128)),
}
FINAL_POOL = tuple(range(2_120_000, 2_120_256))
DEVELOPMENT_TRAIN_SEEDS = (2_130_000, 2_130_100, 2_130_200)
PRODUCTION_TRAIN_SEED = 2_130_900

# Both recipes, their order, and the lower-cost tie break are frozen before any
# development episode is opened.  32 is comparable to the 30–50 cumulative
# exposure loss diagnosed in the burned R6 fresh failures; 64 tests whether
# that survival signal was still too weak.  No other coefficient may be added
# after viewing the new development pools.
RECIPES = {
    "risk32": {"additional_terminal_death_cost": 32.0},
    "risk64": {"additional_terminal_death_cost": 64.0},
}
RECIPE_PREFERENCE = ("risk32", "risk64")
MIN_REPLICATIONS_PASSING_BOTH_POOLS = 2
DEVELOPMENT_DEATH_MARGIN = 0.05
FINAL_DEATH_MARGIN = 0.025
FAMILYWISE_ALPHA = 0.05
GEAR_PROGRESSION_GATE_SCHEMA = (
    "diablogym-r7-action14-gear-progression-gate/2"
)
# A single opportunity at the terminal edge is too weak to demand a pickup,
# while a persistent legal item naturally appears on several Worker decision
# boundaries.  Once the paired pool exposes at least this many aggregate
# opportunities, zero requests or zero native utility growth is a scientific
# failure rather than an allowed "no drop" outcome.
MIN_ACTION14_MASK_OPPORTUNITIES = 8
METRIC_RULES = (
    r7_statistics.MetricRule("farm_worker_wage"),
    r7_statistics.MetricRule("farm_worker_kills"),
    r7_statistics.MetricRule("ret"),
    r7_statistics.MetricRule("kills"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _critic_init_seed(training_seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(
            b"diablogym/full-game-worker-critic-v1\0"
            + str(training_seed).encode("ascii")
        ).digest()[:8],
        byteorder="big",
    ) & ((1 << 63) - 1)


_CANONICAL_MIGRATION_EVIDENCE_CACHE: dict[tuple, dict] = {}


def _canonical_migration_evidence(training_seed: int) -> dict:
    """Independently reconstruct the current migration/reset ground truth."""
    _require(
        _is_plain_int(training_seed) and 0 <= training_seed < 2**32,
        "canonical migration evidence seed 非法",
    )
    key = (
        training_seed,
        V28_SHA256,
        DUAL_WORKER_LAYOUT.schema,
        DUAL_WORKER_LAYOUT_SHA256,
    )
    cached = _CANONICAL_MIGRATION_EVIDENCE_CACHE.get(key)
    if cached is not None:
        return json.loads(json.dumps(cached))
    payload = _stable_read(V28_ZIP)
    _require(
        hashlib.sha256(payload).hexdigest() == V28_SHA256,
        "canonical migration evidence 的 V28 payload 漂移",
    )
    try:
        evidence = (
            train_ppo._canonical_asymmetric_worker_migration_evidence(
                source_checkpoint_payload=payload,
                source_checkpoint_sha256=V28_SHA256,
                training_seed=training_seed,
            )
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CampaignError(
            "canonical asymmetric migration/reset 重构失败") from exc
    _require(
        isinstance(evidence, dict)
        and evidence.get("schema")
        == train_ppo._ASYMMETRIC_CANONICAL_EVIDENCE_SCHEMA
        and evidence.get("source_checkpoint_sha256") == V28_SHA256
        and evidence.get("training_seed") == training_seed
        and isinstance(evidence.get("actor_migration"), dict)
        and isinstance(evidence.get("critic_reset"), dict)
        and isinstance(evidence.get("runtime"), dict),
        "canonical asymmetric migration/reset evidence 非法",
    )
    _CANONICAL_MIGRATION_EVIDENCE_CACHE[key] = json.loads(
        json.dumps(evidence))
    return json.loads(json.dumps(evidence))


def _teacher_actor_sha256() -> str:
    """Prove KING is the exact V28 root, not an unrelated policy leash."""
    import torch

    try:
        state = torch.load(
            KING_SD, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise CampaignError("KING actor state_dict 无法取证") from exc
    keys = (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    )
    try:
        return train_ppo._stable_named_tensor_sha256(state, keys)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CampaignError("KING actor tensors 无法摘要") from exc


def _checkpoint_policy_branch_sha256(
        checkpoint_payload: bytes, *,
        require_asymmetric: bool = False) -> dict:
    """Verify and hash the asymmetric actor root/context and critic."""
    import io
    import zipfile

    import numpy as np
    import torch

    try:
        with zipfile.ZipFile(io.BytesIO(checkpoint_payload)) as archive:
            state = torch.load(
                io.BytesIO(archive.read("policy.pth")),
                map_location="cpu",
                weights_only=True,
            )
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise CampaignError("checkpoint policy.pth 无法取证") from exc
    root_actor_keys = (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    )
    context_enabled_key = "mlp_extractor._context_enabled"

    def digest(keys: tuple[str, ...]) -> str:
        _require(
            isinstance(state, dict)
            and all(key in state for key in keys),
            "checkpoint policy 分支张量缺失",
        )
        result = hashlib.sha256()
        for key in sorted(keys):
            tensor = state[key]
            _require(
                isinstance(tensor, torch.Tensor)
                and bool(torch.isfinite(tensor).all().item()),
                f"checkpoint policy 张量非法:{key}",
            )
            array = tensor.detach().cpu().contiguous().numpy()
            result.update(key.encode("utf-8"))
            result.update(str(array.dtype).encode("ascii"))
            result.update(
                np.asarray(array.shape, dtype=np.int64).tobytes())
            result.update(array.tobytes())
        return result.hexdigest()

    context_keys = tuple(sorted(
        key for key in state
        if key.startswith("mlp_extractor.context_adapter.")
    )) if isinstance(state, dict) else ()
    critic_keys = tuple(sorted(
        key for key in state
        if (
            key.startswith("mlp_extractor.value_net.")
            or key.startswith("value_net.")
        )
    )) if isinstance(state, dict) else ()
    if not context_keys or context_enabled_key not in state:
        _require(
            not require_asymmetric,
            "asymmetric checkpoint 缺 context adapter/buffer",
        )
        return {
            "root_actor_sha256": digest(root_actor_keys),
            "critic_sha256": digest(critic_keys),
        }
    try:
        from leashed_ppo import (
            LeashedMaskablePPO,
            asymmetric_worker_runtime_evidence,
        )
        runtime_model = LeashedMaskablePPO.load(
            io.BytesIO(checkpoint_payload),
            env=None,
            device="cpu",
            teacher_path=None,
            teacher_sha256=None,
        )
        runtime_evidence = asymmetric_worker_runtime_evidence(
            runtime_model.policy)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise CampaignError(
            "structured asymmetric checkpoint 无法生成 runtime evidence"
        ) from exc
    _require(
        train_ppo._asymmetric_worker_deployment_evidence_complete(
            runtime_model),
        "structured asymmetric checkpoint deployment evidence 未闭合",
    )
    context_enabled = state[context_enabled_key]
    encoder_keys = tuple(
        key for key in context_keys
        if (
            ".context_adapter.encoder." in key
            or ".context_adapter.context_projection." in key
        )
    )
    interaction_keys = tuple(
        key for key in context_keys
        if any(
            marker in key
            for marker in (
                ".context_adapter.context_gate.",
                ".context_adapter.legacy_gate.",
                ".context_adapter.fusion.",
            )
        )
    )
    output_keys = tuple(
        key for key in context_keys
        if ".context_adapter.output." in key
    )
    context_parameter_count = sum(
        int(state[key].numel()) for key in context_keys)
    critic_parameter_count = sum(
        int(state[key].numel()) for key in critic_keys)
    _require(
        len(context_keys) > 0
        and len(encoder_keys) > 0
        and len(interaction_keys) > 0
        and len(output_keys) > 0
        and len(critic_keys) > 0
        and set(context_keys)
        == set((*encoder_keys, *interaction_keys, *output_keys))
        and context_parameter_count
        == runtime_evidence["context"]["parameter_count"]
        and len(context_keys)
        == runtime_evidence["context"]["tensor_count"]
        and critic_parameter_count
        == runtime_evidence["policy"]["critic_parameter_count"]
        and len(critic_keys)
        == runtime_evidence["policy"]["critic_tensor_count"]
        and set(state) == set((
            *root_actor_keys,
            *context_keys,
            *critic_keys,
            context_enabled_key,
        ))
        and all(
            isinstance(state[key], torch.Tensor)
            and bool(torch.isfinite(state[key]).all().item())
            for key in (*context_keys, *critic_keys)
        )
        and isinstance(context_enabled, torch.Tensor)
        and context_enabled.numel() == 1
        and context_enabled.dtype == torch.bool
        and bool(context_enabled.item()),
        "structured asymmetric checkpoint 分支/有限性/启用状态非法",
    )
    context_arrays = [
        state[key].detach().cpu().contiguous().numpy()
        for key in context_keys
    ]
    context_nonzero = sum(
        int(np.count_nonzero(array)) for array in context_arrays)
    context_l2_norm = float(
        math.sqrt(sum(
            float(np.square(
                array.astype(np.float64, copy=False)).sum())
            for array in context_arrays
        ))
    )
    action_weight = state["action_net.weight"].detach().cpu()
    output = state[output_keys[0]].detach().cpu()
    fusion = state[
        "mlp_extractor.context_adapter.fusion.linear.weight"
    ].detach().cpu()
    _require(
        action_weight.ndim == 2
        and output.ndim == 2
        and fusion.ndim == 2
        and action_weight.shape[1] == output.shape[0]
        and output.shape[1] == fusion.shape[0]
        and fusion.shape[1]
        == (
            ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM
            + ASYMMETRIC_WORKER_CONTEXT_INTERACTION_DIM
        ),
        "structured asymmetric output/fusion 张量形状漂移",
    )
    context_action_effect = (
        action_weight
        @ output
        @ fusion[:, :ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM]
    ).contiguous().numpy()
    interaction_action_effect = (
        action_weight
        @ output
        @ fusion[:, ASYMMETRIC_WORKER_CONTEXT_HIDDEN_DIM:]
    ).contiguous().numpy()
    context_action_effect_nonzero = int(
        np.count_nonzero(context_action_effect))
    interaction_action_effect_nonzero = int(
        np.count_nonzero(interaction_action_effect))
    context_action_effect_l2_norm = float(np.linalg.norm(
        context_action_effect.astype(np.float64, copy=False)))
    interaction_action_effect_l2_norm = float(np.linalg.norm(
        interaction_action_effect.astype(np.float64, copy=False)))
    _require(
        context_nonzero > 0
        and math.isfinite(context_l2_norm)
        and context_l2_norm > 0.0
        and context_action_effect_nonzero > 0
        and interaction_action_effect_nonzero > 0
        and math.isfinite(context_action_effect_l2_norm)
        and context_action_effect_l2_norm > 0.0
        and math.isfinite(interaction_action_effect_l2_norm)
        and interaction_action_effect_l2_norm > 0.0,
        "asymmetric checkpoint context 未学到可影响 action logits 的参数",
    )
    full_actor_sha256 = digest((*root_actor_keys, *context_keys))
    critic_sha256 = digest(critic_keys)
    _require(
        full_actor_sha256
        == runtime_evidence["policy"]["actor_sha256"]
        and critic_sha256
        == runtime_evidence["policy"]["critic_sha256"]
        and digest(encoder_keys)
        == runtime_evidence["context"]["parameter_groups"][
            "encoder"]["sha256"]
        and digest(interaction_keys)
        == runtime_evidence["context"]["parameter_groups"][
            "interaction"]["sha256"]
        and digest(output_keys)
        == runtime_evidence["context"]["parameter_groups"][
            "output"]["sha256"],
        "checkpoint raw state/runtime parameter 摘要不一致",
    )
    return {
        "root_actor_sha256": digest(root_actor_keys),
        "full_actor_sha256": full_actor_sha256,
        "critic_sha256": critic_sha256,
        "context_sha256": digest(context_keys),
        "context_encoder_sha256":
            runtime_evidence["context"]["parameter_groups"][
                "encoder"]["sha256"],
        "context_interaction_sha256":
            runtime_evidence["context"]["parameter_groups"][
                "interaction"]["sha256"],
        "context_output_sha256":
            runtime_evidence["context"]["parameter_groups"][
                "output"]["sha256"],
        "context_nonzero": context_nonzero,
        "context_parameter_count": context_parameter_count,
        "critic_parameter_count": critic_parameter_count,
        "context_l2_norm": context_l2_norm,
        "context_action_effect_nonzero":
            context_action_effect_nonzero,
        "interaction_action_effect_nonzero":
            interaction_action_effect_nonzero,
        "context_action_effect_l2_norm":
            context_action_effect_l2_norm,
        "interaction_action_effect_l2_norm":
            interaction_action_effect_l2_norm,
        "context_enabled": True,
    }


def _validate_actor_migration_receipt(
        receipt: dict, *, canonical_receipt: dict | None = None) -> None:
    expected_keys = {
        "schema", "method", "source_checkpoint_sha256",
        "source_actor_sha256", "source_critic_sha256",
        "source_policy_class", "source_policy_observation_shape",
        "migrated_actor_sha256", "source_actor_parameter_tensors",
        "target_actor_parameter_tensors", "target_actor_parameter_count",
        "context_parameter_tensors", "context_parameter_count",
        "context_enabled", "context_architecture", "context_initializer",
        "controller_layout_schema", "controller_layout_sha256",
        "context_sha256", "context_encoder_sha256",
        "context_interaction_sha256", "context_output_sha256",
        "context_hidden_nonzero", "context_output_nonzero",
        "context_excluded_preoutput_bitwise_equal",
        "actor_context_excluded_observation_features",
        "bitwise_probe_rows",
        "bitwise_probe_sha256",
    }
    _require(
        isinstance(receipt, dict) and set(receipt) == expected_keys,
        "actor migration receipt 字段漂移",
    )
    if canonical_receipt is None:
        canonical_receipt = _canonical_migration_evidence(
            0)["actor_migration"]
    _require(
        isinstance(canonical_receipt, dict)
        and set(canonical_receipt) == expected_keys
        and receipt == canonical_receipt,
        "actor migration receipt 不等于独立 canonical 重构",
    )


def _validate_critic_migration_receipt(
        receipt: dict, *, seed: int, actor_receipt: dict,
        canonical_evidence: dict | None = None) -> None:
    expected_keys = {
        "schema", "method", "critic_architecture",
        "controller_layout_schema", "controller_layout_sha256",
        "source_checkpoint_sha256", "training_seed",
        "source_actor_sha256", "source_critic_sha256",
        "init_seed", "actor_sha256_before", "actor_sha256_after",
        "critic_sha256_before", "critic_sha256_after",
        "actor_parameter_tensors", "critic_parameter_tensors",
        "critic_parameter_count",
        "gradient_clip_mode", "warmup_start_timesteps",
        "warmup_until_timesteps", "warmup_steps",
        "warmup_rollout_quantum", "warmup_expected_rollouts",
        "actor_sha256", "optimizer_reset",
        "worker_onpolicy_pg_audit_schema",
    }
    _require(
        isinstance(receipt, dict) and set(receipt) == expected_keys,
        "critic migration receipt 字段漂移",
    )
    if canonical_evidence is None:
        canonical_evidence = _canonical_migration_evidence(seed)
    canonical_actor = canonical_evidence.get("actor_migration")
    canonical_reset = canonical_evidence.get("critic_reset")
    _require(
        isinstance(canonical_actor, dict)
        and isinstance(canonical_reset, dict)
        and actor_receipt == canonical_actor
        and all(
            receipt.get(key) == value
            for key, value in canonical_reset.items()
        ),
        "critic migration receipt 不等于独立 canonical reset",
    )
    _require(
        receipt["training_seed"] == seed
        and receipt["init_seed"] == _critic_init_seed(seed)
        and receipt["actor_sha256"]
        == canonical_actor["migrated_actor_sha256"]
        and receipt["gradient_clip_mode"]
        == "separate-root-context-critic-v2"
        and receipt["warmup_start_timesteps"] == START_STEPS
        and receipt["warmup_until_timesteps"]
        == START_STEPS + CRITIC_WARMUP_STEPS
        and receipt["warmup_steps"] == CRITIC_WARMUP_STEPS
        and receipt["warmup_rollout_quantum"] == ROLLOUT_QUANTUM
        and receipt["warmup_expected_rollouts"] == CRITIC_WARMUP_CALLS
        and receipt["worker_onpolicy_pg_audit_schema"]
        == WORKER_ONPOLICY_PG_AUDIT_SCHEMA
        and receipt["optimizer_reset"] is True,
        "critic migration receipt 内容漂移",
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stable_read(path: pathlib.Path) -> bytes:
    """Read one regular file identity and reject symlink/replace races."""
    path = pathlib.Path(path)
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise CampaignError(f"文件不可读:{path}: {exc}") from exc
    _require(not stat.S_ISLNK(before_path.st_mode), f"拒绝符号链接输入:{path}")
    _require(stat.S_ISREG(before_path.st_mode), f"输入不是普通文件:{path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CampaignError(f"文件不可稳定打开:{path}: {exc}") from exc
    try:
        first = os.fstat(fd)
        _require(stat.S_ISREG(first.st_mode), f"已打开输入不是普通文件:{path}")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        second = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise CampaignError(f"文件读取后身份消失:{path}: {exc}") from exc
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_size,
        item.st_mtime_ns, item.st_ctime_ns,
    )
    _require(
        identity(before_path) == identity(first)
        == identity(second) == identity(after_path),
        f"文件读取期间被替换或修改:{path}",
    )
    return b"".join(chunks)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(_stable_read(path)).hexdigest()


def _stable_json(path: pathlib.Path) -> dict:
    value, _digest = _stable_json_snapshot(path)
    return value


def _stable_json_snapshot(path: pathlib.Path) -> tuple[dict, str]:
    """Return one strictly parsed JSON payload and the SHA of those same bytes."""
    try:
        payload = _stable_read(path)
        value = eval_contract.strict_json_loads(payload)
    except (CampaignError, eval_contract.EvalContractError) as exc:
        raise CampaignError(f"JSON 不可稳定读取:{path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON 顶层不是对象:{path}")
    return value, hashlib.sha256(payload).hexdigest()


def _finite_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_reason_counts(value: Any, label: str) -> dict[str, int]:
    _require(isinstance(value, dict), f"{label} 必须是对象")
    _require(
        all(
            isinstance(key, str) and bool(key)
            and _is_plain_int(count) and count >= 0
            for key, count in value.items()
        ),
        f"{label} 必须是非空字符串→非负普通整数",
    )
    return value


def _terminal_death_reward_evidence(
        sentinel_path: pathlib.Path, *, recipe: str,
        target_step: int = TARGET_STEPS) -> dict:
    """Validate the one final Worker sentinel that proves R7 reward wiring.

    A training contract proves which knobs were requested; this receipt proves
    that every observed direct/manager-transition/reset death or no-progress
    timeout failure was classified and that the configured risk cost was
    conserved in the actual rollout ledger.  A run with neither learning-
    facing event remains a valid training artifact, but is explicitly marked
    as lacking an on-rollout opportunity to exercise either failure mechanism.
    """
    _require(recipe in RECIPES, f"未知 R7 recipe:{recipe}")
    _require(_is_plain_int(target_step) and target_step > 0,
             "sentinel target_step 必须是正普通整数")
    path = pathlib.Path(sentinel_path)
    payload = _stable_read(path)
    lines = payload.splitlines()
    _require(lines, f"Worker sentinel 为空:{path}")
    documents = []
    for index, line in enumerate(lines, start=1):
        _require(bool(line.strip()),
                 f"Worker sentinel 第 {index} 行为空")
        try:
            document = eval_contract.strict_json_loads(line)
        except eval_contract.EvalContractError as exc:
            raise CampaignError(
                f"Worker sentinel 第 {index} 行 JSON 非法:{exc}") from exc
        _require(
            isinstance(document, dict)
            and document.get("sentinel") in {"v23", "dry-anchor"},
            f"Worker sentinel 第 {index} 行类型/schema 非法",
        )
        documents.append(document)

    final_rows = [
        row for row in documents
        if row.get("sentinel") == "v23" and row.get("final") is True
    ]
    target_rows = [
        row for row in documents
        if row.get("sentinel") == "v23"
        and row.get("step") == target_step
    ]
    _require(
        len(final_rows) == 1
        and len(target_rows) == 1
        and final_rows[0] is target_rows[0],
        "Worker sentinel 必须恰有一条 target-step final v23 行",
    )
    row = final_rows[0]
    _require(
        set(row) == _WORKER_FINAL_SENTINEL_KEYS,
        "Worker final sentinel 字段漂移:"
        f"缺={sorted(_WORKER_FINAL_SENTINEL_KEYS - set(row))},"
        f"多={sorted(set(row) - _WORKER_FINAL_SENTINEL_KEYS)}",
    )
    _require(
        row["sentinel"] == "v23"
        and row["final"] is True
        and _is_plain_int(row["step"])
        and row["step"] == target_step,
        "Worker final sentinel 身份/step 非法",
    )
    _require(
        all(
            _is_plain_int(row[key]) and row[key] >= 0
            for key in _WORKER_SENTINEL_COUNT_KEYS
        ),
        "Worker final sentinel count 必须是非负普通整数",
    )
    _require(
        all(_finite_json_number(row[key])
            for key in _WORKER_SENTINEL_REWARD_KEYS),
        "Worker final sentinel reward 含非数值/NaN/Inf",
    )
    reasons = _validate_reason_counts(
        row["reasons"], "Worker final sentinel reasons")
    ff_reasons = _validate_reason_counts(
        row["ff_reasons"], "Worker final sentinel ff_reasons")
    _require(
        _finite_json_number(row["dry_share"])
        and 0.0 <= float(row["dry_share"]) <= 1.0
        and math.isclose(
            float(row["dry_share"]),
            round(
                row["dry"]
                / max(1, row["dry"] + row["fresh"]),
                4,
            ),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and _is_plain_int(row["top1_action"])
        and -1 <= row["top1_action"] < 15
        and _finite_json_number(row["top1_share"])
        and 0.0 <= float(row["top1_share"]) <= 1.0
        and row["beta_initial"] == DISTILL_BETA
        and row["beta"] == 0.0
        and row["distill_actor_rollouts_completed"]
        == ACTOR_TRAIN_CALLS
        and all(
            value is None or _finite_json_number(value)
            for value in (
                row["beta"], row["distill_ce"],
                row["teacher_entropy"], row["distill_kl"],
                row["distill_tv"],
                row["teacher_diverge"],
            )
        ),
        "Worker final sentinel telemetry 非法",
    )
    _require(
        (
            row["distill_kl"] is None
            or float(row["distill_kl"]) >= -1e-6
        )
        and (
            row["distill_tv"] is None
            or 0.0 <= float(row["distill_tv"]) <= 1.0
        ),
        "Worker final sentinel KL/TV 非法",
    )
    _require(
        row["windows"] == row["dry"] + row["fresh"]
        and sum(ff_reasons.values()) == row["ff_windows"]
        and row["ff_terminals"]
        >= (
            row["transition_ff_terminal_deaths"]
            + row["reset_ff_terminal_deaths"]
            + row["manual_ff_terminal_deaths"]
            + row["transition_ff_no_progress_timeouts"]
            + row["reset_ff_no_progress_timeouts"]
            + row["manual_ff_no_progress_timeouts"]
        ),
        "Worker final sentinel window/fast-forward count 不闭合",
    )

    direct = row["direct_terminal_deaths"]
    transition = row["transition_ff_terminal_deaths"]
    reset = row["reset_ff_terminal_deaths"]
    manual = row["manual_ff_terminal_deaths"]
    direct_timeout = row["direct_no_progress_timeouts"]
    transition_timeout = row["transition_ff_no_progress_timeouts"]
    reset_timeout = row["reset_ff_no_progress_timeouts"]
    manual_timeout = row["manual_ff_no_progress_timeouts"]
    _require(
        reasons.get("death", 0) == direct + transition + reset + manual
        and ff_reasons.get("death", 0) == transition + reset + manual,
        "Worker final sentinel direct/FF death reason 计数不闭合",
    )
    _require(
        manual == 0
        and manual_timeout == 0
        and row["manual_ff_calls"] == 0
        and row["interrupted_resets"] == 0
        and math.isclose(
            float(row["manual_ff_reward"]), 0.0,
            rel_tol=0.0, abs_tol=1e-9)
        and math.isclose(
            float(row[
                "manual_ff_no_progress_timeout_failure_reward"
            ]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "R7 训练路径禁止消费 manual fast-forward",
    )
    expected_cost = float(
        RECIPES[recipe]["additional_terminal_death_cost"])
    _require(
        row["fast_forward_reward_credit_mode"] == FAST_FORWARD_CREDIT
        and _finite_json_number(
            row["configured_additional_terminal_death_cost"])
        and float(row["configured_additional_terminal_death_cost"])
        == expected_cost,
        "Worker final sentinel 奖励配置与 R7 recipe 漂移",
    )

    def close(actual: Any, expected: float, label: str) -> None:
        _require(
            math.isclose(
                float(actual), float(expected),
                rel_tol=0.0, abs_tol=1e-6),
            f"Worker final sentinel {label} 不守恒:"
            f"{actual} != {expected}",
        )

    direct_additional = row[
        "direct_additional_terminal_death_reward"]
    transition_existing = row[
        "transition_ff_terminal_death_reward"]
    transition_additional = row[
        "transition_ff_additional_terminal_death_reward"]
    reset_additional = row[
        "reset_ff_additional_terminal_death_reward"]
    close(direct_additional, -expected_cost * direct,
          "direct additional death reward")
    close(transition_additional, -expected_cost * transition,
          "transition additional death reward")
    close(reset_additional, -expected_cost * reset,
          "reset additional death reward")
    close(
        row["additional_terminal_death_reward"],
        float(direct_additional) + float(transition_additional),
        "learning-facing total additional death reward",
    )
    close(
        row["credited_ff_terminal_death_reward"],
        float(transition_existing) + float(transition_additional),
        "credited transition death reward",
    )
    for count, reward_key in (
        (direct, "direct_existing_terminal_death_reward"),
        (transition, "transition_ff_terminal_death_reward"),
        (reset, "reset_ff_terminal_death_reward"),
    ):
        reward = float(row[reward_key])
        _require(
            (count == 0 and math.isclose(
                reward, 0.0, rel_tol=0.0, abs_tol=1e-9))
            or (count > 0 and reward < 0.0),
            f"Worker final sentinel {reward_key} 与 death count 矛盾",
        )

    timeout_reward_pairs = (
        (
            direct_timeout,
            "direct_no_progress_timeout_failure_reward",
        ),
        (
            transition_timeout,
            "transition_ff_no_progress_timeout_failure_reward",
        ),
        (
            reset_timeout,
            "reset_ff_no_progress_timeout_failure_reward",
        ),
    )
    for count, reward_key in timeout_reward_pairs:
        reward = float(row[reward_key])
        _require(
            (
                count == 0
                and math.isclose(
                    reward, 0.0, rel_tol=0.0, abs_tol=1e-9)
            )
            or (
                count > 0
                and reward < -expected_cost * count
            ),
            f"Worker final sentinel {reward_key} "
            "与 no-progress timeout count/cost 矛盾",
        )
    close(
        row["credited_no_progress_timeout_failure_reward"],
        float(row[
            "direct_no_progress_timeout_failure_reward"
        ]) + float(row[
            "transition_ff_no_progress_timeout_failure_reward"
        ]),
        "credited no-progress timeout failure reward",
    )

    learning_deaths = direct + transition
    learning_timeouts = direct_timeout + transition_timeout
    opportunity_status = (
        "TRAINING_FAILURE_OBSERVED"
        if learning_deaths + learning_timeouts > 0
        else "NO_TRAINING_FAILURE_OBSERVED"
    )
    evidence = {
        "schema_version": TERMINAL_DEATH_EVIDENCE_SCHEMA,
        "sentinel_sha256": hashlib.sha256(payload).hexdigest(),
        "sentinel_step": target_step,
        "opportunity_status": opportunity_status,
        "reward_mechanism_triggered":
            learning_deaths + learning_timeouts > 0,
        "fast_forward_reward_credit_mode":
            row["fast_forward_reward_credit_mode"],
        "configured_additional_terminal_death_cost": expected_cost,
        "control_path_counts": {
            "interrupted_resets": row["interrupted_resets"],
            "manual_ff_calls": row["manual_ff_calls"],
        },
        "death_counts": {
            "direct": direct,
            "transition_ff": transition,
            "reset_ff": reset,
            "manual_ff": manual,
        },
        "death_rewards": {
            key: float(row[key])
            for key in (
                "direct_existing_terminal_death_reward",
                "direct_additional_terminal_death_reward",
                "transition_ff_terminal_death_reward",
                "transition_ff_additional_terminal_death_reward",
                "credited_ff_terminal_death_reward",
                "reset_ff_terminal_death_reward",
                "reset_ff_additional_terminal_death_reward",
                "additional_terminal_death_reward",
            )
        },
        "no_progress_timeout_counts": {
            "direct": direct_timeout,
            "transition_ff": transition_timeout,
            "reset_ff": reset_timeout,
            "manual_ff": manual_timeout,
        },
        "no_progress_timeout_rewards": {
            key: float(row[key])
            for key in (
                "direct_no_progress_timeout_failure_reward",
                "transition_ff_no_progress_timeout_failure_reward",
                "reset_ff_no_progress_timeout_failure_reward",
                "manual_ff_no_progress_timeout_failure_reward",
                "credited_no_progress_timeout_failure_reward",
            )
        },
    }
    if target_step == TARGET_STEPS:
        return _validate_terminal_death_evidence_document(
            evidence, recipe=recipe)
    return evidence


def _validate_terminal_death_evidence_document(
        evidence: Any, *, recipe: str) -> dict:
    """Revalidate the compact receipt after it crosses an artifact boundary."""
    _require(recipe in RECIPES, f"未知 R7 recipe:{recipe}")
    reward_keys = {
        "direct_existing_terminal_death_reward",
        "direct_additional_terminal_death_reward",
        "transition_ff_terminal_death_reward",
        "transition_ff_additional_terminal_death_reward",
        "credited_ff_terminal_death_reward",
        "reset_ff_terminal_death_reward",
        "reset_ff_additional_terminal_death_reward",
        "additional_terminal_death_reward",
    }
    timeout_reward_keys = {
        "direct_no_progress_timeout_failure_reward",
        "transition_ff_no_progress_timeout_failure_reward",
        "reset_ff_no_progress_timeout_failure_reward",
        "manual_ff_no_progress_timeout_failure_reward",
        "credited_no_progress_timeout_failure_reward",
    }
    _require(
        isinstance(evidence, dict)
        and set(evidence) == {
            "schema_version", "sentinel_sha256", "sentinel_step",
            "opportunity_status", "reward_mechanism_triggered",
            "fast_forward_reward_credit_mode",
            "configured_additional_terminal_death_cost",
            "control_path_counts",
            "death_counts", "death_rewards",
            "no_progress_timeout_counts",
            "no_progress_timeout_rewards",
        }
        and evidence.get("schema_version")
        == TERMINAL_DEATH_EVIDENCE_SCHEMA
        and _is_sha256(evidence.get("sentinel_sha256"))
        and evidence.get("sentinel_step") == TARGET_STEPS
        and evidence.get("opportunity_status") in {
            "TRAINING_FAILURE_OBSERVED",
            "NO_TRAINING_FAILURE_OBSERVED",
        }
        and evidence.get("reward_mechanism_triggered")
        is (
            evidence.get("opportunity_status")
            == "TRAINING_FAILURE_OBSERVED"
        )
        and evidence.get("fast_forward_reward_credit_mode")
        == FAST_FORWARD_CREDIT
        and _finite_json_number(
            evidence.get("configured_additional_terminal_death_cost"))
        and float(evidence[
            "configured_additional_terminal_death_cost"])
        == float(RECIPES[recipe]["additional_terminal_death_cost"])
        and evidence.get("control_path_counts") == {
            "interrupted_resets": 0,
            "manual_ff_calls": 0,
        }
        and isinstance(evidence.get("death_counts"), dict)
        and set(evidence["death_counts"])
        == {"direct", "transition_ff", "reset_ff", "manual_ff"}
        and all(
            _is_plain_int(value) and value >= 0
            for value in evidence["death_counts"].values()
        )
        and evidence["death_counts"]["manual_ff"] == 0
        and isinstance(evidence.get("death_rewards"), dict)
        and set(evidence["death_rewards"]) == reward_keys
        and all(_finite_json_number(value)
                for value in evidence["death_rewards"].values())
        and isinstance(
            evidence.get("no_progress_timeout_counts"), dict)
        and set(evidence["no_progress_timeout_counts"])
        == {"direct", "transition_ff", "reset_ff", "manual_ff"}
        and all(
            _is_plain_int(value) and value >= 0
            for value in evidence[
                "no_progress_timeout_counts"].values()
        )
        and evidence[
            "no_progress_timeout_counts"]["manual_ff"] == 0
        and isinstance(
            evidence.get("no_progress_timeout_rewards"), dict)
        and set(evidence["no_progress_timeout_rewards"])
        == timeout_reward_keys
        and all(
            _finite_json_number(value)
            for value in evidence[
                "no_progress_timeout_rewards"].values()
        ),
        "terminal-death reward evidence 文档字段/身份非法",
    )
    counts = evidence["death_counts"]
    rewards = evidence["death_rewards"]
    timeout_counts = evidence["no_progress_timeout_counts"]
    timeout_rewards = evidence["no_progress_timeout_rewards"]
    cost = float(evidence["configured_additional_terminal_death_cost"])
    identities = (
        (
            rewards["direct_additional_terminal_death_reward"],
            -cost * counts["direct"],
        ),
        (
            rewards["transition_ff_additional_terminal_death_reward"],
            -cost * counts["transition_ff"],
        ),
        (
            rewards["reset_ff_additional_terminal_death_reward"],
            -cost * counts["reset_ff"],
        ),
        (
            rewards["additional_terminal_death_reward"],
            rewards["direct_additional_terminal_death_reward"]
            + rewards["transition_ff_additional_terminal_death_reward"],
        ),
        (
            rewards["credited_ff_terminal_death_reward"],
            rewards["transition_ff_terminal_death_reward"]
            + rewards[
                "transition_ff_additional_terminal_death_reward"],
        ),
    )
    _require(
        all(math.isclose(
            float(actual), float(expected),
            rel_tol=0.0, abs_tol=1e-6)
            for actual, expected in identities)
        and evidence["reward_mechanism_triggered"]
        is (
            counts["direct"]
            + counts["transition_ff"]
            + timeout_counts["direct"]
            + timeout_counts["transition_ff"]
            > 0
        )
        and math.isclose(
            float(timeout_rewards[
                "credited_no_progress_timeout_failure_reward"
            ]),
            float(timeout_rewards[
                "direct_no_progress_timeout_failure_reward"
            ]) + float(timeout_rewards[
                "transition_ff_no_progress_timeout_failure_reward"
            ]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "terminal-death reward evidence 守恒/机会状态非法",
    )
    for scope in ("direct", "transition_ff", "reset_ff"):
        count = timeout_counts[scope]
        reward = float(timeout_rewards[
            f"{scope}_no_progress_timeout_failure_reward"])
        _require(
            (
                count == 0
                and math.isclose(
                    reward, 0.0, rel_tol=0.0, abs_tol=1e-9)
            )
            or (
                count > 0
                and reward < -cost * count
            ),
            "terminal-death reward evidence no-progress "
            f"{scope} count/cost 不闭合",
        )
    _require(
        math.isclose(
            float(timeout_rewards[
                "manual_ff_no_progress_timeout_failure_reward"
            ]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "terminal-death reward evidence manual timeout reward 非零",
    )
    return evidence


def _fsync_directory(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise CampaignError(f"目录不可打开以 fsync:{path}: {exc}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_directory_durable(path: pathlib.Path) -> None:
    path = pathlib.Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    _require(cursor.is_dir() and not cursor.is_symlink(),
             f"目录祖先非法:{cursor}")
    path.mkdir(parents=True, exist_ok=True)
    _require(path.is_dir() and not path.is_symlink(), f"目标目录非法:{path}")
    for created in reversed(missing):
        _fsync_directory(created)
        _fsync_directory(created.parent)


def _write_bytes_exclusive(
        path: pathlib.Path, payload: bytes, *, mode: int = 0o644) -> None:
    _ensure_directory_durable(path.parent)
    try:
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
    except FileExistsError:
        _require(_stable_read(path) == payload,
                 f"不可变文件已存在但内容漂移:{path}")
        return
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except Exception:
        # A partial exclusive file remains evidence that a supposedly one-shot
        # commit was interrupted; future calls fail closed on its bytes.
        raise


def _write_json_exclusive(path: pathlib.Path, value: dict) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    if path.exists():
        _require(_stable_json(path) == value,
                 f"不可变 JSON 已存在但内容漂移:{path}")
        return
    _write_bytes_exclusive(path, payload)


def _write_json_atomic(path: pathlib.Path, value: dict) -> None:
    _ensure_directory_durable(path.parent)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    _write_bytes_exclusive(tmp, payload)
    os.replace(tmp, path)
    _fsync_directory(path.parent)


def _lock_identity(item: os.stat_result) -> tuple[int, int]:
    return item.st_dev, item.st_ino


@contextlib.contextmanager
def _regular_path_lock(
        path: pathlib.Path, purpose: str, *, nonblocking: bool):
    """Lock one non-symlink regular inode and prove the pathname names that inode."""
    path = pathlib.Path(path)
    _ensure_directory_durable(path.parent)
    _require(hasattr(os, "O_NOFOLLOW"),
             f"{purpose} 所在平台缺 O_NOFOLLOW，拒绝降级")
    flags = (
        os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise CampaignError(f"{purpose} lock 不可安全打开:{path}: {exc}") from exc
    locked = False
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode) and opened.st_nlink == 1,
            f"{purpose} lock 必须是单链接普通文件:{path}",
        )
        operation = fcntl.LOCK_EX | (
            fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(descriptor, operation)
            locked = True
        except BlockingIOError as exc:
            raise CampaignError(f"{purpose} 正由另一进程持有") from exc
        try:
            named = path.lstat()
        except OSError as exc:
            raise CampaignError(
                f"{purpose} lock 加锁后路径消失:{path}: {exc}") from exc
        after = os.fstat(descriptor)
        _require(
            stat.S_ISREG(named.st_mode)
            and not stat.S_ISLNK(named.st_mode)
            and named.st_nlink == after.st_nlink == 1
            and _lock_identity(opened)
            == _lock_identity(after)
            == _lock_identity(named),
            f"{purpose} lock 路径/inode 在加锁期间漂移:{path}",
        )
        os.fsync(descriptor)
        _fsync_directory(path.parent)
        yield
        try:
            final_named = path.lstat()
        except OSError as exc:
            raise CampaignError(
                f"{purpose} lock 持有期间路径消失:{path}: {exc}") from exc
        final_opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(final_named.st_mode)
            and not stat.S_ISLNK(final_named.st_mode)
            and final_named.st_nlink == final_opened.st_nlink == 1
            and _lock_identity(opened)
            == _lock_identity(final_opened)
            == _lock_identity(final_named),
            f"{purpose} lock 路径/inode 在持有期间漂移:{path}",
        )
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def _campaign_lock():
    with _regular_path_lock(
            LOCK_PATH, "R7 campaign", nonblocking=True):
        yield


def _python() -> str:
    executable = ROOT / ".venv" / "bin" / "python"
    _require(executable.is_file(), f"项目 Python 不存在:{executable}")
    return str(executable)


def _invoke(command: list[str], label: str) -> None:
    # Every external process is a point of no return for either a training RNG
    # stream or an evaluation pool.  Fail before spawn if any frozen brain has
    # drifted, even when that particular subprocess would notice only later.
    _frozen_inputs_identity()
    print(f"== {label} ==")
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise CampaignError(f"{label} 退出码 {completed.returncode}")


def _launcher_sha256() -> str:
    return _sha256(pathlib.Path(__file__).resolve())


def _seed_registry_identity() -> dict:
    value = {
        "bc_reserved": [list(pair) for pair in BC_RESERVED_SEED_RANGES],
        "eval_reserved": [list(pair) for pair in EVAL_RESERVED_SEED_RANGES],
        "bc_v1": list(BC_V1_RANGE),
        "bc_v2": list(BC_V2_RANGE),
        "dev_pools": {name: list(seeds) for name, seeds in DEV_POOLS.items()},
        "final_pool": list(FINAL_POOL),
        "development_train_seeds": list(DEVELOPMENT_TRAIN_SEEDS),
        "production_train_seed": PRODUCTION_TRAIN_SEED,
    }
    value["sha256"] = _canonical_sha256(value)
    return value


def _recipe_document() -> dict:
    return {
        "schema_version": "diablogym-r7-recipe/1",
        "campaign_revision": CAMPAIGN_REVISION,
        "baseline_sha256": V28_SHA256,
        "manager_sha256": M29_SHA256,
        "teacher_sha256": KING_SHA256,
        "teacher_actor_sha256": V28_ACTOR_PARAMETER_SHA256,
        "start_steps": START_STEPS,
        "leg_steps": LEG_STEPS,
        "critic_warmup_steps": CRITIC_WARMUP_STEPS,
        "actor_leg_steps": ACTOR_LEG_STEPS,
        "target_steps": TARGET_STEPS,
        "rollout_quantum": ROLLOUT_QUANTUM,
        "train_calls": TRAIN_CALLS,
        "critic_warmup_calls": CRITIC_WARMUP_CALLS,
        "actor_train_calls": ACTOR_TRAIN_CALLS,
        "checkpoint_every_steps": CHECKPOINT_EVERY_STEPS,
        "learning_rate": LEARNING_RATE,
        "ent_coef": ENT_COEF,
        "target_kl": TARGET_KL,
        "distill_beta": DISTILL_BETA,
        "distillation": {
            "initial_beta": DISTILL_BETA,
            "scope": "legacy-root-logits",
            "excluded_actions": [12, 14],
            "anneal_actor_rollouts":
                DISTILL_ANNEAL_ACTOR_ROLLOUTS,
            "schedule": "linear-inclusive-zero-v1",
            "interpretation": "short-lived-v28-root-soft-prior",
        },
        "worker_action14_logit_bonus": ACTION14_LOGIT_BONUS,
        "policy_source_roles": {
            "schema": train_ppo._POLICY_SOURCE_ROLES_SCHEMA,
            "initialization": "resume-checkpoint",
            "bc_v1_direct_policy_uses": [],
            "bc_v1_dataset_uses": [
                "pass-gate",
                "read-only-dry-anchor-instrumentation",
            ],
            "distillation_teacher": "teacher-override",
            "worker_action14_policy_sources": [
                "fixed-logit-prior",
                "native-reward-bound-on-policy-ppo",
            ],
        },
        "actor_migration":
            "copy-v28-root-plus-zero-structured-centered-context-v3",
        "context_architecture": (
            "layout-v1-shared-blocks-centered-context-"
            "product-legacy-zero-output-v2"
        ),
        "critic_reset":
            "structured-layout-v1-orthogonal-value-only-v2",
        "critic_architecture":
            "layout-v1-independent-shared-blocks-centered-value-v2",
        "controller_layout": {
            "schema": DUAL_WORKER_LAYOUT.schema,
            "sha256": DUAL_WORKER_LAYOUT_SHA256,
        },
        "gradient_clip_mode": "separate-root-context-critic-v2",
        "worker_onpolicy_pg_audit_schema":
            WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
        "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout":
            WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        "worker_failure_evidence_schema":
            TERMINAL_DEATH_EVIDENCE_SCHEMA,
        "curriculum": CURRICULUM,
        "game_data_requirement": {
            "mode": FULL_GAME_DATA_MODE,
            "allowed_basenames": list(FULL_GAME_DATA_BASENAMES),
            "sha256": FULL_GAME_DATA_SHA256,
        },
        "drink_sovereignty": False,
        "legacy_policy_observation_view": False,
        "worker_policy_observation_view":
            WORKER_POLICY_OBSERVATION_VIEW,
        "manager_policy_observation_view": "legacy-v3",
        "worker_episode_boundary":
            train_ppo._WORKER_EPISODE_BOUNDARY_V24,
        "worker_window_bootstrap": "next-learning-window",
        "worker_no_progress_timeout":
            dict(train_ppo._WORKER_NO_PROGRESS_TIMEOUT_CONTRACT),
        "gradient_clipping": {
            "mode": "separate-root-context-critic-v2",
            "root_max_norm":
                train_ppo._ALGORITHM_RECIPE["max_grad_norm"] / math.sqrt(2.0),
            "context_max_norm":
                train_ppo._ALGORITHM_RECIPE["max_grad_norm"] / math.sqrt(2.0),
            "combined_actor_max_norm":
                train_ppo._ALGORITHM_RECIPE["max_grad_norm"],
            "critic_max_norm": train_ppo._ALGORITHM_RECIPE["max_grad_norm"],
            "optimizer": "single",
            "trainable_shared_parameters": "forbidden",
        },
        "fast_forward_reward_credit": FAST_FORWARD_CREDIT,
        "recipes": RECIPES,
        "recipe_preference": list(RECIPE_PREFERENCE),
        "development_repetitions": len(DEVELOPMENT_TRAIN_SEEDS),
        "minimum_repetitions_passing_both_pools":
            MIN_REPLICATIONS_PASSING_BOTH_POOLS,
        "production_model_selection": "independent-seed-retrain-not-dev-model",
        "production_training_artifact_scope": "candidate",
        "publication_rule": "atomic-promote-candidate-only-after-final-pass",
        "statistics": {
            "schema": r7_statistics.R7_STATISTICS_SCHEMA,
            "method": r7_statistics.R7_METHOD_REVISION,
            "metric_rules": [
                {
                    "key": rule.key,
                    "direction": rule.direction,
                    "minimum_effect": rule.minimum_effect,
                    "require_sign_test": rule.require_sign_test,
                }
                for rule in METRIC_RULES
            ],
            "familywise_alpha": FAMILYWISE_ALPHA,
            "development_death_margin": DEVELOPMENT_DEATH_MARGIN,
            "final_death_margin": FINAL_DEATH_MARGIN,
        },
        "gear_progression_gate": {
            "schema": GEAR_PROGRESSION_GATE_SCHEMA,
            "minimum_action14_mask_opportunities":
                MIN_ACTION14_MASK_OPPORTUNITIES,
            "required_when_informative": {
                "minimum_requests": 1,
                "minimum_native_successes": 1,
                "minimum_gear_utility_delta": 1,
            },
        },
        "seed_registry": _seed_registry_identity(),
    }


CAMPAIGN_RECIPE = _recipe_document()
CAMPAIGN_RECIPE_SHA256 = _canonical_sha256(CAMPAIGN_RECIPE)


def _require_sha256(path: pathlib.Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    _require(actual == expected,
             f"{label} SHA 漂移:{actual} != {expected}: {path}")
    return actual


def _require_curriculum_discipline() -> None:
    table = train_ppo._parse_dry_curriculum_schedule(CURRICULUM)
    _require(len(table) == TRAIN_CALLS,
             f"R7 curriculum 长度 {len(table)} != train_calls {TRAIN_CALLS}")
    _require(
        all(value == 1.0 for value in table[:CRITIC_WARMUP_CALLS]),
        "R7 critic warmup 课程不是固定 p_skip=1.0",
    )
    actor = table[CRITIC_WARMUP_CALLS:]
    _require(
        len(actor) == ACTOR_TRAIN_CALLS
        and actor[0] == 1.0
        and actor[91] == 0.0
        and all(value == 0.0 for value in actor[92:])
        and all(
            left >= right
            for left, right in zip(actor, actor[1:])
        ),
        "R7 actor curriculum 未精确到达并保持部署 p_skip=0",
    )


def _dry_curriculum_ledger_evidence(path: pathlib.Path) -> dict:
    """Strictly close every R7 rollout against the frozen probability table."""
    path = pathlib.Path(path)
    payload = _stable_read(path)
    lines = payload.splitlines()
    table = train_ppo._parse_dry_curriculum_schedule(CURRICULUM)
    _require(
        len(lines) == TRAIN_CALLS == len(table),
        "R7 dry curriculum ledger 行数不闭合:"
        f"{len(lines)} != {TRAIN_CALLS}",
    )
    expected_keys = {
        "rollout_index",
        "p",
        "num_timesteps",
        "boundary_preapplied",
        "cached_dual_observation_refreshed",
    }
    rows = []
    for index, (line, expected_p) in enumerate(zip(lines, table)):
        _require(
            bool(line.strip()),
            f"R7 dry curriculum ledger 第 {index} 行为空",
        )
        try:
            row = eval_contract.strict_json_loads(line)
        except eval_contract.EvalContractError as exc:
            raise CampaignError(
                f"R7 dry curriculum ledger 第 {index} 行 JSON 非法:{exc}"
            ) from exc
        expected_step = START_STEPS + index * ROLLOUT_QUANTUM
        _require(
            isinstance(row, dict)
            and set(row) == expected_keys
            and _is_plain_int(row["rollout_index"])
            and row["rollout_index"] == index
            and isinstance(row["p"], float)
            and math.isfinite(row["p"])
            and row["p"] == expected_p
            and _is_plain_int(row["num_timesteps"])
            and row["num_timesteps"] == expected_step
            and isinstance(row["boundary_preapplied"], bool)
            and row["boundary_preapplied"] is (index > 0)
            and isinstance(row["cached_dual_observation_refreshed"], bool)
            and row["cached_dual_observation_refreshed"] is True,
            "R7 dry curriculum ledger 与冻结表/全局步不一致:"
            f"index={index},row={row!r},"
            f"expected_p={expected_p},expected_step={expected_step}",
        )
        rows.append(row)
    return {
        "schema_version": DRY_CURRICULUM_LEDGER_SCHEMA,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schedule_sha256": _canonical_sha256(list(table)),
        "rollout_rows": len(rows),
        "first_rollout_index": rows[0]["rollout_index"],
        "last_rollout_index": rows[-1]["rollout_index"],
        "first_global_step": rows[0]["num_timesteps"],
        "last_global_step": rows[-1]["num_timesteps"],
    }


def _full_game_data_identity() -> dict:
    """Bind formal R7 to the one pre-registered full Diablo archive.

    ``eval_contract.game_data_identity`` intentionally remains generic and
    supports ``spawn.mpq``.  Formal R7 is narrower: accepting the shareware
    archive would make its full-game recovery claim self-consistently false.
    """
    try:
        source = eval_contract.game_data_identity()
    except eval_contract.EvalContractError as exc:
        raise CampaignError(f"R7 full-game 主档案不可冻结:{exc}") from exc
    _require(
        isinstance(source, dict)
        and set(source) == {"path", "sha256"}
        and isinstance(source["path"], str)
        and pathlib.Path(source["path"]).is_absolute()
        and _is_sha256(source["sha256"]),
        "R7 full-game 主档案身份字段非法",
    )
    path = pathlib.Path(source["path"])
    identity = {
        "mode": FULL_GAME_DATA_MODE,
        "basename": path.name,
        "path": str(path),
        "sha256": source["sha256"],
    }
    _require(
        identity["basename"] in FULL_GAME_DATA_BASENAMES,
        "R7 formal campaign 禁止 spawn/shareware 主档案:"
        f"{identity['basename']}",
    )
    _require(
        identity["sha256"] == FULL_GAME_DATA_SHA256,
        "R7 full-game DIABDAT 主档案 SHA 漂移:"
        f"{identity['sha256']} != {FULL_GAME_DATA_SHA256}",
    )
    return identity


def _frozen_inputs_identity() -> dict:
    game_data = _full_game_data_identity()
    _require_curriculum_discipline()
    v28_branches = _checkpoint_policy_branch_sha256(
        _stable_read(V28_ZIP))
    _require(
        v28_branches["root_actor_sha256"]
        == V28_ACTOR_PARAMETER_SHA256
        and v28_branches["critic_sha256"]
        == V28_CRITIC_PARAMETER_SHA256,
        "V28 actor/critic tensor identity 漂移",
    )
    teacher_actor_sha256 = _teacher_actor_sha256()
    _require(
        teacher_actor_sha256 == V28_ACTOR_PARAMETER_SHA256,
        "KING 不再是 V28 root 的逐位自锚",
    )
    return {
        "v28_sha256": _require_sha256(V28_ZIP, V28_SHA256, "V28"),
        "manager_sha256": _require_sha256(M29_NPZ, M29_SHA256, "M29"),
        "teacher_sha256": _require_sha256(KING_SD, KING_SHA256, "KING"),
        "teacher_actor_sha256": teacher_actor_sha256,
        "game_data": game_data,
    }


def _implementation_identity() -> dict:
    frozen = _frozen_inputs_identity()
    return {
        "implementation_sha256": train_ppo._implementation_bundle_sha256(),
        "evaluator_sha256": _sha256(TRAIN / "eval_assembled.py"),
        "launcher_sha256": _launcher_sha256(),
        "statistics_sha256": _sha256(TRAIN / "r7_statistics.py"),
        "recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        **frozen,
    }


def _require_seed_discipline() -> None:
    _frozen_inputs_identity()
    _require(BC_V1_RANGE in BC_RESERVED_SEED_RANGES, "R7 BC-v1 未进入拒采表")
    _require(BC_V2_RANGE in BC_RESERVED_SEED_RANGES, "R7 BC-v2 未进入拒采表")
    burned_ranges = (
        (2_100_000, 2_100_128),
        (2_101_000, 2_101_384),
    )
    for burned in burned_ranges:
        _require(
            burned in BC_RESERVED_SEED_RANGES,
            f"R7 已消费 BC 池从拒采表消失:{burned}",
        )
    _require(
        tuple(range(*BC_V1_RANGE))
        == tuple(train_ppo._WORKER_BC_DEMO_SEEDS),
        "R7 BC-v1 range 与 producer/consumer registry 漂移",
    )
    _require(
        tuple(range(*BC_V2_RANGE))
        == tuple(train_ppo._BC_V2_COLLECTION_EPISODES),
        "R7 BC-v2 range 与 producer/consumer registry 漂移",
    )
    _require(EVAL_BANK_RANGE in EVAL_RESERVED_SEED_RANGES, "R7 eval 银行未进入拒采表")
    named = {
        "bc-v1": set(range(*BC_V1_RANGE)),
        "bc-v2": set(range(*BC_V2_RANGE)),
        **{name: set(seeds) for name, seeds in DEV_POOLS.items()},
        "final": set(FINAL_POOL),
    }
    burned_episodes = set().union(
        *(set(range(*registered)) for registered in burned_ranges))
    eval_bank = set(range(*EVAL_BANK_RANGE))
    for scope in ("bc-v1", "bc-v2"):
        _require(
            named[scope].isdisjoint(burned_episodes),
            f"R7 {scope} active pool 与已消费 BC 池重叠",
        )
        _require(
            named[scope].isdisjoint(eval_bank),
            f"R7 {scope} active pool 与 eval 银行重叠",
        )
    names = tuple(named)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            _require(named[left].isdisjoint(named[right]),
                     f"R7 seed pools 重叠:{left}/{right}")
    for seeds in named.values():
        _require(all(is_reserved_train_seed(seed) for seed in seeds),
                 "R7 留出 seed 未被普通训练拒采")
    all_training = (*DEVELOPMENT_TRAIN_SEEDS, PRODUCTION_TRAIN_SEED)
    _require(len(set(all_training)) == len(all_training), "训练 RNG seed 重复")
    _require(
        all(
            not is_reserved_train_seed(seed + rank)
            for seed in all_training
            for rank in range(NUM_ENVS)
        ),
        "训练 RNG seed+rank 撞留出池",
    )


def _initial_state() -> dict:
    return {
        "schema_version": STATE_SCHEMA,
        "campaign_revision": CAMPAIGN_REVISION,
        "recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "launcher_sha256": _launcher_sha256(),
        # Freeze evaluator/statistics/training identity before any registered
        # pool can be opened; baseline/candidate equality alone is too late.
        "implementation": _implementation_identity(),
        "phases": {},
        "terminal_status": None,
    }


def _load_state() -> dict:
    if not STATE_PATH.exists():
        state = _initial_state()
        _write_json_atomic(STATE_PATH, state)
        return state
    state = _stable_json(STATE_PATH)
    _require(
        set(state) == {
            "schema_version", "campaign_revision", "recipe_sha256",
            "launcher_sha256", "implementation", "phases",
            "terminal_status"}
        and state.get("schema_version") == STATE_SCHEMA
        and state.get("campaign_revision") == CAMPAIGN_REVISION
        and state.get("recipe_sha256") == CAMPAIGN_RECIPE_SHA256
        and state.get("launcher_sha256") == _launcher_sha256()
        and state.get("implementation") == _implementation_identity()
        and isinstance(state.get("phases"), dict)
        and all(
            isinstance(name, str) and isinstance(phase, dict)
            and isinstance(phase.get("status"), str)
            for name, phase in state["phases"].items())
        and state.get("terminal_status") in {
            None, "DEVELOPMENT_SCIENTIFIC_FAIL", "FINAL_OPERATIONAL_FAIL",
            "PASS", "FINAL_SCIENTIFIC_FAIL",
        },
        "R7 state 与当前不可变 launcher/recipe 不一致",
    )
    return state


def _set_phase(state: dict, name: str, status: str, **fields) -> None:
    state["phases"][name] = {"status": status, **fields}
    _write_json_atomic(STATE_PATH, state)


def _require_not_terminal(state: dict) -> None:
    _require(
        state.get("terminal_status") is None,
        f"R7 已进入终态:{state.get('terminal_status')}",
    )


def _bc_identity() -> dict:
    report = train_ppo._validate_bc_report(BC_V1_POLICY, "data_gate")
    _require(
        report["implementation_sha256"]
        == train_ppo._implementation_bundle_sha256(),
        "BC-v1 不是当前 R7 implementation 重采件",
    )
    return {
        "policy_sha256": _sha256(BC_V1_POLICY),
        "demos_sha256": _sha256(BC_V1_DEMOS),
        "report_sha256": _sha256(BC_V1_REPORT),
        "implementation_sha256": report["implementation_sha256"],
        "final_pool_sha256": report["final_pool_sha256"],
        "marker_sha256": report["final_holdout_marker_sha256"],
    }


def command_prepare_bc() -> None:
    _require_seed_discipline()
    state = _load_state()
    _require_not_terminal(state)
    try:
        identity = _bc_identity()
    except Exception:
        phase = state["phases"].get("prepare_bc", {})
        _require(not phase, "BC 准备曾点火但没有可复验 PASS 工件；禁止同池重试")
        _set_phase(state, "prepare_bc", "running", attempts=1)
        try:
            _invoke([_python(), str(TRAIN / "bc_worker.py")], "R7 BC-v1 重采")
            identity = _bc_identity()
        except Exception as exc:
            _set_phase(
                state, "prepare_bc", "locked-failed", attempts=1,
                error=f"{type(exc).__name__}: {exc}", retry_forbidden=True,
            )
            raise
    _set_phase(state, "prepare_bc", "complete", **identity)
    print("R7 BC-v1 当前实现绑定重采件 PASS。")


def _run_name(recipe: str, seed: int, scope: str) -> str:
    _require(scope in {"development", "candidate"}, f"未知训练 scope:{scope}")
    suffix = "dev" if scope == "development" else "candidate"
    return f"r7-{recipe}-{suffix}-s{seed}"


def _training_model_path(recipe: str, seed: int, scope: str) -> pathlib.Path:
    filename = {
        "development": "model_development.zip",
        "candidate": "model_candidate.zip",
    }.get(scope)
    _require(filename is not None, f"未知训练 scope:{scope}")
    return TRAIN / "runs" / _run_name(recipe, seed, scope) / filename


def _training_receipt_path(recipe: str, seed: int, scope: str) -> pathlib.Path:
    return (
        TRAIN / "runs" / _run_name(recipe, seed, scope)
        / "r7_training_receipt.json"
    )


def _training_fired_path(recipe: str, seed: int, scope: str) -> pathlib.Path:
    return TRAINING_FIRED_DIR / f"{_run_name(recipe, seed, scope)}.json"


def _training_command(recipe: str, seed: int, scope: str) -> list[str]:
    _require(recipe in RECIPES, f"未知 recipe:{recipe}")
    _require(scope in {"development", "candidate"}, f"未知 scope:{scope}")
    return [
        _python(),
        str(TRAIN / "train_ppo.py"),
        "--worker",
        "--algo", "mppo",
        "--gamma", "1.0",
        "--max-steps", "3000",
        "--num-envs", str(NUM_ENVS),
        "--n-steps", str(N_STEPS),
        "--total-steps", str(LEG_STEPS),
        "--seed", str(seed),
        "--device", "cpu",
        "--run-name", _run_name(recipe, seed, scope),
        "--artifact-scope", scope,
        "--resume-from", str(V28_ZIP),
        "--allow-legacy-resume",
        "--reset-optimizer",
        "--reset-worker-critic",
        "--critic-warmup-steps", str(CRITIC_WARMUP_STEPS),
        "--gradient-clip-mode", "separate-root-context-critic-v2",
        "--manager-npz", str(M29_NPZ),
        "--teacher-override", str(KING_SD),
        "--distill-beta", str(DISTILL_BETA),
        "--distill-anneal-actor-rollouts",
        str(DISTILL_ANNEAL_ACTOR_ROLLOUTS),
        "--worker-action14-logit-bonus",
        str(ACTION14_LOGIT_BONUS),
        "--ckpt-every-steps", str(CHECKPOINT_EVERY_STEPS),
        "--lr", str(LEARNING_RATE),
        "--ent-coef", str(ENT_COEF),
        "--target-kl", str(TARGET_KL),
        "--dry-curriculum-schedule", CURRICULUM,
        "--no-drink-sovereignty",
        "--worker-policy-observation-view",
        WORKER_POLICY_OBSERVATION_VIEW,
        "--worker-fast-forward-reward-credit", FAST_FORWARD_CREDIT,
        "--worker-additional-terminal-death-cost",
        str(RECIPES[recipe]["additional_terminal_death_cost"]),
    ]


def _training_fired_core(
        recipe: str, seed: int, scope: str, bc: dict) -> dict:
    return {
        "schema_version": TRAINING_FIRED_SCHEMA,
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "recipe": recipe,
        "scope": scope,
        "seed": seed,
        "command": _training_command(recipe, seed, scope),
        "attempt": 1,
        "frozen_inputs": _frozen_inputs_identity(),
        "implementation": _implementation_identity(),
        "bc_v1": bc,
    }


def _validate_training_fired(
        recipe: str, seed: int, scope: str, bc: dict) -> dict:
    path = _training_fired_path(recipe, seed, scope)
    record = _stable_json(path)
    core = _training_fired_core(recipe, seed, scope, bc)
    _require(
        set(record) == {*core, "fired_at_ns"}
        and _is_plain_int(record["fired_at_ns"])
        and record["fired_at_ns"] > 0
        and all(record[key] == value for key, value in core.items()),
        f"{recipe}/{seed}/{scope} training fired marker 漂移",
    )
    return record


def _expected_training_contract(scope: str, recipe: str, bc: dict) -> dict:
    canonical = _canonical_migration_evidence(0)
    actor_receipt = canonical["actor_migration"]
    critic_reset = canonical["critic_reset"]
    return {
        "schema_version": 2,
        "contract_revision": train_ppo._CONTRACT_REVISION,
        "implementation_sha256": train_ppo._implementation_bundle_sha256(),
        "mode": "worker",
        "arch": "mlp",
        "max_steps": 3000,
        "num_envs": NUM_ENVS,
        "n_steps": N_STEPS,
        "batch_size": train_ppo._select_batch_size(N_STEPS, NUM_ENVS),
        "gamma": 1.0,
        "learning_rate": LEARNING_RATE,
        "ent_coef": ENT_COEF,
        "distill_beta": DISTILL_BETA,
        "distillation": {
            "initial_beta": DISTILL_BETA,
            "scope": "legacy-root-logits",
            "excluded_actions": [12, 14],
            "anneal_actor_rollouts":
                DISTILL_ANNEAL_ACTOR_ROLLOUTS,
            "schedule": "linear-inclusive-zero-v1",
        },
        "teacher_sha256": KING_SHA256,
        "calib_record_only": False,
        "device": "cpu",
        "skip_dry": False,
        "drink_sovereignty": False,
        "legacy_policy_observation_view": False,
        "worker_policy_observation_view":
            WORKER_POLICY_OBSERVATION_VIEW,
        "worker_action14_logit_bonus": ACTION14_LOGIT_BONUS,
        "manager_policy_observation_view": "legacy-v3",
        "worker_episode_boundary":
            train_ppo._WORKER_EPISODE_BOUNDARY_V24,
        "worker_window_bootstrap": "next-learning-window",
        "worker_no_progress_timeout":
            dict(train_ppo._WORKER_NO_PROGRESS_TIMEOUT_CONTRACT),
        "gradient_clipping":
            train_ppo._root_context_critic_gradient_clipping(
                train_ppo._ALGORITHM_RECIPE["max_grad_norm"]),
        "actor_migration": {
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
                "output":
                    "exact-zero-disabled-through-critic-warmup",
            },
            "actor_context_excluded_observation_features":
                actor_receipt[
                    "actor_context_excluded_observation_features"],
        },
        "critic_migration": {
            "method": critic_reset["method"],
            "critic_architecture":
                critic_reset["critic_architecture"],
            "controller_layout_schema":
                critic_reset["controller_layout_schema"],
            "controller_layout_sha256":
                critic_reset["controller_layout_sha256"],
            "critic_parameter_tensors":
                critic_reset["critic_parameter_tensors"],
            "critic_parameter_count":
                critic_reset["critic_parameter_count"],
            "source_checkpoint_sha256":
                critic_reset["source_checkpoint_sha256"],
            "warmup_steps": CRITIC_WARMUP_STEPS,
            "gradient_clip_mode": "separate-root-context-critic-v2",
            "source_actor_sha256":
                critic_reset["source_actor_sha256"],
            "worker_onpolicy_pg_audit_schema":
                WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
            "worker_onpolicy_pg_min_optimizer_steps_per_joint_rollout":
                WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT,
        },
        "worker_fast_forward_reward_credit": FAST_FORWARD_CREDIT,
        "worker_additional_terminal_death_cost":
            RECIPES[recipe]["additional_terminal_death_cost"],
        "artifact_scope": scope,
        "dry_curriculum": {"schedule": CURRICULUM},
        "bc_aux": "disabled",
        "manager_npz_sha256": M29_SHA256,
        "worker_npz_sha256": None,
        "demos_sha256": bc["demos_sha256"],
        "policy_source_roles": {
            "schema": train_ppo._POLICY_SOURCE_ROLES_SCHEMA,
            "initialization": "resume-checkpoint",
            "bc_v1_direct_policy_uses": [],
            "bc_v1_dataset_uses": [
                "pass-gate",
                "read-only-dry-anchor-instrumentation",
            ],
            "distillation_teacher": "teacher-override",
            "worker_action14_policy_sources": [
                "fixed-logit-prior",
                "native-reward-bound-on-policy-ppo",
            ],
        },
        "observation_shape": [DUAL_WORKER_OBSERVATION_DIM],
        "action_n": 15,
        "runtime_versions": dict(train_ppo._RUNTIME_VERSIONS),
        "algorithm_recipe": {
            **train_ppo._ALGORITHM_RECIPE,
            "target_kl": TARGET_KL,
        },
    }


def _require_config_values(config: dict, expected: dict, label: str) -> None:
    _require(isinstance(config, dict), f"{label} config 不是对象")
    differences = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    _require(not differences, f"{label} config 漂移:{differences}")


def _worker_onpolicy_pg_evidence(
        checkpoint: dict, label: str, *,
        expected_additional_terminal_death_cost: float) -> dict:
    """Validate every joint rollout and return a compact signed summary."""
    _require(
        _finite_json_number(expected_additional_terminal_death_cost)
        and float(expected_additional_terminal_death_cost) >= 0.0,
        f"{label} formal Worker timeout additional cost 非法",
    )
    expected_cost = float(expected_additional_terminal_death_cost)
    receipts = checkpoint.get(
        "_worker_onpolicy_pg_rollout_receipts")
    joint_rollouts = checkpoint.get(
        "_worker_onpolicy_pg_joint_rollouts")
    qualifying_rollouts = checkpoint.get(
        "_worker_onpolicy_pg_qualifying_rollouts")
    expected_endpoints = list(range(
        START_STEPS + CRITIC_WARMUP_STEPS + ROLLOUT_QUANTUM,
        TARGET_STEPS + 1,
        ROLLOUT_QUANTUM,
    ))
    _require(
        checkpoint.get("_worker_onpolicy_pg_audit_required") is True
        and isinstance(receipts, list)
        and joint_rollouts == ACTOR_TRAIN_CALLS
        and len(receipts) == ACTOR_TRAIN_CALLS
        and [receipt.get("rollout_end_timesteps")
             if isinstance(receipt, dict) else None
             for receipt in receipts] == expected_endpoints
        and all(validate_worker_onpolicy_pg_receipt(
            receipt, expected_samples=ROLLOUT_QUANTUM)
            for receipt in receipts)
        and qualifying_rollouts
        == sum(receipt["qualifies"] is True for receipt in receipts)
        and _is_plain_int(qualifying_rollouts)
        and 3 <= qualifying_rollouts <= ACTOR_TRAIN_CALLS
        and any(
            receipt["qualifies"] is True
            for receipt in receipts[:8]
        )
        and any(
            receipt["qualifies"] is True
            for receipt in receipts[
                DISTILL_ANNEAL_ACTOR_ROLLOUTS:
            ]
        )
        and sum(receipt["optimizer_steps"] for receipt in receipts)
        == checkpoint.get("_actor_optimizer_steps_completed")
        and sum(
            receipt["optimizer_steps"] for receipt in receipts
        ) >= (
            WORKER_ONPOLICY_PG_MIN_OPTIMIZER_STEPS_PER_JOINT_ROLLOUT
            * ACTOR_TRAIN_CALLS
        ),
        f"{label} formal Worker on-policy PG evidence 未闭合",
    )
    qualifying = [
        receipt for receipt in receipts if receipt["qualifies"]]
    timeout_samples = sum(
        receipt["no_progress_timeout_samples"]
        for receipt in receipts
    )
    timeout_base_sum = math.fsum(
        receipt["no_progress_timeout_base_failure_reward_sum"]
        for receipt in receipts
    )
    timeout_additional_sum = math.fsum(
        receipt["no_progress_timeout_additional_failure_reward_sum"]
        for receipt in receipts
    )
    timeout_total_sum = math.fsum(
        receipt["no_progress_timeout_failure_reward_sum"]
        for receipt in receipts
    )
    _require(
        _is_plain_int(timeout_samples)
        and 0 <= timeout_samples
        <= ACTOR_TRAIN_CALLS * ROLLOUT_QUANTUM
        and all(math.isfinite(value) for value in (
            timeout_base_sum,
            timeout_additional_sum,
            timeout_total_sum,
        ))
        and math.isclose(
            timeout_additional_sum,
            -expected_cost * timeout_samples,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            timeout_total_sum,
            timeout_base_sum + timeout_additional_sum,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and (
            (
                timeout_base_sum == 0.0
                and timeout_additional_sum == 0.0
                and timeout_total_sum == 0.0
            )
            if timeout_samples == 0
            else (
                timeout_base_sum < 0.0
                and timeout_total_sum < timeout_additional_sum
            )
        ),
        f"{label} formal Worker no-progress timeout 分账未闭合",
    )
    return {
        "schema": WORKER_ONPOLICY_PG_AUDIT_SCHEMA,
        "joint_rollouts": joint_rollouts,
        "qualifying_rollouts": qualifying_rollouts,
        "configured_additional_terminal_death_cost": expected_cost,
        "first_rollout_end_timesteps":
            receipts[0]["rollout_end_timesteps"],
        "last_rollout_end_timesteps":
            receipts[-1]["rollout_end_timesteps"],
        "first_qualifying_rollout_end_timesteps":
            qualifying[0]["rollout_end_timesteps"],
        "transition_reward_nonzero_samples":
            sum(receipt["transition_reward_nonzero_samples"]
                for receipt in receipts),
        "no_progress_timeout_samples": timeout_samples,
        "no_progress_timeout_base_failure_reward_sum":
            timeout_base_sum,
        "no_progress_timeout_additional_failure_reward_sum":
            timeout_additional_sum,
        "no_progress_timeout_failure_reward_sum":
            timeout_total_sum,
        "requested_action_counts": [
            sum(receipt["requested_action_counts"][action]
                for receipt in receipts)
            for action in range(15)
        ],
        "executed_action_counts": [
            sum(receipt["executed_action_counts"][action]
                for receipt in receipts)
            for action in range(15)
        ],
        "combat_effect_samples":
            sum(receipt["combat_effect_samples"]
                for receipt in receipts),
        "combat_transition_reward_nonzero_samples":
            sum(
                receipt[
                    "combat_transition_reward_nonzero_samples"
                ]
                for receipt in receipts
            ),
        "combat_transition_reward_positive_samples":
            sum(
                receipt[
                    "combat_transition_reward_positive_samples"
                ]
                for receipt in receipts
            ),
        "combat_transition_reward_sum":
            math.fsum(
                receipt["combat_transition_reward_sum"]
                for receipt in receipts
            ),
        "combat_positive_advantage_samples":
            sum(receipt["combat_positive_advantage_samples"]
                for receipt in receipts),
        "combat_reward_centered_actor_grad_norm_max":
            max(
                receipt[
                    "combat_reward_centered_actor_grad_norm"
                ]
                for receipt in receipts
            ),
        "combat_reward_centered_context_encoder_grad_norm_max":
            max(
                receipt[
                    "combat_reward_centered_context_encoder_grad_norm"
                ]
                for receipt in receipts
            ),
        "combat_reward_centered_context_interaction_grad_norm_max":
            max(
                receipt[
                    "combat_reward_centered_context_interaction_grad_norm"
                ]
                for receipt in receipts
            ),
        "reward_centered_actor_grad_norm_max":
            max(receipt["reward_centered_actor_grad_norm"]
                for receipt in receipts),
        "reward_centered_context_encoder_grad_norm_max":
            max(
                receipt[
                    "reward_centered_context_encoder_grad_norm"
                ]
                for receipt in receipts
            ),
        "reward_centered_context_interaction_grad_norm_max":
            max(
                receipt[
                    "reward_centered_context_interaction_grad_norm"
                ]
                for receipt in receipts
            ),
        "pure_ppo_actor_grad_norm_max":
            max(receipt["pure_ppo_actor_grad_norm_max"]
                for receipt in receipts),
        "pure_ppo_root_grad_norm_max":
            max(receipt["pure_ppo_root_grad_norm_max"]
                for receipt in receipts),
        "pure_ppo_context_encoder_grad_norm_max":
            max(
                receipt[
                    "pure_ppo_context_encoder_grad_norm_max"
                ]
                for receipt in receipts
            ),
        "pure_ppo_context_interaction_grad_norm_max":
            max(
                receipt[
                    "pure_ppo_context_interaction_grad_norm_max"
                ]
                for receipt in receipts
            ),
        "combined_context_on_pure_ppo_projection_min":
            min(
                receipt[
                    "combined_context_on_pure_ppo_projection"
                ]
                for receipt in qualifying
            ),
        "combined_root_on_pure_ppo_projection_min":
            min(
                receipt[
                    "combined_root_on_pure_ppo_projection"
                ]
                for receipt in qualifying
            ),
        "pure_ppo_actor_on_combat_reward_projection_min":
            min(
                receipt[
                    "pure_ppo_actor_on_combat_reward_projection"
                ]
                for receipt in qualifying
            ),
        "pure_ppo_root_on_combat_reward_projection_min":
            min(
                receipt[
                    "pure_ppo_root_on_combat_reward_projection"
                ]
                for receipt in qualifying
            ),
        "pure_ppo_context_on_combat_reward_projection_min":
            min(
                receipt[
                    "pure_ppo_context_on_combat_reward_projection"
                ]
                for receipt in qualifying
            ),
        "optimizer_delta_actor_on_combat_reward_descent_projection_min":
            min(
                receipt[
                    "optimizer_delta_actor_on_combat_reward_descent_projection"
                ]
                for receipt in qualifying
            ),
        "optimizer_delta_root_on_combat_reward_descent_projection_min":
            min(
                receipt[
                    "optimizer_delta_root_on_combat_reward_descent_projection"
                ]
                for receipt in qualifying
            ),
        "optimizer_delta_context_on_combat_reward_descent_projection_min":
            min(
                receipt[
                    "optimizer_delta_context_on_combat_reward_descent_projection"
                ]
                for receipt in qualifying
            ),
        "optimizer_delta_actor_l2_qualifying_min":
            min(
                receipt["optimizer_delta_actor_l2"]
                for receipt in qualifying
            ),
        "optimizer_delta_root_l2_qualifying_min":
            min(
                receipt["optimizer_delta_root_l2"]
                for receipt in qualifying
            ),
        "optimizer_delta_context_l2_qualifying_min":
            min(
                receipt["optimizer_delta_context_l2"]
                for receipt in qualifying
            ),
        "optimizer_delta_actor_on_combat_reward_descent_cosine_qualifying_min":
            min(
                receipt[
                    "optimizer_delta_actor_on_combat_reward_descent_cosine"
                ]
                for receipt in qualifying
            ),
        "optimizer_delta_root_on_combat_reward_descent_cosine_qualifying_min":
            min(
                receipt[
                    "optimizer_delta_root_on_combat_reward_descent_cosine"
                ]
                for receipt in qualifying
            ),
        "optimizer_delta_context_on_combat_reward_descent_cosine_qualifying_min":
            min(
                receipt[
                    "optimizer_delta_context_on_combat_reward_descent_cosine"
                ]
                for receipt in qualifying
            ),
        "optimizer_steps":
            sum(receipt["optimizer_steps"] for receipt in receipts),
        "rollout_receipts_sha256":
            _canonical_sha256(receipts),
    }


def _training_artifact_evidence(
        recipe: str, seed: int, scope: str) -> dict:
    bc = _bc_identity()
    _validate_training_fired(recipe, seed, scope, bc)
    run_dir = TRAIN / "runs" / _run_name(recipe, seed, scope)
    model = _training_model_path(recipe, seed, scope)
    status_path = run_dir / "status.json"
    sentinel_path = run_dir / "sentinel.jsonl"
    dry_curriculum_path = run_dir / "dry_curriculum.jsonl"
    status_payload = _stable_read(status_path)
    try:
        status = eval_contract.strict_json_loads(status_payload)
    except eval_contract.EvalContractError as exc:
        raise CampaignError(f"训练 status 非法:{status_path}: {exc}") from exc
    _require(isinstance(status, dict), f"训练 status 顶层不是对象:{status_path}")
    status_sha256 = hashlib.sha256(status_payload).hexdigest()
    model_payload = _stable_read(model)
    model_sha256 = hashlib.sha256(model_payload).hexdigest()
    expected_status = {
        "development": "DEVELOPMENT_ONLY",
        "candidate": "PRODUCTION_CANDIDATE",
    }[scope]
    _require(
        status.get("publication_status") == expected_status
        and status.get("training_ended") is True
        and status.get("target_reached") is True
        and status.get("run") == _run_name(recipe, seed, scope)
        and status.get("total_steps") == TARGET_STEPS
        and status.get("target_steps") == TARGET_STEPS
        and status.get("start_steps") == START_STEPS
        and status.get("leg_steps") == LEG_STEPS
        and status.get("leg_target_steps") == LEG_STEPS
        and status.get("rollout_full") is True
        and status.get("model_sha256") == model_sha256,
        f"{recipe}/{seed}/{scope} status 未闭合",
    )
    _require(
        status.get("model_published") is False
        and status.get("model_development_only")
        is (scope == "development")
        and status.get("model_production_candidate")
        is (scope == "candidate"),
        f"{recipe}/{seed}/{scope} publication flags 未闭合",
    )
    config = status.get("config")
    expected_contract = _expected_training_contract(scope, recipe, bc)
    _require(
        isinstance(config, dict)
        and _is_plain_int(config.get("dry_curriculum_start_index"))
        and config["dry_curriculum_start_index"] == 0
        and isinstance(
            config.get("dry_curriculum_start_probability"), float)
        and config["dry_curriculum_start_probability"] == 1.0,
        f"{recipe}/{seed}/{scope} dry curriculum 配置起点未严格闭合",
    )
    _require_config_values(
        config,
        {
            "total_steps": LEG_STEPS,
            "num_envs": NUM_ENVS,
            "device": "cpu",
            "lr": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": train_ppo._select_batch_size(N_STEPS, NUM_ENVS),
            "max_steps": 3000,
            "seed": seed,
            "algo": "MaskablePPO/MlpPolicy(gear-key mask)",
            "gamma": 1.0,
            "worker": True,
            "options": False,
            "flat_clock": False,
            "deep": True,
            "death_ladder": True,
            "skip_dry": False,
            "drink_sovereignty": False,
            "legacy_policy_observation_view": False,
            "worker_policy_observation_view":
                WORKER_POLICY_OBSERVATION_VIEW,
            "worker_action14_logit_bonus": ACTION14_LOGIT_BONUS,
            "manager_policy_observation_view": "legacy-v3",
            "worker_fast_forward_reward_credit": FAST_FORWARD_CREDIT,
            "worker_additional_terminal_death_cost":
                RECIPES[recipe]["additional_terminal_death_cost"],
            "artifact_scope": scope,
            "dry_curriculum": {"schedule": CURRICULUM},
            "dry_curriculum_start_index": 0,
            "dry_curriculum_start_probability": 1.0,
            "bc_aux": "disabled",
            "bc_aux_liveness_preflight": "disabled",
            "ent_coef": ENT_COEF,
            "target_kl": TARGET_KL,
            "reset_optimizer": True,
            "reset_worker_critic": True,
            "critic_warmup_steps": CRITIC_WARMUP_STEPS,
            "gradient_clip_mode": "separate-root-context-critic-v2",
            "freeze_policy_steps": 0,
            "distill_beta": DISTILL_BETA,
            "distill_anneal_actor_rollouts":
                DISTILL_ANNEAL_ACTOR_ROLLOUTS,
            "ckpt_every_steps": CHECKPOINT_EVERY_STEPS,
            "resume_from": str(V28_ZIP),
            "resume_checkpoint_sha256": V28_SHA256,
            "manager_npz": str(M29_NPZ),
            "teacher_override": str(KING_SD),
            "allow_manager_change": False,
            "allow_legacy_resume": True,
            "start_steps": START_STEPS,
            "target_global_steps": TARGET_STEPS,
            "training_contract": expected_contract,
            "teacher_sha256": KING_SHA256,
            "invocation_argv":
                _training_command(recipe, seed, scope)[2:],
        },
        f"{recipe}/{seed}/{scope}",
    )
    checkpoint = train_ppo._validate_checkpoint_bytes(
        model_payload, str(model), require_leashed=True)
    contract = checkpoint.get("diablogym_contract")
    canonical_migration = _canonical_migration_evidence(seed)
    actor_migration_receipt = checkpoint.get(
        "_actor_migration_receipt")
    _validate_actor_migration_receipt(
        actor_migration_receipt,
        canonical_receipt=canonical_migration["actor_migration"],
    )
    _require(
        config.get("actor_migration_receipt")
        == actor_migration_receipt,
        f"{recipe}/{seed}/{scope} config/ZIP actor migration receipt 不一致",
    )
    migration_receipt = checkpoint.get("_critic_migration_receipt")
    _validate_critic_migration_receipt(
        migration_receipt, seed=seed,
        actor_receipt=actor_migration_receipt,
        canonical_evidence=canonical_migration)
    _require(
        config.get("critic_migration_receipt") == migration_receipt,
        f"{recipe}/{seed}/{scope} config/ZIP critic migration receipt 不一致",
    )
    worker_pg_evidence = _worker_onpolicy_pg_evidence(
        checkpoint,
        f"{recipe}/{seed}/{scope}",
        expected_additional_terminal_death_cost=(
            RECIPES[recipe]["additional_terminal_death_cost"]),
    )
    final_branches = _checkpoint_policy_branch_sha256(
        model_payload, require_asymmetric=True)
    _require(
        final_branches["full_actor_sha256"]
        != actor_migration_receipt["migrated_actor_sha256"]
        and final_branches["context_sha256"]
        != actor_migration_receipt["context_sha256"]
        and final_branches["context_encoder_sha256"]
        != actor_migration_receipt["context_encoder_sha256"]
        and final_branches["context_interaction_sha256"]
        != actor_migration_receipt["context_interaction_sha256"]
        and final_branches["context_output_sha256"]
        != actor_migration_receipt["context_output_sha256"]
        and final_branches["critic_sha256"]
        != migration_receipt["critic_sha256_after"]
        and final_branches["context_parameter_count"]
        == actor_migration_receipt["context_parameter_count"]
        and final_branches["critic_parameter_count"]
        == migration_receipt["critic_parameter_count"],
        f"{recipe}/{seed}/{scope} optimizer 虽计步但"
        " context actor 或 critic 权重未学习",
    )
    try:
        train_ppo._validate_current_dual_worker_contract(contract)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CampaignError(
            f"{recipe}/{seed}/{scope} rev"
            f"{train_ppo._CONTRACT_REVISION} 完整 contract 未闭合"
        ) from exc
    policy_class_record = checkpoint.get("policy_class")
    _require(
        checkpoint.get("num_timesteps") == TARGET_STEPS
        and checkpoint.get("seed") == seed
        and isinstance(policy_class_record, dict)
        and policy_class_record.get("__module__") == "leashed_ppo"
        and checkpoint.get("_last_completed_ppo_rollout_steps")
        == TARGET_STEPS
        and _is_plain_int(checkpoint.get("_ppo_optimizer_steps_completed"))
        and checkpoint["_ppo_optimizer_steps_completed"] > 0
        and checkpoint.get("gradient_clip_mode")
        == "separate-root-context-critic-v2"
        and checkpoint.get("_critic_warmup_start_timesteps") == START_STEPS
        and checkpoint.get("_critic_warmup_until_timesteps")
        == START_STEPS + CRITIC_WARMUP_STEPS
        and checkpoint.get("_critic_warmup_expected_rollouts")
        == CRITIC_WARMUP_CALLS
        and checkpoint.get("_critic_warmup_rollouts_completed")
        == CRITIC_WARMUP_CALLS
        and checkpoint.get("_critic_warmup_completed") is True
        and _is_plain_int(
            checkpoint.get("_critic_warmup_optimizer_steps_completed"))
        and checkpoint["_critic_warmup_optimizer_steps_completed"] > 0
        and _is_plain_int(checkpoint.get("_actor_optimizer_steps_completed"))
        and checkpoint["_actor_optimizer_steps_completed"] > 0
        and checkpoint.get("distill_anneal_actor_rollouts")
        == DISTILL_ANNEAL_ACTOR_ROLLOUTS
        and checkpoint.get("_distill_actor_rollouts_completed")
        == ACTOR_TRAIN_CALLS
        and checkpoint.get("_last_effective_distill_beta") == 0.0
        and contract == expected_contract
        and contract == config["training_contract"],
        f"{recipe}/{seed}/{scope} checkpoint/contract 未闭合",
    )
    expected_model_name = model.name
    for name in (
            "model_development.zip", "model_candidate.zip", "model_final.zip"):
        wrong = run_dir / name
        if name != expected_model_name:
            _require(not wrong.exists(),
                     f"{scope} run 产生了错误 scope 工件:{wrong}")
    identity = _implementation_identity()
    dry_curriculum_ledger = _dry_curriculum_ledger_evidence(
        dry_curriculum_path)
    terminal_death_evidence = _terminal_death_reward_evidence(
        sentinel_path, recipe=recipe, target_step=TARGET_STEPS)
    pg_timeout_samples = worker_pg_evidence[
        "no_progress_timeout_samples"]
    pg_timeout_total = worker_pg_evidence[
        "no_progress_timeout_failure_reward_sum"]
    sentinel_learning_timeout_samples = sum(
        terminal_death_evidence["no_progress_timeout_counts"][scope]
        for scope in ("direct", "transition_ff")
    )
    sentinel_learning_timeout_total = (
        terminal_death_evidence["no_progress_timeout_rewards"][
            "credited_no_progress_timeout_failure_reward"
        ]
    )
    _require(
        pg_timeout_samples <= sentinel_learning_timeout_samples
        and float(sentinel_learning_timeout_total)
        <= float(pg_timeout_total) + 1e-6
        and (
            pg_timeout_samples != sentinel_learning_timeout_samples
            or math.isclose(
                float(sentinel_learning_timeout_total),
                float(pg_timeout_total),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ),
        f"{recipe}/{seed}/{scope} formal PG/sentinel "
        "no-progress timeout 证据不闭合",
    )
    _require(
        _sha256(model) == model_sha256
        and _sha256(status_path) == status_sha256
        and _sha256(dry_curriculum_path)
        == dry_curriculum_ledger["sha256"]
        and _sha256(sentinel_path)
        == terminal_death_evidence["sentinel_sha256"],
        f"{recipe}/{seed}/{scope} 工件验证期间发生替换",
    )
    return {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "recipe": recipe,
        "scope": scope,
        "seed": seed,
        "command": _training_command(recipe, seed, scope),
        "model_path": str(model),
        "model_sha256": model_sha256,
        "status_sha256": status_sha256,
        "training_contract_sha256":
            train_ppo._canonical_json_sha256(contract),
        "final_actor_sha256": final_branches["full_actor_sha256"],
        "final_root_actor_sha256":
            final_branches["root_actor_sha256"],
        "final_critic_sha256": final_branches["critic_sha256"],
        "final_context_sha256":
            final_branches["context_sha256"],
        "final_context_encoder_sha256":
            final_branches["context_encoder_sha256"],
        "final_context_interaction_sha256":
            final_branches["context_interaction_sha256"],
        "final_context_output_sha256":
            final_branches["context_output_sha256"],
        "final_context_nonzero":
            final_branches["context_nonzero"],
        "final_context_parameter_count":
            final_branches["context_parameter_count"],
        "final_critic_parameter_count":
            final_branches["critic_parameter_count"],
        "final_context_l2_norm":
            final_branches["context_l2_norm"],
        "final_context_action_effect_nonzero":
            final_branches["context_action_effect_nonzero"],
        "final_context_action_effect_l2_norm":
            final_branches["context_action_effect_l2_norm"],
        "final_interaction_action_effect_nonzero":
            final_branches["interaction_action_effect_nonzero"],
        "final_interaction_action_effect_l2_norm":
            final_branches["interaction_action_effect_l2_norm"],
        "worker_onpolicy_pg_evidence": worker_pg_evidence,
        "dry_curriculum_ledger": dry_curriculum_ledger,
        "terminal_death_reward_evidence": terminal_death_evidence,
        "training_fired_sha256":
            _sha256(_training_fired_path(recipe, seed, scope)),
        "implementation": identity,
        "bc_v1": bc,
    }


def _capture_training_artifact(
        recipe: str, seed: int, scope: str) -> dict:
    receipt = _training_artifact_evidence(recipe, seed, scope)
    _write_json_exclusive(
        _training_receipt_path(recipe, seed, scope), receipt)
    receipt["receipt_sha256"] = _sha256(
        _training_receipt_path(recipe, seed, scope))
    return receipt


def _validate_training_artifact(
        recipe: str, seed: int, scope: str) -> dict:
    receipt_path = _training_receipt_path(recipe, seed, scope)
    _require(receipt_path.exists(),
             f"{recipe}/{seed}/{scope} 缺训练回执:{receipt_path}")
    expected = _training_artifact_evidence(recipe, seed, scope)
    _require(
        _stable_json(receipt_path) == expected,
        f"{recipe}/{seed}/{scope} 训练回执与冻结工件漂移",
    )
    expected["receipt_sha256"] = _sha256(receipt_path)
    return expected


def _run_training_once(recipe: str, seed: int, scope: str) -> dict:
    receipt_path = _training_receipt_path(recipe, seed, scope)
    if receipt_path.exists():
        return _validate_training_artifact(recipe, seed, scope)
    bc = _bc_identity()
    fired_path = _training_fired_path(recipe, seed, scope)
    run_dir = receipt_path.parent
    if fired_path.exists():
        # A process may die after the trainer atomically committed model,
        # status, sentinel and curriculum ledger but before this launcher
        # wrote its derived receipt.  Never re-run that seed.  Recompute every
        # frozen invariant from the completed artifacts and adopt them only if
        # the normal evidence function passes in full; partial output remains
        # permanently fail-closed.
        _require(
            run_dir.is_dir(),
            f"{recipe}/{seed}/{scope} 已点火但训练目录缺失；禁止重发",
        )
        return _capture_training_artifact(recipe, seed, scope)
    _require(
        not run_dir.exists(),
        f"{recipe}/{seed}/{scope} 有未登记残件；禁止静默重发:{run_dir}",
    )
    fired = {
        **_training_fired_core(recipe, seed, scope, bc),
        "fired_at_ns": time.time_ns(),
    }
    _write_json_exclusive(fired_path, fired)
    _invoke(
        _training_command(recipe, seed, scope),
        f"R7 {scope} train {recipe} seed={seed}",
    )
    _validate_training_fired(recipe, seed, scope, bc)
    return _capture_training_artifact(recipe, seed, scope)


def command_train_development() -> None:
    _require_seed_discipline()
    state = _load_state()
    _require_not_terminal(state)
    bc = _bc_identity()
    _require(
        state["phases"].get("prepare_bc", {}).get("status") == "complete",
        "须先 prepare-bc",
    )
    phase = state["phases"].get("train_development", {})
    _require(phase.get("status") != "locked-failed", "开发训练已 locked-failed")
    records = dict(phase.get("artifacts", {}))
    _set_phase(
        state, "train_development", "running",
        artifacts=records, bc=bc,
    )
    try:
        for recipe in RECIPE_PREFERENCE:
            for seed in DEVELOPMENT_TRAIN_SEEDS:
                key = f"{recipe}:{seed}"
                record = _run_training_once(recipe, seed, "development")
                records[key] = record
                _set_phase(
                    state, "train_development", "running",
                    artifacts=records, bc=bc,
                )
    except Exception as exc:
        _set_phase(
            state, "train_development", "locked-failed",
            artifacts=records, bc=bc, retry_forbidden=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    _set_phase(
        state, "train_development", "complete",
        artifacts=records, bc=bc,
    )
    print("R7 两配方 × 三固定 RNG 的 DEVELOPMENT_ONLY cohort 已冻结。")


def _eval_tag(pool: str, arm: str, recipe: str | None = None,
              seed: int | None = None) -> str:
    if pool == "final":
        return f"official-r7-final-{arm}-2120000"
    if arm == "baseline":
        return f"r7-{pool}-baseline-v28"
    _require(recipe is not None and seed is not None, "开发 candidate tag 缺身份")
    return f"r7-{pool}-{recipe}-s{seed}"


def _eval_path(tag: str) -> pathlib.Path:
    return EVAL_DIR / f"{tag}.json"


def _eval_lock_path(tag: str) -> pathlib.Path:
    path = _eval_path(tag)
    return path.with_name(f".{path.name}.lock")


def _eval_fired_path(tag: str) -> pathlib.Path:
    return CONTROL_DIR / "eval-fired" / f"{tag}.json"


def _eval_attestation_path(tag: str) -> pathlib.Path:
    return EVAL_ATTESTATION_DIR / f"{tag}.json"


def _seed_arg(seeds: tuple[int, ...]) -> str:
    _require(
        seeds == tuple(range(seeds[0], seeds[-1] + 1)),
        "eval seeds 必须为连续升序范围",
    )
    return f"{seeds[0]}-{seeds[-1]}"


def _eval_command(
        worker: pathlib.Path, seeds: tuple[int, ...], tag: str,
        manager: pathlib.Path = M29_NPZ) -> list[str]:
    return [
        _python(),
        str(TRAIN / "eval_assembled.py"),
        "--worker", str(worker),
        "--manager-npz", str(manager),
        "--manager-policy-observation-view", "legacy-v3",
        "--seeds", _seed_arg(seeds),
        "--tag", tag,
    ]


def _stage_eval_file(
        source: pathlib.Path, destination: pathlib.Path,
        *, expected_sha256: str | None = None) -> str:
    payload = _stable_read(source)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        _require(
            digest == expected_sha256,
            f"eval staging 源 SHA 漂移:{source}:{digest} != {expected_sha256}",
        )
    _write_bytes_exclusive(destination, payload, mode=0o444)
    try:
        os.chmod(destination, 0o444)
    except OSError as exc:
        raise CampaignError(f"eval staging 无法设为只读:{destination}: {exc}") from exc
    _fsync_directory(destination.parent)
    _require(_sha256(destination) == digest,
             f"eval staging 副本 SHA 漂移:{destination}")
    return digest


def _prepare_eval_launch(
        worker: pathlib.Path, seeds: tuple[int, ...], tag: str, *,
        training_receipt: dict | None = None,
        final_bind_sha256: str | None = None,
        require_existing_staging: bool = False) -> dict:
    _frozen_inputs_identity()
    worker = pathlib.Path(worker)
    source_sha = _sha256(worker)
    # Baseline launches have no training receipt, so their immutable V28
    # identity must be rebound at the exact source read used for staging.
    # Merely checking V28 inside _frozen_inputs_identity() above leaves a
    # replace-between-reads gap: a foreign checkpoint could be hashed/staged
    # and the registered V28 restored before the later implementation audit.
    if (
        os.path.abspath(os.fspath(worker))
        == os.path.abspath(os.fspath(V28_ZIP))
    ):
        _require(
            source_sha == V28_SHA256,
            f"eval {tag} V28 baseline SHA 漂移:"
            f"{source_sha} != {V28_SHA256}",
        )
    receipt_sha = None
    if training_receipt is not None:
        _require(
            isinstance(training_receipt, dict)
            and training_receipt.get("model_path") == str(worker)
            and training_receipt.get("model_sha256") == source_sha
            and _is_sha256(training_receipt.get("receipt_sha256")),
            f"eval {tag} 未绑定精确训练回执",
        )
        receipt_sha = training_receipt["receipt_sha256"]
    if final_bind_sha256 is not None:
        _require(_is_sha256(final_bind_sha256),
                 f"eval {tag} final bind SHA 非法")
    stage_dir = EVAL_INPUT_DIR / tag
    staged_worker = stage_dir / "worker.zip"
    staged_manager = stage_dir / "manager.npz"
    if require_existing_staging:
        _require(
            staged_worker.exists() and staged_manager.exists(),
            f"eval {tag} 冻结 staging 缺失",
        )
        worker_sha = _sha256(staged_worker)
        manager_sha = _sha256(staged_manager)
        _require(
            worker_sha == source_sha
            and manager_sha == M29_SHA256,
            f"eval {tag} 冻结 staging/source 身份漂移",
        )
    else:
        worker_sha = _stage_eval_file(
            worker, staged_worker, expected_sha256=source_sha)
        manager_sha = _stage_eval_file(
            M29_NPZ, staged_manager, expected_sha256=M29_SHA256)
    _require(
        stage_dir.is_dir() and not stage_dir.is_symlink()
        and set(stage_dir.iterdir()) == {staged_worker, staged_manager}
        and all(
            path.is_file() and not path.is_symlink()
            for path in (staged_worker, staged_manager)
        ),
        f"eval {tag} staging 成员集合必须精确为 worker/manager",
    )
    command = _eval_command(staged_worker, seeds, tag, staged_manager)
    return {
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "tag": tag,
        "seeds": list(seeds),
        "source_worker_path": str(worker),
        "source_worker_sha256": source_sha,
        "staged_worker_path": str(staged_worker),
        "worker_sha256": worker_sha,
        "staged_manager_path": str(staged_manager),
        "manager_sha256": manager_sha,
        "training_receipt_sha256": receipt_sha,
        "final_bind_sha256": final_bind_sha256,
        "implementation": _implementation_identity(),
        "command": command,
        "attempt": 1,
    }


def _eval_fired_core(launch: dict) -> dict:
    return {"schema_version": EVAL_FIRED_SCHEMA, **launch}


def _validate_eval_fired(tag: str, launch: dict) -> dict:
    path = _eval_fired_path(tag)
    record = _stable_json(path)
    core = _eval_fired_core(launch)
    _require(
        set(record) == {*core, "fired_at_ns"}
        and _is_plain_int(record["fired_at_ns"])
        and record["fired_at_ns"] > 0
        and all(record[key] == value for key, value in core.items()),
        f"eval fired marker 漂移:{tag}",
    )
    return record


def _validate_eval_archive(
        launch: dict, seeds: tuple[int, ...], tag: str) -> tuple[dict, str]:
    worker = pathlib.Path(launch["staged_worker_path"])
    manager = pathlib.Path(launch["staged_manager_path"])
    _require(
        _sha256(worker) == launch["worker_sha256"]
        and _sha256(manager) == launch["manager_sha256"],
        f"eval staging 身份漂移:{tag}",
    )
    snapshot = eval_contract.freeze_eval_identity(
        ROOT, str(worker), str(manager))
    expected = eval_contract.expected_eval_identity(
        snapshot, tag=tag, seeds=seeds)
    path = _eval_path(tag)
    try:
        archive_payload = _stable_read(path)
        document = eval_contract.strict_json_loads(archive_payload)
        document = eval_contract.validate_eval_archive(document, **expected)
    except eval_contract.EvalContractError as exc:
        raise CampaignError(f"eval archive 非法:{path}: {exc}") from exc
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    _require(
        document["meta"]["worker"]["sha256"] == launch["worker_sha256"]
        and document["meta"]["manager"]["sha256"] == launch["manager_sha256"]
        and document["meta"]["worker"]["path"] == str(worker.resolve())
        and document["meta"]["manager"]["path"] == str(manager.resolve())
        and _sha256(worker) == launch["worker_sha256"]
        and _sha256(manager) == launch["manager_sha256"],
        f"eval archive worker/manager 身份异常:{tag}",
    )
    # R7 refuses legacy v5 archives that predate the action-14 execution
    # ledger.  The generic archive reader still accepts them for historical
    # forensics, but they cannot support the current publication claim.
    _gear_progression_gate(document)
    _require(
        _sha256(path) == archive_sha256,
        f"eval archive 验证期间路径漂移:{tag}",
    )
    return document, archive_sha256


def _eval_attestation(
        launch: dict, archive_sha256: str, fired_sha256: str) -> dict:
    return {
        "schema_version": EVAL_ATTESTATION_SCHEMA,
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "tag": launch["tag"],
        "seeds": launch["seeds"],
        "worker_sha256": launch["worker_sha256"],
        "manager_sha256": launch["manager_sha256"],
        "training_receipt_sha256": launch["training_receipt_sha256"],
        "final_bind_sha256": launch["final_bind_sha256"],
        "fired_sha256": fired_sha256,
        "archive_sha256": archive_sha256,
        "command": launch["command"],
    }


def _validate_existing_eval(
        launch: dict, seeds: tuple[int, ...], tag: str) -> tuple[dict, str]:
    _validate_eval_fired(tag, launch)
    document, archive_sha = _validate_eval_archive(launch, seeds, tag)
    fired_sha = _sha256(_eval_fired_path(tag))
    expected = _eval_attestation(launch, archive_sha, fired_sha)
    path = _eval_attestation_path(tag)
    _require(path.exists(), f"eval archive 缺独立 attestation:{tag}")
    _require(_stable_json(path) == expected,
             f"eval attestation 与 fired/archive 漂移:{tag}")
    _require(
        _sha256(_eval_path(tag)) == archive_sha,
        f"eval archive 在 attestation 复验期间漂移:{tag}",
    )
    return document, archive_sha


def _run_eval_once(
        worker: pathlib.Path, seeds: tuple[int, ...], tag: str, *,
        training_receipt: dict | None = None,
        final_bind_sha256: str | None = None,
        prepared_launch: dict | None = None) -> tuple[dict, str]:
    marker_path = _eval_fired_path(tag)
    current_launch = _prepare_eval_launch(
        worker, seeds, tag,
        training_receipt=training_receipt,
        final_bind_sha256=final_bind_sha256,
        require_existing_staging=marker_path.exists(),
    )
    if prepared_launch is not None:
        _require(prepared_launch == current_launch,
                 f"eval {tag} prepared launch 身份漂移")
    launch = current_launch
    path = _eval_path(tag)
    if path.exists():
        attestation_path = _eval_attestation_path(tag)
        if attestation_path.exists():
            return _validate_existing_eval(launch, seeds, tag)
        # Same crash window as training receipt creation: the evaluator may
        # have atomically committed its complete archive before the launcher
        # wrote the independent attestation.  Revalidate the original fired
        # marker, frozen implementation, source/staging bytes and archive,
        # then create only the missing derived attestation.  No evaluation is
        # ever re-fired and an incomplete archive still fails closed.
        _validate_eval_fired(tag, launch)
        _require(
            _implementation_identity() == launch["implementation"],
            f"eval {tag} adoption 时 implementation 漂移",
        )
        _require(
            _sha256(worker) == launch["source_worker_sha256"],
            f"eval {tag} adoption 时源 worker 漂移",
        )
        document, archive_sha = _validate_eval_archive(
            launch, seeds, tag)
        attestation = _eval_attestation(
            launch, archive_sha, _sha256(marker_path))
        _write_json_exclusive(attestation_path, attestation)
        _require(
            _sha256(path) == archive_sha,
            f"eval archive 在 adoption attestation 提交期间漂移:{tag}",
        )
        return document, archive_sha
    _require(
        not marker_path.exists(),
        f"eval {tag} 已点火但无完整档案；禁止观察部分输出后重试",
    )
    _require(
        not _eval_attestation_path(tag).exists(),
        f"eval {tag} 无 archive 却已有 attestation",
    )
    marker = {
        **_eval_fired_core(launch),
        "fired_at_ns": time.time_ns(),
    }
    _write_json_exclusive(marker_path, marker)
    _invoke(launch["command"], f"R7 eval {tag}")
    _validate_eval_fired(tag, launch)
    _require(
        _implementation_identity() == launch["implementation"],
        f"eval {tag} 期间 launcher/statistics/implementation 漂移",
    )
    # The evaluator consumes staging copies, but the source receipt must remain
    # true through archive commit as well.
    _require(_sha256(worker) == launch["source_worker_sha256"],
             f"eval {tag} 期间源 worker 漂移")
    document, archive_sha = _validate_eval_archive(launch, seeds, tag)
    attestation = _eval_attestation(
        launch, archive_sha, _sha256(marker_path))
    _write_json_exclusive(_eval_attestation_path(tag), attestation)
    _require(
        _sha256(path) == archive_sha,
        f"eval archive 在 attestation 提交期间漂移:{tag}",
    )
    return document, archive_sha


def _read_eval_once(
        worker: pathlib.Path, seeds: tuple[int, ...], tag: str, *,
        training_receipt: dict | None = None,
        final_bind_sha256: str | None = None) -> tuple[dict, str]:
    """Pure validation path: never creates staging, markers, or archives."""
    launch = _prepare_eval_launch(
        worker, seeds, tag,
        training_receipt=training_receipt,
        final_bind_sha256=final_bind_sha256,
        require_existing_staging=True,
    )
    _require(_eval_path(tag).exists(), f"冻结 eval archive 缺失:{tag}")
    return _validate_existing_eval(launch, seeds, tag)


def _analysis_path(pool: str, recipe: str, seed: int) -> pathlib.Path:
    return CONTROL_DIR / f"analysis-{pool}-{recipe}-s{seed}.json"


def _gear_progression_gate(candidate: dict) -> dict:
    """Validate and gate action-14 growth on aggregate legal opportunity.

    Zero/few opportunities are explicit and non-failing: a paired pool cannot
    manufacture a useful gear drop.  Once the opportunity ledger is
    informative, however, publication requires the complete causal chain:
    the policy requested action 14, native execution produced at least one
    strictly improving equip, and the authoritative whole-loadout utility
    increased.
    """
    _require(isinstance(candidate, dict), "gear gate candidate archive 非对象")
    agg = candidate.get("agg")
    _require(isinstance(agg, dict), "gear gate 缺 candidate agg")
    names = (
        "worker_calls",
        "worker_action14_mask_opportunities",
        "worker_action14_requests",
        "worker_action14_native_successes",
        "worker_action14_gear_utility_delta",
    )
    _require(
        all(name in agg for name in names),
        "R7 eval archive 缺 action14 gear progression ledger",
    )
    values = {name: agg[name] for name in names}
    _require(
        all(_is_plain_int(value) and value >= 0
            for value in values.values()),
        "R7 action14 gear progression ledger 必须是非负整数",
    )
    calls = values["worker_calls"]
    opportunities = values[
        "worker_action14_mask_opportunities"]
    requests = values["worker_action14_requests"]
    successes = values["worker_action14_native_successes"]
    utility_delta = values[
        "worker_action14_gear_utility_delta"]
    _require(
        successes <= requests <= opportunities <= calls,
        "R7 action14 gear progression 次序不满足 "
        "success<=request<=opportunity<=calls",
    )
    _require(
        (successes == 0) == (utility_delta == 0),
        "R7 action14 native success 与 utility delta 零性不一致",
    )

    informative = opportunities >= MIN_ACTION14_MASK_OPPORTUNITIES
    if opportunities == 0:
        opportunity_status = "NO_MASK_OPPORTUNITY_OBSERVED"
    elif informative:
        opportunity_status = "INFORMATIVE_MASK_OPPORTUNITY"
    else:
        opportunity_status = "INSUFFICIENT_MASK_OPPORTUNITY"
    checks = {
        "request_observed_when_informative":
            (not informative or requests >= 1),
        "native_success_observed_when_informative":
            (not informative or successes >= 1),
        "utility_growth_observed_when_informative":
            (not informative or utility_delta >= 1),
    }
    passed = all(checks.values())
    return {
        "schema_version": GEAR_PROGRESSION_GATE_SCHEMA,
        "minimum_action14_mask_opportunities":
            MIN_ACTION14_MASK_OPPORTUNITIES,
        "opportunity_status": opportunity_status,
        "informative": informative,
        "worker_calls": calls,
        "action14_mask_opportunities": opportunities,
        "action14_requests": requests,
        "action14_native_successes": successes,
        "action14_gear_utility_delta": utility_delta,
        "checks": checks,
        "passed": passed,
    }



# === rev21(PREREG-R7-rev21-proposal,总设计师 2026-07-27 批)===
# 每步效率记分肢与存活分解诊断:只记不裁——不进 METRIC_RULES、不占 familywise α、
# 不作任何 pass/fail 输入。预注册参考带(判读用,非门):点 +0.004/微步,带 [0, +0.010]。
# 口径立法:效率一律步数口径(farm_worker_wage / micro_steps);
# 分解为对称切分 Δwage = m̄·Δr + r̄·Δm(逐种子精确恒等)。
REV21_DIAGNOSTICS_SCHEMA = "diablogym-r7-rev21-diagnostics/1"
REV21_RATE_POINT = 0.004
REV21_RATE_BAND = (0.0, 0.010)


def _rev21_diagnostics(baseline_archive: dict, candidate_archive: dict) -> dict:
    """Record-only per-step efficiency + survival decomposition (rev21)."""
    brows = baseline_archive["rows"]
    crows = candidate_archive["rows"]
    _require(
        len(brows) == len(crows)
        and all(b.get("seed") == c.get("seed")
                for b, c in zip(brows, crows)),
        "rev21 diagnostics 需要种子逐位配对的行",
    )
    rate_rows = []
    time_component = 0.0
    rate_component = 0.0
    deaths_flipped_to_died = []
    deaths_flipped_to_survived = []
    for b, c in zip(brows, crows):
        mb, mc = int(b["micro_steps"]), int(c["micro_steps"])
        _require(mb > 0 and mc > 0, "rev21 diagnostics: micro_steps 必须为正")
        wb = float(b["farm_worker_wage"])
        wc = float(c["farm_worker_wage"])
        rb, rc = wb / mb, wc / mc
        mean_rate = (rb + rc) / 2.0
        mean_steps = (mb + mc) / 2.0
        time_component += mean_rate * (mc - mb)
        rate_component += mean_steps * (rc - rb)
        rate_rows.append({
            "seed": int(b["seed"]),
            "baseline_rate": rb,
            "candidate_rate": rc,
            "delta_rate": rc - rb,
        })
        if bool(c.get("died")) and not bool(b.get("died")):
            deaths_flipped_to_died.append(int(b["seed"]))
        if bool(b.get("died")) and not bool(c.get("died")):
            deaths_flipped_to_survived.append(int(b["seed"]))
    deltas = [row["delta_rate"] for row in rate_rows]
    n = len(deltas)
    ordered = sorted(deltas)
    median = (ordered[n // 2] if n % 2
              else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0)
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = n - wins - losses
    non_ties = wins + losses
    if non_ties:
        tail = sum(math.comb(non_ties, k)
                   for k in range(max(wins, losses), non_ties + 1))
        sign_p_two_sided = min(1.0, 2.0 * tail / (2 ** non_ties))
    else:
        sign_p_two_sided = 1.0
    mean = sum(deltas) / n
    leverage_index = max(range(n), key=lambda i: abs(deltas[i]))
    deleveraged = (
        (sum(deltas) - deltas[leverage_index]) / (n - 1) if n > 1 else mean)
    lo, hi = REV21_RATE_BAND
    return {
        "schema_version": REV21_DIAGNOSTICS_SCHEMA,
        "record_only": True,
        "rate_report": {
            "metric": "farm_worker_wage / micro_steps (per-seed paired delta)",
            "n_pairs": n,
            "mean": mean,
            "median": median,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "exact_sign_p_two_sided": sign_p_two_sided,
            "deleveraged_mean": deleveraged,
            "max_leverage_seed": rate_rows[leverage_index]["seed"],
            "registered_reference": {
                "point": REV21_RATE_POINT,
                "band": [lo, hi],
                "in_band": bool(lo <= mean <= hi),
            },
            "rows": rate_rows,
        },
        "survival_decomposition": {
            "identity": "sum Δwage == rate_component + time_component"
                        " (对称切分,逐种子精确)",
            "rate_component": rate_component,
            "time_component": time_component,
            "total_delta_wage": rate_component + time_component,
            "deaths_flipped_to_died": deaths_flipped_to_died,
            "deaths_flipped_to_survived": deaths_flipped_to_survived,
        },
    }


def _analyze_pair(
        baseline: dict, baseline_sha: str,
        candidate: dict, candidate_sha: str,
        *, phase: str, death_margin: float,
        training_death_reward_evidence: dict | None = None) -> dict:
    analysis = r7_statistics.analyze_paired_archives(
        baseline,
        candidate,
        baseline_archive_sha256=baseline_sha,
        candidate_archive_sha256=candidate_sha,
        phase=phase,
        metric_rules=METRIC_RULES,
        death_noninferiority_margin=death_margin,
        familywise_alpha=FAMILYWISE_ALPHA,
    )
    gear_gate = _gear_progression_gate(candidate)
    verdict = dict(analysis["verdict"])
    checks = dict(verdict["checks"])
    checks["gear.action14_progression"] = gear_gate["passed"]
    failed_checks = sorted(
        name for name, passed in checks.items() if not passed)
    analysis = {
        **analysis,
        "gear_progression_gate": gear_gate,
        "verdict": {
            **verdict,
            "status": "PASS" if not failed_checks else "FAIL",
            "checks": checks,
            "failed_checks": failed_checks,
        },
    }
    # rev21:只记不裁诊断随一切配对分析落档(development 与 final 同享)。
    analysis["rev21_diagnostics"] = _rev21_diagnostics(baseline, candidate)
    if training_death_reward_evidence is not None:
        _require(
            isinstance(training_death_reward_evidence, dict)
            and training_death_reward_evidence.get("schema_version")
            == TERMINAL_DEATH_EVIDENCE_SCHEMA
            and training_death_reward_evidence.get("opportunity_status")
            in {
                "TRAINING_FAILURE_OBSERVED",
                "NO_TRAINING_FAILURE_OBSERVED",
            }
            and training_death_reward_evidence.get(
                "reward_mechanism_triggered")
            is (
                training_death_reward_evidence.get("opportunity_status")
                == "TRAINING_FAILURE_OBSERVED"
            ),
            "analysis 缺严格 terminal-death/timeout training evidence",
        )
        analysis = {
            **analysis,
            "training_death_reward_evidence":
                training_death_reward_evidence,
        }
    return analysis


def _development_analysis_keys() -> tuple[str, ...]:
    return tuple(
        f"{pool}:{recipe}:{seed}"
        for pool in DEV_POOLS
        for recipe in RECIPE_PREFERENCE
        for seed in DEVELOPMENT_TRAIN_SEEDS
    )


def _development_decision_document(
        analyses: dict[str, dict],
        analysis_sha256s: dict[str, str] | None = None) -> dict:
    _require(
        set(analyses) == set(_development_analysis_keys()),
        "development analyses 键集合不等于冻结的 12 份计划",
    )
    if analysis_sha256s is None:
        analysis_sha256s = {
            key: _canonical_sha256(value)
            for key, value in analyses.items()
        }
    _require(
        set(analysis_sha256s) == set(analyses)
        and all(_is_sha256(value) for value in analysis_sha256s.values()),
        "development analysis SHA 集合不闭合",
    )
    training_death_evidence: dict[str, dict] = {}
    for recipe in RECIPE_PREFERENCE:
        for seed in DEVELOPMENT_TRAIN_SEEDS:
            per_pool = [
                analyses[f"{pool}:{recipe}:{seed}"].get(
                    "training_death_reward_evidence")
                for pool in DEV_POOLS
            ]
            _require(
                all(isinstance(value, dict) for value in per_pool)
                and all(value == per_pool[0] for value in per_pool[1:]),
                "development 同一训练 replication 的 death evidence "
                f"跨 pool 缺失/漂移:{recipe}/{seed}",
            )
            training_death_evidence[f"{recipe}:{seed}"] = (
                _validate_terminal_death_evidence_document(
                    per_pool[0], recipe=recipe))
    replication_pass: dict[str, dict[str, bool]] = {}
    eligibility: dict[str, dict] = {}
    for recipe in RECIPE_PREFERENCE:
        by_seed = {}
        for seed in DEVELOPMENT_TRAIN_SEEDS:
            by_seed[str(seed)] = all(
                analyses[f"{pool}:{recipe}:{seed}"]["verdict"]["status"]
                == "PASS"
                for pool in DEV_POOLS
            )
        replication_pass[recipe] = by_seed
        passed = sum(by_seed.values())
        per_pool = {
            pool: sum(
                analyses[f"{pool}:{recipe}:{seed}"]["verdict"]["status"]
                == "PASS"
                for seed in DEVELOPMENT_TRAIN_SEEDS
            )
            for pool in DEV_POOLS
        }
        eligible = (
            passed >= MIN_REPLICATIONS_PASSING_BOTH_POOLS
            and all(
                count >= MIN_REPLICATIONS_PASSING_BOTH_POOLS
                for count in per_pool.values()
            )
        )
        eligibility[recipe] = {
            "replications_passing_both_pools": passed,
            "per_pool_passes": per_pool,
            "training_failure_opportunity": {
                str(seed): training_death_evidence[
                    f"{recipe}:{seed}"]["opportunity_status"]
                for seed in DEVELOPMENT_TRAIN_SEEDS
            },
            "training_failure_observed_replications": sum(
                training_death_evidence[
                    f"{recipe}:{seed}"]["reward_mechanism_triggered"]
                for seed in DEVELOPMENT_TRAIN_SEEDS
            ),
            "eligible": eligible,
        }
    selected = next(
        (recipe for recipe in RECIPE_PREFERENCE
         if eligibility[recipe]["eligible"]),
        None,
    )
    return {
        "schema_version": DEVELOPMENT_DECISION_SCHEMA,
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "selection_rule":
            "first-preferred-recipe-with-at-least-two-of-three-"
            "replications-passing-both-128-pair-pools",
        "replication_pass": replication_pass,
        "eligibility": eligibility,
        "selected_recipe": selected,
        "analysis_sha256s": dict(sorted(analysis_sha256s.items())),
        "training_death_reward_evidence":
            dict(sorted(training_death_evidence.items())),
    }


def _analysis_file_sha256s(analyses: dict[str, dict]) -> dict[str, str]:
    result = {}
    for key, analysis in analyses.items():
        pool, recipe, raw_seed = key.split(":")
        path = _analysis_path(pool, recipe, int(raw_seed))
        _require(path.exists(), f"development analysis 文件缺失:{key}")
        _require(_stable_json(path) == analysis,
                 f"development analysis 文件语义漂移:{key}")
        result[key] = _sha256(path)
    return result


def _freeze_development_decision(analyses: dict[str, dict]) -> dict:
    decision = _development_decision_document(
        analyses, _analysis_file_sha256s(analyses))
    _write_json_exclusive(DEVELOPMENT_DECISION_PATH, decision)
    return decision


def _recompute_development_analyses() -> dict[str, dict]:
    analyses: dict[str, dict] = {}
    for pool, seeds in DEV_POOLS.items():
        base_tag = _eval_tag(pool, "baseline")
        baseline, baseline_sha = _read_eval_once(
            V28_ZIP, seeds, base_tag)
        for recipe in RECIPE_PREFERENCE:
            for seed in DEVELOPMENT_TRAIN_SEEDS:
                receipt = _validate_training_artifact(
                    recipe, seed, "development")
                worker = _training_model_path(
                    recipe, seed, "development")
                tag = _eval_tag(
                    pool, "candidate", recipe=recipe, seed=seed)
                candidate, candidate_sha = _read_eval_once(
                    worker, seeds, tag, training_receipt=receipt)
                analysis = _analyze_pair(
                    baseline, baseline_sha,
                    candidate, candidate_sha,
                    phase="development",
                    death_margin=DEVELOPMENT_DEATH_MARGIN,
                    training_death_reward_evidence=receipt[
                        "terminal_death_reward_evidence"],
                )
                key = f"{pool}:{recipe}:{seed}"
                path = _analysis_path(pool, recipe, seed)
                _require(path.exists(), f"development analysis 缺失:{key}")
                frozen = _stable_json(path)
                _require(
                    frozen == analysis
                    and _canonical_sha256(frozen)
                    == _canonical_sha256(analysis),
                    f"development analysis 无法由冻结档案重算:{key}",
                )
                analyses[key] = analysis
    return analyses


def _validate_development_decision(state: dict) -> dict:
    analyses = _recompute_development_analyses()
    expected = _development_decision_document(
        analyses, _analysis_file_sha256s(analyses))
    _require(DEVELOPMENT_DECISION_PATH.exists(),
             "development decision 文件缺失")
    actual = _stable_json(DEVELOPMENT_DECISION_PATH)
    _require(
        set(actual) == set(expected) and actual == expected,
        "development decision 不能由冻结的 12 份 analysis 重算",
    )
    phase = state["phases"].get("eval_development", {})
    _require(
        set(phase) == {
            "status", "completed", "decision_sha256", "selected_recipe"}
        and phase["status"] == "complete"
        and phase["completed"] == sorted(_development_analysis_keys())
        and phase["decision_sha256"] == _sha256(DEVELOPMENT_DECISION_PATH)
        and phase["selected_recipe"] == expected["selected_recipe"]
        and expected["selected_recipe"] in RECIPES,
        "development state/decision SHA/选择闭合失败",
    )
    return expected


def command_eval_development() -> None:
    _require_seed_discipline()
    state = _load_state()
    _require_not_terminal(state)
    _require(
        state["phases"].get("train_development", {}).get("status")
        == "complete",
        "开发 cohort 尚未完整冻结",
    )
    analyses: dict[str, dict] = {}
    try:
        for pool, seeds in DEV_POOLS.items():
            base_tag = _eval_tag(pool, "baseline")
            baseline, baseline_sha = _run_eval_once(
                V28_ZIP, seeds, base_tag)
            for recipe in RECIPE_PREFERENCE:
                for seed in DEVELOPMENT_TRAIN_SEEDS:
                    receipt = _validate_training_artifact(
                        recipe, seed, "development")
                    worker = _training_model_path(
                        recipe, seed, "development")
                    tag = _eval_tag(
                        pool, "candidate", recipe=recipe, seed=seed)
                    candidate, candidate_sha = _run_eval_once(
                        worker, seeds, tag, training_receipt=receipt)
                    analysis = _analyze_pair(
                        baseline, baseline_sha,
                        candidate, candidate_sha,
                        phase="development",
                        death_margin=DEVELOPMENT_DEATH_MARGIN,
                        training_death_reward_evidence=receipt[
                            "terminal_death_reward_evidence"],
                    )
                    key = f"{pool}:{recipe}:{seed}"
                    analyses[key] = analysis
                    _write_json_exclusive(
                        _analysis_path(pool, recipe, seed), analysis)
                    _set_phase(
                        state, "eval_development", "running",
                        completed=sorted(analyses),
                    )
        decision = _freeze_development_decision(analyses)
    except Exception as exc:
        _set_phase(
            state, "eval_development", "locked-failed",
            completed=sorted(analyses), retry_forbidden=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    if decision["selected_recipe"] is None:
        state["terminal_status"] = "DEVELOPMENT_SCIENTIFIC_FAIL"
        _set_phase(
            state, "eval_development", "scientific-fail",
            completed=sorted(analyses),
            decision_sha256=_sha256(DEVELOPMENT_DECISION_PATH),
            selected_recipe=None,
        )
        _write_json_atomic(STATE_PATH, state)
        raise CampaignError(
            "两档预注册风险配方均未在两个 128-pair pools 下达到"
            "至少 2/3 跨训练 RNG 复现；final 保持未打开")
    _set_phase(
        state, "eval_development", "complete",
        completed=sorted(analyses),
        decision_sha256=_sha256(DEVELOPMENT_DECISION_PATH),
        selected_recipe=decision["selected_recipe"],
    )
    print(f"R7 development 选中配方:{decision['selected_recipe']}；未挑选任何 dev 模型。")


def _selected_recipe(state: dict) -> str:
    decision = _validate_development_decision(state)
    return decision["selected_recipe"]


def command_train_production() -> None:
    _require_seed_discipline()
    state = _load_state()
    _require_not_terminal(state)
    recipe = _selected_recipe(state)
    phase = state["phases"].get("train_production", {})
    _require(phase.get("status") != "locked-failed", "production 已 locked-failed")
    _set_phase(
        state, "train_production", "running",
        recipe=recipe, seed=PRODUCTION_TRAIN_SEED, attempts=1,
    )
    try:
        receipt = _run_training_once(
            recipe, PRODUCTION_TRAIN_SEED, "candidate")
    except Exception as exc:
        _set_phase(
            state, "train_production", "locked-failed",
            recipe=recipe, seed=PRODUCTION_TRAIN_SEED, attempts=1,
            retry_forbidden=True, error=f"{type(exc).__name__}: {exc}",
        )
        raise
    _set_phase(
        state, "train_production", "complete",
        recipe=recipe, seed=PRODUCTION_TRAIN_SEED, attempts=1,
        artifact=receipt,
    )
    print("独立 production seed 已从 V28 重训并冻结；未查看 development 结果的模型权重。")


def _find_final_eval_residue(allowed: Iterable[pathlib.Path] = ()) -> list[str]:
    allowed_resolved = {
        os.path.abspath(os.fspath(path)) for path in allowed
    }
    final = set(FINAL_POOL)
    residue = []
    if not EVAL_DIR.exists():
        return residue
    final_literals = tuple(str(seed).encode("ascii") for seed in FINAL_POOL)
    for path in sorted(EVAL_DIR.rglob("*")):
        if path.is_symlink():
            residue.append(str(path))
            continue
        if path.is_dir():
            continue
        if os.path.abspath(os.fspath(path)) in allowed_resolved:
            continue
        lower_name = path.name.lower()
        if lower_name in {
                ".gold-evaluation.lock", ".reservation.lock",
                ".leaderboard.lock"}:
            continue
        suspicious_name = (
            "official-r7-final" in lower_name
            or "2120000" in lower_name
        )
        try:
            payload = _stable_read(path)
        except CampaignError:
            residue.append(str(path))
            continue
        raw_overlap = any(
            re.search(rb"(?<![0-9])" + literal + rb"(?![0-9])", payload)
            is not None
            for literal in final_literals
        )
        try:
            document = eval_contract.strict_json_loads(payload)
        except eval_contract.EvalContractError:
            if suspicious_name or raw_overlap:
                residue.append(str(path))
            continue
        if not isinstance(document, dict):
            if suspicious_name or raw_overlap:
                residue.append(str(path))
            continue
        discovered = set()
        meta = document.get("meta")
        if isinstance(meta, dict):
            protocol = meta.get("protocol")
            if isinstance(protocol, dict) and isinstance(
                    protocol.get("seeds"), list):
                discovered.update(
                    seed for seed in protocol["seeds"]
                    if _is_plain_int(seed))
        rows = document.get("rows")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    for key in ("seed", "episode_seed"):
                        if _is_plain_int(row.get(key)):
                            discovered.add(row[key])
        if final.intersection(discovered) or suspicious_name or raw_overlap:
            residue.append(str(path))
    return residue


def _final_pool_spec() -> dict:
    return {
        "pool_name": "r7-official-final",
        "seeds": list(FINAL_POOL),
    }


def _final_pool_sha256() -> str:
    return _canonical_sha256(_final_pool_spec())


def _final_registry_path() -> pathlib.Path:
    return FINAL_REGISTRY_DIR / f"{_final_pool_sha256()}.json"


@contextlib.contextmanager
def _final_registry_lock():
    with _regular_path_lock(
            FINAL_REGISTRY_LOCK, "R7 final registry", nonblocking=False):
        yield


def _validate_final_registry_document(
        path: pathlib.Path, record: dict) -> dict:
    expected_keys = {
        "schema_version", "pool_name", "seeds", "pool_sha256",
        "campaign_recipe_sha256", "bind_sha256", "control_path",
        "opened_at_ns", "consumption_stage",
    }
    _require(set(record) == expected_keys,
             f"final registry marker 字段不精确:{path}")
    seeds = record["seeds"]
    spec = {"pool_name": record["pool_name"], "seeds": seeds}
    _require(
        record["schema_version"] == FINAL_REGISTRY_SCHEMA
        and record["pool_name"] == "r7-official-final"
        and isinstance(seeds, list)
        and bool(seeds)
        and all(_is_plain_int(seed) for seed in seeds)
        and len(seeds) == len(set(seeds))
        and record["pool_sha256"] == _canonical_sha256(spec)
        and path.name == f"{record['pool_sha256']}.json"
        and _is_sha256(record["campaign_recipe_sha256"])
        and _is_sha256(record["bind_sha256"])
        and isinstance(record["control_path"], str)
        and bool(record["control_path"])
        and _is_plain_int(record["opened_at_ns"])
        and record["opened_at_ns"] > 0
        and record["consumption_stage"] == "before_baseline_evaluation",
        f"final registry marker 身份非法:{path}",
    )
    return record


def _validate_final_registry_record(path: pathlib.Path) -> dict:
    record = _stable_json(path)
    return _validate_final_registry_document(path, record)


def _final_registry_core(bind_sha256: str) -> dict:
    return {
        "schema_version": FINAL_REGISTRY_SCHEMA,
        **_final_pool_spec(),
        "pool_sha256": _final_pool_sha256(),
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "bind_sha256": bind_sha256,
        "control_path": str(CONTROL_DIR.resolve()),
        "consumption_stage": "before_baseline_evaluation",
    }


def _register_final_pool(bind_sha256: str, *, continuing: bool) -> dict:
    _require(_is_sha256(bind_sha256), "final bind SHA 非法")
    if continuing:
        _require(FINAL_OPENED_PATH.exists(),
                 "continuing final 缺本地 final-opened marker")
    marker_path = _final_registry_path()
    core = _final_registry_core(bind_sha256)
    requested = set(FINAL_POOL)
    with _final_registry_lock():
        records = []
        for path in sorted(FINAL_REGISTRY_DIR.iterdir()):
            if path == FINAL_REGISTRY_LOCK:
                continue
            _require(path.is_file() and path.suffix == ".json",
                     f"final registry 含未知残件:{path}")
            records.append((path, _validate_final_registry_record(path)))
        if continuing:
            _require(marker_path.exists(),
                     "本地 final-opened 存在但全局 registry marker 缺失")
            record = _validate_final_registry_record(marker_path)
            _require(
                set(record) == {*core, "opened_at_ns"}
                and all(record[key] == value for key, value in core.items()),
                "全局 final registry 与本地 bind 漂移",
            )
            return record
        _require(not marker_path.exists(),
                 "final pool 已在全局 registry 点火；本地 marker 缺失时禁止恢复")
        for path, record in records:
            overlap = requested.intersection(record["seeds"])
            _require(
                not overlap,
                "final pool 与全局历史 registry 重叠:"
                f"{path}, overlap={sorted(overlap)[:16]}",
            )
        record = {**core, "opened_at_ns": time.time_ns()}
        _write_json_exclusive(marker_path, record)
        return record


_FINAL_REGISTRY_EVIDENCE_KEYS = frozenset({
    "final_registry_path",
    "final_registry_record_sha256",
})


def _final_registry_evidence(bind_core: dict) -> dict:
    """Validate and snapshot the global reservation for one semantic bind."""
    path = _final_registry_path()
    record, record_sha256 = _stable_json_snapshot(path)
    _validate_final_registry_document(path, record)
    expected = _final_registry_core(_canonical_sha256(bind_core))
    _require(
        set(record) == {*expected, "opened_at_ns"}
        and all(record[key] == value for key, value in expected.items()),
        "全局 final registry 与 final bind core 漂移",
    )
    evidence = {
        "final_registry_path": str(path.resolve()),
        "final_registry_record_sha256": record_sha256,
    }
    _require(
        record["pool_sha256"] == bind_core.get("final_pool_sha256")
        == _final_pool_sha256()
        and _sha256(path) == record_sha256,
        "全局 final registry pool/record SHA 闭合失败",
    )
    return evidence


def _final_opened_document(bind_core: dict) -> dict:
    return {**bind_core, **_final_registry_evidence(bind_core)}


def _final_bind_core_from_opened(opened: dict) -> dict:
    _require(
        isinstance(opened, dict)
        and _FINAL_REGISTRY_EVIDENCE_KEYS.issubset(opened),
        "final-opened 缺全局 registry 证据",
    )
    return {
        key: value for key, value in opened.items()
        if key not in _FINAL_REGISTRY_EVIDENCE_KEYS
    }


def _validate_final_opened_registry(bind_core: dict) -> tuple[dict, dict]:
    """Close local final-opened bytes against the still-live global registry."""
    opened, opened_sha256 = _stable_json_snapshot(FINAL_OPENED_PATH)
    expected = _final_opened_document(bind_core)
    _require(opened == expected, "final-opened/registry/bind 证据链漂移")
    evidence = {
        key: opened[key] for key in _FINAL_REGISTRY_EVIDENCE_KEYS
    }
    _require(
        opened["final_pool_sha256"] == _final_pool_sha256()
        and opened_sha256 == _sha256(FINAL_OPENED_PATH)
        and evidence["final_registry_record_sha256"]
        == _sha256(_final_registry_path()),
        "final-opened 或 registry 在闭环复验期间漂移",
    )
    return opened, evidence


def _final_bind(state: dict, recipe: str, candidate: pathlib.Path) -> dict:
    receipt = _validate_training_artifact(
        recipe, PRODUCTION_TRAIN_SEED, "candidate")
    phase = state["phases"].get("train_production", {})
    _require(
        set(phase) == {"status", "recipe", "seed", "attempts", "artifact"}
        and phase["status"] == "complete"
        and phase["recipe"] == recipe
        and phase["seed"] == PRODUCTION_TRAIN_SEED
        and phase["attempts"] == 1
        and phase["artifact"] == receipt,
        "production candidate state/receipt 未闭合",
    )
    return {
        "schema_version": FINAL_OPENED_SCHEMA,
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "implementation": _implementation_identity(),
        "seed_registry_sha256": _seed_registry_identity()["sha256"],
        "seeds": list(FINAL_POOL),
        "final_pool_sha256": _final_pool_sha256(),
        "development_decision_sha256": _sha256(DEVELOPMENT_DECISION_PATH),
        "baseline_sha256": V28_SHA256,
        "candidate_sha256": _sha256(candidate),
        "candidate_receipt_sha256": receipt["receipt_sha256"],
        "candidate_training_fired_sha256": receipt["training_fired_sha256"],
        "manager_sha256": M29_SHA256,
        "recipe": recipe,
        "candidate_scope": "candidate",
        "production_seed": PRODUCTION_TRAIN_SEED,
        "baseline_tag": _eval_tag("final", "baseline"),
        "candidate_tag": _eval_tag("final", "candidate"),
    }


def _final_fired_core(bind: dict, launch: dict) -> dict:
    return {
        "schema_version": FINAL_FIRED_SCHEMA,
        "bind_sha256": _sha256(FINAL_OPENED_PATH),
        "candidate_sha256": bind["candidate_sha256"],
        "candidate_receipt_sha256": bind["candidate_receipt_sha256"],
        "candidate_eval_launch_sha256": _canonical_sha256(launch),
        "candidate_command": launch["command"],
        "attempt": 1,
    }


def _validate_final_fired_document(
        bind: dict, launch: dict, record: dict) -> dict:
    core = _final_fired_core(bind, launch)
    _require(
        set(record) == {*core, "fired_at_ns"}
        and _is_plain_int(record["fired_at_ns"])
        and record["fired_at_ns"] > 0
        and all(record[key] == value for key, value in core.items()),
        "final candidate fired marker 漂移",
    )
    return record


def _validate_final_fired(bind: dict, launch: dict) -> dict:
    return _validate_final_fired_document(
        bind, launch, _stable_json(FINAL_FIRED_PATH))


def _capture_final_evidence(
        bind: dict, candidate: pathlib.Path,
        *, expected_analysis: dict | None = None) -> dict:
    """Capture and close every mutable file supporting the final verdict."""
    bind_core = _final_bind_core_from_opened(bind)
    opened, registry_evidence = _validate_final_opened_registry(bind_core)
    _require(opened == bind, "final evidence 使用的 bind 不是冻结 final-opened")
    bind_sha256 = _sha256(FINAL_OPENED_PATH)

    receipt = _validate_training_artifact(
        bind["recipe"], PRODUCTION_TRAIN_SEED, "candidate")
    candidate_payload = _stable_read(candidate)
    candidate_model_sha256 = hashlib.sha256(candidate_payload).hexdigest()
    _require(
        candidate_model_sha256 == bind["candidate_sha256"]
        and receipt["receipt_sha256"] == bind["candidate_receipt_sha256"]
        and _sha256(DEVELOPMENT_DECISION_PATH)
        == bind["development_decision_sha256"],
        "final evidence 的 candidate/receipt/development decision 漂移",
    )

    baseline_launch = _prepare_eval_launch(
        V28_ZIP, FINAL_POOL, bind["baseline_tag"],
        final_bind_sha256=bind_sha256,
        require_existing_staging=True)
    baseline, baseline_sha256 = _validate_existing_eval(
        baseline_launch, FINAL_POOL, bind["baseline_tag"])
    candidate_launch = _prepare_eval_launch(
        candidate, FINAL_POOL, bind["candidate_tag"],
        training_receipt=receipt,
        final_bind_sha256=bind_sha256,
        require_existing_staging=True)
    candidate_document, candidate_sha256 = _validate_existing_eval(
        candidate_launch, FINAL_POOL, bind["candidate_tag"])

    fired, fired_sha256 = _stable_json_snapshot(FINAL_FIRED_PATH)
    _validate_final_fired_document(bind, candidate_launch, fired)
    analysis = _analyze_pair(
        baseline, baseline_sha256,
        candidate_document, candidate_sha256,
        phase="final",
        death_margin=FINAL_DEATH_MARGIN,
        training_death_reward_evidence=receipt[
            "terminal_death_reward_evidence"],
    )
    frozen_analysis, analysis_sha256 = _stable_json_snapshot(
        FINAL_ANALYSIS_PATH)
    _require(
        frozen_analysis == analysis
        and (expected_analysis is None or expected_analysis == analysis),
        "final analysis 不能由冻结 archive 重算或与调用结果不一致",
    )

    identity = {
        "bind_sha256": bind_sha256,
        "final_pool_sha256": bind["final_pool_sha256"],
        **registry_evidence,
        "final_fired_sha256": fired_sha256,
        "analysis_sha256": analysis_sha256,
        "baseline_archive_sha256": baseline_sha256,
        "candidate_archive_sha256": candidate_sha256,
        "baseline_eval_fired_sha256":
            _sha256(_eval_fired_path(bind["baseline_tag"])),
        "candidate_eval_fired_sha256":
            _sha256(_eval_fired_path(bind["candidate_tag"])),
        "baseline_attestation_sha256":
            _sha256(_eval_attestation_path(bind["baseline_tag"])),
        "candidate_attestation_sha256":
            _sha256(_eval_attestation_path(bind["candidate_tag"])),
    }
    path_sha256s = {
        str(FINAL_OPENED_PATH): identity["bind_sha256"],
        str(_final_registry_path()):
            identity["final_registry_record_sha256"],
        str(FINAL_FIRED_PATH): identity["final_fired_sha256"],
        str(FINAL_ANALYSIS_PATH): identity["analysis_sha256"],
        str(DEVELOPMENT_DECISION_PATH):
            bind["development_decision_sha256"],
        str(candidate): candidate_model_sha256,
        str(_training_receipt_path(
            bind["recipe"], PRODUCTION_TRAIN_SEED, "candidate")):
            receipt["receipt_sha256"],
        str(_training_fired_path(
            bind["recipe"], PRODUCTION_TRAIN_SEED, "candidate")):
            receipt["training_fired_sha256"],
        str(_eval_path(bind["baseline_tag"])):
            identity["baseline_archive_sha256"],
        str(_eval_path(bind["candidate_tag"])):
            identity["candidate_archive_sha256"],
        str(_eval_fired_path(bind["baseline_tag"])):
            identity["baseline_eval_fired_sha256"],
        str(_eval_fired_path(bind["candidate_tag"])):
            identity["candidate_eval_fired_sha256"],
        str(_eval_attestation_path(bind["baseline_tag"])):
            identity["baseline_attestation_sha256"],
        str(_eval_attestation_path(bind["candidate_tag"])):
            identity["candidate_attestation_sha256"],
        baseline_launch["staged_worker_path"]:
            baseline_launch["worker_sha256"],
        baseline_launch["staged_manager_path"]:
            baseline_launch["manager_sha256"],
        candidate_launch["staged_worker_path"]:
            candidate_launch["worker_sha256"],
        candidate_launch["staged_manager_path"]:
            candidate_launch["manager_sha256"],
    }
    _require(
        all(_sha256(pathlib.Path(path)) == digest
            for path, digest in path_sha256s.items())
        and _final_registry_evidence(bind_core) == registry_evidence,
        "final evidence snapshot 闭环重哈希失败",
    )
    return {
        "identity": identity,
        "analysis": analysis,
        "candidate_receipt": receipt,
        "candidate_payload": candidate_payload,
        "path_sha256s": path_sha256s,
    }


_FINAL_EVIDENCE_IDENTITY_KEYS = frozenset({
    "bind_sha256",
    "final_pool_sha256",
    "final_registry_path",
    "final_registry_record_sha256",
    "final_fired_sha256",
    "analysis_sha256",
    "baseline_archive_sha256",
    "candidate_archive_sha256",
    "baseline_eval_fired_sha256",
    "candidate_eval_fired_sha256",
    "baseline_attestation_sha256",
    "candidate_attestation_sha256",
})


def _publication_document(
        bind: dict, candidate_receipt: dict, analysis: dict,
        *, evidence: dict) -> dict:
    _require(
        set(evidence) == _FINAL_EVIDENCE_IDENTITY_KEYS
        and isinstance(evidence["final_registry_path"], str)
        and bool(evidence["final_registry_path"])
        and all(
            _is_sha256(value)
            for key, value in evidence.items()
            if key != "final_registry_path"
        )
        and bind.get("final_pool_sha256")
        == evidence["final_pool_sha256"]
        and bind.get("final_registry_path")
        == evidence["final_registry_path"]
        and bind.get("final_registry_record_sha256")
        == evidence["final_registry_record_sha256"],
        "publication final evidence identity 字段/registry 绑定非法",
    )
    return {
        "schema_version": PUBLICATION_RECEIPT_SCHEMA,
        "publication_status": "PUBLISHED",
        "campaign_recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        **evidence,
        "recipe": bind["recipe"],
        "production_seed": bind["production_seed"],
        "candidate_model_sha256": bind["candidate_sha256"],
        "candidate_receipt_sha256": candidate_receipt["receipt_sha256"],
        "published_model_sha256": bind["candidate_sha256"],
        "final_bind": bind,
        "candidate_training_receipt": candidate_receipt,
        "verdict": analysis["verdict"],
    }


def _promote_passed_candidate(
        bind: dict, candidate: pathlib.Path, analysis: dict,
        *, baseline_sha: str, candidate_sha: str) -> dict:
    _require(analysis["verdict"]["status"] == "PASS",
             "只有 final PASS 才允许发布提升")
    frozen = _capture_final_evidence(
        bind, candidate, expected_analysis=analysis)
    evidence = frozen["identity"]
    receipt = frozen["candidate_receipt"]
    candidate_payload = frozen["candidate_payload"]
    _require(
        hashlib.sha256(candidate_payload).hexdigest()
        == bind["candidate_sha256"]
        and receipt["receipt_sha256"] == bind["candidate_receipt_sha256"]
        and evidence["baseline_archive_sha256"] == baseline_sha
        and evidence["candidate_archive_sha256"] == candidate_sha,
        "发布前 candidate/bind/训练回执/archive 漂移",
    )
    publication = _publication_document(
        bind, receipt, analysis, evidence=evidence)
    _require(not PUBLISHED_DIR.is_symlink(),
             f"published 路径不得为符号链接:{PUBLISHED_DIR}")
    if PUBLISHED_DIR.exists():
        _require(
            PUBLISHED_DIR.is_dir() and not PUBLISHED_DIR.is_symlink()
            and set(PUBLISHED_DIR.iterdir())
            == {PUBLISHED_MODEL_PATH, PUBLISHED_RECEIPT_PATH}
            and all(
                path.is_file() and not path.is_symlink()
                for path in (PUBLISHED_MODEL_PATH, PUBLISHED_RECEIPT_PATH)
            ),
            "已存在发布目录成员集合非法",
        )
        _require(
            _sha256(PUBLISHED_MODEL_PATH) == bind["candidate_sha256"]
            and _stable_json(PUBLISHED_RECEIPT_PATH) == publication,
            "已存在发布目录与 final PASS 证据不一致",
        )
        _require(
            _capture_final_evidence(
                bind, candidate, expected_analysis=analysis) == frozen,
            "已存在 publication 复验期间 final evidence 漂移",
        )
        publication["receipt_sha256"] = _sha256(PUBLISHED_RECEIPT_PATH)
        return publication
    staging = PUBLISHED_DIR.with_name(
        f".{PUBLISHED_DIR.name}.{os.getpid()}.{time.time_ns()}.tmp")
    _require(not staging.exists(), f"发布 staging 已存在:{staging}")
    _ensure_directory_durable(staging)
    try:
        _write_bytes_exclusive(
            staging / PUBLISHED_MODEL_PATH.name, candidate_payload, mode=0o444)
        _write_json_exclusive(
            staging / PUBLISHED_RECEIPT_PATH.name, publication)
        _fsync_directory(staging)
        _require(
            _capture_final_evidence(
                bind, candidate, expected_analysis=analysis) == frozen,
            "publication commit 前 final evidence 漂移",
        )
        os.replace(staging, PUBLISHED_DIR)
        _fsync_directory(PUBLISHED_DIR.parent)
    finally:
        # A successful directory rename makes the staging pathname disappear.
        # On failure leave any partial directory as explicit forensic residue.
        pass
    _require(
        PUBLISHED_DIR.is_dir() and not PUBLISHED_DIR.is_symlink()
        and set(PUBLISHED_DIR.iterdir())
        == {PUBLISHED_MODEL_PATH, PUBLISHED_RECEIPT_PATH}
        and all(
            path.is_file() and not path.is_symlink()
            for path in (PUBLISHED_MODEL_PATH, PUBLISHED_RECEIPT_PATH)
        )
        and _sha256(PUBLISHED_MODEL_PATH) == bind["candidate_sha256"]
        and _stable_json(PUBLISHED_RECEIPT_PATH) == publication,
        "原子发布目录提交后复验失败",
    )
    _require(
        _capture_final_evidence(
            bind, candidate, expected_analysis=analysis) == frozen,
        "publication commit 后 final evidence 漂移",
    )
    publication["receipt_sha256"] = _sha256(PUBLISHED_RECEIPT_PATH)
    return publication


def _audit_terminal_publication(state: dict) -> dict:
    """Purely rederive a terminal PASS and its complete publication chain."""
    _require(state.get("terminal_status") == "PASS",
             "只有 terminal PASS 可审计 publication")
    recipe = _selected_recipe(state)
    candidate = _training_model_path(
        recipe, PRODUCTION_TRAIN_SEED, "candidate")
    bind_core = _final_bind(state, recipe, candidate)
    bind, _registry_evidence = _validate_final_opened_registry(bind_core)
    snapshot = _capture_final_evidence(bind, candidate)
    analysis = snapshot["analysis"]
    _require(analysis["verdict"]["status"] == "PASS",
             "terminal PASS 无法由 final archives 重算")
    publication = _publication_document(
        bind,
        snapshot["candidate_receipt"],
        analysis,
        evidence=snapshot["identity"],
    )
    _require(
        PUBLISHED_DIR.is_dir() and not PUBLISHED_DIR.is_symlink()
        and set(PUBLISHED_DIR.iterdir())
        == {PUBLISHED_MODEL_PATH, PUBLISHED_RECEIPT_PATH}
        and all(
            path.is_file() and not path.is_symlink()
            for path in (PUBLISHED_MODEL_PATH, PUBLISHED_RECEIPT_PATH)
        ),
        "terminal PASS publication 目录成员集合非法",
    )
    published_model_payload = _stable_read(PUBLISHED_MODEL_PATH)
    published_receipt, published_receipt_sha256 = _stable_json_snapshot(
        PUBLISHED_RECEIPT_PATH)
    _require(
        hashlib.sha256(published_model_payload).hexdigest()
        == bind["candidate_sha256"]
        and published_model_payload == snapshot["candidate_payload"]
        and published_receipt == publication,
        "terminal PASS published model/receipt 与 final evidence 不一致",
    )
    publication_state = {
        **publication,
        "receipt_sha256": published_receipt_sha256,
    }
    phase = state["phases"].get("eval_final", {})
    expected_phase_keys = {
        "status",
        "bind_sha256",
        "final_pool_sha256",
        "final_registry_path",
        "final_registry_record_sha256",
        "fired_sha256",
        "baseline_archive_sha256",
        "candidate_archive_sha256",
        "analysis_sha256",
        "verdict",
        "publication",
    }
    identity = snapshot["identity"]
    _require(
        set(phase) == expected_phase_keys
        and phase["status"] == "complete"
        and phase["bind_sha256"] == identity["bind_sha256"]
        and phase["final_pool_sha256"] == identity["final_pool_sha256"]
        and phase["final_registry_path"] == identity["final_registry_path"]
        and phase["final_registry_record_sha256"]
        == identity["final_registry_record_sha256"]
        and phase["fired_sha256"] == identity["final_fired_sha256"]
        and phase["baseline_archive_sha256"]
        == identity["baseline_archive_sha256"]
        and phase["candidate_archive_sha256"]
        == identity["candidate_archive_sha256"]
        and phase["analysis_sha256"] == identity["analysis_sha256"]
        and phase["verdict"] == analysis["verdict"]
        and phase["publication"] == publication_state,
        "terminal PASS state/publication/final evidence 未闭合",
    )
    _require(
        _capture_final_evidence(bind, candidate) == snapshot
        and _sha256(PUBLISHED_MODEL_PATH) == bind["candidate_sha256"]
        and _sha256(PUBLISHED_RECEIPT_PATH) == published_receipt_sha256,
        "terminal publication 只读审计期间 evidence 漂移",
    )
    return {
        "publication_status": "PUBLISHED",
        "published_model_sha256": bind["candidate_sha256"],
        "publication_receipt_sha256": published_receipt_sha256,
        "final_pool_sha256": identity["final_pool_sha256"],
        "final_registry_record_sha256":
            identity["final_registry_record_sha256"],
        "analysis_sha256": identity["analysis_sha256"],
    }


def command_eval_final() -> None:
    _require_seed_discipline()
    state = _load_state()
    _require_not_terminal(state)
    _require(
        state["phases"].get("eval_final", {}).get("status")
        != "locked-failed",
        "final 曾发生不可恢复的 operational failure",
    )
    recipe = _selected_recipe(state)
    _require(
        state["phases"].get("train_production", {}).get("status")
        == "complete",
        "production candidate 尚未冻结",
    )
    _require(not PUBLISHED_DIR.is_symlink(),
             f"published 路径不得为符号链接:{PUBLISHED_DIR}")
    if PUBLISHED_DIR.exists():
        _require(
            FINAL_OPENED_PATH.exists()
            and FINAL_FIRED_PATH.exists()
            and FINAL_ANALYSIS_PATH.exists(),
            "缺 final 完整证据却已有 published 目录",
        )
    candidate = _training_model_path(
        recipe, PRODUCTION_TRAIN_SEED, "candidate")
    candidate_receipt = _validate_training_artifact(
        recipe, PRODUCTION_TRAIN_SEED, "candidate")
    bind_core = _final_bind(state, recipe, candidate)
    baseline_path = _eval_path(bind_core["baseline_tag"])
    candidate_path = _eval_path(bind_core["candidate_tag"])

    if not FINAL_OPENED_PATH.exists():
        residue = _find_final_eval_residue()
        _require(not residue, f"final 池已有未登记 eval residue:{residue}")
        _register_final_pool(
            _canonical_sha256(bind_core), continuing=False)
        _write_json_exclusive(
            FINAL_OPENED_PATH, _final_opened_document(bind_core))
    else:
        existing = _stable_json(FINAL_OPENED_PATH)
        _require(
            _final_bind_core_from_opened(existing) == bind_core,
            "final opened marker 与冻结 candidate/recipe 漂移")
        _register_final_pool(
            _canonical_sha256(bind_core), continuing=True)

    bind, registry_evidence = _validate_final_opened_registry(bind_core)
    bind_sha = _sha256(FINAL_OPENED_PATH)
    _set_phase(
        state, "eval_final", "opened",
        bind_sha256=bind_sha,
        final_pool_sha256=bind["final_pool_sha256"],
        final_registry_path=bind["final_registry_path"],
        final_registry_record_sha256=
            bind["final_registry_record_sha256"],
    )
    allowed = []
    if _eval_fired_path(bind["baseline_tag"]).exists():
        allowed.extend((
            baseline_path,
            _eval_lock_path(bind["baseline_tag"]),
        ))
    if (FINAL_FIRED_PATH.exists()
            and _eval_fired_path(bind["candidate_tag"]).exists()):
        allowed.extend((
            candidate_path,
            _eval_lock_path(bind["candidate_tag"]),
        ))
    residue = _find_final_eval_residue(allowed)
    _require(not residue, f"final 池出现未登记 eval residue:{residue}")

    try:
        current_bind, current_registry = _validate_final_opened_registry(
            bind_core)
        _require(
            current_bind == bind and current_registry == registry_evidence,
            "baseline 发车前 final registry 证据漂移",
        )
        baseline, baseline_sha = _run_eval_once(
            V28_ZIP, FINAL_POOL, bind["baseline_tag"],
            final_bind_sha256=bind_sha)
        current_bind, current_registry = _validate_final_opened_registry(
            bind_core)
        _require(
            current_bind == bind and current_registry == registry_evidence,
            "baseline 完成后 final registry 证据漂移",
        )
        _set_phase(
            state, "eval_final", "baseline-complete",
            bind_sha256=_sha256(FINAL_OPENED_PATH),
            final_pool_sha256=bind["final_pool_sha256"],
            final_registry_path=bind["final_registry_path"],
            final_registry_record_sha256=
                bind["final_registry_record_sha256"],
            baseline_archive_sha256=baseline_sha,
        )
        current_bind, current_registry = _validate_final_opened_registry(
            bind_core)
        _require(
            current_bind == bind and current_registry == registry_evidence,
            "candidate 发车前 final registry 证据漂移",
        )
        candidate_launch = _prepare_eval_launch(
            candidate, FINAL_POOL, bind["candidate_tag"],
            training_receipt=candidate_receipt,
            final_bind_sha256=bind_sha,
        )
        fired = {
            **_final_fired_core(bind, candidate_launch),
            "fired_at_ns": time.time_ns(),
        }
        if not FINAL_FIRED_PATH.exists():
            _write_json_exclusive(FINAL_FIRED_PATH, fired)
        else:
            _validate_final_fired(bind, candidate_launch)
            _require(
                candidate_path.exists(),
                "final candidate 已点火但无完整档案；为防观察部分输出后重试，"
                "本 campaign 禁止第二次发射",
            )
        candidate_doc, candidate_sha = _run_eval_once(
            candidate, FINAL_POOL, bind["candidate_tag"],
            training_receipt=candidate_receipt,
            final_bind_sha256=bind_sha,
            prepared_launch=candidate_launch)
        _validate_final_fired(bind, candidate_launch)
        current_bind, current_registry = _validate_final_opened_registry(
            bind_core)
        _require(
            current_bind == bind and current_registry == registry_evidence,
            "candidate 完成后 final registry 证据漂移",
        )
        analysis = _analyze_pair(
            baseline, baseline_sha,
            candidate_doc, candidate_sha,
            phase="final",
            death_margin=FINAL_DEATH_MARGIN,
            training_death_reward_evidence=candidate_receipt[
                "terminal_death_reward_evidence"],
        )
        _write_json_exclusive(FINAL_ANALYSIS_PATH, analysis)
        _require(_stable_json(FINAL_ANALYSIS_PATH) == analysis,
                 "final analysis 无法由冻结档案复验")
        verdict = analysis["verdict"]["status"]
        publication = None
        if verdict == "PASS":
            publication = _promote_passed_candidate(
                bind, candidate, analysis,
                baseline_sha=baseline_sha, candidate_sha=candidate_sha)
        else:
            _require(not PUBLISHED_DIR.exists()
                     and not PUBLISHED_DIR.is_symlink(),
                     "final FAIL 不得产生 published 工件")
        final_snapshot = _capture_final_evidence(
            bind, candidate, expected_analysis=analysis)
        final_identity = final_snapshot["identity"]
        _require(
            final_identity["baseline_archive_sha256"] == baseline_sha
            and final_identity["candidate_archive_sha256"] == candidate_sha
            and final_identity["final_registry_record_sha256"]
            == bind["final_registry_record_sha256"],
            "terminal state 提交前 final evidence 漂移",
        )
    except Exception as exc:
        state["terminal_status"] = "FINAL_OPERATIONAL_FAIL"
        _set_phase(
            state, "eval_final", "locked-failed",
            bind_sha256=_sha256(FINAL_OPENED_PATH),
            final_pool_sha256=bind["final_pool_sha256"],
            final_registry_path=bind["final_registry_path"],
            final_registry_record_sha256=
                bind["final_registry_record_sha256"],
            fired=FINAL_FIRED_PATH.exists(),
            retry_forbidden=FINAL_FIRED_PATH.exists(),
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json_atomic(STATE_PATH, state)
        raise

    state["terminal_status"] = (
        "PASS" if verdict == "PASS" else "FINAL_SCIENTIFIC_FAIL")
    _set_phase(
        state, "eval_final",
        "complete" if verdict == "PASS" else "scientific-fail",
        bind_sha256=final_identity["bind_sha256"],
        final_pool_sha256=final_identity["final_pool_sha256"],
        final_registry_path=final_identity["final_registry_path"],
        final_registry_record_sha256=
            final_identity["final_registry_record_sha256"],
        fired_sha256=final_identity["final_fired_sha256"],
        baseline_archive_sha256=
            final_identity["baseline_archive_sha256"],
        candidate_archive_sha256=
            final_identity["candidate_archive_sha256"],
        analysis_sha256=final_identity["analysis_sha256"],
        verdict=analysis["verdict"],
        publication=publication,
    )
    _write_json_atomic(STATE_PATH, state)
    print(f"R7 256-pair one-shot final verdict: {verdict}")
    _require(verdict == "PASS",
             f"R7 final 科学失败:{analysis['verdict']['failed_checks']}")


def command_status() -> None:
    _require_seed_discipline()
    state = _load_state()
    health: dict[str, Any] = {
        "state": state,
        "recipe_sha256": CAMPAIGN_RECIPE_SHA256,
        "seed_registry": _seed_registry_identity(),
    }
    try:
        health["implementation"] = {
            "status": "PASS", **_implementation_identity()}
    except Exception as exc:
        health["implementation"] = {
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    try:
        health["bc_v1"] = {"status": "PASS", **_bc_identity()}
    except Exception as exc:
        health["bc_v1"] = {
            "status": "NOT_READY", "error": f"{type(exc).__name__}: {exc}"}
    if DEVELOPMENT_DECISION_PATH.exists():
        try:
            health["development_decision"] = _stable_json(
                DEVELOPMENT_DECISION_PATH)
        except Exception as exc:
            health["development_decision"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if FINAL_ANALYSIS_PATH.exists():
        try:
            health["final_analysis"] = _stable_json(FINAL_ANALYSIS_PATH)
        except Exception as exc:
            health["final_analysis"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if state.get("terminal_status") == "PASS":
        try:
            health["publication_audit"] = {
                "status": "PASS",
                **_audit_terminal_publication(state),
            }
        except Exception as exc:
            health["publication_audit"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
    print(json.dumps(health, ensure_ascii=False, indent=2, allow_nan=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare-bc",
            "train-development",
            "eval-development",
            "train-production",
            "eval-final",
            "status",
        ],
    )
    args = parser.parse_args()
    if args.command == "status":
        command_status()
        return
    with _campaign_lock():
        {
            "prepare-bc": command_prepare_bc,
            "train-development": command_train_development,
            "eval-development": command_eval_development,
            "train-production": command_train_production,
            "eval-final": command_eval_final,
        }[args.command]()


if __name__ == "__main__":
    main()
