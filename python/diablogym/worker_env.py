"""v23 WorkerWindowEnv: in-place training environment for the FARM operating brain (docs/prereg/PREREG-v23.md).

One Gym episode = one real underlying game, not one FARM window.
  - reset(): fast-forward within the same game: the manager (numpy forward, argmax + manager mask) decides window by window,
    DIVE/RESUPPLY windows are run by the script's inner loop (the original OptionsEnv.step path, with identical bookkeeping);
    on FARM it opens the window, drains reflexes and hands the first reflex-free observation to the worker. If the base game dies/is truncated it
    rolls into a new game (including the natural cost of the spawn fast-forward). Cross-window wrapper state (stagnation clock/exhausted flag/
    fuse/previous option) is never reset; the state machine of the manager's 303-dim observation is the same code as OptionsEnv.
  - step(a): one worker tick + reflex tail drain (shared window core _win_step_worker).
    When FARM closes naturally, the same transition keeps advancing the frozen manager/script until the next
    learnable FARM; the policy reward equals the current FARM wage. Positive returns of the fast-forwarded windows in between and of the next window's opening
    reflexes only go to info/stats audits, so the combat/descend returns of a scripted DIVE are never
    misattributed to the last worker action; if the same transition really dies in the manager/script phase,
    only the explicit ``terminal-death-only`` mode passes back just the death penalty of the underlying terminal
    tick (the default ``none`` keeps the old return exactly).
    Natural window boundaries no longer fake a terminal; PPO bootstraps on the real next FARM state.
    A real underlying end is terminated; when the base game uses up 3000 steps at an idle decision boundary,
    only a safe boundary with progress is truncated. A boundary OptionsEnv proves to be a no-progress timeout
    is promoted at the Worker layer to terminated with a death-equivalent failure cost. If the limit happens
    to fall inside a walk/hit animation not encoded in the 298 dims, the underlying layer still fails closed as
    unsettled_budget_terminal, forbidding SB3 from a wrong TimeLimit bootstrap with an aliased
    terminal_observation.
  - Training seeds: the explicit sampler permanently rejects the historically burned BC pools [100,484),
    [1000,1384) and every currently registered BC/evaluation pool. The consumed protocol-v7 BC pools
    [2100000,2100128), [2101000,2101384) stay in the rejection table permanently; the current active
    v1/v2 pools are [2142000,2142128) and [2143000,2143384). The unified development/final-exam
    seed bank is [2110000,2130000), constructively disjoint from both generations of BC pools.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import time
import hashlib
import io
import json
import pathlib
import re
from types import MappingProxyType
from typing import Mapping

import gymnasium as gym
import numpy as np

from .env import (
    HUNT_SCOPES,
    REWARD_ECONOMY_V1,
    terminal_death_reward_component,
)
from .options_env import (
    DIVE,
    DUAL_WORKER_OBSERVATION_DIM,
    DUAL_WORKER_ACTION_MASK_SLICE,
    DUAL_WORKER_MANAGER_MASK_SLICE,
    DUAL_WORKER_V5_APPENDIX_DIM,
    FARM,
    KILL_PATIENCE,
    RESUPPLY,
    MANAGER_OBSERVATION_VIEW_LEGACY_V3,
    N_EXTRA_WORKER,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
    WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    WORKER_OBSERVATION_VIEW_RAW_V4,
    WORKER_OBSERVATION_VIEWS,
    WORKER_ACTION12_ENVIRONMENT_MASK,
    WORKER_ACTION12_PERMANENTLY_MASKED,
    OptionsEnv,
)

# Single source of truth: the training CLI's first-game seed/rank and the environment's automatic game rollover must both read this table.
# The 12000 pool was reserved before the protocol-v4 code freeze and may be used only once, after training, for a
# paired generalization evaluation; v7's large bank reserves the future frozen dev/final sub-pools in one go,
# so that growing a pool never changes the implementation identity. Leaving any block
# in the training sampling path would invalidate the "fresh hold-out" claim.
EVAL_RESERVED_SEED_RANGES = (
    (7000, 7032),
    (9000, 9032),
    (12000, 12032),
    (2_110_000, 2_130_000),
)
# These early BC pools were already opened by historical producers. They no longer authorize the current bc-v1/v2
# producer, but must stay in the ordinary training rejection domain permanently, so that an "old held-out" pool is not re-sampled
# and then wrongly labelled fresh. BC_RESERVED_SEED_RANGES also keeps the newer burned pools
# and registers the current active pools; it is the training rejection table, not the producer authorization table.
HISTORICAL_BURNED_BC_SEED_RANGES = (
    (100, 484),
    (1000, 1384),
)
BC_RESERVED_SEED_RANGES = (
    (2000, 2128),
    (3000, 3384),
    (2_100_000, 2_100_128),
    (2_101_000, 2_101_384),
    (2_102_000, 2_102_128),
    (2_103_000, 2_103_384),
    # 2026-07-27: 2_102_000..127 were burned once during the first collection by a crash on a missing a14 fuse receipt
    # (the marker was written; the discipline engaged correctly); 2_104_000..127 were then burned once by a candidate-gate FAIL
    # (the origin of amendment A2); the v1 active segment moved on to the 2_106 segment.
    (2_104_000, 2_104_128),
    (2_106_000, 2_106_128),
    (2_108_000, 2_108_128),
    # Registered 2026-07-27: 2_108/2_103 are void with the A3 implementation-bundle change;
    # the BC registration block 2_140_000-2_158_xxx is reserved (avoiding the evaluation bank's 2_11x segment),
    # active pair v1=2_140 / v2=2_141; each later pair is appended to the table as it is consumed.
    (2_140_000, 2_140_128),
    (2_141_000, 2_141_384),
    # 2026-07-27: 2_140/2_141 are void with the A4 implementation-bundle change; the table moves on (within the registration block).
    (2_142_000, 2_142_128),
    (2_143_000, 2_143_384),
)
TRAIN_RESERVED_SEED_RANGES = (
    *HISTORICAL_BURNED_BC_SEED_RANGES,
    *BC_RESERVED_SEED_RANGES,
    *EVAL_RESERVED_SEED_RANGES,
)
_MAX_EMPTY_FARM_EPISODES = 8
_FAST_FORWARD_REWARD_CREDIT_MODES = frozenset({
    "none",
    "terminal-death-only",
})
# R13 classroom reform: farm-only = the old rule (non-FARM windows fast-forwarded by script); farm-dive-v1 =
# live collection in DIVE windows (only for the sovereignty-transfer pre-registration recipe). RESUPPLY/skipped dry windows are always scripted.
_LEARNING_WINDOW_SCOPES = frozenset({
    "farm-only",
    "farm-dive-v1",
    "earned-dive-suffix-v1",
})
_LIVE_DIVE_WINDOW_SCOPES = frozenset({
    "farm-dive-v1", "earned-dive-suffix-v1",
})
_PREFIX_WORKER_SHA256 = (
    "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0")
_MAX_PREFIX_ZERO_TICK_WINDOWS = 32


class PrefixBudgetExceeded(RuntimeError):
    """A lifetime collection limit was reached; no learner terminal is emitted."""

    def __init__(self, reason, audit=None):
        self.reason = str(reason)
        self.audit = copy.deepcopy(audit)
        message = self.reason
        if audit is not None:
            message += "\nEARNED_PREFIX_AUDIT=" + json.dumps(
                audit, ensure_ascii=False, sort_keys=True,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray)
                else value.item() if isinstance(value, np.generic) else str(value))
        super().__init__(message)


# R15 readiness-gate coach: readiness-v1 = release by a tiered readiness table
# (weakest-link logic: no descent if any dimension falls short; an exhausted level is the escape hatch).
_MANAGER_HEURISTICS = frozenset({
    "level-margin-1",
    "readiness-v1",
    # R15 amendment 1: v1's exhausted escape hatch was freeloaded by the worker (deliberately not killing monsters to trigger the no-progress
    # exhausted flag and trick the descend gate open, kills 35->11, death rate 87%); v2 switches to a real-clear
    # escape: release only when ≤25% of this level's monsters are alive; the pass has to be earned by killing.
    "readiness-v2",
    # R16 amendment (audit cluster C1/C2): v3 = readiness table v0.2 (derived from engine arithmetic) + clear-out
    # by kill definition (the roster holds only live monsters, so v2's criterion was dead code).
    "readiness-v3",
})
# R16 readiness table v0.2 (audit finding C1: v0.1's L2 gate HP90 <=> clvl4 <=> 8040 XP exceeds the whole
# L1 experience pool; the AC column scale was off: a fresh warrior's AC 7 against the L1 gate 12 is only 0.58).
# Definition: the gate of level d = what a well-farmed worker can reach on level d-1 (L1-L4 from
# measurements/engine arithmetic; L5+ follow the v0.1 recurrence, to be recalibrated with deep-level data). HP column removed
# (under 3 vit : 2 str auto-allocation HP is a deterministic function of clvl, so it is redundant).
_READINESS_V2_CLVL = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 19)
_READINESS_V2_AC = (7, 9, 11, 13, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55, 60, 60)
_READINESS_V2_DMG = (5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 26)
_READINESS_V2_FIRE_RESIST = {13: 30, 14: 30, 15: 50, 16: 50}
_GOLEM_TYPE = 109   # 4 golem placeholder slots in the roster (mlvl12/exp0/hp1, never die)
# R17.0 gauge (review panel correction 2.3): bounded cap of the per-crossing telemetry of the escrow readiness gate
# (per env; the training arm descends ~244 times, 4096 is enough; once full it stops appending without judging).
_DESCEND_ESCROW_GATE_LOG_CAP = 4096


def readiness_power_ratio_v2(raw, target_dlvl) -> float:
    """R16 v0.2 combat power ratio (three-dimension weakest link + deep-level fire resistance). ≥1.0 releases."""
    d = max(1, min(16, int(target_dlvl)))
    i = d - 1
    ratios = [
        float(raw.get("char_level", 0)) / _READINESS_V2_CLVL[i],
        float(raw.get("armor_class", 0)) / _READINESS_V2_AC[i],
        (float(raw.get("item_max_damage", 0))
         + float(raw.get("damage_mod", 0))) / _READINESS_V2_DMG[i],
    ]
    fire_gate = _READINESS_V2_FIRE_RESIST.get(d)
    if fire_gate is not None:
        ratios.append(float(raw.get("fire_resist", 0)) / fire_gate)
    return min(ratios)


def roster_alive_count(raw) -> int:
    """Number of live monsters in this level's roster (excluding golem placeholder slots). The roster holds only live monsters."""
    return sum(
        1 for m in (raw.get("monsters") or [])
        if int(m.get("hp", 0)) > 0 and int(m.get("type", -1)) != _GOLEM_TYPE)


def readiness_clear_ratio(raw, kills_at_floor_entry) -> float:
    """R16 clear-out ratio by kill definition = kills on this level / (kills on this level + live non-golem monsters).

    The roster already drops the dead when the bridge exports it (C2), so kills on this level are counted with monster_kill_total relative to
    the base on entering the level; idling cannot raise this ratio, and the pass has to be earned by killing.
    """
    kills_here = max(
        0, int(raw.get("monster_kill_total", 0)) - int(kills_at_floor_entry))
    alive = roster_alive_count(raw)
    denom = kills_here + alive
    return (kills_here / denom) if denom > 0 else 1.0


def readiness_floor_cleared(raw, fraction: float = 0.25) -> bool:
    """R15 v2 escape criterion: share of live monsters on this level ≤ fraction (tolerating locked rooms/unreachable leftovers).

    monsters is the whole-level roster (the dead kept with hp≤0); idle freeloading cannot change the share;
    the only way to lower it is to really go and kill. Conservatively returns False if the roster is missing.
    """
    monsters = raw.get("monsters") or []
    total = len(monsters)
    if total <= 0:
        return False
    alive = sum(1 for m in monsters if int(m.get("hp", 0)) > 0)
    return alive <= max(2.0, fraction * total)
# R15 "tiered readiness table" v0.1 lower bounds (r15-READINESS-TABLE-draft.md, cross-checked against
# thirty years of strategy guides: Jarulf's mechanics guide + monstdat source + community build guides).
# index = dlvl-1 (target levels 1..16).
_READINESS_CLVL = (1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22)
_READINESS_HP = (70, 90, 100, 120, 140, 150, 160, 180,
                 200, 210, 220, 240, 260, 280, 300, 350)
_READINESS_AC = (12, 15, 30, 35, 40, 45, 50, 55,
                 60, 65, 70, 80, 90, 100, 110, 110)
_READINESS_DMG = (5, 8, 10, 12, 14, 16, 17, 19,
                  22, 26, 25, 27, 29, 31, 33, 33)
_READINESS_FIRE_RESIST = {13: 50, 14: 50, 15: 75, 16: 75}
_READINESS_MAGIC_RESIST = {15: 50, 16: 50}


def readiness_power_ratio(raw, target_dlvl) -> float:
    """R15 combat power ratio: min(four-dimension pass ratio, plus the resistance ratio on deep levels).

    ≥1.0 = readiness met, may release. Weakest-link logic: enough HP but not enough damage = cannot stun and gets mobbed,
    enough damage but not enough HP = wiped out by one wave; any weak link should block. Field names verified by a live probe
    (probe_r15_raw): char_level/max_hp/armor_class/
    item_max_damage+damage_mod/fire_resist/magic_resist.
    """
    d = max(1, min(16, int(target_dlvl)))
    i = d - 1
    ratios = [
        float(raw.get("char_level", 0)) / _READINESS_CLVL[i],
        float(raw.get("max_hp", 0)) / _READINESS_HP[i],
        float(raw.get("armor_class", 0)) / _READINESS_AC[i],
        (float(raw.get("item_max_damage", 0))
         + float(raw.get("damage_mod", 0))) / _READINESS_DMG[i],
    ]
    fire_gate = _READINESS_FIRE_RESIST.get(d)
    if fire_gate is not None:
        ratios.append(float(raw.get("fire_resist", 0)) / fire_gate)
    magic_gate = _READINESS_MAGIC_RESIST.get(d)
    if magic_gate is not None:
        ratios.append(float(raw.get("magic_resist", 0)) / magic_gate)
    return min(ratios)
_SEED_SCOPES = frozenset({"train", "bc-v1", "bc-v2", "replay"})
_CURRENT_BC_V1_RANGE = (2_142_000, 2_142_128)
_CURRENT_BC_V2_RANGE = (2_143_000, 2_143_384)
_LEGACY_BELT_FEATURE = 286
_LEGACY_EXHAUSTED_FEATURE = 297

# The Worker NPZ is a deployment protocol, not just six matrices without semantics. The manager NPZ keeps the historical
# six-member format; the 298->15 Worker must additionally carry this one canonical JSON scalar.
WORKER_NPZ_CONTRACT_MEMBER = "worker_contract_json"
WORKER_NPZ_SCHEMA = "diablogym-worker-npz/1"
WORKER_NPZ_ROLE = "worker"
WORKER_NPZ_REPRESENTATION = "plain-maskable-mlp-argmax"
_WORKER_WEIGHT_MEMBERS = frozenset({"w0", "b0", "w1", "b1", "wa", "ba"})
_WORKER_CONTRACT_FIELDS = frozenset({
    "schema",
    "role",
    "representation",
    "observation_view",
    "action12_mode",
    "source_checkpoint_sha256",
    "source_training_contract_sha256",
})
_WORKER_OBSERVATION_VIEWS = frozenset({
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    WORKER_OBSERVATION_VIEW_RAW_V4,
})
_WORKER_ACTION12_MODES = frozenset({
    WORKER_ACTION12_PERMANENTLY_MASKED,
    WORKER_ACTION12_ENVIRONMENT_MASK,
})
_LOWER_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_sha256(value: object, field: str, *, nullable: bool) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or _LOWER_SHA256_RE.fullmatch(value) is None:
        suffix = " or null" if nullable else ""
        raise ValueError(f"Worker NPZ metadata.{field} must be a 64-character lowercase SHA256{suffix}")


def validate_worker_npz_contract(contract: object) -> dict:
    """Validate and copy the complete Worker deployment contract.

    Observation selection happens before this network API is called.  In
    particular, ``legacy-v3`` means a complete protocol-v3 vector produced
    at the environment/raw-state boundary; it never claims that a v4
    298-vector can be losslessly decoded back into v3.
    """
    if not isinstance(contract, dict):
        raise ValueError("Worker NPZ metadata must be a JSON object")
    fields = set(contract)
    if fields != _WORKER_CONTRACT_FIELDS:
        missing = sorted(_WORKER_CONTRACT_FIELDS.difference(fields))
        extra = sorted(fields.difference(_WORKER_CONTRACT_FIELDS))
        raise ValueError(
            "Worker NPZ metadata fields must match the schema exactly; "
            f"missing={missing}, extra={extra}")
    checked = dict(contract)
    for field, expected in (
        ("schema", WORKER_NPZ_SCHEMA),
        ("role", WORKER_NPZ_ROLE),
        ("representation", WORKER_NPZ_REPRESENTATION),
    ):
        if checked[field] != expected:
            raise ValueError(
                f"Worker NPZ metadata.{field} must be {expected!r}, "
                f"got {checked[field]!r}")
    view = checked["observation_view"]
    if not isinstance(view, str) or view not in _WORKER_OBSERVATION_VIEWS:
        raise ValueError(
            "Worker NPZ metadata.observation_view must be one of "
            f"{sorted(_WORKER_OBSERVATION_VIEWS)}")
    action12_mode = checked["action12_mode"]
    if (not isinstance(action12_mode, str)
            or action12_mode not in _WORKER_ACTION12_MODES):
        raise ValueError(
            "Worker NPZ metadata.action12_mode must be one of "
            f"{sorted(_WORKER_ACTION12_MODES)}")
    _validate_sha256(
        checked["source_checkpoint_sha256"],
        "source_checkpoint_sha256",
        nullable=False,
    )
    _validate_sha256(
        checked["source_training_contract_sha256"],
        "source_training_contract_sha256",
        nullable=True,
    )
    return checked


def make_worker_npz_contract(
        *,
        observation_view: str,
        action12_mode: str,
        source_checkpoint_sha256: str,
        source_training_contract_sha256: str | None) -> dict:
    """Build the sole accepted Worker NPZ metadata schema."""
    return validate_worker_npz_contract({
        "schema": WORKER_NPZ_SCHEMA,
        "role": WORKER_NPZ_ROLE,
        "representation": WORKER_NPZ_REPRESENTATION,
        "observation_view": observation_view,
        "action12_mode": action12_mode,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_training_contract_sha256": source_training_contract_sha256,
    })


def canonical_worker_npz_contract_json(contract: object) -> str:
    """Return the only byte-level JSON spelling accepted inside a Worker NPZ."""
    return _canonical_json_text(validate_worker_npz_contract(contract))


def _parse_worker_npz_contract(value: np.ndarray, path: pathlib.Path) -> dict:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(
            f"{WORKER_NPZ_CONTRACT_MEMBER} of {path} must be a single JSON string scalar")
    scalar = array.item()
    if isinstance(scalar, bytes):
        try:
            encoded = scalar.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{WORKER_NPZ_CONTRACT_MEMBER} of {path} is not UTF-8") from exc
    elif isinstance(scalar, str):
        encoded = scalar
    else:
        raise ValueError(
            f"{WORKER_NPZ_CONTRACT_MEMBER} of {path} must be a JSON string")

    def reject_duplicate_keys(pairs):
        document = {}
        for key, item in pairs:
            if key in document:
                raise ValueError(f"Worker NPZ metadata contains duplicate field {key!r}")
            document[key] = item
        return document

    def reject_nonfinite(token):
        raise ValueError(f"Worker NPZ metadata forbids non-finite JSON number {token}")

    try:
        document = json.loads(
            encoded,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(
            f"{WORKER_NPZ_CONTRACT_MEMBER} of {path} is not strict JSON") from exc
    checked = validate_worker_npz_contract(document)
    canonical = _canonical_json_text(checked)
    if encoded != canonical:
        raise ValueError(
            f"{WORKER_NPZ_CONTRACT_MEMBER} of {path} must use canonical JSON")
    return checked


def legacy_worker_policy_observation_view(observation: np.ndarray) -> np.ndarray:
    """Decode the reversible A12 overlay into a protocol-v3 Worker row.

    Formal Worker boundaries first reconstruct every non-overlay feature from
    lossless native state.  This array-only helper therefore decodes only the
    two deliberately reversible overlay fields: packed belt slots at 286 and
    the signed drink latch at 297.  It must not be used to claim that an
    arbitrary filtered raw-v4 vector can recover the old monster/item channels.
    Arrays that predate this Worker contract are returned unchanged, matching
    ``leashed_ppo._legacy_worker_observation_view``.
    """
    value = np.asarray(observation)
    if value.ndim == 0 or value.shape[-1] <= _LEGACY_EXHAUSTED_FEATURE:
        return value
    legacy = value.copy()
    packed_belt = legacy[..., _LEGACY_BELT_FEATURE]
    legacy[..., _LEGACY_BELT_FEATURE] = (
        np.floor(np.clip(
            packed_belt * np.float32(8.0),
            np.float32(0.0),
            np.float32(8.0),
        ))
        / np.float32(8.0)
    )
    encoded_exhausted = legacy[..., _LEGACY_EXHAUSTED_FEATURE]
    legacy[..., _LEGACY_EXHAUSTED_FEATURE] = np.where(
        encoded_exhausted < np.float32(0.0),
        -encoded_exhausted - np.float32(1.0),
        encoded_exhausted,
    )
    return legacy

# E1 item 5A (docs/prereg/PREREG-v33-content-case.md): fixed offset of the dedicated p-draw stream; following the dry-anchor rng(26) precedent it takes 26,
# plus 2**33 to move out of the [0, 2**32) training seed domain, so the dedicated stream never shares a seed with any env training RNG.
_P_SKIP_SEED_OFFSET = 2**33 + 26


@dataclass(frozen=True)
class _AdvanceOutcome:
    """Complete result of the frozen manager advancing from a window boundary to the next learning state."""

    obs: np.ndarray | None
    reward: float
    terminated: bool
    truncated: bool
    extras: tuple[dict, ...]
    opening_reward: float = 0.0
    terminal_base_info: dict | None = None
    opening_recovery_action: int | None = None
    # Holds only the death penalty of the real terminal underlying tick, not the same tick's XP/damage/movement rewards, nor
    # any return of earlier manager/script windows. Always 0 for non-death terminals.
    terminal_death_reward: float = 0.0
    # OptionsEnv has proven that this budget boundary made no progress within KILL_PATIENCE. The Worker promotes
    # this kind of safe base TimeLimit to a non-bootstrappable policy-failure termination.
    timeout_without_progress: bool = False


def _coerce_p_skip(value) -> float:
    """E1 item 5A: the skip_dry bool is promoted to a skip probability p_skip (True≡1.0, False≡0.0, backward compatible)."""
    p = float(value)
    if not (np.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError(f"skip_dry/p_skip must be within [0, 1], got {value!r}")
    return p


def _coerce_fast_forward_reward_credit(value) -> str:
    """Accept only named credit modes, so a bool/typo cannot silently change the training objective."""
    if not isinstance(value, str) or value not in _FAST_FORWARD_REWARD_CREDIT_MODES:
        raise ValueError(
            "fast_forward_reward_credit must be one of "
            f"{sorted(_FAST_FORWARD_REWARD_CREDIT_MODES)}, got {value!r}")
    return value


def _coerce_learning_window_scope(value) -> str:
    """R13 classroom reform: only named classroom scopes are accepted; the default farm-only reproduces the old rule bit for bit."""
    if not isinstance(value, str) or value not in _LEARNING_WINDOW_SCOPES:
        raise ValueError(
            "learning_window_scope must be one of "
            f"{sorted(_LEARNING_WINDOW_SCOPES)}, got {value!r}")
    return value


def _coerce_depth_shaping_unit(value) -> float:
    """R13.2 option A: potential-based depth shaping unit (φ = unit × (dungeon_level − 1)).

    Only affects the worker policy_reward layer during training; 0.0 (default) = the old rule untouched.
    """
    unit = float(value)
    if not np.isfinite(unit) or not 0.0 <= unit <= 100.0:
        raise ValueError(
            f"depth_shaping_unit must be a finite number within [0,100], got {value!r}")
    return unit


def _coerce_additional_terminal_death_cost(value) -> float:
    """The extra death cost is configured as a non-negative cost and converted internally into a negative reward."""
    cost = float(value)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(
            "additional_terminal_death_cost must be a finite non-negative number, got "
            f"{value!r}")
    return cost


def _coerce_seed_scope(value) -> str:
    """Make permission to consume held-out seeds explicit at construction."""
    if not isinstance(value, str) or value not in _SEED_SCOPES:
        raise ValueError(
            f"seed_scope must be one of {sorted(_SEED_SCOPES)}, got {value!r}")
    return value


def _seed_in_half_open_range(seed: int, registered: tuple[int, int]) -> bool:
    return registered[0] <= int(seed) < registered[1]


def _derive_p_skip_rng(seed: int) -> np.random.Generator:
    """E1 item 3 per-env seeding: the dedicated p-draw stream is derived deterministically from the episode seed with a fixed offset (no contamination of the training RNG)."""
    return np.random.default_rng(int(seed) + _P_SKIP_SEED_OFFSET)


class NumpyManager:
    """numpy forward of the frozen v22-H (MlpPolicy(64,64) policy side; G0' reconciled bit for bit with SB3)."""

    def __init__(self, npz_path: str | pathlib.Path,
                 expected_sha256: str | None = None):
        path = pathlib.Path(npz_path)
        payload = path.read_bytes()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError("NumpyManager expected_sha256 must be a 64-character string")
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{path} SHA256 mismatch: {actual_sha256} != {expected_sha256}")
        # Hash and parse the exact same immutable byte string.  Opening the
        # path once for validation and again for np.load would leave a replace
        # race in which subprocesses could silently consume different brains.
        self.source_sha256 = actual_sha256
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            files = tuple(z.files)
            if len(files) != len(set(files)):
                raise ValueError(f"{path} NPZ contains duplicate members")
            members = set(files)
            missing = _WORKER_WEIGHT_MEMBERS.difference(members)
            if missing:
                raise ValueError(f"{path} is missing weights: {sorted(missing)}")
            allowed = _WORKER_WEIGHT_MEMBERS | {WORKER_NPZ_CONTRACT_MEMBER}
            extra = members.difference(allowed)
            if extra:
                raise ValueError(f"{path} contains unregistered NPZ members: {sorted(extra)}")
            self._worker_contract = (
                _parse_worker_npz_contract(
                    z[WORKER_NPZ_CONTRACT_MEMBER], path)
                if WORKER_NPZ_CONTRACT_MEMBER in members
                else None
            )
            self.w0, self.b0 = z["w0"].astype(np.float32), z["b0"].astype(np.float32)
            self.w1, self.b1 = z["w1"].astype(np.float32), z["b1"].astype(np.float32)
            self.wa, self.ba = z["wa"].astype(np.float32), z["ba"].astype(np.float32)
        if (self.w0.ndim != 2 or self.b0.shape != (self.w0.shape[0],)
                or self.w1.ndim != 2 or self.w1.shape[1] != self.w0.shape[0]
                or self.b1.shape != (self.w1.shape[0],)
                or self.wa.ndim != 2 or self.wa.shape[1] != self.w1.shape[0]
                or self.ba.shape != (self.wa.shape[0],)):
            raise ValueError(f"{path} weight shapes do not form a connectable MLP")
        if not all(np.isfinite(a).all()
                   for a in (self.w0, self.b0, self.w1, self.b1, self.wa, self.ba)):
            raise ValueError(f"{path} weights contain NaN/Inf")
        self._io_shape = (int(self.w0.shape[1]), int(self.wa.shape[0]))
        if self._worker_contract is not None and self._io_shape != (298, 15):
            raise ValueError(
                f"{path} carries Worker metadata, but its weight shape is "
                f"{self._io_shape[0]}→{self._io_shape[1]}; it must be 298→15")
        if self._worker_contract is not None:
            self._worker_contract = MappingProxyType(self._worker_contract)

    def require_io_shape(self, observation_dim: int, action_count: int,
                         label: str = "NumpyManager") -> None:
        actual = self._io_shape
        expected = (int(observation_dim), int(action_count))
        if actual != expected:
            raise ValueError(
                f"{label} input→action shape must be {expected[0]}→{expected[1]}, "
                f"actual {actual[0]}→{actual[1]}")

    @property
    def worker_contract(self) -> Mapping | None:
        """Immutable deployment metadata, or ``None`` for historical NPZs."""
        return self._worker_contract

    def require_worker_contract(self) -> Mapping:
        if self._io_shape != (298, 15):
            raise ValueError(
                "Worker NPZ weight shape must be 298→15, "
                f"actual {self._io_shape[0]}→{self._io_shape[1]}")
        if self._worker_contract is None:
            raise ValueError(
                "298→15 Worker NPZ lacks strict worker_contract_json metadata; "
                "old six-member NPZs may no longer be deployed silently")
        return self._worker_contract

    @property
    def worker_observation_view(self) -> str:
        return str(self.require_worker_contract()["observation_view"])

    @property
    def worker_action12_mode(self) -> str:
        return str(self.require_worker_contract()["action12_mode"])

    def _raw_logits(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.w0.shape[1],):
            raise ValueError(f"observation shape should be {(self.w0.shape[1],)}, got {obs.shape}")
        h = np.tanh(self.w0 @ obs + self.b0)
        h = np.tanh(self.w1 @ h + self.b1)
        logits = self.wa @ h + self.ba
        if not np.isfinite(logits).all():
            raise FloatingPointError("NumpyManager forward produced NaN/Inf logits")
        return logits

    def forensic_worker_logits(self, policy_obs: np.ndarray) -> np.ndarray:
        """Inspect an old contractless Worker only for parity/archaeology.

        No observation selection or deployment mask is applied.  Callers such
        as the historical BC state-dict adapter must separately declare the
        OptionsEnv view and apply its frozen mask contract; Worker NPZ
        deployment must use ``worker_logits``/``worker_callback`` instead.
        """
        actual = (int(self.w0.shape[1]), int(self.wa.shape[0]))
        if actual != (298, 15):
            raise ValueError(
                "forensic_worker_logits only accepts a 298→15 Worker, "
                f"actual {actual[0]}→{actual[1]}")
        return self._raw_logits(policy_obs)

    def _reject_generic_worker_inference(self) -> None:
        actual = getattr(
            self,
            "_io_shape",
            (int(self.w0.shape[1]), int(self.wa.shape[0])),
        )
        if actual == (298, 15):
            raise ValueError(
                "a 298→15 Worker must not call the generic logits/choose: "
                "use worker_logits/choose_worker and declare the provided observation_view explicitly")

    def logits(self, obs: np.ndarray) -> np.ndarray:
        self._reject_generic_worker_inference()
        return self._raw_logits(obs)

    def choose(self, obs: np.ndarray, mask: np.ndarray) -> int:
        lg = self.logits(obs)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != lg.shape:
            raise ValueError(f"mask shape should be {lg.shape}, got {mask.shape}")
        if not mask.any():
            raise ValueError("the action mask must not be all False")
        lg = np.where(mask, lg, -np.inf)
        return int(np.argmax(lg))

    def worker_logits(
            self,
            policy_obs: np.ndarray,
            *,
            observation_view: str) -> np.ndarray:
        """Forward an already-selected Worker policy observation.

        The explicit view label prevents callers from presenting a raw-v4
        vector to a legacy-v3 network (or vice versa).  This method performs
        no observation conversion because the complete v3 view must be built
        from the original raw game state at the environment boundary.
        """
        contract = self.require_worker_contract()
        expected_view = contract["observation_view"]
        if not isinstance(observation_view, str) or observation_view != expected_view:
            raise ValueError(
                "Worker policy observation_view mismatch: "
                f"the NPZ requires {expected_view!r}, the caller declared {observation_view!r}")
        return self._raw_logits(policy_obs)

    def worker_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """Apply the NPZ's action-12 deployment contract without mutation."""
        contract = self.require_worker_contract()
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (15,):
            raise ValueError(f"Worker action mask shape should be (15,), got {mask.shape}")
        policy_mask = mask.copy()
        if contract["action12_mode"] == WORKER_ACTION12_PERMANENTLY_MASKED:
            policy_mask[12] = False
        if not policy_mask.any():
            raise ValueError("the Worker action mask must not be all False after applying the deployment contract")
        return policy_mask

    def choose_worker(
            self,
            policy_obs: np.ndarray,
            raw_mask: np.ndarray,
            *,
            observation_view: str) -> int:
        logits = self.worker_logits(
            policy_obs,
            observation_view=observation_view,
        )
        policy_mask = self.worker_mask(raw_mask)
        return int(np.argmax(np.where(policy_mask, logits, -np.inf)))

    def worker_callback(self):
        """Return an OptionsEnv callback carrying its lossless view request."""
        view = self.worker_observation_view

        def policy(policy_obs: np.ndarray, raw_mask: np.ndarray) -> int:
            return self.choose_worker(
                policy_obs,
                raw_mask,
                observation_view=view,
            )

        # OptionsEnv reads this before constructing the callback's observation.
        # A plain lambda without this declaration would fall back to an
        # environment default and reintroduce a same-shape semantic mismatch.
        policy.diablogym_worker_observation_view = view
        policy.diablogym_worker_action12_mode = self.worker_action12_mode
        return policy


def is_reserved_eval_seed(seed: int) -> bool:
    """The training path must not consume any registered evaluation-pool seed."""
    value = int(seed)
    return any(lo <= value < hi for lo, hi in EVAL_RESERVED_SEED_RANGES)


def is_reserved_train_seed(seed: int) -> bool:
    """Whether an ordinary training reset would contaminate any held-out pool."""
    value = int(seed)
    return any(lo <= value < hi for lo, hi in TRAIN_RESERVED_SEED_RANGES)


def sample_train_seed(rng: np.random.Generator) -> int:
    """Training seed sampler: rejects the historically burned and all currently registered BC/evaluation pools."""
    while True:
        s = int(rng.integers(0, 2**31))
        if not is_reserved_train_seed(s):
            return s


class WorkerWindowEnv(gym.Env):
    """SB3 view: explicit policy view, Discrete(15), episode = one real underlying game.

    The historical bool still maps to the two 298-wide compatibility views; the lossless dual
    ``dual-v4-asymmetric-v3`` must be named by ``policy_observation_view``,
    so that an old CLI cannot silently widen the dimensions because a default changed.
    """

    metadata = {"render_modes": []}

    def __init__(self, manager_npz: str | None, max_steps: int = 3000,
                 rng_seed: int | None = None, log_windows: bool = False,
                 skip_dry: float | bool = False, manager_sha256: str | None = None,
                 manager_heuristic: str | None = None,
                 drink_sovereignty: bool = True,
                 legacy_policy_observation_view: bool = False,
                 policy_observation_view: str | None = None,
                 fast_forward_reward_credit: str = "none",
                 additional_terminal_death_cost: float = 0.0,
                 seed_scope: str = "train",
                 learning_window_scope: str = "farm-only",
                 depth_shaping_unit: float = 0.0,
                 worker_descend_bonus_fraction: float = 0.0,
                 worker_descend_escrow_fraction: float = 0.0,
                 worker_descend_escrow_power: float = 1.0,
                 worker_descend_escrow_readiness_gate: bool = False,
                 worker_descend_escrow_readiness_table: str = "v1",
                 worker_hp_loss_price: float = 0.0,
                 worker_potion_pickup_bonus: float = 0.0,
                 worker_no_progress_timeout_credit: str = "death-equivalent",
                 resource_protocol: str = "off",
                 resource_purchase_mode: str = "full",
                 resource_service_policy: str = "legacy-v1",
                 resource_readiness_law: str = "veto-v1",
                 worker_time_protocol: str = "legacy",
                 resource_retreat: str = "off",
                 resource_portal: str = "off",
                 resource_sweep: str = "off",
                 resource_identify: str = "off",
                 resource_weapon_upgrade: str = "off",
                 hunt_scope: str = "all",
                 prefix_worker=None,
                 prefix_worker_sha256: str | None = None,
                 prefix_max_attempts: int | None = None,
                 prefix_max_microsteps: int | None = None,
                 **env_kwargs):
        super().__init__()
        if "resource_calibration" in env_kwargs:
            raise ValueError("Resource calibration is diagnostic-only; use OptionsEnv, not WorkerWindowEnv")
        from .resource_protocol import validate_resource_config
        self.resource_protocol, self.resource_purchase_mode = validate_resource_config(
            resource_protocol, resource_purchase_mode)
        from .resource_sustain import validate_service_policy
        self.resource_service_policy = validate_service_policy(
            self.resource_protocol, self.resource_purchase_mode, resource_service_policy)
        env_kwargs["resource_protocol"] = self.resource_protocol
        env_kwargs["resource_purchase_mode"] = self.resource_purchase_mode
        if self.resource_service_policy != "legacy-v1":
            env_kwargs["resource_service_policy"] = self.resource_service_policy
        # R17.1 readiness rule 3: readiness law passthrough (default keeps kwargs byte-identical).
        from .resource_protocol import validate_readiness_law
        self.resource_readiness_law = validate_readiness_law(
            self.resource_protocol, resource_readiness_law)
        if self.resource_readiness_law != "veto-v1":
            env_kwargs["resource_readiness_law"] = self.resource_readiness_law
        # R18-B retreat-v1 passthrough (default off keeps kwargs byte-identical).
        from .resource_protocol import validate_retreat_protocol
        self.resource_retreat = validate_retreat_protocol(
            self.resource_protocol, self.resource_readiness_law, resource_retreat)
        if self.resource_retreat != "off":
            env_kwargs["resource_retreat"] = self.resource_retreat
        # R18-B5 (2026-09-07): portal-v1 passthrough (default off keeps the kwargs byte-identical).
        # The Scroll of Town Portal is the vehicle of retreat, not its neighbour, so the same deployment-side validator is reused:
        # portal-v1 requires l2-town-v1 + coach-v03 + retreat-v1, otherwise fail closed.
        from .resource_protocol import validate_portal_protocol
        self.resource_portal = validate_portal_protocol(
            self.resource_protocol, self.resource_readiness_law,
            self.resource_retreat, resource_portal)
        if self.resource_portal != "off":
            env_kwargs["resource_portal"] = self.resource_portal
        # R18-B6 (2026-09-07): sweep-v1 passthrough (default off keeps the kwargs byte-identical).
        # The training world must equal the tested world, so this reuses verbatim the one validator
        # used by the deployment-side OptionsEnv (options_env.py:477-481): the chest sweep only holds inside the
        # sustain-loot-v1 loot economy of l2-town-v1 (it only creates drops; collecting them is the
        # loot economy's job), otherwise fail closed.
        from .resource_protocol import validate_sweep_protocol
        self.resource_sweep = validate_sweep_protocol(
            self.resource_protocol, self.resource_service_policy, resource_sweep)
        if self.resource_sweep != "off":
            env_kwargs["resource_sweep"] = self.resource_sweep
        # R18-B6 (2026-09-07): cain-v1 passthrough (default off keeps the kwargs byte-identical).
        # Again the one deployment-side validator (options_env.py:483-487): the identify leg is
        # a segment of the loot town trip and only holds under sustain-loot-v1.
        from .resource_identify import validate_identify_protocol
        self.resource_identify = validate_identify_protocol(
            self.resource_protocol, self.resource_service_policy, resource_identify)
        if self.resource_identify != "off":
            env_kwargs["resource_identify"] = self.resource_identify
        # R18-B6 (2026-09-07): smith-v1 passthrough (default off keeps the kwargs byte-identical).
        # The same deployment-side validator (options_env.py:489-494); on the native side
        # configure_resource_weapon_purchase is written uniformly by DiabloGymEnv's
        # _configure_native_resource_protocol (env.py:2341-2352), and through this kwargs passthrough
        # the training environment takes exactly the deployment environment's configure path.
        from .resource_weapon_upgrade import validate_weapon_upgrade
        self.resource_weapon_upgrade = validate_weapon_upgrade(
            self.resource_protocol, self.resource_service_policy,
            self.resource_purchase_mode, resource_weapon_upgrade)
        if self.resource_weapon_upgrade != "off":
            env_kwargs["resource_weapon_upgrade"] = self.resource_weapon_upgrade
        # R18-B5 (2026-09-07): hunt_scope passthrough (default all keeps the kwargs byte-identical).
        # DiabloGymEnv has no separate validator function (the rule is inlined in env.py's constructor),
        # so this copies the same wording verbatim; it is a DiabloGymEnv-level switch and needs no resource protocol.
        # Review correction: the vocabulary itself is no longer copied; it refers directly to the one env.HUNT_SCOPES.
        if hunt_scope not in HUNT_SCOPES:
            raise ValueError(f"Unknown hunt_scope {hunt_scope!r}; expected all or l1-only")
        self.hunt_scope = hunt_scope
        if self.hunt_scope != "all":
            env_kwargs["hunt_scope"] = self.hunt_scope
        # R13 classroom reform main flag: the default farm-only reproduces the old rule bit for bit (non-FARM windows
        # fast-forwarded by script); farm-dive-v1 makes DIVE windows first-class live learning windows and hands
        # in-window sovereignty (a11/stepping) to OptionsEnv.
        self.learning_window_scope = _coerce_learning_window_scope(
            learning_window_scope)
        # R18-B2: every immutable completion-l2 recipe is accepted here; the
        # completion-l2-v1 path keeps the identical guards and kwargs.
        from .completion_clock import COMPLETION_PROTOCOLS
        if worker_time_protocol not in ("legacy", *COMPLETION_PROTOCOLS):
            raise ValueError("Unknown worker_time_protocol")
        self.worker_time_protocol = worker_time_protocol
        if worker_time_protocol != "legacy":
            if self.learning_window_scope != "earned-dive-suffix-v1":
                raise ValueError("completion-l2 protocols require earned-dive-suffix-v1")
            env_kwargs["worker_time_protocol"] = worker_time_protocol
        prefix_values = (prefix_worker, prefix_worker_sha256,
                         prefix_max_attempts, prefix_max_microsteps)
        if self.learning_window_scope == "earned-dive-suffix-v1":
            # R18-B3 (2026-09-07): design decision: selling gear in town is the core of the snowball plan;
            # sustain-loot-v1 (the two-trip L1 loot economy) is a first-class training service law alongside
            # sustain-v6; both require l2-town-v1/full. loot only holds under the completion clock
            # (the old legacy clock has no collect command window/service micro-step budget), so loot +
            # legacy always fails closed: the training world must equal the tested world.
            if (self.resource_protocol != "l2-town-v1"
                    or self.resource_purchase_mode != "full"
                    or self.resource_service_policy not in ("sustain-v6", "sustain-loot-v1")):
                raise ValueError(
                    "earned-dive-suffix-v1 requires l2-town-v1/full/sustain-v6 or sustain-loot-v1")
            if (self.resource_service_policy == "sustain-loot-v1"
                    and self.worker_time_protocol not in COMPLETION_PROTOCOLS):
                raise ValueError("sustain-loot-v1 requires an explicit completion-l2 time protocol")
            if (not callable(prefix_worker)
                    or prefix_worker_sha256 != _PREFIX_WORKER_SHA256
                    or getattr(prefix_worker, "source_sha256", None)
                    != prefix_worker_sha256):
                raise ValueError("earned prefix requires the exact verified original R16 callback")
            if (not callable(getattr(prefix_worker, "episode_reseed", None))
                    or getattr(prefix_worker, "on_beat", None) is not None):
                raise ValueError("prefix callback requires episode_reseed and no on_beat hook")
            if (getattr(prefix_worker, "diablogym_worker_observation_view", None)
                    != WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC
                    or getattr(prefix_worker, "diablogym_worker_action12_mode", None)
                    != WORKER_ACTION12_ENVIRONMENT_MASK
                    or not drink_sovereignty):
                raise ValueError("prefix callback requires dual-v4-asymmetric-v3/environment-mask")
            for name, value in (("prefix_max_attempts", prefix_max_attempts),
                                ("prefix_max_microsteps", prefix_max_microsteps)):
                if (isinstance(value, (bool, np.bool_))
                        or not isinstance(value, (int, np.integer)) or int(value) <= 0):
                    raise ValueError(f"{name} requires an explicit positive integer")
            if _coerce_p_skip(skip_dry) != 0.0 or "workers" in env_kwargs:
                raise ValueError("earned prefix forbids skip_dry and external workers")
            self.prefix_worker = prefix_worker
            self.prefix_worker_sha256 = prefix_worker_sha256
            self.prefix_max_attempts = int(prefix_max_attempts)
            self.prefix_max_microsteps = int(prefix_max_microsteps)
        elif any(value is not None for value in prefix_values):
            raise ValueError("prefix arguments require earned-dive-suffix-v1")
        # R13.2 option A (approved 2026-08-31): potential-based depth shaping.
        # F = unit × Δdepth per transition; a true terminal refunds −unit×(d−1)
        # (φ(absorbing)=0); truncation gives no refund and is absorbed by bootstrapping. Only enters the worker's
        # policy_reward, never the manager ledger/wage identity/exam scoring; the wage-stripping rule
        # (descend bonus belongs to the manager) is untouched, and policy invariance is backed by the potential-function theorem.
        self.depth_shaping_unit = _coerce_depth_shaping_unit(
            depth_shaping_unit)
        if (self.depth_shaping_unit > 0.0
                and self.learning_window_scope not in _LIVE_DIVE_WINDOW_SCOPES):
            raise ValueError(
                "depth_shaping_unit only applies to the farm-dive-v1 classroom"
                " (the old rule has no live descent to shape)")
        # R14 option B: bounded refund of the descend bonus, only in the farm-dive-v1 classroom (the rule-domain check
        # is enforced a second time by OptionsEnv); for bookkeeping/identities see options_env._win_beat.
        if (float(worker_descend_bonus_fraction) > 0.0
                and self.learning_window_scope not in _LIVE_DIVE_WINDOW_SCOPES):
            raise ValueError(
                "worker_descend_bonus_fraction only applies to the farm-dive-v1 classroom")
        # R14.2 option D (adopted after option B failed): escrowed payout of the descend fee.
        # The descend fee of a new deepest level × fraction goes into escrow; at the next non-death window close it vests into
        # policy_reward; death (live or in a fast-forward segment) forfeits all of it. The same component as the R11-certified
        # kill lock's "forfeiting anti-suicide valve"; the worker controls its own survival, so it is learnable.
        # Runs only through the policy_reward layer: the W ledger/identities/manager ledger are untouched.
        _ef = float(worker_descend_escrow_fraction)
        if not np.isfinite(_ef) or not 0.0 <= _ef <= 1.0:
            raise ValueError(
                "worker_descend_escrow_fraction must be within [0,1], "
                f"got {worker_descend_escrow_fraction!r}")
        if _ef > 0.0 and self.learning_window_scope not in _LIVE_DIVE_WINDOW_SCOPES:
            raise ValueError(
                "worker_descend_escrow_fraction only applies to the farm-dive-v1 classroom")
        if _ef > 0.0 and float(worker_descend_bonus_fraction) > 0.0:
            raise ValueError(
                "option B (bonus_fraction) and option D (escrow_fraction) are mutually exclusive "
                "and must not be enabled together")
        self.descend_escrow_fraction = _ef
        # R14.3 option E (design goal: find a usable curve): convex escrow wage curve.
        # Escrow per level = fraction × unit × d^power; power=1.0 takes the original integer-sum
        # path, identical to the old one bit for bit. The measured risk h(d)≈0.23×1.53^(d−1) rises geometrically, so a linear
        # wage must stall at a break-even level; the convexity must catch up with the risk convexity (fit p≈1.6).
        _ep = float(worker_descend_escrow_power)
        if not np.isfinite(_ep) or not 1.0 <= _ep <= 3.0:
            raise ValueError(
                "worker_descend_escrow_power must be within [1,3], "
                f"got {worker_descend_escrow_power!r}")
        if _ep != 1.0 and _ef <= 0.0:
            raise ValueError(
                "worker_descend_escrow_power ≠ 1 only takes effect together with escrow (escrow_fraction"
                ">0)")
        self.descend_escrow_power = _ep
        # R15 amendment 2 (wage-side enforcement): readiness-conditional escrow: escrow is booked only for
        # descents with readiness>=1. The mask rule (exhausted forced hand-over) can force the worker into
        # DIVE windows, but an under-ready descent puts nothing into escrow: force can move it, but cannot be gamed for money.
        self.descend_escrow_readiness_gate = bool(
            worker_descend_escrow_readiness_gate)
        if (self.descend_escrow_readiness_gate
                and self.descend_escrow_fraction <= 0.0):
            raise ValueError(
                "descend_escrow_readiness_gate only takes effect together with "
                "escrow (escrow_fraction>0)")
        # R17.0 amendment 1 (review panel correction 2.1): ruler selector of the escrow readiness gate.
        # v1 = the existing readiness_power_ratio (v0.1 table, old rule bit for bit); v2 =
        # readiness_power_ratio_v2 (v0.2 table, the same ruler as the readiness-v3 coach).
        # Default v1 has zero drift; v2 only takes effect together with the readiness gate (otherwise the ruler has no use).
        _table = str(worker_descend_escrow_readiness_table)
        if _table not in ("v1", "v2"):
            raise ValueError(
                "worker_descend_escrow_readiness_table must be v1/v2, got "
                f"{worker_descend_escrow_readiness_table!r}")
        if _table != "v1" and not self.descend_escrow_readiness_gate:
            raise ValueError(
                "worker_descend_escrow_readiness_table=v2 only takes effect together with the readiness gate"
                " (descend_escrow_readiness_gate)")
        self.descend_escrow_readiness_table = _table
        self._descend_escrow = 0.0
        # R16 amendment (audit cluster C7/C8): HP economy pricing and terminal bookkeeping fixes.
        # hp_loss_price: deduct price per HP point lost (asymmetric: only HP loss is penalized, healing is not rewarded;
        # symmetric potential-style bookkeeping does not change the optimum, see the option-A lesson);
        # potion_pickup_bonus: a successful a13 potion pickup is rewarded (the belt count rises);
        # no_progress_timeout_credit="zero": a no-progress timeout is no longer penalized as death-equivalent
        # (the survival premium is no longer negative). All three default to off = old rule unchanged bit for bit.
        _hp_price = float(worker_hp_loss_price)
        _pot_bonus = float(worker_potion_pickup_bonus)
        if not np.isfinite(_hp_price) or _hp_price < 0.0:
            raise ValueError(
                f"worker_hp_loss_price must be a finite non-negative number, got {_hp_price!r}")
        if not np.isfinite(_pot_bonus) or _pot_bonus < 0.0:
            raise ValueError(
                "worker_potion_pickup_bonus must be a finite non-negative number, got "
                f"{_pot_bonus!r}")
        if worker_no_progress_timeout_credit not in (
                "death-equivalent", "zero"):
            raise ValueError(
                "worker_no_progress_timeout_credit must be death-equivalent/"
                f"zero, got {worker_no_progress_timeout_credit!r}")
        self.hp_loss_price = _hp_price
        self.potion_pickup_bonus = _pot_bonus
        self.no_progress_timeout_credit = worker_no_progress_timeout_credit
        # The default none reproduces exactly the old policy return where "all fast-forward rewards are audit-only";
        # terminal-death-only passes only the existing death penalty across the frozen-policy boundary. The extra death
        # cost is independent of this mode and is appended exactly once on a direct/fast-forwarded real death.
        self.fast_forward_reward_credit = (
            _coerce_fast_forward_reward_credit(fast_forward_reward_credit))
        self.additional_terminal_death_cost = (
            _coerce_additional_terminal_death_cost(
                additional_terminal_death_cost))
        self.seed_scope = _coerce_seed_scope(seed_scope)
        legacy_view = bool(legacy_policy_observation_view)
        if policy_observation_view is None:
            resolved_policy_view = (
                WORKER_OBSERVATION_VIEW_LEGACY_V3
                if legacy_view else WORKER_OBSERVATION_VIEW_A12_OVERLAY
            )
        else:
            if (
                not isinstance(policy_observation_view, str)
                or policy_observation_view not in WORKER_OBSERVATION_VIEWS
            ):
                raise ValueError(
                    "policy_observation_view must be one of "
                    f"{sorted(WORKER_OBSERVATION_VIEWS)}, got "
                    f"{policy_observation_view!r}")
            resolved_policy_view = policy_observation_view
            if (
                legacy_view
                and resolved_policy_view
                != WORKER_OBSERVATION_VIEW_LEGACY_V3
            ):
                raise ValueError(
                    "legacy_policy_observation_view=True conflicts with the explicit "
                    f"policy_observation_view={resolved_policy_view!r}")
        self.policy_observation_view = resolved_policy_view
        if (self.learning_window_scope == "earned-dive-suffix-v1"
                and resolved_policy_view != WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC):
            raise ValueError("earned prefix requires the existing dual-v4-asymmetric-v3 learner view")
        # Retain this public compatibility attribute for existing contracts
        # and probes.  New code must bind the string-valued view above.
        self.legacy_policy_observation_view = (
            resolved_policy_view == WORKER_OBSERVATION_VIEW_LEGACY_V3)
        # R12 amendment 2: scripted coach (the repository's canonical heuristic, literally the margin-1 version of the hypothesis-gate
        # scripted-teacher rule). The demon coach's structural failure of zero FARM windows (ledger
        # 2026-08-30) proved that deep classrooms need a coach that "starts class after diving down".
        if manager_heuristic is not None:
            if manager_heuristic not in _MANAGER_HEURISTICS:
                raise ValueError(
                    "manager_heuristic only supports "
                    f"{sorted(_MANAGER_HEURISTICS)}, got "
                    f"{manager_heuristic!r}")
            if manager_npz is not None:
                raise ValueError("manager_heuristic and manager_npz are mutually exclusive")
            self.mgr = None
            self.manager_heuristic = manager_heuristic
        else:
            if manager_npz is None:
                raise ValueError("either manager_npz or manager_heuristic must be given")
            self.manager_heuristic = None
            self.mgr = NumpyManager(manager_npz, expected_sha256=manager_sha256)
            self.mgr.require_io_shape(303, 3, "WorkerWindow manager")
        self.oe = OptionsEnv(max_steps=max_steps,
                             drink_sovereignty=drink_sovereignty,
                             worker_observation_view=(
                                 self.policy_observation_view),
                             manager_observation_view=(
                                 MANAGER_OBSERVATION_VIEW_LEGACY_V3),
                             dive_live_sovereignty=(
                                 self.learning_window_scope
                                 in _LIVE_DIVE_WINDOW_SCOPES),
                             worker_descend_bonus_fraction=(
                                 float(worker_descend_bonus_fraction)),
                             **env_kwargs)
        if (self.depth_shaping_unit > 0.0
                or self.descend_escrow_fraction > 0.0):
            # The argument of the monotone potential/escrow, _econ_episode_max_depth, is maintained per tick only under v2+
            # economies; under v1 it is always 0, which would make it silently idle, so this fails closed.
            _econ = getattr(self.oe.env, "reward_economy", None)
            if getattr(_econ, "name", "v1") == "v1":
                raise ValueError(
                    "depth_shaping_unit/descend_escrow_fraction depend on the v2+ "
                    "economy's per-episode deepest-level tracker; refused under the v1 economy")
        base = self.oe.env.observation_space.shape[0]
        if (
            self.policy_observation_view
            == WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC
        ):
            observation_dim = DUAL_WORKER_OBSERVATION_DIM
        elif (
            self.policy_observation_view
            == WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE
        ):
            observation_dim = (
                DUAL_WORKER_OBSERVATION_DIM + DUAL_WORKER_V5_APPENDIX_DIM)
        else:
            observation_dim = base + N_EXTRA_WORKER
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(observation_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(15)
        self._rng = np.random.default_rng(rng_seed)
        self._alive = False
        self._episode_seed: int | None = None
        self.log_windows = log_windows
        # v26 oasis -> E1 item 5A: skip probability p_skip for dry-level revisit windows (bool promoted to float; 1.0/0.0
        # keep the original boolean semantics bit for bit); the p draw uses its own dedicated stream _p_rng, no contamination of the training RNG (_rng).
        self.skip_dry = _coerce_p_skip(skip_dry)
        # VecEnv may auto-reset a terminated environment before callbacks
        # regain control.  A rollout-boundary curriculum change therefore
        # cannot be applied safely by an after-step callback.  The callback
        # registers the next probability plus a per-environment Worker-step
        # countdown; reset() deliberately preserves both fields.
        self._pending_skip_dry_probability: float | None = None
        self._pending_skip_dry_remaining_env_steps = 0
        self._p_rng = (_derive_p_skip_rng(rng_seed) if rng_seed is not None
                       else np.random.default_rng())
        self.window_log = []      # with log_windows=True: all windows (including fast-forwarded ones) are logged in order
        self.stats = {"windows": 0, "dry": 0, "fresh": 0, "ff_windows": 0,
                      "ff_dry": 0, "episodes": 0, "reseeds": 0,
                      "reasons": {}, "ff_reasons": {}}
        # The old stats keys are kept; three reward/termination ledgers are appended so that the initial reset fast-forward and the
        # fast-forward inside a worker transition can be audited, no longer mixing up hidden consequences.
        self.stats.update({"ff_terminals": 0, "transition_ff_reward": 0.0,
                           "reset_ff_reward": 0.0, "manual_ff_reward": 0.0})
        self.stats.update({
            "interrupted_resets": 0,
            "manual_ff_calls": 0,
            "direct_terminal_deaths": 0,
            "transition_ff_terminal_deaths": 0,
            "reset_ff_terminal_deaths": 0,
            "manual_ff_terminal_deaths": 0,
            "direct_existing_terminal_death_reward": 0.0,
            "direct_additional_terminal_death_reward": 0.0,
            "transition_ff_terminal_death_reward": 0.0,
            "transition_ff_additional_terminal_death_reward": 0.0,
            "credited_ff_terminal_death_reward": 0.0,
            "reset_ff_terminal_death_reward": 0.0,
            "reset_ff_additional_terminal_death_reward": 0.0,
            "additional_terminal_death_reward": 0.0,
            "direct_no_progress_timeouts": 0,
            "transition_ff_no_progress_timeouts": 0,
            "reset_ff_no_progress_timeouts": 0,
            "manual_ff_no_progress_timeouts": 0,
            "direct_no_progress_timeout_failure_reward": 0.0,
            "transition_ff_no_progress_timeout_failure_reward": 0.0,
            "reset_ff_no_progress_timeout_failure_reward": 0.0,
            "manual_ff_no_progress_timeout_failure_reward": 0.0,
            "credited_no_progress_timeout_failure_reward": 0.0,
        })

        if self.learning_window_scope == "earned-dive-suffix-v1":
            self._initialize_prefix_state()

    # ---- housekeeping ----
    def _policy_observation(self, observation: np.ndarray) -> np.ndarray:
        """Apply the explicitly selected policy view at the lossless edge."""
        # A few pure transition probes instantiate the environment with
        # ``object.__new__`` to avoid native-engine setup.  Missing means the
        # constructor's A12-overlay default.  Real environments always rebuild
        # the complete v3 base from ``oe.env._raw``; the array-only fallback is
        # intentionally limited to synthetic transition tests.
        view = getattr(self, "policy_observation_view", None)
        if view is None:
            # Compatibility for pure transition probes created with
            # ``object.__new__`` and for pre-selector callers.
            view = (
                WORKER_OBSERVATION_VIEW_LEGACY_V3
                if getattr(
                    self, "legacy_policy_observation_view", False)
                else WORKER_OBSERVATION_VIEW_A12_OVERLAY
            )
        options_env = getattr(self, "oe", None)
        if (
            options_env is not None
            and callable(getattr(
                options_env, "_worker_policy_observation", None))
            and getattr(getattr(options_env, "env", None), "_raw", None)
            is not None
        ):
            return options_env._worker_policy_observation(
                view,
                skip_dry_probability=(
                    getattr(self, "skip_dry", 0.0)
                    if view in (
                        WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
                        WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
                    )
                    else 0.0
                ),
            )
        if view in (
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
            WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
        ):
            raise RuntimeError(
                f"{view} requires lossless active OptionsEnv raw")
        if view == WORKER_OBSERVATION_VIEW_LEGACY_V3:
            return legacy_worker_policy_observation_view(observation)
        return np.asarray(observation)

    def _new_episode(self, seed=None, options=None):
        if self._earned_prefix_enabled():
            self._prefix_require_capacity()
        if seed is None:
            if self.seed_scope in {"bc-v1", "bc-v2"}:
                raise RuntimeError(
                    f"{self.seed_scope}: automatic rollover into an unregistered seed after a demonstration game is forbidden")
            s = sample_train_seed(self._rng)
        else:
            s = int(seed)
            if self.seed_scope == "train" and is_reserved_train_seed(s):
                raise ValueError(f"training Worker reset refuses reserved seed {s}")
            expected = (
                _CURRENT_BC_V1_RANGE
                if self.seed_scope == "bc-v1"
                else _CURRENT_BC_V2_RANGE
                if self.seed_scope == "bc-v2"
                else None
            )
            if expected is not None and not _seed_in_half_open_range(
                    s, expected):
                raise ValueError(
                    f"{self.seed_scope} only allows the registered pool "
                    f"[{expected[0]},{expected[1]}), got {s}")
        # p_skip is derived per "real underlying game", not derived once at an external reset(seed) and
        # consumed continuously across auto-resets. This way any episode_seed can
        # replay its course draws on its own, independent of how many games/dry windows came before.
        self._p_rng = _derive_p_skip_rng(s)
        try:
            if self._earned_prefix_enabled():
                self._prefix_begin_attempt(s)
            self.oe.reset(seed=s, options=options)
            if self._earned_prefix_enabled():
                self._prefix_after_reset()
        except Exception as exc:
            if self._earned_prefix_enabled():
                self._prefix_abort_error(exc)
            raise
        self._episode_seed = s
        self._alive = True
        # R14.2 option D: the escrow balance is cleared per game (death forfeits/terminal vests were settled in the previous game;
        # defensive clearing prevents leaks across games).
        self._descend_escrow = 0.0
        self.stats["episodes"] += 1

    def _log(self, extra, fast_forward: bool):
        if self.log_windows:
            self.window_log.append(dict(extra, ff=fast_forward))
        reason = str(extra.get("reason", "unknown"))
        reasons = self.stats["reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        if fast_forward:
            self.stats["ff_windows"] += 1
            ff_reasons = self.stats["ff_reasons"]
            ff_reasons[reason] = ff_reasons.get(reason, 0) + 1

    def _mgr_choose(self) -> int:
        if getattr(self, "resource_protocol", "off") != "off":
            return self.oe.resource_option_choice()
        if self.manager_heuristic is not None:
            # level-margin-1: dive if exhausted or one level higher, otherwise farm;
            # readiness-v1 (R15): dive only when the readiness table's weakest link is met; exhausted is the escape hatch
            # (an exhausted floor = readiness cannot improve, best-effort descent). The mask fallback follows the
            # legal order FARM->DIVE->RESUPPLY, the same as the G0 probe.
            raw = self.oe.env._raw
            mask = self.oe.action_masks()
            if self.manager_heuristic == "readiness-v3":
                # R16: v0.2 readiness table + kill-definition clear-out escape. The level-entry base resets on
                # dungeon_level changes (getattr tolerates the missing attribute in __new__ shells).
                # G0-3/4/5 calibration decision: a threshold of 0.75 opens DIVE windows while 25% of monsters on the level are still alive,
                # and the a11 main line gets blocked by monsters -> stalls (39 stalls in 57 windows); changed to 1.0
                # (dive as soon as the roster has no live monsters, without waiting for the exhausted flag's KILL_PATIENCE);
                # unreachable leftovers are still covered by the exhausted flag.
                dlvl = int(raw["dungeon_level"])
                floor_state = getattr(self, "_readiness_floor_state", None)
                if floor_state is None or floor_state[0] != dlvl:
                    floor_state = (dlvl, int(raw.get("monster_kill_total", 0)))
                    self._readiness_floor_state = floor_state
                ready = readiness_power_ratio_v2(raw, dlvl + 1) >= 1.0
                cleared = readiness_clear_ratio(raw, floor_state[1]) >= 1.0
                want = DIVE if (ready or cleared) else FARM
            elif self.manager_heuristic == "readiness-v2":
                ready = readiness_power_ratio(
                    raw, int(raw["dungeon_level"]) + 1) >= 1.0
                want = (DIVE if (ready or readiness_floor_cleared(raw))
                        else FARM)
            elif self.manager_heuristic == "readiness-v1":
                ready = readiness_power_ratio(
                    raw, int(raw["dungeon_level"]) + 1) >= 1.0
                want = (DIVE if (ready or self.oe.exhausted) else FARM)
            else:
                want = (DIVE if (self.oe.exhausted
                                 or int(raw["char_level"])
                                 >= int(raw["dungeon_level"]) + 1)
                        else FARM)
            if mask[want]:
                return int(want)
            for cand in (FARM, DIVE, RESUPPLY):
                if mask[cand]:
                    return int(cand)
            raise RuntimeError("all three manager actions are masked; the protocol invariant is broken")
        mobs = self.oe._mgr_obs(self.oe._last_base_obs)
        return self.mgr.choose(mobs, self.oe.action_masks())

    def set_skip_dry_p(self, p: float | bool) -> float:
        """Set the live probability only while no boundary switch is armed."""
        pending = getattr(
            self, "_pending_skip_dry_probability", None)
        remaining = getattr(
            self, "_pending_skip_dry_remaining_env_steps", 0)
        if pending is not None or remaining != 0:
            raise RuntimeError(
                "a rollout-boundary p_skip switch countdown already exists; "
                "directly overwriting the current probability is forbidden")
        self.skip_dry = _coerce_p_skip(p)
        return self.skip_dry

    def schedule_skip_dry_p(
            self, p: float | bool, remaining_env_steps: int) -> dict:
        """Register an atomic probability switch after N Worker actions.

        The countdown is local to one environment, not the global VecEnv
        sample count.  The Nth action and its reflex tail run under the current
        value; the pending value becomes live immediately afterwards, before
        next-state construction, next-window selection, or VecEnv auto-reset.
        """
        if (
            not isinstance(remaining_env_steps, (int, np.integer))
            or isinstance(remaining_env_steps, (bool, np.bool_))
            or int(remaining_env_steps) <= 0
        ):
            raise ValueError(
                "rollout-boundary p_skip remaining_env_steps must be a positive integer")
        pending = getattr(
            self, "_pending_skip_dry_probability", None)
        remaining = getattr(
            self, "_pending_skip_dry_remaining_env_steps", 0)
        if pending is not None or remaining != 0:
            raise RuntimeError(
                "a rollout-boundary p_skip switch is already registered; overlapping countdowns are forbidden")
        self._pending_skip_dry_probability = _coerce_p_skip(p)
        self._pending_skip_dry_remaining_env_steps = int(
            remaining_env_steps)
        return self.skip_dry_schedule_state()

    def skip_dry_schedule_state(self) -> dict:
        """Return a pickle-safe curriculum state for callback audits."""
        pending = getattr(
            self, "_pending_skip_dry_probability", None)
        remaining = getattr(
            self, "_pending_skip_dry_remaining_env_steps", 0)
        if pending is None:
            if remaining != 0:
                raise RuntimeError(
                    "p_skip pending probability missing but the countdown is non-zero")
            normalized_pending = None
        else:
            if (
                not isinstance(remaining, (int, np.integer))
                or isinstance(remaining, (bool, np.bool_))
                or int(remaining) <= 0
            ):
                raise RuntimeError("p_skip pending countdown state is invalid")
            normalized_pending = float(pending)
            remaining = int(remaining)
        return {
            "current_probability": float(self.skip_dry),
            "pending_probability": normalized_pending,
            "remaining_env_steps": int(remaining),
        }

    def _tick_skip_dry_schedule(self) -> bool:
        """Advance the countdown after one successful Worker action."""
        state = self.skip_dry_schedule_state()
        pending = state["pending_probability"]
        if pending is None:
            return False
        remaining = int(state["remaining_env_steps"]) - 1
        if remaining > 0:
            self._pending_skip_dry_remaining_env_steps = remaining
            return False
        # Commit before any observation/advance/reset can expose a mixed
        # old-selection/new-feature state.
        self.skip_dry = _coerce_p_skip(pending)
        self._pending_skip_dry_probability = None
        self._pending_skip_dry_remaining_env_steps = 0
        return True

    def _skip_dry_draw(self) -> bool:
        """p_skip draw: the end points 1.0/0.0 consume no stream (bit-identical to the original boolean behaviour); intermediate values use the dedicated stream."""
        p = self.skip_dry
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return float(self._p_rng.random()) < p

    @staticmethod
    def _base_boundary(done, trunc) -> tuple[bool, bool]:
        """Map the underlying end to mutually exclusive Gymnasium terminated/truncated."""
        terminated = bool(done)
        truncated = bool(trunc) and not terminated
        return terminated, truncated

    @classmethod
    def _worker_boundary(
            cls, done, trunc, option_extra: dict,
            base_info: dict | None) -> tuple[bool, bool, bool]:
        """Map an exhausted no-progress budget to a Worker policy failure rather than a TimeLimit.

        OptionsEnv is the single source of truth for the no-progress fact; this only changes the Worker Gym
        boundary semantics. A normal safe budget boundary with progress can still bootstrap, and an
        unsettled boundary with an unfinished animation still keeps the underlying fail-closed terminated.
        """
        terminated, truncated = cls._base_boundary(done, trunc)
        if not isinstance(option_extra, dict):
            raise RuntimeError("Worker terminal is missing option_extra")
        if not isinstance(base_info, dict):
            raise RuntimeError("Worker terminal is missing the underlying base_info")
        required = {
            "base_done",
            "base_trunc",
            "budget_boundary",
            "no_progress_micro_steps",
            "timeout_without_progress",
        }
        missing = required.difference(option_extra)
        if missing:
            raise RuntimeError(
                "Worker terminal option_extra missing fields: "
                f"{sorted(missing)}")
        marker = option_extra["timeout_without_progress"]
        budget_boundary = option_extra["budget_boundary"]
        base_done = option_extra["base_done"]
        base_trunc = option_extra["base_trunc"]
        no_progress_micro_steps = option_extra[
            "no_progress_micro_steps"]
        if (
            not isinstance(marker, (bool, np.bool_))
            or not isinstance(budget_boundary, (bool, np.bool_))
            or not isinstance(base_done, (bool, np.bool_))
            or not isinstance(base_trunc, (bool, np.bool_))
        ):
            raise RuntimeError(
                "Worker terminal boundary flags must be bool")
        if (
            isinstance(no_progress_micro_steps, (bool, np.bool_))
            or not isinstance(
                no_progress_micro_steps, (int, np.integer))
            or int(no_progress_micro_steps) < 0
        ):
            raise RuntimeError(
                "option_extra.no_progress_micro_steps "
                "must be a non-negative plain integer")
        if (
            bool(base_done) is not True
            or bool(base_trunc) != truncated
        ):
            raise RuntimeError(
                "Worker terminal base_done/base_trunc inconsistent with the underlying return: "
                f"done={done!r},trunc={trunc!r},"
                f"base_done={base_done!r},base_trunc={base_trunc!r}")

        terminal = dict(base_info)
        safe_time_limit = bool(
            not terminated
            and truncated
            and bool(base_trunc)
            and terminal.get("time_limit_bootstrap_safe") is True
            and terminal.get("unsettled_budget_terminal") is False
        )
        unsettled_terminal = bool(
            terminated
            and not truncated
            and not bool(base_trunc)
            and terminal.get("time_limit_bootstrap_safe") is False
            and terminal.get("unsettled_budget_terminal") is True
        )
        derived_budget_boundary = (
            safe_time_limit or unsettled_terminal)
        if bool(budget_boundary) != derived_budget_boundary:
            raise RuntimeError(
                "Worker terminal budget_boundary inconsistent with the underlying boundary facts: "
                f"terminated={terminated},truncated={truncated},"
                f"budget_boundary={budget_boundary!r},"
                "time_limit_bootstrap_safe="
                f"{terminal.get('time_limit_bootstrap_safe')!r},"
                "unsettled_budget_terminal="
                f"{terminal.get('unsettled_budget_terminal')!r}")
        # A Gymnasium truncation is only legitimate when the lower layer
        # explicitly certified a settled TimeLimit boundary.  Without this
        # independent check, a simultaneous producer bug could omit both
        # base markers and publish ``budget_boundary=False``; the two false
        # values would agree and silently re-enable value bootstrap.
        if truncated and not safe_time_limit:
            raise RuntimeError(
                "Worker truncated terminal lacks an explicit safe TimeLimit certificate: "
                "time_limit_bootstrap_safe="
                f"{terminal.get('time_limit_bootstrap_safe')!r},"
                "unsettled_budget_terminal="
                f"{terminal.get('unsettled_budget_terminal')!r}")

        expected_timeout = bool(
            derived_budget_boundary
            and int(no_progress_micro_steps) >= KILL_PATIENCE
        )
        timeout_without_progress = bool(marker)
        if timeout_without_progress != expected_timeout:
            raise RuntimeError(
                "option_extra.timeout_without_progress inconsistent with the "
                "budget/no-progress facts: "
                f"marker={timeout_without_progress},"
                f"budget_boundary={derived_budget_boundary},"
                f"no_progress_micro_steps={no_progress_micro_steps},"
                f"threshold={KILL_PATIENCE}")
        if not timeout_without_progress:
            return terminated, truncated, False

        if safe_time_limit == unsettled_terminal:
            raise RuntimeError(
                "a no-progress timeout can only come from a settled safe TimeLimit or a "
                "fail-closed unsettled terminal: "
                f"terminated={terminated},truncated={truncated},"
                f"base_done={option_extra.get('base_done')!r},"
                f"base_trunc={option_extra.get('base_trunc')!r},"
                f"budget_boundary={option_extra.get('budget_boundary')!r},"
                "time_limit_bootstrap_safe="
                f"{terminal.get('time_limit_bootstrap_safe')!r},"
                "unsettled_budget_terminal="
                f"{terminal.get('unsettled_budget_terminal')!r}")
        return True, False, True

    @staticmethod
    def _option_base_info(info: dict) -> dict:
        """Extract the underlying env's original info from the OptionsEnv.step info."""
        return {key: value for key, value in dict(info).items()
                if key != "option_extra"}

    @staticmethod
    def _attach_terminal_base_info(
            info: dict, base_info: dict | None, *,
            terminated: bool, truncated: bool) -> None:
        """Wrap the underlying terminal and publish the final Worker bootstrap semantics at the top level."""
        if bool(terminated) == bool(truncated):
            raise RuntimeError(
                "a Worker terminal must have exactly one of terminated/truncated")
        terminal = dict(base_info or {})
        info["terminal_base_info"] = terminal
        for key, value in terminal.items():
            info.setdefault(key, value)
        # A settled base TimeLimit may have been promoted to a no-progress
        # policy failure.  Keep that lower-layer fact only in
        # terminal_base_info; top-level consumers must see the final Worker
        # boundary and can bootstrap iff the wrapper itself returned truncation.
        info["time_limit_bootstrap_safe"] = bool(truncated)

    def _terminal_death_reward(
            self, terminated: bool, truncated: bool,
            base_info: dict | None) -> float:
        """Rebuild the death component of the underlying terminal tick that may safely cross the frozen-policy boundary.

        OptionsEnv's window ``R`` only publishes the whole-window sum, and the terminal tick may also contain XP, damage,
        movement and other positive/negative terms. Handing the sum or ``min(R, 0)`` to the worker would leak the frozen policy's
        behaviour credit. This rebuilds, from the underlying episode_extra already written and the actual death_ladder
        configuration, only the death term of DiabloGymEnv._reward; any non-death terminal gives 0.
        """
        if not (terminated or truncated):
            return 0.0
        terminal = dict(base_info or {})
        episode = terminal.get("episode_extra")
        if not isinstance(episode, dict):
            raise RuntimeError("the underlying terminal lacks episode_extra; the death penalty cannot be checked")
        died = bool(episode.get("died", False))
        if not died:
            return 0.0
        if not terminated or truncated:
            raise RuntimeError("died=True but not mapped to a mutually exclusive terminated")

        base_env = getattr(self.oe, "env", None)
        if base_env is None or not hasattr(base_env, "death_ladder"):
            raise RuntimeError("cannot read the underlying death_ladder configuration")
        try:
            reward = terminal_death_reward_component(
                dead=True,
                dungeon_level=episode["depth"],
                death_ladder=bool(base_env.death_ladder),
                economy=getattr(base_env, "reward_economy",
                                REWARD_ECONOMY_V1),
            )
            depth = int(episode["depth"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("death terminal lacks a valid depth") from exc
        raw = getattr(base_env, "_raw", None)
        if isinstance(raw, dict):
            if not bool(raw.get("dead", False)):
                raise RuntimeError("episode_extra.died=True but the underlying raw.dead=False")
            raw_depth = int(raw.get("dungeon_level", depth))
            if raw_depth != depth:
                raise RuntimeError(
                    f"death terminal depth ledger mismatch: info={depth},raw={raw_depth}")
        return reward

    def _additional_terminal_death_reward(
            self, terminated: bool, truncated: bool,
            base_info: dict | None) -> float:
        """Map the non-negative configured cost only onto real death terminals; non-death/truncation is always 0."""
        if not terminated or truncated:
            return 0.0
        episode = dict(base_info or {}).get("episode_extra")
        if not isinstance(episode, dict):
            raise RuntimeError("the underlying terminal lacks episode_extra; the death cost cannot be appended")
        return (
            -self.additional_terminal_death_cost
            if bool(episode.get("died", False))
            else 0.0
        )

    def _no_progress_timeout_failure_components(
            self, timeout_without_progress: bool,
            base_info: dict | None) -> tuple[float, float, float]:
        """Return (base death-equivalent term, extra term, total term) of a no-progress failure."""
        if not isinstance(timeout_without_progress, (bool, np.bool_)):
            raise RuntimeError("timeout_without_progress must be a bool")
        if not bool(timeout_without_progress):
            return 0.0, 0.0, 0.0
        if getattr(self, "no_progress_timeout_credit",
                   "death-equivalent") == "zero":
            # R16 (C8): timeout != death; the terminal is still terminated, but no death-equivalent penalty is charged,
            # so trajectories that survive to the end are no longer booked as "worse than dying on L1".
            return 0.0, 0.0, 0.0
        terminal = dict(base_info or {})
        episode = terminal.get("episode_extra")
        if not isinstance(episode, dict):
            raise RuntimeError(
                "no-progress timeout lacks episode_extra; the current depth cannot be checked")
        if bool(episode.get("died", False)):
            raise RuntimeError("a no-progress timeout must not also be marked as a real death")
        base_env = getattr(self.oe, "env", None)
        if base_env is None or not hasattr(base_env, "death_ladder"):
            raise RuntimeError("cannot read the underlying death_ladder configuration")
        try:
            base_failure = terminal_death_reward_component(
                dead=True,
                dungeon_level=episode["depth"],
                death_ladder=bool(base_env.death_ladder),
                economy=getattr(base_env, "reward_economy",
                                REWARD_ECONOMY_V1),
            )
            depth = int(episode["depth"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "no-progress timeout lacks a valid current depth") from exc
        raw = getattr(base_env, "_raw", None)
        if isinstance(raw, dict):
            if bool(raw.get("dead", False)):
                raise RuntimeError(
                    "no-progress timeout but the underlying raw.dead=True")
            raw_depth = int(raw.get("dungeon_level", depth))
            if raw_depth != depth:
                raise RuntimeError(
                    "no-progress timeout depth ledger mismatch: "
                    f"info={depth},raw={raw_depth}")
        additional_failure = -float(self.additional_terminal_death_cost)
        total_failure = float(base_failure) + additional_failure
        if (
            not all(np.isfinite(value) for value in (
                base_failure, additional_failure, total_failure))
            or base_failure >= 0.0
            or additional_failure > 0.0
            or total_failure != float(base_failure) + additional_failure
        ):
            raise RuntimeError(
                "no-progress timeout failure-cost split is abnormal: "
                f"base={base_failure},additional={additional_failure},"
                f"total={total_failure}")
        return float(base_failure), additional_failure, total_failure

    @staticmethod
    def _attach_no_progress_timeout_audit(
            info: dict, timeout_without_progress: bool,
            components: tuple[float, float, float],
            credit_mode: str = "death-equivalent") -> None:
        base_failure, additional_failure, total_failure = components
        timeout = bool(timeout_without_progress)
        if credit_mode not in ("death-equivalent", "zero"):
            raise RuntimeError(
                f"no_progress_timeout_credit is invalid: {credit_mode!r}")
        # R16 (C8) zero-penalty rule domain: timeout is still True and the split is always (0,0,0); under the old rule a timeout
        # always carries a negative base_failure, so this special case masks no old-rule violation; it is released only
        # under credit_mode=="zero" (fail-closed), and the rule domain is passed down with info to the formal PG
        # gatekeeper of leashed_ppo.
        zero_credit_timeout = (timeout and credit_mode == "zero"
                               and components == (0.0, 0.0, 0.0))
        if (
            (not timeout and components != (0.0, 0.0, 0.0))
            or (timeout and not zero_credit_timeout and not (
                base_failure < 0.0
                and additional_failure <= 0.0
                and total_failure == base_failure + additional_failure
            ))
        ):
            raise RuntimeError(
                "Worker no-progress timeout audit split does not add up")
        info.update({
            "worker_no_progress_timeout": timeout,
            "no_progress_timeout_base_failure_reward": base_failure,
            "no_progress_timeout_additional_failure_reward": (
                additional_failure),
            "no_progress_timeout_failure_reward": total_failure,
            "no_progress_timeout_credit": credit_mode,
        })

    def _record_no_progress_timeout(
            self, scope: str, total_failure: float, *,
            credited: bool) -> None:
        if scope not in {"direct", "transition_ff", "reset_ff", "manual_ff"}:
            raise RuntimeError(f"invalid no-progress timeout ledger scope={scope!r}")
        if not np.isfinite(total_failure) or (
                total_failure >= 0.0
                and not (
                    total_failure == 0.0
                    and getattr(self, "no_progress_timeout_credit",
                                "death-equivalent") == "zero")):
            raise RuntimeError(
                "no-progress timeout total failure cost must be finite and negative")
        count_key = f"{scope}_no_progress_timeouts"
        reward_key = f"{scope}_no_progress_timeout_failure_reward"
        self.stats[count_key] = int(self.stats.get(count_key, 0)) + 1
        self.stats[reward_key] = (
            float(self.stats.get(reward_key, 0.0)) + total_failure)
        if credited:
            self.stats["credited_no_progress_timeout_failure_reward"] = (
                float(self.stats.get(
                    "credited_no_progress_timeout_failure_reward", 0.0))
                + total_failure)

    def _earned_prefix_enabled(self) -> bool:
        return getattr(self, "learning_window_scope", "farm-only") == "earned-dive-suffix-v1"

    def _initialize_prefix_state(self):
        self._prefix_original_physical_limit = int(self.oe.env.max_steps)
        self._prefix_observation_limit = int(self.oe.max_steps)
        self._prefix_deadline = None
        self._prefix_handed_off = False
        self._prefix_failure = None
        self._prefix_attempt = None
        self._prefix_attempts = []
        self._prefix_lifetime_attempts = 0
        self._prefix_lifetime_microsteps = 0
        self.stats.update({
            "prefix_attempts": 0, "prefix_microsteps": 0,
            "prefix_handoffs": 0, "prefix_no_handoff_episodes": 0,
            "prefix_parent_decisions": 0, "prefix_raw_reward": 0.0,
            "prefix_worker_wage": 0.0, "prefix_budget_exhausted": False,
        })

        def parent(observation, action_mask):
            self._prefix_validate_limits(active=True)
            action = self.prefix_worker(observation, action_mask)
            if not self.action_space.contains(action) or not bool(action_mask[int(action)]):
                raise RuntimeError("fixed prefix callback returned an illegal action")
            self.stats["prefix_parent_decisions"] += 1
            self._prefix_attempt["parent_decisions"] += 1
            self._prefix_attempt["requested_actions"][int(action)] += 1
            return action

        for name in ("diablogym_worker_observation_view", "diablogym_worker_action12_mode"):
            setattr(parent, name, getattr(self.prefix_worker, name))
        self._prefix_policy = parent

    def get_prefix_ledger(self) -> dict:
        """Return detached audit data; this performs no native or policy query."""
        if not self._earned_prefix_enabled():
            return {"enabled": False}
        return copy.deepcopy({
            "enabled": True, "scope": self.learning_window_scope,
            "prefix_worker_sha256": self.prefix_worker_sha256,
            "prefix_max_attempts": self.prefix_max_attempts,
            "prefix_max_microsteps": self.prefix_max_microsteps,
            "budget_scope": "per-env-lifetime-across-resets",
            "budget_start": "after_normal_env_reset",
            "bootstrap_navigation_in_budget": False,
            "bootstrap_note": "Every attempt executes the normal reset/navigation; its wall time is included below.",
            "original_physical_limit": self._prefix_original_physical_limit,
            "observation_limit": self._prefix_observation_limit,
            "lifetime": {
                **{key: value for key, value in self.stats.items() if key.startswith("prefix_")},
                "prefix_attempts": self._prefix_lifetime_attempts,
                "prefix_microsteps": self._prefix_lifetime_microsteps,
            },
            "failure": self._prefix_failure,
            "attempts": self._prefix_attempts,
        })

    def _prefix_snapshot(self):
        raw = getattr(self.oe.env, "_raw", None)
        if not isinstance(raw, dict):
            return None
        keys = ("dungeon_level", "is_set_level", "set_level_number", "player_x", "player_y",
                "future_x", "future_y", "player_mode", "walkpath0", "walkpath", "dest_action",
                "dead", "game_over", "victory", "hp", "max_hp", "xp", "char_level",
                "gold", "belt_heals", "belt_heal_kinds", "armor_class", "equipped_items",
                "gear_combat_profile", "monster_kill_total")
        state = {key: copy.deepcopy(raw[key]) for key in keys if key in raw}
        state["resource_state"] = {key: copy.deepcopy(value)
                                   for key, value in raw.get("resource_state", {}).items()
                                   if key in ("enabled", "readiness", "transition", "service_trip", "service_authorized",
                                              "ordinary_armor_scope", "preserve_equipment_readiness")}
        return state

    def _prefix_validate_limits(self, *, active):
        expected = self._prefix_deadline if active else self._prefix_original_physical_limit
        if (not active and getattr(self, "worker_time_protocol", "legacy").startswith("completion-l2")):
            expected = self.oe._completion_clock.state.physical_deadline
        if (expected is None or int(self.oe.env.max_steps) != expected
                or int(self.oe.max_steps) != self._prefix_observation_limit):
            raise RuntimeError("earned prefix physical/observation limit changed outside its owner")

    def _prefix_budget_exception(self, reason):
        audit = self.get_prefix_ledger()
        audit["attempt_summary"] = [
            {key: attempt.get(key) for key in
             ("attempt", "episode_seed", "status", "microsteps", "raw_reward", "worker_wage", "wall_seconds")}
            for attempt in audit["attempts"]]
        audit["attempts"] = audit["attempts"][-1:]
        return PrefixBudgetExceeded(reason, audit)

    def _prefix_require_capacity(self):
        if self._prefix_failure is not None:
            raise self._prefix_budget_exception(f"earned prefix is sealed after failure: {self._prefix_failure}")
        if (self._prefix_lifetime_attempts >= self.prefix_max_attempts
                or self._prefix_lifetime_microsteps >= self.prefix_max_microsteps):
            self.stats["prefix_budget_exhausted"] = True
            self._prefix_failure = ("attempt_budget_exhausted"
                                    if self._prefix_lifetime_attempts >= self.prefix_max_attempts
                                    else "microstep_budget_exhausted")
            raise self._prefix_budget_exception(self._prefix_failure)

    def _prefix_begin_attempt(self, seed):
        self._prefix_validate_limits(active=False)
        self._prefix_handed_off = False
        self._prefix_lifetime_attempts += 1
        self.stats["prefix_attempts"] = self._prefix_lifetime_attempts
        self._prefix_wall_start = time.perf_counter()
        self._prefix_attempt = {
            "attempt": self.stats["prefix_attempts"], "episode_seed": int(seed),
            "status": "resetting", "parent_decisions": 0, "requested_actions": [0] * 15,
            "microsteps": 0, "raw_reward": 0.0, "worker_wage": 0.0, "windows": [],
            "budget_start": "after_normal_env_reset", "bootstrap_navigation_in_budget": False,
        }
        self._prefix_attempts.append(self._prefix_attempt)
        self.prefix_worker.episode_reseed(int(seed))

    def _prefix_after_reset(self):
        self._prefix_validate_limits(active=False)
        clock = int(self.oe.env._resource_actual_microsteps)
        if int(self.oe.env._steps) != clock:
            raise RuntimeError("earned prefix requires aligned settled physical clocks")
        remaining = self.prefix_max_microsteps - self._prefix_lifetime_microsteps
        self._prefix_deadline = min(self._prefix_original_physical_limit, clock + remaining)
        if self._prefix_deadline <= clock:
            raise RuntimeError("normal reset returned beyond the prefix physical budget")
        self.oe.env.max_steps = self._prefix_deadline
        if getattr(self, "worker_time_protocol", "legacy").startswith("completion-l2"):
            self.oe.env._completion_prefix_active = True
            self.oe.env._completion_prefix_deadline = self._prefix_deadline
        self._prefix_accounted_clock = clock
        self._prefix_zero_tick_windows = 0
        self._prefix_attempt.update({
            "status": "prefix", "start_native_microstep": clock,
            "start_env_steps": int(self.oe.env._steps),
            "physical_deadline": self._prefix_deadline,
            "reset_wall_seconds": time.perf_counter() - self._prefix_wall_start,
            "initial_state": self._prefix_snapshot(),
        })

    def _prefix_account_clock(self):
        self._prefix_validate_limits(active=True)
        clock = int(self.oe.env._resource_actual_microsteps)
        if (clock < self._prefix_accounted_clock or clock > self._prefix_deadline
                or int(self.oe.env._steps) != clock):
            raise RuntimeError("earned prefix actual microstep accounting violated its physical deadline")
        delta = clock - self._prefix_accounted_clock
        self._prefix_lifetime_microsteps += delta
        self.stats["prefix_microsteps"] = self._prefix_lifetime_microsteps
        self._prefix_attempt["microsteps"] += delta
        self._prefix_accounted_clock = clock
        return delta

    def _prefix_restore_limit(self):
        if self._prefix_deadline is not None:
            self._prefix_validate_limits(active=True)
            self.oe.env.max_steps = self._prefix_original_physical_limit
            self._prefix_deadline = None
            if getattr(self, "worker_time_protocol", "legacy").startswith("completion-l2"):
                self.oe.env._completion_prefix_active = False
                self.oe.env._completion_prefix_deadline = None

    def _prefix_finish_attempt(self, status, **details):
        self._prefix_attempt.update({
            "status": status, "end_native_microstep": int(self.oe.env._resource_actual_microsteps),
            "end_env_steps": int(self.oe.env._steps),
            "wall_seconds": time.perf_counter() - self._prefix_wall_start,
            "final_state": self._prefix_snapshot(), **copy.deepcopy(details),
        })
        self._prefix_restore_limit()
        if status != "handoff":
            self.stats["prefix_no_handoff_episodes"] += 1

    def _prefix_abort_error(self, exc):
        # Unexpected failures remain failures: no fabricated terminal, observation,
        # payroll completion or retry. Do not overwrite a foreign changed deadline.
        if self._prefix_failure is None:
            self._prefix_failure = f"engineering_error:{type(exc).__name__}:{exc}"
        if self._prefix_attempt is not None and self._prefix_attempt.get("status") in ("prefix", "resetting"):
            self._prefix_attempt.update({
                "status": "engineering_error", "error": str(exc),
                "wall_seconds": time.perf_counter() - self._prefix_wall_start,
                "final_state": self._prefix_snapshot(),
                "window_incomplete": self.oe._win is not None,
                "observed_native_microstep": getattr(self.oe.env, "_resource_actual_microsteps", None),
            })
            if getattr(self.oe.env, "_completion_time_failure", None) is not None:
                self._prefix_attempt["completion_time_failure"] = copy.deepcopy(
                    self.oe.env._completion_time_failure)
        if (self._prefix_deadline is not None
                and int(self.oe.env.max_steps) == self._prefix_deadline
                and int(self.oe.max_steps) == self._prefix_observation_limit):
            self._prefix_restore_limit()
        self._alive = False

    def _prefix_record_window(self, extra, base_info, start_clock):
        delta = self._prefix_account_clock()
        self._prefix_zero_tick_windows = (self._prefix_zero_tick_windows + 1 if delta == 0 else 0)
        self._prefix_attempt["windows"].append({
            "start_native_microstep": int(start_clock),
            "end_native_microstep": int(self.oe.env._resource_actual_microsteps),
            "extra": copy.deepcopy(extra), "terminal_info": copy.deepcopy(base_info)
            if extra.get("base_done") else None,
        })
        for field, stat, value in (("raw_reward", "prefix_raw_reward", float(extra["R"])),
                                   ("worker_wage", "prefix_worker_wage", float(extra["worker_wage"]))):
            self._prefix_attempt[field] += value
            self.stats[stat] += value
        if self._prefix_zero_tick_windows >= _MAX_PREFIX_ZERO_TICK_WINDOWS:
            raise RuntimeError("earned prefix repeated zero-native-tick windows without a learner handoff")

    def _prefix_open_dive(self):
        # Match the real Options worker opening. A failed opening verdict runs
        # this entire window with the parent; it is never closed and reopened.
        self.oe._win_begin(DIVE)
        ending = self.oe._consume_fuse_recovery()
        if ending is None or ending.reason is None:
            ending = self.oe._drain()
        observation = None
        if ending is None:
            observation = self.oe._worker_policy_observation(self.policy_observation_view)
            mask = np.asarray(observation[DUAL_WORKER_ACTION_MASK_SLICE])
            manager_mask = np.asarray(observation[DUAL_WORKER_MANAGER_MASK_SLICE])
            if (mask.shape != (15,) or manager_mask.shape != (3,)
                    or not np.all((mask == 0) | (mask == 1))
                    or not np.all((manager_mask == 0) | (manager_mask == 1))):
                raise RuntimeError("earned prefix requires the canonical embedded action masks")
            from .resource_protocol import native_readiness
            raw = self.oe.env._raw
            eligible = (int(raw["dungeon_level"]) == 1 and not raw.get("is_set_level")
                        and not any(raw.get(key) for key in ("dead", "game_over", "victory"))
                        and self.oe.env._decision_idle(raw)
                        and native_readiness(raw)["ready"]
                        and bool(manager_mask[DIVE]) and bool(np.any(mask)))
            self._prefix_attempt.setdefault("dive_openings", []).append({
                "window_id": int(self.oe._win["window_id"]),
                "native_microstep": int(self.oe.env._resource_actual_microsteps),
                "eligible": bool(eligible), "state": self._prefix_snapshot(),
                "worker_mask": mask.astype(bool).tolist(),
                "manager_mask": manager_mask.astype(bool).tolist(),
            })
            if eligible:
                self._prefix_account_clock()
                if self._prefix_lifetime_microsteps >= self.prefix_max_microsteps:
                    raise RuntimeError("prefix handoff reached an unclosed physical budget boundary")
                win = self.oe._win
                opening_reward = float(win["R"])
                self.stats["prefix_raw_reward"] += opening_reward
                self._prefix_attempt["raw_reward"] += opening_reward
                self.stats["prefix_handoffs"] += 1
                self.stats["windows"] += 1
                self.stats["fresh"] += 1
                self.stats["dive_live_windows"] = int(self.stats.get("dive_live_windows", 0)) + 1
                self._prefix_finish_attempt(
                    "handoff", window_id=int(win["window_id"]),
                    opening_reward=opening_reward,
                    learner_observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest(),
                    handoff_window={key: copy.deepcopy(win[key]) for key in
                                    ("window_id", "opt", "t0", "R", "W", "worker_wage", "beats")},
                    learner_reward_credited=0.0)
                self._prefix_handed_off = True
                return observation, None
        while ending is None:
            if observation is None:
                observation = self.oe._worker_policy_observation(self.policy_observation_view)
            # R18-B7 prefix guard (2026-09-07): a DIVE window whose opening verdict was
            # not eligible is played by the frozen parent; the parent must never descend
            # under the prefix (the completion clock fails closed with "L2 arrival before
            # learner handoff" - the R18-B arm crashed on it). Under coach-v03 the
            # manager opens DIVE on the six-condition law while the handoff needs the
            # seven-condition native verdict (HP >= 80%), so the parent legitimately plays
            # DIVE windows on L1 while wounded. Inside such a window the descend macro and
            # the trigger-tile steps are withheld from the parent; it farms until the window
            # closes and the next opening is judged again.
            masks = self._prefix_guard_masks(self.oe.env._raw, self.oe._worker_masks())
            action = self._prefix_policy(observation, masks)
            outcome = self.oe._win_step_worker(action)
            if outcome.reason is not None:
                ending = outcome
            observation = None
        return None, self.oe._win_end(ending.reason)

    @staticmethod
    def _prefix_guard_masks(raw, masks):
        """R18-B7: withhold a11 and trigger-tile steps from the parent inside a prefix
        DIVE window (main L1 only); a0 stays legal so the parent can always act."""
        from .env import DiabloGymEnv
        guarded = np.array(masks, dtype=bool, copy=True)
        if int(raw.get("dungeon_level", 0)) != 1 or raw.get("is_set_level"):
            return guarded
        guarded[11] = False
        for action in DiabloGymEnv._protected_walk_actions(raw):
            guarded[action] = False
        if not guarded.any():
            guarded[0] = True
        return guarded

    def _advance_earned_prefix(self) -> _AdvanceOutcome:
        if not self._alive or self._prefix_attempt is None:
            raise RuntimeError("earned prefix requires a real reset before advancement")
        reward = 0.0
        extras = []
        try:
            while True:
                self._prefix_validate_limits(active=True)
                start_clock = int(self.oe.env._resource_actual_microsteps)
                option = self._mgr_choose()
                if self.oe._win is not None:
                    raise RuntimeError("earned prefix cannot replace an open window")
                if option == DIVE:
                    observation, closed = self._prefix_open_dive()
                    if observation is not None:
                        return _AdvanceOutcome(
                            observation, reward + float(self._prefix_attempt["opening_reward"]),
                            False, False, tuple(extras),
                            opening_reward=float(self._prefix_attempt["opening_reward"]),
                            opening_recovery_action=(self.oe._win["last_recovery_action"]
                                                     if self.oe._win["recovery_actions"] else None))
                    extra, base_info, done, trunc = closed
                else:
                    original_workers = self.oe._workers
                    if option == FARM:
                        self.oe._workers = {**original_workers, FARM: self._prefix_policy}
                    try:
                        _, _, done, trunc, info = self.oe.step(option)
                    finally:
                        self.oe._workers = original_workers
                    extra = info["option_extra"]
                    base_info = self._option_base_info(info)
                self._prefix_record_window(extra, base_info, start_clock)
                reward += float(extra["R"])
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if self._prefix_lifetime_microsteps >= self.prefix_max_microsteps:
                    self.stats["prefix_budget_exhausted"] = True
                    self._prefix_failure = "microstep_budget_exhausted"
                    self._prefix_finish_attempt(
                        "microstep_budget_exhausted", terminal_info=base_info,
                        terminal_option_extra=extra, base_done=bool(done), base_truncated=bool(trunc))
                    self._alive = False
                    raise self._prefix_budget_exception(self._prefix_failure)
                raw = self.oe.env._raw
                if int(raw["dungeon_level"]) >= 2 and not raw.get("is_set_level"):
                    self._prefix_failure = "early_main_depth_before_handoff"
                    self._prefix_finish_attempt(self._prefix_failure, terminal_info=base_info,
                                                terminal_option_extra=extra)
                    self._alive = False
                    raise RuntimeError(self._prefix_failure)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    terminated, truncated, timeout = self._worker_boundary(done, trunc, extra, base_info)
                    self._prefix_finish_attempt(
                        "prefix_terminal", reason=extra.get("reason"),
                        terminated=terminated, truncated=truncated,
                        timeout_without_progress=timeout, terminal_info=base_info,
                        terminal_option_extra=extra)
                    return _AdvanceOutcome(
                        None, reward, terminated, truncated, tuple(extras),
                        terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(terminated, truncated, base_info),
                        timeout_without_progress=timeout)
        except Exception as exc:
            self._prefix_abort_error(exc)
            raise

    def _advance_to_learning_window(self) -> _AdvanceOutcome:
        """Advance to the next learnable FARM and return all intermediate consequences; never roll into a new game.

        DIVE/RESUPPLY, dry FARM windows skipped by the course, and the next FARM's opening reflexes are all
        environment dynamics that happen after the previous worker action. Their raw rewards are accumulated here for
        conservation audits; ``step()`` returns the real next state/termination in the same transition,
        but never credits these frozen-policy returns to the worker action.
        """
        if (self._earned_prefix_enabled()
                and not self._prefix_handed_off):
            return self._advance_earned_prefix()
        if not self._alive:
            return _AdvanceOutcome(None, 0.0, False, False, ())
        if self.oe._win is not None:
            raise RuntimeError("the current window must be closed before advancing to the next FARM")
        reward = 0.0
        extras = []
        while True:
            opt = self._mgr_choose()
            if opt == FARM:
                dry = self.oe.exhausted
                if dry and self._skip_dry_draw():
                    # v26 oasis: dry-level revisit windows (exhausted flag set) are run by the script's inner loop,
                    # the same path as DIVE/RESUPPLY with the same bookkeeping, and do not become learning episodes
                    # (E1 item 5A: run by script with probability p_skip; p=1.0 always runs them ≡ the original skip_dry=True)
                    _, r, done, trunc, info = self.oe.step(FARM)
                    extra = info["option_extra"]
                    reward += float(r)
                    extras.append(extra)
                    self._log(extra, fast_forward=True)
                    self.stats["ff_dry"] += 1
                    if done or trunc:
                        self._alive = False
                        self.stats["ff_terminals"] += 1
                        base_info = self._option_base_info(info)
                        terminated, truncated, timeout = (
                            self._worker_boundary(
                                done, trunc, extra, base_info))
                        return _AdvanceOutcome(
                            self._policy_observation(self.oe._worker_obs()),
                            reward, terminated, truncated,
                            tuple(extras),
                            terminal_base_info=base_info,
                            terminal_death_reward=self._terminal_death_reward(
                                terminated, truncated, base_info),
                            timeout_without_progress=timeout)
                    continue
                self.oe._win_begin(FARM)
                # If the previous worker proposal triggered the fuse, the refusal tick itself still does not execute the action;
                # the recovery tick is booked explicitly at the start of this new manager window, then reflexes are drained.
                ending = self.oe._consume_fuse_recovery()
                if ending is None or ending.reason is None:
                    ending = self.oe._drain()
                if ending is None:
                    # The opening reflexes happen before the worker gets its next observation; the worker's first tick
                    # counts from the current W, so they must be credited to the previous transition here,
                    # otherwise this part of the reward would still fall outside the books.
                    opening_reward = float(self.oe._win["R"])
                    reward += opening_reward
                    self.stats["windows"] += 1
                    self.stats["dry" if dry else "fresh"] += 1
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, False, False,
                        tuple(extras), opening_reward=opening_reward,
                        opening_recovery_action=(
                            self.oe._win["last_recovery_action"]
                            if self.oe._win["recovery_actions"] else None))
                # The drain tick ended the window directly (death/exhausted/CAP ...): log it as a fast-forwarded window and keep looking
                extra, base_info, done, trunc = self.oe._win_end(ending.reason)
                reward += float(extra["R"])
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    terminated, truncated, timeout = (
                        self._worker_boundary(
                            done, trunc, extra, base_info))
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, terminated, truncated,
                        tuple(extras), terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(
                            terminated, truncated, base_info),
                        timeout_without_progress=timeout)
            elif (opt == DIVE
                  and getattr(self, "learning_window_scope", "farm-only")
                  in _LIVE_DIVE_WINDOW_SCOPES):
                # R13 classroom reform: DIVE windows open live like FARM windows, and the worker enters the gradient per micro tick;
                # the window-opening recovery tick/reflex drain still belongs to the previous transition, with the same rule and books as
                # FARM. RESUPPLY and skipped dry windows stay scripted fast-forwards (the else below).
                self.oe._win_begin(DIVE)
                ending = self.oe._consume_fuse_recovery()
                if ending is None or ending.reason is None:
                    ending = self.oe._drain()
                if ending is None:
                    opening_reward = float(self.oe._win["R"])
                    reward += opening_reward
                    self.stats["windows"] += 1
                    self.stats["fresh"] += 1
                    self.stats["dive_live_windows"] = (
                        int(self.stats.get("dive_live_windows", 0)) + 1)
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, False, False,
                        tuple(extras), opening_reward=opening_reward,
                        opening_recovery_action=(
                            self.oe._win["last_recovery_action"]
                            if self.oe._win["recovery_actions"] else None))
                # The drain tick ended the DIVE window directly: log it as a fast-forwarded window and keep looking
                extra, base_info, done, trunc = self.oe._win_end(ending.reason)
                reward += float(extra["R"])
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    terminated, truncated, timeout = (
                        self._worker_boundary(
                            done, trunc, extra, base_info))
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, terminated, truncated,
                        tuple(extras), terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(
                            terminated, truncated, base_info),
                        timeout_without_progress=timeout)
            else:
                _, r, done, trunc, info = self.oe.step(opt)   # script inner loop, same-source bookkeeping
                extra = info["option_extra"]
                reward += float(r)
                extras.append(extra)
                self._log(extra, fast_forward=True)
                if done or trunc:
                    self._alive = False
                    self.stats["ff_terminals"] += 1
                    base_info = self._option_base_info(info)
                    terminated, truncated, timeout = (
                        self._worker_boundary(
                            done, trunc, extra, base_info))
                    return _AdvanceOutcome(
                        self._policy_observation(self.oe._worker_obs()),
                        reward, terminated, truncated,
                        tuple(extras),
                        terminal_base_info=base_info,
                        terminal_death_reward=self._terminal_death_reward(
                            terminated, truncated, base_info),
                        timeout_without_progress=timeout)

    def next_window(self):
        """Manually advance to this game's next FARM; returns None when the game ends and never rolls into a new game.

        The new training path has ``step()`` consume the complete _AdvanceOutcome directly. This compatibility entry is only for
        manual traversal by demonstrations/probes; if a caller uses it, fast-forward rewards are booked explicitly into
        ``manual_ff_reward`` instead of pretending to have been handed to PPO.
        """
        self.stats["manual_ff_calls"] = int(
            self.stats.get("manual_ff_calls", 0)) + 1
        outcome = self._advance_to_learning_window()
        self.stats["manual_ff_reward"] += float(outcome.reward)
        if float(outcome.terminal_death_reward) < 0.0:
            self.stats["manual_ff_terminal_deaths"] += 1
        if outcome.timeout_without_progress:
            timeout_components = (
                self._no_progress_timeout_failure_components(
                    True, outcome.terminal_base_info))
            self._record_no_progress_timeout(
                "manual_ff", timeout_components[2], credited=False)
        if outcome.terminated or outcome.truncated:
            return None
        return outcome.obs

    # ---- gym interface ----
    def reset(self, *, seed=None, options=None):
        # SB3/VecEnv seeds the environment only through reset(seed). If the worker's game-seed sampler were not
        # reset with this seed, --seed would only control the first game, and later automatic
        # rollovers would still come from system entropy, making the whole training run irreproducible.
        if self._earned_prefix_enabled():
            self._prefix_require_capacity()
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            # E1 item 3 per-env seeding pinned: the dedicated p-draw stream is derived deterministically from the episode seed with a fixed offset
            # (SB3 seeds per env with seed+rank, which makes the reproducibility claim hold; no contamination of the training RNG).
            self._p_rng = _derive_p_skip_rng(seed)
        # Gym allows the caller to reset before the episode has ended. _win still exists then, and
        # calling next_window() on the same underlying game would overwrite the unsettled window and tie the manager state,
        # wage ledger and new Gym episode together. Abandon the old game explicitly and restart from a clean boundary.
        interrupted_window = self.oe._win is not None
        if interrupted_window:
            self.stats["interrupted_resets"] = int(
                self.stats.get("interrupted_resets", 0)) + 1
            self._alive = False
        if seed is not None or not self._alive:
            self._new_episode(seed, options=options)
        empty_episodes = 0
        while True:
            outcome = self._advance_to_learning_window()
            self.stats["reset_ff_reward"] += float(outcome.reward)
            reset_terminal_death = float(outcome.terminal_death_reward)
            if reset_terminal_death < 0.0:
                reset_additional_terminal_death = (
                    self._additional_terminal_death_reward(
                        outcome.terminated,
                        outcome.truncated,
                        outcome.terminal_base_info))
                self.stats["reset_ff_terminal_deaths"] += 1
                self.stats["reset_ff_terminal_death_reward"] += (
                    reset_terminal_death)
                # The reset fast-forward happens before PPO gets this game's first state, so the extra cost cannot
                # be booked to any worker transition; only the counterfactual audit ledger is kept here.
                self.stats[
                    "reset_ff_additional_terminal_death_reward"
                ] += reset_additional_terminal_death
            if outcome.timeout_without_progress:
                reset_timeout_components = (
                    self._no_progress_timeout_failure_components(
                        True, outcome.terminal_base_info))
                # The reset fast-forward precedes any Worker observation/action and only books the counterfactual failure cost;
                # it is never pushed into the next game's first tick and never fabricates a transition.
                self._record_no_progress_timeout(
                    "reset_ff", reset_timeout_components[2],
                    credited=False)
            if (outcome.obs is not None
                    and not (outcome.terminated or outcome.truncated)):
                return outcome.obs, {
                    "episode_seed": self._episode_seed,
                    "window_id": int(self.oe._win["window_id"]),
                    "reset_fast_forward_reward": float(outcome.reward),
                    "reset_fast_forward_extras": list(outcome.extras),
                    "reset_recovery_action": outcome.opening_recovery_action,
                }
            self.stats["reseeds"] += 1    # fallback rollover (also reached when an explicit-seed game has zero FARM windows;
            empty_episodes += 1
            if (not self._earned_prefix_enabled()
                    and empty_episodes >= _MAX_EMPTY_FARM_EPISODES):
                raise RuntimeError(
                    "frozen manager produced no FARM window in "
                    f"{empty_episodes} consecutive games; refusing to roll over forever; "
                    "manager="
                    + (f"heuristic:{self.manager_heuristic}"
                       if self.mgr is None
                       else f"sha256={self.mgr.source_sha256}"))
            self._new_episode()           # the BC side seals the demonstration-pool escape hatch with stats assertions)

    def _portal_close_is_death_equivalent(self) -> bool:
        """R18-B5 (2026-09-07) review correction: is this portal_trigger window close a dangerous close?

        True: main L2+ (not a quest set level) and one of the portal service's three danger clauses holds now;
        same rule and thresholds as retreat_trigger; the escrow is forfeited.
        False: the in-town scroll purchase/return errand, a door already underfoot (portal_standing), and
        any case where the service/state cannot be obtained; vests as usual. Better to under-penalize than to penalize an errand.
        """
        portal = getattr(getattr(self, "oe", None), "portal_service", None)
        if portal is None or getattr(portal, "policy", None) is None:
            return False
        raw = getattr(getattr(getattr(self, "oe", None), "env", None), "_raw", None)
        if not isinstance(raw, dict):
            return False
        if raw.get("is_set_level"):
            return False
        if int(raw.get("dungeon_level", 0) or 0) < 2:
            return False
        return portal.danger_reason(raw) is not None

    def _descend_escrow_settlement(self, d_before, close_reason) -> float:
        """R14.2 option D: settle at window close and return the amount this transition should vest (always 0 with the flag off).

        Non-death close: first vest all existing escrow, then put this window's new-deepest-level descend fee × fraction
        into escrow (waiting for the next non-death close); death close: forfeit all of it. Forfeiture for deaths in fast-forward
        segments is executed separately by the continuation branch. Only enters policy_reward and never touches the
        W ledger/identities/manager ledger (a worker-side transplant of the R11 forfeiting valve).
        """
        f = getattr(self, "descend_escrow_fraction", 0.0)
        if f <= 0.0:
            return 0.0
        pending = float(getattr(self, "_descend_escrow", 0.0))
        # R18-B windows triggered by retreat pay no escrow: retreat_trigger closes the window exactly at hp <= 50% (the point of maximum
        # death risk), which is death-equivalent and forfeits; otherwise it would open a cash-out channel of "diving wounded to half HP and
        # then retreating", turning forfeitable danger pay into cash. With the flag off this branch
        # is never reachable and, short-circuited, is structurally identical to the old rule bit for bit.
        # R18-B5 (2026-09-07): danger-triggered portal windows likewise pay no escrow: "dive wounded to half HP
        # and then open a door to town" is exactly the cash-out channel the retreat clause just closed, and it does not even lose depth.
        # Review correction (same day): close_reason is not enough to judge danger. OptionsEnv._win_term collapses
        # **any** non-None result of PortalService.trigger_reason into the same
        # "portal_trigger", but buy_scroll / portal_return among them fire in town at full HP, and
        # portal_standing does not look at HP at all; those are errands, not death-equivalents. So this asks
        # the portal service again for the three danger clauses (danger_reason, same thresholds as the retreat rule),
        # and forfeits only dangerous closes on main L2+. With the flag off this branch is never reachable and, short-circuited, is
        # structurally identical to the old rule bit for bit.
        # R18-B6 (2026-09-07): the two new close reasons introduced by sweep-v1
        # ("sweep_trigger"/"sweep_complete", options_env.py:1452/1462) do **not**
        # forfeit, and this is provable, not arguable: the sweep law only holds on main L1
        # (the L1-only clause of resource_sweep.py refuses any dungeon_level != 1),
        # while both existing forfeiture clauses only hold on main L2+
        # (RetreatService.trigger_reason returns None directly when dungeon_level < 2;
        # _portal_close_is_death_equivalent likewise). The two sets are constructively disjoint by depth,
        # so a sweep close can neither steal the forfeiture of a dangerous close nor be death-equivalent itself;
        # it hands control back on its own when a monster comes within 3 tiles or HP drops below the threshold. On L1
        # max_after > d_before does not hold, so the window stores no new escrow either; its effect is structurally identical
        # to the existing ordinary closes such as "scene"/"cap".
        # cain-v1 / smith-v1 introduce no new close reason: they are legs inside the RESUPPLY town trip
        # (options_env.py:2448-2479 / 2569-2574), and the window is still closed by the resource service
        # under the old rule, so no new branch is needed here, nor allowed.
        _forfeit_reason = close_reason == "death" or (
            getattr(self, "resource_retreat", "off") != "off"
            and close_reason == "retreat_trigger") or (
            getattr(self, "resource_portal", "off") != "off"
            and close_reason == "portal_trigger"
            and self._portal_close_is_death_equivalent())
        if _forfeit_reason:
            if pending:
                self.stats["descend_escrow_forfeited"] = float(
                    self.stats.get(
                        "descend_escrow_forfeited", 0.0)) + pending
            self._descend_escrow = 0.0
            return 0.0
        vest = pending
        if vest:
            self.stats["descend_escrow_vested"] = float(
                self.stats.get("descend_escrow_vested", 0.0)) + vest
        max_after = max(1, int(getattr(
            self.oe.env, "_econ_episode_max_depth", 1)))
        new_escrow = 0.0
        if max_after > max(1, int(d_before)):
            _unit = float(getattr(getattr(
                self.oe.env, "reward_economy", None),
                "descend_unit", 0.0))
            _pw = float(getattr(self, "descend_escrow_power", 1.0))
            if _pw == 1.0:
                _floors = float(
                    sum(range(max(1, int(d_before)), max_after)))
            else:
                # R14.3 option E: convex hazard pay; d^power priced per level
                _floors = float(sum(
                    d ** _pw
                    for d in range(max(1, int(d_before)), max_after)))
            new_escrow = f * _unit * _floors
            if (new_escrow > 0.0
                    and getattr(
                        self, "descend_escrow_readiness_gate", False)):
                # R15 amendment 2: an under-ready descent gets nothing (the character panel is unchanged before and after the level
                # change, so evaluating at the arrival level at window close is equivalent to evaluating at descent).
                raw = getattr(self.oe.env, "_raw", None) or {}
                # R17.0 amendment 1: ruler selector (v1 old ruler bit for bit; v2 the same ruler as the coach)
                _table = getattr(
                    self, "descend_escrow_readiness_table", "v1")
                _ratio_v1 = readiness_power_ratio(raw, max_after)
                _ratio_v2 = readiness_power_ratio_v2(raw, max_after)
                _ratio = _ratio_v2 if _table == "v2" else _ratio_v1
                if getattr(self, "resource_protocol", "off") != "off":
                    receipts = getattr(self.oe.env, "_resource_transition_receipts", [])
                    receipt = next((item for item in reversed(receipts)
                                    if int(item.get("target_depth", -1)) == max_after), None)
                    if getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03":
                        # Six-condition law (health excluded): a forced-unready
                        # descent is accepted by the engine but vests nothing.
                        _ratio = (1.0 if receipt and receipt.get("accepted")
                                  and receipt.get("pretransition_ready_law") else 0.0)
                        _table = "native-l2-town-v1/coach-v03"
                    else:
                        _ratio = 1.0 if receipt and receipt.get("accepted") and receipt.get("pretransition_ready") else 0.0
                        _table = "native-l2-town-v1"
                # R17.0 gauge (review panel correction 2.3): per-crossing telemetry: both rulers'
                # ratios and the panel (clvl/HP/AC/dmg), arrival level, verdict, escrow amount due;
                # a bounded list that records without judging, untouched with the flag off (no readiness gate). The audit callback
                # (R13DiveAuditCallback) rewrites r17_descend_gate.jsonl as a whole.
                _gate_log = self.stats.setdefault(
                    "descend_escrow_gate_log", [])
                if len(_gate_log) < _DESCEND_ESCROW_GATE_LOG_CAP:
                    _gate_log.append({
                        "depth": int(max_after),
                        "char_level": int(raw.get("char_level", 0)),
                        "max_hp": int(raw.get("max_hp", 0)),
                        "armor_class": int(raw.get("armor_class", 0)),
                        "dmg": (int(raw.get("item_max_damage", 0))
                                + int(raw.get("damage_mod", 0))),
                        "ratio_v1": round(float(_ratio_v1), 4),
                        "ratio_v2": round(float(_ratio_v2), 4),
                        "table": str(_table),
                        "passed": bool(_ratio >= 1.0),
                        "escrow": round(float(new_escrow), 3),
                    })
                if _ratio < 1.0:
                    self.stats["descend_escrow_unready_denied"] = float(
                        self.stats.get(
                            "descend_escrow_unready_denied", 0.0)
                    ) + new_escrow
                    new_escrow = 0.0
                else:
                    # R17.0: count of gate-passing descents (both rulers count), so audits can report
                    # "number of ready descents" rather than only the vested amount.
                    self.stats["descend_escrow_ready_vested_count"] = (
                        int(self.stats.get(
                            "descend_escrow_ready_vested_count", 0)) + 1)
        self._descend_escrow = new_escrow
        return vest

    def _forfeit_descend_escrow_on_ff_death(self) -> None:
        """R14.2: a real death in a fast-forward segment (RESUPPLY/skipped dry window) also forfeits the escrow."""
        if getattr(self, "descend_escrow_fraction", 0.0) <= 0.0:
            return
        pending = float(getattr(self, "_descend_escrow", 0.0))
        if pending:
            self.stats["descend_escrow_forfeited"] = float(
                self.stats.get(
                    "descend_escrow_forfeited", 0.0)) + pending
        self._descend_escrow = 0.0

    def _hp_economy_component(self, hp_before, belt_before) -> float:
        """R16 (C7): HP economy component (always 0.0 with the flag off).

        −hp_loss_price × max(0, hp_before − hp_now) (asymmetric: only HP loss is penalized);
        +potion_pickup_bonus × max(0, belt_heals_now − belt_before).
        Only enters policy_reward, never touching the wage identity/manager ledger.
        """
        price = float(getattr(self, "hp_loss_price", 0.0))
        bonus = float(getattr(self, "potion_pickup_bonus", 0.0))
        if price <= 0.0 and bonus <= 0.0:
            return 0.0
        raw = getattr(self.oe.env, "_raw", None) or {}
        component = 0.0
        if price > 0.0:
            lost = max(0, int(hp_before) - int(raw.get("hp", hp_before)))
            if lost:
                charge = -price * float(lost)
                component += charge
                self.stats["hp_loss_charged"] = float(
                    self.stats.get("hp_loss_charged", 0.0)) + charge
        if bonus > 0.0:
            gained = max(0, int(raw.get("belt_heals", belt_before))
                         - int(belt_before))
            if gained:
                credit = bonus * float(gained)
                component += credit
                self.stats["potion_pickup_credited"] = float(
                    self.stats.get("potion_pickup_credited", 0.0)) + credit
        return component

    def _depth_shaping_component(
            self, d_before, terminated, truncated) -> float:
        """R13.2 option A: this transition's potential-based shaping component (always 0.0 with the flag off).

        φ = unit × (deepest level of the game − 1) (monotone potential, a theory-review correction: once live DIVE windows are exempt
        from step protection, upward transitions exist and a current-level potential would heavily penalize going up; a monotone potential
        charges nothing for going up, is immune to cycles and pays only for new progress). F = φ(s′)−φ(s); a true terminal
        refunds −φ(s_T) so that φ(absorbing)=0; truncation gives no refund (the critic learns
        V−φ and bootstrapping hedges exactly; the review proved this is the only self-consistent combination). Only enters
        policy_reward, never touching the wage identity/manager ledger (the sanctioned slot is the same layer as
        additional_terminal_death_cost). Conservation: for any true-terminal
        trajectory ΣF ≡ 0.
        """
        unit = getattr(self, "depth_shaping_unit", 0.0)
        if unit <= 0.0:
            return 0.0
        max_after = max(1, int(getattr(
            self.oe.env, "_econ_episode_max_depth", 1)))
        delta = unit * float(max_after - max(1, int(d_before)))
        if delta != 0.0:
            self.stats["depth_shaping_credited"] = float(
                self.stats.get("depth_shaping_credited", 0.0)) + delta
        refund = 0.0
        if terminated and not truncated:
            refund = -unit * float(max_after - 1)
            if refund != 0.0:
                self.stats["depth_shaping_refunded"] = float(
                    self.stats.get("depth_shaping_refunded", 0.0)) + refund
        return delta + refund

    def step(self, action):
        if self._earned_prefix_enabled():
            if self._prefix_failure is not None or not self._prefix_handed_off:
                raise gym.error.ResetNeeded("earned prefix has not handed a live window to the learner")
            self._prefix_validate_limits(active=False)
        if self.oe._win is None:
            raise gym.error.ResetNeeded("WorkerWindowEnv.step() needs reset() to open a FARM window first")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in {self.action_space}, got {action!r}")
        win = self.oe._win
        window_id = int(win["window_id"])
        w_before = win["W"]
        # R13.2: the shaping baseline = this game's deepest level (monotone potential; zero reads with the flag off, protecting __new__ shells
        # and the zero-touch discipline). A level change on a reason-None tick always triggers a scene close, so Δφ≠0
        # only appears on window-close transitions, and the two policy_reward assembly points cover every case.
        _d_before = (
            max(1, int(getattr(
                self.oe.env, "_econ_episode_max_depth", 1)))
            if (getattr(self, "depth_shaping_unit", 0.0) > 0.0
                or getattr(
                    self, "descend_escrow_fraction", 0.0) > 0.0)
            else 0)
        # R16 (C7): HP economy baseline (zero reads with the flag off)
        if (getattr(self, "hp_loss_price", 0.0) > 0.0
                or getattr(self, "potion_pickup_bonus", 0.0) > 0.0):
            _raw0 = getattr(self.oe.env, "_raw", None) or {}
            _hp_before = int(_raw0.get("hp", 0))
            _belt_before = int(_raw0.get("belt_heals", 0))
        else:
            _hp_before = _belt_before = 0
        outcome = self.oe._win_step_worker(action)
        wage = win["W"] - w_before
        if (getattr(self, "learning_window_scope", "farm-only")
                in _LIVE_DIVE_WINDOW_SCOPES):
            # R13 per-window metering ledger (zero touch on the flag-off path): live micro ticks are bucketed by window mode,
            # farm_live_steps + dive_live_steps ≡ the num_timesteps increment.
            _in_dive = int(win.get("opt", FARM)) == DIVE
            _step_key = "dive_live_steps" if _in_dive else "farm_live_steps"
            self.stats[_step_key] = int(self.stats.get(_step_key, 0)) + 1
            # R14 option B audit: mirrors OptionsEnv's retained cumulative descend bonus
            _kept = float(getattr(
                self.oe, "descend_bonus_kept_total", 0.0))
            if _kept:
                self.stats["descend_bonus_kept"] = round(_kept, 6)
            if _in_dive:
                if int(outcome.requested_action) == 11:
                    self.stats["dive_live_a11_requests"] = (
                        int(self.stats.get("dive_live_a11_requests", 0)) + 1)
                if outcome.executed_action == 11:
                    self.stats["dive_live_a11_executed"] = (
                        int(self.stats.get(
                            "dive_live_a11_executed", 0)) + 1)
        # The action belongs to the current rollout.  Its resulting state
        # belongs to the next one, so an armed curriculum switch commits here:
        # after action execution, but before observation construction,
        # next-window advancement, or a terminal return that VecEnv auto-resets.
        self._tick_skip_dry_schedule()
        overridden = bool(outcome.fuse_tripped)   # compatible BC exclusion key; current semantics = proposal refused
        obs = self._policy_observation(
            self.oe._worker_obs())           # take the final observation before the window closes (the window still holds τ)
        info = {
            "episode_seed": self._episode_seed,
            "window_id": window_id,
            "farm_window_end": False,
            "worker_wage": float(wage),
            "transition_reward": float(wage),
            "requested_action": int(outcome.requested_action),
            "executed_action": outcome.executed_action,
            "action_effect_audit": getattr(
                outcome, "action_effect_audit", None),
            "action14_audit": getattr(
                outcome, "action14_audit", None),
            "overridden": overridden,
            "fuse_tripped": bool(outcome.fuse_tripped),
            "fuse_requested_action": outcome.fuse_requested_action,
            "fast_forward_reward_credit_mode": (
                self.fast_forward_reward_credit),
            "additional_terminal_death_cost": (
                self.additional_terminal_death_cost),
            # R17.0 amendment 1 (review panel correction 2.2): per-transition receipts of the three new policy_reward
            # components (always 0.0 with the flag off), so the leashed_ppo formal-PG
            # gatekeeper can check reward == wage + timeout + shaping + vest + hp_econ
            # (summed in the same order as the two assembly points below, verifiable bit for bit in floating point).
            "worker_depth_shaping_reward": 0.0,
            "worker_descend_escrow_vest": 0.0,
            "worker_hp_economy_reward": 0.0,
        }
        self._attach_no_progress_timeout_audit(
            info, False, (0.0, 0.0, 0.0),
            credit_mode=getattr(self, "no_progress_timeout_credit",
                                "death-equivalent"))
        _hp_econ = self._hp_economy_component(_hp_before, _belt_before)
        info["worker_hp_economy_reward"] = float(_hp_econ)
        if outcome.reason is None:
            if _hp_econ:
                info["transition_reward"] = float(wage) + _hp_econ
                return obs, float(wage) + _hp_econ, False, False, info
            return obs, wage, False, False, info
        extra, base_info, done, trunc = self.oe._win_end(outcome.reason)
        self._log(extra, fast_forward=False)
        if (getattr(self, "learning_window_scope", "farm-only")
                in _LIVE_DIVE_WINDOW_SCOPES):
            # R13 audit: live closes record the mode; live DIVE closes are bucketed by reason
            # (descend = classroom success/stall = idling/death = deep-level death).
            info["window_mode"] = (
                ("farm", "dive", "resupply")[int(extra.get("opt", FARM))])
            if int(extra.get("opt", FARM)) == DIVE:
                _reason_key = {
                    "descend": "dive_live_descends",
                    "stall": "dive_live_stalls",
                    "death": "dive_live_deaths",
                }.get(extra.get("reason"))
                if _reason_key is not None:
                    self.stats[_reason_key] = (
                        int(self.stats.get(_reason_key, 0)) + 1)
        # R14.2 option D: settle at window close (non-death vests existing escrow + puts the new descent into escrow;
        # death forfeits). Always 0.0 with the flag off, zero touch.
        _escrow_vest = self._descend_escrow_settlement(
            _d_before, extra.get("reason"))
        info["worker_descend_escrow_vest"] = float(_escrow_vest)
        info.update({
            "farm_window_end": True,
            "option_extra": extra,
            "fast_forward_reward": 0.0,
            "credited_fast_forward_reward": 0.0,
            "existing_terminal_death_reward": 0.0,
            "additional_terminal_death_reward": 0.0,
            "total_terminal_death_reward": 0.0,
            "credited_fast_forward_terminal_death_reward": 0.0,
            "fast_forward_extras": [],
            "next_window_id": None,
            "next_window_opening_reward": 0.0,
            "next_window_recovery_action": None,
            "terminal_option_extra": extra if (done or trunc) else None,
            "terminal_base_info": None,
        })
        if done or trunc:
            self._alive = False
            terminated, truncated, timeout_without_progress = (
                self._worker_boundary(done, trunc, extra, base_info))
            # This death component was already counted into the wage by _win_step_worker. Only publish it for audit,
            # without going through fast-forward credit, so the same underlying death tick is never penalized twice.
            existing_terminal_death = self._terminal_death_reward(
                terminated, truncated, base_info)
            additional_terminal_death = (
                self._additional_terminal_death_reward(
                    terminated, truncated, base_info))
            total_terminal_death = (
                existing_terminal_death + additional_terminal_death)
            timeout_components = (
                self._no_progress_timeout_failure_components(
                    timeout_without_progress, base_info))
            self._attach_no_progress_timeout_audit(
                info, timeout_without_progress, timeout_components,
                credit_mode=getattr(self, "no_progress_timeout_credit",
                                    "death-equivalent"))
            info.update({
                "existing_terminal_death_reward": existing_terminal_death,
                "additional_terminal_death_reward": additional_terminal_death,
                "total_terminal_death_reward": total_terminal_death,
            })
            if existing_terminal_death < 0.0:
                self.stats["direct_terminal_deaths"] += 1
                self.stats[
                    "direct_existing_terminal_death_reward"
                ] += existing_terminal_death
                self.stats[
                    "direct_additional_terminal_death_reward"
                ] += additional_terminal_death
            self.stats["additional_terminal_death_reward"] += (
                additional_terminal_death)
            if timeout_without_progress:
                self._record_no_progress_timeout(
                    "direct", timeout_components[2], credited=True)
            _shaping = self._depth_shaping_component(
                _d_before, terminated, truncated)
            info["worker_depth_shaping_reward"] = float(_shaping)
            policy_reward = (
                float(wage)
                + additional_terminal_death
                + timeout_components[2]
                + _shaping
                + _escrow_vest
                + _hp_econ
            )
            info["transition_reward"] = policy_reward
            # R17.1 (W/P review H3/M1): unified death-component receipt key + receipt schema marker,
            # so the leashed_ppo gatekeeper checks the six-term identity for every transition
            # (wage + death + timeout + shaping + vest + hp_econ, same order as above).
            info["worker_credited_terminal_death_reward"] = float(
                additional_terminal_death)
            info["worker_policy_reward_receipt"] = "v1"
            self._attach_terminal_base_info(
                info, base_info,
                terminated=terminated, truncated=truncated)
            return obs, policy_reward, terminated, truncated, info

        # A natural FARM boundary is not a Gym terminal. All consequences of the frozen manager/script
        # are advanced and returned within this transition, with raw rewards only in the audit ledger; then the
        # real next FARM observation is returned for value-function bootstrapping.
        continuation = self._advance_to_learning_window()
        self.stats["transition_ff_reward"] += float(continuation.reward)
        ff_terminal_death = float(continuation.terminal_death_reward)
        if (not np.isfinite(ff_terminal_death)
                or ff_terminal_death > 0.0
                or (ff_terminal_death != 0.0
                    and not continuation.terminated)):
            raise RuntimeError(
                "fast-forward terminal death component abnormal: "
                f"reward={ff_terminal_death},"
                f"terminated={continuation.terminated},"
                f"truncated={continuation.truncated}")
        if ff_terminal_death < 0.0:
            # R14.2 option D: a real death in a fast-forward segment also forfeits the escrow (the part newly stored at this
            # transition's close); death means loss, the same rule as a live death.
            self._forfeit_descend_escrow_on_ff_death()
        additional_terminal_death = (
            self._additional_terminal_death_reward(
                continuation.terminated,
                continuation.truncated,
                continuation.terminal_base_info))
        total_terminal_death = (
            ff_terminal_death + additional_terminal_death)
        timeout_components = (
            self._no_progress_timeout_failure_components(
                continuation.timeout_without_progress,
                continuation.terminal_base_info))
        credited_existing_terminal_death = (
            ff_terminal_death
            if self.fast_forward_reward_credit == "terminal-death-only"
            else 0.0
        )
        credited_terminal_death = (
            credited_existing_terminal_death
            + additional_terminal_death)
        self.stats["transition_ff_terminal_death_reward"] += (
            ff_terminal_death)
        if ff_terminal_death < 0.0:
            self.stats["transition_ff_terminal_deaths"] += 1
            self.stats[
                "transition_ff_additional_terminal_death_reward"
            ] += additional_terminal_death
        self.stats["credited_ff_terminal_death_reward"] += (
            credited_terminal_death)
        self.stats["additional_terminal_death_reward"] += (
            additional_terminal_death)
        if continuation.timeout_without_progress:
            self._record_no_progress_timeout(
                "transition_ff", timeout_components[2], credited=True)
        self._attach_no_progress_timeout_audit(
            info, continuation.timeout_without_progress,
            timeout_components,
            credit_mode=getattr(self, "no_progress_timeout_credit",
                                "death-equivalent"))
        info.update({
            "fast_forward_reward": float(continuation.reward),
            # The frozen manager/script is not a worker action. The full raw reward is kept for
            # conservation audits; the policy only receives the real terminal death component or the no-progress failure
            # cost; otherwise a0 could wait for exhausted and claim the positive combat/descend returns of DIVE
            # as its own.
            "credited_fast_forward_reward": (
                credited_terminal_death + timeout_components[2]),
            "existing_terminal_death_reward": ff_terminal_death,
            "additional_terminal_death_reward": additional_terminal_death,
            "total_terminal_death_reward": total_terminal_death,
            "credited_fast_forward_terminal_death_reward": (
                credited_terminal_death),
            "fast_forward_extras": list(continuation.extras),
            "next_window_opening_reward": float(continuation.opening_reward),
            "next_window_recovery_action": continuation.opening_recovery_action,
            "terminal_option_extra": (
                continuation.extras[-1]
                if ((continuation.terminated or continuation.truncated)
                    and continuation.extras)
                else None),
        })
        _shaping = self._depth_shaping_component(
            _d_before,
            continuation.terminated, continuation.truncated)
        info["worker_depth_shaping_reward"] = float(_shaping)
        policy_reward = (
            float(wage)
            + credited_terminal_death
            + timeout_components[2]
            + _shaping
            + _escrow_vest
            + _hp_econ
        )
        info["transition_reward"] = policy_reward
        # R17.1 (W/P review H3/M1): as above, the death component of the fast-forward branch is credited_terminal_death.
        info["worker_credited_terminal_death_reward"] = float(credited_terminal_death)
        info["worker_policy_reward_receipt"] = "v1"
        if continuation.terminated or continuation.truncated:
            self._attach_terminal_base_info(
                info, continuation.terminal_base_info,
                terminated=continuation.terminated,
                truncated=continuation.truncated)
            return (continuation.obs, policy_reward, continuation.terminated,
                    continuation.truncated, info)
        if continuation.obs is None or self.oe._win is None:
            raise RuntimeError("no next learning state and no real termination after a natural FARM boundary")
        info["next_window_id"] = int(self.oe._win["window_id"])
        return continuation.obs, policy_reward, False, False, info

    def action_masks(self) -> np.ndarray:
        if self.oe._win is None:
            raise gym.error.ResetNeeded("WorkerWindowEnv.action_masks() needs reset() first")
        return self.oe._worker_masks()

    def close(self):
        self.oe.close()
