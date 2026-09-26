"""v22 strategy brain/operating brain: OptionsEnv, an SMDP wrapper over frozen macros.

Design: docs/design/DESIGN.md, chapter v22.
Core commitments:
  - operating brain = a verbatim, frozen port of the inner loop of the oracle oracle_mountain (a stationary SMDP);
  - "exhausted -> dive" is not hard-coded in the script but promoted to a decision of the strategy brain (the only exam question of this chapter);
  - a level change always hands control back (explicit invariant: at least one new decision per level, the foundation for extending to 16 levels);
  - γ_mgr=1.0 is guaranteed on the training side and in-option rewards accumulate undiscounted, so the quantity the strategy brain optimizes
    equals the oracle ledger verbatim (the undiscounted 3000-step return);
  - drinking is a brainstem reflex (hp<0.5 ∧ belt>0 -> 12), deliberately not an option (keeping the v12 ghost from coming back).

Option vocabulary Discrete(3):
  0 FARM     clear the current level: engage when monsters exist, otherwise pick up potions/gear/explore. Terminates on: level-up/no progress
             140/cumulative FARM 1800 in this scene/level change.
  1 DIVE     push combat toward the next main-line target: break through blockers (≤6 tiles), otherwise the story/descend macro.
             Terminates on: scene change/level loss/stall 140.
  2 RESUPPLY pre-dive resupply: pick up potions repeatedly. Mask: no floor potion or belt full. Terminates on: full/none/zero progress.
Common termination: death/truncation/TAU_CAP=600 micro ticks (hit rate goes to info, alarm above 5%).
Anti-hideout: FARM is usually the fallback; but when a story target has appeared and no enemy is near, control is forcibly handed to DIVE,
so that action10 does not overstep its authority. When the exhausted flag is set and DIVE is legal, control is also forcibly handed over, cutting off the endless dry windows
the old manager formed by relying on action10 to descend covertly; FARM is kept only when DIVE is illegal, with a forced
floor revisit of ≥25 ticks.

v23 (docs/prereg/PREREG-v23.md): the per-tick bookkeeping of the window loop (fuse/reflexes/termination ladder) is extracted into
shared methods: OptionsEnv (assembly/evaluation) and WorkerWindowEnv (in-place training) run the same
code, eliminating a "third implementation". workers={option: policy} replaces an option's scripted inner loop with a
learnable worker:
  - reflex ownership moved up: the worker never observes a "reflex pending" state; at window start and after every worker action
    the wrapper drains (tick by tick through the fuse/clock/termination ladder). For the scripted path this is an identity transformation
    (the first dispatch branch is the same check, so deciding early does not change the action sequence).
  - worker wage w_t = r_t − level-change bonus (the only stripped term; the dive-arbitrage fix, lesson 16).
    Ledger identity: Σw ≡ window R − DESCEND_UNIT×ΣΔdlvl⁺. The manager ledger is untouched.
  - worker observation 298 dims = base 295 +
    [τ/TAU_CAP, old no-new-kill clock/140, old exhausted]. The last two keep the frozen
    V28's protocol-v3 semantics verbatim; the new "no positive progress" clock and the scene FARM cumulative budget are only
    responsible for the current window-closing dynamics and do not swap out the frozen network's input contract. The third dimension also publishes, by sign,
    this window's drinking history: not drunk stays the old 0/1; once drunk from any source in this window (worker-initiated or brainstem reflex)
    it is encoded as -1/-2, so a hidden latch never creates different labels for the same observation.
  - worker action 11 is always masked (descending belongs to the manager's DIVE authority), and 10 is also masked in the story hand-over state; 12 is
    controlled by drink_sovereignty and is legal only when the visible hp∈[0.5,0.75), the belt has potions and no drinking
    has happened in this window yet. The brainstem reflex itself is not limited by this latch, so emergencies can still drain
    continuously; the latch only stops the worker from wasting another potion after already being rescued.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from . import bridge
from .controller_wire import *  # noqa: F403 - re-export canonical wire schema
from .env import (
    DESCEND_UNIT,
    REWARD_ECONOMY_V1,
    DiabloGymEnv,
    _scene_identity,
    gear_combat_utility_value,
)

FARM, DIVE, RESUPPLY = 0, 1, 2
KILL_PATIENCE = 140   # micro ticks: stall limit for FARM without positive progress / DIVE without a level change
TAU_CAP = 600         # longest option occupancy (cap on the straggler tax)
REVISIT_FLOOR = 25    # minimum occupancy when re-choosing FARM under the exhausted flag (blocks instant-termination key stirring)
RESUPPLY_CAP = 60
# Cumulative hard budget for FARM occupancy in this scene. The short clock is reset by real progress such as new tiles/damage, so it cannot
# also decide "when to hand control back to DIVE"; otherwise continuous exploration could eat the whole 3000-tick
# game. 1800 gives the spawn level about 60% of the total budget; fixed-seed calibration can still clear 40-90 monsters,
# while leaving at least 1200 ticks to find/execute DIVE.
FARM_SCENE_CAP = 1800
BLOCKER_RADIUS = 6    # shared by story hand-over and DIVE path clearing; forbids two thresholds that leave a 4..6 gap
GEAR_GRACE_MAX_DECISIONS = 3
# Active drinking is a "limited sovereignty", not a licence to squander potions at any HP loss. The brainstem keeps exclusive control of the <0.5
# emergency drain; the worker only has a choice inside the safe envelope [0.5, 0.75). The upper bound is deliberately above
# the BC-v2 teacher's main setting 0.65 and its only OC 0.70: the mask only seals obviously wasteful states and keeps
# real legal hard negatives in [teacher_threshold, 0.75) so the policy learns the boundary,
# instead of hard-coding the teacher's target policy into action legality.
VOLUNTARY_DRINK_HP_LOW = 0.50
VOLUNTARY_DRINK_HP_HIGH = 0.75
N_EXTRA_WORKER = 3    # extra worker observation dims: τ clock / old no-new-kill clock / old exhausted
# The third worker extra after the 295 base dims. Non-negative values keep the old exhausted verbatim;
# ≤-1 means at least one drink (active or reflex) has happened in this window, never changing any input of the old
# zero-drink policy; ``-v-1`` restores the old 0/1 losslessly.
WORKER_DRINK_LATCH_FEATURE = 297
WORKER_OBSERVATION_VIEW_RAW_V4 = "raw-v4"
WORKER_OBSERVATION_VIEW_LEGACY_V3 = "legacy-v3"
WORKER_OBSERVATION_VIEW_A12_OVERLAY = "legacy-v3-a12-overlay"
WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC = "dual-v4-asymmetric-v3"
# R13 classroom reform: v5 = the complete v4 vector as a bit-identical prefix + a 12-dim window-mode block appended.
# The old v4 layout and sha are untouched; v5 is only named by the R13 pre-registered recipe, and the default path cannot reach it.
WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE = "dual-v5-window-mode-v1"
DUAL_WORKER_V5_APPENDIX_DIM = 12
WORKER_OBSERVATION_VIEWS = frozenset({
    WORKER_OBSERVATION_VIEW_RAW_V4,
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
    WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
})
WORKER_ACTION12_PERMANENTLY_MASKED = "permanently-masked"
WORKER_ACTION12_ENVIRONMENT_MASK = "environment-mask"
WORKER_ACTION12_MODES = frozenset({
    WORKER_ACTION12_PERMANENTLY_MASKED,
    WORKER_ACTION12_ENVIRONMENT_MASK,
})
# ``dual-v4-asymmetric-v3`` layout, segment metadata and canonical hash live
# in controller_wire.py.  Runtime/wrapper/training code all import that frozen
# source rather than copying absolute indices.
MANAGER_OBSERVATION_VIEW_RAW_V4 = "raw-v4"
MANAGER_OBSERVATION_VIEW_LEGACY_V3 = "legacy-v3"
MANAGER_OBSERVATION_VIEWS = frozenset({
    MANAGER_OBSERVATION_VIEW_RAW_V4,
    MANAGER_OBSERVATION_VIEW_LEGACY_V3,
})


def _resolve_worker_drink_sovereignty(
        workers: dict,
        requested: bool | None) -> bool:
    """Bind the environment-wide action-12 semantics to tagged callbacks."""
    modes = []
    untagged = []
    for option, worker in workers.items():
        mode = getattr(worker, "diablogym_worker_action12_mode", None)
        if mode is None:
            untagged.append(option)
            continue
        if mode not in WORKER_ACTION12_MODES:
            raise ValueError(
                "Worker callback action12 mode not registered: "
                f"option={option!r},mode={mode!r}")
        modes.append(mode)
    if modes and untagged:
        raise ValueError(
            "a Worker with an action12 contract must not be mixed with a callback without a contract"
            f": untagged={untagged!r}")
    unique_modes = set(modes)
    if len(unique_modes) > 1:
        raise ValueError(
            "OptionsEnv's global drink_sovereignty cannot express multiple Worker "
            f"action12 mode:{sorted(unique_modes)!r}")
    derived = None
    if unique_modes:
        derived = (
            next(iter(unique_modes))
            == WORKER_ACTION12_ENVIRONMENT_MASK
        )
    if requested is None:
        return True if derived is None else derived
    configured = bool(requested)
    if derived is not None and configured != derived:
        raise ValueError(
            "OptionsEnv drink_sovereignty inconsistent with the Worker action12 contract"
            f": configured={configured},contract={next(iter(unique_modes))!r}")
    return configured


@dataclass(frozen=True)
class BeatOutcome:
    """Auditable result of one tick.

    When the fuse fires, the requested action is no longer silently replaced by explore action 10; instead this tick is refused and
    the current option closes its window with ``reason="fuse"``. This way the requested action recorded in the rollout and the action the
    environment actually executed are never mislabelled: ``executed_action=None`` explicitly means not executed.
    """

    reason: str | None
    requested_action: int
    executed_action: int | None
    fuse_tripped: bool
    action14_audit: dict | None = None
    action_effect_audit: dict | None = None


@dataclass(frozen=True)
class WorkerStepOutcome:
    """Combined audit result of a worker proposal and its reflex tail."""

    reason: str | None
    requested_action: int
    executed_action: int | None
    fuse_tripped: bool
    fuse_requested_action: int | None
    action14_audit: dict | None = None
    action_effect_audit: dict | None = None


_ACTION14_AUDIT_KEYS = frozenset({
    "accepted", "commit_attempts", "utility_before",
    "utility_after", "utility_delta",
})
_ACTION_EFFECT_AUDIT_KEYS = frozenset({
    "requested_action", "native_attempts", "native_accepts",
    "request_executed", "material_effect", "effect_reasons",
    "same_scene", "stall_cost_applied",
})


def _validated_action14_audit(info: dict) -> dict:
    """Validate the synchronous native gear-commit receipt from the base env."""
    audit = info.get("action14_audit")
    if not isinstance(audit, dict) or set(audit) != _ACTION14_AUDIT_KEYS:
        raise RuntimeError(
            "action14 native causal receipt missing/corrupt")
    accepted = audit["accepted"]
    integers = {
        key: audit[key]
        for key in (
            "commit_attempts", "utility_before",
            "utility_after", "utility_delta",
        )
    }
    if (
        not isinstance(accepted, bool)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integers.values()
        )
        or integers["commit_attempts"] < 0
        or not 0 <= integers["utility_before"] <= 0xFFFFFFFF
        or not 0 <= integers["utility_after"] <= 0xFFFFFFFF
        or integers["utility_delta"] < 0
        or integers["utility_after"] - integers["utility_before"]
        != integers["utility_delta"]
        or accepted
        != (
            integers["commit_attempts"] >= 1
            and integers["utility_delta"] > 0
        )
    ):
        raise RuntimeError(
            "action14 native causal receipt fields do not balance")
    return {
        "accepted": accepted,
        **integers,
    }


def _validated_action_effect_audit(
        info: dict, requested_action: int) -> dict:
    """Validate the base environment's causal effect/stall receipt."""
    audit = info.get("action_effect_audit")
    resource = info.get("resource_action_audit")
    is_resource = isinstance(resource, dict) and resource.get("source") == "resupply-script"
    if is_resource and (requested_action != 0
            or not isinstance(resource.get("accepted"), bool)
            or type(resource.get("micro_steps")) is not int
            or resource["micro_steps"] < 0):
        raise RuntimeError("Invalid script-owned resource receipt")
    if (
        not isinstance(audit, dict)
        or set(audit) != _ACTION_EFFECT_AUDIT_KEYS
        or (
            isinstance(audit.get("requested_action"), bool)
            or not isinstance(audit.get("requested_action"), int)
        )
        or int(audit["requested_action"]) != int(requested_action)
        or isinstance(audit.get("native_attempts"), bool)
        or not isinstance(audit.get("native_attempts"), int)
        or isinstance(audit.get("native_accepts"), bool)
        or not isinstance(audit.get("native_accepts"), int)
        or not 0 <= audit["native_accepts"] <= audit["native_attempts"]
        or not isinstance(audit.get("request_executed"), bool)
        or not isinstance(audit.get("material_effect"), bool)
        or not isinstance(audit.get("same_scene"), bool)
        or not isinstance(audit.get("stall_cost_applied"), bool)
        or not isinstance(audit.get("effect_reasons"), tuple)
        or any(
            not isinstance(reason, str) or not reason
            for reason in audit["effect_reasons"]
        )
        or len(set(audit["effect_reasons"]))
        != len(audit["effect_reasons"])
        or bool(audit["effect_reasons"]) != audit["material_effect"]
        or audit["request_executed"]
        != (
            bool(resource["accepted"]) if is_resource else
            (int(requested_action) == 0 or audit["native_accepts"] > 0)
        )
        or audit["stall_cost_applied"]
        != (
            (audit["same_scene"] and not audit["request_executed"]
             and resource["micro_steps"] > 0) if is_resource else
            (audit["same_scene"] and (int(requested_action) == 0
                or not audit["request_executed"]))
        )
    ):
        raise RuntimeError("base action causal-effect receipt missing/corrupt")
    return {
        "requested_action": int(audit["requested_action"]),
        "native_attempts": int(audit["native_attempts"]),
        "native_accepts": int(audit["native_accepts"]),
        "request_executed": bool(audit["request_executed"]),
        "material_effect": bool(audit["material_effect"]),
        "effect_reasons": tuple(audit["effect_reasons"]),
        "same_scene": bool(audit["same_scene"]),
        "stall_cost_applied": bool(audit["stall_cost_applied"]),
    }


def _floor_heals(raw) -> bool:
    # Shares the single visible-and-reachable definition with the base env's action targets/masks; the helper stays backward compatible
    # with old fixtures lacking visible/reachable, avoiding a second filtering source of truth.
    return bool(DiabloGymEnv._policy_floor_items(raw, "heal"))


def _reflex(raw) -> bool:
    """Brainstem reflex condition (identical to the first dispatch branch)."""
    return DiabloGymEnv._reflex_eligible(raw)


_DISTANCE_UNSET = object()


def dispatch(
    mode: str,
    raw,
    gear_available: bool,
    action_mask=None,
    nearest_engageable_distance=_DISTANCE_UNSET,
) -> int:
    """Verbatim port of the oracle's inner loop (pure function, frozen). mode ∈ {farm, dive, resupply}.
    Note: the oracle farm phase's stagnant>=140 -> 11 sub-branch is deliberately removed; it belongs to the strategy brain."""
    hp = raw["hp"] / max(1, raw["max_hp"])
    belt = raw.get("belt_heals", 0)
    free_slots = DiabloGymEnv._belt_free_slots(raw)
    has_exact_action_mask = action_mask is not None
    if action_mask is not None:
        action_mask = np.asarray(action_mask, dtype=bool)
        if action_mask.shape != (15,):
            raise ValueError(
                "dispatch action_mask must be (15,), got "
                f"{action_mask.shape}")
        monster_available = bool(action_mask[9])
        heal_available = bool(action_mask[13])
        gear_available = bool(gear_available) and bool(action_mask[14])
    else:
        monster_available = bool(
            DiabloGymEnv._policy_monsters(raw))
        heal_available = free_slots > 0 and _floor_heals(raw)
    if nearest_engageable_distance is _DISTANCE_UNSET:
        near = _nearest(raw)
    else:
        near = nearest_engageable_distance
        if near is not None:
            if isinstance(near, bool) or not isinstance(
                    near, (int, float, np.integer, np.floating)):
                raise TypeError(
                    "nearest_engageable_distance must be a non-negative number or None")
            near = float(near)
            if not np.isfinite(near) or near < 0:
                raise ValueError(
                    "nearest_engageable_distance must be a finite non-negative number")
    if hp < 0.5 and belt > 0:          # brainstem reflex, embedded in every mode
        return 12
    if mode == "resupply":
        # OptionsEnv normally enters this branch only when the RESUPPLY mask is true; the extra guard
        # ensures that even a pure function call never issues an illegal pickup to a belt "short of potions but filled with non-potions".
        return 13 if heal_available else 0
    if mode == "dive":
        if belt <= 2 and heal_available:
            return 13
        if (
            monster_available
            and near is not None
            and near <= BLOCKER_RADIUS
        ):
            return 9
        # action 11 may cross a one-way scene boundary.  Drain a currently
        # executable upgrade first so loot from the cleared blocker is not
        # abandoned with the old scene.  Callers without the exact controller
        # mask have no atomic reachability proof and retain fail-closed a11.
        if has_exact_action_mask and gear_available:
            return 14
        return 11
    # farm
    if raw.get("progression_targets") and (near is None or near > 6):
        # The flat oracle has no manager and must execute DIVE immediately; the old action10 would wait deterministically
        # in the progression state, producing about KILL_PATIENCE empty labels in a row.
        # Hierarchical Options closes the window through _farm_handoff before reaching this point. If the Worker
        # masked 11, fail closed and wait for the hand-over instead of generating an overstepping action.
        if action_mask is not None and not bool(action_mask[11]):
            # A just-cleared blocker may expose both the one-way progression
            # target and a strict gear upgrade.  During the single Worker
            # grace decision the mask is exactly {0,14}; teach/execute 14
            # instead of turning that valuable state into an action0 label.
            return 14 if gear_available else 0
        return 11
    if monster_available:
        return 9
    if heal_available:
        return 13
    if gear_available:
        return 14
    return 10


def _nearest(raw):
    px, py = raw["player_x"], raw["player_y"]
    ds = [max(abs(m["x"] - px), abs(m["y"] - py))
          for m in DiabloGymEnv._policy_monsters(raw)]
    return min(ds) if ds else None


def _farm_handoff(
    raw,
    nearest_engageable_distance=_DISTANCE_UNSET,
) -> bool:
    """When a story target has appeared and no enemy is near, FARM must hand authority back to the manager's DIVE."""
    near = (
        _nearest(raw)
        if nearest_engageable_distance is _DISTANCE_UNSET
        else nearest_engageable_distance
    )
    return (bool(raw.get("progression_targets"))
            and (near is None or near > BLOCKER_RADIUS))


class OptionsEnv(gym.Env):
    """step(option) runs the option to termination and returns (303-dim observation, undiscounted cumulative reward, ...)."""

    N_EXTRA_MGR = 8  # time_remaining/old no-new-kill clock/kills on this level/old time on this level/previous option one-hot(3)/previous option τ

    def __init__(self, max_steps: int = 3000, workers: dict | None = None,
                 drink_sovereignty: bool | None = None,
                 dive_stall_protocol: str = "no-progress-v1",
                 worker_observation_view: str = WORKER_OBSERVATION_VIEW_RAW_V4,
                 manager_observation_view: str = (
                     MANAGER_OBSERVATION_VIEW_RAW_V4),
                 dive_live_sovereignty: bool = False,
                 worker_descend_bonus_fraction: float = 0.0,
                 farm_scene_cap: int = FARM_SCENE_CAP,
                 reset_layer_clock_on_window: bool = False,
                 resource_protocol: str = "off",
                 resource_purchase_mode: str = "full",
                 resource_service_policy: str = "legacy-v1",
                 resource_calibration=None,
                 resource_readiness_law: str = "veto-v1",
                 worker_time_protocol: str = "legacy",
                 resource_retreat: str = "off",
                 resource_portal: str = "off",
                 resource_sweep: str = "off",
                 resource_identify: str = "off",
                 resource_weapon_upgrade: str = "off",
                 **env_kwargs):
        super().__init__()
        from .resource_protocol import (
            validate_resource_config, validate_resource_calibration, RESOURCE_FARM_CAP,
            validate_readiness_law)
        self.resource_protocol, self.resource_purchase_mode = validate_resource_config(
            resource_protocol, resource_purchase_mode)
        from .resource_sustain import validate_service_policy
        self.resource_service_policy = validate_service_policy(
            self.resource_protocol, self.resource_purchase_mode, resource_service_policy)
        self.resource_calibration = validate_resource_calibration(
            self.resource_protocol, resource_calibration)
        # R17.1 readiness rule 3: readiness law; default veto-v1 keeps env kwargs byte-identical.
        self.resource_readiness_law = validate_readiness_law(
            self.resource_protocol, resource_readiness_law)
        if self.resource_readiness_law != "veto-v1":
            env_kwargs["resource_readiness_law"] = self.resource_readiness_law
        # R18-A retreat-v1: return-to-town interface (default off = byte-identical kwargs).
        from .resource_protocol import validate_retreat_protocol
        self.resource_retreat = validate_retreat_protocol(
            self.resource_protocol, self.resource_readiness_law, resource_retreat)
        if self.resource_retreat != "off":
            env_kwargs["resource_retreat"] = self.resource_retreat
        # R18-F portal-v1 (default off = byte-identical kwargs).
        from .resource_protocol import validate_portal_protocol
        self.resource_portal = validate_portal_protocol(
            self.resource_protocol, self.resource_readiness_law,
            self.resource_retreat, resource_portal)
        if self.resource_portal != "off":
            env_kwargs["resource_portal"] = self.resource_portal
        # R18-H (2026-09-07) sweep-v1 (default off = byte-identical kwargs).
        from .resource_protocol import validate_sweep_protocol
        self.resource_sweep = validate_sweep_protocol(
            self.resource_protocol, self.resource_service_policy, resource_sweep)
        if self.resource_sweep != "off":
            env_kwargs["resource_sweep"] = self.resource_sweep
        # R18-H identify-v1 (default off = byte-identical kwargs).
        from .resource_identify import validate_identify_protocol
        self.resource_identify = validate_identify_protocol(
            self.resource_protocol, self.resource_service_policy, resource_identify)
        if self.resource_identify != "off":
            env_kwargs["resource_identify"] = self.resource_identify
        # R18-K (2026-09-07) smith-v1 (default off = byte-identical kwargs).
        from .resource_weapon_upgrade import validate_weapon_upgrade
        self.resource_weapon_upgrade = validate_weapon_upgrade(
            self.resource_protocol, self.resource_service_policy,
            self.resource_purchase_mode, resource_weapon_upgrade)
        if self.resource_weapon_upgrade != "off":
            env_kwargs["resource_weapon_upgrade"] = self.resource_weapon_upgrade
        from .completion_clock import COMPLETION_PROTOCOLS, COMPLETION_RECIPES
        if worker_time_protocol not in ("legacy", *COMPLETION_PROTOCOLS):
            raise ValueError("Unknown worker_time_protocol")
        self.worker_time_protocol = worker_time_protocol
        if worker_time_protocol in COMPLETION_PROTOCOLS:
            recipe = COMPLETION_RECIPES[worker_time_protocol]
            if (type(max_steps) is not int
                    or max_steps != recipe.actor_denominator
                    or self.resource_protocol != "l2-town-v1"
                    or self.resource_purchase_mode != "full"
                    or self.resource_service_policy not in ("sustain-v6", "sustain-loot-v1")
                    or self.resource_calibration is not None):
                raise ValueError("completion-l2-v1 requires legacy observation denominator and l2-town-v1/full with sustain-v6 or sustain-loot-v1 without calibration")
        if self.resource_service_policy == "sustain-loot-v1" and self.worker_time_protocol not in COMPLETION_PROTOCOLS:
            raise ValueError("sustain-loot-v1 requires an explicit completion-l2 time protocol")
        env_kwargs["resource_protocol"] = self.resource_protocol
        env_kwargs["resource_purchase_mode"] = self.resource_purchase_mode
        ordinary_scope = self.resource_service_policy in ("sustain-v5", "sustain-v6", "sustain-loot-v1")
        requested_scope = env_kwargs.pop("resource_ordinary_armor_scope", ordinary_scope)
        if requested_scope is not ordinary_scope:
            raise ValueError("resource_ordinary_armor_scope must match resource_service_policy")
        if ordinary_scope:
            env_kwargs["resource_ordinary_armor_scope"] = True
        preserve_equipment = self.resource_service_policy in ("sustain-v6", "sustain-loot-v1")
        requested_preservation = env_kwargs.pop(
            "resource_preserve_equipment_readiness", preserve_equipment)
        if requested_preservation is not preserve_equipment:
            raise ValueError("resource_preserve_equipment_readiness must match resource_service_policy")
        if preserve_equipment:
            env_kwargs["resource_preserve_equipment_readiness"] = True
        loot_economy = self.resource_service_policy == "sustain-loot-v1"
        requested_loot = env_kwargs.pop("resource_loot_economy", loot_economy)
        if requested_loot is not loot_economy:
            raise ValueError("resource_loot_economy must match resource_service_policy")
        if loot_economy:
            env_kwargs["resource_loot_economy"] = True
        if self.resource_protocol != "off":
            farm_scene_cap = (RESOURCE_FARM_CAP if self.resource_calibration is None
                              else self.resource_calibration.farm_scene_microstep_cap)
        self._workers = workers or {}
        # R16 amendment (audit cluster C6/C3): the spawn-level scene budget becomes configurable (default 1800, old rule bit for bit);
        # the no-progress clock can optionally be reset when a FARM window opens; under argmax, after a DIVE stall
        # layer_clock is inherited and the next FARM window is exhausted again after one decision (clock inheritance).
        _cap = int(farm_scene_cap)
        if _cap <= 0:
            raise ValueError(
                f"farm_scene_cap must be a positive integer, got {farm_scene_cap!r}")
        self.farm_scene_cap = _cap
        self.reset_layer_clock_on_window = bool(reset_layer_clock_on_window)
        # R13 sovereignty transfer (default False = old rule unchanged bit for bit): only inside live DIVE windows
        # are a11 and stepping on trigger tiles with 1-8 unlocked; the a10 story mask and the FARM/RESUPPLY window
        # masks are untouched for any value.
        self.dive_live_sovereignty = bool(dive_live_sovereignty)
        # R14 option B: share of the descend bonus the worker keeps inside live DIVE windows (default 0 = the old rule of full
        # wage stripping); only effective together with the sovereignty transfer; bookkeeping in _win_beat.
        _f = float(worker_descend_bonus_fraction)
        if not np.isfinite(_f) or not 0.0 <= _f <= 1.0:
            raise ValueError(
                "worker_descend_bonus_fraction must be within [0,1], "
                f"got {worker_descend_bonus_fraction!r}")
        if _f > 0.0 and not self.dive_live_sovereignty:
            raise ValueError(
                "worker_descend_bonus_fraction only takes effect together with dive_live_sovereignty "
                "(outside the live-DIVE rule domain there is no descent to reward)")
        self.worker_descend_bonus_fraction = _f
        self.descend_bonus_kept_total = 0.0
        # v32 potion autonomy (course 4C): default True = the normal state of the new protocol; the worker may press 12 on its own;
        # the 0.5 reflex (embedded in _drain/dispatch) is untouched and always the fallback. False is a knob only for
        # control legs/reproducing the old protocol. A Worker with a deployment contract can be the single source of truth under the None
        # default; if the caller also gives a value explicitly, the two must agree.
        self.drink_sovereignty = _resolve_worker_drink_sovereignty(
            self._workers, drink_sovereignty)
        # E-fix B (form 1): DIVE stall-clock protocol. "no-progress-v1" = the normal state of the new protocol:
        # only "strict improvement of the historical minimum target distance ∨ a kill ∨ the full positive_progress
        # set" extends a DIVE window; the KILL_PATIENCE constant (the normalization denominator of the frozen observation)
        # is untouched, and TAU_CAP=600 is still the hard cap. Bare displacement is not progress (a limit cycle moves every tick,
        # so a bare-displacement criterion would burn a dead window up to TAU_CAP). "tau-v3" = the old protocol's purely time-based
        # window close, an end point only for control legs/bit-level replay of old archives.
        if dive_stall_protocol not in ("tau-v3", "no-progress-v1"):
            raise ValueError(
                "dive_stall_protocol must be 'tau-v3' or 'no-progress-v1', "
                f"got {dive_stall_protocol!r}")
        self.dive_stall_protocol = dive_stall_protocol
        if worker_observation_view not in WORKER_OBSERVATION_VIEWS:
            raise ValueError(
                "worker_observation_view must be one of "
                f"{sorted(WORKER_OBSERVATION_VIEWS)}, got "
                f"{worker_observation_view!r}")
        self.worker_observation_view = worker_observation_view
        if manager_observation_view not in MANAGER_OBSERVATION_VIEWS:
            raise ValueError(
                "manager_observation_view must be one of "
                f"{sorted(MANAGER_OBSERVATION_VIEWS)}, got "
                f"{manager_observation_view!r}")
        self.manager_observation_view = manager_observation_view
        env_kwargs.setdefault("descend_ladder", True)
        env_kwargs.setdefault("death_ladder", True)
        env_kwargs.setdefault("start_in_dungeon", True)
        env_kwargs.setdefault("include_raw", False)
        if not bool(env_kwargs["descend_ladder"]):
            raise ValueError(
                "OptionsEnv requires descend_ladder=True; the Worker wage deducts the same depth bonus from the base "
                "reward, so turning it off would create negative wages out of nothing")
        required_controller_snapshot = (
            worker_observation_view in (
                WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
                WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
            )
        )
        requested_controller_snapshot = env_kwargs.setdefault(
            "controller_snapshot_enabled", required_controller_snapshot)
        if bool(requested_controller_snapshot) != required_controller_snapshot:
            raise ValueError(
                "controller_snapshot_enabled must agree with worker_observation_view; "
                "only the dual-family views may enable the fixed controller wire")
        self.env = DiabloGymEnv(max_steps=max_steps, **env_kwargs)
        if self.resource_calibration is not None:
            self.env._resource_calibration = self.resource_calibration
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(3)
        base = self.env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base + self.N_EXTRA_MGR,), dtype=np.float32)
        self._last_base_obs = None
        self._win = None
        self._reset_wrapper_state()

    # ---- wrapper state (persists across options) ----
    def _reset_wrapper_state(self):
        from .resource_protocol import ResourceService
        service_type = ResourceService
        if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v2":
            from .resource_sustain import SustainResourceService
            service_type = SustainResourceService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v3":
            from .resource_sustain_gold import SustainGoldMemoryService
            service_type = SustainGoldMemoryService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v4":
            from .resource_sustain_gold import SustainGoldExtendedService
            service_type = SustainGoldExtendedService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v5":
            from .resource_sustain_armor import SustainOrdinaryArmorService
            service_type = SustainOrdinaryArmorService
        elif getattr(self, "resource_service_policy", "legacy-v1") == "sustain-v6":
            from .resource_sustain_armor import SustainEquipmentReadinessService
            service_type = SustainEquipmentReadinessService
        if getattr(self, "worker_time_protocol", "legacy").startswith("completion-l2"):
            from .resource_sustain_completion import SustainCompletionService
            from .completion_clock import CompletionClock, COMPLETION_RECIPES
            service_type = SustainCompletionService
            self._completion_clock = CompletionClock(COMPLETION_RECIPES[self.worker_time_protocol])
            self.env.max_steps = self._completion_clock.state.physical_deadline
            self.env._completion_prefix_active = False
            self.env._completion_time_failure = None
            self._completion_previous_scene = self._completion_scene(
                getattr(self.env, "_raw", None))
            self.env.resource_time_callback = self._completion_time_tick
            self.env.resource_time_info_callback = self.completion_time_telemetry
        if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
            from .resource_sustain_loot import SustainLootService
            service_type = SustainLootService
        self.resource_service = service_type(
            getattr(self, "resource_purchase_mode", "full"),
            calibration=getattr(self, "resource_calibration", None))
        if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
            service = self.resource_service
            self.env.resource_observation_callback = lambda inner, raw, steps: service.observe(raw, steps)
            if getattr(self.env, "_raw", None) is not None:
                service.observe(self.env._raw, int(self.env._steps))
        elif getattr(self, "resource_service_policy", "legacy-v1") in ("sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6"):
            memory = self.resource_service.gold_memory
            self.env.resource_observation_callback = lambda inner, raw, steps: memory.observe(raw, steps)
            if getattr(self.env, "_raw", None) is not None:
                memory.observe(self.env._raw, int(self.env._steps))
        self._resource_farm_ledgers = {}
        self._resource_command = None
        # R18-A retreat-v1: a fresh scripted retreat service per episode; None when off.
        if getattr(self, "resource_retreat", "off") != "off":
            from .resource_retreat import RetreatService
            self.retreat_service = RetreatService()
        else:
            self.retreat_service = None
        # R18-F portal-v1: a fresh scripted portal service per episode; None when off.
        if getattr(self, "resource_portal", "off") != "off":
            from .resource_portal import PortalService
            self.portal_service = PortalService()
        else:
            self.portal_service = None
        # R18-H sweep-v1: a fresh scripted chest/barrel sweep per episode; None when off.
        if getattr(self, "resource_sweep", "off") != "off":
            from .resource_sweep import SweepService
            self.sweep_service = SweepService()
        else:
            self.sweep_service = None
        # R18-H identify-v1: a fresh scripted Cain identify service per episode; None when off.
        if getattr(self, "resource_identify", "off") != "off":
            from .resource_identify import IdentifyService
            self.identify_service = IdentifyService()
        else:
            self.identify_service = None
        # R18-K smith-v1: a fresh scripted weapon leg per episode; None when off.
        if getattr(self, "resource_weapon_upgrade", "off") != "off":
            from .resource_weapon_upgrade import WeaponUpgradeService
            self.weapon_upgrade_service = WeaponUpgradeService(
                protocol=self.resource_weapon_upgrade)
        else:
            self.weapon_upgrade_service = None
        # The frozen V28 worker and M29 manager were both trained on db7d26c's protocol-v3
        # state. A new protocol may change window-closing dynamics but must not silently change the meaning of a column of the same width;
        # so the old no-new-kill clock/exhausted flag/level start are maintained separately, only for the frozen networks' observations.
        self._legacy_layer_clock = 0
        self._legacy_exhausted = False
        self._legacy_layer_steps0 = 0
        self._layer_steps0 = 0
        self.layer_clock = 0          # micro ticks without positive progress on this level (reset by combat/exploration/supplies/level change)
        self.exhausted = False        # exhausted flag (cleared by any verifiable positive progress/level change)
        self.farm_scene_steps = 0     # cumulative FARM micro ticks in the current scene (not reset by positive progress)
        raw = getattr(self.env, "_raw", None)
        self._farm_scene = _scene_identity(raw) if raw is not None else None
        self._fuse_sig = None         # B4 fuse signature (persists across option boundaries)
        self._fuse = 0
        self._fuse_recovery_pending = False
        self._layer_kills0 = 0
        self._last_opt = -1
        self._last_tau = 0
        self._cap_hits = 0
        self._decisions = 0
        self.mode_seq = []
        self._win = None

    @staticmethod
    def _completion_scene(raw):
        if raw is None:
            return None
        return {key: raw[key] for key in ("engine_level", "is_set_level")}

    def completion_time_telemetry(self):
        """Detached clock facts only; no native query, reset, reward or outcome."""
        if getattr(self, "worker_time_protocol", "legacy") == "legacy":
            return None
        from dataclasses import asdict
        clock = self._completion_clock
        return {"protocol": self.worker_time_protocol,
                "recipe": clock.recipe.as_dict(),
                "clock": asdict(clock.state),
                "effective_physical_limit": int(self.env.max_steps),
                "observation_denominator": int(self.max_steps),
                "prefix_budget_active": bool(getattr(self.env, "_completion_prefix_active", False))}

    def _completion_time_tick(self, inner, raw, micro_step):
        clock = self._completion_clock
        target = self._completion_scene(raw)
        prefix_active = bool(getattr(inner, "_completion_prefix_active", False))
        expected = (getattr(inner, "_completion_prefix_deadline", None) if prefix_active
                    else clock.state.physical_deadline)
        if expected is None or int(inner.max_steps) != expected:
            raise RuntimeError("completion physical limit changed outside its owner")
        entering = (clock.state.first_arrival_micro_step is None
                    and target["engine_level"] == 2 and not target["is_set_level"])
        if entering:
            if prefix_active:
                # The engine already completed this tick; it is an engineering
                # stop, not a settled/credited prefix or permission to extend it.
                inner._completion_time_failure = {
                    "reason": "L2_before_learner_handoff",
                    "actual_native_microstep": int(micro_step),
                    "source_scene": self._completion_previous_scene,
                    "observed_scene": target,
                    "window_incomplete": True,
                    "physical_deadline_unchanged": int(inner.max_steps),
                }
                raise RuntimeError("completion L2 arrival before learner handoff")
            transition = raw.get("resource_state", {}).get("transition", {})
            # R17.1 readiness rule 3: under coach-v03 a forced-unready arrival is legal
            # (accepted, pretransition_ready False, accounted, no escrow); only
            # the receipt's existence/acceptance and the 1->2 geometry are law.
            ready_required = getattr(self, "resource_readiness_law", "veto-v1") != "coach-v03"
            if (not transition.get("accepted")
                    or (ready_required and not transition.get("pretransition_ready"))
                    or transition.get("source_depth") != 1 or transition.get("target_depth") != 2
                    or transition.get("source_is_set") or transition.get("target_is_set")):
                raise RuntimeError("completion L2 arrival lacks the live native readiness receipt")
        clock.observe_native_tick(micro_step, self._completion_previous_scene, target)
        self._completion_previous_scene = target
        if entering:
            inner.max_steps = clock.state.physical_deadline

    def _mark_exhausted(self) -> None:
        """Atomically publish the exhausted state of the current protocol.

        ``layer_clock``/``exhausted`` only govern the current window close and masks; the frozen networks read
        separate protocol-v3 state, so the scene cap or positive progress cannot change the meaning of old inputs.
        """
        self.exhausted = True
        self.layer_clock = max(int(self.layer_clock), KILL_PATIENCE)

    def _clear_exhausted(self) -> None:
        """Clear the hand-over state; the caller must already have confirmed that the scene/budget allows FARM to continue."""
        self.exhausted = False

    def _sig(self, a, raw):
        # The fuse may only recognize "the requested action truly made no positive progress", and must not mistake standing-still output for
        # being stuck. Player hp/mana are deliberately left out of the signature: enemies continuously hitting the player are not positive
        # progress of the requested action, otherwise a real wall-stuck state would keep extending itself by losing HP until death. All digests are sorted,
        # so jitter in the bridge's list enumeration order can never reset/trigger the fuse.
        monsters = tuple(sorted(
            (
             DiabloGymEnv._monster_generation_key(m),
             int(m.get("x", 0)), int(m.get("y", 0)),
             int(m.get("hp", 0)), int(m.get("max_hp", 0)))
            for m in DiabloGymEnv._policy_monsters(raw)
        ))
        def item_word(item, field, bits, *, default=None):
            return DiabloGymEnv._controller_uint(
                item.get(field, default),
                name=f"fuse floor_item.{field}",
                bits=bits,
            )

        floor_items = tuple(sorted(
            (
             item_word(it, "active_id", 7),
             int(it.get("x", 0)), int(it.get("y", 0)),
             bool(it.get("heal")), bool(it.get("gear")),
             item_word(it, "combat_utility_hi", 16),
             item_word(it, "combat_utility_lo", 16),
             item_word(it, "seed_hi", 16, default=0),
             item_word(it, "seed_lo", 16, default=0),
             item_word(it, "create_info", 16, default=0),
             item_word(it, "base_id", 16, default=0))
            for it in raw.get("floor_items", ())
            if bool(it.get("visible", True)) and bool(it.get("reachable", True))
        ))
        progression = tuple(sorted(
            (str(p.get("kind", "")), str(p.get("action", "")),
             int(p.get("x", 0)), int(p.get("y", 0)),
             int(p.get("goal_x", 0)), int(p.get("goal_y", 0)),
             bool(p.get("exact")))
            for p in raw.get("progression_targets", ())
        ))
        combat_floor = tuple(sorted(
            (
                (
                    tuple(int(part) for part in key)
                    if isinstance(key, tuple) and len(key) == 3
                    else (int(key), 0, 0)
                ),
                int(value[0]),
                int(value[1]),
            )
            for key, value in getattr(
                self.env, "_combat_hp_floor", {}
            ).items()
        ))
        return (a, raw["player_x"], raw["player_y"], raw.get("belt_heals", 0),
                raw.get("future_x", raw["player_x"]),
                raw.get("future_y", raw["player_y"]),
                DiabloGymEnv._belt_free_slots(raw),
                raw.get("xp", 0), raw.get("gold", 0),
                gear_combat_utility_value(raw, "fuse_signature"),
                raw["char_level"],
                self.env._ep_kills, _scene_identity(raw),
                int(getattr(self.env, "exploration_progress", 0)),
                combat_floor,
                monsters, floor_items, progression)

    def _tick_layer_clock(
        self,
        kills_before,
        scene_before,
        steps_delta,
        *,
        positive_progress: bool = False,
    ):
        """Advance both the old observation clock and the current FARM "no positive progress" clock.

        A kill is only one kind of progress. A new action10 visit/opening an ordinary softwall, a monster's lowest HP line
        dropping further, successfully getting belt supplies or putting on gear equally prove the current level is not exhausted;
        if only kills counted, exhausted would be misreported after about 12 explore macros while frontier still exists.
        Losing HP oneself, re-hitting a monster's already-paid HP band after it heals, and walking old tiles again do not reset the clock.
        """
        raw = self.env._raw
        scene_changed = _scene_identity(raw) != scene_before

        # protocol-v3 per-tick semantics: only a scene change or a new kill resets the clock; later additions such as exploration, damage,
        # pickups and equipping (positive_progress) must not affect the frozen input.
        if scene_changed:
            self._legacy_layer_clock = 0
            self._legacy_exhausted = False
            self._legacy_layer_steps0 = self.env._steps
        elif self.env._ep_kills > kills_before:
            self._legacy_layer_clock = 0
            self._legacy_exhausted = False
        else:
            self._legacy_layer_clock += steps_delta

        if scene_changed:
            self._sync_farm_scene(_scene_identity(raw))
            self.layer_clock = 0
            self._clear_exhausted()
            self._layer_kills0 = self.env._ep_kills
            self._layer_steps0 = self.env._steps
        elif self.env._ep_kills > kills_before or positive_progress:
            # The cumulative budget is a separate hand-over gate and cannot be reopened by a new floor tile or the next swing.
            if (getattr(self, "farm_scene_steps", 0)
                    < getattr(self, "farm_scene_cap", FARM_SCENE_CAP)):
                self.layer_clock = 0
                self._clear_exhausted()
            else:
                self._mark_exhausted()
        else:
            self.layer_clock += steps_delta
            # The cap may fire exactly inside a macro without positive progress; publish atomically before
            # _win_term, so that no debug/worker observation ever sees the intermediate state
            # "clock not full but flag full".
            if (getattr(self, "farm_scene_steps", 0)
                    >= getattr(self, "farm_scene_cap", FARM_SCENE_CAP)):
                self._mark_exhausted()

    def _sync_farm_scene(self, scene) -> None:
        """Atomically clear the cumulative FARM budget when the scene identity changes (applies to main levels and quest maps)."""
        scene = tuple(scene)
        if getattr(self, "_farm_scene", None) != scene:
            if getattr(self, "resource_protocol", "off") != "off":
                previous = getattr(self, "_farm_scene", None)
                if previous is not None:
                    self._resource_farm_ledgers[previous] = int(self.farm_scene_steps)
                self._farm_scene = scene
                self.farm_scene_steps = int(self._resource_farm_ledgers.get(scene, 0))
            else:
                self._farm_scene = scene
                self.farm_scene_steps = 0

    # ---- gym interface ----
    def reset(self, *, seed=None, options=None):
        # OptionsEnv is itself a Gym Env and must establish its own np_random;
        # passing seed only to the inner env would be judged by env_checker as violating the Gymnasium contract.
        super().reset(seed=seed)
        if getattr(self, "worker_time_protocol", "legacy").startswith("completion-l2"):
            self._completion_clock.reset()
            self.env.max_steps = self._completion_clock.state.physical_deadline
        if getattr(self, "resource_service_policy", "legacy-v1") in ("sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"):
            self.env.resource_observation_callback = None
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_wrapper_state()
        self._last_base_obs = obs
        return self._mgr_obs(obs), info

    def _controller_action_context(
        self,
    ) -> tuple[np.ndarray, int | None]:
        """Delegate to the real env; retain a narrow pure-fixture fallback."""
        builder = getattr(self.env, "controller_action_context", None)
        if callable(builder):
            return builder()
        # Unit-test shells predating the controller wire have no native map.
        # Production OptionsEnv always constructs DiabloGymEnv above and can
        # never enter this branch.
        raw = self.env._raw
        mask = np.ones(15, dtype=bool)
        mask[9] = bool(DiabloGymEnv._policy_monsters(raw))
        mask[12] = (
            int(raw.get("belt_heals", 0)) > 0
            and int(raw.get("hp", 0)) < int(raw.get("max_hp", 0))
        )
        mask[13] = (
            DiabloGymEnv._belt_free_slots(raw) > 0
            and bool(DiabloGymEnv._policy_floor_items(raw, "heal"))
        )
        mask[14] = bool(
            DiabloGymEnv._policy_floor_items(raw, "gear"))
        return mask, _nearest(raw)

    def action_masks(self) -> np.ndarray:
        if self._last_base_obs is None:
            raise gym.error.ResetNeeded("OptionsEnv.action_masks() needs reset() first")
        self.env._ensure_active(allow_ended=True)
        raw = self.env._raw
        self._sync_farm_scene(_scene_identity(raw))
        m = np.ones(3, dtype=bool)
        # DIVE = advance the next main-line target, no longer narrowly the NEXT stairs: quest entrances,
        # Vile mechanisms, the L16 door mechanism and set-level returns all belong to the same authority.
        transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                      else bridge.WM_DIABNEXTLVL)
        m[DIVE] = bool(raw.get("progression_targets")) or any(
            t.get("msg") == transition for t in raw.get("triggers", []))
        controller_mask, nearest = self._controller_action_context()
        # RESUPPLY must mean "a13 can execute now", not merely "a potion
        # exists somewhere in the full raw list".  The latter opened endless
        # two-wait windows for a visible potion just outside radius 12.
        m[RESUPPLY] = bool(controller_mask[13])
        # Story targets are DIVE's exclusive authority. The exhausted state must also hand over when DIVE is legal:
        # v4 removed action10's old leak of "no frontier, so sneak downstairs", and the frozen old H
        # relied on that leak, re-choosing FARM continuously even with layer_clock saturated until the whole game
        # was truncated. If DIVE is currently illegal, FARM remains the fallback, so the mask is never all false, and
        # dry-level revisits stay constrained by REVISIT_FLOOR.
        forced_dive = _farm_handoff(
            raw, nearest) or (self.exhausted and bool(m[DIVE]))
        m[FARM] = not forced_dive
        if getattr(self, "resource_protocol", "off") != "off":
            from .resource_protocol import progression_allowed, native_readiness, RESOURCE_FARM_CAP
            law = getattr(self, "resource_readiness_law", "veto-v1")
            # veto-v1 masks DIVE on the native verdict; coach-v03 leaves the frozen
            # DIVE legality untouched (progression_allowed returns True under it).
            m[DIVE] = bool(m[DIVE] and progression_allowed(raw, law))
            service = self.resource_service
            if getattr(self, "resource_service_policy", "legacy-v1") == "sustain-loot-v1":
                alive = sum(1 for monster in raw.get("monsters", [])
                            if int(monster.get("hp", 0)) > 0 and int(monster.get("type", -1)) != 109)
                cleared_now = alive == 0 and int(raw.get("monster_kill_total", 0)) > 0
                sweep = getattr(self, "sweep_service", None)
                if sweep is not None:
                    # R18-H (2026-09-07) review round: observe() also records the
                    # lit-object memory, so the sweep only ever targets objects
                    # this episode has actually seen (partial observability).
                    sweep.observe(raw)
                    sweep_portal = getattr(self, "portal_service", None)
                    sweep_retreat = getattr(self, "retreat_service", None)
                    if (not sweep.active and not service.active
                            and not getattr(sweep_portal, "active", False)
                            and not getattr(sweep_retreat, "active", False)):
                        # R18-H sweep-v1: consulted BEFORE maybe_start, on exactly
                        # the beat the loot trip would depart (floor cleared / FARM
                        # cap), so the chest and barrel drops are already on the
                        # floor when the collect stage runs. maybe_start is not
                        # called at all while the sweep owns the word.  A live
                        # portal/retreat window keeps the word: the RESUPPLY loop
                        # prefers the sweep, so starting one mid-leg would steal
                        # the body from a script already walking.
                        from .completion_clock import COMPLETION_L2_V1
                        calibration = getattr(self, "resource_calibration", None)
                        # Review round 2026-09-07: the sweep drops loot that only a
                        # town trip's collect stage turns into gold, so it never
                        # opens once the trip limit is spent.  The remaining
                        # maybe_start denials are deliberately NOT copied (see the
                        # resource_sweep module docstring); the window ledger
                        # records the facts so the case stays measurable.
                        sweep_readiness = native_readiness(raw)
                        sweep_deficit = bool(not sweep_readiness["ready"]
                                             and sweep_readiness["failures"])
                        sweep_slots = max(0, int(service.trip_limit(raw))
                                          - int(service.trip_count))
                        trigger = sweep.trigger_reason(
                            raw, int(self.env._steps),
                            farm_scene_steps=self.farm_scene_steps,
                            cleared=cleared_now,
                            farm_trigger=(COMPLETION_L2_V1.farm_microsteps
                                          if calibration is None
                                          else calibration.farm_scene_microstep_cap),
                            town_trip_active=bool(service.active),
                            loot_trip_slots_left=sweep_slots)
                        if trigger is not None:
                            sweep.start(raw, int(self.env._steps), trigger,
                                        loot_trip_slots_left=sweep_slots,
                                        readiness_deficit=sweep_deficit)
                if sweep is None or not sweep.active:
                    service.maybe_start(raw, self.env._steps,
                        farm_scene_steps=self.farm_scene_steps,
                        cleared=cleared_now)
            elif (not service.attempted and int(raw["dungeon_level"]) == 1
                    and not raw.get("is_set_level") and not native_readiness(raw)["ready"]):
                alive = sum(1 for monster in raw.get("monsters", [])
                            if int(monster.get("hp", 0)) > 0 and int(monster.get("type", -1)) != 109)
                cleared = alive == 0 and int(raw.get("monster_kill_total", 0)) > 0
                calibration = getattr(self, "resource_calibration", None)
                farm_trigger = (RESOURCE_FARM_CAP if calibration is None
                                else calibration.farm_scene_microstep_cap)
                if cleared or self.farm_scene_steps >= farm_trigger:
                    service.start(raw, self.env._steps, "cleared" if cleared else "farm_cap",
                                  farm_scene_steps=self.farm_scene_steps)
            retreat = getattr(self, "retreat_service", None)
            portal = getattr(self, "portal_service", None)
            if (portal is not None and portal.awaiting_return and not portal.active
                    and not service.active and int(raw.get("dungeon_level", 0)) == 0
                    and not raw.get("is_set_level")
                    and getattr(portal, "_town_visit_served", -1) != len(portal.attempts)):
                # R18-F portal-v1: a portal arrival in town runs the ordinary town
                # itinerary (heal / armor / potions / sell / scroll) BEFORE the return
                # leg. The loot service normally departs from L1; started here it
                # skips the stairs walk (outbound phase: depth 0 -> smith/healer) and
                # is closed at its return phase by the RESUPPLY-loop intercept, after
                # which the portal return trigger fires. One itinerary per visit.
                portal._town_visit_served = len(portal.attempts)
                if hasattr(service, "_begin_trip"):
                    # sustain-loot-v1 forbids direct start(); its trip entry is _begin_trip
                    # (maybe_start cannot fire in town by design). The visit counts as a trip.
                    service._begin_trip(raw, int(self.env._steps), "portal_arrival",
                                        int(self.farm_scene_steps), ["portal_arrival"])
                else:
                    service.start(raw, self.env._steps, "portal_arrival",
                                  farm_scene_steps=self.farm_scene_steps)
                # The collect phase (gold pickup) belongs to L1 and rejects town as an
                # unexpected scene; a portal visit enters the itinerary at outbound
                # (depth 0 -> sell / smith / healer / potions), which is exactly what
                # collect does when it finds nothing to pick up.
                service.phase = service._last_phase = "outbound"
                bridge.configure_town_service(True)
            if (portal is not None and not portal.active and not service.active
                    and (retreat is None or not retreat.active)):
                # R18-M (2026-09-07) review round — why this guard does NOT
                # mention the sweep, although a sweep may have started earlier in
                # this same action_masks() call (options_env.py:998-1032) against
                # the same immutable `raw`: the depths are disjoint by law.
                # resource_sweep.py:205-206 refuses any dungeon_level != 1;
                # resource_portal.py:240-243 handles depth 0 and refuses depth < 2;
                # resource_retreat.py:138-139 refuses depth < 2. On the only beat a
                # sweep can start, trigger_reason here always returns None, so no
                # second service can be started on top of it and the owner chain at
                # :2341-2344 never has to arbitrate. Widen the sweep's depth (or
                # copy this block for a new leg) and that stops being true — add
                # `and not (sweep is not None and sweep.active)` here and below.
                # R18-F portal-v1: consulted BEFORE the walking retreat. Its
                # outbound law is the retreat law PLUS "a scroll is carried", so
                # the coach prefers the portal exactly when the pair has one and
                # falls through to the stairs retreat otherwise. It also owns the
                # town-side legs (buy the scroll from Adria; walk back through).
                trigger = portal.trigger_reason(raw, int(self.env._steps))
                if trigger is not None:
                    portal.start(raw, int(self.env._steps), bridge, trigger)
            if (retreat is not None and not retreat.active and not service.active
                    and (portal is None or not portal.active)):
                # R18-M (2026-09-07) review round: same disjoint-depth invariant as
                # the portal guard above (resource_sweep.py:205 L1-only vs
                # resource_retreat.py:138 depth >= 2) is what keeps the missing
                # `sweep.active` term inert here.
                # R18-A retreat-v1: the manager-side law fires here, exactly where the
                # town trip starts; the mask then collapses to RESUPPLY (= retreat).
                trigger = retreat.trigger_reason(raw, int(self.env._steps))
                if trigger is not None:
                    retreat.start(raw, int(self.env._steps), bridge, trigger)
            sweep = getattr(self, "sweep_service", None)
            if sweep is not None and sweep.active:
                m[:] = False
                m[RESUPPLY] = True
            elif portal is not None and portal.active:
                m[:] = False
                m[RESUPPLY] = True
            elif retreat is not None and retreat.active:
                m[:] = False
                m[RESUPPLY] = True
            elif service.active:
                m[:] = False
                m[RESUPPLY] = True
            elif law == "coach-v03":
                # R17.1 readiness rule 3: the frozen forced-descent law stands (escape hatch
                # kept); forced-unready descents are recorded by the engine receipt
                # and paid no escrow (worker_env settlement reads pretransition_ready_law).
                m[FARM] = not forced_dive
                m[RESUPPLY] = bool(controller_mask[13])
            else:
                # veto-v1: no exhausted/cleared bypass of the native admission verdict.
                m[FARM] = not (forced_dive and m[DIVE])
                m[RESUPPLY] = bool(controller_mask[13])
            self.env._resource_dive_authority = bool(
                self.dive_live_sovereignty and (self._win or {}).get("opt") == DIVE
                and m[DIVE])
        return m

    def resource_option_choice(self, mask=None):
        if getattr(self, "resource_protocol", "off") == "off":
            raise RuntimeError("Resource manager requires l2-town-v1")
        from .resource_protocol import law_ready
        mask = self.action_masks() if mask is None else mask
        sweep = getattr(self, "sweep_service", None)
        if sweep is not None and sweep.active:
            return RESUPPLY
        portal = getattr(self, "portal_service", None)
        if portal is not None and portal.active:
            return RESUPPLY
        retreat = getattr(self, "retreat_service", None)
        if retreat is not None and retreat.active:
            return RESUPPLY
        if self.resource_service.active:
            return RESUPPLY
        raw = self.env._raw
        # veto-v1: native seven-condition verdict; coach-v03: six-condition law
        # (health excluded). The town-trip trigger above keeps the full native
        # verdict so low HP still sends the manager to Pepin (readiness rule 3).
        if mask[DIVE] and (raw.get("is_set_level") or raw.get("progression_targets")
                           or law_ready(raw, getattr(self, "resource_readiness_law", "veto-v1"))):
            return DIVE
        for option in (FARM, RESUPPLY, DIVE):
            if mask[option]:
                return option
        raise RuntimeError("Resource manager has no legal option")

    # ---- shared window core (v23: the single implementation for OptionsEnv and WorkerWindowEnv) ----
    def _dive_target_distance(self, raw) -> int | None:
        """Chebyshev distance to DIVE's current main-line target (same definition as action_masks:
        the minimum over story targets if any, otherwise the nearest transition trigger)."""
        px, py = raw["player_x"], raw["player_y"]
        targets = raw.get("progression_targets") or []
        if targets:
            return min(
                max(abs(int(t["goal_x"]) - px), abs(int(t["goal_y"]) - py))
                for t in targets)
        transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                      else bridge.WM_DIABNEXTLVL)
        stairs = [t for t in raw.get("triggers", [])
                  if t.get("msg") == transition]
        if not stairs:
            return None
        return min(
            max(abs(t["x"] - px), abs(t["y"] - py)) for t in stairs)

    def _win_begin(self, option: int):
        if not self.action_space.contains(option):
            raise ValueError(f"option must be an integer in {self.action_space}, got {option!r}")
        option = int(option)
        if getattr(self, "resource_protocol", "off") != "off":
            self.env._resource_dive_authority = False
        if not self.action_masks()[option]:
            raise ValueError(f"option {option} is masked but was chosen")
        raw = self.env._raw
        if (option == FARM
                and getattr(self, "reset_layer_clock_on_window", False)):
            # R16 (C3 clock inheritance): a new FARM window counts no-progress from zero and does not inherit the balance of the previous
            # DIVE stall; the exhausted flag and the legacy clock are untouched (frozen observation semantics).
            self.layer_clock = 0
        self._win = {
            # _decisions only increments in _win_end, so within one underlying game this is a stable,
            # monotone identifier that can be reported as soon as the window starts.
            "window_id": self._decisions + 1,
            "opt": option,
            "mode": ("farm", "dive", "resupply")[option],
            "t0": self.env._steps,
            "clvl0": raw["char_level"],
            "dlvl0": raw["dungeon_level"],
            "scene0": _scene_identity(raw),
            "kills0": int(self.env._ep_kills),
            "floor": REVISIT_FLOOR if (option == FARM and self.exhausted) else 0,
            "dive_best_d": (self._dive_target_distance(raw)
                            if option == DIVE else None),
            "dive_last_progress_tau": 0,
            "resupply_stall": 0,
            "R": 0.0, "W": 0.0, "bonus": 0.0,
            # These two ledgers only cover _win_step_worker: the fuse recovery/brainstem drain before the window opens
            # are not network transitions and must not be mixed into the wage PPO actually receives
            # or into "kills accompanying the worker". The wage returned by WorkerWindowEnv.step and the increments accumulated
            # step by step here use the same W difference, and the two must come from exactly the same source.
            "worker_wage": 0.0, "worker_kills": 0,
            "worker_action14_requests": 0,
            "worker_action14_native_successes": 0,
            "worker_action14_gear_utility_delta": 0,
            "no_effect_requests": 0,
            "worker_no_effect_requests": 0,
            "executed_requests": 0,
            # After a FARM kill exposes progression, give the Worker exactly
            # one visible {decline, equip} decision before manager handoff.
            "gear_grace_pending": False,
            "gear_grace_consumed": False,
            "gear_grace_opportunities": 0,
            "gear_grace_decisions": 0,
            "beats": 0, "overrides": 0, "fuse_trips": 0,
            # attempts includes hit-recovery rejections; drains counts only
            # native-accepted potion uses.  Conflating them previously made
            # failed rescues look successful in evaluation.
            "drain_attempts": 0, "drains": 0,
            "voluntary_drinks": 0,
            "recovery_actions": 0, "last_recovery_action": None,
            "last_requested_action": None, "last_executed_action": None,
            "fuse_requested_action": None,
            "done": False, "trunc": False, "last_info": {},
        }
        if getattr(self, "resource_protocol", "off") != "off":
            self._win["resource_depth0"] = int(self.env._resource_max_main_depth)
            retreat = getattr(self, "retreat_service", None)
            self._win["retreat_window"] = bool(
                retreat is not None and retreat.active and option == RESUPPLY)
            portal = getattr(self, "portal_service", None)
            self._win["portal_window"] = bool(
                portal is not None and portal.active and option == RESUPPLY)
            sweep = getattr(self, "sweep_service", None)
            self._win["sweep_window"] = bool(
                sweep is not None and sweep.active and option == RESUPPLY)

    def _beat(self, a: int, *, worker_authority: bool = False):
        """One tick: fuse -> env.step -> observation cache -> stall clock.

        Returns ``(r, done, trunc, info, audit, lvl_before, belt_free_before)``.
        When the fuse fires no base action is executed and ``audit.executed_action`` is None;
        the caller must end the current window immediately with ``reason="fuse"``.
        """
        raw = self.env._raw
        requested = int(a)
        sig = (("resource", tuple(self._resource_command), int(self.env._steps))
               if getattr(self, "_resource_command", None) is not None
               else self._sig(a, raw))
        if sig == self._fuse_sig:
            self._fuse += 1
            if self._fuse >= 25:
                # The old behaviour silently executed action 10 here while PPO/BC still labelled this transition
                # as requested, mislabelling action and result. Now this tick is refused and control returns to the
                # manager; the signature is cleared so the next window starts from a clean fuse state.
                self._fuse = 0
                self._fuse_sig = None
                # A fuse refusal executes no native action; a14's receipt enforcement (every request must carry a
                # causal receipt) requires an explicit refusal receipt here too, otherwise the consumer fails closed.
                # (Fixed 2026-07-27: when receipt enforcement was added on 07-25 this early-exit branch was missed;
                # the prepare-bc teacher crashed as soon as pressing a14 repeatedly for 25 ticks triggered the fuse.)
                fuse_action14_audit = None
                if requested == 14:
                    fuse_utility = gear_combat_utility_value(
                        raw, "action14_fuse_reject")
                    fuse_action14_audit = {
                        "accepted": False,
                        "commit_attempts": 0,
                        "utility_before": fuse_utility,
                        "utility_after": fuse_utility,
                        "utility_delta": 0,
                    }
                audit = BeatOutcome(
                    reason="fuse",
                    requested_action=requested,
                    executed_action=None,
                    fuse_tripped=True,
                    action14_audit=fuse_action14_audit,
                )
                return 0.0, False, False, {}, audit, \
                    raw["dungeon_level"], DiabloGymEnv._belt_free_slots(raw)
        else:
            self._fuse = 0
        self._fuse_sig = sig
        kills_b = self.env._ep_kills
        lvl_b = raw["dungeon_level"]
        scene_b = _scene_identity(raw)
        self._sync_farm_scene(scene_b)
        steps_b = self.env._steps
        belt_free_b = DiabloGymEnv._belt_free_slots(raw)
        exploration_b = int(getattr(self.env, "exploration_progress", 0))
        gear_utility_b = gear_combat_utility_value(
            raw, "options_before")
        gold_b = int(raw.get("gold", 0))
        combat_floor_b = dict(getattr(self.env, "_combat_hp_floor", {}))
        if getattr(self, "_resource_command", None) is not None:
            obs, r, done, trunc, info = self.env.step_resource(self._resource_command)
        elif worker_authority:
            obs, r, done, trunc, info = self.env.step(
                a, worker_authority=True)
        else:
            obs, r, done, trunc, info = self.env.step(a)
        self._last_base_obs = obs
        current = self.env._raw
        action14_audit = (
            _validated_action14_audit(info)
            if requested == 14 else None
        )
        action_effect_audit = _validated_action_effect_audit(
            info, requested)
        steps_delta = self.env._steps - steps_b
        current_scene = _scene_identity(current)
        if current_scene != scene_b:
            self._sync_farm_scene(current_scene)
        elif (getattr(self, "_win", None) is not None
              and int(self._win.get("opt", -1)) == FARM):
            self.farm_scene_steps += steps_delta
        combat_floor = getattr(self.env, "_combat_hp_floor", {})
        new_damage_floor = any(
            mid in combat_floor and int(combat_floor[mid][0]) < int(before[0])
            for mid, before in combat_floor_b.items()
        )
        positive_progress = (
            int(getattr(self.env, "exploration_progress", 0)) > exploration_b
            or new_damage_floor
            or (
                action14_audit is not None
                and action14_audit["accepted"]
            )
            or gear_combat_utility_value(
                current, "options_after") > gear_utility_b
            or DiabloGymEnv._belt_free_slots(current) < belt_free_b
            or int(current.get("gold", 0)) > gold_b
        )
        self._tick_layer_clock(
            kills_b,
            scene_b,
            steps_delta,
            positive_progress=positive_progress,
        )
        _dive_w = getattr(self, "_win", None)
        if (_dive_w is not None and _dive_w.get("opt") == DIVE
                and getattr(self, "dive_stall_protocol", "tau-v3")
                == "no-progress-v1"):
            w = _dive_w
            d_now = self._dive_target_distance(current)
            best = w.get("dive_best_d")
            improved = (
                d_now is not None
                and (best is None or d_now < best)
            )
            if improved:
                w["dive_best_d"] = d_now
            if (improved
                    or int(self.env._ep_kills) > int(kills_b)
                    or positive_progress):
                w["dive_last_progress_tau"] = self.env._steps - w["t0"]
        executed_action = (
            int(a)
            if action_effect_audit["request_executed"]
            else None
        )
        if requested == 12:
            drink_audit = info.get("action12_audit")
            if (
                not isinstance(drink_audit, dict)
                or set(drink_audit) != {
                    "accepted", "accepted_belt_before",
                    "belt_before", "consumed", "belt_after",
                }
                or not isinstance(drink_audit["accepted"], bool)
                or not isinstance(drink_audit["consumed"], bool)
                or drink_audit["accepted"] != drink_audit["consumed"]
            ):
                raise RuntimeError(
                    "action12 real execution receipt missing/corrupt")
            executed_action = 12 if drink_audit["consumed"] else None
        elif requested == 14:
            # A14 is a macro: an accepted safe walk/open is a real execution
            # even when this decision has not reached the final gear commit.
            # Keep completion in action14_audit; only require the one-way
            # implication that a successful commit was truly executed.
            if action14_audit["accepted"] and executed_action != 14:
                raise RuntimeError(
                    "action14 successful commit lacks the generic execution receipt")
        audit = BeatOutcome(
            reason=None,
            requested_action=requested,
            executed_action=executed_action,
            fuse_tripped=False,
            action14_audit=action14_audit,
            action_effect_audit=action_effect_audit,
        )
        return float(r), done, trunc, info, audit, lvl_b, belt_free_b

    def _win_term(self, done, trunc, belt_free_b):
        """Seven-step termination ladder (order structurally identical to v22 line by line). Returns reason or None."""
        w = self._win
        raw = self.env._raw
        tau = self.env._steps - w["t0"]
        if done or trunc:
            return "death" if raw.get("dead") else "end"
        if _scene_identity(raw) != w["scene0"]:
            # Resource service returns town -> visited L1. Only a newly reached
            # main depth is descent; a scene return must not inflate statistics.
            if getattr(self, "resource_protocol", "off") != "off":
                return ("descend" if int(self.env._resource_max_main_depth)
                        > w["resource_depth0"] else "scene")
            # Entering/leaving a quest set-level must also hand control back, but only an increase in main-line depth
            # counts as descend and may collect the dive bonus.
            return ("descend" if raw["dungeon_level"] > w["dlvl0"]
                    else "scene")
        opt = w["opt"]
        sweep = getattr(self, "sweep_service", None)
        if sweep is not None:
            # R18-H sweep-v1: a sweep window closes exactly when the script hands
            # back, before the ordinary RESUPPLY belt ladder can close it early.
            if w.get("sweep_window"):
                return None if sweep.active else "sweep_complete"
            if opt != RESUPPLY and sweep.active:
                # Review round 2026-09-07: action_masks() also runs inside the
                # frozen worker's dual observation build, so the sweep law can
                # fire mid FARM/DIVE window.  Close that window at once — its
                # embedded manager mask is already collapsed to RESUPPLY, and the
                # sweep must not be charged for a tail it did not own.  (The
                # service re-baselines its own clocks on its first command; this
                # rung is what stops the frozen worker acting under a mask that
                # no longer describes its window.)
                return "sweep_trigger"
        portal = getattr(self, "portal_service", None)
        if portal is not None:
            # R18-F portal-v1 (constructed only when the flag is on): a portal
            # window closes when the script hands back; any other window closes
            # the beat a portal leg becomes legal so the mask can take the word.
            if w.get("portal_window"):
                return None if portal.active else "portal_complete"
            if (opt != RESUPPLY and not portal.active
                    and portal.trigger_reason(raw, self.env._steps) is not None):
                return "portal_trigger"
        retreat = getattr(self, "retreat_service", None)
        if retreat is not None:
            # R18-A retreat-v1 (constructed only when the flag is on): a retreat
            # window closes when the script hands back; any other window closes
            # the beat the retreat law fires so the mask can take the word.
            if w.get("retreat_window"):
                return None if retreat.active else "retreat_complete"
            if (opt != RESUPPLY and not retreat.active
                    and retreat.trigger_reason(raw, self.env._steps) is not None):
                return "retreat_trigger"
        # Once FARM clears the last nearby enemy, a story target may become the current state in the same tick;
        # the window must close before the revisit floor/stall clock, so action10 cannot overstep its authority.
        if opt == FARM:
            controller_mask, nearest = self._controller_action_context()
            if _farm_handoff(raw, nearest):
                exact_gear = bool(controller_mask[14])
                if (
                    not w.get("gear_grace_consumed", False)
                    and exact_gear
                ):
                    if (
                        int(w.get("gear_grace_decisions", 0))
                        >= GEAR_GRACE_MAX_DECISIONS
                    ):
                        w["gear_grace_consumed"] = True
                        w["gear_grace_pending"] = False
                        return "handoff"
                    if not w.get("gear_grace_pending", False):
                        w["gear_grace_pending"] = True
                        if int(w.get(
                                "gear_grace_opportunities", 0)) == 0:
                            w["gear_grace_opportunities"] = 1
                    return None
                w["gear_grace_pending"] = False
                return "handoff"
        if tau < w["floor"]:
            return None
        if opt == FARM and raw["char_level"] > w["clvl0"]:
            return "levelup"
        # db7d26c publishes the old exhausted only at this level of FARM, and after done/scene/floor/levelup.
        # Even if the current no-progress does not close the window because of real positive progress, the frozen V28
        # should still see the 296/297 it would have received at this moment under the old protocol.
        if opt == FARM and self._legacy_layer_clock >= KILL_PATIENCE:
            self._legacy_exhausted = True
        if (opt == FARM
                and (self.layer_clock >= KILL_PATIENCE
                     or self.farm_scene_steps
                     >= getattr(self, "farm_scene_cap", FARM_SCENE_CAP))):
            self._mark_exhausted()
            return "exhausted"
        if opt == DIVE:
            if (getattr(self, "dive_stall_protocol", "tau-v3")
                    == "no-progress-v1"):
                if (tau - int(w.get("dive_last_progress_tau", 0))
                        >= KILL_PATIENCE):
                    return "stall"
            elif tau >= KILL_PATIENCE:
                return "stall"
        if opt == RESUPPLY and getattr(self, "resource_protocol", "off") != "off" and self.resource_service.attempted:
            if self.resource_service.active:
                return None
            return "resource_complete"
        if opt == RESUPPLY:
            belt_free = DiabloGymEnv._belt_free_slots(raw)
            if belt_free >= belt_free_b:
                w["resupply_stall"] += 1
            else:
                w["resupply_stall"] = 0
            exact_heal_available = bool(
                self._controller_action_context()[0][13])
            if (belt_free <= 0
                    or not exact_heal_available
                    or w["resupply_stall"] >= 2 or tau >= RESUPPLY_CAP):
                return "done"
        if tau >= TAU_CAP:
            self._cap_hits += 1
            return "cap"
        return None

    def _win_beat(self, a: int, *, worker_authority: bool = False):
        """One tick + ledgers (manager R / worker W) + termination check."""
        w = self._win
        # Narrow compatibility for hand-built unit-test windows.  Production
        # windows initialize every ledger in _win_begin.
        for key in (
            "no_effect_requests",
            "worker_no_effect_requests",
            "executed_requests",
            "gear_grace_opportunities",
            "gear_grace_decisions",
        ):
            w.setdefault(key, 0)
        w.setdefault("gear_grace_pending", False)
        w.setdefault("gear_grace_consumed", False)
        r, done, trunc, info, audit, lvl_b, belt_free_b = self._beat(
            a, worker_authority=worker_authority)
        w["last_requested_action"] = audit.requested_action
        w["last_executed_action"] = audit.executed_action
        if audit.fuse_tripped:
            # A refused proposal consumes no base micro tick and produces no fake reward; it only registers an auditable
            # manager intervention. The minimum occupancy floor of a dry revisit is still a hard invariant:
            # while the floor is not reached this tick only refuses and does not close the window; the cleared fuse lets the next proposal
            # advance normally, avoiding a zero-micro-step infinite loop.
            w["overrides"] += 1
            w["fuse_trips"] += 1
            w["fuse_requested_action"] = audit.requested_action
            tau = self.env._steps - w["t0"]
            reason = "fuse" if tau >= w["floor"] else None
            if reason is not None:
                # The refusal tick itself secretly executes nothing. At the start of the next manager window one explicitly registered
                # scripted recovery tick runs, so the manager does not keep re-choosing the same bad geometric state.
                self._fuse_recovery_pending = True
            return BeatOutcome(
                reason=reason,
                requested_action=audit.requested_action,
                executed_action=None,
                fuse_tripped=True,
                # Second hole fixed 2026-07-27: the window-level fuse wrapper must propagate the inner
                # a14 refusal receipt (already synthesized on the _beat fuse path), otherwise an a14 request's receipt on
                # this path is None and the consumer fails closed (the cause of death of the 2_106 pool).
                action14_audit=audit.action14_audit,
                action_effect_audit=audit.action_effect_audit,
            )
        w["beats"] += 1
        if (
            audit.requested_action != 0
            and audit.action_effect_audit is not None
            and not audit.action_effect_audit["request_executed"]
        ):
            w["no_effect_requests"] += 1
        elif (
            audit.action_effect_audit is not None
            and audit.action_effect_audit["request_executed"]
            and audit.requested_action != 0
        ):
            w["executed_requests"] += 1
        w["R"] += r
        cur_lvl = self.env._raw["dungeon_level"]
        # The stripping unit price follows the economy spec (== DESCEND_UNIT under v1, values identical to the old bit for bit;
        # after R10 v2 changes the unit price the identity Σw ≡ window R − unit×ΣΔdlvl⁺ still holds,
        # and the anti-arbitrage semantics of "descending mid-fight to avoid penalties" are unchanged). Test doubles/old wrappers without this attribute
        # fall back to v1, identical to historical behaviour bit for bit.
        descend_unit = getattr(
            self.env, "reward_economy", REWARD_ECONOMY_V1).descend_unit
        bonus = descend_unit * sum(range(lvl_b, cur_lvl)) if cur_lvl > lvl_b else 0.0
        if getattr(self, "resource_protocol", "off") != "off":
            bonus = float(info["resource_protocol"]["new_main_depth_bonus"])
        # R14 option B (tried first, to see whether the "naked runner" behaviour still appears under
        # option B): inside live DIVE windows the worker keeps fraction of the descend bonus.
        # w["bonus"] means the amount actually stripped; the identity R≡W+bonus keeps holding bit for bit;
        # with fraction=0 (default) or outside live-DIVE windows this path is identical to the old rule bit for bit.
        if (bonus > 0.0
                and getattr(
                    self, "worker_descend_bonus_fraction", 0.0) > 0.0
                and getattr(self, "dive_live_sovereignty", False)
                and w.get("opt") == DIVE):
            _kept = bonus * self.worker_descend_bonus_fraction
            bonus -= _kept
            self.descend_bonus_kept_total = float(getattr(
                self, "descend_bonus_kept_total", 0.0)) + _kept
        w["bonus"] += bonus
        w["W"] += r - bonus
        w["done"], w["trunc"], w["last_info"] = done, trunc, info
        return BeatOutcome(
            reason=self._win_term(done, trunc, belt_free_b),
            requested_action=audit.requested_action,
            executed_action=audit.executed_action,
            fuse_tripped=False,
            action14_audit=audit.action14_audit,
            action_effect_audit=audit.action_effect_audit,
        )

    def _drain(self):
        """Reflex drain: while hp<0.5∧belt>0 the wrapper drinks tick by tick (worker path only;
        every tick goes through the fuse/clock/termination ladder as usual; a drink tick can cross exhausted/CAP/death).

        Returns the BeatOutcome that made the drain end the window; returns None if the drain finished and the window is still active.
        """
        w = self._win
        while _reflex(self.env._raw):
            w["drain_attempts"] += 1
            outcome = self._win_beat(12)
            if outcome.executed_action == 12:
                w["drains"] += 1
            if outcome.reason is not None:
                return outcome
        return None

    def _win_step_worker(self, a: int):
        """One worker step = one worker-action tick + reflex tail drain."""
        if not self.env.action_space.contains(a):
            raise ValueError(f"worker action must be an integer in {self.env.action_space}, got {a!r}")
        a = int(a)
        if (
            1 <= a <= 8
            and a in DiabloGymEnv._protected_walk_actions(self.env._raw)
            and not (
                getattr(self, "dive_live_sovereignty", False)
                and (self._win or {}).get("opt") == DIVE
            )   # R13: inside live DIVE windows stepping on the tile is within authority, same condition as the mask release
        ):
            raise ValueError(
                f"worker action {a} tried to step onto a DIVE-only trigger/story tile")
        if not self._worker_masks()[a]:
            raise ValueError(f"worker action {a} is masked but was executed")
        wage_before = float(self._win["W"])
        kills_before = int(self.env._ep_kills)
        grace_decision = bool(
            self._win.get("gear_grace_pending", False))
        if grace_decision:
            self._win["gear_grace_pending"] = False
            # a0 explicitly declines.  a14 keeps the grace open until the
            # exact native gear commit removes the target, or the bounded
            # retry budget is exhausted; one macro may need several decisions
            # to traverse the fixed radius-12 path.
            self._win["gear_grace_consumed"] = a == 0
            self._win["gear_grace_decisions"] += 1
        primary = (
            self._win_beat(a, worker_authority=True)
            if 1 <= a <= 8
            else self._win_beat(a)
        )
        if grace_decision and primary.fuse_tripped:
            # Fuse rejection executes no base action, so it cannot consume the
            # one learning opportunity.
            self._win["gear_grace_pending"] = True
            self._win["gear_grace_consumed"] = False
            self._win["gear_grace_decisions"] -= 1
        if a == 12 and primary.executed_action == 12:
            # This is worker-initiated drinking, excluding the brainstem reflexes of _drain. The count is only for audits;
            # it is published in the sign of feature 297 of the next policy observation and closes
            # later worker-owned a12 in this window; the brainstem _drain can still drain continuously in emergencies.
            # This makes "at most one worker drink per window" both a visible state and an execution constraint.
            self._win["voluntary_drinks"] += 1
        if a == 14:
            self._win.setdefault("worker_action14_requests", 0)
            self._win.setdefault(
                "worker_action14_native_successes", 0)
            self._win.setdefault(
                "worker_action14_gear_utility_delta", 0)
            self._win["worker_action14_requests"] += 1
            gear_audit = primary.action14_audit
            if gear_audit is None:
                raise RuntimeError("action14 lacks the native gear-commit receipt")
            if gear_audit["accepted"]:
                if (
                    primary.executed_action != 14
                    or gear_audit["utility_delta"] <= 0
                ):
                    raise RuntimeError(
                        "action14 successful commit did not publish a real execution receipt")
                self._win[
                    "worker_action14_native_successes"
                ] += 1
                self._win[
                    "worker_action14_gear_utility_delta"
                ] += int(gear_audit["utility_delta"])
                if grace_decision:
                    # The grace budget exists to let one fixed pickup macro
                    # finish walking to its target.  Once the native atomic
                    # commit succeeds, hand control back even if another
                    # upgrade happens to become the new nearest target.
                    self._win["gear_grace_consumed"] = True
        if (
            a != 0
            and primary.action_effect_audit is not None
            and not primary.action_effect_audit["request_executed"]
        ):
            self._win["worker_no_effect_requests"] += 1
        ending = primary
        if primary.reason is None:
            drain_ending = self._drain()
            if drain_ending is not None:
                ending = drain_ending
        # Exactly the same as the policy reward of WorkerWindowEnv.step: the W increment caused by the current worker proposal
        # and its reflex tail. The window-opening recovery/drain happens before this method is called,
        # so it is never misattributed to the network.
        self._win["worker_wage"] += float(self._win["W"]) - wage_before
        worker_kills = int(self.env._ep_kills) - kills_before
        if worker_kills < 0:
            raise RuntimeError("per-game kill count went backwards within a worker transition")
        self._win["worker_kills"] += worker_kills
        fuse = primary if primary.fuse_tripped else (
            ending if ending.fuse_tripped else None)
        return WorkerStepOutcome(
            reason=ending.reason,
            requested_action=a,
            executed_action=primary.executed_action,
            fuse_tripped=fuse is not None,
            fuse_requested_action=(None if fuse is None
                                   else fuse.requested_action),
            action14_audit=primary.action14_audit,
            action_effect_audit=primary.action_effect_audit,
        )

    def _consume_fuse_recovery(self):
        """Execute one auditable recovery tick at the start of the next window; not attributed to the refused worker action."""
        if not self._fuse_recovery_pending:
            return None
        mode = self._win["mode"]
        if mode == "farm":
            raw = self.env._raw
            controller_mask, _nearest = (
                self._controller_action_context())
            # FARM recovery must not use action10 to operate story targets either; with a story target it keeps clearing
            # visible blockers (the monster-free hand-over state never opens FARM anyway).
            action = (
                9 if (raw.get("progression_targets")
                      and bool(controller_mask[9]))
                else (0 if raw.get("progression_targets") else 10)
            )
        else:
            action = {"dive": 11, "resupply": 0}[mode]
        self._fuse_recovery_pending = False
        self._win["recovery_actions"] += 1
        self._win["last_recovery_action"] = action
        return self._win_beat(action)

    def _win_end(self, reason: str):
        """Close the window: advance the manager state machine + option_extra. Returns (extra, base_info, done, trunc)."""
        if reason is None:
            raise RuntimeError("attempted to close the window while the option has no termination reason")
        w = self._win
        for key in (
            "no_effect_requests",
            "worker_no_effect_requests",
            "executed_requests",
            "gear_grace_opportunities",
            "gear_grace_decisions",
        ):
            w.setdefault(key, 0)
        tau = self.env._steps - w["t0"]
        terminal_info = (
            w["last_info"] if isinstance(w.get("last_info"), dict) else {})
        safe_time_limit = (
            terminal_info.get("time_limit_bootstrap_safe") is True)
        unsettled_time_limit = (
            terminal_info.get("unsettled_budget_terminal") is True)
        native_terminal = bool(
            self.env._raw.get("dead")
            or self.env._raw.get("game_over")
            or self.env._raw.get("victory")
        )
        budget_boundary = bool(
            not native_terminal
            and safe_time_limit != unsettled_time_limit
        )
        self._decisions += 1
        self._last_opt, self._last_tau = w["opt"], tau
        self.mode_seq.append("FDR"[w["opt"]] + ("†" if reason == "death" else ""))
        extra = {
            "window_id": w["window_id"],
            "opt": w["opt"], "tau": tau, "reason": reason,
            "micro_steps": self.env._steps, "decisions": self._decisions,
            "cap_hits": self._cap_hits, "mode_seq": "".join(self.mode_seq),
            "R": w["R"], "W": w["W"], "bonus": w["bonus"],
            "kills_delta": int(self.env._ep_kills) - int(w["kills0"]),
            "worker_wage": w["worker_wage"],
            "worker_kills": w["worker_kills"],
            "worker_action14_requests":
                w["worker_action14_requests"],
            "worker_action14_native_successes":
                w["worker_action14_native_successes"],
            "worker_action14_gear_utility_delta":
                w["worker_action14_gear_utility_delta"],
            "no_effect_requests": w["no_effect_requests"],
            "worker_no_effect_requests": w["worker_no_effect_requests"],
            "executed_requests": w["executed_requests"],
            "gear_grace_opportunities": w["gear_grace_opportunities"],
            "gear_grace_decisions": w["gear_grace_decisions"],
            "beats": w["beats"], "overrides": w["overrides"],
            "fuse_trips": w["fuse_trips"],
            "drain_attempts": w["drain_attempts"],
            "drains": w["drains"],
            "voluntary_drinks": w["voluntary_drinks"],
            "recovery_actions": w["recovery_actions"],
            "last_recovery_action": w["last_recovery_action"],
            "last_requested_action": w["last_requested_action"],
            "last_executed_action": w["last_executed_action"],
            "fuse_requested_action": w["fuse_requested_action"],
            "dlvl0": w["dlvl0"], "dlvl_end": self.env._raw["dungeon_level"],
            "farm_scene_steps": self.farm_scene_steps,
            "dry": w["floor"] > 0,      # exhausted flag set when the window opened (dry-level revisit window)
            "base_done": w["done"] or w["trunc"],
            "base_trunc": w["trunc"] and not w["done"],
            "no_progress_micro_steps": int(self.layer_clock),
            "budget_boundary": budget_boundary,
            "timeout_without_progress": bool(
                budget_boundary
                and int(self.layer_clock) >= KILL_PATIENCE
            ),
        }
        if (not all(math.isfinite(float(extra[key]))
                    for key in ("R", "W", "bonus", "worker_wage"))
                or not math.isclose(
                    float(extra["R"]),
                    float(extra["W"]) + float(extra["bonus"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12)):
            raise RuntimeError(
                "option return split abnormal: "
                f"R={extra['R']}, W={extra['W']}, bonus={extra['bonus']}, "
                f"worker_wage={extra['worker_wage']}")
        if (extra["kills_delta"] < 0
                or not 0 <= extra["worker_kills"] <= extra["kills_delta"]):
            raise RuntimeError(
                "option kill split abnormal: "
                f"kills_delta={extra['kills_delta']}, "
                f"worker_kills={extra['worker_kills']}")
        if (
            not 0
            <= extra["worker_action14_native_successes"]
            <= extra["worker_action14_requests"]
            or extra["worker_action14_gear_utility_delta"] < 0
            or (
                extra["worker_action14_native_successes"] == 0
            ) != (
                extra["worker_action14_gear_utility_delta"] == 0
            )
        ):
            raise RuntimeError(
                "option action14 native receipt split abnormal: "
                f"requests={extra['worker_action14_requests']}, "
                "successes="
                f"{extra['worker_action14_native_successes']}, "
                "utility_delta="
                f"{extra['worker_action14_gear_utility_delta']}")
        if (
            not 0 <= extra["worker_no_effect_requests"]
            <= extra["no_effect_requests"] <= extra["beats"]
            or not 0 <= extra["executed_requests"] <= extra["beats"]
            or (
                extra["no_effect_requests"]
                + extra["executed_requests"]
                > extra["beats"]
            )
            or not 0 <= extra["gear_grace_opportunities"] <= 1
            or not 0 <= extra["gear_grace_decisions"] <= (
                GEAR_GRACE_MAX_DECISIONS
                * max(1, extra["gear_grace_opportunities"])
            )
        ):
            raise RuntimeError(
                "option action-effect/gear learning-window split abnormal: "
                f"no_effect={extra['no_effect_requests']},"
                f"worker_no_effect={extra['worker_no_effect_requests']},"
                f"executed={extra['executed_requests']},"
                f"beats={extra['beats']},"
                f"gear_grace={extra['gear_grace_decisions']}/"
                f"{extra['gear_grace_opportunities']}")
        base_info, done, trunc = w["last_info"], w["done"], w["trunc"]
        self._win = None
        return extra, base_info, done, trunc

    # ---- worker view (v23) ----
    def _worker_obs(self) -> np.ndarray:
        w = self._win
        tau = self.env._steps - w["t0"] if w is not None else self._last_tau
        # The actor and critic of V28/KING/root were trained on protocol-v3: 296 is
        # the no-new-kill clock and 297 the old exhausted. The current no-progress/scene-cap state
        # still drives the window but must not enter these two frozen input slots.
        legacy_exhausted = 1.0 if self._legacy_exhausted else 0.0
        exhausted_and_latch = legacy_exhausted
        if w is not None and (
            int(w.get("voluntary_drinks", 0)) > 0
            or int(w.get("drains", 0)) > 0
        ):
            # The sign domain is strictly separate from the undrunk 0/1; ``-v-1`` restores the old value losslessly.
            exhausted_and_latch = -(1.0 + legacy_exhausted)
        extra = np.asarray([
            min(1.0, tau / TAU_CAP),
            min(1.0, self._legacy_layer_clock / KILL_PATIENCE),
            exhausted_and_latch,
        ], dtype=np.float32)
        return np.concatenate([np.asarray(self._last_base_obs, dtype=np.float32), extra])

    def _worker_v5_window_appendix(self, w, tau, raw) -> np.ndarray:
        """R13 v5 appended block (12 dims): window-mode awareness + DIVE progress/value decomposition features.

        Only the dual-v5-window-mode-v1 view builds this block; the v4 prefix is assembled bit for bit by the caller,
        and this function consumes no RNG and writes no state. The actor/critic migrated by zero padding have zero
        initial weights for this block, so behaviour continues from the v4 starting point (the implementation premise of the D0 gate).
        """
        mode = (w or {}).get("mode")
        best_d = (w or {}).get("dive_best_d")
        stall = 0.0
        if w is not None and w.get("opt") == DIVE:
            stall = min(1.0, max(0.0, (
                float(tau) - float(w.get("dive_last_progress_tau", 0))
            ) / max(1, KILL_PATIENCE)))
        window_kills = 0.0
        worker_wage_frac = 0.0
        drinks_frac = 0.0
        if w is not None:
            window_kills = min(1.0, max(0.0, (
                float(self.env._ep_kills) - float(w.get("kills0", 0))
            ) / 50.0))
            worker_wage_frac = min(1.0, max(
                -1.0, float(w.get("worker_wage", 0.0)) / 50.0))
            drinks_frac = min(1.0, max(0.0, (
                float(w.get("drains", 0))
                + float(w.get("voluntary_drinks", 0))
            ) / 5.0))
        dlvl = float(raw.get("dungeon_level", 0))
        clvl = float(raw.get("char_level", 0))
        hp_frac = min(1.0, max(0.0, (
            float(raw.get("hp", 0)) / max(1.0, float(raw.get("max_hp", 0))))))
        appendix = np.asarray([
            1.0 if mode == "farm" else 0.0,       # [0] mode one-hot
            1.0 if mode == "dive" else 0.0,       # [1]
            1.0 if mode == "resupply" else 0.0,   # [2]
            (min(1.0, max(0.0, float(best_d) / 50.0))
             if best_d is not None else 0.0),     # [3] nearest main-line target distance
            stall,                                # [4] DIVE stall clock
            min(1.0, max(0.0, float(tau) / max(1, TAU_CAP))),  # [5] window age
            min(1.0, max(0.0, dlvl / 15.0)),      # [6] dungeon level
            min(1.0, max(-1.0, (clvl - dlvl) / 10.0)),  # [7] level margin
            window_kills,                         # [8] kills in the window
            hp_frac,                              # [9] HP fraction
            worker_wage_frac,                     # [10] worker wage in the window
            drinks_frac,                          # [11] drinks in the window
        ], dtype=np.float32)
        if (
            appendix.shape != (DUAL_WORKER_V5_APPENDIX_DIM,)
            or not np.isfinite(appendix).all()
        ):
            raise RuntimeError(
                "dual Worker v5 appended block shape/finiteness drift: "
                f"shape={appendix.shape},"
                f"finite={np.isfinite(appendix).all()}")
        return appendix

    def _worker_policy_observation(
            self,
            view: str,
            *,
            skip_dry_probability: float = 0.0,
    ) -> np.ndarray:
        """Build a declared Worker representation at the lossless raw edge.

        A complete protocol-v3 row cannot be reconstructed from the v4 vector
        after invisible/unreachable monsters and items have been filtered.
        Rebuild the 295-wide base from native ``raw``.  The A12 representation
        overlays only the two reversible v4 fields used by its adapter:
        packed belt economy (286) and the signed drink latch (297).  Thus the
        inherited V28/KING actor can recover an exact v3 row while the adapter
        still observes its resource/latch state.  The asymmetric representation
        instead keeps that exact 298-wide legacy row intact, then appends the
        current v4 base, wrapper dynamics, real fuse streak, exact Worker /
        Manager masks, and the immutable controller snapshot according to the
        constants above.
        """
        if view not in WORKER_OBSERVATION_VIEWS:
            raise ValueError(
                f"Worker policy observation view not registered: {view!r}")
        current = np.asarray(self._worker_obs(), dtype=np.float32)
        if current.shape != (298,):
            raise RuntimeError(
                f"Worker raw-v4 observation shape drift: {current.shape} != (298,)")
        if view == WORKER_OBSERVATION_VIEW_RAW_V4:
            return current

        raw = getattr(self.env, "_raw", None)
        if not isinstance(raw, dict):
            raise RuntimeError(
                "lossless Worker policy observation requires active native raw")
        legacy_base = self.env._legacy_policy_vectorize(self.env._raw)
        legacy_result = np.concatenate([
            legacy_base,
            np.asarray([
                current[295],
                current[296],
                1.0 if self._legacy_exhausted else 0.0,
            ], dtype=np.float32),
        ]).astype(np.float32, copy=False)
        if legacy_result.shape != (298,):
            raise RuntimeError(
                "Worker legacy observation shape drift: "
                f"{legacy_result.shape} != (298,)")
        if view in (
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
            WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE,
        ):
            try:
                p_skip = float(skip_dry_probability)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "dual Worker skip_dry_probability must be a finite number in [0,1]"
                ) from exc
            if not math.isfinite(p_skip) or not 0.0 <= p_skip <= 1.0:
                raise ValueError(
                    "dual Worker skip_dry_probability must be a finite number in [0,1]")
            current_base = np.asarray(
                self._last_base_obs, dtype=np.float32)
            if current_base.shape != (295,):
                raise RuntimeError(
                    "dual Worker current-v4 base shape drift: "
                    f"{current_base.shape} != (295,)")
            snapshot_vectorizer = getattr(
                self.env, "controller_snapshot_vector", None)
            if not callable(snapshot_vectorizer):
                raise RuntimeError(
                    "dual Worker v3 requires controller_snapshot_vector()")
            controller_snapshot = np.asarray(
                snapshot_vectorizer(), dtype=np.float32)
            if (
                controller_snapshot.shape
                != (CONTROLLER_SNAPSHOT_VECTOR_DIM,)
                or not np.isfinite(controller_snapshot).all()
            ):
                raise RuntimeError(
                    "dual Worker controller snapshot shape/finiteness drift: "
                    f"shape={controller_snapshot.shape},"
                    f"finite={np.isfinite(controller_snapshot).all()}")

            # action_masks() synchronizes the scene-budget identity.  Do this
            # before reading any context scalar so masks and context describe
            # one post-sync state rather than opposite sides of a scene reset.
            manager_mask = np.asarray(
                self.action_masks(), dtype=bool)
            worker_mask = np.asarray(
                self._worker_masks(), dtype=bool)
            if manager_mask.shape != (3,) or worker_mask.shape != (15,):
                raise RuntimeError(
                    "dual Worker mask shape drift: "
                    f"worker={worker_mask.shape},manager={manager_mask.shape}")

            w = self._win
            tau = (
                int(self.env._steps) - int(w["t0"])
                if w is not None else int(self._last_tau)
            )
            drank = bool(
                w is not None
                and (
                    int(w.get("voluntary_drinks", 0)) > 0
                    or int(w.get("drains", 0)) > 0
                )
            )
            dry_floor_remaining = 0.0
            if w is not None:
                dry_floor_remaining = min(
                    1.0,
                    max(
                        0.0,
                        (int(w.get("floor", 0)) - tau)
                        / max(1, REVISIT_FLOOR),
                    ),
                )

            calibration = getattr(self, "resource_calibration", None)
            farm_scene_denominator = (
                calibration.worker_farm_scene_denominator if calibration is not None
                else getattr(self, "farm_scene_cap", FARM_SCENE_CAP))
            context = np.asarray([
                min(1.0, max(0.0, self.layer_clock / KILL_PATIENCE)),
                1.0 if self.exhausted else 0.0,
                min(
                    1.0,
                    max(
                        0.0,
                        self.farm_scene_steps / max(1, farm_scene_denominator),
                    ),
                ),
                min(
                    1.0,
                    max(
                        0.0,
                        1.0 - self.env._steps / max(1, self.max_steps),
                    ),
                ),
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            int(self.env._ep_kills)
                            - int(self._layer_kills0)
                        ) / 50.0,
                    ),
                ),
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            int(self.env._steps)
                            - int(self._legacy_layer_steps0)
                        ) / 1500.0,
                    ),
                ),
                dry_floor_remaining,
                1.0 if drank else 0.0,
                p_skip,
            ], dtype=np.float32)
            if context.shape != (9,):
                raise RuntimeError(
                    f"dual Worker context shape drift: {context.shape} != (9,)")

            # ``_fuse`` counts repeats after the first matching request and
            # trips at 25.  A matching signature with counter zero is already
            # armed and one request closer to the trip than ``_fuse_sig=None``;
            # v1 encoded both as zero.  Publish (counter+1)/25 while armed so
            # the observation is Markov at that boundary.  Merely remembering
            # a previous action is not evidence of a current streak:
            # movement/entity/item changes invalidate it.
            fuse_streak = np.zeros(15, dtype=np.float32)
            fuse_signature = self._fuse_sig
            if (
                isinstance(fuse_signature, tuple)
                and fuse_signature
                and isinstance(fuse_signature[0], (int, np.integer))
                and not isinstance(fuse_signature[0], (bool, np.bool_))
            ):
                fuse_action = int(fuse_signature[0])
                if (
                    0 <= fuse_action < 15
                    and self._sig(fuse_action, raw) == fuse_signature
                ):
                    fuse_counter = int(self._fuse)
                    if not 0 <= fuse_counter < 25:
                        raise RuntimeError(
                            "dual Worker fuse counter must be within [0,24]: "
                            f"{fuse_counter}")
                    fuse_streak[fuse_action] = (
                        fuse_counter + 1) / 25.0

            result = np.concatenate([
                legacy_result,
                current_base,
                context,
                fuse_streak,
                worker_mask.astype(np.float32),
                manager_mask.astype(np.float32),
                controller_snapshot,
            ]).astype(np.float32, copy=False)
            expected_dim = DUAL_WORKER_OBSERVATION_DIM
            if view == WORKER_OBSERVATION_VIEW_DUAL_V5_WINDOW_MODE:
                # R13: the v4 prefix was fully assembled above (bit-identical to v4); this only appends.
                result = np.concatenate([
                    result,
                    self._worker_v5_window_appendix(w, tau, raw),
                ]).astype(np.float32, copy=False)
                expected_dim += DUAL_WORKER_V5_APPENDIX_DIM
            if (
                result.shape != (expected_dim,)
                or not np.isfinite(result).all()
            ):
                raise RuntimeError(
                    "dual Worker policy observation shape/finiteness drift: "
                    f"shape={result.shape},finite={np.isfinite(result).all()}")
            return result

        result = legacy_result.copy()
        if view == WORKER_OBSERVATION_VIEW_A12_OVERLAY:
            # All other base fields remain exact v3.  In particular features
            # 8/9 and the entity/map/item channels must not retain v4 filtering:
            # that information loss cannot be undone inside the policy.
            # Main ticks must use the exact v3 heal classification (which
            # historically included Healing scrolls); otherwise flooring in
            # the inherited actor cannot recover its training input.  The v4
            # free-slot count occupies the reversible sub-tick.  Actual
            # drinkability remains enforced by the native v4 action mask.
            free_slots = min(
                8, max(0, int(self.env._raw.get("belt_free_slots", 0))))
            result[286] = legacy_base[286] + free_slots / 128.0
            result[WORKER_DRINK_LATCH_FEATURE] = current[
                WORKER_DRINK_LATCH_FEATURE]
        else:
            result[WORKER_DRINK_LATCH_FEATURE] = (
                1.0 if self._legacy_exhausted else 0.0)
        return result

    def _worker_masks_and_distance(
        self,
    ) -> tuple[np.ndarray, int | None]:
        if getattr(self, "resource_protocol", "off") != "off":
            from .resource_protocol import progression_allowed
            self.env._resource_dive_authority = bool(
                self.dive_live_sovereignty and (self._win or {}).get("opt") == DIVE
                and progression_allowed(self.env._raw))
        m, nearest = self._controller_action_context()
        m = np.array(m, dtype=bool)
        raw = self.env._raw
        # R13 sovereignty transfer: inside live DIVE windows the worker gets a11 and the stepping authority (default off,
        # old rule unchanged bit for bit). FARM/RESUPPLY windows and the flag-off path keep the permanent mask;
        # the escape loophole of descending mid-fight to avoid the death cost only fails to apply inside DIVE windows,
        # where "descending is this window's task" anyway.
        _dive_live_window = (
            getattr(self, "dive_live_sovereignty", False)
            and (self._win or {}).get("opt") == DIVE
        )
        if not _dive_live_window:
            m[11] = False   # main-line progress belongs to the manager (DIVE authority)
            # 1..8 can also step directly onto adjacent trigger/story stances. Stripping the level-change bonus alone cannot
            # seal the authority: descending mid-fight can dodge a huge death cost, and the Worker would learn to run away.
            for action in DiabloGymEnv._protected_walk_actions(raw):
                m[action] = False
        # action10 handles the story allowlist first; as long as a target exists it must be masked, even if
        # nearby enemies still need FARM clearing, so the learning worker cannot bypass the manager's DIVE.
        m[10] = m[10] and not bool(
            self.env._raw.get("progression_targets"))
        if bool((self._win or {}).get("gear_grace_pending", False)):
            # Two legal actions are essential: masking to a14 alone would make
            # its probability exactly one and produce zero policy gradient.
            # a0 is an explicit, costly decline; a14 is the exact native
            # upgrade receipt the Worker must learn to choose.
            grace = np.zeros_like(m)
            grace[0] = True
            grace[14] = bool(m[14])
            return grace, nearest
        if self.drink_sovereignty:
            # Sovereignty only opens inside the currently visible safe envelope. Integer cross-multiplication keeps
            # the 0.5/0.75 boundary from flipping between collection and deployment through float rounding. A successful active drink
            # or reflex drink sets the visible latch and closes later worker-owned a12 in this window;
            # the emergency _drain() itself can still keep draining.
            hp = int(raw.get("hp", 0))
            max_hp = max(1, int(raw.get("max_hp", 0)))
            in_safe_envelope = (
                2 * hp >= max_hp
                and 4 * hp < 3 * max_hp
            )
            # A reflex drain is just as much a consumed potion as a voluntary
            # one.  The old latch counted only voluntary presses, so the
            # worker could immediately spend another bottle after a successful
            # brainstem rescue.  Keep emergency _drain() unrestricted (it
            # bypasses worker masks), but allow at most one worker-owned drink
            # before this option window returns control to the manager.
            already_drank = (
                int((self._win or {}).get("voluntary_drinks", 0)) > 0
                or int((self._win or {}).get("drains", 0)) > 0
            )
            m[12] = (
                m[12]
                and raw.get("belt_heals", 0) > 0
                and in_safe_envelope
                and not already_drank
            )
        else:
            m[12] = False   # drinking belongs to the brainstem (the old protocol before v32; the reflex still backs it up)
        return m, nearest

    def _worker_masks(self) -> np.ndarray:
        return self._worker_masks_and_distance()[0]

    def _validate_worker_action12_contract(
            self, worker, declared_view: str) -> None:
        """Keep a dual observation's embedded mask equal to deployment.

        Historical 298-wide policies do not embed their action mask.  The
        asymmetric lossless dual view does, so allowing a callback to apply a
        different action-12 contract after the observation is built would
        present the actor with a state that contradicts its actual legal
        actions.
        """
        expected_mode = (
            WORKER_ACTION12_ENVIRONMENT_MASK
            if self.drink_sovereignty
            else WORKER_ACTION12_PERMANENTLY_MASKED
        )
        declared_mode = getattr(
            worker, "diablogym_worker_action12_mode", None)
        if declared_mode is None:
            if declared_view == WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC:
                raise RuntimeError(
                    "dual Worker callback lacks the action12 deployment contract")
            return
        if declared_mode != expected_mode:
            raise RuntimeError(
                "Worker action12 contract inconsistent with OptionsEnv "
                "drink_sovereignty: "
                f"callback={declared_mode!r},environment={expected_mode!r}")

    def step(self, option: int):
        if self._win is not None:
            raise RuntimeError("the previous option window has not closed yet")
        self.env._ensure_active()
        self._win_begin(option)
        mode = self._win["mode"]
        worker = self._workers.get(int(option))
        ending = self._consume_fuse_recovery()
        retreat = getattr(self, "retreat_service", None)
        portal = getattr(self, "portal_service", None)
        sweep = getattr(self, "sweep_service", None)
        # R18-H identify-v1 never owns a window on its own: the Cain leg runs
        # strictly INSIDE an already-open town trip, before anything is sold.
        identify = getattr(self, "identify_service", None)
        # R18-M (2026-09-07) review round — the invariant that makes "no rung"
        # safe, stated where a pass-2 leg will copy the pattern: the leg is only
        # ever active while self.resource_service.active, and _win_term
        # (options_env.py:1516-1519) returns None for the whole time that flag
        # holds, so the ONLY window close reachable while identify.active is the
        # episode-terminating `done or trunc` path at :1418-1419. An interrupted
        # leg therefore cannot outlive the episode that stranded it. A future leg
        # that can be interrupted by a NON-terminating close does not inherit
        # this and must take a rung (guarded on its own .active) instead, or the
        # resume clause below (`service is identify`) will never fire for it and
        # its trip stays in _served_trips (resource_identify.py:367) forever.
        # R18-M (2026-09-07) conflict resolution H1/H2 (options_env.py, the
        # RESUPPLY owner chain): both patches rewrote this header. BOTH survive.
        # H1 (sweep-v1) adds a rung to the ownership chain; H2 (identify-v1) adds
        # no rung at all -- it only binds the local, because the Cain leg runs
        # inside a town trip the town service already owns. H1's rung ORDER is
        # kept byte-for-byte (sweep first): sweep is main-L1 only while
        # retreat/portal are main-L2+ only, so no two rungs can ever contend for
        # the same window and the order carries no behaviour.
        # R18-F portal-v1 extends the two-rung chain to three; R18-H sweep-v1 to
        # four. The town service is still the default owner of a RESUPPLY window.
        # The sweep is main-L1 only and the retreat/portal are main-L2+ only, so
        # the new rung can never contend with them for the same window.
        service = (sweep if sweep is not None and sweep.active
                   else portal if portal is not None and portal.active
                   else retreat if retreat is not None and retreat.active
                   else self.resource_service)
        if (getattr(self, "resource_protocol", "off") != "off"
                and int(option) == RESUPPLY and service.active):
            while ending is None or ending.reason is None:
                if (portal is not None and service is portal and not portal.active
                        and self.resource_service.active):
                    # R18-F: a witch leg run inside a town trip has completed; the
                    # town service resumes its own (return) phase.
                    service = self.resource_service
                if (identify is not None and service is identify and not identify.active
                        and self.resource_service.active):
                    # R18-H: the Cain leg run inside a town trip has completed;
                    # the town service resumes its own itinerary, and only now
                    # does anything get sold -- at its identified value.
                    service = self.resource_service
                # R18-M2 (2026-09-07) conflict resolution K2b: a third leg
                # hands back at the same seam.  Separate `if`s in trip order
                # (identify, then weapon); each is a no-op unless it owns the
                # beat, so the order carries no behaviour.
                weapon = getattr(self, "weapon_upgrade_service", None)
                if (weapon is not None and service is weapon and not weapon.active
                        and self.resource_service.active):
                    # R18-K: same hand-back for the surplus weapon leg, so its own
                    # final beat is still charged to the weapon service's phases.
                    service = self.resource_service
                if (getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03"
                        and not getattr(service, "settle_exempt", False)):
                    # R17.1 readiness rule 3 (T0′ crash, seed 2133003): a deadline-truncated
                    # service walk can leave the player mid-tile; the service's next
                    # decision — and, at completion, the worker's first decision —
                    # must start from a decision-idle player (frozen worker law).
                    # Settle through the audited script-owned wait command
                    # (bounded).  Inert under veto-v1, so R18–R23 receipts stand.
                    settle = 0
                    while (not self.env._decision_idle(self.env._raw)
                           and not self.env._raw.get("dead") and settle < 40
                           and (ending is None or ending.reason is None)):
                        self._resource_command = ("wait",)
                        try:
                            ending = self._win_beat(0)
                        finally:
                            self._resource_command = None
                        service.record_steps(
                            self.env._steps, service.phase)
                        settle += 1
                    if ending is not None and ending.reason is not None:
                        break
                if (portal is not None and not portal.active and not portal.awaiting_return
                        and service is self.resource_service and service.active
                        and getattr(service, "phase", None) == "return"
                        and int(self.env._raw["dungeon_level"]) == 0
                        and portal.trigger_reason(self.env._raw, int(self.env._steps)) == "buy_scroll"):
                    # R18-F portal-v1: the witch leg runs INSIDE the town trip, after the
                    # ordinary itinerary (heal / armor / potions / sell) and before the walk
                    # back, so the scroll is bought last with the belt already full. The
                    # town service never hands back while in town, so this is the only
                    # moment the purchase can happen. Decided BEFORE the town service is
                    # asked for a command (a discarded command would leave its pending
                    # request without a receipt). The portal script owns the next
                    # commands; the town service resumes its return phase afterwards.
                    portal.start(self.env._raw, int(self.env._steps), bridge, "buy_scroll")
                    service = portal
                identify_trip_budget = None
                if identify is not None and service is self.resource_service and service.active:
                    # R18-H review round (2026-09-07): the leg's beats are charged
                    # to the town trip's own clock, so hand it that clock.
                    identify_trip_budget = (int(getattr(service, "start_steps", 0))
                                            + int(service.service_microstep_cap)
                                            - int(self.env._steps))
                if (identify is not None and not identify.active
                        and service is self.resource_service and service.active
                        and int(self.env._raw["dungeon_level"]) == 0
                        and not self.env._raw.get("is_set_level")
                        and getattr(service, "phase", None) == "outbound"
                        and identify.trigger_reason(
                            self.env._raw, int(self.env._steps),
                            int(getattr(service, "trip_count", 0)),
                            trip_budget_left=identify_trip_budget) == "identify"):
                    # R18-H identify-v1 (design decision, 2026-09-07): the Cain
                    # leg runs at the EARLIEST town phase of every town trip -- the
                    # town service reaches town in "outbound" and turns it into
                    # "sell_idle" INSIDE its own next command, so this beat is the
                    # only moment before sell_idle_smith_equipment can sell anything.
                    # R18-H review round: the gate is exactly "outbound" -- admitting
                    # "sell_idle" let a leg deferred here (no funds) fire later, after
                    # the first sale had raised the wallet, i.e. AFTER something was
                    # sold. A deferral is now recorded instead (identify.deferrals).
                    # Decided BEFORE the town service is asked for a command (a
                    # discarded command would leave its pending request without a
                    # receipt). The identify script owns the next commands; the town
                    # service resumes its untouched itinerary afterwards.
                    identify.start(self.env._raw, int(self.env._steps), "identify",
                                   int(getattr(service, "trip_count", 0)),
                                   trip_budget_left=identify_trip_budget)
                    service = identify
                # R18-M2 (2026-09-07) conflict resolution K2b (leg order inside a
                # town trip): identify runs at phase "outbound" -- the only beat
                # before sell_idle_smith_equipment can sell anything -- and the
                # weapon leg at phase "return", after the readiness basket and the
                # potion target are already paid and before the walk back.  The two
                # phase gates plus `service is self.resource_service` already make
                # them mutually exclusive within one beat (identify owns the beats
                # it runs, so the town service cannot advance its phase while
                # identify.active).  The `not identify.active` term above states
                # that law instead of deriving it: it is inert on every reachable
                # path and fail-closed if a later change breaks the derivation.
                if (weapon is not None and not weapon.active
                        and (identify is None or not identify.active)
                        and (portal is None or (not portal.active and not portal.awaiting_return))
                        and service is self.resource_service and service.active
                        and getattr(service, "phase", None) == "return"
                        and int(self.env._raw["dungeon_level"]) == 0):
                    # R18-K (2026-09-07) smith-v1: the surplus weapon leg runs INSIDE the
                    # town trip, after the readiness basket and the potion target are
                    # already paid and before the walk back -- the same seam R18-F uses
                    # for the witch, and for the same reason (the town service never
                    # hands back while in town).  Decided BEFORE the town service is
                    # asked for a command, so no generated command loses its receipt.
                    # The portal keeps priority; one leg per town trip.
                    trip = int(getattr(service, "trip_count", 0) or 0)
                    stock = list(getattr(service, "_smith_stock", []) or [])
                    if weapon.protocol == "dry-v1":
                        # Observation only: record the counterfactual and issue
                        # nothing, so this arm stays bit-identical to control.
                        weapon.observe_seam(self.env._raw, stock, trip, int(self.env._steps))
                    else:
                        trigger = weapon.trip_trigger_reason(self.env._raw, stock, trip)
                        if trigger is not None:
                            weapon.start(self.env._raw, int(self.env._steps), trigger, trip)
                            service = weapon
                synthesized = False
                if (portal is not None and portal.awaiting_return
                        and service is self.resource_service and service.active
                        and getattr(service, "phase", None) == "return"
                        and int(self.env._raw["dungeon_level"]) == 0):
                    # R18-F portal-v1. ResourceService's return phase walks to the
                    # cathedral stairs and completes only on L1, which would throw away
                    # the depth the scroll was bought to keep. Close the town service
                    # here (its itinerary is finished by construction at this phase),
                    # decided BEFORE asking it for a command so no pending request is
                    # left without a receipt; the next window's manager law hands the
                    # return leg to the portal script.
                    service.active = False
                    service.phase = "done"
                    service.reason = "portal_return_pending"
                    bridge.configure_town_service(False)
                    command = ("complete",)
                    synthesized = True
                else:
                    command = service.command(self.env, bridge)
                self.env._resource_service_deadline = (
                    service.start_steps + service.service_microstep_cap)
                if service is self.resource_service and getattr(self, "resource_service_policy", "legacy-v1") in ("sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"):
                    self.env._resource_service_deadline = self.resource_service.command_microstep_deadline
                self._resource_command = command
                try:
                    ending = self._win_beat(0)
                finally:
                    self._resource_command = None
                receipt = self._win.get("last_info", {}).get("resource_action_audit", {})
                if not synthesized:
                    service.receipt(command, receipt)
                if (portal is not None and service is portal and command[0] == "buy"
                        and receipt.get("accepted") and self.resource_service.active):
                    # R18-F: the scroll was bought inside the town trip; book it in the
                    # town service ledger so the trip cash reconciliation stays exact.
                    self.resource_service.purchases += 1
                    self.resource_service.gold_spent += int(receipt.get("price", 0) or 0)
                if (identify is not None and service is identify and command[0] == "identify"
                        and receipt.get("accepted") and self.resource_service.active):
                    # R18-H: the fee was paid inside the town trip. Book it exactly the
                    # way the scroll was booked so _seal_trip's cash residual stays 0
                    # and the economy audit raises no "unexplained cash movement".
                    # It is a paid service, not a purchase: only gold_spent moves.
                    self.resource_service.gold_spent += int(receipt.get("price", 0) or 0)
                if (identify is not None and service is self.resource_service
                        and command[0] == "sell" and receipt.get("accepted")):
                    # Attribute the Smith's payment to the identify leg when the item
                    # is one Cain opened for us (the identity never changes).
                    identify.note_sale(tuple(command[3:7]), int(receipt.get("received", 0) or 0))
                if weapon is not None:
                    # R18-K: identical booking for the weapon bought inside the
                    # trip.  R18-K review round: the body moved into
                    # resource_weapon_upgrade.book_weapon_purchase so the ledger
                    # test drives the shipped code instead of a copy of it; the
                    # guard terms are unchanged and, with the flag off, `weapon`
                    # is None and this whole branch is the old short circuit.
                    from .resource_weapon_upgrade import book_weapon_purchase
                    book_weapon_purchase(self.resource_service, weapon, service,
                                         command, receipt)
                service.record_steps(self.env._steps, service.phase)
            extra, base_info, done, trunc = self._win_end(ending.reason)
            self._finish_loot_episode(done, trunc, base_info)
            extra["resource"] = self.resource_service.telemetry()
            if retreat is not None:
                extra["retreat"] = retreat.telemetry()
            if portal is not None:
                extra["portal"] = portal.telemetry()
            if sweep is not None:
                sweep.observe(self.env._raw)
                extra["sweep"] = sweep.telemetry()
            if identify is not None:
                extra["identify"] = identify.telemetry()
            weapon = getattr(self, "weapon_upgrade_service", None)
            if weapon is not None:
                # R18-K: also nested in the per-row resource dict the probe reads.
                extra["weapon_upgrade"] = weapon.telemetry()
                extra["resource"]["weapon_upgrade"] = weapon.telemetry()
            info = dict(base_info)
            info["option_extra"] = extra
            return self._mgr_obs(self._last_base_obs), extra["R"], done, trunc, info
        if worker is not None:
            if ending is None or ending.reason is None:
                ending = self._drain()  # the worker's first observation must be reflex-free
            while ending is None:
                declared_view = getattr(
                    worker,
                    "diablogym_worker_observation_view",
                    self.worker_observation_view,
                )
                self._validate_worker_action12_contract(
                    worker, declared_view)
                a = worker(
                    self._worker_policy_observation(declared_view),
                    self._worker_masks(),
                )
                outcome = self._win_step_worker(a)
                if outcome.reason is not None:
                    ending = BeatOutcome(
                        reason=outcome.reason,
                        requested_action=outcome.requested_action,
                        executed_action=outcome.executed_action,
                        fuse_tripped=outcome.fuse_tripped,
                        action14_audit=outcome.action14_audit,
                        action_effect_audit=outcome.action_effect_audit,
                    )
        else:
            while ending is None or ending.reason is None:
                raw = self.env._raw
                action_mask, nearest = self._controller_action_context()
                action_mask = np.asarray(action_mask, dtype=bool)
                grace_decision = bool(
                    self._win.get("gear_grace_pending", False))
                if grace_decision:
                    # Scripted Options evaluation consumes the same single
                    # opportunity, without charging it to Worker counters.
                    self._win["gear_grace_pending"] = False
                    self._win["gear_grace_consumed"] = False
                    self._win["gear_grace_decisions"] += 1
                    a = 14 if bool(action_mask[14]) else 0
                    if a == 0:
                        self._win["gear_grace_consumed"] = True
                else:
                    a = 12 if _reflex(raw) else dispatch(
                        mode,
                        raw,
                        bool(action_mask[14]),
                        action_mask=action_mask,
                        nearest_engageable_distance=nearest,
                    )
                ending = self._win_beat(a)
                if grace_decision and ending.fuse_tripped:
                    self._win["gear_grace_pending"] = True
                    self._win["gear_grace_consumed"] = False
                    self._win["gear_grace_decisions"] -= 1
        extra, base_info, done, trunc = self._win_end(ending.reason)
        self._finish_loot_episode(done, trunc, base_info)
        info = dict(base_info)
        if getattr(self, "resource_protocol", "off") != "off":
            extra["resource"] = self.resource_service.telemetry()
            retreat = getattr(self, "retreat_service", None)
            if retreat is not None:
                extra["retreat"] = retreat.telemetry()
            portal = getattr(self, "portal_service", None)
            if portal is not None:
                extra["portal"] = portal.telemetry()
            sweep = getattr(self, "sweep_service", None)
            if sweep is not None:
                sweep.observe(self.env._raw)
                extra["sweep"] = sweep.telemetry()
            identify = getattr(self, "identify_service", None)
            if identify is not None:
                extra["identify"] = identify.telemetry()
            weapon = getattr(self, "weapon_upgrade_service", None)
            if weapon is not None:
                # R18-K: also nested in the per-row resource dict the probe reads.
                extra["weapon_upgrade"] = weapon.telemetry()
                extra["resource"]["weapon_upgrade"] = weapon.telemetry()
        info["option_extra"] = extra
        return self._mgr_obs(self._last_base_obs), extra["R"], done, trunc, info

    def _finish_loot_episode(self, done, trunc, base_info):
        if (getattr(self, "resource_service_policy", "legacy-v1") != "sustain-loot-v1"
                or not (done or trunc)):
            return
        raw = self.env._raw
        reason = (getattr(self.env, "_resource_terminal_reason", None)
                  or base_info.get("terminal_reason")
                  or ("death" if raw.get("dead") or int(raw.get("hp", 0)) <= 0
                      else "episode_terminated" if done else "episode_truncated"))
        self.resource_service.finish_episode(raw, int(self.env._steps), str(reason))

    def _mgr_obs(self, base_obs) -> np.ndarray:
        view = getattr(
            self, "manager_observation_view",
            MANAGER_OBSERVATION_VIEW_RAW_V4)
        if view not in MANAGER_OBSERVATION_VIEWS:
            raise RuntimeError(
                f"manager observation view not registered: {view!r}")
        if view == MANAGER_OBSERVATION_VIEW_LEGACY_V3:
            # M29 was trained on the same protocol-v3 base as V28.  Restoring
            # only belt feature 286 is insufficient: v4 also filtered monster
            # count/nearest/slots, the 121-cell monster map, and item slots.
            # Rebuild before information is lost at the vector boundary.
            rebuild_legacy = getattr(
                self.env, "_legacy_policy_vectorize", None)
            raw = getattr(self.env, "_raw", None)
            if not callable(rebuild_legacy) or raw is None:
                raise RuntimeError(
                    "legacy-v3 manager view requires a lossless native raw "
                    "record and legacy vectorizer")
            manager_base = rebuild_legacy(raw)
        else:
            manager_base = np.asarray(base_obs, dtype=np.float32).copy()
        if manager_base.shape != (295,):
            raise ValueError(
                f"manager base observation must be (295,), got {manager_base.shape}")
        if view == MANAGER_OBSERVATION_VIEW_LEGACY_V3:
            # The vectorizer already used native legacy_belt_heals.  Keep this
            # assertion local so no future refactor can reintroduce a packed
            # v4 sub-tick into M29's feature 286.
            expected_belt = min(
                8, max(0, int(self.env._raw["legacy_belt_heals"]))) / 8.0
            if not np.isclose(
                    manager_base[286], expected_belt, rtol=0.0, atol=1e-7):
                raise RuntimeError(
                    "legacy-v3 manager belt feature was not reconstructed "
                    "from native legacy_belt_heals")
        one_hot = [0.0, 0.0, 0.0]
        if self._last_opt >= 0:
            one_hot[self._last_opt] = 1.0
        if view == MANAGER_OBSERVATION_VIEW_LEGACY_V3:
            clock = self._legacy_layer_clock
            layer_steps0 = self._legacy_layer_steps0
        else:
            clock = self.layer_clock
            layer_steps0 = self._layer_steps0
        extra = np.asarray([
            max(0.0, 1.0 - self.env._steps / max(1, self.max_steps)), # time remaining
            min(1.0, clock / KILL_PATIENCE),
            min(1.0, (self.env._ep_kills - self._layer_kills0) / 50.0),
            min(1.0, (self.env._steps - layer_steps0) / 1500.0),
            *one_hot,
            min(1.0, self._last_tau / TAU_CAP),
        ], dtype=np.float32)
        return np.concatenate([manager_base, extra])

    def close(self):
        self.env.close()


class StagnationClockWrapper(gym.Wrapper):
    """For demon arm F only: a flat 295+1=296-dim wrapper.

    Clock progress uses the same factual signals as OptionsEnv.  Counting only
    kills taught the flat oracle to abandon a high-HP monster after 140 beats
    even while every attack was lowering its conserved HP floor.
    """

    def __init__(self, env: DiabloGymEnv):
        super().__init__(env)
        base = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base + 1,), dtype=np.float32)
        self._clock = 0
        self._kills = 0
        self._scene = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._clock, self._kills = 0, 0
        self._scene = _scene_identity(self.env._raw)
        return self._obs(obs), info

    def action_masks(self):
        return self.env.action_masks()

    def step(self, action):
        steps_b = self.env._steps
        exploration_b = int(getattr(
            self.env, "exploration_progress", 0))
        gear_utility_b = gear_combat_utility_value(
            self.env._raw, "flat_clock_before")
        belt_free_b = DiabloGymEnv._belt_free_slots(self.env._raw)
        gold_b = int(self.env._raw.get("gold", 0))
        combat_floor_b = dict(getattr(
            self.env, "_combat_hp_floor", {}))
        obs, r, done, trunc, info = self.env.step(action)
        raw = self.env._raw
        combat_floor = getattr(self.env, "_combat_hp_floor", {})
        new_damage_floor = any(
            key in combat_floor
            and int(combat_floor[key][0]) < int(before[0])
            for key, before in combat_floor_b.items()
        )
        positive_progress = (
            int(getattr(self.env, "exploration_progress", 0))
            > exploration_b
            or new_damage_floor
            or gear_combat_utility_value(
                raw, "flat_clock_after") > gear_utility_b
            or DiabloGymEnv._belt_free_slots(raw) < belt_free_b
            or int(raw.get("gold", 0)) > gold_b
        )
        if _scene_identity(raw) != self._scene or self.env._ep_kills > self._kills:
            self._clock = 0
            self._scene = _scene_identity(raw)
            self._kills = self.env._ep_kills
        elif positive_progress:
            self._clock = 0
        else:
            self._clock += self.env._steps - steps_b
        return self._obs(obs), r, done, trunc, info

    def _obs(self, base_obs):
        return np.concatenate([np.asarray(base_obs, dtype=np.float32),
                               np.asarray([min(1.0, self._clock / KILL_PATIENCE)],
                                          dtype=np.float32)])
