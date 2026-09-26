"""DiabloGymEnv: Gymnasium wrapper (v0: structured vector observation + discrete actions).

Observation vector (float32, length 12 + K*4 + 2*(2R+1)² + 9; 295 when R=5):
  [hp/maxhp, mana/maxmana, xp(log1p/10), gold/1000, char_level/50,
   dungeon_level/16, player_x/112, player_y/112,
   visible reachable monster count/50, distance to the nearest visible reachable monster/30 (no monster = 1),
   direction dx/56, dy/56 to the next required main-quest target (the down stairs on ordinary levels; 0,0 if none)]
  + (dx/20, dy/20, hp/max_hp, 1 presence flag) of the K nearest visible reachable monsters
  + 11×11 local map, two channels (walkability, visible reachable monster occupancy); run4 lesson: without spatial awareness
    even a perfect reward is "a blind man with a perfect ledger" (locking on through walls, shaping through walls, never finding the door)
  + [belt heal potions/8 + free slots/128,
     nearest visible reachable floor heal potion dx/20, dy/20 (clipped to ±1), presence flag]
    (v4 encodes two 0..8 integer domains unambiguously inside the original belt scalar: heal potions keep the 1/8
    main scale and free slots only take the 1/16 sub-scale below it, max perturbation 0.0625. This makes every resource
    precondition of drinking/potion pickup observable without changing the 295-dim shape)
  + [armor class/50 (clipped to 1), nearest visible reachable whole-set combat upgrade dx/20,
     dy/20 (clipped to ±1), presence flag]
    (the presence flag comes from native PlanGearUpgrade: candidates are identified first, then the occupied-slot replacement,
    dual-ring choice and one-/two-hand switch are simulated; it is 1 only if the whole set's conservative combat power strictly increases)
  + [min(2, character level/max(1, dungeon level))/2]: the v19 strength gauge

Actions (Discrete(15)):
  0      explicit wait: cancels leftover pathing/chase/destAction; an attack animation stops immediately, a committed
         single-tile move finishes naturally within this step's billed settle ticks, but the previous
         action's long path or repeated attack does not continue (v4)
  1-8    walk one tile in one of eight directions (pathing)
  9      engage macro: lock onto the nearest visible reachable monster and keep chasing it until it dies/we die/the level changes/
         timeout (≤10 ticks)
         (v2 lesson: a single-tick attack is interrupted by the next move action, so the policy never learns to "keep attacking")
  10     explore macro: if a story target required for completion exists, fail closed by waiting/handing back to the manager,
         never overstepping action11 to push forward; otherwise walk to the nearest "walkable and not yet visited"
         frontier point within the 25×25 view; on spotting prey
         (nearest monster ≤6 tiles) hand control back immediately; opens ordinary closed doors to new areas on the way,
         and may deal with a blocking barrel when there is no frontier, but never steps on any up/down trigger
         (run5 lesson: when no monster is reachable from the spawn area, a reactive policy never "tries another room")
  11     main-quest progress macro (v11+): first completes the strict allowlist of staff stand/staff/L15 quest entrance/Vile books and
         pentagram/L16 mechanism, then walks to the down or quest-return trigger when nothing is pending.
         Unlike the explore macro, spotting prey does **not** interrupt it: this is the policy's deliberate withdraw/level-change key
         (the escape hatch from a dead end + the next-chapter button after clearing a level); control returns naturally after 12 ticks.
         (v10 lesson: a dead end is a dead 0, and more time does not help; it needs a door)
  12     drink key (v12): drinks one healing potion if the belt has one (same path as the engine's gamepad shortcut),
         otherwise an empty tick. v12 deliberately kept the belt potion count out of the observation (to keep all 286-dim generations
         re-evaluable), and 99.5% of presses landed on an empty belt (lesson 11, "bottle blindness");
         since v13 the belt potion count and the nearest floor potion direction are in the observation.
  13     potion pickup macro (v13): walks 4-direction safe steps one by one along the radius-12 fixed snapshot bound to the observation,
         opening an ordinary door from the adjacent tile first; native pickup is committed only when standing exactly on the target tile, and
         native code re-checks the item identity and belt capacity. Waits if there is no target/no free slot or the safe path is incomplete.
  14     gear pickup macro (v14): walks the same step-by-step safe path toward the strict whole-set upgrade approved by
         PlanGearUpgrade; once exactly in place, native code re-validates the item identity and swap plan, copies the body slots
         and replaces atomically (including occupied slots, weapon/shield, dual rings and one-/two-hand switching). If after CalcPlrInv
         the conservative whole-set combat power has not strictly increased, it rolls back completely; it never uses AutoEquip's backpack fallback.

Action masks and path safety:
  action_masks only constrains 9/12/13/14 dynamically: 9 needs an encoded monster that is visible in the radius-12 snapshot and locally
  engageable; 12 needs a belt heal potion and actual missing HP; 13 needs a free belt slot and a valid
  heal potion target; 14 needs a valid whole-set upgrade target. The other keys stay legal. Masks only remove deterministic
  empty presses and do not promise that a macro avoids later dynamic blocking, hits taken or the time limit.
  The dual Worker executes the same snapshot installed when the observation was generated; the old 295/298 views only take a read-only snapshot
  at the mask/step boundary. The controller plans 4-direction adjacent steps only from that radius-12 view; hazards,
  explosive softwalls and protected story tiles are never crossed; item macros also require the path to reach the
  exact target tile completely, and must not use a native short-range command as a second pathing shortcut across a dangerous tile.

Decision boundary (v4): except at death/victory/total-step boundaries, when step returns the player must be in
PM_STAND, with future==tile and walkpath/destAction empty. The engine beats used to settle walking, hits taken and other
hidden animations all count toward micro steps, the reward difference and Options τ;
the 295/298/303-dim observations therefore need no invisible execution state, and no state aliasing of the kind "same tile,
different pending command" can occur. If max_steps runs out exactly during an animation, it cannot settle
beyond the budget; that boundary fails closed as terminal and forbids value bootstrapping. Only an idle
max_steps boundary is a standard truncation.

action9/10 are recoverable controllers with memory: failed-target rotation, the sticky frontier and the visited
set are internal scheduling state of the macro, not another action the policy can choose; they must not change the
action mask, and once the failure table is exhausted a new round must start instead of deleting legal targets permanently. Formally,
if the 15 keys were treated as fully atomic flat-MDP actions, these scheduling tables (together with unobserved parts of the game)
would still be partially observable state; a strict expansion would need target directions and a new exploration-map channel, invalidating the old 295-dim
weights. The current interface explicitly assigns them to the option controller, and puts only the wrapper hand-back clock/budget counters
that change the manager mask or the FARM window-closing time directly into the 298/303-dim policy observation;
"is the next tile visited for the first time" remains, through the explore controller's visited map, part of this explicit partially observable
abstraction, and is not claimed to be fully reconstructible from the 295 dims.

Reward (v4, state-potential conserving):
  +Φ(q_before)-Φ(q_after), Φ(q)=0.75q-0.125q²
                                              q is the lowest HP/maxHP already paid for in this
                                              monster's lifetime; paid once per new low, no repeat
                                              for healing and re-hitting. Damage shaping from full
                                              to 0 is always 0.625, regardless of cuts, settle
                                              ticks or attack rhythm
  +1.0 * Δmonster_kill_total                  native monotone kill ledger; monsters spawned and killed within
                                              a macro, or a scene change in the kill tick, are never missed
  +0.01 * ΔXP                                  the real goal (leveling)
  +8.0  * Δdungeon level                       flat mode; depth-ladder mode settles 8N for each
                                              N→N+1 crossed
  +min(1, max(0, Δgear_utility)/4096)          weapon damage, to-hit, AC, resistances,
                                              blocking, affixes and durability share the native whole-set
                                              uint32 ledger; only increases are rewarded, a negative Δ is not penalized
  +0.005 * player-only approach difference to a fixed live target   may be negative when walking away
  -0.002  same-scene request for the action0 wait
  -0.002  any non-zero request not executed natively in the same scene   attacks/drinks/pickups that really executed
                                              but stayed in place do not pay this fixed penalty
  -2.0 death (-8×current level in death-ladder mode)  +10.0 victory
  Historical lessons: v0's HP-loss penalty -> collapse into facing the wall; v1's "score even with a monster in your face" -> standing still and fishing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
import os
import pathlib
import shutil
import tempfile
from collections import deque
from contextlib import contextmanager

import gymnasium as gym
import numpy as np

from . import bridge, nav
from .controller_wire import *  # noqa: F403 - single schema source, re-exported

# Eight directions (tile coordinates of the isometric dungeon)
_DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
_K_MONSTERS = 8
_MAP_RADIUS = 5  # 11×11 local map

# Unit price of the level-change bonus (the Δdungeon-level term of _reward; a named constant since v23, because worker wage stripping
# must invert it on the wrapper side with the same formula, and the number may exist in only one place)
DESCEND_UNIT = 8.0
GEAR_COMBAT_UTILITY_REWARD_SCALE = 4096.0
GEAR_COMBAT_UTILITY_REWARD_CAP = 1.0
STALL_ACTION_REWARD = -0.002
# R18-B5 (2026-09-07) review correction: the scope vocabulary of the a10 whole-map monster hunt. This tuple used to
# live only as a literal in env.py's constructor, while the training/evaluation side each kept a copy; if the engine added a
# scope, the training world could not express a law the tested world can run, and no test would raise an alarm.
# From this version the deployment side (env.py/worker_env.py) recognizes only this copy; the train/ side does not import diablogym
# (eval_contract is a pure-stdlib contract module), and a new test compares the literals site by site.
HUNT_SCOPES = ("all", "l1-only")


@dataclass(frozen=True)
class TerminalDeathRewardSpec:
    """Immutable source of truth for the native terminal-death component."""

    flat_cost: float
    ladder_cost_per_depth: float


TERMINAL_DEATH_REWARD_SPEC = TerminalDeathRewardSpec(
    flat_cost=2.0,
    ladder_cost_per_depth=8.0,
)


@dataclass(frozen=True)
class RewardEconomy:
    """R10 economy act: the single source of truth for all wage constants (v1 = history unchanged bit for bit).

    v2 (R10 depth economy, approved 2026-08-27; all bookings go to the manager):
      - A descend bonus: descend_unit raised (still the v17 progressive 8×N structure, with a new unit price);
      - B depth multiplier: farm income (xp + kills) × (1 + kill_depth_beta × (d-1));
      - C revised death penalty: flat + c0×gamma^(d-1), decreasing with depth (reversing v1's
        ladder of increasing fear);
      - D anti-idling: after staying on the same level beyond idle_threshold_steps, farm income × idle_factor
        (reset on reaching a new deepest level of the episode);
      - E1 gear repricing: scale down / cap up, one real upgrade ≈ several monsters.
    """

    name: str
    descend_unit: float
    kill_depth_beta: float
    death_flat: float
    death_c0: float
    death_gamma: float
    death_uses_ladder_spec: bool
    gear_scale: float
    gear_cap: float
    idle_threshold_steps: int
    idle_kill_factor: float
    # R11 kill lock: when >0 the descend bonus is booked into an "unvested" escrow; it vests once this many kills
    # are made on that level; dying before vesting forfeits all of it (mathematically equal to vested escrow at gamma=1.0).
    # 0 = off (v1/v2 semantics unchanged).
    descend_vest_kills: int
    # R16 amendment (C8, 2026-09-01) redefines anti-idling with two fields that default to off:
    #   idle_counts_micro_beats: the clause-D idle clock counts engine micro beats (increments of self._steps)
    #     instead of _reward calls (= decisions/macros);
    #   idle_reset_on_kill: a native kill in this decision resets the idle clock (this decision's farm multiplier is settled
    #     with the clock before the reset, then it is reset; reaching a new deepest level still resets it).
    # v1/v2/v3/v3b all take the default False -> old behaviour unchanged bit for bit.
    idle_counts_micro_beats: bool = False
    idle_reset_on_kill: bool = False


REWARD_ECONOMY_V1 = RewardEconomy(
    name="v1",
    descend_unit=DESCEND_UNIT,
    kill_depth_beta=0.0,
    death_flat=TERMINAL_DEATH_REWARD_SPEC.flat_cost,
    death_c0=0.0,
    death_gamma=1.0,
    death_uses_ladder_spec=True,
    gear_scale=GEAR_COMBAT_UTILITY_REWARD_SCALE,
    gear_cap=GEAR_COMBAT_UTILITY_REWARD_CAP,
    idle_threshold_steps=0,
    idle_kill_factor=1.0,
    descend_vest_kills=0,
)

REWARD_ECONOMY_V2 = RewardEconomy(
    name="v2",
    # rev2 (G0 calibration 2026-08-27): the rev1 probe DIVE-FARM=-29.3 did not flip sign;
    # the B multiplier fattened the conservatives indiscriminately. Descend bonus 24->48, beta 0.5->0.25,
    # anti-idling 900->300 steps. The frozen pre-registration holds the final values.
    descend_unit=48.0,
    kill_depth_beta=0.25,
    death_flat=2.0,
    death_c0=24.0,
    death_gamma=0.7,
    death_uses_ladder_spec=False,
    gear_scale=1024.0,
    gear_cap=12.0,
    idle_threshold_steps=300,
    idle_kill_factor=0.5,
    descend_vest_kills=0,
)

# R11 trial version (2026-08-28, a first trial round): v2 + kill lock K=3.
# Single-variable discipline: identical to v2 except vest_kills, for clean attribution.
REWARD_ECONOMY_V3 = RewardEconomy(
    name="v3",
    descend_unit=48.0,
    kill_depth_beta=0.25,
    death_flat=2.0,
    death_c0=24.0,
    death_gamma=0.7,
    death_uses_ladder_spec=False,
    gear_scale=1024.0,
    gear_cap=12.0,
    idle_threshold_steps=300,
    idle_kill_factor=0.5,
    descend_vest_kills=3,
)

# R11 trial two (bisection: the K=3 end point refuted sneaking, step back one notch): v3 changes only vest_kills=1.
REWARD_ECONOMY_V3B = RewardEconomy(
    name="v3b",
    descend_unit=48.0,
    kill_depth_beta=0.25,
    death_flat=2.0,
    death_c0=24.0,
    death_gamma=0.7,
    death_uses_ladder_spec=False,
    gear_scale=1024.0,
    gear_cap=12.0,
    idle_threshold_steps=300,
    idle_kill_factor=0.5,
    descend_vest_kills=1,
)

# R16 amendment v4 (C8 verdict: v2's clause D counts by decision, resets only at a new deepest level, does not reset
# on kills, and halves farm income after >300, which hits exactly the trajectories grinding levels on the current floor). v4 = all v2 parameters
# (dataclasses.replace guarantees field-by-field identity) + redefined anti-idling: the idle clock counts engine micro beats
# and resets on any kill. v2 itself is untouched.
REWARD_ECONOMY_V4 = dataclasses.replace(
    REWARD_ECONOMY_V2,
    name="v4",
    idle_counts_micro_beats=True,
    idle_reset_on_kill=True,
)

REWARD_ECONOMIES = {
    "v1": REWARD_ECONOMY_V1,
    "v2": REWARD_ECONOMY_V2,
    "v3": REWARD_ECONOMY_V3,
    "v3b": REWARD_ECONOMY_V3B,
    "v4": REWARD_ECONOMY_V4,
}


def gear_combat_utility_value(raw, label: str) -> int:
    """Validate and return the native whole-loadout uint32 utility."""
    if "gear_combat_utility" not in raw:
        raise RuntimeError(
            f"{label} is missing native gear_combat_utility")
    value = raw["gear_combat_utility"]
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(
            f"{label}.gear_combat_utility must be a non-negative integer")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"{label}.gear_combat_utility must be a non-negative integer") from exc
    if (
        not math.isfinite(numeric)
        or numeric != float(integer)
        or integer < 0
        or integer > 0xFFFFFFFF
    ):
        raise RuntimeError(
            f"{label}.gear_combat_utility must be a uint32")
    return integer


def gear_upgrade_reward_delta_component(delta: int) -> float:
    """Bounded shaping for one causally attributed native utility increase."""
    if isinstance(delta, (bool, np.bool_)) or not isinstance(
            delta, (int, np.integer)):
        raise RuntimeError("gear utility delta must be a non-negative integer")
    delta = int(delta)
    if not 0 <= delta <= 0xFFFFFFFF:
        raise RuntimeError("gear utility delta must be a uint32")
    return min(
        GEAR_COMBAT_UTILITY_REWARD_CAP,
        float(delta) / GEAR_COMBAT_UTILITY_REWARD_SCALE,
    )


def gear_upgrade_reward_component(previous, current) -> float:
    """Bounded positive shaping from the native gear comparator's own ledger."""

    delta = max(
        0,
        gear_combat_utility_value(current, "current")
        - gear_combat_utility_value(previous, "previous"),
    )
    return gear_upgrade_reward_delta_component(delta)


@dataclass(frozen=True)
class _ControllerMonster:
    """One canonical visible monster row in the controller snapshot."""

    monster_id: int
    generation_key: tuple[int, int, int]
    monster_type: int
    x: int
    y: int
    future_x: int
    future_y: int
    hp: int
    max_hp: int
    ledger_low: int
    ledger_max: int
    blocked: bool
    visible: bool
    native_reachable: bool
    locally_engageable: bool
    dynamic_quantities: tuple[float, ...]
    combat_flags: int
    # R18-E: runtime AI / level / damage for the engagement selector (not on the wire).
    ai: int = 0
    monster_level: int = 0
    max_damage: int = 0


@dataclass(frozen=True)
class _ControllerMissile:
    """One canonical radius-12 projectile slot, already float32-safe."""

    quantities: tuple[float, ...]


@dataclass(frozen=True)
class _ControllerItemTarget:
    x: int
    y: int
    active_id: int
    seed_hi: int
    seed_lo: int
    create_info: int
    base_id: int
    heal_kind: int
    gear_quantities: tuple[float, ...]
    effect_flags: int
    dam_ac_flags: int


@dataclass(frozen=True)
class _ControllerSnapshot:
    """Immutable state shared by one Worker observation and its macro action."""

    raw_identity: int
    steps: int
    scene: tuple[int, bool, int]
    player_x: int
    player_y: int
    player_future_x: int
    player_future_y: int
    walkable: tuple[int, ...]
    visible_monster: tuple[int, ...]
    physical_monster: tuple[int, ...]
    softwall: tuple[int, ...]
    closed_door: tuple[int, ...]
    hazard: tuple[int, ...]
    explosive_softwall: tuple[int, ...]
    visited_mask: tuple[int, ...]
    blocked_mask: tuple[int, ...]
    protected_mask: tuple[int, ...]
    visited_tiles: frozenset[tuple[int, int]]
    explore_blocked_targets: frozenset[tuple[int, int]]
    protected_tiles: frozenset[tuple[int, int]]
    sticky_target: tuple[int, int] | None
    monsters: tuple[_ControllerMonster, ...]
    monster_overflow_quantities: tuple[float, ...]
    candidates: tuple[_ControllerMonster, ...]
    missiles: tuple[_ControllerMissile, ...]
    missile_overflow_quantities: tuple[float, ...]
    heal_target: _ControllerItemTarget | None
    gear_target: _ControllerItemTarget | None
    belt_slot_kinds: tuple[int, ...]
    exact_quantities: tuple[float, ...]
    combat_quantities: tuple[float, ...]
    combat_effect_flags: int
    combat_dam_ac_flags: int
    equipped_quantities: tuple[float, ...]


def terminal_death_reward_component(
        *, dead: bool, dungeon_level, death_ladder: bool,
        economy: RewardEconomy = REWARD_ECONOMY_V1) -> float:
    """Return only the terminal-death component of the native reward.

    This pure function is shared by :class:`DiabloGymEnv` when it settles the
    real transition and by ``WorkerWindowEnv`` when it reconstructs that one
    component across a frozen manager/script boundary.  No XP, combat,
    movement, progression, or victory credit is included.

    economy=v1 (default) reproduces historical behaviour bit for bit; with economy=v2 the death penalty becomes the revised
    decreasing function flat + c0×gamma^(max(d,1)-1), and the death_ladder branch is overridden by v2.
    """
    if not isinstance(dead, (bool, np.bool_)):
        raise TypeError(f"dead must be a bool, got {dead!r}")
    if not dead:
        return 0.0
    if not isinstance(death_ladder, (bool, np.bool_)):
        raise TypeError(
            f"death_ladder must be a bool, got {death_ladder!r}")
    if isinstance(dungeon_level, (bool, np.bool_)):
        raise ValueError(
            f"terminal-death dungeon_level must be a non-negative integer, got {dungeon_level!r}")
    try:
        depth = int(dungeon_level)
        numeric_depth = float(dungeon_level)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"terminal-death dungeon_level must be a non-negative integer, got {dungeon_level!r}"
        ) from exc
    if (not math.isfinite(numeric_depth)
            or numeric_depth != float(depth)
            or depth < 0):
        raise ValueError(
            f"terminal-death dungeon_level must be a non-negative integer, got {dungeon_level!r}")
    if not economy.death_uses_ladder_spec:
        # v2 revised version: dying on level 1 costs the most, less the deeper (approved 2026-08-27).
        cost = economy.death_flat + (
            economy.death_c0
            * (economy.death_gamma ** (max(depth, 1) - 1)))
        return -float(cost)
    cost = (
        TERMINAL_DEATH_REWARD_SPEC.ladder_cost_per_depth * depth
        if bool(death_ladder)
        else TERMINAL_DEATH_REWARD_SPEC.flat_cost
    )
    return -float(cost)


_DEFAULT_ASSETS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "build" / "engine" / "devilutionx.app" / "Contents" / "Resources"
)
_TEMP_SAVE_LEGACY_PREFIX = "diablogym-saves-"
_TEMP_SAVE_PREFIX = "diablogym-saves-v2-"
_TEMP_SAVE_LOCK = ".owner.lock"
_TEMP_SAVE_REGISTRY_LOCK = (
    f".diablogym-saves-v2.{os.getuid() if hasattr(os, 'getuid') else 0}.global.lock"
)


def _scene_identity(raw) -> tuple[int, bool, int]:
    """A quest set-level with the same main-line depth is still a different map; no difference may be taken across maps."""
    depth = int(raw["dungeon_level"])
    is_set = bool(raw.get("is_set_level", False))
    set_id = int(raw.get("set_level_id", 0)) if is_set else 0
    return depth, is_set, set_id


@contextmanager
def _temp_save_registry_lock(root: pathlib.Path):
    """Serialize scratch publication and reclamation across spawn workers.

    The registry file is intentionally persistent and outside the scratch glob.
    Unlinking it would let waiters retain the old inode while newcomers lock a
    new inode, splitting the critical section in two.
    """
    import fcntl

    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(root / _TEMP_SAVE_REGISTRY_LOCK, flags, 0o600)
    try:
        registry = os.fdopen(fd, "a+", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise
    acquired = False
    try:
        fcntl.flock(registry.fileno(), fcntl.LOCK_EX)
        acquired = True
        yield
    finally:
        if acquired:
            fcntl.flock(registry.fileno(), fcntl.LOCK_UN)
        registry.close()


def _cleanup_stale_temp_save_dirs_locked(base: pathlib.Path) -> int:
    """Reclaim stale scratch while the caller holds the registry lock."""
    import fcntl

    removed = 0
    for candidate in base.glob(f"{_TEMP_SAVE_LEGACY_PREFIX}*"):
        if not candidate.is_dir():
            continue
        marker = candidate / _TEMP_SAVE_LOCK
        if not marker.is_file():
            # A v2 creator publishes the directory and owner marker while
            # holding the same registry lock.  Therefore a visible markerless
            # v2 directory can only be debris from a crashed creator.  Older
            # directories predate that invariant and remain fail-closed.
            if candidate.name.startswith(_TEMP_SAVE_PREFIX):
                try:
                    shutil.rmtree(candidate)
                    removed += 1
                except OSError:
                    pass
            continue
        try:
            owner = open(marker, "a+", encoding="utf-8")
        except OSError:
            continue
        acquired = False
        try:
            try:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                continue
            try:
                shutil.rmtree(candidate)
                removed += 1
            except OSError:
                # The owner may have completed cleanup after our path lookup;
                # genuine I/O failures remain for a later startup to retry.
                pass
        finally:
            if acquired:
                fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
            owner.close()
    return removed


def _cleanup_stale_temp_save_dirs(root: pathlib.Path | None = None) -> int:
    """Reclaim new-style scratch directories left behind by SIGKILL that no process holds a lock on any more.

    v2 creation/cleanup is serialized by a global transaction lock, so half-built directories without a marker left by a crash
    can be reclaimed; older versions have no such ownership evidence, so they are kept rather than guessed. flock is released by the
    kernel when the process dies, so PID reuse cannot cause wrong or missed deletions.
    """
    base = (pathlib.Path(root) if root is not None
            else pathlib.Path(tempfile.gettempdir()))
    with _temp_save_registry_lock(base):
        return _cleanup_stale_temp_save_dirs_locked(base)


def _create_locked_temp_save_dir():
    import fcntl

    base = pathlib.Path(tempfile.gettempdir())
    with _temp_save_registry_lock(base):
        _cleanup_stale_temp_save_dirs_locked(base)
        directory = tempfile.TemporaryDirectory(
            prefix=_TEMP_SAVE_PREFIX, dir=base)
        owner = open(pathlib.Path(directory.name) / _TEMP_SAVE_LOCK,
                     "a+", encoding="utf-8")
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner.write(f"pid={os.getpid()}\n")
            owner.flush()
            os.fsync(owner.fileno())
        except Exception:
            owner.close()
            directory.cleanup()
            raise
    return directory, owner


class DiabloGymEnv(gym.Env):
    metadata = {"render_modes": []}

    # Class-level defaults of the R10 economy act: v1 = history unchanged bit for bit. Existing tests/forensic scripts often
    # build a bare instance via __new__ and call _reward directly; class-level defaults keep that path on v1 forever;
    # the normal __init__ overwrites the instance attributes from its arguments.
    reward_economy = REWARD_ECONOMY_V1
    _econ_steps_on_level = 0
    _econ_idle_prev_steps = 0  # R16 v4: self._steps at the last _reward (for micro-beat counting)
    _econ_episode_max_depth = 0
    _econ_unvested = 0.0
    _econ_kills_on_floor = 0
    _econ_prev_epkills = 0

    # DevilutionX is an in-process global singleton, not a re-entrant multi-instance engine. The same
    # process may reuse several wrappers sequentially, but must not interleave their steps; multiple environments need
    # a multi-process scheme such as SubprocVecEnv. Keeping explicit books here turns silent state crosstalk
    # into a loud exception.
    _engine_initialized = False
    _engine_pid: int | None = None
    _engine_config: tuple[str, str, str, int] | None = None
    _active_token = None
    _temp_save_dir: tempfile.TemporaryDirectory | None = None
    _temp_save_lock = None
    _atfork_registered = False
    # R18-H sweep-v1 / R18-M (2026-09-07) review round: the raw "objects"
    # observation channel is a bridge global (src/resource_sweep.hpp:17
    # gObjectObservation), i.e. the same process-singleton kind of state as
    # _engine_initialized above, so it is booked HERE and not per instance.
    # H1 declared exactly this law in its own comment at
    # _configure_native_resource_protocol ("an env that asked once is also
    # responsible for turning it back off, so a later sweep-off env in the same
    # process sees the frozen dict again") but wrote the flag as an instance
    # attribute, where a fresh env always reads False and therefore never turns
    # the channel off: after any sweep-v1 env, the next sweep-off or
    # identify-only env in the same process died at its first reset() with
    # "Native sweep identity mismatch; the objects channel is on while sweep is
    # off" (validate_native_sweep, resource_protocol.py:234-236).  Reproduced
    # against the merged bridge before the fix; see M-REPORT §8.
    _sweep_object_channel = False

    @classmethod
    def _after_fork_child(cls) -> None:
        """A child process must not destroy the scratch directory or native engine still used by the parent."""
        directory, cls._temp_save_dir = cls._temp_save_dir, None
        owner, cls._temp_save_lock = cls._temp_save_lock, None
        if directory is not None:
            finalizer = getattr(directory, "_finalizer", None)
            if finalizer is not None and finalizer.alive:
                finalizer.detach()  # only cancels the child's copy; must not rmtree the parent's directory
        if owner is not None:
            owner.close()

    def __init__(
        self,
        assets_dir: str | None = None,
        save_dir: str | None = None,
        data_dir: str | None = None,
        ticks_per_step: int = 4,
        max_steps: int = 5000,
        start_in_dungeon: bool = False,
        include_raw: bool = True,
        descend_ladder: bool = False,
        death_ladder: bool = False,
        hero_class: int = 0,
        controller_snapshot_enabled: bool = False,
        descend_fallback_promotion: bool = True,
        reward_economy: str = "v1",
        explore_global_fallback: bool = False,
        progress_far_tiles: int = 0,
        explore_global_hunt: bool = False,
        resource_protocol: str = "off",
        resource_purchase_mode: str = "full",
        dive_blocker_recovery: str = "off",
        resource_ordinary_armor_scope: bool = False,
        resource_preserve_equipment_readiness: bool = False,
        resource_loot_economy: bool = False,
        resource_readiness_law: str = "veto-v1",
        resource_retreat: str = "off",
        resource_portal: str = "off",
        resource_sweep: str = "off",
        resource_identify: str = "off",
        resource_weapon_upgrade: str = "off",
        aggro_cap: str = "off",
        engagement_priority: str = "off",
        hunt_scope: str = "all",
    ):
        super().__init__()
        from .resource_protocol import validate_resource_config, validate_ordinary_armor_scope
        self.resource_protocol, self.resource_purchase_mode = validate_resource_config(
            resource_protocol, resource_purchase_mode)
        self.resource_ordinary_armor_scope = validate_ordinary_armor_scope(
            self.resource_protocol, self.resource_purchase_mode, resource_ordinary_armor_scope)
        from .resource_protocol import validate_equipment_readiness_preservation
        self._resource_preserve_equipment_readiness = validate_equipment_readiness_preservation(
            self.resource_protocol, resource_preserve_equipment_readiness)
        if type(resource_loot_economy) is not bool:
            raise ValueError("resource_loot_economy must be bool")
        if resource_loot_economy and (
                self.resource_protocol != "l2-town-v1" or self.resource_purchase_mode != "full"
                or not self.resource_ordinary_armor_scope
                or not self._resource_preserve_equipment_readiness):
            raise ValueError("loot economy requires l2-town-v1/full with ordinary armor and equipment preservation")
        self._resource_loot_economy = resource_loot_economy
        # R17.1 readiness rule 3: readiness law (veto-v1 = R18-R23 native veto, bit-identical;
        # coach-v03 = six-condition coach + accounted forced descents).
        from .resource_protocol import validate_readiness_law
        self.resource_readiness_law = validate_readiness_law(
            self.resource_protocol, resource_readiness_law)
        # R18-A retreat-v1: return-to-town interface (default off = byte-identical).
        from .resource_protocol import validate_retreat_protocol
        self.resource_retreat = validate_retreat_protocol(
            self.resource_protocol, self.resource_readiness_law, resource_retreat)
        # R18-F portal-v1: the Scroll of Town Portal vehicle for that same
        # interface -- the retreat without the lost depth (default off =
        # byte-identical; portal-v1 requires retreat-v1 to be on).
        from .resource_protocol import validate_portal_protocol
        self.resource_portal = validate_portal_protocol(
            self.resource_protocol, self.resource_readiness_law,
            self.resource_retreat, resource_portal)
        # R18-H (2026-09-07) sweep-v1: the chest/barrel sweep on main L1. The
        # loot economy is what turns the drops into gold, so the validator
        # demands it; here that fact is the already-derived loot flag.
        # Default off = byte-identical (no native call, no new observation key).
        from .resource_protocol import validate_sweep_protocol
        self.resource_sweep = validate_sweep_protocol(
            self.resource_protocol,
            "sustain-loot-v1" if self._resource_loot_economy else "legacy-v1",
            resource_sweep)
        # Process-global native channel: only ever touched by an env that wants
        # it, plus by an env that must turn OFF what an earlier env turned on.
        # R18-M (2026-09-07) review round: NOT reset here.  The channel it
        # tracks is a bridge global that outlives this instance, so the flag
        # lives on the class (DiabloGymEnv._sweep_object_channel, declared with
        # the other process-singleton state above); zeroing it per instance is
        # what made a fresh sweep-off env skip the turn-off call.
        # R18-H identify-v1 (2026-09-07) Cain identify: the Cain identify leg of the
        # loot economy's town trip (default off = byte-identical).
        from .resource_identify import validate_identify_env
        self.resource_identify = validate_identify_env(
            self.resource_protocol, self._resource_loot_economy, resource_identify)
        # R18-K (2026-09-07) smith-v1: the surplus weapon upgrade leg at Griswold.
        # Default off passes no new native flag and leaves every path byte-identical.
        from .resource_weapon_upgrade import RESOURCE_WEAPON_UPGRADES
        if resource_weapon_upgrade not in RESOURCE_WEAPON_UPGRADES:
            raise ValueError(
                f"Unknown resource_weapon_upgrade {resource_weapon_upgrade!r}; "
                f"expected one of {RESOURCE_WEAPON_UPGRADES}")
        if resource_weapon_upgrade != "off" and (
                self.resource_protocol != "l2-town-v1"
                or self.resource_purchase_mode != "full"
                or not self._resource_loot_economy):
            raise ValueError("smith-v1 requires l2-town-v1/full with the loot economy")
        self.resource_weapon_upgrade = resource_weapon_upgrade
        # R18-D aggro cap / R18-E engagement priority (defaults off = byte-identical).
        from .aggro_cap import validate_aggro_cap, AggroCapPolicy
        self.aggro_cap = validate_aggro_cap(aggro_cap)
        self._aggro_cap_policy = AggroCapPolicy() if self.aggro_cap != "off" else None
        self._aggro_cap_fired = 0
        from .engagement import validate_engagement_priority
        self.engagement_priority = validate_engagement_priority(engagement_priority)
        self._engagement_decisions = 0
        self._engagement_reordered = 0
        # R18-G (2026-09-07): scope of the a10 global hunt (the pull mechanism).
        # "all" = frozen behaviour; "l1-only" = no global hunt on main L2+.
        if hunt_scope not in HUNT_SCOPES:
            raise ValueError(f"Unknown hunt_scope {hunt_scope!r}; expected all or l1-only")
        self.hunt_scope = hunt_scope
        if dive_blocker_recovery not in ("off", "adjacent-v1"):
            raise ValueError("dive_blocker_recovery must be off or adjacent-v1")
        if dive_blocker_recovery != "off" and self.resource_protocol == "off":
            raise ValueError("dive_blocker_recovery requires the resource protocol")
        self.dive_blocker_recovery = dive_blocker_recovery
        self._resource_pending_command = None
        self._resource_terminal_reason = None
        self._resource_dive_authority = False
        self._resource_actual_microsteps = 0
        self._resource_scene_ledgers = {}
        self._resource_max_main_depth = 0
        self._resource_transition_receipts = []
        self._resource_reward_depth_before = 0
        self._resource_step_bonus = 0.0
        # R16 amendment (2026-09-01): two switches that default to off; with the defaults the code path is unchanged bit for bit:
        #   explore_global_fallback (C5): when a10 finds no frontier/softwall candidate in the 25×25 window,
        #     instead of waiting it runs a whole-map BFS to the nearest unvisited reachable tile or live-monster
        #     tile and takes the farthest locally reachable prefix point of that path inside the window as this frontier;
        #   progress_far_tiles (C6): when >0, only new tiles at Chebyshev distance >= this value from the existing
        #     "progress anchors" advance exploration_progress (pacing does not count); 0 = the old rule
        #     (any first-visited tile counts).
        if not isinstance(explore_global_fallback, (bool, np.bool_)):
            raise TypeError(
                "explore_global_fallback must be a bool, got "
                f"{explore_global_fallback!r}")
        self._explore_global_fallback = bool(explore_global_fallback)
        if (isinstance(progress_far_tiles, bool)
                or not isinstance(progress_far_tiles, (int, np.integer))
                or int(progress_far_tiles) < 0):
            raise ValueError(
                "progress_far_tiles must be a non-negative integer (0 = old rule), got "
                f"{progress_far_tiles!r}")
        self._progress_far_tiles = int(progress_far_tiles)
        # R16 C5 extra variant (default off; a probe found the fallback alone almost never fires): when a10 sees
        # no visible monster in the 25×25 window, it does not wait for the local frontier to run out but runs a whole-map BFS
        # toward the nearest live monster (including ones behind walls/unlit) (still taking only in-window waypoints and using the
        # existing step-by-step walking); only when no monster on the map is reachable does it return to local frontier exploration.
        if not isinstance(explore_global_hunt, (bool, np.bool_)):
            raise TypeError(
                "explore_global_hunt must be a bool, got "
                f"{explore_global_hunt!r}")
        self._explore_global_hunt = bool(explore_global_hunt)
        if (isinstance(ticks_per_step, bool)
                or not isinstance(ticks_per_step, (int, np.integer))
                or int(ticks_per_step) <= 0):
            raise ValueError(f"ticks_per_step must be a positive integer, got {ticks_per_step!r}")
        if (isinstance(max_steps, bool)
                or not isinstance(max_steps, (int, np.integer))
                or int(max_steps) <= 0):
            raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
        if (isinstance(hero_class, bool)
                or not isinstance(hero_class, (int, np.integer))
                or int(hero_class) != 0):
            raise ValueError(
                "the current action/auto stat-allocation contract only supports hero_class=0 (Warrior); "
                f"got {hero_class!r}")
        if not isinstance(controller_snapshot_enabled, (bool, np.bool_)):
            raise TypeError(
                "controller_snapshot_enabled must be a bool, got "
                f"{controller_snapshot_enabled!r}")
        # E-fix A (form 1): when monster-avoiding planning "succeeds but cannot step onto the stairs", promote the
        # lenient planner by rank comparison (the same existing rule as _macro_progression); False = the old-behaviour end point,
        # used only for control legs/bit-level replay of old archives.
        if not isinstance(descend_fallback_promotion, (bool, np.bool_)):
            raise TypeError(
                "descend_fallback_promotion must be a bool, got "
                f"{descend_fallback_promotion!r}")
        self._descend_fallback_promotion = bool(descend_fallback_promotion)

        assets = str(pathlib.Path(assets_dir or _DEFAULT_ASSETS).expanduser().resolve())
        data = str(pathlib.Path(
            data_dir
            or pathlib.Path.home() / "Library/Application Support/diasurgical/devilution"
        ).expanduser().resolve())
        cls = DiabloGymEnv
        pid = os.getpid()
        if not cls._atfork_registered and hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=cls._after_fork_child)
            cls._atfork_registered = True
        if cls._engine_initialized and cls._engine_pid != pid:
            raise RuntimeError(
                "the DiabloGym engine was already initialized in the parent process and cannot be reused after fork; "
                "a fork child may only exec/os._exit immediately; use spawn for multi-environment training")

        # bridge.init() is a long C++ call; SIGINT may turn into KeyboardInterrupt exactly after it returns successfully
        # but before Python has written the three class attributes.
        # The native configuration is the committed source of truth; every construction first uses it to repair a Python
        # ledger possibly torn by an async exception, and must not delete the scratch save directory native code still uses.
        native_config = bridge.engine_config()
        if native_config is not None:
            recovered = (str(native_config[0]), str(native_config[1]),
                         str(native_config[2]), int(native_config[3]))
            cls._engine_config = recovered
            cls._engine_pid = pid
            cls._engine_initialized = True
        elif cls._engine_initialized:
            raise RuntimeError(
                "DiabloGym Python/native singleton ledgers disagree: Python marks it initialized, "
                "but the native bridge is not initialized")
        if cls._engine_initialized:
            if cls._engine_config is None:
                raise RuntimeError("DiabloGym engine singleton state corrupt: initialized but configuration missing")
            _, old_saves, _, _ = cls._engine_config
            requested_save = (str(pathlib.Path(save_dir).expanduser().resolve())
                              if save_dir is not None else old_saves)
            requested = (assets, requested_save, data, int(hero_class))
            if requested != cls._engine_config:
                raise RuntimeError(
                    "DevilutionX is an in-process singleton and cannot be re-initialized with a different assets/save/data/"
                    f"hero_class; existing={cls._engine_config!r}, requested={requested!r}")
            saves = old_saves
        else:
            if save_dir is not None:
                saves = str(pathlib.Path(save_dir).expanduser().resolve())
            else:
                # The save scratch must live until the in-process engine exits, but should not, like
                # mkdtemp, leave permanent disk garbage behind after every multi-process training run.
                cls._temp_save_dir, cls._temp_save_lock = _create_locked_temp_save_dir()
                saves = cls._temp_save_dir.name
            try:
                bridge.init(assets_dir=assets, save_dir=saves, data_dir=data,
                            hero_class=int(hero_class))
            except BaseException:
                # If the async exception happens after the native commit, keep native and scratch
                # and complete the Python ledger; roll back the disk only if native really did not commit.
                committed = bridge.engine_config()
                if committed is not None:
                    cls._engine_config = (
                        str(committed[0]), str(committed[1]),
                        str(committed[2]), int(committed[3]))
                    cls._engine_pid = pid
                    cls._engine_initialized = True
                elif save_dir is None and cls._temp_save_dir is not None:
                    cls._temp_save_dir.cleanup()
                    cls._temp_save_dir = None
                    if cls._temp_save_lock is not None:
                        cls._temp_save_lock.close()
                        cls._temp_save_lock = None
                raise
            cls._engine_config = (assets, saves, data, int(hero_class))
            cls._engine_pid = pid
            # The commit bit must be written last: if SIGINT lands between the config/PID assignments, the next
            # construction recovers from native engine_config; setting True first would instead misread the missing PID
            # as a fork and never even reach the recovery logic.
            cls._engine_initialized = True

        self.ticks_per_step = int(ticks_per_step)
        self.max_steps = int(max_steps)
        self.start_in_dungeon = start_in_dungeon
        self.include_raw = include_raw
        # v17 deep water: the descend bonus grows with depth (N->N+1 pays 8×N; False = the flat 8.0 of v6-v16,
        # leaving the world rules of the old gold standard untouched)
        self.descend_ladder = descend_ladder
        # v18: death cost priced in step with the ladder (dying on level N costs 8×N; False = constant -2.0).
        # Lesson 16: a ladder of 8/16/24 against a death cost of -2 makes sprinting a sure win in expectation (+5.8);
        # "arriving alive" must outbid "touching the depth"
        self.death_ladder = death_ladder
        # R10 economy act: v1 = all historical constants unchanged bit for bit (default); v2 = depth economy
        # (A/B/C/D/E1, approved 2026-08-27). String entry point; the single source of truth is
        # REWARD_ECONOMIES.
        if reward_economy not in REWARD_ECONOMIES:
            raise ValueError(
                f"reward_economy must be one of {sorted(REWARD_ECONOMIES)},"
                f" got {reward_economy!r}")
        self.reward_economy = REWARD_ECONOMIES[reward_economy]
        # Clause-D state: steps spent on the same level and the deepest level of this episode (cleared on reset).
        self._econ_steps_on_level = 0
        self._econ_episode_max_depth = 0
        # The old 295/298 views do not consume the controller wire; by default no extra 25×25 map is fetched
        # per decision, and old raw data need not carry the new protocol fields. The dual Worker enables it explicitly
        # when the wrapper is constructed; macro actions still take one local, uncached snapshot at the key-press boundary.
        self._controller_snapshot_enabled = bool(
            controller_snapshot_enabled)
        side = 2 * _MAP_RADIUS + 1
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(12 + _K_MONSTERS * 4 + 2 * side * side + 9,), dtype=np.float32,
        )  # +9 = v13 potions 4 dims + v14 gear 4 dims + v19 strength gauge 1 dim
        self._token = object()
        self._raw = None
        self._native_generation: int | None = None
        self._episode_seed: int | None = None
        self._episode_ended = True
        self._steps = 0
        self._ep_kills = 0
        self._ep_start_xp = 0
        self._visited: set[tuple[int, int]] = set()
        # Per-episode monotone exploration progress clock: +1 whenever any action first steps on a new tile or action10 really
        # opens a softwall. Options FARM resets its "no progress" clock from the difference, so that
        # teacher (action10) and learned worker (direction keys/chase) do not use two sets of termination
        # semantics. It does not wrap on scene changes; only reset zeroes it.
        self._exploration_progress = 0
        self._softwalls_opened = 0
        # action10's frontier target must persist across macros. If every 12 ticks it re-picked "the one nearest to the
        # current position", maze forks would form a stable 2-cycle: after walking toward A, B is nearer;
        # after walking toward B, A is nearer again, and it would finally report exhausted with many rooms left.
        self._explore_target: tuple[int, int] | None = None
        self._explore_blocked_targets: set[tuple[int, int]] = set()
        # v4: native FindPath's reachable is an instantaneous geometric snapshot; monster/player
        # animations and dynamic occupancy can still leave a target without progress during an actual chase. Remember the
        # targets proven to fail in this scene, so the next action9 rotates first instead of always being pulled back by the
        # stable ActiveMonsters/id order to the same equidistant target.
        self._engage_blocked_keys: set[tuple[int, int, int]] = set()
        # v4: the lowest HP line already paid for in this scene during each monster's lifetime. The key is
        # (active slot, rndItemSeed hi, lo), not just the slot id, which runtime spawns can reuse;
        # the value is (lowest_hp, denominator_max_hp).
        self._combat_hp_floor: dict[
            tuple[int, int, int], tuple[int, int]
        ] = {}
        self._controller_snapshot: _ControllerSnapshot | None = None

    @property
    def resource_preserve_equipment_readiness(self):
        """Constructor-only protocol identity; reset reapplies the same native flag."""
        return getattr(self, "_resource_preserve_equipment_readiness", False)

    @property
    def resource_loot_economy(self):
        return getattr(self, "_resource_loot_economy", False)

    def _validate_native_resource_flags(self, raw):
        from .resource_protocol import (
            validate_native_armor_scope, validate_native_equipment_readiness_preservation)
        validate_native_armor_scope(raw, getattr(self, "resource_ordinary_armor_scope", False))
        validate_native_equipment_readiness_preservation(
            raw, getattr(self, "resource_preserve_equipment_readiness", False))
        expected_loot = getattr(self, "resource_loot_economy", False)
        observed_loot = raw.get("resource_state", {}).get("loot_economy", False)
        if type(observed_loot) is not bool or observed_loot != expected_loot:
            raise RuntimeError("Native loot economy identity mismatch; isolated rebuild required")
        from .resource_protocol import validate_native_readiness_law
        validate_native_readiness_law(raw, getattr(self, "resource_readiness_law", "veto-v1"))
        from .resource_protocol import validate_native_retreat
        validate_native_retreat(raw, getattr(self, "resource_retreat", "off"))
        from .resource_protocol import validate_native_portal
        validate_native_portal(raw, getattr(self, "resource_portal", "off"))
        from .resource_protocol import validate_native_sweep
        validate_native_sweep(raw, getattr(self, "resource_sweep", "off"))
        from .resource_identify import validate_native_identify
        validate_native_identify(raw, getattr(self, "resource_identify", "off"))
        # R18-K2b (2026-09-07): the weapon-purchase handshake is checked on
        # EVERY episode, not only smith-v1 ones. gResourceWeaponPurchase is a
        # process-wide native flag, so this is what proves a frozen arm never
        # inherited it from another env in the same worker. A bridge older than
        # R18-K2b exports no key at all, which reads as the default (off).
        observed_weapon = raw.get("resource_state", {}).get("weapon_purchase_enabled", False)
        expected_weapon = getattr(self, "resource_weapon_upgrade", "off") == "smith-v1"
        if type(observed_weapon) is not bool or observed_weapon != expected_weapon:
            raise RuntimeError(
                "Native weapon-purchase scope identity mismatch "
                f"(observed {observed_weapon!r}, expected {expected_weapon!r}); "
                "isolated rebuild or a leaked configure_resource_weapon_purchase")
        if getattr(self, "resource_weapon_upgrade", "off") != "off":
            from .resource_weapon_upgrade import validate_native_weapon_upgrade
            validate_native_weapon_upgrade(raw, self.resource_weapon_upgrade)

    # ---------- gymnasium API ----------

    def reset(self, *, seed: int | None = None, options=None):
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            raise RuntimeError(
                "resetting a DevilutionX instance initialized by the parent process in a fork child is forbidden; "
                "multi-environment training must use spawn")
        super().reset(seed=seed)
        # Clause-D state of R10 economy v2 is cleared per episode (harmlessly always zero under v1).
        self._econ_steps_on_level = 0
        self._econ_idle_prev_steps = 0  # R16 v4 micro-beat baseline (read only by v4)
        self._econ_episode_max_depth = 0
        # R11 kill-lock state is cleared per episode.
        self._econ_unvested = 0.0
        self._econ_kills_on_floor = 0
        self._econ_prev_epkills = 0
        actual_seed = seed if seed is not None else int(self.np_random.integers(2**31))
        actual_seed = int(actual_seed)
        if not 0 <= actual_seed <= np.iinfo(np.uint32).max:
            raise ValueError(f"seed must be within the uint32 range [0, 2**32-1], got {actual_seed}")
        try:
            DiabloGymEnv._active_token = self._token
            self._configure_native_resource_protocol()
            self._raw = bridge.reset(seed=actual_seed)
            self._validate_native_resource_flags(self._raw)
            self._native_generation = int(bridge.episode_generation())
            if self.start_in_dungeon:
                # The town layout is fixed; walk to the cathedral stairs by script (about 500-900 ticks, ~0.05 s)
                if getattr(self, "resource_preserve_equipment_readiness", False):
                    self._raw = nav.descend_to_dungeon(
                        bridge, raw_validator=self._validate_native_resource_flags)
                else:
                    self._raw = nav.descend_to_dungeon(bridge)
                self._validate_native_resource_flags(self._raw)
            if self._native_monster_kill_delta(
                    self._raw, self._raw) is None:
                raise RuntimeError(
                    "the current native bridge lacks monster_kill_total; "
                    "spawn->death within a macro cannot be counted completely")
            self._steps = 0
            self._episode_seed = actual_seed
            self._episode_ended = False
            self._ep_kills = 0
            self._ep_start_xp = int(self._raw["xp"])
            self._visited = {(self._raw["player_x"], self._raw["player_y"])}
            # R16 C6 progress anchors start from the same source as the footprints (read only when progress_far_tiles>0).
            self._resource_actual_microsteps = 0
            self._aggro_cap_fired = 0
            self._engagement_decisions = 0
            self._engagement_reordered = 0
            self._resource_terminal_reason = None
            self._resource_pending_command = None
            self._resource_dive_authority = False
            self._resource_scene_ledgers = {}
            self._resource_max_main_depth = int(self._raw["dungeon_level"])
            self._resource_transition_receipts = []
            self._resource_reward_depth_before = self._resource_max_main_depth
            self._resource_step_bonus = 0.0
            self._progress_anchors = set(self._visited)
            self._exploration_progress = 0
            self._softwalls_opened = 0
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._reset_combat_ledger(self._raw)
            self._controller_snapshot = (
                self._capture_controller_snapshot(self._raw)
                if self._controller_snapshot_enabled else None
            )
            obs = self._vectorize(self._raw)
            info = self._info(self._raw)
        except BaseException:
            # Navigation/observation construction is also part of the reset transaction; a failure midway must not
            # leave a half-initialized episode that looks steppable.
            try:
                bridge.end_game()
            except Exception:
                pass
            if DiabloGymEnv._active_token is self._token:
                DiabloGymEnv._active_token = None
            self._raw = None
            self._native_generation = None
            self._episode_seed = None
            self._episode_ended = True
            self._exploration_progress = 0
            self._softwalls_opened = 0
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._combat_hp_floor = {}
            self._controller_snapshot = None
            raise
        return obs, info

    @staticmethod
    def _policy_monsters(raw) -> list[dict]:
        """The monster subset the policy may consume; the full-level monsters remain for the reward/kill ledger.

        The branch without visible/reachable only serves old pure-Python synthetic fixtures;
        v4 native raw always provides both fields explicitly.
        """
        return [
            m for m in raw.get("monsters", ())
            if bool(m.get("visible", True)) and bool(m.get("reachable", True))
        ]

    @staticmethod
    def _policy_floor_items(raw, flag: str) -> list[dict]:
        if flag not in {"heal", "gear"}:
            raise ValueError(f"unknown floor-item policy flag: {flag!r}")
        return [
            it for it in raw.get("floor_items", ())
            if bool(it.get(flag))
            and bool(it.get("visible", True))
            and bool(it.get("reachable", True))
        ]

    @staticmethod
    def _belt_free_slots(raw) -> int:
        if "belt_free_slots" in raw:
            return max(0, int(raw["belt_free_slots"]))
        # Only an approximation for compatibility with protocol-v3 synthetic fixtures/old bridges; only the v4 native field
        # can tell "2 potions + 6 other belt items" from "2 potions + 6 empty slots".
        return max(0, 8 - int(raw.get("belt_heals", 0)))

    @staticmethod
    def _reflex_eligible(raw) -> bool:
        """Shared emergency handoff predicate for every multi-beat macro."""
        return (
            2 * int(raw.get("hp", 0))
            < max(1, int(raw.get("max_hp", 0)))
            and int(raw.get("belt_heals", 0)) > 0
        )

    @classmethod
    def _belt_observation_scalar(cls, raw) -> float:
        """Publish the heal potion count and the real free-slot count unambiguously in the one existing scalar.

        heals/free are both 0..8 integers of the native belt. heals uses the historical 1/8 main scale,
        free uses a 1/128 sub-scale, so adjacent heals buckets still keep a 1/16 gap;
        old policy inputs shift by at most 8/128=0.0625, and the shape and old main scale are unchanged.
        """
        heals = min(8, max(0, int(raw.get("belt_heals", 0))))
        free = min(8, cls._belt_free_slots(raw))
        return heals / 8.0 + free / 128.0

    @staticmethod
    def _controller_binary_channel(values, name: str) -> tuple[int, ...]:
        expected = CONTROLLER_SNAPSHOT_CELLS
        try:
            frozen = tuple(int(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"controller snapshot {name} channel cannot be converted to integers") from exc
        if len(frozen) != expected or any(value not in (0, 1) for value in frozen):
            raise RuntimeError(
                f"controller snapshot {name} must be {expected} values of 0/1, "
                f"got len={len(frozen)}")
        return frozen

    @staticmethod
    def _controller_uint(value, *, name: str, bits: int) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise RuntimeError(
                f"controller snapshot {name} must be uint{bits} integers")
        try:
            integer = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"controller snapshot {name} must be uint{bits} integers") from exc
        if (
            not math.isfinite(numeric)
            or numeric != float(integer)
            or not 0 <= integer < (1 << bits)
        ):
            raise RuntimeError(
                f"controller snapshot {name} must be uint{bits} integers")
        return integer

    @classmethod
    def _monster_generation_key(cls, monster) -> tuple[int, int, int]:
        """Stable native monster lifetime key, never serialized to policy."""
        if not isinstance(monster, dict):
            raise RuntimeError("monster generation needs a dict")
        monster_id = cls._controller_uint(
            monster.get("id"), name="monster.id", bits=16)
        seed_hi = cls._controller_uint(
            monster.get("rnd_item_seed_hi"),
            name=f"monster[{monster_id}].rnd_item_seed_hi",
            bits=16,
        )
        seed_lo = cls._controller_uint(
            monster.get("rnd_item_seed_lo"),
            name=f"monster[{monster_id}].rnd_item_seed_lo",
            bits=16,
        )
        return monster_id, seed_hi, seed_lo

    @staticmethod
    def _controller_int32_words(value, *, name: str) -> tuple[float, float]:
        if isinstance(value, (bool, np.bool_)):
            raise RuntimeError(
                f"controller snapshot {name} must be an int32 integer")
        try:
            integer = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"controller snapshot {name} must be an int32 integer") from exc
        if (
            not math.isfinite(numeric)
            or numeric != float(integer)
            or not -(1 << 31) <= integer < (1 << 32)
        ):
            raise RuntimeError(
                f"controller snapshot {name} must fit in a 32-bit word")
        unsigned = integer & 0xFFFFFFFF
        return (
            float((unsigned >> 16) & 0xFFFF) / 65536.0,
            float(unsigned & 0xFFFF) / 65536.0,
        )

    @staticmethod
    def _controller_target(
        raw,
        flag: str,
        *,
        player_x: int,
        player_y: int,
        reachable_tiles: frozenset[tuple[int, int]],
    ) -> _ControllerItemTarget | None:
        targets = [
            item for item in raw.get("floor_items", ())
            if bool(item.get(flag)) and bool(item.get("visible", True))
            and (
                int(item["x"]), int(item["y"])
            ) in reachable_tiles
            and abs(int(item["x"]) - player_x)
            <= CONTROLLER_SNAPSHOT_RADIUS
            and abs(int(item["y"]) - player_y)
            <= CONTROLLER_SNAPSHOT_RADIUS
        ]
        if not targets:
            return None

        def active_id(item) -> int:
            if "active_id" not in item:
                raise RuntimeError(
                    f"controller snapshot {flag} target is missing active_id")
            return DiabloGymEnv._controller_uint(
                item["active_id"], name=f"{flag}.active_id", bits=7)

        target = min(
            targets,
            key=lambda item: (
                max(
                    abs(int(item["x"]) - player_x),
                    abs(int(item["y"]) - player_y),
                ),
                int(item["x"]),
                int(item["y"]),
                active_id(item),
            ),
        )
        selected_id = active_id(target)
        identity = {}
        for field in ("seed_hi", "seed_lo", "create_info", "base_id"):
            if field not in target:
                raise RuntimeError(
                    f"controller snapshot {flag} target is missing {field}")
            identity[field] = DiabloGymEnv._controller_uint(
                target[field],
                name=f"{flag}.{field}",
                bits=16,
            )
        if flag == "heal":
            if "heal_kind" not in target:
                raise RuntimeError(
                    "controller snapshot heal target is missing heal_kind")
            heal_kind = DiabloGymEnv._controller_uint(
                target["heal_kind"], name="heal.heal_kind", bits=3)
            if not 1 <= heal_kind <= CONTROLLER_SNAPSHOT_INSTANT_HEAL_KINDS:
                raise RuntimeError(
                    "controller snapshot heal_kind must be an integer in [1,4]")
            gear_quantities: tuple[float, ...] = ()
            effect_flags = 0
            dam_ac_flags = 0
        else:
            missing = [
                field for field in CONTROLLER_SNAPSHOT_GEAR_FIELDS
                if field not in target
            ]
            if missing:
                raise RuntimeError(
                    "controller snapshot gear target is missing fields: "
                    + ",".join(missing))
            bool_fields = {
                "identified", "effects_active", "stat_usable"}
            gear_quantities = tuple(
                (
                    float(bool(target[field]))
                    if field in bool_fields else float(target[field])
                ) / scale
                for field, scale in zip(
                    CONTROLLER_SNAPSHOT_GEAR_FIELDS,
                    CONTROLLER_SNAPSHOT_GEAR_SCALES,
                    strict=True,
                )
            )
            if not all(math.isfinite(value) for value in gear_quantities):
                raise RuntimeError(
                    "controller snapshot gear exact values contain NaN/Inf")
            effect_flags = DiabloGymEnv._controller_uint(
                target.get("effect_flags"),
                name="gear.effect_flags",
                bits=CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS,
            )
            dam_ac_flags = DiabloGymEnv._controller_uint(
                target.get("effect_dam_ac_flags"),
                name="gear.effect_dam_ac_flags",
                bits=CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS,
            )
            heal_kind = 0
        return _ControllerItemTarget(
            x=int(target["x"]),
            y=int(target["y"]),
            active_id=selected_id,
            seed_hi=identity["seed_hi"],
            seed_lo=identity["seed_lo"],
            create_info=identity["create_info"],
            base_id=identity["base_id"],
            heal_kind=heal_kind,
            gear_quantities=gear_quantities,
            effect_flags=effect_flags,
            dam_ac_flags=dam_ac_flags,
        )

    def _capture_controller_snapshot(self, raw) -> _ControllerSnapshot:
        """Read the radius-12 controller state exactly once at a decision edge."""
        if not isinstance(raw, dict):
            raise RuntimeError("controller snapshot needs active native raw")
        radius = CONTROLLER_SNAPSHOT_RADIUS
        side = CONTROLLER_SNAPSHOT_SIDE
        local_map = bridge.local_map(radius=radius)
        walkable = self._controller_binary_channel(
            local_map.get("walkable", ()), "walkable")
        physical_monster = self._controller_binary_channel(
            local_map.get("monster", ()), "physical_monster")
        softwall = self._controller_binary_channel(
            local_map.get("door", ()), "softwall")
        closed_door = self._controller_binary_channel(
            local_map.get("closed_door", local_map.get("door", ())),
            "closed_door",
        )
        hazard = self._controller_binary_channel(
            local_map.get("hazard", ()), "hazard")
        explosive_softwall = self._controller_binary_channel(
            local_map.get("explosive_softwall", ()),
            "explosive_softwall",
        )

        px, py = int(raw["player_x"]), int(raw["player_y"])
        player_future_x = int(raw.get("future_x", px))
        player_future_y = int(raw.get("future_y", py))
        def in_window(point) -> bool:
            return (
                abs(int(point[0]) - px) <= radius
                and abs(int(point[1]) - py) <= radius
            )

        # planner and wire must consume exactly the same 25×25 facts. Keeping a one-tile visited/protected halo
        # outside the window would let near_visited inflation or the door unseen-side check make radius-13 history
        # change radius-12 edge actions without appearing in the observation.
        protected = frozenset(
            (int(x), int(y))
            for x, y in self._explore_protected_tiles(raw)
            if in_window((x, y))
        )
        visited = frozenset(
            (int(x), int(y))
            for x, y in getattr(self, "_visited", set())
            if in_window((x, y))
        )
        explore_blocked = frozenset(
            (int(x), int(y))
            for x, y in getattr(self, "_explore_blocked_targets", set())
            if in_window((x, y))
        )

        def tile_mask(tiles) -> tuple[int, ...]:
            return tuple(
                1 if (px + dx, py + dy) in tiles else 0
                for dy in range(-radius, radius + 1)
                for dx in range(-radius, radius + 1)
            )

        def map_index(x: int, y: int) -> int:
            return (y - py + radius) * side + (x - px + radius)

        raw_monsters = raw.get("monsters")
        if not isinstance(raw_monsters, (list, tuple)):
            raise RuntimeError(
                "controller snapshot is missing the native monsters list")
        visible_local_monsters: list[dict] = []
        for monster_index, monster in enumerate(raw_monsters):
            if not isinstance(monster, dict):
                raise RuntimeError(
                    "controller snapshot monsters"
                    f"[{monster_index}] must be a dict")
            missing_identity = [
                field for field in (
                    "id", "x", "y", "hp", "max_hp",
                    "visible", "reachable",
                    "rnd_item_seed_hi", "rnd_item_seed_lo",
                )
                if field not in monster
            ]
            if missing_identity:
                raise RuntimeError(
                    "controller snapshot monsters"
                    f"[{monster_index}] missing fields: "
                    + ",".join(missing_identity))
            # Hidden/native-inactive monsters may remain in the immutable raw
            # collision snapshot for macro planning, but neither identity nor
            # coordinates may enter an actor row or actor occupancy channel.
            if (
                bool(monster["visible"])
                and in_window((monster["x"], monster["y"]))
            ):
                visible_local_monsters.append(monster)

        visible_monster_values = [0] * CONTROLLER_SNAPSHOT_CELLS
        for monster in visible_local_monsters:
            visible_monster_values[
                map_index(int(monster["x"]), int(monster["y"]))
            ] = 1
        visible_monster = tuple(visible_monster_values)

        def local_reachable(
            *,
            allow_softwalls: bool,
            avoid_monsters: bool,
        ) -> frozenset[tuple[int, int]]:
            start = (px, py)
            reached = {start}
            queue = deque([start])
            while queue:
                cx, cy = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    point = (cx + dx, cy + dy)
                    if (
                        point in reached
                        or not in_window(point)
                        or point in protected
                    ):
                        continue
                    i = map_index(*point)
                    if hazard[i] or explosive_softwall[i]:
                        continue
                    if (
                        not walkable[i]
                        and not (allow_softwalls and softwall[i])
                    ):
                        continue
                    if avoid_monsters and physical_monster[i]:
                        continue
                    reached.add(point)
                    queue.append(point)
            return frozenset(reached)

        engage_reachable = local_reachable(
            allow_softwalls=False, avoid_monsters=True)
        pickup_reachable = local_reachable(
            allow_softwalls=True, avoid_monsters=False)

        def locally_engageable(monster) -> bool:
            target = (
                int(monster.get("future_x", monster["x"])),
                int(monster.get("future_y", monster["y"])),
            )
            # The native controller attack guard and _macro_engage both bind
            # the command to this observation window.  A monster may still
            # have its visible current tile on the radius-12 edge while its
            # already-committed future tile lies at radius 13; keep its row,
            # but do not advertise an action-9 candidate that execution must
            # reject.
            if not in_window(target):
                return False
            return any(
                max(abs(rx - target[0]), abs(ry - target[1])) <= 1
                for rx, ry in engage_reachable
            )

        ranked_monsters = [
            (monster, locally_engageable(monster))
            for monster in visible_local_monsters
        ]
        ranked_monsters.sort(key=lambda entry: (
            not entry[1],
            max(
                abs(
                    int(entry[0].get("future_x", entry[0]["x"]))
                    - player_future_x
                ),
                abs(
                    int(entry[0].get("future_y", entry[0]["y"]))
                    - player_future_y
                ),
            ),
            int(entry[0]["id"]),
        ))
        selected_monsters = ranked_monsters[
            :CONTROLLER_SNAPSHOT_MONSTER_LIMIT]
        overflow_monsters = ranked_monsters[
            CONTROLLER_SNAPSHOT_MONSTER_LIMIT:]

        blocked_keys = getattr(self, "_engage_blocked_keys", set())
        ledger = getattr(self, "_combat_hp_floor", {})
        monster_rows = []
        candidates = []
        for monster, is_locally_engageable in selected_monsters:
            monster_id = int(monster["id"])
            generation_key = self._monster_generation_key(monster)
            hp = max(0, int(monster["hp"]))
            max_hp = max(1, int(monster["max_hp"]))
            ledger_low, ledger_max = ledger.get(
                generation_key, (hp, max_hp))
            missing_dynamics = [
                field
                for field in (
                    CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS
                    + CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS
                )
                if field not in monster
            ]
            if missing_dynamics:
                raise RuntimeError(
                    "controller snapshot monster"
                    f"[{monster_id}] missing dynamic combat fields: "
                    + ",".join(missing_dynamics))
            dynamic_quantities_list = list(
                (
                    float(bool(monster[field]))
                    if field in {"anim_petrified", "is_invalid"}
                    else float(monster[field])
                ) / scale
                for field, scale in zip(
                    CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS,
                    CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_SCALES,
                    strict=True,
                )
            )
            for field in CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS:
                dynamic_quantities_list.extend(
                    self._controller_int32_words(
                        monster[field],
                        name=f"monster[{monster_id}].{field}",
                    )
                )
            dynamic_quantities = tuple(dynamic_quantities_list)
            if not all(
                math.isfinite(value) for value in dynamic_quantities
            ):
                raise RuntimeError(
                    "controller snapshot monster"
                    f"[{monster_id}] dynamic combat values contain NaN/Inf")
            combat_flags = self._controller_uint(
                monster.get("combat_flags"),
                name=f"monster[{monster_id}].combat_flags",
                bits=CONTROLLER_SNAPSHOT_MONSTER_FLAG_BITS,
            )
            row = _ControllerMonster(
                monster_id=monster_id,
                generation_key=generation_key,
                monster_type=int(monster.get("type", 0)),
                x=int(monster["x"]),
                y=int(monster["y"]),
                future_x=int(monster.get("future_x", monster["x"])),
                future_y=int(monster.get("future_y", monster["y"])),
                hp=hp,
                max_hp=max_hp,
                ledger_low=max(0, int(ledger_low)),
                ledger_max=max(1, int(ledger_max)),
                blocked=generation_key in blocked_keys,
                visible=True,
                native_reachable=bool(monster["reachable"]),
                locally_engageable=is_locally_engageable,
                dynamic_quantities=dynamic_quantities,
                combat_flags=combat_flags,
                ai=int(monster.get("ai", 0)),
                monster_level=int(monster.get("monster_level", 0)),
                max_damage=max(int(monster.get("max_damage", 0)),
                               int(monster.get("max_damage_special", 0))),
            )
            monster_rows.append(row)
            if is_locally_engageable:
                # Action 9 must never target a row that the actor did not
                # receive.  Engageable rows sort first, so this remains useful
                # even in pathological over-capacity scenes.
                candidates.append(row)

        overflow_values = (
            len(ranked_monsters),
            len(overflow_monsters),
            sum(is_engageable for _, is_engageable in ranked_monsters),
            sum(is_engageable for _, is_engageable in overflow_monsters),
            sum(max(0, int(monster["hp"]))
                for monster, _ in overflow_monsters),
            sum(max(1, int(monster["max_hp"]))
                for monster, _ in overflow_monsters),
            min(
                (
                    max(
                        abs(
                            int(monster.get("future_x", monster["x"]))
                            - player_future_x
                        ),
                        abs(
                            int(monster.get("future_y", monster["y"]))
                            - player_future_y
                        ),
                    )
                    for monster, _ in overflow_monsters
                ),
                default=0,
            ),
            max(
                (
                    max(
                        max(0, int(monster["max_damage"])),
                        max(0, int(monster["max_damage_special"])),
                    )
                    for monster, _ in overflow_monsters
                ),
                default=0,
            ),
        )
        monster_overflow_quantities = tuple(
            float(value) / scale
            for value, scale in zip(
                overflow_values,
                CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_SCALES,
                strict=True,
            )
        )

        raw_missiles = raw.get("missiles")
        if not isinstance(raw_missiles, (list, tuple)):
            raise RuntimeError(
                "controller snapshot is missing the native missiles list")
        missile_required = (
            CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS
            + CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS
        )
        missile_bool_fields = {
            "deleted", "draw", "light", "pre", "hit",
            "limit_reached", "hostile",
            "visible", "source_visible", "start_visible",
        }
        encoded_monster_ids = {
            entry.monster_id for entry in monster_rows}

        def missile_source_is_visible(missile) -> bool:
            if not bool(missile["source_visible"]):
                return False
            # Native MissileSource::Monster is 1.  A lit-but-out-of-window or
            # over-capacity source still has no actor entity row, so exposing
            # its engine index would be an identity side channel.
            return (
                int(missile["source_type"]) != 1
                or int(missile["source_id"]) in encoded_monster_ids
            )

        def missile_policy_value(missile, field):
            if field == "source_visible":
                return missile_source_is_visible(missile)
            if (
                field == "source_id"
                and not missile_source_is_visible(missile)
            ):
                return -1
            if (
                field in {"start_dx", "start_dy"}
                and not bool(missile["start_visible"])
            ):
                return 0
            return missile[field]

        local_missiles = []
        for missile_index, missile in enumerate(raw_missiles):
            if not isinstance(missile, dict):
                raise RuntimeError(
                    f"controller snapshot missiles[{missile_index}] must be a dict")
            missing = [
                field for field in missile_required
                if field not in missile
            ]
            if missing:
                raise RuntimeError(
                    f"controller snapshot missiles[{missile_index}] missing fields: "
                    + ",".join(missing))
            try:
                tile_dx = int(missile["tile_dx"])
                tile_dy = int(missile["tile_dy"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    "controller snapshot missile tile offset is not an integer") from exc
            if abs(tile_dx) > radius or abs(tile_dy) > radius:
                continue
            if not bool(missile["visible"]):
                continue
            local_missiles.append(missile)

        def missile_sort_key(missile):
            return (
                0 if bool(missile["hostile"]) else 1,
                max(abs(int(missile["tile_dx"])),
                    abs(int(missile["tile_dy"]))),
                max(0, int(missile["duration"])),
                int(missile["type"]),
                int(missile_policy_value(missile, "source_id")),
                tuple(
                    int(bool(missile[field]))
                    if field in missile_bool_fields
                    else int(missile_policy_value(missile, field))
                    for field in missile_required
                ),
            )

        local_missiles.sort(key=missile_sort_key)
        selected_missiles = local_missiles[
            :CONTROLLER_SNAPSHOT_MISSILE_LIMIT]
        overflow_missiles = local_missiles[
            CONTROLLER_SNAPSHOT_MISSILE_LIMIT:]
        missile_slots = []
        for missile_index, missile in enumerate(selected_missiles):
            quantities: list[float] = []
            for field, scale in zip(
                CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS,
                CONTROLLER_SNAPSHOT_MISSILE_DIRECT_SCALES,
                strict=True,
            ):
                value = (
                    float(bool(missile_policy_value(missile, field)))
                    if field in missile_bool_fields
                    else float(missile_policy_value(missile, field))
                )
                quantities.append(value / scale)
            for field in CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS:
                quantities.extend(self._controller_int32_words(
                    missile_policy_value(missile, field),
                    name=f"missile[{missile_index}].{field}",
                ))
            expected = CONTROLLER_SNAPSHOT_MISSILE_FIELDS - 1
            if (
                len(quantities) != expected
                or not all(math.isfinite(value) for value in quantities)
            ):
                raise RuntimeError(
                    "controller snapshot missile slot has an abnormal shape/finiteness")
            missile_slots.append(_ControllerMissile(tuple(quantities)))

        overflow_hostile = [
            missile for missile in overflow_missiles
            if bool(missile["hostile"])
        ]
        overflow_values = (
            len(local_missiles),
            len(overflow_missiles),
            sum(bool(missile["hostile"]) for missile in local_missiles),
            len(overflow_hostile),
            sum(abs(int(missile["damage"]))
                for missile in overflow_missiles),
            max(
                (abs(int(missile["damage"]))
                 for missile in overflow_missiles),
                default=0,
            ),
            min(
                (max(abs(int(missile["tile_dx"])),
                     abs(int(missile["tile_dy"])))
                 for missile in overflow_hostile),
                default=0,
            ),
            min(
                (max(0, int(missile["duration"]))
                 for missile in overflow_hostile),
                default=0,
            ),
            sum(bool(missile["deleted"]) for missile in local_missiles),
        )
        missile_overflow_quantities = tuple(
            float(value) / scale
            for value, scale in zip(
                overflow_values,
                CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_SCALES,
                strict=True,
            )
        )

        if "belt_slot_kinds" not in raw:
            raise RuntimeError(
                "controller snapshot is missing native belt_slot_kinds; "
                "cannot confuse belt empty/other or guess the action12 consumption order")
        try:
            belt_slot_kinds = tuple(
                int(kind) for kind in raw["belt_slot_kinds"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "controller snapshot belt_slot_kinds cannot be converted to integers") from exc
        if (
            len(belt_slot_kinds) != CONTROLLER_SNAPSHOT_BELT_SLOTS
            or any(
                kind < 0 or kind >= CONTROLLER_SNAPSHOT_BELT_KINDS
                for kind in belt_slot_kinds
            )
        ):
            raise RuntimeError(
                "controller snapshot belt_slot_kinds must be 8 integers in [0,5]")

        missing_exact = [
            field for field in CONTROLLER_SNAPSHOT_EXACT_FIELDS
            if field not in raw
        ]
        if missing_exact:
            raise RuntimeError(
                "controller snapshot is missing player/scene/quest exact values: "
                + ",".join(missing_exact))
        exact_quantities = tuple(
            (
                float(bool(raw[field]))
                if field in {
                    "is_set_level", "monotonic_quest_turn_in_used"}
                else float(raw[field])
            ) / scale
            for field, scale in zip(
                CONTROLLER_SNAPSHOT_EXACT_FIELDS,
                CONTROLLER_SNAPSHOT_EXACT_SCALES,
                strict=True,
            )
        )
        if not all(math.isfinite(value) for value in exact_quantities):
            raise RuntimeError("controller snapshot player/scene/quest exact values contain NaN/Inf")

        missing_combat = [
            field for field in CONTROLLER_SNAPSHOT_COMBAT_FIELDS
            if field not in raw
        ]
        if missing_combat:
            raise RuntimeError(
                "controller snapshot is missing currently effective combat values: "
                + ",".join(missing_combat))
        combat_quantities = tuple(
            (
                float(bool(raw[field]))
                if field == "block_enabled" else float(raw[field])
            ) / scale
            for field, scale in zip(
                CONTROLLER_SNAPSHOT_COMBAT_FIELDS,
                CONTROLLER_SNAPSHOT_COMBAT_SCALES,
                strict=True,
            )
        )
        if not all(math.isfinite(value) for value in combat_quantities):
            raise RuntimeError(
                "controller snapshot currently effective combat values contain NaN/Inf")
        combat_effect_flags = self._controller_uint(
            raw.get("item_effect_flags"),
            name="item_effect_flags",
            bits=CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS,
        )
        combat_dam_ac_flags = self._controller_uint(
            raw.get("item_dam_ac_flags"),
            name="item_dam_ac_flags",
            bits=CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS,
        )

        equipped_items = raw.get("equipped_items")
        if (
            not isinstance(equipped_items, (list, tuple))
            or len(equipped_items) != CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS
        ):
            raise RuntimeError(
                "controller snapshot equipped_items must have exactly 7 slots")
        equipped_quantities_list: list[float] = []
        equipped_bool_fields = {
            "identified", "effects_active", "stat_usable"}
        for slot_index, item in enumerate(equipped_items):
            if not isinstance(item, dict):
                raise RuntimeError(
                    "controller snapshot equipped_items"
                    f"[{slot_index}] must be a dict")
            missing = [
                field for field in (
                    ("present",)
                    + CONTROLLER_SNAPSHOT_GEAR_FIELDS
                    + ("effect_flags", "effect_dam_ac_flags")
                )
                if field not in item
            ]
            if missing:
                raise RuntimeError(
                    "controller snapshot equipped_items"
                    f"[{slot_index}] missing fields: " + ",".join(missing))
            equipped_quantities_list.append(
                1.0 if bool(item["present"]) else 0.0)
            equipped_quantities_list.extend(
                (
                    float(bool(item[field]))
                    if field in equipped_bool_fields else float(item[field])
                ) / scale
                for field, scale in zip(
                    CONTROLLER_SNAPSHOT_GEAR_FIELDS,
                    CONTROLLER_SNAPSHOT_GEAR_SCALES,
                    strict=True,
                )
            )
            effect_flags = self._controller_uint(
                item["effect_flags"],
                name=f"equipped[{slot_index}].effect_flags",
                bits=CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS,
            )
            dam_ac_flags = self._controller_uint(
                item["effect_dam_ac_flags"],
                name=f"equipped[{slot_index}].effect_dam_ac_flags",
                bits=CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS,
            )
            equipped_quantities_list.extend(
                1.0 if effect_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS)
            )
            equipped_quantities_list.extend(
                1.0 if dam_ac_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS)
            )
        equipped_quantities = tuple(equipped_quantities_list)
        if (
            len(equipped_quantities)
            != CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS
            * CONTROLLER_SNAPSHOT_EQUIPPED_FIELDS
            or not all(
                math.isfinite(value) for value in equipped_quantities)
        ):
            raise RuntimeError(
                "controller snapshot equipped_items exact values have an abnormal shape/finiteness")
        sticky = getattr(self, "_explore_target", None)
        sticky_target = (
            (int(sticky[0]), int(sticky[1]))
            if sticky is not None else None
        )
        return _ControllerSnapshot(
            raw_identity=id(raw),
            steps=int(getattr(self, "_steps", 0)),
            scene=_scene_identity(raw),
            player_x=px,
            player_y=py,
            player_future_x=player_future_x,
            player_future_y=player_future_y,
            walkable=walkable,
            visible_monster=visible_monster,
            physical_monster=physical_monster,
            softwall=softwall,
            closed_door=closed_door,
            hazard=hazard,
            explosive_softwall=explosive_softwall,
            visited_mask=tile_mask(visited),
            blocked_mask=tile_mask(explore_blocked),
            protected_mask=tile_mask(protected),
            visited_tiles=visited,
            explore_blocked_targets=explore_blocked,
            protected_tiles=protected,
            sticky_target=sticky_target,
            monsters=tuple(monster_rows),
            monster_overflow_quantities=monster_overflow_quantities,
            candidates=tuple(candidates),
            missiles=tuple(missile_slots),
            missile_overflow_quantities=missile_overflow_quantities,
            heal_target=self._controller_target(
                raw,
                "heal",
                player_x=px,
                player_y=py,
                reachable_tiles=pickup_reachable,
            ),
            gear_target=self._controller_target(
                raw,
                "gear",
                player_x=px,
                player_y=py,
                reachable_tiles=pickup_reachable,
            ),
            belt_slot_kinds=belt_slot_kinds,
            exact_quantities=exact_quantities,
            combat_quantities=combat_quantities,
            combat_effect_flags=combat_effect_flags,
            combat_dam_ac_flags=combat_dam_ac_flags,
            equipped_quantities=equipped_quantities,
        )

    def _controller_snapshot_for(self, raw) -> _ControllerSnapshot:
        """Return an already-installed decision snapshot without mutation."""
        snapshot = getattr(self, "_controller_snapshot", None)
        if (
            snapshot is None
            or snapshot.raw_identity != id(raw)
            or snapshot.steps != int(getattr(self, "_steps", 0))
            or snapshot.scene != _scene_identity(raw)
        ):
            raise RuntimeError(
                "controller snapshot not installed or expired; "
                "observation/mask reads must not implicitly fetch another map")
        return snapshot

    def controller_snapshot_vector(self) -> np.ndarray:
        """Return the exact fixed controller wire appended to the dual Worker."""
        self._ensure_active(allow_ended=True)
        snapshot = self._controller_snapshot_for(self._raw)
        values: list[float] = []
        softwall_kind = tuple(
            (
                softwall
                + 2 * closed_door
                + 4 * explosive
            ) / CONTROLLER_SNAPSHOT_SOFTWALL_KIND_DENOMINATOR
            for softwall, closed_door, explosive in zip(
                snapshot.softwall,
                snapshot.closed_door,
                snapshot.explosive_softwall,
                strict=True,
            )
        )
        for channel in (
            snapshot.walkable,
            snapshot.visible_monster,
            softwall_kind,
            snapshot.visited_mask,
            snapshot.blocked_mask,
            snapshot.protected_mask,
            snapshot.hazard,
        ):
            values.extend(float(value) for value in channel)
        for candidate in snapshot.monsters:
            values.extend((
                1.0,
                float(candidate.monster_id) / 200.0,
                float(candidate.monster_type) / 200.0,
                float(candidate.x - snapshot.player_x) / 112.0,
                float(candidate.y - snapshot.player_y) / 112.0,
                float(candidate.future_x - snapshot.player_future_x) / 112.0,
                float(candidate.future_y - snapshot.player_future_y) / 112.0,
                float(candidate.hp) / 1024.0,
                float(candidate.max_hp) / 1024.0,
                float(candidate.ledger_low) / 1024.0,
                float(candidate.ledger_max) / 1024.0,
                1.0 if candidate.blocked else 0.0,
                1.0 if candidate.visible else 0.0,
                1.0 if candidate.native_reachable else 0.0,
                1.0 if candidate.locally_engageable else 0.0,
            ))
            values.extend(candidate.dynamic_quantities)
            values.extend(
                1.0 if candidate.combat_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_MONSTER_FLAG_BITS)
            )
        values.extend(
            [0.0] * (
                CONTROLLER_SNAPSHOT_MONSTER_DIM
                - CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM
                - len(snapshot.monsters)
                * CONTROLLER_SNAPSHOT_MONSTER_FIELDS
            )
        )
        values.extend(snapshot.monster_overflow_quantities)
        for missile in snapshot.missiles:
            values.append(1.0)
            values.extend(missile.quantities)
        values.extend(
            [0.0] * (
                CONTROLLER_SNAPSHOT_MISSILE_LIMIT
                - len(snapshot.missiles)
            ) * CONTROLLER_SNAPSHOT_MISSILE_FIELDS
        )
        values.extend(snapshot.missile_overflow_quantities)
        for kind in snapshot.belt_slot_kinds:
            values.extend(
                1.0 if kind == expected else 0.0
                for expected in range(CONTROLLER_SNAPSHOT_BELT_KINDS)
            )
        values.extend(snapshot.exact_quantities)
        values.extend(snapshot.combat_quantities)
        values.extend(
            1.0 if snapshot.combat_effect_flags & (1 << bit) else 0.0
            for bit in range(CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS)
        )
        values.extend(
            1.0 if snapshot.combat_dam_ac_flags & (1 << bit) else 0.0
            for bit in range(CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS)
        )
        values.extend(snapshot.equipped_quantities)
        if snapshot.sticky_target is None:
            values.extend((0.0, 0.0, 0.0))
        else:
            values.extend((
                1.0,
                float(snapshot.sticky_target[0] - snapshot.player_x) / 112.0,
                float(snapshot.sticky_target[1] - snapshot.player_y) / 112.0,
            ))
        heal_target = snapshot.heal_target
        if heal_target is None:
            values.extend([0.0] * CONTROLLER_SNAPSHOT_HEAL_TARGET_DIM)
        else:
            values.extend((
                1.0,
                float(heal_target.x - snapshot.player_x) / 112.0,
                float(heal_target.y - snapshot.player_y) / 112.0,
                float(heal_target.active_id) / 128.0,
                float(heal_target.heal_kind)
                / CONTROLLER_SNAPSHOT_INSTANT_HEAL_KINDS,
            ))

        gear_target = snapshot.gear_target
        if gear_target is None:
            values.extend([0.0] * CONTROLLER_SNAPSHOT_GEAR_TARGET_DIM)
        else:
            values.extend((
                1.0,
                float(gear_target.x - snapshot.player_x) / 112.0,
                float(gear_target.y - snapshot.player_y) / 112.0,
            ))
            values.extend(gear_target.gear_quantities)
            values.extend(
                1.0 if gear_target.effect_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_EFFECT_FLAG_BITS)
            )
            values.extend(
                1.0 if gear_target.dam_ac_flags & (1 << bit) else 0.0
                for bit in range(CONTROLLER_SNAPSHOT_DAM_AC_FLAG_BITS)
            )
        result = np.asarray(values, dtype=np.float32)
        if (
            result.shape != (CONTROLLER_SNAPSHOT_VECTOR_DIM,)
            or not np.isfinite(result).all()
        ):
            raise RuntimeError(
                "controller snapshot wire shape/finiteness drift: "
                f"shape={result.shape},finite={np.isfinite(result).all()}")
        return result

    @property
    def exploration_progress(self) -> int:
        """Monotone count of first-visited tiles this episode + softwalls opened by action10."""
        return int(self._exploration_progress)

    @property
    def softwalls_opened(self) -> int:
        """Number of closed doors / blocking barrels actually opened by action10 this episode."""
        return int(self._softwalls_opened)

    def action_masks(self) -> np.ndarray:
        """v4 invalid-action mask (MaskablePPO protocol method).

        9 is legal only when the radius-12 snapshot holds an encoded monster that is visible and locally engageable;
        each row also publishes native reachable separately, and the two kinds of reachability must not be merged into one fact.
        12 requires a belt heal potion and actual missing HP; 13 requires a free belt slot and a valid
        heal potion target in the snapshot; 14 requires a strict whole-set upgrade approved by PlanGearUpgrade
        in the snapshot. Other actions stay legal. The mask removes deterministic empty presses, but does not promise that a macro
        necessarily finishes despite dynamic monster blocking, hits taken or the time limit.

        This is a behavioural semantic break of protocol-v4: even with identical action/observation shapes, old policies
        have their logits renormalized, must be re-baselined, and cannot be mixed with the v3 leaderboard.
        """
        self._ensure_active(allow_ended=True)
        mask, _nearest = self.controller_action_context()
        return mask

    def _decision_controller_snapshot(self, raw) -> _ControllerSnapshot:
        """Return the exact snapshot that a controller action would execute.

        The dual view installs its snapshot while producing the observation;
        silently recapturing it here would break observation/action atomicity.
        Legacy/flat views do not expose the controller wire, so they capture a
        read-only snapshot at the mask edge, exactly as ``step`` does.
        """
        snapshot = getattr(self, "_controller_snapshot", None)
        snapshot_is_current = (
            snapshot is not None
            and snapshot.raw_identity == id(raw)
            and snapshot.steps == int(getattr(self, "_steps", 0))
            and snapshot.scene == _scene_identity(raw)
        )
        if snapshot_is_current:
            return snapshot
        if getattr(self, "_controller_snapshot_enabled", False):
            return self._controller_snapshot_for(raw)
        return self._capture_controller_snapshot(raw)

    def controller_action_context(
        self,
    ) -> tuple[np.ndarray, int | None]:
        """Exact action mask plus nearest locally engageable monster distance.

        Scripted teachers need the distance from the same radius-12 candidate
        set as action 9.  Using the global raw monster list here made a hidden
        or locally blocked monster suppress DIVE handoff and generate labels
        for a different target than the macro could execute.
        """
        self._ensure_active(allow_ended=True)
        mask = np.ones(15, dtype=bool)
        raw = self._raw
        snapshot = self._decision_controller_snapshot(raw)
        # Flat/base policies receive the same hard authority boundary as the
        # hierarchical Worker.  A nominal adjacent CMD_WALKXY may otherwise
        # re-plan around a blocked edge and step on a stair/quest trigger.
        if getattr(self, "resource_protocol", "off") != "off":
            from .resource_protocol import progression_allowed
            protected = self._resource_direction_protected_tiles(raw)
            px, py = int(raw["player_x"]), int(raw["player_y"])
            for action, (dx, dy) in enumerate(_DIRS, start=1):
                if (px + dx, py + dy) in protected:
                    mask[action] = False
            mask[11] = progression_allowed(
                raw, getattr(self, "resource_readiness_law", "veto-v1"))
        else:
            for action in self._protected_walk_actions(raw):
                mask[action] = False
        mask[9] = bool(snapshot.candidates)
        if (getattr(self, "aggro_cap", "off") != "off" and mask[9]
                and int(raw.get("dungeon_level") or 0) >= 2 and not raw.get("is_set_level")):
            # v1 scope: main L2+ only, so the L1 prefix stays byte-identical to the
            # retreat arm and paired comparisons isolate the L2 effect.
            # R18-D hold-v1: a crowd is on us and we can fight it -> no new pulls
            # (a10 masked). The mask[9] guard keeps a fight available: no deadlock.
            from .aggro_cap import cap_fires
            if cap_fires(raw, self._aggro_cap_policy, mask[9]):
                mask[10] = False
                self._aggro_cap_fired += 1  # mask-build hits (several per step), not steps
        mask[12] = (
            int(raw.get("belt_heals", 0)) > 0
            and int(raw.get("hp", 0)) < int(raw.get("max_hp", 0))
        )
        mask[13] = (
            self._belt_free_slots(raw) > 0
            and snapshot.heal_target is not None
        )
        mask[14] = snapshot.gear_target is not None
        nearest = min(
            (
                max(
                    abs(candidate.future_x - snapshot.player_future_x),
                    abs(candidate.future_y - snapshot.player_future_y),
                )
                for candidate in snapshot.candidates
            ),
            default=None,
        )
        return mask, nearest

    def controller_action_masks(self) -> np.ndarray:
        """Backward-compatible name for the now-canonical exact mask."""
        return self.action_masks()

    def _configure_native_resource_protocol(self):
        enabled = getattr(self, "resource_protocol", "off") != "off"
        # R18-H sweep-v1: the raw "objects" channel. An env that never asks for
        # it makes NO native call here (the off path stays byte-identical); an
        # env that asked once is also responsible for turning it back off, so a
        # later sweep-off env in the same process sees the frozen dict again.
        sweep = getattr(self, "resource_sweep", "off") != "off"
        if sweep or DiabloGymEnv._sweep_object_channel:
            if not hasattr(bridge, "configure_object_observation"):
                raise RuntimeError("Native bridge lacks the object observation channel; rebuild required")
            bridge.end_game()
            bridge.configure_object_observation(sweep)
            # R18-M (2026-09-07) review round: written on the CLASS, so the next
            # env in this process sees what the bridge is actually doing.  The
            # off path is untouched (both operands false ⇒ no native call), and
            # a process that only ever runs sweep-on envs takes exactly the same
            # branch as before, so H1's tested arms are byte-for-byte unchanged.
            DiabloGymEnv._sweep_object_channel = sweep
        if hasattr(bridge, "configure_resource_protocol"):
            bridge.end_game()
            advisory = getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03"
            loot = bool(getattr(self, "resource_loot_economy", False))
            if loot and not advisory:
                # R17.1-D loot economy, exact R21 ABI (no advisory keyword).
                bridge.configure_resource_protocol(enabled, ordinary_armor_scope=True,
                    preserve_equipment_readiness=True, loot_economy=True,
                    **({"identify": True} if getattr(self, "resource_identify", "off") != "off" else {}))
            elif advisory:
                # R17.1 readiness-rule-3 advisory law (optionally with the loot economy):
                # explicit full-keyword call, needs a bridge built after both.
                bridge.configure_resource_protocol(enabled,
                    ordinary_armor_scope=(True if loot else getattr(
                        self, "resource_ordinary_armor_scope", False)),
                    preserve_equipment_readiness=(True if loot else getattr(
                        self, "resource_preserve_equipment_readiness", False)),
                    loot_economy=loot,
                    readiness_advisory=True,
                    **({"retreat": True} if getattr(self, "resource_retreat", "off") != "off" else {}),
                    **({"portal": True} if getattr(self, "resource_portal", "off") != "off" else {}),
                    **({"identify": True} if getattr(self, "resource_identify", "off") != "off" else {}))
            elif getattr(self, "resource_preserve_equipment_readiness", False):
                bridge.configure_resource_protocol(enabled,
                    ordinary_armor_scope=getattr(self, "resource_ordinary_armor_scope", False),
                    preserve_equipment_readiness=True)
            elif getattr(self, "resource_ordinary_armor_scope", False):
                bridge.configure_resource_protocol(enabled, ordinary_armor_scope=True)
            else:
                # Preserve the original one-argument ABI for every old policy.
                bridge.configure_resource_protocol(enabled)
        elif enabled:
            raise RuntimeError("Native bridge lacks the resource protocol; rebuild required")
        # R18-K2b weapon-purchase-v1 (2026-09-07): a SEPARATE native switch, so
        # every frozen configure_resource_protocol ABI above is untouched.
        # R18-K2b review round: written on EVERY configure, in both directions,
        # exactly like retreat/portal -- the flag is process-wide and nothing
        # else clears it, so an off arm constructed after a smith-v1 arm in the
        # same worker used to inherit the leaked True and hard-fail forever.
        # A bridge that predates the entry point still runs every off arm;
        # smith-v1 fails closed there and again at the per-observation
        # handshake in _validate_native_resource_flags.
        from .resource_weapon_upgrade import configure_native_weapon_purchase
        configure_native_weapon_purchase(
            bridge, getattr(self, "resource_weapon_upgrade", "off"))

    def _step_native(self):
        raw = bridge.step(ticks=self.ticks_per_step)
        if getattr(self, "resource_protocol", "off") != "off":
            self._resource_actual_microsteps += 1
        DiabloGymEnv._validate_native_resource_flags(self, raw)
        if getattr(self, "resource_protocol", "off") != "off":
            time_callback = getattr(self, "resource_time_callback", None)
            if time_callback is not None:
                time_callback(self, raw, self._resource_actual_microsteps)
            observer = getattr(self, "resource_observation_callback", None)
            if observer is not None:
                observer(self, raw, self._resource_actual_microsteps)
            callback = getattr(self, "resource_tick_callback", None)
            if callback is not None:
                callback(self, raw, self._resource_actual_microsteps)
        return raw

    def step_resource(self, command):
        """Execute a script-owned service command through normal env accounting.

        The internal wait action number is never a Worker-labelled transition;
        resource_action_audit identifies the actual source and command.
        """
        if getattr(self, "resource_protocol", "off") == "off":
            raise RuntimeError("Resource command requires l2-town-v1")
        if self._resource_pending_command is not None:
            raise RuntimeError("Nested resource command")
        self._resource_pending_command = tuple(command)
        try:
            return self.step(0)
        finally:
            self._resource_pending_command = None

    def _execute_resource_command(self, command):
        kind, *args = command
        receipt = {"accepted": False, "reason": kind, "price": 0}
        if kind in ("finish", "complete"):
            if kind == "finish":
                self._resource_terminal_reason = str(args[0])
            return self._raw, 0, receipt
        if kind in ("loot", "sell"):
            if not getattr(self, "resource_loot_economy", False):
                raise ValueError("Loot pickup and sale require the explicit loot economy protocol")
            # A refused pre-dispatch request still carries its real wallet and
            # requested identity so the service can seal an honest zero-action
            # ledger. It is explicitly not a native transaction receipt.
            receipt.update(received=0, native_executed=False,
                           source="environment-precheck", gold_before=int(self._raw.get("gold", 0)),
                           gold_after=int(self._raw.get("gold", 0)))
            if len(args) == 7:
                identity = args[2:6] if kind == "sell" else args[3:7]
                receipt.update(zip(("seed_hi", "seed_lo", "create_info", "base_id"), identity))
                if kind == "sell":
                    receipt.update(vendor=args[0], index=args[1], quoted_price=args[6])
                else:
                    receipt.update(active_id=args[0], index=-1)
            deadline = min(self.max_steps, int(getattr(
                self, "_resource_service_deadline", self.max_steps)))
            if self._resource_actual_microsteps >= deadline:
                receipt["reason"] = "resource_command_deadline"
                return self._raw, 0, receipt
            if (self._raw.get("dead") or self._raw.get("game_over") or self._raw.get("victory")
                    or int(self._raw.get("hp", 0)) <= 0):
                receipt["reason"] = "resource_command_terminal"
                return self._raw, 0, receipt
        if kind in ("attack_monster", "drink", "unequip"):
            # New script-only commands. Keep the legacy service dispatch below
            # untouched; every mutation here is bounded by actual native clocks.
            if kind == "unequip" and self.resource_purchase_mode != "full":
                raise ValueError("Unequip is restricted to the full resource arm")
            if kind == "attack_monster":
                if (len(args) != 2 or type(args[1]) is not int or args[1] <= 0):
                    raise ValueError("attack_monster requires id and positive microstep budget")
                command_budget = min(12, args[1])
            else:
                command_budget = 1
            clock_before = self._resource_actual_microsteps
            fixed_deadline = clock_before + command_budget

            def time_available():
                return self._resource_actual_microsteps < min(
                    fixed_deadline, self.max_steps,
                    int(getattr(self, "_resource_service_deadline", self.max_steps)))

            if not time_available():
                receipt["reason"] = "resource_command_deadline"
                return self._raw, 0, receipt
            before = self._raw
            if (before.get("dead") or before.get("game_over") or before.get("victory")
                    or int(before.get("hp", 0)) <= 0):
                receipt["reason"] = "resource_command_terminal"
                return before, 0, receipt
            start_scene = _scene_identity(before)
            px, py = int(before["player_x"]), int(before["player_y"])
            if kind == "attack_monster":
                target_before = next((monster for monster in before.get("monsters", ())
                                      if int(monster.get("id", -1)) == int(args[0])
                                      and bool(monster.get("visible", False))
                                      and not bool(monster.get("is_invalid", False))
                                      and int(monster.get("type", -1)) != 109
                                      and int(monster.get("hp", 0)) > 0), None)
                target_key = (None if target_before is None else tuple(
                    target_before.get(name) for name in
                    ("id", "type", "rnd_item_seed_hi", "rnd_item_seed_lo")))
                adjacent = target_before is not None and max(
                    abs(int(target_before.get("future_x", target_before["x"]))
                        - int(before.get("future_x", px))),
                    abs(int(target_before.get("future_y", target_before["y"]))
                        - int(before.get("future_y", py)))) <= 1
                result = (bridge.act_controller_attack_monster(int(args[0]), px, py, 1)
                          if adjacent else 0)
                receipt["accepted"] = int(result) == 1
                receipt["monster_id"] = int(args[0])
                receipt["microstep_budget"] = command_budget
            elif kind == "drink":
                kinds = before.get("belt_heal_kinds")
                belt_before = (sum(int(value) in (1, 2, 3, 4) for value in kinds)
                               if kinds is not None else int(before.get(
                                   "resource_state", {}).get("readiness", {}).get("belt_heals", 0)))
                # Same FIFO fence and certified return convention as action12:
                # the result is zero or the PRE-drink belt count, not bool(1).
                bridge.act_wait()
                result = bridge.act_drink()
                if not isinstance(result, (bool, np.bool_, int, np.integer)):
                    raise RuntimeError("resource drink native receipt must be an integer")
                accepted = int(result)
                if accepted not in (0, belt_before):
                    raise RuntimeError("resource drink native receipt disagrees with pre-drink belt")
                receipt.update(accepted=accepted > 0, accepted_belt_before=accepted,
                               belt_before=belt_before, hp_before=int(before.get("hp", 0)),
                               consumed=accepted > 0)
            else:
                result = bridge.act_unequip_equipped_item(*args)
                if not isinstance(result, dict):
                    raise RuntimeError("resource unequip native receipt must be a dictionary")
                receipt = dict(result)
            if not receipt.get("accepted") and kind != "drink":
                bridge.act_wait()
            raw = self._step_native()
            beats = 1
            if kind == "attack_monster" and receipt.get("accepted"):
                # Do not cancel a swing after its first native beat. Preserve
                # its ordinary animation, but never chase an out-of-range or
                # no-longer-visible target and never cross a live deadline.
                while (time_available()
                       and not (raw.get("dead") or raw.get("game_over") or raw.get("victory"))
                       and _scene_identity(raw) == start_scene
                       and not self._decision_idle(raw)):
                    target = next((monster for monster in raw.get("monsters", ())
                                   if int(monster.get("id", -1)) == int(args[0])
                                   and bool(monster.get("visible", False))
                                   and int(monster.get("hp", 0)) > 0), None)
                    if (target is None or tuple(target.get(name) for name in
                            ("id", "type", "rnd_item_seed_hi", "rnd_item_seed_lo")) != target_key):
                        break
                    if max(
                            abs(int(target.get("future_x", target["x"]))
                                - int(raw.get("future_x", raw["player_x"]))),
                            abs(int(target.get("future_y", target["y"]))
                                - int(raw.get("future_y", raw["player_y"])))) > 1:
                        break
                    raw = self._step_native()
                    beats += 1
            if kind == "drink":
                kinds = raw.get("belt_heal_kinds")
                belt_after = (sum(int(value) in (1, 2, 3, 4) for value in kinds)
                              if kinds is not None else int(raw.get(
                                  "resource_state", {}).get("readiness", {}).get("belt_heals", 0)))
                hp_after = int(raw.get("hp", 0))
                receipt.update(belt_after=belt_after, hp_after=hp_after,
                               belt_consumed_observed=belt_after < receipt["belt_before"],
                               post_tick_effect_visible=(belt_after < receipt["belt_before"]
                                                         or hp_after > receipt["hp_before"]))
                # Native ActDrink already certifies synchronous consumption or
                # HP increase. An intervening attack/auto-refill may hide that
                # effect in the net post-tick delta; retain both facts explicitly.
            return raw, beats, receipt
        px, py = int(self._raw["player_x"]), int(self._raw["player_y"])
        if kind == "walk":
            result = bridge.act_explore_walk(*args, (), px, py, self._DESCEND_RADIUS)
        elif kind == "open":
            result = bridge.act_controller_operate(*args, px, py, self._DESCEND_RADIUS)
        elif kind == "gold":
            result = bridge.act_pickup_gold_at(*args)
        elif kind in ("loot", "sell"):
            if not getattr(self, "resource_loot_economy", False):
                raise ValueError("Loot pickup and sale require the explicit loot economy protocol")
            operation = bridge.act_pickup_loot_at if kind == "loot" else bridge.act_sell_inventory_item
            result = operation(*args)
            if not isinstance(result, dict):
                raise RuntimeError("Loot economy operation requires a real native receipt")
        elif kind == "talk":
            result = bridge.act_talk_towner(*args)
        elif kind == "dismiss":
            result = bridge.act_dismiss_dialog()
        elif kind == "buy":
            result = bridge.act_buy_store_item(*args)
        elif kind == "cast_portal":
            # R18-F portal-v1. The native action emits CMD_SPELLXY directly;
            # routing a targeted scroll through UseInvItem would strand the
            # cursor in CURSOR_TELEPORT and silently disable every later drink.
            if getattr(self, "resource_portal", "off") == "off":
                raise ValueError("Reading a Scroll of Town Portal requires portal-v1")
            result = bridge.act_cast_town_portal(*args)
            if not isinstance(result, dict):
                raise RuntimeError("Portal cast requires a real native receipt")
        elif kind == "equip":
            result = bridge.act_equip_inventory_item(*args)
        elif kind == "repair":
            if self.resource_purchase_mode != "full":
                raise ValueError("Paid repair is restricted to the full resource arm")
            result = bridge.act_repair_equipped_item(*args)
        elif kind == "identify":
            # R18-H identify-v1. Cain's fixed-fee identify transaction; the native
            # action needs no store page (TryIdentifyItem reads no UI state), only
            # a real adjacency to the storyteller inside an authorized town trip.
            if getattr(self, "resource_identify", "off") == "off":
                raise ValueError("Paying Cain to identify an item requires identify-v1")
            result = bridge.act_identify(*args)
            if not isinstance(result, dict):
                raise RuntimeError("Identification requires a real native receipt")
        elif kind == "hold":
            # R18-F portal-v1: advance the engine without the wait/cancel semantics
            # (act_wait calls player.Stop and StartStand on PM_SPELL, which kills a
            # queued Scroll of Town Portal before its animation fires). Script-only.
            if getattr(self, "resource_portal", "off") == "off":
                raise ValueError("The hold command requires portal-v1")
            result = 1
        elif kind == "wait":
            result = bridge.act_wait()
        else:
            raise ValueError(f"Unknown resource command: {kind!r}")
        if isinstance(result, dict):
            receipt = dict(result)
        else:
            receipt["accepted"] = int(result) == 1
        # Rejections still advance a real, explicit wait. This avoids zero-time
        # service retry loops while never inventing a phantom microstep.
        if not receipt.get("accepted"):
            bridge.act_wait()
        clock_before = self._resource_actual_microsteps
        start_scene = _scene_identity(self._raw)
        raw = self._step_native()
        beats = 1
        if kind == "cast_portal" and receipt.get("accepted"):
            # R18-F portal-v1: the queued Scroll of Town Portal must run to its cast
            # frame BEFORE the generic settle fence (_settle_to_idle cancels intent
            # with act_wait). Mirrors the accepted-open handling: step until the
            # portal exists or the spell animation (PM_SPELL=9) has returned to
            # PM_STAND, bounded by the service deadline and 16 beats.
            deadline = min(self.max_steps,
                           int(getattr(self, "_resource_service_deadline", self.max_steps)),
                           clock_before + 16)
            seen_spell = False
            while (self._resource_actual_microsteps < deadline
                   and not (raw.get("dead") or raw.get("game_over") or raw.get("victory"))
                   and _scene_identity(raw) == start_scene):
                mode = int(raw.get("player_mode", 0) or 0)
                seen_spell = seen_spell or mode == 9
                state = raw.get("resource_state", {}) or {}
                if state.get("portal_open") or any(
                        int(m.get("type", -1)) == 10 for m in raw.get("missiles", ())):
                    break
                if seen_spell and mode == 0:
                    break
                raw = self._step_native()
                beats += 1
        if kind == "open" and receipt.get("accepted"):
            # Non-explosive barrels share the planner's softwall channel.
            # Let the accepted native operation finish its attack animation
            # before the generic settle fence cancels further intent. Without
            # this, a one-beat command starts a swing and immediately cancels
            # it before impact on every retry (seed 2114004, object type 57).
            deadline = min(self.max_steps,
                           int(getattr(self, "_resource_service_deadline", self.max_steps)),
                           clock_before + 12)
            while (self._resource_actual_microsteps < deadline
                   and not (raw.get("dead") or raw.get("game_over") or raw.get("victory"))
                   and _scene_identity(raw) == start_scene
                   and not bool(bridge.probe_tile(int(args[0]), int(args[1]))["walkable"])
                   and not self._decision_idle(raw)
                   and (getattr(self, "_resource_calibration", None) is None
                        or self._resource_actual_microsteps < self.max_steps)):
                raw = self._step_native()
                beats += 1
        return raw, beats, receipt

    def step(self, action: int, *, worker_authority: bool = False):
        self._ensure_active()
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in {self.action_space}, got {action!r}")
        prev = self._raw
        resource_command = getattr(self, "_resource_pending_command", None)
        resource_receipt = None
        resource_clock_before = getattr(self, "_resource_actual_microsteps", 0)
        self._resource_reward_depth_before = getattr(self, "_resource_max_main_depth", 0)
        self._resource_step_bonus = 0.0
        action = int(action)
        if getattr(self, "dive_blocker_recovery", "off") != "off":
            self._dive_blocker_audit = None  # receipt is strictly per action
        remaining = self.max_steps - self._steps
        if resource_command is not None:
            remaining = min(remaining, max(0, int(getattr(
                self, "_resource_service_deadline", self.max_steps)) - self._resource_actual_microsteps))
            if resource_command[0] == "attack_monster":
                if (len(resource_command) != 3 or type(resource_command[2]) is not int
                        or resource_command[2] <= 0):
                    raise ValueError("attack_monster requires id and positive microstep budget")
                remaining = min(remaining, 12, resource_command[2])
            elif resource_command[0] in ("drink", "unequip"):
                remaining = min(remaining, 1)
        drink_audit = None
        action14_audit = None
        native_execution = {"attempts": 0, "accepts": 0}
        exploration_before = int(
            getattr(self, "_exploration_progress", 0))
        softwalls_before = int(
            getattr(self, "_softwalls_opened", 0))
        controller_snapshot = None
        engage_target_generation_key = None
        if action in {9, 10, 13, 14}:
            # The dual Worker must execute the installed snapshot it just saw; the old 295/298
            # views do not publish the controller wire, so one local fetch at the key-press boundary is enough, and
            # it is not cached, keeping action_masks/ordinary observations pure reads and the old mode light.
            controller_snapshot = (
                self._controller_snapshot_for(prev)
                if getattr(self, "_controller_snapshot_enabled", False)
                else self._capture_controller_snapshot(prev)
            )
        if resource_command is not None:
            self._raw, micro, resource_receipt = self._execute_resource_command(resource_command)
            if resource_command[0] not in ("finish", "complete"):
                native_execution["attempts"] = 1
                native_execution["accepts"] = int(bool(resource_receipt.get("accepted")))
        elif action == 9:
            engage_candidate = self._engage_candidate_for_action9(
                controller_snapshot)
            if engage_candidate is not None:
                engage_target_generation_key = (
                    engage_candidate.generation_key)
            self._raw, micro = self._macro_engage(
                max_beats=min(10, remaining),
                controller_snapshot=controller_snapshot,
                execution_audit=native_execution,
            )
        elif action == 10:
            self._raw, micro = self._macro_explore(
                max_beats=min(12, remaining),
                controller_snapshot=controller_snapshot,
                execution_audit=native_execution,
            )
        elif action == 11:
            self._raw, micro = self._macro_descend(
                max_beats=min(12, remaining),
                execution_audit=native_execution,
            )
        elif action == 12:
            # Drinking itself does not go through the network command layer, so a wait fence must first cancel any chase
            # the previous action may have left; otherwise the potion key's tick would keep walking/attacking and swallow its reward.
            belt_before = int(prev.get("belt_heals", 0))
            bridge.act_wait()
            accepted_raw = bridge.act_drink()
            if not isinstance(
                    accepted_raw, (bool, np.bool_, int, np.integer)):
                raise RuntimeError(
                    "action12 native receipt must be an integer, got "
                    f"{accepted_raw!r}")
            accepted = int(accepted_raw)
            if accepted not in (0, belt_before):
                raise RuntimeError(
                    "action12 native receipt disagrees with the pre-request belt count: "
                    f"accepted={accepted},before={belt_before}")
            self._raw = self._step_native()
            micro = 1
            # ``act_drink`` can be rejected while the player is in hit/block
            # recovery even though the visible hp/belt predicate is true.
            # A requested key is not an executed drink until a bottle really
            # leaves the belt.  Keep this audit through the later settle so
            # wrappers never count failed emergency attempts as rescues.
            drink_audit = {
                "accepted": accepted > 0,
                "accepted_belt_before": accepted,
                "belt_before": belt_before,
                "consumed": accepted > 0,
            }
            self._record_native_execution(
                native_execution,
                1 if accepted > 0 else 0,
                "action12 drink",
            )
        elif action == 13:
            self._raw, micro = self._macro_pickup(
                "heal",
                max_beats=min(12, remaining),
                controller_snapshot=controller_snapshot,
                execution_audit=native_execution,
            )
        elif action == 14:
            utility = gear_combat_utility_value(
                prev, "action14_request")
            action14_audit = {
                "accepted": False,
                "commit_attempts": 0,
                "utility_before": utility,
                "utility_after": utility,
                "utility_delta": 0,
            }
            self._raw, micro = self._macro_pickup(
                "gear",
                max_beats=min(12, remaining),
                controller_snapshot=controller_snapshot,
                action14_audit=action14_audit,
                execution_audit=native_execution,
            )
        else:
            accepted = self._apply_action(
                action, worker_authority=bool(worker_authority))
            if action != 0:
                self._record_native_execution(
                    native_execution, accepted, "direction")
            self._raw = self._step_native()
            micro = 1
        dive_recovery_audit = getattr(self, "_dive_blocker_audit", None)
        if dive_recovery_audit is not None:
            dive_recovery_audit["attack_core_micro_steps"] = dive_recovery_audit["micro_steps"]
            dive_recovery_audit["core_micro_steps"] = self._resource_actual_microsteps - resource_clock_before
            dive_recovery_audit["core_after_microstep"] = self._resource_actual_microsteps
        # The 295-dim policy observation has no future/mode/path. Every non-terminal decision boundary must settle
        # the single-tile/hit-stun animation already committed by this action to PM_STAND; these settle ticks
        # belong to this action and use up max_steps, the reward difference and the Options τ/clock as usual.
        if resource_command is None or resource_command[0] not in ("finish", "complete"):
            self._raw, micro = self._settle_to_idle(
                self._raw,
                micro,
                max_beats=remaining,
                start_scene=_scene_identity(prev),
            )
        if getattr(self, "resource_protocol", "off") != "off":
            micro = self._resource_actual_microsteps - resource_clock_before
        authorized_retreat = (
            getattr(self, "resource_protocol", "off") != "off"
            and int(prev["dungeon_level"]) == 1
            and int(self._raw["dungeon_level"]) == 0
            and bool(self._raw.get("resource_state", {}).get("service_trip")))
        if (not authorized_retreat and getattr(self, "resource_retreat", "off") != "off"
                and int(prev["dungeon_level"]) >= 2
                and int(self._raw["dungeon_level"]) == int(prev["dungeon_level"]) - 1):
            # R18-A retreat-v1: a one-floor ascent is legal only with an accepted
            # engine receipt; anything else stays the fail-closed invariant below.
            receipt = self._raw.get("resource_state", {}).get("transition", {})
            authorized_retreat = bool(receipt.get("accepted")
                                      and receipt.get("reason") == "retreat_ascent")
        if (not authorized_retreat and getattr(self, "resource_portal", "off") != "off"
                and int(self._raw["dungeon_level"]) != int(prev["dungeon_level"])):
            # R18-F portal-v1: the portal is the only sanctioned MULTI-floor
            # transition. Outbound is L2+ -> 0, a depth drop this fail-closed
            # invariant would otherwise kill the engine process for; the return
            # leg 0 -> L2+ is an increase and already legal, and is listed only
            # so both engine receipts are named in one place. Nothing is trusted
            # except an ACCEPTED receipt whose reason and geometry both match.
            receipt = self._raw.get("resource_state", {}).get("transition", {})
            previous_depth = int(prev["dungeon_level"])
            new_depth = int(self._raw["dungeon_level"])
            if bool(receipt.get("accepted")) and receipt.get("reason") == "portal_to_town":
                authorized_retreat = previous_depth >= 2 and new_depth == 0
            elif bool(receipt.get("accepted")) and receipt.get("reason") == "portal_return":
                authorized_retreat = previous_depth == 0 and new_depth >= 2
        if (self.start_in_dungeon and not authorized_retreat
                and self._raw["dungeon_level"] < prev["dungeon_level"]):
            # If a new town-return/upward teleport path ever appears, stopping training is preferable
            # to silently feeding depth=0 wasted trajectories to PPO. The native trigger layer already blocks
            # the usual up stairs and town-return stairs; this is the second fail-closed invariant.
            bad_transition = (prev["dungeon_level"], self._raw["dungeon_level"])
            try:
                bridge.end_game()
            finally:
                if DiabloGymEnv._active_token is self._token:
                    DiabloGymEnv._active_token = None
                self._raw = None
                self._native_generation = None
                self._episode_ended = True
                self._engage_blocked_keys = set()
            raise RuntimeError(
                f"DiabloGym forbids dungeon level regression: {bad_transition[0]}→{bad_transition[1]}")
        self._steps += micro
        same_scene = _scene_identity(self._raw) == _scene_identity(prev)
        if not same_scene and getattr(self, "resource_protocol", "off") != "off":
            names = ("_visited", "_progress_anchors", "_explore_target",
                     "_explore_blocked_targets", "_engage_blocked_keys", "_combat_hp_floor")
            self._resource_scene_ledgers[_scene_identity(prev)] = {
                name: getattr(self, name) for name in names}
        if not same_scene:
            # New main level or quest set-level: clear the footprints. Different maps share one coordinate system; without clearing,
            # the explore macro would treat the old map's footprints as "visited" and the frontier logic would fail for the whole level; the lowest-HP
            # lines also need a new ledger, since monster ids only mean something within one scene.
            self._visited = set()
            self._progress_anchors = set()  # R16 C6: anchors are cleared with the footprints per scene
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._reset_combat_ledger(self._raw)
            if getattr(self, "resource_protocol", "off") != "off":
                for name, value in self._resource_scene_ledgers.get(
                        _scene_identity(self._raw), {}).items():
                    setattr(self, name, value)
        if getattr(self, "resource_protocol", "off") != "off" and not self._raw.get("is_set_level"):
            depth = int(self._raw["dungeon_level"])
            if depth > self._resource_max_main_depth:
                receipt = dict(self._raw.get("resource_state", {}).get("transition", {}))
                if getattr(self, "resource_readiness_law", "veto-v1") == "coach-v03":
                    # Forced-unready descents are legal under the frozen mask law;
                    # the receipt must exist and be accepted, readiness is accounting.
                    if depth > 1 and not receipt.get("accepted"):
                        raise RuntimeError("New main depth lacks an accepted transition receipt")
                elif depth > 1 and (not receipt.get("accepted")
                        or not receipt.get("pretransition_ready")):
                    raise RuntimeError("New main depth lacks an accepted pretransition readiness receipt")
                self._resource_transition_receipts.append(receipt)
                self._resource_max_main_depth = depth
            if self._resource_max_main_depth > self._resource_reward_depth_before:
                self._econ_steps_on_level = 0
            self._econ_episode_max_depth = self._resource_max_main_depth
        self._record_visit((self._raw["player_x"], self._raw["player_y"]))

        # Native MonsterDeath is the only event-complete kill ledger.  Endpoint
        # list differences still miss a monster that spawns and dies inside
        # one multi-beat macro, and slot generation keys cannot recover an
        # entity absent from both observations.
        native_kills = self._native_monster_kill_delta(prev, self._raw)
        if native_kills is None:
            # Compatibility for old synthetic fixtures only.  Current native
            # observations always carry monster_kill_total.
            native_kills = (
                self._disappeared_monster_generations(prev, self._raw)
                if same_scene else 0
            )
        self._ep_kills += native_kills

        effect_reasons = self._action_effect_reasons(
            prev,
            self._raw,
            requested_action=action,
            engage_target_generation_key=engage_target_generation_key,
            native_kills=native_kills,
            exploration_before=exploration_before,
            softwalls_before=softwalls_before,
            drink_audit=drink_audit,
            action14_audit=action14_audit,
        )
        action_effective = bool(effect_reasons)
        request_executed = bool(
            action == 0
            or native_execution["accepts"] > 0
        )
        stall_cost_applied = bool(
            _scene_identity(self._raw) == _scene_identity(prev)
            and (action == 0 or not request_executed)
        )
        if resource_command is not None:
            request_executed = bool((resource_receipt or {}).get("accepted"))
            stall_cost_applied = bool(same_scene and not request_executed and micro > 0)
        same_scene_for_reward = bool(
            _scene_identity(self._raw) == _scene_identity(prev))

        reward = self._reward(
            prev,
            self._raw,
            requested_action=None if resource_command is not None else action,
            engage_target_generation_key=engage_target_generation_key,
            action_executed=request_executed,
            action14_utility_delta=(
                int(action14_audit["utility_delta"])
                if action14_audit is not None else None
            ),
        )
        if resource_command is not None and resource_command[0] in ("finish", "complete"):
            reward = 0.0
            stall_cost_applied = False
        # The reward advances the combat ledger; the next decision's candidate ledger must be frozen after that.
        self._controller_snapshot = (
            self._capture_controller_snapshot(self._raw)
            if getattr(self, "_controller_snapshot_enabled", False)
            else None
        )
        # SB3 automatically adds gamma*V(s_T) for the terminal_observation of every Gym ``truncated``.
        # If the budget runs out exactly during a walk/hit animation, the 295-dim observation has no
        # future/mode/path, so the value network sees a busy state that fully aliases "an idle state that can
        # decide again immediately"; bootstrapping would then assume out of nowhere that the player can skip
        # the committed animation and act at once. The animation must not be settled secretly beyond max_steps, nor may a
        # POMDP state pose as a legal TimeLimit state. So this one representation-level boundary is defined as a
        # fail-closed terminal; a truly idle time limit remains a standard truncation,
        # keeping the correct TimeLimit bootstrap.
        (terminated, truncated, budget_exhausted, decision_idle,
         unsettled_budget_terminal) = self._episode_boundary(
             self._raw, self._steps, self.max_steps)

        resource_failure_terminal = bool(
            getattr(self, "resource_protocol", "off") != "off"
            and self._resource_terminal_reason is not None
        )
        if resource_failure_terminal:
            # A failed service itinerary ends this protocol episode. It is
            # not a settled TimeLimit that SB3 may bootstrap through, even
            # when the player is alive or the episode clock also runs out.
            terminated, truncated = True, False
        info = self._info(self._raw)
        if getattr(self, "resource_protocol", "off") != "off":
            info["resource_protocol"] = {
                "protocol": self.resource_protocol, "purchase_mode": self.resource_purchase_mode,
                "max_main_depth": self._resource_max_main_depth,
                "terminal_reason": self._resource_terminal_reason,
                "new_main_depth_bonus": self._resource_step_bonus,
                "curriculum_boundary": int(self._raw["dungeon_level"]) >= 2,
                "readiness": dict(self._raw["resource_state"]["readiness"]),
                "transition": dict(self._raw["resource_state"].get("transition", {})),
            }
            calibration = getattr(self, "_resource_calibration", None)
            if calibration is not None:
                info["resource_protocol"]["calibration"] = calibration.metadata()
            if resource_command is not None:
                info["resource_action_audit"] = {
                    "source": "resupply-script", "command": resource_command,
                    "micro_steps": micro, **(resource_receipt or {})}
        info["action_effect_audit"] = {
            "requested_action": action,
            "native_attempts": int(native_execution["attempts"]),
            "native_accepts": int(native_execution["accepts"]),
            "request_executed": request_executed,
            "material_effect": action_effective,
            "effect_reasons": tuple(effect_reasons),
            "same_scene": same_scene_for_reward,
            "stall_cost_applied": stall_cost_applied,
        }
        if dive_recovery_audit is not None:
            info["dive_blocker_recovery_audit"] = {
                **dive_recovery_audit,
                "transition_before_microstep": resource_clock_before,
                "after_microstep": self._resource_actual_microsteps,
                "micro_steps": micro,
                "settle_micro_steps": micro - dive_recovery_audit["core_micro_steps"],
                "decision_idle": self._decision_idle(self._raw),
            }
        if drink_audit is not None:
            belt_after = int(self._raw.get("belt_heals", 0))
            info["action12_audit"] = {
                **drink_audit,
                "belt_after": belt_after,
            }
        if action14_audit is not None:
            info["action14_audit"] = dict(action14_audit)
        if budget_exhausted:
            info.update({
                "budget_exhausted": True,
                "decision_idle": bool(decision_idle),
                "unsettled_budget_terminal": unsettled_budget_terminal,
                "time_limit_bootstrap_safe": bool(truncated),
            })
        if resource_failure_terminal:
            info.update({
                "resource_failure_terminal": True,
                "decision_idle": bool(decision_idle),
                "unsettled_budget_terminal": unsettled_budget_terminal,
                "time_limit_bootstrap_safe": False,
            })
        if terminated or truncated:
            info["episode_extra"] = {
                "xp": int(self._raw["xp"]) - self._ep_start_xp,
                "kills": self._ep_kills,
                "char_level": self._raw["char_level"],
                "depth": self._raw["dungeon_level"],
                "died": bool(self._raw["dead"]),
                "gold": self._raw["gold"],
            }
        obs = self._vectorize(self._raw)
        if terminated or truncated:
            self._episode_ended = True
        return obs, reward, terminated, truncated, info

    def _info(self, raw):
        info = {"episode_seed": self._episode_seed}
        time_info = getattr(self, "resource_time_info_callback", None)
        if time_info is not None:
            info["completion_time"] = time_info()
        if self.include_raw:
            # info belongs to the caller; the mutable raw that the internal reward/macro state depends on must not
            # leak out directly, or callbacks or debug code modifying info["raw"] would tamper with the next tick's reward.
            # New protocols keep appending list[dict] (equipped/missiles etc.), so an allowlist that easily misses fields
            # cannot be maintained. Only Python mutable containers are copied recursively;
            # numbers/strings stay shared immutable objects, and the cost is still far below copying native state.
            def clone_mutable(value):
                if isinstance(value, dict):
                    return {
                        key: clone_mutable(child)
                        for key, child in value.items()
                    }
                if isinstance(value, list):
                    return [clone_mutable(child) for child in value]
                if isinstance(value, tuple):
                    return tuple(clone_mutable(child) for child in value)
                if isinstance(value, set):
                    return {clone_mutable(child) for child in value}
                return value

            snapshot = clone_mutable(raw)
            info["raw"] = snapshot
        return info

    def _ensure_active(self, *, allow_ended: bool = False) -> None:
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            raise RuntimeError(
                "using a DevilutionX instance initialized by the parent process in a fork child is forbidden; "
                "multi-environment training must use spawn")
        if self._raw is None:
            raise gym.error.ResetNeeded("reset() must be called before step/action_masks")
        if DiabloGymEnv._active_token is not self._token:
            raise RuntimeError(
                "multiple DiabloGymEnv instances interleaved in the same process; the engine is a global singleton. "
                "Reset/use them sequentially, or use SubprocVecEnv instead")
        if int(bridge.episode_generation()) != self._native_generation:
            raise RuntimeError(
                "the engine was reset directly by bridge.reset() or by another wrapper, "
                "so this environment's cache is invalid; reset() this environment again")
        if self._episode_ended and not allow_ended:
            raise gym.error.ResetNeeded("episode terminated/truncated; reset() is required before step() continues")

    def close(self):
        if (DiabloGymEnv._engine_initialized
                and DiabloGymEnv._engine_pid != os.getpid()):
            # The child inherits an unsafe snapshot of the parent's multi-threaded engine; it must never
            # enter SDL/NetClose/Lua here. The OS reclaims the child's address space.
            if DiabloGymEnv._active_token is self._token:
                DiabloGymEnv._active_token = None
            self._raw = None
            self._native_generation = None
            self._episode_ended = True
            self._exploration_progress = 0
            self._softwalls_opened = 0
            self._explore_target = None
            self._explore_blocked_targets = set()
            self._engage_blocked_keys = set()
            self._combat_hp_floor = {}
            self._controller_snapshot = None
            return
        if DiabloGymEnv._active_token is self._token:
            bridge.end_game()
            DiabloGymEnv._active_token = None
        self._raw = None
        self._native_generation = None
        self._episode_ended = True
        self._exploration_progress = 0
        self._softwalls_opened = 0
        self._explore_target = None
        self._explore_blocked_targets = set()
        self._engage_blocked_keys = set()
        self._combat_hp_floor = {}
        self._controller_snapshot = None

    # ---------- internals ----------

    def _record_visit(self, pos) -> bool:
        """Register the real footing; a first visit also advances the Options exploration progress clock.

        R16 C6 (progress_far_tiles>0): footprint registration is unchanged, but only new tiles at Chebyshev distance
        >= progress_far_tiles from the existing "progress anchors" (the start + tiles already counted as progress)
        advance exploration_progress and become new anchors themselves. Walking straight along a corridor counts once every
        far tiles; pacing near the footprints never counts. If anchors were measured against "all footprints", the heel would
        always be at distance 1 when walking straight and no movement would ever count, hence the anchor set rather than the footprint set as reference.
        """
        point = (int(pos[0]), int(pos[1]))
        if point in self._visited:
            return False
        self._visited.add(point)
        far = int(getattr(self, "_progress_far_tiles", 0))
        if far > 0:
            anchors = getattr(self, "_progress_anchors", None)
            if anchors is None:
                anchors = set(self._visited)
                anchors.discard(point)
                self._progress_anchors = anchors
            if any(
                (point[0] + dx, point[1] + dy) in anchors
                for dx in range(1 - far, far)
                for dy in range(1 - far, far)
            ):
                return True
            anchors.add(point)
        self._exploration_progress += 1
        return True

    def _apply_action(
        self,
        action: int,
        *,
        worker_authority: bool = False,
    ) -> int:
        obs = self._raw
        px, py = obs["player_x"], obs["player_y"]
        if action == 0:
            bridge.act_wait()
            return 1
        elif 1 <= action <= 8:
            dx, dy = _DIRS[action - 1]
            # All learned direction keys use the same exact-one-step native
            # commit.  Restricting this to Worker left flat/base training able
            # to invoke unrestricted path replanning and cross protected
            # progression tiles under a nominal adjacent action.
            direction_protected = (self._resource_direction_protected_tiles(obs)
                if getattr(self, "resource_protocol", "off") != "off"
                else self._explore_protected_tiles(obs))
            protected = sorted(
                point
                for point in direction_protected
                if max(abs(point[0] - px), abs(point[1] - py)) <= 1
            )
            return bridge.act_explore_walk(
                px + dx,
                py + dy,
                protected,
                px,
                py,
                1,
            )
        raise RuntimeError(f"_apply_action got unknown atomic action {action}")

    @staticmethod
    def _record_native_execution(
        audit: dict,
        result,
        label: str,
    ) -> bool:
        """Record one checked native 0/1 command receipt."""
        if (
            isinstance(result, (bool, np.bool_))
            or isinstance(result, (int, np.integer))
        ):
            accepted = int(result)
        else:
            raise RuntimeError(
                f"{label} native execution receipt must be an integer 0/1, "
                f"got {result!r}")
        if accepted not in (0, 1):
            raise RuntimeError(
                f"{label} native execution receipt must be 0/1, got {accepted}")
        audit["attempts"] = int(audit.get("attempts", 0)) + 1
        audit["accepts"] = int(audit.get("accepts", 0)) + accepted
        return bool(accepted)

    def _wait_step(self):
        """Cancel old commands and spend one standard micro-step; used when a macro returns with no target/unreachable."""
        bridge.act_wait()
        raw = self._step_native()
        if (not raw.get("dead") and not raw.get("game_over") and not raw.get("victory")
                and int(raw.get("dest_action", bridge.ACTION_NONE)) != bridge.ACTION_NONE):
            raise RuntimeError(
                f"destAction={raw.get('dest_action')} still set after the wait step")
        if (not raw.get("dead") and not raw.get("game_over") and not raw.get("victory")
                and int(raw.get("walkpath0", bridge.WALK_NONE)) != bridge.WALK_NONE):
            raise RuntimeError(
                f"walkpath0={raw.get('walkpath0')} still set after the wait step")
        return raw, 1

    @staticmethod
    def _finish_macro(raw, beats: int, start_scene):
        """At the end of a macro, clear the path/destAction first; the outer layer settles hidden animations uniformly.

        wait's loopback FIFO fence guarantees that the next policy action inherits no path or attack;
        the costed ticks needed by the current single-tile/hit animation are advanced uniformly by ``_settle_to_idle``,
        not by a loop copied into each macro, and game time is never consumed for free.
        """
        if (raw.get("dead") or raw.get("game_over") or raw.get("victory")
                or _scene_identity(raw) != start_scene):
            return raw, beats
        if not bridge.act_wait():
            return raw, beats
        refreshed = bridge.observe()
        if raw.get("resource_state", {}).get("preserve_equipment_readiness") is True:
            from .resource_protocol import (
                validate_native_armor_scope, validate_native_equipment_readiness_preservation)
            validate_native_armor_scope(
                refreshed, raw["resource_state"].get("ordinary_armor_scope", False))
            validate_native_equipment_readiness_preservation(refreshed, True)
        expected_loot = raw.get("resource_state", {}).get("loot_economy", False)
        observed_loot = refreshed.get("resource_state", {}).get("loot_economy", False)
        if type(observed_loot) is not bool or observed_loot != expected_loot:
            raise RuntimeError("Native loot economy identity mismatch at macro boundary")
        if (int(refreshed.get("dest_action", bridge.ACTION_NONE)) != bridge.ACTION_NONE
                or int(refreshed.get("walkpath0", bridge.WALK_NONE)) != bridge.WALK_NONE):
            raise RuntimeError(
                "wait did not clear the native command state before the macro returned: "
                f"dest_action={refreshed.get('dest_action')}, "
                f"walkpath0={refreshed.get('walkpath0')}")
        return refreshed, beats

    @staticmethod
    def _engage_distance(raw, monster) -> int:
        """Chase distance with exactly the same definition as CMD_ATTACKID (future->future).

        The player/monster tile changes first when a one-tile walk animation starts, whereas future is
        the end point the engine has already committed. Comparing tiles would misreport a legal 8-10 tick animation
        as "not moving".
        """
        px = int(raw.get("future_x", raw["player_x"]))
        py = int(raw.get("future_y", raw["player_y"]))
        mx = int(monster.get("future_x", monster["x"]))
        my = int(monster.get("future_y", monster["y"]))
        return max(abs(mx - px), abs(my - py))

    def _select_engage_target(
        self,
        raw,
        *,
        exclude: set[tuple[int, int, int]] | None = None,
        allow_blocked_cycle: bool = False,
        candidate_keys: tuple[tuple[int, int, int], ...] | None = None,
    ):
        """Take the nearest target not yet proven to fail; stable ties are broken by id, for reproducible audits.

        Failed targets are never re-picked within one macro. The failure set persists across macros; only when all current candidates
        have been tried once is it cleared for another round, so that a single dynamic monster never permanently loses attackability.
        """
        excluded = exclude or set()
        policy_monsters = self._policy_monsters(raw)
        if candidate_keys is None:
            candidates = [
                monster for monster in policy_monsters
                if self._monster_generation_key(monster) not in excluded
            ]
        else:
            # a9's candidate universe was fixed when the policy observed. Inside the macro only members that died/lost reachability
            # may be filtered out in that canonical order; a 33rd monster must never be slipped in.
            by_key = {
                self._monster_generation_key(monster): monster
                for monster in policy_monsters
            }
            candidates = [
                by_key[generation_key]
                for generation_key in candidate_keys
                if generation_key in by_key
                and generation_key not in excluded
            ]
        if not candidates:
            return None
        if candidate_keys is None:
            px = int(raw.get("future_x", raw["player_x"]))
            py = int(raw.get("future_y", raw["player_y"]))
            candidates.sort(key=lambda m: (
                abs(int(m.get("future_x", m["x"])) - px)
                + abs(int(m.get("future_y", m["y"])) - py),
                int(m["id"]),
            ))

        policy_keys = (
            {self._monster_generation_key(m) for m in policy_monsters}
            if candidate_keys is None
            else set(candidate_keys)
        )
        # A generation that disappeared/left the visible-reachable set must not pollute a new lifetime; when the same
        # native slot id is reused by a spawn, rndItemSeed produces a new key.
        self._engage_blocked_keys.intersection_update(policy_keys)
        for monster in candidates:
            if (
                self._monster_generation_key(monster)
                not in self._engage_blocked_keys
            ):
                return monster
        if not allow_blocked_cycle:
            return None
        self._engage_blocked_keys.difference_update(
            self._monster_generation_key(m) for m in candidates)
        return candidates[0]

    @staticmethod
    def _movement_engine_busy(raw) -> bool:
        """The player is still executing/queuing a walk; a paused tile does not mean the animation or path has stopped."""
        walking_modes = {
            bridge.PM_WALK_NORTHWARDS,
            bridge.PM_WALK_SOUTHWARDS,
            bridge.PM_WALK_SIDEWAYS,
        }
        return (
            int(raw.get("walkpath0", bridge.WALK_NONE)) != bridge.WALK_NONE
            or int(raw.get("player_mode", -1)) in walking_modes
            or (
                int(raw.get("future_x", raw["player_x"])) != int(raw["player_x"])
                or int(raw.get("future_y", raw["player_y"])) != int(raw["player_y"])
            )
        )

    @classmethod
    def _engage_engine_busy(cls, raw) -> bool:
        """With a chase packet queued, a tile still being walked or a swing in progress, unchanged geometry does not mean stuck."""
        return (
            cls._movement_engine_busy(raw)
            or int(raw.get("dest_action", bridge.ACTION_NONE))
            == bridge.ACTION_ATTACKMON
            or int(raw.get("player_mode", -1)) == bridge.PM_ATTACK
        )

    @staticmethod
    def _decision_idle(raw) -> bool:
        """Policy-observable boundary: no player execution state exists outside the 295-dim vector."""
        return (
            int(raw.get("player_mode", -1)) == bridge.PM_STAND
            and int(raw.get("dest_action", bridge.ACTION_NONE))
            == bridge.ACTION_NONE
            and int(raw.get("walkpath0", bridge.WALK_NONE))
            == bridge.WALK_NONE
            and int(raw.get("future_x", raw["player_x"]))
            == int(raw["player_x"])
            and int(raw.get("future_y", raw["player_y"]))
            == int(raw["player_y"])
        )

    @classmethod
    def _episode_boundary(cls, raw, steps: int, max_steps: int):
        """Classify native terminal states, bootstrappable time limits and unobservable budget interruptions.

        Returns ``(terminated, truncated, budget_exhausted, decision_idle,
        unsettled_budget_terminal)``. The latter two Gym boundaries are strictly mutually exclusive.
        """
        native_terminated = bool(
            raw.get("dead") or raw.get("game_over") or raw.get("victory"))
        budget_exhausted = int(steps) >= int(max_steps)
        decision_idle = cls._decision_idle(raw)
        unsettled_budget_terminal = bool(
            budget_exhausted and not native_terminated and not decision_idle)
        terminated = bool(native_terminated or unsettled_budget_terminal)
        truncated = bool(budget_exhausted and not terminated)
        return (
            terminated,
            truncated,
            budget_exhausted,
            decision_idle,
            unsettled_budget_terminal,
        )

    def _settle_to_idle(
        self,
        raw,
        beats: int,
        *,
        max_beats: int,
        start_scene,
    ):
        """Cancel long commands and settle hidden animations with costed engine beats.

        Death/victory is handed immediately to the outer termination logic; after a scene change settling continues on the new map,
        so that the manager's first observation is also idle. If the total step budget runs out first, a busy raw may be
        returned, but the same ``step`` marks it as
        ``unsettled_budget_terminal`` instead of a bootstrappable truncation;
        it never becomes the input of the next policy decision and is never taken by the value function as an idle alias.
        """
        beats = int(beats)
        max_beats = int(max_beats)
        terminal = (
            raw.get("dead") or raw.get("game_over") or raw.get("victory")
        )
        calibration = getattr(self, "_resource_calibration", None)
        resource_command = getattr(self, "_resource_pending_command", None)
        bounded_resource = bool(resource_command and resource_command[0]
                                in ("attack_monster", "drink", "unequip"))

        bounded_dive = getattr(self, "_dive_blocker_audit", None) is not None
        completion_time = getattr(self, "resource_time_callback", None) is not None
        settle_clock_start = (self._resource_actual_microsteps - beats
                              if completion_time else None)

        def settle_limit():
            if not completion_time:
                return max_beats
            physical = self.max_steps - settle_clock_start
            # A new arrival may extend or shorten the physical horizon inside
            # this very action. Only its pending animation gets the new room;
            # resource-command budgets retain their existing tighter limits.
            return physical if resource_command is None else min(max_beats, physical)

        def resource_time_available():
            return ((not bounded_resource or self._resource_actual_microsteps < min(
                self.max_steps,
                int(getattr(self, "_resource_service_deadline", self.max_steps))))
                and (not bounded_dive or self._resource_actual_microsteps < self.max_steps)
                and (not completion_time or self._resource_actual_microsteps < self.max_steps))

        if (terminal or beats >= settle_limit() or not resource_time_available()
                or (calibration is not None
                    and self._resource_actual_microsteps >= self.max_steps)):
            return raw, beats

        # ActWait immediately clears path/dest/attack; an already committed single-tile walk or hit/block
        # animation cannot be cut hard, only continued with game_loop. The first call is also a FIFO fence, so that
        # old network packets cannot put a long command back on the next tick.
        settle_scene = _scene_identity(raw)
        bridge.act_wait()
        while (beats < settle_limit()
               and not self._decision_idle(raw)
               and resource_time_available()
               and (calibration is None or self._resource_actual_microsteps < self.max_steps)):
            raw = self._step_native()
            beats += 1
            current_scene = _scene_identity(raw)
            if current_scene == start_scene:
                self._record_visit((raw["player_x"], raw["player_y"]))
            if (raw.get("dead") or raw.get("game_over") or raw.get("victory")
                    or not resource_time_available()
                    or (calibration is not None
                        and self._resource_actual_microsteps >= self.max_steps)):
                break
            if current_scene != settle_scene:
                # The first manager/worker observation after a map change must also be idle; put a new FIFO fence down
                # in the new scene and keep settling instead of leaking PM_NEWLVL.
                settle_scene = current_scene
                bridge.act_wait()
        return raw, beats

    @staticmethod
    def _canonical_engage_candidate(
        snapshot: _ControllerSnapshot,
    ) -> _ControllerMonster | None:
        """Return exactly the candidate action 9 will bind for this snapshot."""
        if not snapshot.candidates:
            return None
        return next(
            (
                candidate
                for candidate in snapshot.candidates
                if not candidate.blocked
            ),
            snapshot.candidates[0],
        )

    def _hunt_allowed_here(self, raw):
        """R18-G hunt_scope: "all" (byte-identical) or "l1-only" (main L2+ never
        hunts across the level; quest set-levels keep the frozen behaviour)."""
        if getattr(self, "hunt_scope", "all") == "all":
            return True
        return bool(raw.get("is_set_level")) or int(raw.get("dungeon_level") or 0) <= 1

    def _engage_candidate_for_action9(self, snapshot, record=True):
        """R18-E: the target action 9 binds. threat-v1 re-ranks the same frozen
        candidate set on main L2+ (a function of the snapshot plus the decision
        floor, constant within a decision, so the a9 approach reward and the macro
        agree); off = the wire-canonical first item, byte-identical."""
        raw = getattr(self, "_raw", None) or {}
        if (getattr(self, "engagement_priority", "off") == "off"
                or int(raw.get("dungeon_level") or 0) < 2 or raw.get("is_set_level")):
            # v1 scope: main L2+ only (L1 prefix byte-identical to the retreat arm).
            return self._canonical_engage_candidate(snapshot)
        from .engagement import choose_engage_candidate
        chosen = choose_engage_candidate(snapshot)
        if record:
            # counted once per a9 step (the reward-key call in step()); _macro_engage
            # re-selects on the same snapshot with record=False.
            canonical = self._canonical_engage_candidate(snapshot)
            self._engagement_decisions += 1
            if (chosen is not None and canonical is not None
                    and chosen.monster_id != canonical.monster_id):
                self._engagement_reordered += 1
        return chosen

    def _macro_engage(
        self,
        max_beats: int = 10,
        *,
        controller_snapshot: _ControllerSnapshot | None = None,
        execution_audit: dict | None = None,
    ):
        """Chase one engageable target; on a real stall, rotate within the same action9 budget.

        A warrior's one-tile walk takes about 8-10 engine ticks, and a swing needs another ten-odd ticks to reach its damage
        frame. The old implementation only looked at tile/HP in two 4-tick samples and called ActWait right before the first swing's
        damage frame, aborting the attack and never dealing damage. Now it counts as a stall only when the engine is already idle and
        two consecutive ticks neither approached nor damaged; a pending attack, an unfinished walk step and
        PM_ATTACK are all in progress and must not be cancelled early.
        """
        snapshot = (
            controller_snapshot
            if controller_snapshot is not None
            else self._capture_controller_snapshot(self._raw)
        )
        candidates = list(snapshot.candidates)
        candidate = self._engage_candidate_for_action9(snapshot, record=False)
        if candidate is None:
            return self._wait_step()
        if candidate.blocked:
            # The fact "all have been tried" is already published in each snapshot slot's blocked bit;
            # action9 atomically starts the next round when it executes, still choosing the wire-canonical first entry.
            self._engage_blocked_keys.difference_update(
                entry.generation_key for entry in candidates)
        tid = int(candidate.monster_id)
        target_key = candidate.generation_key
        target = next(
            (
                monster for monster in self._raw.get("monsters", ())
                if self._monster_generation_key(monster) == target_key
                and bool(monster.get("visible", True))
            ),
            None,
        )
        if target is None:
            self._engage_blocked_keys.add(target_key)
            return self._wait_step()

        start_scene = _scene_identity(self._raw)
        raw = self._raw
        path = self._plan_controller_path(
            snapshot,
            candidate.future_x,
            candidate.future_y,
            avoid_monsters=True,
            allow_softwalls=False,
        )
        initial_distance = max(
            abs(int(raw.get("future_x", raw["player_x"]))
                - int(target.get("future_x", target["x"]))),
            abs(int(raw.get("future_y", raw["player_y"]))
                - int(target.get("future_y", target["y"]))),
        )
        if path is None and initial_distance > 1:
            self._engage_blocked_keys.add(target_key)
            return self._wait_step()

        protected = sorted(snapshot.protected_tiles)
        pi = 0
        active_step: tuple[int, int] | None = None
        beats = 0
        last_hp = int(target["hp"])
        last_distance = self._engage_distance(raw, target)
        target_start_hp = last_hp
        target_start_distance = last_distance
        target_progress = False
        idle_stall = 0

        for beats in range(1, max_beats + 1):
            current_target = next(
                (monster for monster in raw.get("monsters", ())
                 if self._monster_generation_key(monster) == target_key),
                None,
            )
            if current_target is None:
                self._engage_blocked_keys.discard(target_key)
                break
            target_x = int(current_target.get(
                "future_x", current_target["x"]))
            target_y = int(current_target.get(
                "future_y", current_target["y"]))
            if (
                not bool(current_target.get("visible", True))
                or abs(target_x - snapshot.player_x)
                > CONTROLLER_SNAPSHOT_RADIUS
                or abs(target_y - snapshot.player_y)
                > CONTROLLER_SNAPSHOT_RADIUS
            ):
                self._engage_blocked_keys.add(target_key)
                break

            player_x = int(raw.get("future_x", raw["player_x"]))
            player_y = int(raw.get("future_y", raw["player_y"]))
            adjacent = max(
                abs(player_x - target_x), abs(player_y - target_y)) <= 1
            if self._decision_idle(raw):
                if adjacent:
                    accepted = bridge.act_controller_attack_monster(
                        tid,
                        snapshot.player_x,
                        snapshot.player_y,
                        CONTROLLER_SNAPSHOT_RADIUS,
                    )
                    if execution_audit is not None:
                        self._record_native_execution(
                            execution_audit, accepted, "action9 attack")
                elif active_step is None:
                    if path is None or pi >= len(path):
                        self._engage_blocked_keys.add(target_key)
                        break
                    nx, ny, is_softwall = path[pi]
                    if is_softwall:
                        self._engage_blocked_keys.add(target_key)
                        break
                    if abs(nx - player_x) + abs(ny - player_y) != 1:
                        self._engage_blocked_keys.add(target_key)
                        break
                    accepted = bridge.act_explore_walk(
                        nx,
                        ny,
                        protected,
                        snapshot.player_x,
                        snapshot.player_y,
                        CONTROLLER_SNAPSHOT_RADIUS,
                    )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted, "action9 walk")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                        self._engage_blocked_keys.add(target_key)
                        break
                    active_step = (nx, ny)

            raw = self._step_native()
            if _scene_identity(raw) == start_scene:
                self._record_visit((raw["player_x"], raw["player_y"]))
            if (raw["dead"] or _scene_identity(raw) != start_scene):
                break
            # action9 is a multi-beat controller, while the wrapper's
            # emergency potion reflex runs only after control returns.  The
            # old loop could keep attacking for the rest of its ten-beat
            # budget after HP had already crossed below 50%, killing players
            # that still had several potions.  End the macro at the first
            # observed reflex-eligible state so the normal tail drain gets a
            # chance; no potion is consumed or hidden inside action9.
            if self._reflex_eligible(raw):
                break

            cur_target = next(
                (
                    m for m in raw["monsters"]
                    if self._monster_generation_key(m) == target_key
                ),
                None,
            )
            if cur_target is None:
                # Death/disappearance/slot reuse all end the old generation successfully; a new monster must
                # wait for the next policy observation and cannot inherit this macro's attack.
                self._engage_blocked_keys.discard(target_key)
                break

            cur_hp = int(cur_target["hp"])
            cur_distance = self._engage_distance(raw, cur_target)
            position = (int(raw["player_x"]), int(raw["player_y"]))
            if active_step is not None and position == active_step:
                pi += 1
                active_step = None
            hp_progress = cur_hp < last_hp
            distance_progress = cur_distance < last_distance
            if hp_progress or distance_progress:
                target_progress = True
                idle_stall = 0
                self._engage_blocked_keys.discard(target_key)
            elif self._engage_engine_busy(raw):
                # Animation/command still active; in particular, never cut losses before PM_ATTACK's damage frame.
                idle_stall = 0
            else:
                idle_stall += 1

            # The fixed snapshot path allows no re-planning/target switch within the macro; true idle with no progress in a row
            # records a failure and hands control back, and the next observation decides again.
            if idle_stall >= 2:
                self._engage_blocked_keys.add(target_key)
                break

            last_hp = cur_hp
            last_distance = cur_distance

        # Having used up the whole macro without dealing damage or getting closer than the target's starting point, the next action9
        # tries other candidates first. An animation in progress still gets the full budget this time, and this cross-macro
        # rotation rule never triggers an early ActWait.
        if (not raw.get("dead")
                and _scene_identity(raw) == start_scene):
            surviving = next(
                (m for m in raw.get("monsters", ())
                 if self._monster_generation_key(m) == target_key), None)
            if surviving is not None:
                effective_progress = (
                    target_progress
                    or int(surviving["hp"]) < target_start_hp
                    or self._engage_distance(raw, surviving)
                    < target_start_distance
                )
                if not effective_progress:
                    self._engage_blocked_keys.add(target_key)

        return self._finish_macro(raw, beats, start_scene)

    _EXPLORE_RADIUS = CONTROLLER_SNAPSHOT_RADIUS  # 25×25 fixed control snapshot

    _PROGRESSION_PRIORITY = {
        "lazarus_stand": 0,
        "lazarus_staff": 1,
        "vile_entrance": 2,
        "vile_book": 3,
        "vile_center_circle": 4,
        "diablo_switch": 5,
    }

    @staticmethod
    def _progression_present(raw, target) -> bool:
        """Position + kind is a stable identity within one scene; the target disappearing means this interaction was committed."""
        return any(
            p.get("kind") == target.get("kind")
            and int(p.get("x", -1)) == int(target["x"])
            and int(p.get("y", -1)) == int(target["y"])
            for p in raw.get("progression_targets", ())
        )

    @staticmethod
    def _progression_ready(raw, target) -> bool:
        px, py = int(raw["player_x"]), int(raw["player_y"])
        tx, ty = int(target["x"]), int(target["y"])
        gx, gy = int(target["goal_x"]), int(target["goal_y"])
        action = target["action"]
        if action == "walk":
            return (px, py) == (gx, gy)
        if action == "pickup":
            # CMD_GOTOAGETITEM may otherwise install a new multi-tile route
            # after the safe planner has finished.  Walk onto the exact item
            # tile first, then the pickup command has no route left to invent.
            return (px, py) == (tx, ty)
        if action == "operate" and bool(target.get("exact")):
            return (px, py) == (gx, gy)
        if action == "operate":
            return max(abs(px - tx), abs(py - ty)) <= 1
        raise RuntimeError(f"unknown story-target action: {action!r}")

    @staticmethod
    def _issue_progression(
        target,
        *,
        center_x: int,
        center_y: int,
        radius: int,
    ) -> bool:
        action = target["action"]
        if action == "operate":
            result = bridge.act_controller_operate(
                int(target["x"]),
                int(target["y"]),
                center_x,
                center_y,
                radius,
            )
        elif action == "pickup":
            result = bridge.act_pickup_progression(
                int(target["x"]), int(target["y"]))
        elif action == "walk":
            result = bridge.act_explore_walk(
                int(target["goal_x"]),
                int(target["goal_y"]),
                (),
                center_x,
                center_y,
                radius,
            )
        else:
            raise RuntimeError(f"unknown story-target action: {action!r}")
        if not isinstance(result, (bool, np.bool_, int, np.integer)):
            raise RuntimeError(
                "action11 progression native receipt must be an integer 0/1, "
                f"got {result!r}")
        accepted = int(result)
        if accepted not in (0, 1):
            raise RuntimeError(
                "action11 progression native receipt must be 0/1, "
                f"got {accepted}")
        return accepted == 1

    def _macro_progression(
        self,
        max_beats: int = 12,
        *,
        execution_audit: dict | None = None,
    ):
        """Advance the next completion-required interaction on the strict allowlist.

        Only action 11 / the DIVE manager calls this macro; action 10 / FARM fails closed and hands control back in
        story state and may not push the story forward. Vile books require standing exactly in the circle;
        ordinary mechanisms only need adjacency. The global BFS still treats only doors/barrels as softwalls; no teleporting, no walking through walls.
        """
        raw = self._raw
        targets = [dict(p) for p in raw.get("progression_targets", ())]
        if not targets:
            return self._wait_step()

        candidates = []
        for target in targets:
            required = {"kind", "action", "x", "y", "goal_x", "goal_y", "exact"}
            if set(target) != required:
                raise RuntimeError(f"story target schema anomaly: {target!r}")
            ready = self._progression_ready(raw, target)
            path = [] if ready else self._plan_descend_path(
                raw, int(target["goal_x"]), int(target["goal_y"]),
                avoid_monsters=True)
            px, py = int(raw["player_x"]), int(raw["player_y"])

            def assess(candidate_path):
                ex, ey = ((candidate_path[-1][0], candidate_path[-1][1])
                          if candidate_path else (px, py))
                if target["action"] == "operate" and not bool(target["exact"]):
                    remaining_ = max(abs(ex - int(target["x"])),
                                     abs(ey - int(target["y"])))
                    limit = 1
                elif target["action"] == "pickup":
                    remaining_ = max(abs(ex - int(target["x"])),
                                     abs(ey - int(target["y"])))
                    limit = 0
                else:
                    remaining_ = max(abs(ex - int(target["goal_x"])),
                                     abs(ey - int(target["goal_y"])))
                    limit = 0
                reachable_ = ready or (
                    candidate_path is not None and remaining_ <= limit)
                return remaining_, reachable_

            remaining, reachable = assess(path)
            if not ready and not reachable:
                # The "monster-avoiding BFS" returns a partial path even if it can only reach the wall of monsters,
                # not None. If it fell back only on None, L16 would misjudge the truly reachable second
                # switch as a far target and keep walking toward another wall that is not open yet.
                fallback = self._plan_descend_path(
                    raw, int(target["goal_x"]), int(target["goal_y"]))
                fallback_remaining, fallback_reachable = assess(fallback)
                old_rank = (not reachable, remaining,
                            len(path) if path is not None else 10**9)
                new_rank = (not fallback_reachable, fallback_remaining,
                            len(fallback) if fallback is not None else 10**9)
                if new_rank < old_rank:
                    path = fallback
                    remaining, reachable = fallback_remaining, fallback_reachable
            priority = self._PROGRESSION_PRIORITY.get(target["kind"], 999)
            candidates.append((not reachable, priority, remaining,
                               len(path) if path is not None else 10**9,
                               target, path))

        _, _, _, _, target, path = min(candidates, key=lambda c: c[:4])
        if path is None and not self._progression_ready(raw, target):
            return self._wait_step()

        start_scene = _scene_identity(raw)
        center_x = int(raw["player_x"])
        center_y = int(raw["player_y"])
        pi = 0
        command = None  # ("open"/"walk"/"progress", x, y, path_index)
        last_pos = (raw["player_x"], raw["player_y"])
        stall = 0
        beats = 0
        for beats in range(1, max_beats + 1):
            if not self._progression_present(raw, target):
                break
            if command is None:
                if self._progression_ready(raw, target):
                    accepted = self._issue_progression(
                        target,
                        center_x=center_x,
                        center_y=center_y,
                        radius=self._DESCEND_RADIUS,
                    )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit,
                            1 if accepted else 0,
                            "action11 progression",
                        )
                    if not accepted:
                        break
                    command = ("progress", int(target["x"]), int(target["y"]), pi)
                elif pi >= len(path):
                    break
                else:
                    j = pi
                    command = (
                        ("open", path[j][0], path[j][1], j)
                        if path[j][2]
                        else ("walk", path[j][0], path[j][1], j)
                    )
                    if command[0] == "open":
                        accepted = bridge.act_controller_operate(
                            command[1],
                            command[2],
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    else:
                        accepted = bridge.act_explore_walk(
                            command[1],
                            command[2],
                            (),
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted,
                            f"action11 {command[0]}")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                        break

            raw = self._step_native()
            pos = (raw["player_x"], raw["player_y"])
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            if raw["dead"] or _scene_identity(raw) != start_scene:
                break
            if self._reflex_eligible(raw):
                break
            if not self._progression_present(raw, target):
                break

            if command[0] == "open":
                if bridge.probe_tile(command[1], command[2])["walkable"]:
                    path[command[3]] = (command[1], command[2], False)
                    pi = command[3]
                    command = None
                    stall = 0
                    last_pos = pos
                    continue
            elif command[0] == "walk":
                # act_explore_walk now proves and installs exactly one adjacent
                # edge.  Do not consume that edge while the player's tile is
                # merely still adjacent to its endpoint (which is also true at
                # the starting tile); wait until the animation really lands.
                if pos == (command[1], command[2]):
                    pi = command[3] + 1
                    command = None
                    stall = 0
                    last_pos = pos
                    continue

            if pos == last_pos:
                stall += 1
                if stall == 3:
                    if command[0] == "open":
                        accepted = bridge.act_controller_operate(
                            command[1],
                            command[2],
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    elif command[0] == "walk":
                        accepted = bridge.act_explore_walk(
                            command[1],
                            command[2],
                            (),
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    else:
                        accepted = self._issue_progression(
                            target,
                            center_x=center_x,
                            center_y=center_y,
                            radius=self._DESCEND_RADIUS,
                        )
                        accepted = 1 if accepted else 0
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted,
                            f"action11 retry {command[0]}")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                            break
                if stall >= 6:
                    break
            else:
                stall = 0
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    def _resource_direction_protected_tiles(self, raw):
        protected = self._explore_protected_tiles(raw)
        if not getattr(self, "_resource_dive_authority", False):
            return protected
        from .resource_protocol import progression_allowed
        if not progression_allowed(raw, getattr(self, "resource_readiness_law", "veto-v1")):
            return protected
        transition = bridge.WM_DIABRTNLVL if raw.get("is_set_level") else bridge.WM_DIABNEXTLVL
        allowed = {(int(t["x"]), int(t["y"])) for t in raw.get("triggers", [])
                   if t.get("msg") == transition}
        for target in raw.get("progression_targets", []):
            allowed.add((int(target["x"]), int(target["y"])))
            allowed.add((int(target["goal_x"]), int(target["goal_y"])))
        return protected - allowed

    @staticmethod
    def _explore_protected_tiles(raw) -> set[tuple[int, int]]:
        protected = {
            (int(t["x"]), int(t["y"]))
            for t in raw.get("triggers", ())
        }
        for target in raw.get("progression_targets", ()):
            protected.add((int(target["x"]), int(target["y"])))
            protected.add((int(target["goal_x"]), int(target["goal_y"])))
        return protected

    @classmethod
    def _protected_walk_actions(cls, raw) -> frozenset[int]:
        """Adjacent direction keys that would trespass on DIVE-only tiles."""
        px, py = int(raw["player_x"]), int(raw["player_y"])
        protected = cls._explore_protected_tiles(raw)
        return frozenset(
            action
            for action, (dx, dy) in enumerate(_DIRS, start=1)
            if (px + dx, py + dy) in protected
        )

    def _plan_explore_step(
        self,
        raw,
        *,
        blocked_softwalls: set[tuple[int, int]] | None = None,
        controller_snapshot: _ControllerSnapshot | None = None,
    ):
        """Plan action10's original frontier point, or a reachable stance next to an ordinary softwall.

        Planning only crosses the current walkable component; triggers and story targets/goals are hard no-go zones,
        so exploration never steps on stairs or uses an ordinary operate to override DIVE's story authority. Ordinary
        closed doors within 8 steps may be handled on the way first; a blocking barrel is only a fallback once the component
        has no unvisited frontier left. With an existing frontier, action10 keeps its original far-waypoint continuous-pathing semantics.
        """
        snapshot = controller_snapshot
        if snapshot is not None:
            px, py = snapshot.player_x, snapshot.player_y
        else:
            px, py = int(raw["player_x"]), int(raw["player_y"])
        r = self._EXPLORE_RADIUS
        side = 2 * r + 1
        if snapshot is None:
            lm = bridge.local_map(radius=r)
            walk = lm["walkable"]
            occupied = lm["monster"]
            softwall = lm.get("door", [0] * (side * side))
            ordinary_door = lm.get("closed_door", softwall)
            hazard = lm.get("hazard", [0] * (side * side))
            explosive_softwall = lm.get(
                "explosive_softwall", [0] * (side * side))
            protected = self._explore_protected_tiles(raw)
            visited = self._visited
            blocked_targets = getattr(
                self, "_explore_blocked_targets", set())
            sticky = getattr(self, "_explore_target", None)
        else:
            walk = snapshot.walkable
            occupied = snapshot.physical_monster
            softwall = snapshot.softwall
            ordinary_door = snapshot.closed_door
            hazard = snapshot.hazard
            explosive_softwall = snapshot.explosive_softwall
            protected = set(snapshot.protected_tiles)
            visited = set(snapshot.visited_tiles)
            blocked_targets = set(snapshot.explore_blocked_targets)
            sticky = snapshot.sticky_target
        blocked = blocked_softwalls or set()

        # The old version picked frontiers only by the target tile's own walkability + geometric distance, and could pick a floor tile
        # five tiles away that actually needs a 160-tile maze detour; after the engine's 100-tile path cap refused it,
        # every action10 picked the same tile again and faced the wall forever. First build a conservative 4-direction component in the
        # local window, and use only tiles native pathing can really reach this time as candidates. Monster occupancy and authority no-go zones cannot be crossed.
        reachable = {(px, py)}
        depth = {(px, py): 0}
        queue = deque([(px, py)])

        def local_index(tx, ty):
            return (ty - py + r) * side + (tx - px + r)

        def in_window(tx, ty):
            return abs(tx - px) <= r and abs(ty - py) <= r

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if ((nx, ny) in reachable
                        or not in_window(nx, ny)
                        or (nx, ny) in protected):
                    continue
                i = local_index(nx, ny)
                if (
                    not walk[i]
                    or occupied[i]
                    or hazard[i]
                    or explosive_softwall[i]
                ):
                    continue
                reachable.add((nx, ny))
                depth[(nx, ny)] = depth[(cx, cy)] + 1
                queue.append((nx, ny))

        # Candidates: walkable frontier points ≥5 tiles from the player and not within the footprint neighbourhood (±1)
        near_visited = visited | {
            (x + dx, y + dy) for x, y in visited for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        }
        candidates = []
        retry_candidates = []
        for tx, ty in reachable:
            d_player = max(abs(tx - px), abs(ty - py))
            if (d_player >= 5
                    and (tx, ty) not in near_visited):
                entry = (d_player, tx, ty)
                if (tx, ty) in blocked_targets:
                    retry_candidates.append(entry)
                else:
                    candidates.append(entry)

        # Search the boundary of the current component for ordinary softwalls that really connect to another potential space.
        # local_map["door"] contains exactly closedDoor or a solid breakable barrel;
        # progression/trigger coordinates are still doubly excluded by protected.
        softwall_candidates = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                door = (px + dx, py + dy)
                if (door in blocked
                        or door in protected
                        or not softwall[local_index(*door)]
                        or explosive_softwall[local_index(*door)]
                        or hazard[local_index(*door)]):
                    continue
                approaches = []
                has_unseen_side = False
                for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (door[0] + ox, door[1] + oy)
                    if neighbor in reachable:
                        approaches.append(neighbor)
                        continue
                    if neighbor in protected:
                        continue
                    if not in_window(*neighbor):
                        has_unseen_side = True
                        continue
                    ni = local_index(*neighbor)
                    # Monster occupancy is dynamic; the floor itself existing is enough to prove there is no solid wall behind the door.
                    if (
                        not hazard[ni]
                        and not explosive_softwall[ni]
                        and (walk[ni] or softwall[ni])
                    ):
                        has_unseen_side = True
                if not approaches or not has_unseen_side:
                    continue
                approach = min(
                    approaches,
                    key=lambda p: (depth[p], p[0], p[1]),
                )
                softwall_candidates.append((
                    0 if ordinary_door[local_index(*door)] else 1,
                    depth[approach],
                    max(abs(door[0] - px), abs(door[1] - py)),
                    door[0], door[1], approach,
                ))

        chosen_softwall = None
        if softwall_candidates:
            # With an ordinary frontier, only real doors within 8 steps are handled on the way; barrels are kept as the last softwall
            # fallback for when the component has no frontier at all, so FARM does not get hooked on smashing barrels along the way.
            nearby_doors = [
                entry for entry in softwall_candidates
                if entry[0] == 0 and entry[1] <= 8
            ]
            if nearby_doors:
                chosen_softwall = min(nearby_doors)
            elif not candidates:
                chosen_softwall = min(softwall_candidates)
        if chosen_softwall is not None:
            _, _, _, door_x, door_y, approach = chosen_softwall
            if approach != (px, py):
                return ("approach", approach[0], approach[1])
            return ("open", door_x, door_y)

        # The frontier failure memory only means "rotate to other candidates first"; it must not turn one failure caused by dynamic
        # occupancy into a permanent whole-scene blacklist. Once the current unblocked candidates are exhausted,
        # atomically start the next round and retry targets that are still reachable/unvisited; otherwise the same hidden
        # set would permanently turn an identical visible map from "explorable" into wait.
        if not candidates and retry_candidates:
            self._explore_blocked_targets.difference_update(
                (tx, ty) for _, tx, ty in retry_candidates)
            candidates = retry_candidates

        # The frontier target persists across action10 until it is really reached/invalid. The "nearest point" greedy choice must not
        # be redone from each macro's new current position, or the two sides of a fork would keep stealing the nearest rank,
        # forming the permanent A<->B back-and-forth confirmed on seed7002.
        if sticky is not None:
            sticky = (int(sticky[0]), int(sticky[1]))
            if (sticky in protected
                    or sticky in blocked_targets
                    or sticky in visited
                    or sticky not in reachable
                    or max(abs(sticky[0] - px), abs(sticky[1] - py)) <= 1):
                self._explore_target = None
            else:
                return ("frontier", sticky[0], sticky[1])
        # R16 C5 extra hunt (default off): when the window has no visible monster (monsters behind walls/unlit do not
        # count as visible), prefer the whole-map monster hunt; as soon as a visible monster is in the window, hand back to local logic/the a9 mask/reflexes
        # (probes: gating on "no engageable candidate" made a9<->a10 oscillate next to monsters visible but not yet engageable,
        # seed7002 kills 96->48; gating on "no occupancy in the window" let dark monsters behind walls lock the hunt,
        # seed9005 wandered in place). Only when no live monster on the map is reachable (None) does it fall back to local frontier/retreat.
        if getattr(self, "_explore_global_hunt", False) and self._hunt_allowed_here(raw):
            if snapshot is not None:
                visible_in_window = any(snapshot.visible_monster)
            else:
                visible_in_window = any(
                    bool(m.get("visible"))
                    and in_window(int(m["x"]), int(m["y"]))
                    for m in raw.get("monsters", ())
                )
            if not visible_in_window:
                command = self._plan_explore_global_waypoint(
                    raw, px, py, reachable, blocked_targets,
                    monsters_only=True)
                if command is not None:
                    self._explore_target = (
                        (command[1], command[2])
                        if command[0] == "frontier" else None)
                    return command
        if candidates:
            _, tx, ty = min(candidates)  # nearest frontier point (cheap and stable)
            self._explore_target = (tx, ty)
            return ("frontier", tx, ty)
        # R16 C5 (default off): when the window has no candidate, fall back to a whole-map BFS and take the farthest prefix point,
        # inside this window and locally reachable, of the path to the nearest unvisited reachable tile/live-monster tile as
        # this frontier (if the prefix is cut by an in-window softwall, issue approach/open instead); then hand over to
        # the existing step-by-step walking/sticky target mechanism. Returns None only if there is still no candidate.
        if getattr(self, "_explore_global_fallback", False):
            command = self._plan_explore_global_waypoint(
                raw, px, py, reachable, blocked_targets)
            if command is not None:
                self._explore_target = (
                    (command[1], command[2])
                    if command[0] == "frontier" else None)
                return command
        self._explore_target = None
        return None

    def _plan_explore_global_waypoint(
        self,
        raw,
        px: int,
        py: int,
        reachable,
        blocked_targets,
        *,
        monsters_only: bool = False,
    ):
        """R16 C5: whole-map 4-direction BFS (same definition as _plan_descend_path: closed doors count as passable,
        hazard/explosive-softwall/trigger/story tiles are walls) to the nearest target, returning a command
        of the same kind as local planning: ("frontier", x, y) = the farthest prefix point of its path that lies inside the 25×25 window
        and belongs to this local reachable component; if the prefix is cut by an in-window softwall,
        returns ("approach", x, y) / ("open", x, y); returns None if there is no target/no prefix.

        Targets (the first BFS pop is the nearest):
          - live-monster tiles (radius-112 monster channel, dMonster≠0);
          - walkable tiles outside the window that are unvisited and not within the ±1 halo of the global footprints.
        Walkable unvisited tiles inside the window are already exhausted by the local candidates (those <5 from the player or inside the halo do not count),
        and are not repeated as targets. Monster tiles are only targets and are not expanded. The whole routine uses no random numbers.
        monsters_only=True (the hunt variant) treats only monster tiles as targets.
        If a prefix point is already in blocked_targets, it falls back to a nearer prefix point; when all are blocked,
        like the local candidates it clears these blocking memories and still takes the farthest prefix point, so that the same visible
        map never permanently degrades into wait.
        """
        r = self._EXPLORE_RADIUS
        big = self._DESCEND_RADIUS
        side = 2 * big + 1
        lm = bridge.local_map(radius=big)
        walk, door = lm["walkable"], lm["door"]
        hazard = lm.get("hazard", [0] * (side * side))
        explosive = lm.get("explosive_softwall", [0] * (side * side))
        mon = lm["monster"]
        protected = self._explore_protected_tiles(raw)
        visited = getattr(self, "_visited", set())

        def idx(tx, ty):
            return (ty - py + big) * side + (tx - px + big)

        def is_goal(tx, ty):
            if mon[idx(tx, ty)]:
                return True
            if monsters_only or (abs(tx - px) <= r and abs(ty - py) <= r):
                return False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (tx + dx, ty + dy) in visited:
                        return False
            return True

        start = (px, py)
        prev = {start: None}
        queue = deque([start])
        goal = None
        while queue and goal is None:
            cx, cy = queue.popleft()
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + ddx, cy + ddy
                if (abs(nx - px) > big or abs(ny - py) > big
                        or (nx, ny) in prev or (nx, ny) in protected):
                    continue
                i = idx(nx, ny)
                if hazard[i] or explosive[i]:
                    continue
                if not walk[i] and not door[i]:
                    continue
                prev[(nx, ny)] = (cx, cy)
                if is_goal(nx, ny):
                    goal = (nx, ny)
                    break
                queue.append((nx, ny))
        if goal is None:
            return None
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        prefix = []
        for point in path:
            if point not in reachable:
                break
            prefix.append(point)
        if len(prefix) < len(path):
            # The prefix is cut by an in-window softwall (an ordinary closed door/non-explosive solid barrel, passable for the BFS): issue
            # the existing approach/open command for act_controller_operate instead of stopping the
            # waypoint one tile before the softwall; otherwise the same waypoint is picked every time and it circles forever (confirmed on the seed7002
            # hunt: the same waypoint before a barrel repeated 25 times). If monsters block the door, it still walks to the waypoint.
            blocker = path[len(prefix)]
            bi = idx(*blocker)
            if (abs(blocker[0] - px) <= r and abs(blocker[1] - py) <= r
                    and door[bi] and not mon[bi]):
                approach = prefix[-1] if prefix else start
                if approach == start:
                    return ("open", blocker[0], blocker[1])
                return ("approach", approach[0], approach[1])
        if not prefix:
            return None
        for point in reversed(prefix):
            if point not in blocked_targets:
                return ("frontier", point[0], point[1])
        self._explore_blocked_targets.difference_update(prefix)
        return ("frontier", prefix[-1][0], prefix[-1][1])

    def _macro_explore(
        self,
        max_beats: int = 12,
        *,
        controller_snapshot: _ControllerSnapshot | None = None,
        execution_audit: dict | None = None,
    ):
        """Explore macro: executes only the one fixed command chosen by the radius-12 snapshot at observation time.

        Story progress is the exclusive authority of a11/the manager; in story state a10 fails closed to wait.
        Ordinary exploration never walks onto up/down triggers and never operates progression coordinates. Once a door is opened
        or the approach stance is reached, the policy gets control back immediately and the next snapshot decides the next step; a second
        local_map re-plan after the same observation is forbidden.
        """
        if self._raw.get("progression_targets"):
            self._explore_target = None
            return self._wait_step()
        raw = self._raw
        snapshot = (
            controller_snapshot
            if controller_snapshot is not None
            else self._capture_controller_snapshot(raw)
        )
        start_scene = _scene_identity(raw)
        command = self._plan_explore_step(
            raw,
            blocked_softwalls=set(),
            controller_snapshot=snapshot,
        )
        if command is None:
            return self._wait_step()

        protected = sorted(snapshot.protected_tiles)
        path = None
        if command[0] != "open":
            path = self._plan_controller_path(
                snapshot,
                command[1],
                command[2],
                avoid_monsters=True,
                allow_softwalls=False,
            )
            if path is None:
                path = self._plan_controller_path(
                    snapshot,
                    command[1],
                    command[2],
                    avoid_monsters=False,
                    allow_softwalls=False,
                )
            if path is None:
                return self._wait_step()
        elif max(
            abs(int(raw["player_x"]) - int(command[1])),
            abs(int(raw["player_y"]) - int(command[2])),
        ) > 1:
            raise RuntimeError(
                "controller snapshot planned a non-adjacent operate")

        pi = 0
        active_step: tuple[int, int] | None = None
        command_issued = False
        last_pos = (int(raw["player_x"]), int(raw["player_y"]))
        stall = 0
        beats = 0
        while beats < max_beats:
            if command[0] == "open":
                if not command_issued and self._decision_idle(raw):
                    accepted = bridge.act_controller_operate(
                        command[1],
                        command[2],
                        snapshot.player_x,
                        snapshot.player_y,
                        CONTROLLER_SNAPSHOT_RADIUS,
                    )
                    if execution_audit is not None:
                        accepted = self._record_native_execution(
                            execution_audit, accepted, "action10 open")
                    else:
                        accepted = int(accepted) == 1
                    if not accepted:
                        break
                    command_issued = True
            elif active_step is None and self._decision_idle(raw):
                if pi >= len(path):
                    break
                nx, ny, is_softwall = path[pi]
                if is_softwall:
                    raise RuntimeError(
                        "action10 ordinary walking path unexpectedly crosses an unopened softwall")
                if abs(nx - int(raw["player_x"])) + abs(
                        ny - int(raw["player_y"])) != 1:
                    raise RuntimeError(
                        "controller snapshot path is not 4-direction adjacent steps")
                accepted = bridge.act_explore_walk(
                    nx,
                    ny,
                    protected,
                    snapshot.player_x,
                    snapshot.player_y,
                    CONTROLLER_SNAPSHOT_RADIUS,
                )
                if execution_audit is not None:
                    accepted = self._record_native_execution(
                        execution_audit, accepted, "action10 walk")
                else:
                    accepted = int(accepted) == 1
                if not accepted:
                    break
                active_step = (nx, ny)

            raw = self._step_native()
            beats += 1
            pos = (int(raw["player_x"]), int(raw["player_y"]))
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            nd = self._nearest_dist(raw)
            if (
                raw["dead"]
                or _scene_identity(raw) != start_scene
                or self._reflex_eligible(raw)
                or (nd is not None and nd <= 6)
            ):
                break

            if command[0] == "open":
                if bridge.probe_tile(command[1], command[2])["walkable"]:
                    self._softwalls_opened += 1
                    self._exploration_progress += 1
                    break
                interaction_busy = (
                    self._movement_engine_busy(raw)
                    or int(raw.get("player_mode", -1)) == bridge.PM_ATTACK
                    or int(raw.get("dest_action", bridge.ACTION_NONE))
                    != bridge.ACTION_NONE
                )
            else:
                interaction_busy = self._movement_engine_busy(raw)
                if active_step is not None and pos == active_step:
                    pi += 1
                    active_step = None
                    stall = 0
                    reached = (
                        pos == (int(command[1]), int(command[2]))
                        if command[0] == "approach"
                        else max(
                            abs(pos[0] - int(command[1])),
                            abs(pos[1] - int(command[2])),
                        ) <= 1
                    )
                    if reached:
                        if (
                            command[0] == "frontier"
                            and getattr(self, "_explore_target", None)
                            == (int(command[1]), int(command[2]))
                        ):
                            self._explore_target = None
                        break
                    last_pos = pos
                    continue

            if pos != last_pos or interaction_busy:
                stall = 0
            else:
                stall += 1
                if stall == 3 and self._decision_idle(raw):
                    if command[0] == "open":
                        accepted = bridge.act_controller_operate(
                            command[1],
                            command[2],
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                "action10 reopen")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break
                    elif active_step is not None:
                        accepted = bridge.act_explore_walk(
                            active_step[0],
                            active_step[1],
                            protected,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            self._record_native_execution(
                                execution_audit, accepted,
                                "action10 rewalk")
                if stall >= 6:
                    break
            if command[0] == "frontier" and stall >= 2:
                failed = (int(command[1]), int(command[2]))
                self._explore_blocked_targets.add(failed)
                if getattr(self, "_explore_target", None) == failed:
                    self._explore_target = None
                break
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    _DESCEND_RADIUS = 112  # the planning window covers the whole map (dungeon 112×112): some levels' connecting corridors loop widely,
                           # and a 40-tile window once missed the western detour on seed 9005. Each key press plans only once,
                           # with one C++ call producing the map in milliseconds, so the global optimum is worth it

    def _plan_descend_path(self, raw, sx, sy, avoid_monsters: bool = False):
        """Whole-window 4-direction BFS (closed doors count as passable); returns the path to "the reachable tile nearest the stairs"
        as [(x, y, is_closed_door), ...] (without the start). None = no tile in the reachable region is closer to the stairs than
        the current one (truly trapped). 4 directions guarantee the engine pathing accepts every segment (it refuses diagonal
        cuts across wall corners); a greedy "only pick nearer tiles" dies in concave mazes, while BFS allows detours first.

        Fire/acid pools and other hazards and explosive-softwalls are always walls; ``door``
        includes both ordinary doors and blocking barrels, so an explosive barrel must never be mistaken for a safe softwall.

        With avoid_monsters=True monster tiles count as walls (v14 fix: engine pathing refuses to go through monsters,
        and a monster-blind planner facing an idle monster blocking a corridor falls into a stalled loop of "re-planning the same path";
        caught red-handed with a 1-HP skeleton on seed 9024; callers should fall back to
        avoid_monsters=False when this returns None, so the worst case degrades to the old stall-and-hand-back)."""
        px, py = raw["player_x"], raw["player_y"]
        r = self._DESCEND_RADIUS
        side = 2 * r + 1
        lm = bridge.local_map(radius=r)
        walk, door = lm["walkable"], lm["door"]
        hazard = lm.get("hazard", [0] * (side * side))
        explosive = lm.get(
            "explosive_softwall", [0] * (side * side))
        mon = lm["monster"] if avoid_monsters else None

        def idx(tx, ty):
            return (ty - py + r) * side + (tx - px + r)

        start = (px, py)
        prev = {start: None}
        depth = {start: 0}
        best = (max(abs(sx - px), abs(sy - py)), 0, start)
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + ddx, cy + ddy
                if abs(nx - px) > r or abs(ny - py) > r or (nx, ny) in prev:
                    continue
                i = idx(nx, ny)
                if hazard[i] or explosive[i]:
                    continue
                if not walk[i] and not door[i]:
                    continue
                if mon is not None and mon[i]:
                    continue  # monster tile = wall (engine pathing refuses to go through monsters; see the docstring)
                prev[(nx, ny)] = (cx, cy)
                depth[(nx, ny)] = depth[(cx, cy)] + 1
                d_stairs = max(abs(sx - nx), abs(sy - ny))
                if (d_stairs, depth[(nx, ny)]) < best[:2]:
                    best = (d_stairs, depth[(nx, ny)], (nx, ny))
                queue.append((nx, ny))
        if best[2] == start:
            return None
        path = []
        cur = best[2]
        while cur != start:
            path.append((cur[0], cur[1], bool(door[idx(*cur)])))
            cur = prev[cur]
        path.reverse()
        return path

    @staticmethod
    def _plan_controller_path(
        snapshot: _ControllerSnapshot,
        target_x: int,
        target_y: int,
        *,
        avoid_monsters: bool,
        allow_softwalls: bool = True,
    ):
        """Plan only from the observation-bound radius-12 controller snapshot.

        Targets beyond the window are not deleted: BFS advances to the locally
        reachable cell that most reduces Chebyshev distance, then the next
        policy decision receives a re-centered snapshot.  This preserves
        ordinary long-range reachability without consulting an unobserved
        radius-112 map behind the Worker's back.
        """
        px, py = snapshot.player_x, snapshot.player_y
        radius = CONTROLLER_SNAPSHOT_RADIUS
        side = CONTROLLER_SNAPSHOT_SIDE

        def index(x, y):
            return (y - py + radius) * side + (x - px + radius)

        start = (px, py)
        predecessors = {start: None}
        depths = {start: 0}
        best = (
            max(abs(int(target_x) - px), abs(int(target_y) - py)),
            0,
            start,
        )
        queue = deque([start])
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                point = (nx, ny)
                if (
                    abs(nx - px) > radius
                    or abs(ny - py) > radius
                    or point in predecessors
                    or point in snapshot.protected_tiles
                ):
                    continue
                i = index(nx, ny)
                if snapshot.hazard[i] or snapshot.explosive_softwall[i]:
                    continue
                if (
                    not snapshot.walkable[i]
                    and not (
                        allow_softwalls and snapshot.softwall[i])
                ):
                    continue
                if avoid_monsters and snapshot.physical_monster[i]:
                    continue
                predecessors[point] = (cx, cy)
                depths[point] = depths[(cx, cy)] + 1
                distance = max(
                    abs(int(target_x) - nx), abs(int(target_y) - ny))
                if (distance, depths[point]) < best[:2]:
                    best = (distance, depths[point], point)
                queue.append(point)
        if best[2] == start:
            return None
        path = []
        current = best[2]
        while current != start:
            path.append((
                current[0],
                current[1],
                bool(snapshot.softwall[index(*current)]),
            ))
            current = predecessors[current]
        path.reverse()
        return path

    @staticmethod
    def _descend_blocker_health(monster):
        if "hp_fixed_hi" in monster and "hp_fixed_lo" in monster:
            return (int(monster["hp_fixed_hi"]) << 16) + int(monster["hp_fixed_lo"])
        return 64 * int(monster.get("hp", 0))

    @classmethod
    def _descend_observed_blocker(cls, raw, target_x, target_y):
        """Bind only a visible live hostile occupying this exact next edge.

        A moving monster can reserve the edge with future while its tile still
        lies elsewhere. Adjacency uses future on both sides, matching native
        radius-one attack validation; this never permits a chase.
        """
        px = int(raw.get("future_x", raw["player_x"]))
        py = int(raw.get("future_y", raw["player_y"]))
        edge = (int(target_x), int(target_y))
        for monster in raw.get("monsters", ()):
            if (not bool(monster.get("visible", False))
                    or bool(monster.get("is_invalid", False))
                    or bool(monster.get("is_player_minion", False))
                    or int(monster.get("type", -1)) == 109
                    or cls._descend_blocker_health(monster) <= 0):
                continue
            tile = (int(monster["x"]), int(monster["y"]))
            future = (int(monster.get("future_x", tile[0])),
                      int(monster.get("future_y", tile[1])))
            if edge in (tile, future) and max(abs(future[0] - px), abs(future[1] - py)) <= 1:
                return monster
        return None

    def _macro_descend_blocker(self, raw, blocker, *, beats_before,
                              macro_deadline, start_scene, execution_audit):
        """One native adjacent attack command, then return for real replanning.

        The command may span multiple ordinary attack frames within the core
        budget; this does not promise exactly one swing or one damage event.

        There is no cross-action recovery state. The attack core shares this
        a11 call's remaining budget; normal idle settle is charged separately
        and remains bounded by the live episode deadline. Receipts distinguish
        native acceptance, observed damage and actual committed displacement.
        """
        start_clock = self._resource_actual_microsteps
        initial_position = (int(raw["player_x"]), int(raw["player_y"]))
        key = self._monster_generation_key(blocker)
        initial_hp = self._descend_blocker_health(blocker)
        receipt = {"source": "dive-blocker:adjacent-v1", "monster_generation": key,
                   "before_microstep": start_clock, "macro_deadline": int(macro_deadline),
                   "accepted": False, "monster_hp_fixed_before": initial_hp,
                   "monster_hp_fixed_after": initial_hp, "damage_observed": False,
                   "target_alive_observed": True, "position_changed": False,
                   "after_microstep": start_clock, "micro_steps": 0,
                   "reason": "deadline"}
        self._dive_blocker_audit = receipt

        def time_available():
            return self._resource_actual_microsteps < min(macro_deadline, self.max_steps)

        def canonical_ready(observed):
            return bool(observed.get("resource_state", {}).get("readiness", {}).get("ready", False))

        if not time_available() or not canonical_ready(raw):
            receipt["reason"] = "deadline" if not time_available() else "readiness_lost"
            return self._finish_macro(raw, beats_before, start_scene)
        result = bridge.act_controller_attack_monster(
            int(blocker["id"]), int(raw["player_x"]), int(raw["player_y"]), 1)
        if execution_audit is not None:
            accepted = self._record_native_execution(execution_audit, result, "action11 blocker attack")
        else:
            accepted = int(result) == 1
        receipt["accepted"] = bool(accepted)
        if not accepted:
            bridge.act_wait()
        receipt["reason"] = "attack_rejected" if not accepted else "macro_cap"
        beats = int(beats_before)
        while time_available():
            raw = self._step_native()
            beats += 1
            if _scene_identity(raw) == start_scene:
                self._record_visit((raw["player_x"], raw["player_y"]))
            if raw.get("dead") or raw.get("game_over") or raw.get("victory"):
                receipt["reason"] = "terminal"
                break
            if _scene_identity(raw) != start_scene:
                receipt["reason"] = "scene_changed"
                break
            target = next((m for m in raw.get("monsters", ())
                           if self._monster_generation_key(m) == key), None)
            if target is not None:
                current_hp = self._descend_blocker_health(target)
                receipt["monster_hp_fixed_after"] = current_hp
                receipt["damage_observed"] = receipt["damage_observed"] or current_hp < initial_hp
                receipt["target_alive_observed"] = current_hp > 0
            else:
                # Disappearance / generation replacement is not proof of a kill.
                receipt["monster_hp_fixed_after"] = None
                receipt["target_alive_observed"] = None
            if not canonical_ready(raw):
                receipt["reason"] = "readiness_lost"
                break
            if not accepted:
                break
            if target is None or self._descend_blocker_health(target) <= 0:
                receipt["reason"] = "target_changed" if target is None else "target_dead_observed"
                break
            if (not bool(target.get("visible", False))
                    or self._engage_distance(raw, target) > 1):
                receipt["reason"] = "target_left_adjacency"
                break
            if self._decision_idle(raw):
                receipt["reason"] = "swing_complete"
                break
        receipt["after_microstep"] = self._resource_actual_microsteps
        receipt["micro_steps"] = self._resource_actual_microsteps - start_clock
        receipt["position_changed"] = (
            int(raw["player_x"]), int(raw["player_y"])) != initial_position
        return self._finish_macro(raw, beats, start_scene)

    def _macro_descend(
        self,
        max_beats: int = 12,
        *,
        execution_audit: dict | None = None,
    ):
        """Descend macro: plan once with the global BFS, then walk to the down stairs in 4-direction adjacent safe steps one by one.

        Doors are operated only when already adjacent; walk/operate both use the native constrained controller entry points,
        forbidding CMD_WALKXY/CMD_OPOBJXY from computing a separate shortest path through hazards during execution.
        Dungeon rooms connect through doors, and a closed door looks like a wall in the walkable channel,
        so the path must still carry the door state explicitly.

        Spotting prey does not interrupt it (this is the deliberate withdraw key); a level change/death/persistent stall ends it early;
        after 12 ticks control returns naturally and the next key press re-plans. Fully deterministic, no random numbers.
        """
        if self._raw.get("progression_targets"):
            return self._macro_progression(
                max_beats=max_beats,
                execution_audit=execution_audit,
            )
        raw = self._raw
        transition = (bridge.WM_DIABRTNLVL if raw.get("is_set_level")
                      else bridge.WM_DIABNEXTLVL)
        stairs = [t for t in raw.get("triggers", [])
                  if t["msg"] == transition]
        if not stairs:
            return self._wait_step()
        px, py = raw["player_x"], raw["player_y"]
        st = min(stairs, key=lambda t: max(abs(t["x"] - px), abs(t["y"] - py)))
        sx, sy = st["x"], st["y"]
        start_scene = _scene_identity(raw)

        path = self._plan_descend_path(raw, sx, sy, avoid_monsters=True)
        if (getattr(self, "_descend_fallback_promotion", False)
                and path is not None):
            # E-fix A: the pocket reachable region of the monster-avoiding BFS can produce a partial path where "going back is closer",
            # forming a deterministic limit cycle with lenient planning (R8 seed 2122004:
            # a 6-window, 24-micro-step cycle pinned in place). Same as the existing rule
            # in _macro_progression: when the monster-avoiding path cannot reach the stairs tile, consult lenient planning, and promote it
            # only if its end point is strictly closer; when the stairs can be reached (remaining==0) the second BFS does not happen,
            # and the behaviour equals the old one bit for bit.
            def _remaining(candidate):
                if not candidate:
                    return max(abs(sx - int(px)), abs(sy - int(py)))
                tail = candidate[-1]
                return max(abs(sx - int(tail[0])), abs(sy - int(tail[1])))
            remaining = _remaining(path)
            if remaining > 0:
                fallback = self._plan_descend_path(raw, sx, sy)
                if fallback is not None and _remaining(fallback) < remaining:
                    path = fallback
        if path is None:
            path = self._plan_descend_path(raw, sx, sy)  # monsters seal the only route: fall back to the old behaviour
        if path is None:
            return self._wait_step()  # truly trapped: one tick in place, hand control back

        pi = 0            # path consumption pointer
        target = None     # (kind, x, y, path_index)
        center_x, center_y = int(px), int(py)
        stall = 0
        beats = 0
        last_pos = (px, py)
        blocker_recovery = (
            getattr(self, "dive_blocker_recovery", "off") == "adjacent-v1"
            and int(raw.get("dungeon_level", 0)) > 0
            and not bool(raw.get("is_set_level", False))
        )
        macro_deadline = (self._resource_actual_microsteps + max_beats
                          if blocker_recovery else None)
        for beats in range(1, max_beats + 1):
            if blocker_recovery:
                next_edge = (target[1:3] if target is not None else
                             path[pi][:2] if pi < len(path) else None)
                blocker = (self._descend_observed_blocker(raw, *next_edge)
                           if next_edge is not None else None)
                if blocker is not None:
                    return self._macro_descend_blocker(
                        raw, blocker, beats_before=beats - 1,
                        macro_deadline=macro_deadline, start_scene=start_scene,
                        execution_audit=execution_audit)
            if target is None:
                if pi >= len(path):
                    break  # path finished (happens when the nearest reachable tile is not the stairs); hand control back
                j = pi
                target = (
                    ("open", path[j][0], path[j][1], j)
                    if path[j][2]
                    else ("walk", path[j][0], path[j][1], j)
                )
                if target[0] == "open":
                    accepted = bridge.act_controller_operate(
                        target[1],
                        target[2],
                        center_x,
                        center_y,
                        self._DESCEND_RADIUS,
                    )
                else:
                    accepted = bridge.act_explore_walk(
                        target[1],
                        target[2],
                        (),
                        center_x,
                        center_y,
                        self._DESCEND_RADIUS,
                    )
                if execution_audit is not None:
                    accepted = self._record_native_execution(
                        execution_audit, accepted,
                        f"action11 descend {target[0]}")
                else:
                    accepted = int(accepted) == 1
                if not accepted:
                    break
            raw = self._step_native()
            pos = (raw["player_x"], raw["player_y"])
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            if raw["dead"] or _scene_identity(raw) != start_scene:
                break  # level changed (or died); step() resets footprints per level uniformly
            if self._reflex_eligible(raw):
                break
            if pos == (sx, sy):
                continue  # standing on the stairs tile, waiting for the level-change trigger; standing still is not a stall
            if target[0] == "open":
                # Door-type target: complete only when the door tile really becomes walkable (touching it != opened, the animation takes a few ticks)
                if bridge.probe_tile(target[1], target[2])["walkable"]:
                    path[target[3]] = (target[1], target[2], False)
                    pi = target[3]  # continue consuming the path from the door tile
                    target = None
                    stall = 0
                    last_pos = pos
                    continue
            elif pos == (target[1], target[2]):
                # Every native walk request is one exact controller edge.
                # The old multi-hop macro accepted "within one tile", which
                # could advance pi before the committed animation had landed.
                pi = target[3] + 1  # waypoint reached, continue with the next segment
                target = None
                stall = 0
                last_pos = pos
                continue
            if pos == last_pos:
                stall += 1
                if stall == 3 and target is not None:
                    # The command may have been interrupted (path knocked off by a monster, etc.): resend once in place
                    if target[0] == "open":
                        accepted = bridge.act_controller_operate(
                            target[1],
                            target[2],
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    else:
                        accepted = bridge.act_explore_walk(
                            target[1],
                            target[2],
                            (),
                            center_x,
                            center_y,
                            self._DESCEND_RADIUS,
                        )
                    if execution_audit is not None:
                        self._record_native_execution(
                            execution_audit, accepted,
                            f"action11 descend retry {target[0]}")
                if stall >= 6:
                    break  # still no progress after the resend -> hand control back, re-plan on the next key press
            else:
                stall = 0
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    def _macro_pickup(
        self,
        kind: str = "heal",
        max_beats: int = 12,
        *,
        controller_snapshot: _ControllerSnapshot | None = None,
        action14_audit: dict | None = None,
        execution_audit: dict | None = None,
    ):
        """Pickup macro (v13 potions / v14 gear): consumes only the target and radius-12 map bound to the observation.

        Walks toward the target item along the fixed snapshot path, opening doors, and commits only after standing exactly on the target tile.
        The wrapper and both native pinned entry points require the player's future to be exactly on that tile, locking the whole
        movement trajectory onto the path verified step by step by the fixed snapshot; the public convenience entry points likewise cannot
        pick a nearby item to bypass this proof.

        The native commit re-checks the active id, position, seed, create info and base id again.
        Gear also recomputes PlanGearUpgrade, copies the whole set of body slots, replaces atomically and
        re-verifies with CalcPlrInv, rolling back completely on failure; no AutoEquip/backpack fallback. A successful
        0/1 native receipt synchronously observes right after the commit and records the strictly positive whole-set
        combat delta, so later hits/durability loss cannot tamper with this causal evidence. The macro ends when the target
        disappears or becomes invalid; potions also have the quick criterion of a rising belt count. Death/level change/path
        exhaustion/persistent stall end it early; after 12 ticks control returns naturally and the next key press re-plans.
        Fully deterministic, no random numbers.
        """
        raw = self._raw
        flag = "heal" if kind == "heal" else "gear"
        snapshot = (
            controller_snapshot
            if controller_snapshot is not None
            else self._capture_controller_snapshot(raw)
        )
        target_point = (
            snapshot.heal_target if kind == "heal"
            else snapshot.gear_target
        )
        if target_point is None or (
                kind == "heal" and self._belt_free_slots(raw) <= 0):
            return self._wait_step()

        if kind == "gear":
            utility = gear_combat_utility_value(
                raw, "action14_macro_start")
            if action14_audit is None:
                action14_audit = {
                    "accepted": False,
                    "commit_attempts": 0,
                    "utility_before": utility,
                    "utility_after": utility,
                    "utility_delta": 0,
                }
            expected_audit_keys = {
                "accepted", "commit_attempts", "utility_before",
                "utility_after", "utility_delta",
            }
            if (
                set(action14_audit) != expected_audit_keys
                or action14_audit["accepted"] is not False
                or int(action14_audit["commit_attempts"]) != 0
                or int(action14_audit["utility_delta"]) != 0
            ):
                raise RuntimeError(
                    "action14_audit initial state corrupt")
        elif action14_audit is not None:
            raise RuntimeError(
                "action14_audit may only be used by the gear pickup macro")

        px, py = int(raw["player_x"]), int(raw["player_y"])
        hx, hy = target_point.x, target_point.y

        def act():
            if kind == "heal":
                accepted = bridge.act_pickup_at(
                    target_point.active_id,
                    hx,
                    hy,
                    target_point.seed_hi,
                    target_point.seed_lo,
                    target_point.create_info,
                    target_point.base_id,
                )
            else:
                utility_before = gear_combat_utility_value(
                    raw, "action14_before_native_commit")
                accepted_raw = bridge.act_pickup_gear_at(
                    target_point.active_id,
                    hx,
                    hy,
                    target_point.seed_hi,
                    target_point.seed_lo,
                    target_point.create_info,
                    target_point.base_id,
                )
                if not isinstance(
                        accepted_raw, (bool, np.bool_, int, np.integer)):
                    raise RuntimeError(
                        "action14 native receipt must be an integer 0/1, got "
                        f"{accepted_raw!r}")
                accepted = int(accepted_raw)
                if accepted not in (0, 1):
                    raise RuntimeError(
                        "action14 native receipt must be 0/1, got "
                        f"{accepted}")
                action14_audit["commit_attempts"] += 1
                if accepted:
                    if action14_audit["accepted"]:
                        raise RuntimeError(
                            "action14 duplicate successful commit within the same policy action")
                    committed = bridge.observe()
                    if getattr(self, "resource_preserve_equipment_readiness", False):
                        self._validate_native_resource_flags(committed)
                    utility_after = gear_combat_utility_value(
                        committed, "action14_after_native_commit")
                    utility_delta = utility_after - utility_before
                    if utility_delta <= 0:
                        raise RuntimeError(
                            "action14 whole-set combat power did not strictly increase after native acceptance: "
                            f"{utility_before}->{utility_after}")
                    action14_audit.update({
                        "accepted": True,
                        "utility_before": utility_before,
                        "utility_after": utility_after,
                        "utility_delta": utility_delta,
                    })
            if execution_audit is not None:
                return int(self._record_native_execution(
                    execution_audit, accepted,
                    f"action{13 if kind == 'heal' else 14} pickup"))
            return int(accepted)

        start_belt = int(raw["belt_heals"])
        start_scene = _scene_identity(raw)
        pickup_radius = 0
        near0 = max(abs(hx - px), abs(hy - py)) <= pickup_radius
        path = self._plan_controller_path(
            snapshot, hx, hy, avoid_monsters=True)
        if path is None:
            path = self._plan_controller_path(
                snapshot, hx, hy, avoid_monsters=False)
        center_index = CONTROLLER_SNAPSHOT_CELLS // 2
        same_tile_safe = (
            (px, py) == (hx, hy)
            and not snapshot.hazard[center_index]
            and not snapshot.explosive_softwall[center_index]
        )
        item_path_complete = (
            same_tile_safe
            or (
                path is not None
                and bool(path)
                and (int(path[-1][0]), int(path[-1][1])) == (hx, hy)
            )
        )
        if (
            path is None and not near0
            or not item_path_complete
        ):
            return self._wait_step()

        protected = sorted(snapshot.protected_tiles)
        pi = 0
        target: tuple[str, int, int] | None = None
        stall = 0
        beats = 0
        last_pos = (px, py)
        for beats in range(1, max_beats + 1):
            if target is None and self._decision_idle(raw):
                cur = (int(raw["player_x"]), int(raw["player_y"]))
                near = (
                    max(abs(hx - cur[0]), abs(hy - cur[1]))
                    <= pickup_radius
                )
                door_pending = bool(path) and any(
                    point[2] for point in path[pi:])
                if near and not door_pending:
                    target = ("pick", hx, hy)
                    act()
                elif path is None or pi >= len(path):
                    break
                else:
                    nx, ny, is_softwall = path[pi]
                    if abs(nx - cur[0]) + abs(ny - cur[1]) != 1:
                        raise RuntimeError(
                            "controller pickup path is not 4-direction adjacent steps")
                    target = (
                        ("open", nx, ny)
                        if is_softwall else ("walk", nx, ny)
                    )
                    if is_softwall:
                        accepted = bridge.act_controller_operate(
                            nx,
                            ny,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} open")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break
                    else:
                        accepted = bridge.act_explore_walk(
                            nx,
                            ny,
                            protected,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} walk")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break

            raw = self._step_native()
            pos = (int(raw["player_x"]), int(raw["player_y"]))
            if _scene_identity(raw) == start_scene:
                self._record_visit(pos)
            if raw["dead"] or _scene_identity(raw) != start_scene:
                break
            if self._reflex_eligible(raw):
                break
            if kind == "heal" and int(raw["belt_heals"]) > start_belt:
                break

            tracked_items = [
                it for it in raw.get("floor_items", ())
                if (
                    int(it.get("active_id", -1))
                    == target_point.active_id
                    and int(it.get("seed_hi", -1))
                    == target_point.seed_hi
                    and int(it.get("seed_lo", -1))
                    == target_point.seed_lo
                    and int(it.get("create_info", -1))
                    == target_point.create_info
                    and int(it.get("base_id", -1))
                    == target_point.base_id
                )
            ]
            if not tracked_items:
                break
            if not any(bool(it.get(flag)) for it in tracked_items):
                break
            if not any(it.get("visible", True) for it in tracked_items):
                break

            if target is not None and target[0] == "pick":
                if self._decision_idle(raw):
                    target = None
            elif target is not None and target[0] == "open":
                if bridge.probe_tile(target[1], target[2])["walkable"]:
                    path[pi] = (target[1], target[2], False)
                    target = None
                    stall = 0
                    last_pos = pos
                    continue
            elif (
                target is not None
                and target[0] == "walk"
                and pos == (target[1], target[2])
            ):
                pi += 1
                target = None
                stall = 0
                last_pos = pos
                continue

            interaction_busy = (
                not self._decision_idle(raw)
                or self._movement_engine_busy(raw)
            )
            if pos != last_pos or interaction_busy:
                stall = 0
            else:
                stall += 1
                if (
                    stall == 3
                    and target is not None
                    and self._decision_idle(raw)
                ):
                    if target[0] == "open":
                        accepted = bridge.act_controller_operate(
                            target[1],
                            target[2],
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            accepted = self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} reopen")
                        else:
                            accepted = int(accepted) == 1
                        if not accepted:
                            break
                    elif target[0] == "pick":
                        act()
                    else:
                        accepted = bridge.act_explore_walk(
                            target[1],
                            target[2],
                            protected,
                            snapshot.player_x,
                            snapshot.player_y,
                            CONTROLLER_SNAPSHOT_RADIUS,
                        )
                        if execution_audit is not None:
                            self._record_native_execution(
                                execution_audit, accepted,
                                f"action{13 if kind == 'heal' else 14} rewalk")
                if stall >= 6:
                    break
            last_pos = pos
        return self._finish_macro(raw, beats, start_scene)

    @classmethod
    def _nearest_monster(cls, obs):
        px, py = obs["player_x"], obs["player_y"]
        best, best_d = None, None
        for m in cls._policy_monsters(obs):
            d = abs(m["x"] - px) + abs(m["y"] - py)
            if best_d is None or d < best_d:
                best, best_d = m, d
        return best

    @classmethod
    def _nearest_dist(cls, raw):
        px, py = raw["player_x"], raw["player_y"]
        dists = [
            max(abs(m["x"] - px), abs(m["y"] - py))
            for m in cls._policy_monsters(raw)
        ]
        return min(dists) if dists else None

    @classmethod
    def _player_approach_delta(
        cls,
        prev,
        cur,
        *,
        target_generation_key: tuple[int, int, int] | None = None,
    ) -> int | None:
        """Player-only distance change to one fixed surviving generation.

        Recomputing two independent nearest-monster minima lets monster motion
        and target switching masquerade as player progress.  By default select
        the nearest generation at the start; action 9 instead supplies the
        canonical generation selected by its installed controller snapshot.
        Hold that generation's current endpoint fixed and compare only the
        player's two positions.  A killed/despawned/reused-slot target receives
        no approach shaping; combat reward already credits a real kill.
        """
        previous_monsters = cls._policy_monsters(prev)
        if not previous_monsters:
            return None
        px0, py0 = int(prev["player_x"]), int(prev["player_y"])
        if target_generation_key is None:
            target = min(
                previous_monsters,
                key=lambda monster: (
                    max(
                        abs(int(monster["x"]) - px0),
                        abs(int(monster["y"]) - py0),
                    ),
                    cls._monster_generation_key(monster),
                ),
            )
        else:
            target = next(
                (
                    monster
                    for monster in previous_monsters
                    if cls._monster_generation_key(monster)
                    == target_generation_key
                ),
                None,
            )
            if target is None:
                return None
        target_key = cls._monster_generation_key(target)
        current = next(
            (
                monster
                for monster in cls._policy_monsters(cur)
                if cls._monster_generation_key(monster) == target_key
            ),
            None,
        )
        if current is None:
            return None
        tx, ty = int(current["x"]), int(current["y"])
        px1, py1 = int(cur["player_x"]), int(cur["player_y"])
        return (
            max(abs(tx - px0), abs(ty - py0))
            - max(abs(tx - px1), abs(ty - py1))
        )

    @classmethod
    def _disappeared_monster_generations(cls, prev, cur) -> int:
        """Count same-scene monster lifetimes that ended during a transition."""
        current = {
            cls._monster_generation_key(monster)
            for monster in cur.get("monsters", ())
        }
        return sum(
            cls._monster_generation_key(monster) not in current
            for monster in prev.get("monsters", ())
        )

    @staticmethod
    def _native_monster_kill_delta(prev, cur) -> int | None:
        """Validated delta of the process/hero-monotonic native kill ledger."""
        before_present = "monster_kill_total" in prev
        after_present = "monster_kill_total" in cur
        if not before_present and not after_present:
            return None
        if before_present != after_present:
            raise RuntimeError(
                "monster_kill_total appears at only one end of the transition")

        def value(raw, label):
            candidate = raw["monster_kill_total"]
            if isinstance(candidate, bool):
                raise RuntimeError(
                    f"{label}.monster_kill_total must not be a bool")
            try:
                integer = int(candidate)
                numeric = float(candidate)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"{label}.monster_kill_total must be a non-negative integer") from exc
            if (
                not math.isfinite(numeric)
                or numeric != float(integer)
                or integer < 0
            ):
                raise RuntimeError(
                    f"{label}.monster_kill_total must be a non-negative integer")
            return integer

        before = value(prev, "prev")
        after = value(cur, "cur")
        if after < before:
            raise RuntimeError(
                "native monster_kill_total went backwards within the episode: "
                f"{before}->{after}")
        return after - before

    def _reset_combat_ledger(self, raw) -> None:
        """Zero the current scene's HP lines; called on reset/map change, never across lifetime domains."""
        self._combat_hp_floor = {
            self._monster_generation_key(m): (
                max(0, int(m["hp"])),
                max(1, int(m["max_hp"])),
            )
            for m in raw.get("monsters", ())
        }

    def _combat_reward(self, prev, cur) -> float:
        """Pay damage as the potential difference of each monster's new lowest HP in this scene; intervals already paid are never paid again.

        The ledger keeps only lifetimes still alive; a newly spawned monster opens its account at its first observed HP line.
        When a generation disappears, only its unpaid damage potential is settled to 0; kill units come from the native
        monotone monster_kill_total rather than being inferred from active-list disappearance (only old synthetic fixtures
        lacking that field fall back to counting generation disappearances). When a slot is reused in the same tick,
        the old and new generations are settled separately, and the new monster's HP is never taken as the old monster healing.
        """
        if not hasattr(self, "_combat_hp_floor"):
            self._combat_hp_floor = {}
        if not self._combat_hp_floor and prev.get("monsters"):
            self._reset_combat_ledger(prev)

        prev_by_generation = {
            self._monster_generation_key(m): m
            for m in prev.get("monsters", ())
        }
        cur_by_generation = {
            self._monster_generation_key(m): m
            for m in cur.get("monsters", ())
        }
        next_floor: dict[tuple[int, int, int], tuple[int, int]] = {}
        r = 0.0

        for generation_key, m in prev_by_generation.items():
            ledger_low, ledger_max = self._combat_hp_floor.get(
                generation_key,
                (max(0, int(m["hp"])), max(1, int(m["max_hp"]))),
            )
            # If the caller advanced the bridge directly between two env.step calls and prev is already below
            # the ledger, only lower the zero point; do not pay for damage that happened outside the environment.
            paid_floor = min(ledger_low, max(0, int(m["hp"])))
            denominator = max(ledger_max, int(m["max_hp"]), 1)
            current = cur_by_generation.get(generation_key)
            hp_after = max(0, int(current["hp"])) if current is not None else 0
            new_floor = min(paid_floor, hp_after)
            newly_credited = paid_floor - new_floor
            if newly_credited > 0:
                # Pay the difference of one bounded health potential.  The
                # old endpoint multiplier overpaid a 100→0 one-shot relative
                # to the exact same damage split across several transitions,
                # making reward depend on ticks_per_step and attack cadence.
                q_before = paid_floor / denominator
                q_after = new_floor / denominator
                r += (
                    0.75 * (q_before - q_after)
                    - 0.125 * (
                        q_before * q_before - q_after * q_after
                    )
                )
            if current is None:
                continue
            next_floor[generation_key] = (
                new_floor,
                max(denominator, int(current["max_hp"]), 1),
            )

        # Monsters newly spawned/first entering the active list this tick cannot claim a reward for the low HP seen at first observation.
        for generation_key, m in cur_by_generation.items():
            if generation_key not in prev_by_generation:
                next_floor[generation_key] = (
                    max(0, int(m["hp"])),
                    max(1, int(m["max_hp"])),
                )

        self._combat_hp_floor = next_floor
        native_kills = self._native_monster_kill_delta(prev, cur)
        r += (
            native_kills
            if native_kills is not None
            else self._disappeared_monster_generations(prev, cur)
        )
        return r

    @staticmethod
    def _progression_effect_signature(raw) -> tuple:
        """Stable visible progression identity for causal action receipts."""
        return tuple(sorted(
            (
                str(target.get("kind", "")),
                str(target.get("action", "")),
                int(target.get("x", 0)),
                int(target.get("y", 0)),
                int(target.get("goal_x", 0)),
                int(target.get("goal_y", 0)),
                bool(target.get("exact", False)),
            )
            for target in raw.get("progression_targets", ())
        ))

    def _action_effect_reasons(
        self,
        prev,
        cur,
        *,
        requested_action: int,
        engage_target_generation_key: tuple[int, int, int] | None,
        native_kills: int,
        exploration_before: int,
        softwalls_before: int,
        drink_audit: dict | None,
        action14_audit: dict | None,
    ) -> tuple[str, ...]:
        """Return only effects causally attributable to the requested action.

        Enemy motion and player damage are deliberately absent.  They are
        world consequences, not proof that an explore/attack/pickup request
        executed, and previously let stalled requests masquerade as progress.
        """
        action = int(requested_action)
        if action == 0:
            return ()

        reasons: list[str] = []
        scene_changed = _scene_identity(cur) != _scene_identity(prev)
        position_changed = (
            (
                int(cur.get("player_x", 0)),
                int(cur.get("player_y", 0)),
                int(cur.get("future_x", cur.get("player_x", 0))),
                int(cur.get("future_y", cur.get("player_y", 0))),
            )
            != (
                int(prev.get("player_x", 0)),
                int(prev.get("player_y", 0)),
                int(prev.get("future_x", prev.get("player_x", 0))),
                int(prev.get("future_y", prev.get("player_y", 0))),
            )
        )
        if scene_changed:
            reasons.append("scene")

        if 1 <= action <= 8:
            if position_changed:
                reasons.append("move")
        elif action == 9:
            if position_changed:
                reasons.append("move")
            if int(native_kills) > 0:
                reasons.append("kill")
            target_key = engage_target_generation_key
            if target_key is not None:
                previous = next(
                    (
                        monster
                        for monster in prev.get("monsters", ())
                        if self._monster_generation_key(monster) == target_key
                    ),
                    None,
                )
                current = next(
                    (
                        monster
                        for monster in cur.get("monsters", ())
                        if self._monster_generation_key(monster) == target_key
                    ),
                    None,
                )
                if (
                    previous is not None
                    and current is not None
                    and int(current.get("hp", 0))
                    < int(previous.get("hp", 0))
                ):
                    reasons.append("target_damage")
                elif (
                    previous is not None
                    and current is None
                    and int(native_kills) > 0
                ):
                    reasons.append("target_removed")
        elif action == 10:
            if position_changed:
                reasons.append("move")
            if int(getattr(
                    self, "_exploration_progress", 0)) > int(
                        exploration_before):
                reasons.append("exploration")
            if int(getattr(self, "_softwalls_opened", 0)) > int(
                    softwalls_before):
                reasons.append("softwall")
        elif action == 11:
            if position_changed:
                reasons.append("move")
            recovery = getattr(self, "_dive_blocker_audit", None)
            if (getattr(self, "dive_blocker_recovery", "off") == "adjacent-v1"
                    and recovery is not None
                    and recovery.get("accepted") is True
                    and recovery.get("damage_observed") is True):
                reasons.append("blocker_damage")
            if (
                self._progression_effect_signature(cur)
                != self._progression_effect_signature(prev)
            ):
                reasons.append("progression")
        elif action == 12:
            if (
                isinstance(drink_audit, dict)
                and drink_audit.get("consumed") is True
            ):
                reasons.append("drink")
        elif action == 13:
            if (
                self._belt_free_slots(cur) < self._belt_free_slots(prev)
                or int(cur.get("belt_heals", 0))
                > int(prev.get("belt_heals", 0))
            ):
                reasons.append("heal_pickup")
        elif action == 14:
            if (
                isinstance(action14_audit, dict)
                and action14_audit.get("accepted") is True
                and int(action14_audit.get("utility_delta", 0)) > 0
            ):
                reasons.append("gear_commit")
        return tuple(dict.fromkeys(reasons))

    def _reward(
        self,
        prev,
        cur,
        requested_action: int | None = None,
        *,
        engage_target_generation_key: tuple[int, int, int] | None = None,
        action_executed: bool | None = None,
        action14_utility_delta: int | None = None,
    ) -> float:
        cls = type(self)
        econ = self.reward_economy
        xp_term = 0.01 * (cur["xp"] - prev["xp"])
        r = xp_term
        dl = cur["dungeon_level"] - prev["dungeon_level"]
        actual_prev, actual_cur = prev, cur
        if getattr(self, "resource_protocol", "off") != "off":
            before = self._resource_reward_depth_before
            after = self._resource_max_main_depth
            dl = max(0, after - before)
            paid = econ.descend_unit * (sum(range(before, after))
                    if self.descend_ladder else dl)
            self._resource_step_bonus = float(paid)
            # Preserve the remainder of the old accounting using equivalent
            # synthetic monotonic endpoints; raw game state is never changed.
            prev = dict(prev, dungeon_level=before)
            cur = dict(cur, dungeon_level=after)
        if self.descend_ladder and dl > 0:
            # v17: depth progression; each N->N+1 pays 8×N (L1->2 is still 8, anchored to the old chapter;
            # L2->3 pays 16, L3->4 pays 24 ... the deeper the more valuable, giving "going down alive" a future)
            r += DESCEND_UNIT * sum(range(prev["dungeon_level"], cur["dungeon_level"]))
        else:
            r += DESCEND_UNIT * dl
        if dl > 0 and econ.descend_unit != DESCEND_UNIT:
            # R10 v2 clause A: the descend unit price is raised, appended as an exact increment over v1
            # (level counts are integers, so the price-difference product is lossless in float64; the v1 path is untouched).
            if self.descend_ladder:
                r += (econ.descend_unit - DESCEND_UNIT) * sum(
                    range(prev["dungeon_level"], cur["dungeon_level"]))
            else:
                r += (econ.descend_unit - DESCEND_UNIT) * dl
        if econ.descend_vest_kills > 0:
            # R11 kill lock (2026-08-28): the descend bonus is first booked into an "unvested" escrow;
            # vest_kills kills on that level only vest it (escrow cleared, money stays booked); dying before vesting
            # forfeits all of it. Mathematically equal to real escrow payout at gamma=1.0, and it leaves the wage-stripping
            # identity intact. Timed-out episodes are not penalized (the rule only penalizes death).
            if dl > 0:
                paid = econ.descend_unit * (
                    sum(range(prev["dungeon_level"], cur["dungeon_level"]))
                    if self.descend_ladder else dl)
                self._econ_unvested += float(paid)
                self._econ_kills_on_floor = 0
            kills_now = int(getattr(self, "_ep_kills", 0))
            kill_delta = kills_now - self._econ_prev_epkills
            self._econ_prev_epkills = kills_now
            if kill_delta > 0 and dl == 0 and self._econ_unvested > 0.0:
                self._econ_kills_on_floor += kill_delta
                if self._econ_kills_on_floor >= econ.descend_vest_kills:
                    self._econ_unvested = 0.0
            if bool(cur["dead"]) and self._econ_unvested > 0.0:
                r -= self._econ_unvested
                self._econ_unvested = 0.0
        prev, cur = actual_prev, actual_cur
        # The native replacement comparator, observation and this shaping
        # consume one shared uint32 ledger.  Unlike the old ΔAC-only reward,
        # weapon damage/to-hit, shields/block, resistances, affixes and usable
        # durability all receive credit.  Strict replacement makes growth
        # monotonic at pickup time; the cap bounds any rare large unique.
        if action14_utility_delta is None:
            gear_term = gear_upgrade_reward_component(prev, cur)
            r += gear_term
        else:
            # Action 14 publishes the comparator result synchronously at the
            # native commit.  Its later settle endpoint may already include
            # durability loss or other combat changes, so endpoint Δutility
            # is not a causal receipt and must neither erase nor double-pay
            # the accepted upgrade.
            gear_term = gear_upgrade_reward_delta_component(
                action14_utility_delta)
            r += gear_term
        if econ.name != "v1" and gear_term > 0.0:
            # R10 v2 clause E1: gear repricing (scale down/cap up). The a14 causal branch
            # can recompute the exact Δutility; the (prev,cur) branch converts the v1 component proportionally,
            # underestimating conservatively only when v1 saturates (=cap); the pre-registration records this approximation truthfully.
            if action14_utility_delta is not None:
                repriced = min(
                    econ.gear_cap,
                    float(int(action14_utility_delta)) / econ.gear_scale)
            else:
                repriced = min(
                    econ.gear_cap,
                    gear_term * (GEAR_COMBAT_UTILITY_REWARD_SCALE
                                 / econ.gear_scale))
            r += repriced - gear_term
        same_scene = _scene_identity(cur) == _scene_identity(prev)
        combat_term = 0.0
        if same_scene:
            combat_term = self._combat_reward(prev, cur)
            r += combat_term
        else:
            # Damage ledgers are scene-local, but a monster killed earlier in
            # the same macro must not lose its terminal unit merely because
            # the final micro-beat also crossed a level/quest boundary.
            native_kills = self._native_monster_kill_delta(prev, cur)
            if native_kills is not None:
                combat_term = float(native_kills)
                r += native_kills
        # Approach shaping: rewarded only when this tick requested a non-wait action and "walked closer by itself".
        # To avoid breaking engine occupancy, ActWait lets the single-tile animation committed in the previous tick finish naturally;
        # if only the before/after positions were compared, that displacement of the old action would be credited to the current action0, and a real
        # a0-only probe occasionally got +0.005 to +0.02. requested_action pins the credit
        # back to the requested action: waiting never earns approach shaping, and pays -0.002 even while an old step finishes.
        # Environment events such as XP/kills/real damage are still booked as facts, not erased by the action number.
        if same_scene:
            moved = (cur["player_x"], cur["player_y"]) != (prev["player_x"], prev["player_y"])
            if requested_action == 0 or action_executed is False:
                # A rejected/stalled non-zero request is behaviorally the same
                # as an explicit wait.  Charging only action0 let action10 (and
                # stale masked requests) consume the TimeLimit at zero cost.
                r += STALL_ACTION_REWARD
            elif requested_action is not None:
                if not moved:
                    approach_delta = None
                elif requested_action == 9:
                    # action 9's path/attack is already bound to the controller snapshot's
                    # canonical generation. Re-picking the nearest monster by tile here would let
                    # a moving monster or a locally unengageable decoy book the walk toward the real target
                    # in reverse as negative reward. With no candidate it must also fail closed and must not
                    # fall back to an arbitrary raw monster to fake progress for an empty action.
                    approach_delta = (
                        cls._player_approach_delta(
                            prev,
                            cur,
                            target_generation_key=(
                                engage_target_generation_key),
                        )
                        if engage_target_generation_key is not None
                        else None
                    )
                else:
                    approach_delta = cls._player_approach_delta(prev, cur)
                if approach_delta is not None:
                    r += 0.005 * approach_delta
        r += terminal_death_reward_component(
            dead=bool(cur["dead"]),
            dungeon_level=cur["dungeon_level"],
            death_ladder=bool(self.death_ladder),
            economy=econ,
        )
        if econ.name != "v1":
            # R10 v2 clauses B×D: depth multiplier and anti-idling, applied only to positive farm income
            # (xp + kills), never touching descend/gear/shaping/death/victory terms; booked to the manager.
            d_now = max(int(cur["dungeon_level"]), 1)
            if econ.idle_counts_micro_beats:
                # R16 v4: the idle clock counts engine micro beats. step() has already added all of this decision's
                # ticks (including settle) to self._steps before calling _reward; take the increment.
                steps_now = int(getattr(self, "_steps", 0))
                idle_tick = max(
                    0, steps_now - int(getattr(
                        self, "_econ_idle_prev_steps", 0)))
                self._econ_idle_prev_steps = steps_now
            else:
                idle_tick = 1
            if int(cur["dungeon_level"]) > self._econ_episode_max_depth:
                self._econ_episode_max_depth = int(cur["dungeon_level"])
                self._econ_steps_on_level = 0
            elif dl == 0:
                self._econ_steps_on_level += idle_tick
            else:
                self._econ_steps_on_level = 0
            mult = 1.0 + econ.kill_depth_beta * (d_now - 1)
            if (econ.idle_threshold_steps > 0
                    and self._econ_steps_on_level
                    > econ.idle_threshold_steps):
                mult *= econ.idle_kill_factor
            farm = xp_term + combat_term
            if farm > 0.0:
                r += farm * (mult - 1.0)
            if econ.idle_reset_on_kill:
                # R16 v4: a kill resets the clock (settled first: the kill that ends a long idle spell
                # is still priced with the clock before the reset; the clock then starts again).
                idle_kills = self._native_monster_kill_delta(prev, cur)
                if idle_kills is not None and idle_kills > 0:
                    self._econ_steps_on_level = 0
        if cur["victory"]:
            r += 10.0
        return float(r)

    @classmethod
    def _legacy_policy_vectorize(cls, obs) -> np.ndarray:
        """Reconstruct the exact protocol-v3 295-wide policy observation.

        Protocol-v4 deliberately hid invisible/unreachable entities from the
        live observation.  Frozen V28/KING/M29 networks were trained before
        that semantic break, so their compatibility view cannot be recovered
        from the already-filtered 295-vector: monster count/nearest/slots, the
        121-cell monster channel, and floor-item slots may all have lost
        information.  Rebuild those fields from the still-lossless native
        ``raw`` record and local map instead of pretending that decoding the
        packed belt scalar alone restores v3.

        Keep this implementation literal and independent of v4 policy helpers.
        Calling ``_policy_monsters`` or ``_policy_floor_items`` here would
        silently reintroduce the distribution shift this boundary exists to
        prevent.
        """
        if "monsters" not in obs or "floor_items" not in obs:
            raise RuntimeError(
                "protocol-v3 compatibility view requires lossless native "
                "monsters and floor_items records")
        if "legacy_belt_heals" not in obs:
            raise RuntimeError(
                "protocol-v3 compatibility view requires native "
                "legacy_belt_heals; refusing a v4 belt fallback")
        missing_legacy_item = [
            index for index, item in enumerate(obs["floor_items"])
            if "legacy_heal" not in item
        ]
        if missing_legacy_item:
            raise RuntimeError(
                "protocol-v3 compatibility view requires native legacy_heal "
                "on every floor item; missing indices "
                f"{missing_legacy_item[:8]}")
        px, py = obs["player_x"], obs["player_y"]
        all_monsters = list(obs["monsters"])
        nearest = None
        if all_monsters:
            nearest = min(
                max(abs(m["x"] - px), abs(m["y"] - py))
                for m in all_monsters
            )
        advance = list(obs.get("progression_targets", ()))
        if advance:
            st = min(advance, key=lambda t: max(
                abs(t["goal_x"] - px), abs(t["goal_y"] - py)))
            sx, sy = st["goal_x"], st["goal_y"]
        else:
            transition = (bridge.WM_DIABRTNLVL if obs.get("is_set_level")
                          else bridge.WM_DIABNEXTLVL)
            stairs = [t for t in obs.get("triggers", ())
                      if t["msg"] == transition]
            st = min(stairs, key=lambda t: max(
                abs(t["x"] - px), abs(t["y"] - py))) if stairs else None
            sx, sy = (st["x"], st["y"]) if st is not None else (px, py)
        if advance or st is not None:
            stair_dx, stair_dy = (sx - px) / 56.0, (sy - py) / 56.0
        else:
            stair_dx = stair_dy = 0.0
        vec = [
            obs["hp"] / max(1, obs["max_hp"]),
            obs["mana"] / max(1, obs["max_mana"]),
            math.log1p(obs["xp"]) / 10.0,
            obs["gold"] / 1000.0,
            obs["char_level"] / 50.0,
            obs["dungeon_level"] / 16.0,
            px / 112.0,
            py / 112.0,
            min(1.0, len(all_monsters) / 50.0),
            min(1.0, nearest / 30.0) if nearest is not None else 1.0,
            stair_dx,
            stair_dy,
        ]
        monsters = sorted(
            all_monsters,
            key=lambda m: abs(m["x"] - px) + abs(m["y"] - py),
        )[:_K_MONSTERS]
        for monster in monsters:
            vec += [
                (monster["x"] - px) / 20.0,
                (monster["y"] - py) / 20.0,
                monster["hp"] / max(1, monster["max_hp"]),
                1.0,
            ]
        vec += [0.0, 0.0, 0.0, 0.0] * (
            _K_MONSTERS - len(monsters))
        local_map = bridge.local_map(radius=_MAP_RADIUS)
        vec += [float(value) for value in local_map["walkable"]]
        vec += [float(value) for value in local_map["monster"]]
        heals = [
            item for item in obs["floor_items"]
            if bool(item["legacy_heal"])
        ]
        legacy_belt_heals = obs["legacy_belt_heals"]
        if heals:
            heal = min(heals, key=lambda item: max(
                abs(item["x"] - px), abs(item["y"] - py)))
            vec += [
                legacy_belt_heals / 8.0,
                max(-1.0, min(1.0, (heal["x"] - px) / 20.0)),
                max(-1.0, min(1.0, (heal["y"] - py) / 20.0)),
                1.0,
            ]
        else:
            vec += [legacy_belt_heals / 8.0, 0.0, 0.0, 0.0]
        gears = [
            item for item in obs["floor_items"]
            if item.get("gear")
        ]
        armor_class = max(
            0.0, min(1.0, obs.get("armor_class", 0) / 50.0))
        if gears:
            gear = min(gears, key=lambda item: max(
                abs(item["x"] - px), abs(item["y"] - py)))
            vec += [
                armor_class,
                max(-1.0, min(1.0, (gear["x"] - px) / 20.0)),
                max(-1.0, min(1.0, (gear["y"] - py) / 20.0)),
                1.0,
            ]
        else:
            vec += [armor_class, 0.0, 0.0, 0.0]
        vec += [
            min(
                2.0,
                obs["char_level"] / max(1, obs["dungeon_level"]),
            ) / 2.0
        ]
        result = np.asarray(vec, dtype=np.float32)
        if result.shape != (295,):
            raise RuntimeError(
                "protocol-v3 policy observation shape drift: "
                f"{result.shape} != (295,)")
        return result

    @classmethod
    def _vectorize(cls, obs) -> np.ndarray:
        px, py = obs["player_x"], obs["player_y"]
        policy_monsters = cls._policy_monsters(obs)
        nearest = cls._nearest_dist(obs)
        advance = list(obs.get("progression_targets", ()))
        if advance:
            st = min(advance, key=lambda t: max(
                abs(t["goal_x"] - px), abs(t["goal_y"] - py)))
            sx, sy = st["goal_x"], st["goal_y"]
        else:
            transition = (bridge.WM_DIABRTNLVL if obs.get("is_set_level")
                          else bridge.WM_DIABNEXTLVL)
            stairs = [t for t in obs.get("triggers", [])
                      if t["msg"] == transition]
            st = min(stairs, key=lambda t: max(
                abs(t["x"] - px), abs(t["y"] - py))) if stairs else None
            sx, sy = (st["x"], st["y"]) if st is not None else (px, py)
        if advance or st is not None:
            stair_dx, stair_dy = (sx - px) / 56.0, (sy - py) / 56.0
        else:
            stair_dx = stair_dy = 0.0
        vec = [
            obs["hp"] / max(1, obs["max_hp"]),
            obs["mana"] / max(1, obs["max_mana"]),
            math.log1p(obs["xp"]) / 10.0,
            obs["gold"] / 1000.0,
            obs["char_level"] / 50.0,
            obs["dungeon_level"] / 16.0,
            px / 112.0,
            py / 112.0,
            min(1.0, len(policy_monsters) / 50.0),
            min(1.0, nearest / 30.0) if nearest is not None else 1.0,
            stair_dx,
            stair_dy,
        ]
        monsters = sorted(
            policy_monsters, key=lambda m: abs(m["x"] - px) + abs(m["y"] - py)
        )[:_K_MONSTERS]
        for m in monsters:
            vec += [(m["x"] - px) / 20.0, (m["y"] - py) / 20.0, m["hp"] / max(1, m["max_hp"]), 1.0]
        vec += [0.0, 0.0, 0.0, 0.0] * (_K_MONSTERS - len(monsters))
        lm = bridge.local_map(radius=_MAP_RADIUS)
        vec += [float(v) for v in lm["walkable"]]
        # The local_map monster channel is the physical collision source of truth and includes monsters behind walls/unlit;
        # the policy observation must share the visible-reachable source of truth with token/a9, and cannot peek at
        # omniscient occupancy through this 121-bit side channel. Planning macros may still use the raw channel to avoid walking through bodies.
        policy_tiles = {(int(m["x"]), int(m["y"])) for m in policy_monsters}
        vec += [
            1.0 if (px + dx, py + dy) in policy_tiles else 0.0
            for dy in range(-_MAP_RADIUS, _MAP_RADIUS + 1)
            for dx in range(-_MAP_RADIUS, _MAP_RADIUS + 1)
        ]
        heals = cls._policy_floor_items(obs, "heal")
        belt_scalar = cls._belt_observation_scalar(obs)
        if heals:  # v13: bottle-blindness fix; the preconditions of the drink/potion-pickup keys enter the observation
            h = min(heals, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
            vec += [belt_scalar,
                    max(-1.0, min(1.0, (h["x"] - px) / 20.0)),
                    max(-1.0, min(1.0, (h["y"] - py) / 20.0)), 1.0]
        else:
            vec += [belt_scalar, 0.0, 0.0, 0.0]
        gears = cls._policy_floor_items(obs, "gear")
        ac = max(0.0, min(1.0, obs.get("armor_class", 0) / 50.0))
        if gears:  # v14: gear chapter; the precondition of the gear-pickup key enters the observation (lesson 11 acceptance list)
            g = min(gears, key=lambda it: max(abs(it["x"] - px), abs(it["y"] - py)))
            vec += [ac,
                    max(-1.0, min(1.0, (g["x"] - px) / 20.0)),
                    max(-1.0, min(1.0, (g["y"] - py) / 20.0)), 1.0]
        else:
            vec += [ac, 0.0, 0.0, 0.0]
        # v19: strength gauge (lesson-5 family). The decision variable for "strong enough, time to go down?" is
        # the level/depth ratio, but dim5's char_level/50 makes level 1 and level 3 differ by only 0.04,
        # nearly invisible to the policy; to decide how strong to farm before descending, "how strong" must first be visible.
        # Ratio 1.0 = level equals depth, >1 over-levelled and crushing, <1 under-levelled and dying; capped at 2 and normalized.
        vec += [min(2.0, obs["char_level"] / max(1, obs["dungeon_level"])) / 2.0]
        return np.asarray(vec, dtype=np.float32)
