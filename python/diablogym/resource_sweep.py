"""R18-H (2026-09-07) sweep-v1: the scripted chest/barrel sweep on main L1.

Chairman ruling 2026-09-07 14:00: the pair must play a complete Diablo, and
today it cannot see a chest and never smashes a barrel for loot.  The eyes are
the new native ``raw["objects"]`` channel (``src/resource_sweep.hpp``); the
hands already existed (``("open", x, y)`` -> ``act_controller_operate`` reaches
``OperateChest`` for every chest type and ``StartAttack`` for every barrel via
``_oBreak == 1``).  This module is the missing *decision*.

Law (v1):

* main L1 only, never in town, never on L2+, never during a town trip and never
  on a set level;
* the sweep opens on the loot economy's own FIRST-TRIP facts (floor cleared, or
  the FARM cap) **and only while a town-trip slot is left**, and it is consulted
  BEFORE ``SustainLootService.maybe_start`` in ``OptionsEnv.action_masks``, so
  the drops it creates are on the floor before the collect stage runs.  It is
  NOT the whole of ``maybe_start``'s admission law: the native-deficit gate is
  deliberately NOT copied (review round 2026-09-07 — the ruling asks the pair to
  open chests whenever it plays, a deficit appears later in every episode, and
  the sweep's own drops are themselves a legal second-trip trigger).  The window
  ledger records ``trip_slots0`` / ``readiness_deficit0`` so the orphaned-drop
  case is measurable rather than assumed;
* only ``"chest"`` and ``"barrel"`` are opened.  ``"chest_trapped"`` fires a
  trap missile at the player, ``"sarcophagus"`` can spawn a skeleton and
  ``"barrel_explosive"`` damages the player: v1 records them as skipped and
  never touches them;
* only objects this episode has actually SEEN LIT are candidates.  The native
  channel carries ``visible`` (``IsTileLit``) exactly like ``floor_items``, and
  the service keeps a per-episode memory of the objects it has seen lit — the
  same partial-observability 口径 as the loot memory, so the sweep never walks
  to a chest on an unexplored side of the floor;
* bounded — ``SWEEP_MICROSTEP_BUDGET`` microsteps and ``SWEEP_MAX_TARGETS``
  targets per EPISODE — and abandoned the beat an alive monster comes within
  ``SWEEP_MONSTER_ABORT_RADIUS`` tiles or hp falls to
  ``SWEEP_HP_ABORT_FRACTION`` of max, so the ordinary FARM fight (or the
  retreat law) resumes.  The remaining budget is carried to the next trigger;
* the window LENGTH is frozen when the script issues its first command
  (``service_microstep_cap`` is a constant for the life of a window, the same
  contract ``RetreatService`` / ``PortalService`` / ``ResourceService`` expose);
  ``budget_remaining`` stays the separate decaying per-episode ledger;
* loot is booked only from ACCEPTED operate receipts;
* the window collapses the manager mask to RESUPPLY exactly like retreat and
  portal, and is NOT settle-exempt (only the portal's spell needed that).

``resource_sweep="off"`` (the default) constructs no service, passes no new
env kwarg, makes no native call and leaves the observation dict byte-identical.
"""
from dataclasses import dataclass, field

from .resource_protocol import (
    RESOURCE_SWEEP_PROTOCOLS, validate_sweep_protocol, validate_native_sweep)

SWEEP_MICROSTEP_BUDGET = 900        # microsteps per EPISODE (all windows together)
SWEEP_MAX_TARGETS = 16              # objects attempted per episode
# 16, not 8: on the R18 smoke seeds the nearest-first order is dominated by
# chests, and a cap of 8 was spent before a single barrel became the target
# (see H1-REPORT.md). The 900-microstep episode budget stays the real bound.
SWEEP_MONSTER_ABORT_RADIUS = 3      # an alive monster this close hands control back
SWEEP_WALK_REJECTIONS = 8           # consecutive rejected walks before abandoning a target
SWEEP_OPEN_ATTEMPTS = 6             # accepted operate commands per target before giving up
SWEEP_WINDOW_COOLDOWN = 60          # microsteps before a new sweep window may open
SWEEP_EMPTY_WINDOWS = 3             # consecutive windows that reached no target before the
                                    # episode stops re-opening (review round 2026-09-07:
                                    # replaces the old blanket retirement of every candidate)
SWEEP_HP_ABORT_FRACTION = 0.5       # hand back below this hp share; the same number as
                                    # RetreatPolicy.hp_fraction, so the sweep never holds the
                                    # body through the beat the retreat law wants it
SWEEP_TAKEN_KINDS = ("chest", "barrel")
SWEEP_SKIPPED_KINDS = ("chest_trapped", "sarcophagus", "barrel_explosive")

__all__ = ["RESOURCE_SWEEP_PROTOCOLS", "SweepService", "validate_sweep_protocol",
           "validate_native_sweep", "SWEEP_MICROSTEP_BUDGET", "SWEEP_MAX_TARGETS",
           "SWEEP_MONSTER_ABORT_RADIUS", "SWEEP_TAKEN_KINDS", "SWEEP_SKIPPED_KINDS",
           "SWEEP_EMPTY_WINDOWS", "SWEEP_HP_ABORT_FRACTION"]


def sweep_objects(raw):
    """The new native channel, or () when the bridge flag is off.

    Never invents an object: an absent key is an absent channel, not an empty
    floor (``validate_native_sweep`` is the fail-closed identity check)."""
    objects = raw.get("objects") if isinstance(raw, dict) else None
    return tuple(objects) if isinstance(objects, (list, tuple)) else ()


def alive_monsters_within(raw, radius):
    """Same alive-monster口径 as RetreatService (type 109 = the invalid marker)."""
    px, py = int(raw["player_x"]), int(raw["player_y"])
    count = 0
    for monster in raw.get("monsters", ()):
        if int(monster.get("hp", 0)) <= 0 or int(monster.get("type", -1)) == 109:
            continue
        if bool(monster.get("is_invalid", False)):
            continue
        if "x" not in monster or "y" not in monster:
            continue
        if max(abs(int(monster["x"]) - px), abs(int(monster["y"]) - py)) <= radius:
            count += 1
    return count


def _item_identity(item):
    return (int(item.get("seed_hi", 0)), int(item.get("seed_lo", 0)),
            int(item.get("create_info", 0)), int(item.get("base_id", 0)))


@dataclass
class SweepService:
    """One instance per episode; the caller builds a fresh one on every reset."""
    attempted: bool = False
    active: bool = False
    phase: str = "idle"
    reason: str | None = None
    trigger: str | None = None
    start_steps: int = 0
    steps: int = 0
    microsteps_used: int = 0
    chests_opened: int = 0
    barrels_smashed: int = 0
    targets_attempted: int = 0
    open_commands: int = 0
    open_accepted: int = 0
    open_rejected: int = 0
    path_open_commands: int = 0
    walk_commands: int = 0
    empty_windows: int = 0
    gold_peak_after_first_sweep: int | None = None
    targets_skipped_by_kind: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    phase_steps: dict = field(default_factory=dict)
    gold_before: int | None = None
    gold_after: int | None = None
    items_dropped_seen: int = 0
    _seen_drop_identities: set = field(default_factory=set, repr=False)
    _snapshot_tiles: set = field(default_factory=set, repr=False)
    _operated_tiles: set = field(default_factory=set, repr=False)
    _seen_objects: set = field(default_factory=set, repr=False)
    _booked: set = field(default_factory=set, repr=False)
    _target: dict | None = field(default=None, repr=False)
    _target_opens: int = field(default=0, repr=False)
    _done_targets: set = field(default_factory=set, repr=False)
    _skipped_recorded: set = field(default_factory=set, repr=False)
    _last_phase: str = field(default="idle", repr=False)
    _last_phase_step: int = field(default=0, repr=False)
    _walk_rejections: int = field(default=0, repr=False)
    _give_up: str | None = field(default=None, repr=False)
    _cooldown_until: int = field(default=0, repr=False)
    _window_cap: int = field(default=SWEEP_MICROSTEP_BUDGET, repr=False)
    _window_baselined: bool = field(default=False, repr=False)
    _window_targets0: int = field(default=0, repr=False)
    _hp_min: int | None = field(default=None, repr=False)

    # ---- clocks (the shape the RESUPPLY window loop expects of a service) ----
    @property
    def budget_remaining(self):
        """The decaying per-EPISODE ledger. NOT the window length."""
        return max(0, SWEEP_MICROSTEP_BUDGET - int(self.microsteps_used))

    @property
    def service_microstep_cap(self):
        """CONSTANT for the life of a window, like RetreatService.

        The RESUPPLY loop computes ``env._resource_service_deadline =
        start_steps + service_microstep_cap`` on every iteration, so a cap that
        decayed with the wall clock produced a deadline that moved BACKWARDS one
        microstep per elapsed microstep and expired at half the sanctioned
        budget — which silently cancelled env.py's 12-beat accepted-``open``
        follow-through from the middle of every window on (review round
        2026-09-07)."""
        return max(1, int(self._window_cap))

    @property
    def command_microstep_deadline(self):
        """Same shape (and same "only ResourceService reads it" status) as
        ``RetreatService.command_microstep_deadline``; kept so every service in
        the RESUPPLY loop answers the same three clock questions."""
        return self.start_steps + self.service_microstep_cap

    def record_steps(self, absolute_steps, phase=None):
        elapsed = int(absolute_steps) - self._last_phase_step
        charged = self._last_phase if phase is None else phase
        if elapsed > 0:
            self.phase_steps[charged] = self.phase_steps.get(charged, 0) + elapsed
            self.microsteps_used += elapsed
        self._last_phase_step = int(absolute_steps)
        self.steps = int(absolute_steps) - self.start_steps

    # ---- the manager-side law ----
    def trigger_reason(self, raw, steps, *, farm_scene_steps, cleared, farm_trigger,
                       town_trip_active, loot_trip_slots_left=1):
        """Return the trigger name if the sweep law fires now, else None.

        ``cleared`` / ``farm_scene_steps`` / ``farm_trigger`` are the two facts
        the loot economy's own FIRST-trip trigger consumes;
        ``loot_trip_slots_left`` is ``trip_limit(raw) - trip_count``, so the
        sweep never drops loot that no collect stage can ever pick up.  The rest
        of ``maybe_start``'s admission law is deliberately not copied — see the
        module docstring."""
        if self.active or raw is None:
            return None
        if raw.get("dead") or raw.get("game_over") or raw.get("is_set_level"):
            return None
        if int(raw.get("dungeon_level", 0)) != 1:
            return None            # v1: main L1 only, and never in town
        if town_trip_active:
            return None
        if int(loot_trip_slots_left) <= 0:
            return None            # no trip left = nobody would collect the drops
        if int(raw.get("player_mode", 0) or 0) != 0:
            return None
        if int(steps) < self._cooldown_until:
            return None            # a window that just handed back never re-opens on the same beat
        if self.budget_remaining <= 0 or self.targets_attempted >= SWEEP_MAX_TARGETS:
            return None
        if self.empty_windows >= SWEEP_EMPTY_WINDOWS:
            return None            # this floor keeps refusing; stop paying for empty windows
        if alive_monsters_within(raw, SWEEP_MONSTER_ABORT_RADIUS):
            return None
        max_hp = int(raw.get("max_hp", 0) or 0)
        if max_hp > 0 and int(raw.get("hp", 0) or 0) <= SWEEP_HP_ABORT_FRACTION * max_hp:
            return None            # the retreat law owns a hurt body, not the sweep
        self._observe_objects(raw)
        if not self._reachable_targets(raw):
            return None
        if bool(cleared):
            return "cleared"
        if int(farm_scene_steps) >= int(farm_trigger):
            return "farm_cap"
        return None

    def start(self, raw, steps, trigger, *, loot_trip_slots_left=None,
              readiness_deficit=None):
        if self.active:
            raise RuntimeError("sweep already active")
        if int(raw.get("dungeon_level", 0)) != 1 or raw.get("is_set_level"):
            raise RuntimeError("sweep-v1 runs only on main L1")
        self.attempted = self.active = True
        self.phase = self._last_phase = "sweep"
        self.reason = None
        self.trigger = trigger
        self.start_steps = self._last_phase_step = int(steps)
        self.steps = 0
        self._target = None
        self._target_opens = 0
        self._walk_rejections = 0
        self._give_up = None
        self._window_targets0 = int(self.targets_attempted)
        self._hp_min = int(raw.get("hp", 0) or 0)
        # The window length is a CONSTANT from here on. It is re-frozen on the
        # first command() of the window, because action_masks() (and therefore
        # this start) also runs inside the frozen worker's observation build,
        # several beats before the RESUPPLY loop can ask for a command.
        self._window_cap = max(1, self.budget_remaining)
        self._window_baselined = False
        gold = int(raw.get("gold", 0))
        if self.gold_before is None:
            self.gold_before = gold
        self.gold_after = gold
        self.gold_peak_after_first_sweep = (
            gold if self.gold_peak_after_first_sweep is None
            else max(int(self.gold_peak_after_first_sweep), gold))
        self._observe_objects(raw)
        self._record_skips(raw)
        self.windows.append({
            "trigger": trigger, "beat0": int(steps), "gold0": gold,
            "budget_remaining0": self.budget_remaining,
            "trip_slots0": (None if loot_trip_slots_left is None
                            else int(loot_trip_slots_left)),
            "readiness_deficit0": (None if readiness_deficit is None
                                   else bool(readiness_deficit)),
            "chests0": self.chests_opened, "barrels0": self.barrels_smashed,
            "outcome": None, "reason": None, "beat1": None, "gold1": None,
            "hp_min": self._hp_min})

    # ---- targets ----
    def _record_skips(self, raw):
        """Book every dangerous object once, so the telemetry says what v1 left."""
        for obj in sweep_objects(raw):
            kind = str(obj.get("kind"))
            if kind not in SWEEP_SKIPPED_KINDS or not bool(obj.get("interactable")):
                continue
            key = (kind, int(obj["x"]), int(obj["y"]))
            if key in self._skipped_recorded:
                continue
            self._skipped_recorded.add(key)
            self.targets_skipped_by_kind[kind] = self.targets_skipped_by_kind.get(kind, 0) + 1

    def _observe_objects(self, raw):
        """Remember every object seen LIT this episode.

        Partial observability, same 口径 as the floor-item channel's
        ``visible`` (``IsTileLit``) and the loot service's identity memory: an
        object the pair has never had in the light is not a legal target, and
        one it HAS seen stays a target after the light moves on."""
        for obj in sweep_objects(raw):
            if bool(obj.get("visible")):
                self._seen_objects.add(self._key(obj))

    def observe(self, raw):
        """Everything the manager beat must record whether or not a window is
        open: the wallet (for the gold pair) and the lit-object memory."""
        self.observe_gold(raw)
        if isinstance(raw, dict):
            self._observe_objects(raw)

    @staticmethod
    def _key(obj):
        return (str(obj.get("kind")), int(obj["x"]), int(obj["y"]))

    def _candidates(self, raw):
        for obj in sweep_objects(raw):
            kind = str(obj.get("kind"))
            if kind not in SWEEP_TAKEN_KINDS or not bool(obj.get("interactable")):
                continue
            key = self._key(obj)
            if key in self._done_targets or key not in self._seen_objects:
                continue
            yield obj

    def _reachable_targets(self, raw, env=None):
        """Reachable = a BFS lands adjacent to (or on) the object.  Without an
        env (the manager-side law) fall back to Chebyshev line of sight over the
        raw, which is a superset the command loop re-checks with the real
        planner before it walks.

        Same planner口径 as ``_walk``: the monster-avoiding BFS first, the plain
        BFS as the fallback.  Without the fallback an idle monster standing in a
        corridor made every object behind it "unreachable" (review round
        2026-09-07) even though the walker would have gone there."""
        px, py = int(raw["player_x"]), int(raw["player_y"])
        found = []
        for obj in self._candidates(raw):
            tx, ty = int(obj["x"]), int(obj["y"])
            distance = max(abs(tx - px), abs(ty - py))
            if env is None or distance <= 1:
                found.append((distance, 0, obj))
                continue
            path = env._plan_descend_path(raw, tx, ty, avoid_monsters=True)
            if not self._lands_beside(path, tx, ty):
                path = env._plan_descend_path(raw, tx, ty, avoid_monsters=False)
            if not self._lands_beside(path, tx, ty):
                continue
            found.append((distance, len(path), obj))
        found.sort(key=lambda row: (row[1], row[0]))
        return found

    @staticmethod
    def _lands_beside(path, tx, ty):
        if not path:
            return False
        ex, ey = int(path[-1][0]), int(path[-1][1])
        return max(abs(tx - ex), abs(ty - ey)) <= 1

    # ---- the worker-side hands ----
    def command(self, env, bridge):
        if not self._window_baselined:
            # The manager law fires inside action_masks(), which the frozen
            # worker's dual observation build also calls, so a window can open
            # several beats before the RESUPPLY loop asks for its first command.
            # Re-baseline both clocks here: the episode budget is charged only
            # for beats this script actually owned, and the frozen window length
            # starts at the first command rather than at the trigger.
            self._window_baselined = True
            self.start_steps = self._last_phase_step = int(env._steps)
            self._window_cap = max(1, self.budget_remaining)
            if self.windows:
                self.windows[-1]["beat0"] = int(env._steps)
                self.windows[-1]["budget_remaining0"] = self.budget_remaining
        self.record_steps(env._steps)
        command = self._command(env, bridge)
        self._last_phase = self.phase
        return command

    def _command(self, env, bridge):
        raw = env._raw
        self.gold_after = int(raw.get("gold", 0))
        hp = int(raw.get("hp", 0) or 0)
        self._hp_min = hp if self._hp_min is None else min(int(self._hp_min), hp)
        if raw.get("dead") or raw.get("game_over") or hp <= 0:
            return self._end(env, "sweep_died")
        if raw.get("is_set_level") or int(raw.get("dungeon_level", 0)) != 1:
            return self._end(env, "sweep_unexpected_scene")
        max_hp = int(raw.get("max_hp", 0) or 0)
        if max_hp > 0 and hp <= SWEEP_HP_ABORT_FRACTION * max_hp:
            # Hand the body back the beat the retreat law would want it; the
            # sweep must never hold a hurt player on a live floor for a budget.
            return self._end(env, "sweep_low_hp")
        if self._give_up is not None:
            give_up, self._give_up = self._give_up, None
            return self._end(env, give_up)
        if self.budget_remaining <= 0:
            return self._end(env, "sweep_budget")
        if self.targets_attempted >= SWEEP_MAX_TARGETS:
            return self._end(env, "sweep_target_cap")
        if alive_monsters_within(raw, SWEEP_MONSTER_ABORT_RADIUS):
            # Hand back so the ordinary FARM fight resumes; the remaining
            # budget and the untouched targets survive to the next trigger.
            return self._end(env, "sweep_monster_near")
        self._observe_objects(raw)
        self._record_skips(raw)
        self._book_operated(raw)
        self._settle_target(raw)
        if self._target is None:
            candidates = self._reachable_targets(raw, env)
            if not candidates:
                return self._end(env, "sweep_no_reachable_target")
            obj = candidates[0][2]
            self._target = {"kind": str(obj["kind"]), "x": int(obj["x"]), "y": int(obj["y"])}
            self._target_opens = 0
            self._walk_rejections = 0
            self.targets_attempted += 1
        tx, ty = self._target["x"], self._target["y"]
        px, py = int(raw["player_x"]), int(raw["player_y"])
        if max(abs(tx - px), abs(ty - py)) <= 1:
            if self._target_opens >= SWEEP_OPEN_ATTEMPTS:
                return self._abandon_target("sweep_open_exhausted")
            self._target_opens += 1
            self.open_commands += 1
            self._snapshot_drops(raw, tx, ty)
            return ("open", tx, ty)
        command = self._walk(env, tx, ty)
        if command is None:
            return self._abandon_target("sweep_target_unreachable")
        self.walk_commands += 1
        if command[0] == "open":
            # A blocking barrel (or closed door) ON THE WAY is operated by the
            # navigation channel. A barrel broken there is still a barrel this
            # sweep smashed, and its drop is still loot the collect stage will
            # find, so it is booked exactly like a chosen target.
            self.path_open_commands += 1
            self._snapshot_drops(raw, int(command[1]), int(command[2]))
        return command

    def _book_operated(self, raw):
        """Book every object this script operated the beat the engine says it is
        no longer interactable — that flag is exactly OperateChest / BreakBarrel's
        own "already opened / already broken" predicate. Covers both the chosen
        targets and the blocking barrels smashed by the navigation channel.

        A tile only enters ``_operated_tiles`` from ``receipt()``, on an
        ACCEPTED operate, so an object the worker (or anything else) broke while
        our own command was refused is never claimed."""
        for obj in sweep_objects(raw):
            key = self._key(obj)
            if key in self._booked or (int(obj["x"]), int(obj["y"])) not in self._operated_tiles:
                continue
            if bool(obj.get("interactable")):
                continue
            self._booked.add(key)
            self._done_targets.add(key)
            if key[0] == "chest":
                self.chests_opened += 1
            elif key[0] == "barrel":
                self.barrels_smashed += 1
            else:                       # a door: navigation, not loot
                continue
            self._count_drops(raw, key[1], key[2])

    def _settle_target(self, raw):
        """Close the current target once the engine (or _book_operated) says it
        is finished, so the next beat picks the next object."""
        if self._target is None:
            return
        key = (self._target["kind"], self._target["x"], self._target["y"])
        for obj in sweep_objects(raw):
            if self._key(obj) == key and bool(obj.get("interactable")):
                return
        self._done_targets.add(key)
        self._target = None
        self._target_opens = 0

    def _snapshot_drops(self, raw, tx, ty):
        """Everything already lying beside the object BEFORE we open it.

        Monster drops from the preceding FARM and the collect stage's leftovers
        are not this sweep's loot; without this snapshot ``_count_drops`` booked
        them into ``items_dropped_seen`` (review round 2026-09-07)."""
        if (int(tx), int(ty)) in self._snapshot_tiles:
            return
        self._snapshot_tiles.add((int(tx), int(ty)))
        for item in raw.get("floor_items", ()):
            try:
                ix, iy = int(item["x"]), int(item["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if max(abs(ix - tx), abs(iy - ty)) > 1:
                continue
            self._seen_drop_identities.add(_item_identity(item))

    def _count_drops(self, raw, tx, ty):
        """New floor identities on or beside the object we just opened."""
        for item in raw.get("floor_items", ()):
            try:
                ix, iy = int(item["x"]), int(item["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if max(abs(ix - tx), abs(iy - ty)) > 1:
                continue
            identity = _item_identity(item)
            if identity in self._seen_drop_identities:
                continue
            self._seen_drop_identities.add(identity)
            self.items_dropped_seen += 1

    def _abandon_target(self, reason):
        if self._target is not None:
            self.failures.append({"reason": reason, **dict(self._target)})
            self._done_targets.add(
                (self._target["kind"], self._target["x"], self._target["y"]))
        self._target = None
        self._target_opens = 0
        self._walk_rejections = 0
        # Abandoning one object is not abandoning the sweep: keep the window
        # open and let the next beat pick the next reachable target.
        return ("wait",)

    @staticmethod
    def _walk(env, x, y):
        """Identical planner口径 to RetreatService._walk: the monster-avoiding
        BFS first, the plain BFS as the fallback, and a closed door / blocking
        barrel on the way is opened rather than walked into."""
        raw = env._raw
        path = env._plan_descend_path(raw, x, y, avoid_monsters=True)
        if not path:
            path = env._plan_descend_path(raw, x, y, avoid_monsters=False)
        if not path:
            return None
        nx, ny, door = path[0]
        if door:
            from . import bridge
            if not bool(bridge.probe_tile(int(nx), int(ny))["walkable"]):
                return ("open", int(nx), int(ny))
        return ("walk", int(nx), int(ny))

    def receipt(self, command, receipt):
        kind = command[0]
        if kind in ("walk", "open"):
            if receipt.get("accepted"):
                self._walk_rejections = 0
            else:
                self._walk_rejections += 1
                if self._walk_rejections >= SWEEP_WALK_REJECTIONS:
                    self._give_up = "sweep_walk_rejected"
        if kind != "open":
            return
        if receipt.get("accepted"):
            # ONLY an accepted operate registers the tile. act_controller_operate
            # fails closed on MyPlayer->position.future, so a mid-tile player is
            # refused and nothing reached the engine: booking such a tile would
            # let the sweep claim a barrel the worker later smashed itself
            # (review round 2026-09-07).
            self.open_accepted += 1
            self._operated_tiles.add((int(command[1]), int(command[2])))
        else:
            self.open_rejected += 1

    def _end(self, env, reason):
        self.active = False
        self.phase = "done"
        self.reason = reason
        self._target = None
        self._target_opens = 0
        self._window_baselined = False
        self._cooldown_until = int(env._steps) + SWEEP_WINDOW_COOLDOWN
        if self.targets_attempted > self._window_targets0:
            self.empty_windows = 0
        elif reason in ("sweep_no_reachable_target", "sweep_walk_rejected"):
            # No blanket retirement: an object unreachable from HERE is often
            # reachable from the next trigger's standing spot, and the walker
            # has a fallback this test did not. Bound the cost with a counter
            # instead (review round 2026-09-07).
            self.empty_windows += 1
        if self.windows:
            record = self.windows[-1]
            record["outcome"] = "swept" if reason in (
                "sweep_no_reachable_target", "sweep_budget", "sweep_target_cap") else "handed_back"
            record["reason"] = reason
            record["beat1"] = int(env._steps)
            record["gold1"] = int(env._raw.get("gold", 0))
            record["chests1"] = self.chests_opened
            record["barrels1"] = self.barrels_smashed
            record["hp_min"] = self._hp_min
        self.gold_after = int(env._raw.get("gold", 0))
        return ("complete",)

    def observe_gold(self, raw):
        """Keep the gold pair honest after the sweep hands back: the loot service
        collects the drops in the trip that follows, and the ruling asks for the
        gold across sweep + collect.

        ``gold_after`` is the LATEST wallet, which by episode end has also been
        SPENT at the smith/healer, so it is not by itself an income measure;
        ``gold_peak_after_first_sweep`` is the high-water mark since the sweep
        opened, which is the collected total before the trip spends it."""
        if not isinstance(raw, dict) or "gold" not in raw:
            return
        gold = int(raw["gold"])
        self.gold_after = gold
        if self.gold_before is not None:
            self.gold_peak_after_first_sweep = (
                gold if self.gold_peak_after_first_sweep is None
                else max(int(self.gold_peak_after_first_sweep), gold))

    def telemetry(self):
        return {
            "protocol": "sweep-v1",
            "attempted": self.attempted, "active": self.active, "phase": self.phase,
            "reason": self.reason, "trigger": self.trigger,
            "chests_opened": self.chests_opened,
            "barrels_smashed": self.barrels_smashed,
            "targets_skipped_by_kind": dict(self.targets_skipped_by_kind),
            "sweep_microsteps": int(self.microsteps_used),
            "gold_before": self.gold_before, "gold_after": self.gold_after,
            "gold_delta": (None if self.gold_before is None or self.gold_after is None
                           else int(self.gold_after) - int(self.gold_before)),
            "gold_peak_after_first_sweep": self.gold_peak_after_first_sweep,
            "gold_peak_delta": (
                None if self.gold_before is None or self.gold_peak_after_first_sweep is None
                else int(self.gold_peak_after_first_sweep) - int(self.gold_before)),
            "items_dropped_seen": int(self.items_dropped_seen),
            "targets_attempted": int(self.targets_attempted),
            "open_commands": int(self.open_commands),
            "open_accepted": int(self.open_accepted),
            "open_rejected": int(self.open_rejected),
            "path_open_commands": int(self.path_open_commands),
            "walk_commands": int(self.walk_commands),
            "empty_windows": int(self.empty_windows),
            "objects_seen_lit": len(self._seen_objects),
            "budget_remaining": self.budget_remaining,
            "window_microstep_cap": int(self._window_cap),
            "microstep_budget": SWEEP_MICROSTEP_BUDGET,
            "max_targets": SWEEP_MAX_TARGETS,
            "monster_abort_radius": SWEEP_MONSTER_ABORT_RADIUS,
            "hp_abort_fraction": SWEEP_HP_ABORT_FRACTION,
            "windows": [dict(w) for w in self.windows],
            "failures": [dict(f) for f in self.failures],
            "phase_steps": dict(self.phase_steps),
        }
