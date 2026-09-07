"""R18-F portal-v1: the Scroll of Town Portal interface for main L2+.

R18-A gave the pair a *walking* retreat (one floor per trip, up the stairs).
It costs depth: three retreats from L4 end on L1.  The Scroll of Town Portal is
the same interface without that cost -- the pair leaves L2+ for town, restocks
at the ordinary shops, and comes back to *the same floor*.

The three legs this script owns (each a separate RESUPPLY window; the manager
keeps its three words):

* ``restock`` -- in town, after the ordinary town service has finished its
  healer/smith/potions itinerary, walk to Adria (80, 20), open her store and buy
  the pinned Scroll of Town Portal (``WitchItems[2]``, 200 gold at any dlvl).
  Deliberately last: ``StoreAutoPlace`` prefers the belt, so a scroll bought
  early would eat a slot the four-heal readiness law needs.  The engine refuses
  the sale outright unless those four heals are already in the belt
  (``belt_reserved_for_heals``).
* ``outbound`` -- on main L2+, read the scroll (``act_cast_town_portal``, never
  ``UseInvItem``), walk onto the portal that appears within five tiles and cross.
  The engine receipt is ``portal_to_town`` and the arrival sets the town-service
  trip flag, so the ordinary shops open exactly as after a stairs departure.
* ``return`` -- in town, walk onto the town-side portal missile at (57, 40).
  The engine receipt is ``portal_return`` and the pair lands back on the floor
  it left, on the tile it left from.

The town SHOPPING itself is not duplicated here: the outbound leg hands control
back with ``("complete",)`` the moment the pair is in town, and the ordinary
``ResourceService`` runs its own phases.  See ``docs`` in
``r17_work/r18/patch_portal_env.py`` for the one place that needs a change: the
town service's ``return`` phase walks to the cathedral stairs, which would throw
away the depth the portal just preserved, so the coach has to close it there and
hand the return leg to this script.

Failure never ends the episode: every leg hands back with ``("complete",)`` and
a cooldown (``("finish", ...)`` stays reserved for the town service's
fail-closed itinerary).  ``resource_portal="off"`` (the default) constructs no
service, buys nothing from the witch and leaves every frozen path byte-identical.
"""
from dataclasses import dataclass, field

from .resource_protocol import (
    RESOURCE_PORTAL_PROTOCOLS, validate_portal_protocol, validate_native_portal)

PORTAL_SERVICE_CAP = 900          # microsteps per portal leg
PORTAL_MAX_PER_EPISODE = 2        # mirrors MaxResourcePortals in the engine
PORTAL_FAILURE_COOLDOWN = 300     # microsteps before a failed leg may be retried
PORTAL_DRINK_SPACING = 20         # microsteps between script-owned drinks
PORTAL_WALK_REJECTIONS = 8        # consecutive rejected walks before giving up
PORTAL_CAST_BEATS = 12            # microsteps to wait for the missile to appear
PORTAL_LIFETIME_TICKS = 100       # missiles.cpp AddTownPortal duration
PORTAL_TALK_ATTEMPTS = 8          # Adria can open a Q_MUSHROOM dialog first
PORTAL_CLEAR_WALK_BEATS = 60      # walk toward the up-stairs this long for a clear cast beat

TOWN_PORTAL_MISSILE = 10          # MissileID::TownPortal
WITCH_PORTAL_INDEX = 2            # SpawnWitch pins {mana, full mana, portal}
PORTAL_SCROLL_BASE_ID = 27        # IDI_PORTAL
PORTAL_SCROLL_PRICE = 200         # itemdat.tsv, level-independent
WITCH_POSITION = (80, 20)         # towners.tsv TOWN_WITCH
TOWN_PORTAL_POSITION = (57, 40)   # portal.cpp PortalTownPosition[0]

__all__ = ["RESOURCE_PORTAL_PROTOCOLS", "PortalPolicy", "PortalService",
           "validate_portal_protocol", "validate_native_portal",
           "PORTAL_SERVICE_CAP", "PORTAL_MAX_PER_EPISODE",
           "PORTAL_SCROLL_BASE_ID", "PORTAL_SCROLL_PRICE",
           "TOWN_PORTAL_MISSILE", "WITCH_PORTAL_INDEX"]


@dataclass(frozen=True)
class PortalPolicy:
    """The manager-side portal law.  Thresholds mirror RetreatPolicy so the two
    interfaces fire on exactly the same evidence; only the vehicle differs."""
    hp_fraction: float = 0.5              # leave when hp <= hp_fraction * max_hp
    empty_belt_hp_fraction: float = 0.75  # ... or the belt is empty and hp <= this
    pressure_radius: int = 6
    pressure_count: int = 0               # 0 = pressure trigger off
    drink_hp_fraction: float = 0.4        # drink while travelling below this
    cast_clear_radius: int = 2            # cast only when no monster stands within this radius (0 = off)
    scroll_budget: int = 2                # witch purchases allowed per episode

    def __post_init__(self):
        for name in ("hp_fraction", "empty_belt_hp_fraction", "drink_hp_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1]")
        for name in ("pressure_radius", "pressure_count", "cast_clear_radius", "scroll_budget"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")

    def as_dict(self):
        return {"hp_fraction": self.hp_fraction,
                "empty_belt_hp_fraction": self.empty_belt_hp_fraction,
                "pressure_radius": self.pressure_radius,
                "pressure_count": self.pressure_count,
                "drink_hp_fraction": self.drink_hp_fraction,
                "cast_clear_radius": self.cast_clear_radius,
                "scroll_budget": self.scroll_budget}


# ---------------------------------------------------------------- raw readers
def _state(raw):
    state = raw.get("resource_state") if isinstance(raw, dict) else None
    return state if isinstance(state, dict) else {}


def belt_heals(raw):
    kinds = raw.get("belt_heal_kinds")
    if kinds is not None:
        return sum(int(value) in (1, 2, 3, 4) for value in kinds)
    return int(_state(raw).get("readiness", {}).get("belt_heals", 0) or 0)


def alive_monsters_within(raw, radius):
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


def portal_scrolls(raw):
    """Carried Scrolls of Town Portal, with the INVITEM_* ``spell_from`` code the
    native cast action needs.  ``inventory_items`` is armor-only, so this list is
    the ONLY place a scroll is observable."""
    scrolls = _state(raw).get("portal_scrolls")
    return list(scrolls) if isinstance(scrolls, list) else []


def portals_started(raw):
    """Engine counter of COMPLETED round trips (incremented on arrival back in
    the dungeon, not at authorization)."""
    return int(_state(raw).get("portals_started", 0) or 0)


def portal_open(raw):
    return bool(_state(raw).get("portal_open", False))


def portal_level(raw):
    return int(_state(raw).get("portal_level", 0) or 0)


def portal_missile_tile(raw):
    """Absolute tile of this level's town-portal missile, or None.

    ``portal_x``/``portal_y`` carry the DUNGEON tile the scroll was read on, so
    in town they are the wrong answer; the missile list is authoritative in both
    scenes.  Missile tiles are exported relative to the player."""
    px, py = int(raw["player_x"]), int(raw["player_y"])
    best = None
    for missile in raw.get("missiles", ()):
        if int(missile.get("type", -1)) != TOWN_PORTAL_MISSILE:
            continue
        if bool(missile.get("deleted", False)):
            continue
        tile = (px + int(missile.get("tile_dx", 0)), py + int(missile.get("tile_dy", 0)))
        distance = max(abs(tile[0] - px), abs(tile[1] - py))
        if best is None or distance < best[0]:
            best = (distance, tile)
    return None if best is None else best[1]


@dataclass
class PortalService:
    policy: PortalPolicy = field(default_factory=PortalPolicy)
    attempted: bool = False
    active: bool = False
    mission: str | None = None
    phase: str = "idle"
    reason: str | None = None
    trigger: str | None = None
    start_steps: int = 0
    steps: int = 0
    start_depth: int = 0
    drinks: int = 0
    casts: int = 0
    scrolls_bought: int = 0
    gold_spent: int = 0
    awaiting_return: bool = False
    attempts: list = field(default_factory=list)
    phase_steps: dict = field(default_factory=dict)
    _last_phase: str = "idle"
    _last_phase_step: int = 0
    _cooldown_until: int = 0
    _walk_rejections: int = 0
    _talk_attempts: int = 0
    _last_drink_step: int | None = None
    _give_up: str | None = None
    _cast_step: int | None = None
    _cast_spell_from: int | None = None
    _portal_seen: bool = False
    _enter_deadline: int | None = None
    _shop_failures: set = field(default_factory=set)

    # ---- clocks (the shape the RESUPPLY window loop expects of a service) ----
    @property
    def service_microstep_cap(self):
        return PORTAL_SERVICE_CAP

    @property
    def settle_exempt(self):
        """True while a cast is in flight: the coach-v03 settle loop must not
        wait-cancel the queued spell (see the hold command)."""
        return bool(self.active and self.mission == "outbound" and self.phase == "cast"
                    and self._cast_step is not None)

    @property
    def command_microstep_deadline(self):
        return self.start_steps + PORTAL_SERVICE_CAP

    def record_steps(self, absolute_steps, phase=None):
        elapsed = int(absolute_steps) - self._last_phase_step
        charged = self._last_phase if phase is None else phase
        if elapsed > 0:
            self.phase_steps[charged] = self.phase_steps.get(charged, 0) + elapsed
        self._last_phase_step = int(absolute_steps)
        self.steps = int(absolute_steps) - self.start_steps

    # ---- the manager-side law ----
    def trigger_reason(self, raw, steps):
        """Return the trigger name if a portal leg should start now, else None.

        The outbound law is RetreatPolicy's law plus "a scroll is carried", so a
        coach that consults this service first and the walking retreat second
        automatically prefers the portal whenever the pair actually has one."""
        if self.active or raw is None:
            return None
        if raw.get("dead") or raw.get("game_over") or raw.get("is_set_level"):
            return None
        depth = int(raw.get("dungeon_level", 0))
        if int(steps) < self._cooldown_until:
            return None
        if depth == 0:
            return self._town_trigger(raw)
        if depth < 2:
            return None
        if portals_started(raw) >= PORTAL_MAX_PER_EPISODE:
            return None
        if not portal_scrolls(raw):
            return None
        if portal_open(raw) and portal_level(raw) == depth:
            # A portal is already standing here: crossing it is the "enter"
            # phase's job, not a new cast, and it costs no scroll.
            return "portal_standing"
        hp, max_hp = int(raw.get("hp", 0)), max(1, int(raw.get("max_hp", 1)))
        if hp <= 0:
            return None
        belt = belt_heals(raw)
        if hp <= self.policy.hp_fraction * max_hp:
            return "low_hp"
        if belt == 0 and hp <= self.policy.empty_belt_hp_fraction * max_hp:
            return "empty_belt"
        if (self.policy.pressure_count
                and alive_monsters_within(raw, self.policy.pressure_radius) >= self.policy.pressure_count):
            return "pressure"
        return None

    def _town_trigger(self, raw):
        if self.awaiting_return and portal_open(raw) and portal_level(raw) >= 2:
            # Defence in depth: the outbound law already refuses to leave once
            # the engine budget is spent, so this can only fire while a transit
            # is still authorizable -- but never hand start() a request the
            # engine would answer with an exception inside action_masks().
            if portals_started(raw) >= PORTAL_MAX_PER_EPISODE:
                return None
            return "portal_return"
        if self.awaiting_return:
            return None
        if self.scrolls_bought >= self.policy.scroll_budget:
            return None
        if portal_scrolls(raw):
            return None
        if int(raw.get("gold", 0)) < PORTAL_SCROLL_PRICE:
            return None
        if belt_heals(raw) < int(_state(raw).get("readiness", {}).get("required_belt_heals", 4) or 4):
            # The engine refuses the sale until the four heals are in the belt.
            return None
        return "buy_scroll"

    # ---- window start ----
    def start(self, raw, steps, bridge, trigger):
        if self.active:
            raise RuntimeError("portal leg already active")
        depth = int(raw.get("dungeon_level", 0))
        if raw.get("is_set_level"):
            raise RuntimeError("portal legs belong to the main dungeon")
        if trigger == "buy_scroll":
            if depth != 0:
                raise RuntimeError("the witch leg runs in town")
            mission, phase = "restock", "shop"
        elif trigger == "portal_return":
            if depth != 0:
                raise RuntimeError("the return leg starts in town")
            bridge.configure_resource_portal(True)
            mission, phase = "return", "enter"
        else:
            if depth < 2:
                raise RuntimeError("a portal can be read only on main L2 or deeper")
            bridge.configure_resource_portal(True)
            mission = "outbound"
            phase = "enter" if trigger == "portal_standing" else "cast"
        self.attempted = self.active = True
        self.mission = mission
        self.phase = self._last_phase = phase
        self.reason = None
        self.trigger = trigger
        self.start_steps = self._last_phase_step = int(steps)
        self.steps = 0
        self.start_depth = depth
        self._walk_rejections = 0
        self._talk_attempts = 0
        self._give_up = None
        self._last_drink_step = None
        self._cast_step = None
        self._cast_spell_from = None
        self._portal_seen = phase == "enter" and mission == "outbound"
        self._enter_deadline = None
        self.attempts.append({
            "mission": mission, "trigger": trigger, "beat0": int(steps),
            "depth0": depth, "hp0": int(raw.get("hp", 0)),
            "max_hp": int(raw.get("max_hp", 0)), "belt0": belt_heals(raw),
            "scrolls0": len(portal_scrolls(raw)),
            "monsters_near0": (alive_monsters_within(raw, self.policy.pressure_radius)
                               if depth else 0),
            "outcome": None, "reason": None, "beat1": None, "hp1": None,
            "drinks": 0, "gold_spent": 0})

    # ---- the worker-side hands ----
    def command(self, env, bridge):
        self.record_steps(env._steps)
        command = self._command(env, bridge)
        self._last_phase = self.phase
        return command

    def _command(self, env, bridge):
        raw = env._raw
        depth = int(raw["dungeon_level"])
        if raw.get("is_set_level"):
            return self._end(env, bridge, "portal_unexpected_scene", failed=True)
        if raw.get("dead") or raw.get("game_over") or int(raw.get("hp", 0)) <= 0:
            return self._end(env, bridge, "portal_died", failed=True)
        if self._give_up is not None:
            return self._end(env, bridge, self._give_up, failed=True)
        if self.steps >= self.service_microstep_cap:
            return self._end(env, bridge, "portal_cap", failed=True)
        if self.mission == "restock":
            return self._restock(env, bridge, raw, depth)
        if self.mission == "outbound":
            return self._outbound(env, bridge, raw, depth)
        return self._return(env, bridge, raw, depth)

    # ---- leg: buy the scroll from Adria ----
    def _restock(self, env, bridge, raw, depth):
        if depth != 0:
            return self._end(env, bridge, "portal_unexpected_scene", failed=True)
        if portal_scrolls(raw):
            return self._end(env, bridge, "portal_scroll_carried", failed=False)
        if int(raw.get("gold", 0)) < PORTAL_SCROLL_PRICE:
            return self._end(env, bridge, "portal_scroll_unaffordable", failed=True)
        town = _state(raw).get("town", {})
        command = self._visit(env, "witch", town)
        if command is not None:
            return command
        candidates = [item for item in town.get("stock", [])
                      if item.get("vendor") == "witch"
                      and int(item.get("base_id", -1)) == PORTAL_SCROLL_BASE_ID
                      and item.get("can_use") and item.get("can_fit")
                      and int(item.get("price", 0)) <= int(raw.get("gold", 0))
                      and self._stock_key(item) not in self._shop_failures]
        candidates.sort(key=lambda item: (int(item["price"]), int(item["index"])))
        if not candidates:
            return self._end(env, bridge, "portal_scroll_unavailable", failed=True)
        return ("buy", *self._stock_key(candidates[0]))

    # ---- leg: read the scroll and cross to town ----
    def _outbound(self, env, bridge, raw, depth):
        if depth == 0:
            # Arrived. The engine receipt already opened the shops; the ordinary
            # town service owns the itinerary from here.
            self.awaiting_return = True
            return self._end(env, bridge, "portal_to_town", failed=False)
        if depth != self.start_depth:
            return self._end(env, bridge, "portal_unexpected_scene", failed=True)
        drink = self._maybe_drink(env, raw)
        if drink is not None:
            return drink
        if self.phase == "cast":
            if ((portal_open(raw) and portal_level(raw) == depth)
                    or portal_missile_tile(raw) is not None):
                return self._enter_phase(env, bridge, raw, depth)
            if self._cast_step is None:
                scrolls = portal_scrolls(raw)
                if not scrolls:
                    return self._end(env, bridge, "portal_scroll_missing", failed=True)
                if (self.policy.cast_clear_radius
                        and alive_monsters_within(raw, self.policy.cast_clear_radius)):
                    # Reading the scroll under attack is interrupted (PM_GOTHIT) and
                    # never fires: step toward the up-stairs (the walking-retreat
                    # path) and cast the first clear beat; give the leg back after
                    # PORTAL_CLEAR_WALK_BEATS so the stairs retreat can take over.
                    if int(env._steps) - self.start_steps >= PORTAL_CLEAR_WALK_BEATS:
                        return self._end(env, bridge, "portal_cast_crowded", failed=True)
                    command = self._walk_to_upstairs(env, bridge, raw)
                    return command if command is not None else ("wait",)
                self._cast_spell_from = int(scrolls[0]["spell_from"])
                self._cast_step = int(env._steps)
                return ("cast_portal", self._cast_spell_from)
            if int(env._steps) - self._cast_step >= PORTAL_CAST_BEATS:
                # AddTownPortal can set _miDelFlag when no valid tile exists in
                # radius 0..5, yet CastSpell still consumed the scroll. Burned.
                return self._end(env, bridge, "portal_cast_failed", failed=True)
            # The spell is queued / animating: a wait would cancel it (act_wait =
            # player.Stop + StartStand on PM_SPELL). Hold the engine instead.
            return ("hold",)
        return self._enter(env, bridge, raw, depth)

    def _enter_phase(self, env, bridge, raw, depth):
        """The portal exists: switch to the walk-in phase and arm its deadline.

        AddTownPortal gives the missile 100 game ticks, so at the default
        ticks_per_step=4 the pair has only ~25 microsteps to cover the up-to-five
        tiles the Crawl spiral may have placed it at. That deadline is real."""
        self.phase = "enter"
        self._portal_seen = True
        if self._enter_deadline is None:
            ticks = max(1, int(getattr(env, "ticks_per_step", 4) or 4))
            self._enter_deadline = int(env._steps) + max(1, PORTAL_LIFETIME_TICKS // ticks)
        return self._enter(env, bridge, raw, depth)

    def _enter(self, env, bridge, raw, depth):
        if self._enter_deadline is not None and int(env._steps) > self._enter_deadline:
            return self._end(env, bridge, "portal_expired", failed=True)
        tile = portal_missile_tile(raw)
        if tile is None and not (portal_open(raw) and portal_level(raw) == depth):
            if self._portal_seen:
                return self._end(env, bridge, "portal_expired", failed=True)
            return self._end(env, bridge, "portal_missing", failed=True)
        command = self._enter_step(env, raw)
        if command is None:
            return self._end(env, bridge, "portal_unreachable", failed=True)
        return command

    def _enter_step(self, env, raw):
        tile = portal_missile_tile(raw)
        if tile is None:
            state = _state(raw)
            tile = (int(state.get("portal_x", 0)), int(state.get("portal_y", 0)))
        px, py = int(raw["player_x"]), int(raw["player_y"])
        if (px, py) == tile:
            return ("wait",)
        return self._walk(env, tile[0], tile[1])

    # ---- leg: cross the town-side portal back to the dungeon ----
    def _return(self, env, bridge, raw, depth):
        if depth >= 2:
            self.awaiting_return = False
            return self._end(env, bridge, "portal_return", failed=False)
        if depth != 0:
            return self._end(env, bridge, "portal_unexpected_scene", failed=True)
        if not portal_open(raw):
            self.awaiting_return = False
            return self._end(env, bridge, "portal_closed", failed=True)
        town = _state(raw).get("town", {})
        if town.get("dialog_active") or town.get("active_vendor") not in (None, "none"):
            return ("dismiss",)
        tile = portal_missile_tile(raw) or TOWN_PORTAL_POSITION
        px, py = int(raw["player_x"]), int(raw["player_y"])
        if (px, py) == tuple(tile):
            return ("wait",)
        command = self._walk(env, int(tile[0]), int(tile[1]))
        if command is None:
            return self._end(env, bridge, "portal_unreachable", failed=True)
        return command

    # ---- shared helpers ----
    def _maybe_drink(self, env, raw):
        hp, max_hp = int(raw.get("hp", 0)), max(1, int(raw.get("max_hp", 1)))
        if (hp <= self.policy.drink_hp_fraction * max_hp and belt_heals(raw) > 0
                and (self._last_drink_step is None
                     or env._steps - self._last_drink_step >= PORTAL_DRINK_SPACING)):
            self._last_drink_step = int(env._steps)
            return ("drink",)
        return None

    def _visit(self, env, vendor, town):
        """Open ``vendor``'s store, dismissing anything else that is on screen.

        Adria can answer with the Q_MUSHROOM speech instead of her store
        (towners.cpp TalkToWitch), which is exactly the hazard the town
        service's dismiss/8-attempt loop already handles for Pepin."""
        if town.get("active_vendor") == vendor and not town.get("dialog_active"):
            return None
        if town.get("dialog_active") or town.get("active_vendor") not in (None, "none", vendor):
            return ("dismiss",)
        npc = next((n for n in town.get("npcs", []) if n.get("type") == vendor), None)
        if npc is None:
            self._give_up = "portal_npc_missing"
            return ("wait",)
        raw = env._raw
        px, py = int(raw["player_x"]), int(raw["player_y"])
        if max(abs(int(npc["x"]) - px), abs(int(npc["y"]) - py)) <= 1:
            self._talk_attempts += 1
            if self._talk_attempts > PORTAL_TALK_ATTEMPTS:
                self._give_up = "portal_vendor_unavailable"
                return ("wait",)
            return ("talk", int(npc["id"]))
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            command = self._walk(env, int(npc["x"]) + dx, int(npc["y"]) + dy)
            if command is not None:
                return command
        self._give_up = "portal_npc_unreachable"
        return ("wait",)

    @staticmethod
    def _stock_key(item):
        return (item["vendor"], int(item["index"]), int(item["seed_hi"]),
                int(item["seed_lo"]), int(item["create_info"]), int(item["base_id"]))

    def _walk_to_upstairs(self, env, bridge, raw):
        message = getattr(bridge, "WM_DIABPREVLVL", None)
        if message is None:
            return None
        stairs = [t for t in raw.get("triggers", []) if t.get("msg") == message]
        if not stairs:
            return None
        px, py = int(raw["player_x"]), int(raw["player_y"])
        stairs.sort(key=lambda t: max(abs(int(t["x"]) - px), abs(int(t["y"]) - py)))
        target = stairs[0]
        if (px, py) == (int(target["x"]), int(target["y"])):
            return ("wait",)
        return self._walk(env, int(target["x"]), int(target["y"]))

    @staticmethod
    def _walk(env, x, y):
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

    # ---- receipts ----
    def receipt(self, command, receipt):
        kind = command[0]
        if kind in ("walk", "open"):
            if receipt.get("accepted"):
                self._walk_rejections = 0
            else:
                self._walk_rejections += 1
                if self._walk_rejections >= PORTAL_WALK_REJECTIONS:
                    self._give_up = "portal_walk_rejected"
        elif kind == "drink" and receipt.get("accepted"):
            self.drinks += 1
            if self.attempts:
                self.attempts[-1]["drinks"] += 1
        elif kind == "cast_portal":
            if receipt.get("accepted"):
                self.casts += 1
            else:
                # The scroll was never read, so nothing was spent; retry next leg.
                self._cast_step = None
                self._give_up = "portal_cast_rejected"
        elif kind == "buy":
            if receipt.get("accepted"):
                self.scrolls_bought += 1
                paid = int(receipt.get("price", 0))
                self.gold_spent += paid
                if self.attempts:
                    self.attempts[-1]["gold_spent"] += paid
            else:
                self._shop_failures.add(tuple(command[1:]))
                if receipt.get("reason") == "belt_reserved_for_heals":
                    self._give_up = "portal_belt_reserved"

    def _end(self, env, bridge, reason, *, failed):
        self.active = False
        self.phase = "done"
        self.reason = reason
        if failed:
            self._cooldown_until = int(env._steps) + PORTAL_FAILURE_COOLDOWN
        if self.mission in ("outbound", "return"):
            # Never leave a stale authorization behind: the engine consumes it on
            # arrival, so revoking after a SUCCESSFUL leg is a harmless no-op and
            # after a failed one it closes the guard again.
            try:
                bridge.configure_resource_portal(False)
            except RuntimeError as exc:  # engine out of game after death
                if self.attempts:
                    self.attempts[-1]["revoke_error"] = str(exc)
        if failed and self.mission == "outbound":
            self.awaiting_return = False
        if self.attempts:
            record = self.attempts[-1]
            record["outcome"] = "failed" if failed else "completed"
            record["reason"] = reason
            record["beat1"] = int(env._steps)
            record["hp1"] = int(env._raw.get("hp", 0))
        self.mission = None
        return ("complete",)

    def telemetry(self):
        return {"protocol": "portal-v1", "policy": self.policy.as_dict(),
                "attempted": self.attempted, "active": self.active,
                "mission": self.mission, "phase": self.phase, "reason": self.reason,
                "trigger": self.trigger, "steps": self.steps, "drinks": self.drinks,
                "casts": self.casts, "scrolls_bought": self.scrolls_bought,
                "gold_spent": self.gold_spent, "awaiting_return": self.awaiting_return,
                "attempts": [dict(a) for a in self.attempts],
                "to_town": sum(1 for a in self.attempts if a.get("reason") == "portal_to_town"),
                "returned": sum(1 for a in self.attempts if a.get("reason") == "portal_return"),
                "failed": sum(1 for a in self.attempts if a.get("outcome") == "failed"),
                "phase_steps": dict(self.phase_steps)}
