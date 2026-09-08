"""R18-H identify-v1 (2026-09-07) 凯恩鉴定: Cain identifies the loot before it
is sold.

主席 2026-09-07 14:00 裁定:城镇行程只用铁匠、医者、女巫;未鉴定的魔法物品
只能按 ``_ivalue`` 的四分之一卖出,并且装备时不带任何词缀。凯恩(TOWN_STORY)
以固定 100 金币鉴定一件物品(``stores.h`` ``StorytellerIdentifyPrice``),之后
同一件物品按 ``_iIvalue`` 的四分之一卖出(``stores.cpp`` ``NormalStoreSellPrice``),
词缀也随 ``CalcPlrInv`` 生效。

The law this script executes, per that ruling:

* it runs during EVERY town trip, at the earliest town phase, strictly BEFORE
  the loot service's ``sell_idle`` stage sells anything;
* it identifies every unidentified magic item carried or worn, in
  best-expected-gain-first order;
* it stops the moment ``gold - price`` would fall below the joint minimum
  readiness budget the town itinerary needs for potions (and, when the shop
  quotes happen to be observable, armour) -- the potion buy is never starved;
  R18-F portal-v1's 200-gold Scroll of Town Portal, bought later in the same
  trip, is measured next to that budget and binds only under the chairman
  switch ``IDENTIFY_RESERVE_PORTAL_SCROLL``;
* it is bounded by the town trip's OWN microstep clock as well as its own cap,
  so the beats it spends can never push the itinerary into
  ``resource_service_cap`` (R18-H review round 2026-09-07);
* it then hands control back with ``("complete",)`` so the ordinary itinerary
  sells the now-identified items at their identified value or keeps them.

The fee is booked into the loot service's own trip ledger by ``OptionsEnv``
(exactly like R18-F booked the witch's scroll), so the trip cash reconciliation
stays exact and the economy audit raises nothing.

``resource_identify="off"`` (the default) constructs no service, makes no native
call, adds no env kwarg and leaves every frozen path byte-identical.

UI 独立性:引擎侧 ``TryIdentifyItem`` 不读任何商店状态,bridge 也从不为凯恩
开店,所以本脚本只需要真的走到凯恩身边(和真人一样),没有 talk/store 这一步。
"""
from copy import deepcopy
from dataclasses import dataclass, field

RESOURCE_IDENTIFY_PROTOCOLS = ("off", "cain-v1")

IDENTIFY_SERVICE_CAP = 600        # microsteps for the whole identify leg
# R18-H review round (2026-09-07): the leg runs FIRST inside the town trip and
# its beats are charged to that trip's own 3000-microstep budget
# (completion_clock service_microsteps), so an unbounded leg could push the
# itinerary into "resource_service_cap" before the potions are bought. The leg
# therefore always leaves the trip IDENTIFY_TRIP_MARGIN microsteps -- half the
# recipe's trip budget, for the smith sales, the healer, the potions and the
# walk back -- and refuses to start at all when less than IDENTIFY_MIN_LEG is
# left for itself (the measured approach to Cain alone cost 229).
IDENTIFY_TRIP_MARGIN = 1500       # microsteps of the town trip reserved for the itinerary
# R18-H review round (2026-09-07): R18-F portal-v1 buys a 200-gold Scroll of
# Town Portal LATER in the same town trip, and refuses outright below that
# price, so the Cain fee can leave that purchase unaffordable at the moment the
# witch leg runs. Whether the scroll therefore belongs in the readiness reserve
# is a THRESHOLD ruling (it decides how much gold the leg may spend), so it is
# the chairman's, not mine: the amount is ALWAYS computed and ALWAYS reported
# (``portal_scroll_reserve``), and this switch decides whether it BINDS.
# False = the behaviour the H2 smoke measured. True was measured too: on all
# four smoke seeds it defers every leg and the on-arm becomes identical to the
# off-arm (report, review round). Flip the constant, nothing else.
IDENTIFY_RESERVE_PORTAL_SCROLL = False
IDENTIFY_MIN_LEG = 300            # a shorter leg cannot even reach Cain; defer instead
IDENTIFY_FAILURE_COOLDOWN = 300   # microsteps before a failed leg may be retried
IDENTIFY_WALK_REJECTIONS = 8      # consecutive rejected walks before giving up
IDENTIFY_PRICE = 100              # stores.h StorytellerIdentifyPrice (unchanged)
STORYTELLER_POSITION = (62, 71)   # towners.tsv TOWN_STORY "Cain the Elder"
# itemdat.tsv IDI_HEAL (Potion of Healing) value; items.cpp SpawnHealer pins it
# at HealerItems[0] on every restock, so it is the cheapest instant heal the
# town itinerary can ever be quoted. Used only as the FLOOR of the medicine
# reserve while the healer's real stock is not observable (stock is exported
# only inside the corresponding vendor's open store).
HEALER_MIN_HEAL_PRICE = 50

__all__ = ["RESOURCE_IDENTIFY_PROTOCOLS", "IDENTIFY_PRICE", "IDENTIFY_SERVICE_CAP",
           "IDENTIFY_TRIP_MARGIN", "IDENTIFY_MIN_LEG", "IDENTIFY_RESERVE_PORTAL_SCROLL",
           "STORYTELLER_POSITION", "IdentifyService", "identify_reserve",
           "validate_identify_protocol", "validate_identify_env", "validate_native_identify"]


# ------------------------------------------------------------- validators
def validate_identify_protocol(protocol, policy, identify):
    """R18-H identify-v1: the Cain identify leg of the loot economy town trip.

    It spends the town trip's gold and books the fee in that trip's ledger, so
    it is defined only inside sustain-loot-v1 under l2-town-v1. Default off is
    byte-identical."""
    if identify not in RESOURCE_IDENTIFY_PROTOCOLS:
        raise ValueError(
            f"Unknown resource_identify {identify!r}; expected one of {RESOURCE_IDENTIFY_PROTOCOLS}")
    if identify != "off":
        if protocol != "l2-town-v1":
            raise ValueError("identify-v1 requires l2-town-v1")
        if policy != "sustain-loot-v1":
            raise ValueError("identify-v1 requires the sustain-loot-v1 town trip")
    return identify


def validate_native_identify(raw, expected="off"):
    """Fail closed: the native identify flag must match the Python contract."""
    state = raw.get("resource_state", {}) if isinstance(raw, dict) else {}
    if not isinstance(state, dict) or not state:
        if expected != "off":
            raise RuntimeError("Native identify identity mismatch; isolated rebuild required")
        return
    observed = state.get("identify_enabled", False)
    if type(observed) is not bool or observed != (expected != "off"):
        raise RuntimeError("Native identify identity mismatch; isolated rebuild required")


# ---------------------------------------------------------- raw readers
def _state(raw):
    state = raw.get("resource_state") if isinstance(raw, dict) else None
    return state if isinstance(state, dict) else {}


def _town(raw):
    town = _state(raw).get("town")
    return town if isinstance(town, dict) else {}


def _identity(item):
    return tuple(int(item[k]) for k in ("seed_hi", "seed_lo", "create_info", "base_id"))


def _portal_scroll_reserve(raw):
    """R18-H review round (2026-09-07): R18-F portal-v1 buys its 200-gold Scroll
    of Town Portal LATER IN THE SAME TRIP (the witch leg runs at phase
    ``return``), and refuses outright below that price. This is what the leg
    would have to keep back to leave that purchase affordable. Zero whenever
    portal-v1 is off, a scroll is already carried, or the engine round-trip
    budget is spent -- nothing left to protect. Whether it BINDS is
    ``IDENTIFY_RESERVE_PORTAL_SCROLL``; the number is reported either way."""
    from .resource_portal import (PORTAL_MAX_PER_EPISODE, PORTAL_SCROLL_PRICE,
                                  portal_scrolls, portals_started)
    if _state(raw).get("portal_enabled") is not True:
        return 0
    if portal_scrolls(raw) or portals_started(raw) >= PORTAL_MAX_PER_EPISODE:
        return 0
    return PORTAL_SCROLL_PRICE


def identify_reserve(raw):
    """The joint minimum readiness budget the town itinerary must keep.

    Reuses the town service's own accounting: ``plan_basket`` computes
    ``minimum_observed_total`` = the cheapest fully quoted (medicine + gear)
    basket. That number exists only while the shop stock is observable, and the
    identify leg deliberately runs BEFORE any vendor is visited, so in practice
    the answer is the medicine floor: the still-missing belt heals priced at the
    cheapest instant heal Pepin always pins. Whichever is larger wins, so the
    potion buy can never be starved by an identify fee. The portal scroll the
    same trip may want is measured next to it (``_portal_scroll_reserve``) and
    added only when ``IDENTIFY_RESERVE_PORTAL_SCROLL`` says it must bind.

    R18-H review round (2026-09-07): ``town["stock"]`` is only the CURRENTLY
    OPEN vendor's page, so it is split per vendor before it is handed to
    ``plan_basket``; and with no page open at all the planner would answer a
    zero-cost basket nobody quoted, which used to be reported as
    ``source: "joint_basket"``. That verdict is now reserved for a basket that
    really was quoted."""
    from .resource_protocol import native_readiness
    from .resource_sustain import plan_basket
    ready = native_readiness(raw)
    deficit = max(0, int(ready["required_belt_heals"]) - int(ready["belt_heals"]))
    floor = deficit * HEALER_MIN_HEAL_PRICE
    scroll = _portal_scroll_reserve(raw)
    binding = scroll if IDENTIFY_RESERVE_PORTAL_SCROLL else 0
    town = _town(raw)
    stock = town.get("stock", []) or []
    smith_stock = [item for item in stock if item.get("vendor") == "smith"]
    healer_stock = [item for item in stock if item.get("vendor") == "healer"]

    def answer(reserve, observed, source):
        return {"reserve": int(reserve) + binding, "medicine_floor": floor,
                "medicine_deficit": deficit, "observed_minimum_total": observed,
                "portal_scroll_reserve": scroll,
                "portal_scroll_reserved": bool(binding), "source": source}

    if not smith_stock and not healer_stock:
        # The normal case at this phase: no vendor page is open, so no basket
        # was ever quoted. Only the medicine floor (and the scroll) can bind.
        return answer(floor, None, "medicine_floor")
    try:
        plan = plan_basket(raw, smith_stock, town.get("repair_quotes", []) or [], healer_stock)
    except Exception:                      # planning is advisory here, never fatal
        return answer(floor, None, "medicine_floor_plan_failed")
    observed = plan.get("minimum_observed_total")
    if observed is None:
        return answer(floor, None, "medicine_floor")
    reserve = max(floor, int(observed))
    return answer(reserve, int(observed),
                  "joint_basket" if reserve == int(observed) else "medicine_floor")


def identify_candidates(raw):
    """Every quote Cain would list, best expected sale gain first.

    The ruling is "identify every unidentified magic item carried", so there is
    NO profitability filter here -- identification also turns the affixes on for
    the gear plan. The ordering only decides who gets identified first when the
    readiness reserve cuts the leg short."""
    candidates = []
    for quote in _town(raw).get("identify_quotes", []) or []:
        if quote.get("identified") is not False or int(quote.get("quality", 0)) == 0:
            continue
        if quote.get("is_quest") is True:
            continue
        price = quote.get("identify_price")
        if type(price) is not int or price <= 0:
            continue
        before = quote.get("sale_price")
        after = quote.get("sale_price_identified")
        gain = (int(after) - int(before)) if isinstance(before, int) and isinstance(after, int) else 0
        candidates.append((-gain, _identity(quote), dict(quote, expected_sale_gain=gain)))
    candidates.sort(key=lambda row: row[:2])
    return [row[2] for row in candidates]


@dataclass
class IdentifyService:
    """The caller creates a new instance on every normal episode reset."""
    attempted: bool = False
    active: bool = False
    phase: str = "idle"
    reason: str | None = None
    trigger: str | None = None
    start_steps: int = 0
    steps: int = 0
    identified_count: int = 0
    identify_gold_spent: int = 0
    sale_income_identified: int = 0
    identified_items: list = field(default_factory=list)
    skipped_for_reserve: list = field(default_factory=list)
    deferrals: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    phase_steps: dict = field(default_factory=dict)
    last_reserve: dict | None = None
    _last_phase: str = "idle"
    _last_phase_step: int = 0
    _cooldown_until: int = 0
    _budget: int = IDENTIFY_SERVICE_CAP
    _deferred_trips: set = field(default_factory=set)
    _walk_rejections: int = 0
    _give_up: str | None = None
    _served_trips: set = field(default_factory=set)
    _identified: set = field(default_factory=set)
    _attempted_identities: set = field(default_factory=set)
    _trip: int = 0

    # ---- clocks (the shape the RESUPPLY window loop expects of a service) ----
    @property
    def service_microstep_cap(self):
        # The leg's own cap, already narrowed to what the town trip can lend it.
        return self._budget

    @property
    def settle_exempt(self):
        return False

    @property
    def command_microstep_deadline(self):
        return self.start_steps + self.service_microstep_cap

    @staticmethod
    def leg_budget(trip_budget_left):
        """How many microsteps the leg may spend (R18-H review round).

        ``trip_budget_left`` is the town trip's own remaining microsteps
        (``start_steps + service_microstep_cap - now``). ``None`` means the
        caller has no trip clock to protect (unit tests, standalone use)."""
        if trip_budget_left is None:
            return IDENTIFY_SERVICE_CAP
        return min(IDENTIFY_SERVICE_CAP,
                   max(0, int(trip_budget_left) - IDENTIFY_TRIP_MARGIN))

    def record_steps(self, absolute_steps, phase=None):
        elapsed = int(absolute_steps) - self._last_phase_step
        charged = self._last_phase if phase is None else phase
        if elapsed > 0:
            self.phase_steps[charged] = self.phase_steps.get(charged, 0) + elapsed
        self._last_phase_step = int(absolute_steps)
        self.steps = int(absolute_steps) - self.start_steps

    # ---- the manager-side law ----
    def trigger_reason(self, raw, steps, trip, trip_budget_left=None):
        """Return "identify" if the leg should run now for town trip ``trip``.

        One leg per town trip, in town, with something identifiable that the
        readiness reserve can actually afford AND enough of the town trip's own
        clock left that the itinerary behind us still fits (R18-H review round).
        The manager law calls this at the single pre-sale beat of the trip, so a
        refusal here means the trip runs without a Cain leg -- which is why both
        refusals are recorded in ``deferrals`` instead of vanishing."""
        if self.active or raw is None:
            return None
        if raw.get("dead") or raw.get("game_over") or raw.get("is_set_level"):
            return None
        if int(raw.get("dungeon_level", 0)) != 0:
            return None
        if _state(raw).get("identify_enabled") is not True:
            return None
        if int(trip) <= 0 or int(trip) in self._served_trips:
            return None
        if int(steps) < self._cooldown_until:
            return None
        budget = self.leg_budget(trip_budget_left)
        if not self._affordable(raw)[0]:
            if any(_identity(item) not in self._attempted_identities
                   for item in identify_candidates(raw)):
                # There WAS something to open; only the reserve refused it.
                # Nothing to identify at all is not a deferral, it is a quiet trip.
                self._defer(raw, steps, trip, "identify_deferred_no_funds",
                            budget, trip_budget_left)
            return None
        if budget < IDENTIFY_MIN_LEG:
            self._defer(raw, steps, trip, "identify_deferred_trip_budget",
                        budget, trip_budget_left)
            return None
        return "identify"

    def _defer(self, raw, steps, trip, reason, budget, trip_budget_left):
        """Record (once per town trip) that the leg was NOT run and why."""
        if int(trip) in self._deferred_trips:
            return
        self._deferred_trips.add(int(trip))
        self.deferrals.append({
            "trip": int(trip), "beat": int(steps), "reason": reason,
            "gold": int(raw.get("gold", 0)),
            "reserve": deepcopy(identify_reserve(raw)),
            "candidates": len(identify_candidates(raw)),
            "leg_budget": int(budget),
            "trip_budget_left": (None if trip_budget_left is None
                                 else int(trip_budget_left))})

    def _affordable(self, raw):
        reserve = identify_reserve(raw)
        gold = int(raw.get("gold", 0))
        affordable = [item for item in identify_candidates(raw)
                      if _identity(item) not in self._attempted_identities
                      and gold - int(item["identify_price"]) >= reserve["reserve"]]
        return affordable, reserve

    # ---- window start ----
    def start(self, raw, steps, trigger, trip, trip_budget_left=None):
        if self.active:
            raise RuntimeError("identify leg already active")
        if int(raw.get("dungeon_level", 0)) != 0 or raw.get("is_set_level"):
            raise RuntimeError("the Cain identify leg runs in town")
        budget = self.leg_budget(trip_budget_left)
        if budget < IDENTIFY_MIN_LEG:
            # Fail closed: the manager law refuses this case in trigger_reason.
            raise RuntimeError(
                "the Cain identify leg needs the town trip microsteps the manager law checks")
        self._budget = budget
        self.attempted = self.active = True
        self.phase = self._last_phase = "approach"
        self.reason = None
        self.trigger = trigger
        self.start_steps = self._last_phase_step = int(steps)
        self.steps = 0
        self._walk_rejections = 0
        self._give_up = None
        self._trip = int(trip)
        self._served_trips.add(int(trip))
        self._attempted_identities = set()
        affordable, reserve = self._affordable(raw)
        self.last_reserve = reserve
        self.attempts.append({
            "trip": int(trip), "trigger": trigger, "beat0": int(steps),
            "leg_budget": int(budget),
            "trip_budget_left0": (None if trip_budget_left is None
                                  else int(trip_budget_left)),
            "gold0": int(raw.get("gold", 0)), "reserve0": deepcopy(reserve),
            "candidates0": len(identify_candidates(raw)),
            "affordable0": len(affordable), "identified": 0, "gold_spent": 0,
            "outcome": None, "reason": None, "beat1": None, "gold1": None})

    # ---- the worker-side hands ----
    def command(self, env, bridge):
        self.record_steps(env._steps)
        command = self._command(env, bridge)
        self._last_phase = self.phase
        return command

    def _command(self, env, bridge):
        raw = env._raw
        if raw.get("is_set_level") or int(raw["dungeon_level"]) != 0:
            return self._end(env, "identify_unexpected_scene", failed=True)
        if raw.get("dead") or raw.get("game_over") or int(raw.get("hp", 0)) <= 0:
            return self._end(env, "identify_died", failed=True)
        if self._give_up is not None:
            return self._end(env, self._give_up, failed=True)
        if self.steps >= self.service_microstep_cap:
            return self._end(env, "identify_cap", failed=True)
        town = _town(raw)
        if town.get("dialog_active") or town.get("active_vendor") not in (None, "none"):
            # Nothing may be open while the pair stands at Cain; the native
            # action refuses outright (IsPlayerInStore) rather than guessing.
            return ("dismiss",)
        affordable, reserve = self._affordable(raw)
        self.last_reserve = reserve
        if not affordable:
            remaining = [i for i in identify_candidates(raw)
                         if _identity(i) not in self._attempted_identities]
            if remaining:
                self.skipped_for_reserve.extend(
                    {"identity": list(_identity(i)), "name": i.get("name"),
                     "identify_price": int(i["identify_price"]),
                     "gold": int(raw.get("gold", 0)), "reserve": deepcopy(reserve)}
                    for i in remaining)
                return self._end(env, "identify_reserve_exhausted", failed=False)
            return self._end(env, "identify_complete" if self.identified_count
                             else "identify_nothing_to_do", failed=False)
        if _state(raw).get("identify_at_storyteller") is not True:
            self.phase = "approach"
            command = self._approach(env, raw, town)
            if command is None:
                return self._end(env, "identify_unreachable", failed=True)
            return command
        self.phase = "identify"
        item = affordable[0]
        return ("identify", bool(item["equipped"]), int(item["index"]),
                *_identity(item), int(item["identify_price"]))

    # ---- walking to Cain ----
    def _approach(self, env, raw, town):
        npc = next((n for n in town.get("npcs", []) if n.get("type") == "storyteller"), None)
        target = ((int(npc["x"]), int(npc["y"])) if npc is not None else STORYTELLER_POSITION)
        px, py = int(raw["player_x"]), int(raw["player_y"])
        if max(abs(target[0] - px), abs(target[1] - py)) <= 1:
            # Adjacent already, yet the native gate says no (mid-walk, held item,
            # a store still closing): spend one honest beat and look again.
            return ("wait",)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            command = self._walk(env, target[0] + dx, target[1] + dy)
            if command is not None:
                return command
        return None

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
                if self._walk_rejections >= IDENTIFY_WALK_REJECTIONS:
                    self._give_up = "identify_walk_rejected"
            return
        if kind != "identify":
            return
        if not isinstance(receipt, dict) or not isinstance(receipt.get("accepted"), bool):
            # The house guard (resource_sustain_loot.py) BEFORE any identity key
            # is dereferenced: a missing audit must fail closed, not KeyError.
            raise RuntimeError("Resource identify receipt needs a native boolean accepted")
        identity = tuple(int(v) for v in command[3:7])
        if _identity(receipt) != identity:
            raise RuntimeError("Identify receipt source identity mismatch")
        if receipt.get("quoted_price") != command[-1]:
            raise RuntimeError("Identify receipt quote mismatch")
        if receipt.get("received") not in (0, None):
            raise RuntimeError("Identification must never report income")
        self._attempted_identities.add(identity)
        gold_before = receipt.get("gold_before")
        gold_after = receipt.get("gold_after")
        if type(gold_before) is not int or type(gold_after) is not int:
            raise RuntimeError("Identify receipt lacks a native wallet")
        price = int(receipt.get("price", 0) or 0)
        if not receipt.get("accepted"):
            if price != 0 or gold_after != gold_before:
                raise RuntimeError("Rejected identification moved cash")
            self.failures.append({"identity": list(identity),
                                  "reason": receipt.get("reason")})
            return
        if price != int(command[-1]) or gold_before - gold_after != price:
            raise RuntimeError("Accepted identification lacks the exact quoted cash effect")
        item = receipt.get("item") or {}
        if item.get("identified") is not True:
            raise RuntimeError("Accepted identification did not set _iIdentified")
        self.identified_count += 1
        self.identify_gold_spent += price
        self._identified.add(identity)
        self.identified_items.append({
            "identity": list(identity), "name": item.get("name"),
            "equipped": bool(command[1]), "index": int(command[2]),
            "price": price, "value": item.get("value"),
            "identified_value": item.get("identified_value"),
            "sale_price": item.get("sale_price"),
            "sale_price_identified": item.get("sale_price_identified"),
            "sale_income": 0})
        if self.attempts:
            self.attempts[-1]["identified"] += 1
            self.attempts[-1]["gold_spent"] += price

    def note_sale(self, identity, received):
        """Attribute a later Smith sale to this leg when the item is one of ours."""
        key = tuple(int(v) for v in identity)
        if key not in self._identified:
            return 0
        amount = int(received)
        self.sale_income_identified += amount
        for row in self.identified_items:
            if tuple(row["identity"]) == key:
                row["sale_income"] += amount
        return amount

    def _end(self, env, reason, *, failed):
        self.active = False
        self.phase = "done"
        self.reason = reason
        if failed:
            self._cooldown_until = int(env._steps) + IDENTIFY_FAILURE_COOLDOWN
        if self.attempts:
            record = self.attempts[-1]
            record["outcome"] = "failed" if failed else "completed"
            record["reason"] = reason
            record["beat1"] = int(env._steps)
            record["gold1"] = int(env._raw.get("gold", 0))
        return ("complete",)

    def telemetry(self):
        return {"protocol": "cain-v1", "price": IDENTIFY_PRICE,
                "attempted": self.attempted, "active": self.active,
                "phase": self.phase, "reason": self.reason, "trigger": self.trigger,
                "steps": self.steps, "trips_served": sorted(self._served_trips),
                "identified_count": self.identified_count,
                "identify_gold_spent": self.identify_gold_spent,
                "identified_items": deepcopy(self.identified_items),
                "sale_income_identified": self.sale_income_identified,
                "skipped_for_reserve": deepcopy(self.skipped_for_reserve),
                "deferrals": deepcopy(self.deferrals),
                "leg_budget": self._budget, "trip_margin": IDENTIFY_TRIP_MARGIN,
                "failures": deepcopy(self.failures),
                "last_reserve": deepcopy(self.last_reserve),
                "attempts": [dict(a) for a in self.attempts],
                "completed": sum(1 for a in self.attempts if a.get("outcome") == "completed"),
                "failed": sum(1 for a in self.attempts if a.get("outcome") == "failed"),
                "phase_steps": dict(self.phase_steps)}


def validate_identify_env(protocol, loot_economy, identify):
    """DiabloGymEnv-side twin of validate_identify_protocol.

    The base env knows the loot economy flag, not the manager-side service
    policy, so the same law is stated in the terms it owns: identify-v1 lives
    inside the loot economy town trip under l2-town-v1. Default off is
    byte-identical."""
    if identify not in RESOURCE_IDENTIFY_PROTOCOLS:
        raise ValueError(
            f"Unknown resource_identify {identify!r}; expected one of {RESOURCE_IDENTIFY_PROTOCOLS}")
    if identify != "off":
        if protocol != "l2-town-v1":
            raise ValueError("identify-v1 requires l2-town-v1")
        if not loot_economy:
            raise ValueError("identify-v1 requires the loot economy town trip")
    return identify
