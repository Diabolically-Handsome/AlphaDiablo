"""R18-K (2026-09-07) smith-v1: the surplus weapon upgrade leg at Griswold.

The town itinerary buys smith gear only to clear a readiness FAILURE
({armor, damage, weapon}) and potions up to the belt target.  The warrior's
starting short sword already clears the ``damage >= 6`` bar, so nothing in the
itinerary ever proposes a better weapon and the pair carries that sword for the
whole episode (hit_damage 7-9 at clvl 3-5 in the R18 probes) while gold piles
up.  This law spends the SURPLUS -- the gold left after the readiness basket and
the potion target are already paid for -- on exactly one better one-handed melee
weapon per town trip.

Where it runs.  Inside the ordinary town trip, at the same seam R18-F uses for
the witch: the town service has reached its ``return`` phase (basket done,
potions done, sale done) and has not yet walked back to the stairs.  The leg is
decided BEFORE the town service is asked for a command, so no generated command
is ever discarded without a receipt.  The portal leg keeps priority; the weapon
leg never runs while a portal leg is active or awaited.

What it will not do (v1):

* one-handed melee only (``ILOC_ONEHAND``; Sword/Axe/Mace/Staff).  A two-hander
  would displace the shield, and the shield is an ARMOR decision this law is
  forbidden to touch.  A bow would also change the attack style the a12 melee
  worker is frozen on.
* normal quality with no damage affixes only.  The projection below models
  ``_pIMaxDam + _pDamageMod``; an affixed item would make it wrong, so such an
  item is rejected rather than guessed at.
* at most one purchase per trip, and the displaced weapon is never sold in the
  same trip (it stays in the inventory; the next trip's ordinary sale phase may
  quote it like any other carried item).
* never spend the potion floor.  R18-K set the reserve to a flat 0 because the
  belt is normally already full when this leg starts.  R18-K2b replaces that
  with the SAME floor H2's identify leg reserves
  (``resource_identify.identify_reserve``): the still-missing belt heals priced
  at the cheapest instant heal Pepin can ever quote,
  ``(required_belt_heals - belt_heals) * 50`` -- 50 being ``IDI_HEAL``'s value
  in ``assets/txtdata/items/itemdat.tsv``, which ``items.cpp SpawnHealer`` pins
  at ``HealerItems[0]`` on every restock.  With a full belt (the normal case at
  this seam) the floor is 0 and the budget is the whole balance, so the R18-K
  counterfactual still applies; with an unfinished belt the weapon can no longer
  eat the potion money.  The joint-basket half of H2's reserve is deliberately
  NOT reused: Griswold's stock is observable at this seam, so a gear reserve
  would double-count the basket this trip has already paid for.  Any further
  forward reserve (a repair fund, the 200 gold a portal scroll costs) is still
  left to a later version.

Native evidence.  Two native verdicts drive the choice, both already exported
for weapons on every smith stock row:

* ``upgrade`` -- ``PlanGearUpgrade`` validity, i.e. the engine's own strict
  verdict that the candidate raises the AGGREGATE combat utility of the whole
  equipped set after identification, slot replacement, pairing rules and every
  attribute cascade.  This is the same projection ``a14`` uses to accept a floor
  drop; it is authoritative and it is the gate.
* ``max_damage`` / ``effect_damage_mod`` -- the item fields the native readiness
  metric ``damage = _pIMaxDam + _pDamageMod`` is computed from.  These order the
  gated candidates; they never widen the gate.
* R18-K2b adds two more, exported for a weapon only while the native scope is
  on: ``is_upgrade_weapon`` (the item is inside weapon-purchase-v1's scope) and
  ``can_equip`` / ``equip_reason`` (``PlanResourceArmor`` can actually wear it
  right now -- pairing rules, room for the displaced item, life and stat
  cascades, and a strictly higher readiness damage).  When ``is_upgrade_weapon``
  is present, ``can_equip`` becomes a hard gate, so gold is never spent on a
  weapon the native equip path would refuse.  ``dry-v1`` runs with the native
  scope OFF and therefore cannot see those two fields; its counterfactual is
  measured on the gates R18-K already had.

``resource_weapon_upgrade="off"`` (the default) constructs no service, passes no
new env kwarg, adds no telemetry key and leaves every frozen path byte-identical.

FAIL CLOSED -- R18-K2b (2026-09-07) supplies the native leg this law was waiting
for.  ``ConfigureResourceWeaponPurchase`` (``bridge.configure_resource_weapon_
purchase``) is off by default; while it is off Griswold's counter and the town
equip path stay armor-only and answer every weapon ``unsupported_item``, exactly
as the R18-K blocker receipt recorded.  While it is on, ``ActBuyStoreItem`` also
sells an ordinary ONE-HANDED weapon and ``PlanResourceArmor`` plans the hand slot
with a14's own ``PlanGearUpgrade`` (target slot, displaced slots and the strict
conservative full-set utility increase), refusing two-handers so the shield slot
logic never changes.  ``validate_native_weapon_upgrade`` checks the
``weapon_purchase_enabled`` handshake on every observation, exactly as
``validate_native_retreat`` / ``validate_native_portal`` do for their laws, and
``smith-v1`` refuses to run against a bridge that does not acknowledge the scope.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

RESOURCE_WEAPON_UPGRADES = ("off", "dry-v1", "smith-v1")

WEAPON_UPGRADE_SERVICE_CAP = 400   # microsteps for one weapon leg
WEAPON_UPGRADE_RESERVE = 0         # the BASE reserve; the potion floor is added on top
# IDI_HEAL's value in assets/txtdata/items/itemdat.tsv.  items.cpp SpawnHealer
# pins IDI_HEAL at HealerItems[0] on every restock (infinite stock in single
# player), so it is always available and always the cheapest instant heal.
# Same constant, same source, as H2's identify_reserve.
HEALER_MIN_HEAL_PRICE = 50
WEAPON_UPGRADE_TALK_ATTEMPTS = 8   # Griswold can answer with a quest dialog first
WEAPON_UPGRADE_WALK_REJECTIONS = 8 # consecutive rejected walks before giving up
WEAPON_UPGRADE_MIN_GAIN = 1        # native readiness damage points

ILOC_ONEHAND = 1                   # items.h item_equip_type
# ItemType: Misc 0, Sword 1, Axe 2, Bow 3, Mace 4, Shield 5, LightArmor 6,
# Helm 7, MediumArmor 8, HeavyArmor 9, Staff 10.  Bow is deliberately absent.
MELEE_WEAPON_ITEM_TYPES = (1, 2, 4, 10)
SHIELD_ITEM_TYPE = 5               # ItemType::Shield
HAND_SLOTS = (4, 5)                # INVLOC_HAND_LEFT, INVLOC_HAND_RIGHT
# Affixes that would invalidate the damage projection below.
DAMAGE_AFFIX_FIELDS = ("effect_damage", "effect_damage_mod", "effect_strength",
                       "effect_dexterity", "effect_to_hit")
# R18-K (2026-09-07) review round: items.cpp CalcPlrDamage (Source/items.cpp
# 2494-2513) floors an unarmed player at _pIMaxDam = 3 while a shield is held
# and at 1 bare handed (the monk branch cannot apply -- the frozen worker is a
# warrior).  R18-K read "no hand weapon" as a displaced max damage of 0, which
# over-stated every bare-handed projection by 1 to 3 points.
UNARMED_MAX_DAMAGE_WITH_SHIELD = 3
UNARMED_MAX_DAMAGE_BARE = 1

__all__ = ["RESOURCE_WEAPON_UPGRADES", "validate_weapon_upgrade",
           "validate_native_weapon_upgrade", "plan_weapon_upgrade",
           "weapon_trigger_reason", "weapon_upgrade_reserve",
           "book_weapon_purchase", "unarmed_max_damage", "holds_shield",
           "WeaponUpgradeService", "WEAPON_UPGRADE_SERVICE_CAP",
           "WEAPON_UPGRADE_RESERVE", "HEALER_MIN_HEAL_PRICE",
           "MELEE_WEAPON_ITEM_TYPES"]


# ------------------------------------------------------------------ validators
def validate_weapon_upgrade(protocol, policy, mode, upgrade):
    """R18-K rides on the loot itinerary; default off is byte-identical.

    ``dry-v1`` is the OBSERVATION-ONLY twin of ``smith-v1``: at the same seam it
    records the plan smith-v1 would execute and issues no command, no microstep
    and no native call, so its rows stay bit-identical to the control arm.  It
    exists because the executable arm is blocked by the armor-only native
    counter, and the counterfactual is still worth measuring.
    """
    if upgrade not in RESOURCE_WEAPON_UPGRADES:
        raise ValueError(
            f"Unknown resource_weapon_upgrade {upgrade!r}; expected one of {RESOURCE_WEAPON_UPGRADES}")
    if upgrade != "off":
        if protocol != "l2-town-v1":
            raise ValueError("smith-v1 requires l2-town-v1")
        if policy != "sustain-loot-v1":
            raise ValueError("smith-v1 requires the sustain-loot-v1 town itinerary")
        if mode != "full":
            raise ValueError("smith-v1 requires the full purchase mode")
    return upgrade


def validate_native_weapon_upgrade(raw, expected="off"):
    """Fail closed: the native weapon scope must match the Python contract.

    Only ``smith-v1`` needs the native scope; ``dry-v1`` observes and never buys,
    so it must see the scope OFF.  R18-K2b's bridge exports
    ``weapon_purchase_enabled`` on every observation; a bridge that predates it
    exports nothing, which reads as ``False`` and therefore refuses ``smith-v1``.
    Refusing here is deliberate: issuing a purchase the engine answers with
    ``unsupported_item`` would burn the trip's microsteps and quietly produce an
    arm that differs from control only by wasted time.
    """
    state = raw.get("resource_state", {}) if isinstance(raw, dict) else {}
    if not isinstance(state, dict):
        state = {}
    observed = state.get("weapon_purchase_enabled", False)
    if type(observed) is not bool or observed != (expected == "smith-v1"):
        raise RuntimeError(
            "Native weapon-purchase scope identity mismatch "
            f"(observed {observed!r}, resource_weapon_upgrade={expected!r}); "
            "weapon-purchase-v1 is the R18-K2b native leg and must be switched on "
            "with bridge.configure_resource_weapon_purchase(True) by the same "
            "process. A bridge older than R18-K2b sells and equips ordinary armor "
            "only (ActBuyStoreItem/PlanResourceArmor answer isWeapon() with "
            "unsupported_item) and needs an isolated rebuild.")


def configure_native_weapon_purchase(native, upgrade):
    """Write the native weapon scope for THIS episode, in BOTH directions.

    R18-K2b review round (2026-09-07): ``gResourceWeaponPurchase`` lives in the
    ENGINE process, not in the env, and nothing else ever clears it --
    ``ConfigureResourceProtocol`` rewrites retreat and portal on every configure
    but never touches this flag.  Writing it only for ``smith-v1`` therefore
    leaked a ``True`` into every later env in the same worker: its first
    observation raised the handshake mismatch in
    ``_validate_native_resource_flags`` and the process could never recover, so
    an A/B pair could not share a worker the way retreat/portal pairs do.
    Writing it unconditionally is legal at this seam because ``end_game`` has
    already run (the switch may change only between episodes) and because
    ``ConfigureResourceWeaponPurchase(false)`` has no protocol precondition.
    A bridge that predates the entry point still runs every off arm; only
    ``smith-v1`` fails closed there.  Returns the value written.
    """
    wanted = upgrade == "smith-v1"
    if not hasattr(native, "configure_resource_weapon_purchase"):
        if wanted:
            raise RuntimeError(
                "Native bridge lacks weapon-purchase-v1 "
                "(bridge.configure_resource_weapon_purchase); "
                "resource_weapon_upgrade=smith-v1 requires an isolated R18-K2b rebuild")
        return False
    native.configure_resource_weapon_purchase(wanted)
    return wanted


# ------------------------------------------------------------------ raw readers
def _state(raw):
    state = raw.get("resource_state") if isinstance(raw, dict) else None
    return state if isinstance(state, dict) else {}


def _readiness(raw):
    ready = _state(raw).get("readiness")
    return ready if isinstance(ready, dict) else {}


def stock_key(item):
    """The tuple the native buy command carries (identical to ResourceService)."""
    return (item["vendor"], int(item["index"]), int(item["seed_hi"]),
            int(item["seed_lo"]), int(item["create_info"]), int(item["base_id"]))


def identity(item):
    return tuple(int(item[key]) for key in
                 ("seed_hi", "seed_lo", "create_info", "base_id"))


def holds_shield(raw):
    """A shield in either hand (items.cpp isHoldingItem(ItemType::Shield))."""
    for slot, item in enumerate(raw.get("equipped_items", ())):
        if (slot in HAND_SLOTS and item.get("present")
                and int(item.get("item_type", -1)) == SHIELD_ITEM_TYPE):
            return True
    return False


def unarmed_max_damage(raw):
    """The engine's unarmed ``_pIMaxDam`` floor (R18-K review round).

    ``CalcPlrDamage`` never lets a player deal zero: with no weapon equipped it
    sets ``maxDamage = 3`` while a shield is held and ``1`` otherwise.  The
    projection below subtracts what the swap actually removes, so a bare-handed
    player displaces this floor, not 0.
    """
    return UNARMED_MAX_DAMAGE_WITH_SHIELD if holds_shield(raw) else UNARMED_MAX_DAMAGE_BARE


def equipped_weapon(raw):
    """The hand item the swap displaces, or None.

    ``PlanGearUpgrade`` picks the hand itself; for the ordering key only the
    weapon that actually contributes ``_pIMaxDam`` matters, so take the hand
    weapon with the largest max damage.

    R18-K review round: the DISPLACED item is screened for the same affixes the
    candidates are.  ``_pDamageMod`` is a function of ``_pStrength``
    (``items.cpp CalcPlrDamageMod``: warrior ``_pDamageMod = level * strength /
    100``), so unequipping a weapon that carries ``_iPLStr`` -- or ``_iPLDam`` /
    ``_iPLDamMod`` -- lowers the readiness damage by an amount the arithmetic
    below never subtracts.  ``unmodelled`` reports that; ``plan_weapon_upgrade``
    refuses to plan at all rather than sell a wrong number as a damage gain.
    """
    best = None
    for slot, item in enumerate(raw.get("equipped_items", ())):
        if slot not in HAND_SLOTS or not item.get("present"):
            continue
        if int(item.get("max_damage", 0)) <= 0:
            continue          # a shield contributes no damage and is not displaced
        affixes = {name: int(item.get(name, 0)) for name in DAMAGE_AFFIX_FIELDS}
        entry = {"slot": slot, "identity": identity(item),
                 "min_damage": int(item.get("min_damage", 0)),
                 "max_damage": int(item.get("max_damage", 0)),
                 "durability": int(item.get("durability", 0)),
                 "quality": int(item.get("quality", 0)),
                 "affixes": affixes,
                 "unmodelled": int(item.get("quality", 0)) != 0 or any(affixes.values())}
        if best is None or entry["max_damage"] > best["max_damage"]:
            best = entry
    return best


def weapon_upgrade_reserve(raw):
    """The potion floor this leg must never spend (R18-K2b).

    Identical in definition and in source constant to H2's
    ``resource_identify.identify_reserve`` medicine floor: the still-missing belt
    heals priced at the cheapest instant heal Pepin always pins.  H2's joint
    basket half is deliberately not reused -- see the module docstring.
    """
    ready = _readiness(raw)
    required = int(ready.get("required_belt_heals", 4) or 0)
    heals = int(ready.get("belt_heals", 0) or 0)
    deficit = max(0, required - heals)
    floor = deficit * HEALER_MIN_HEAL_PRICE
    return {"reserve": max(WEAPON_UPGRADE_RESERVE, floor),
            "base_reserve": WEAPON_UPGRADE_RESERVE,
            "medicine_floor": floor, "medicine_deficit": deficit,
            # R18-K2b review round (2026-09-07): name the term that actually
            # binds. The base reserve is 0 and the floor is never negative, so
            # ">=" made this a tautology that reported medicine_floor even when
            # the belt was full and nothing at all was reserved.
            "source": "medicine_floor" if floor > WEAPON_UPGRADE_RESERVE else "base"}


# ------------------------------------------------------------- the choice rule
def plan_weapon_upgrade(raw, smith_stock, *, reserve=None, failed=(), require_native=False):
    """Rank the affordable, natively approved one-handed melee weapons.

    Every rejection is recorded with its reason: this catalog is evidence, not a
    score.  ``chosen`` is the maximum native projected damage gain, then the
    cheaper item, then the lower observed stock index (a deterministic tie).

    ``require_native`` is what ``smith-v1`` passes.  The executable arm runs with
    weapon-purchase-v1 ON, so every smith row carries the two native verdicts
    (``is_upgrade_weapon`` and ``can_equip``/``equip_reason``); demanding them
    turns ``can_equip`` -- which includes the engine's own ``no_damage_gain``
    refusal -- into an unconditional gate instead of one that quietly disappears
    when the field is absent.  ``dry-v1`` observes with the scope OFF and cannot
    see them, so it keeps the default.
    """
    cash = int(raw.get("gold", 0))
    ready = _readiness(raw)
    damage_before = int(ready.get("damage", 0))
    current = equipped_weapon(raw)
    replaced = unarmed_max_damage(raw) if current is None else current["max_damage"]
    # R18-K2b: the potion floor is the default reserve; an explicit number still
    # wins so a caller (and every boundary test) can pin it.
    detail = weapon_upgrade_reserve(raw)
    reserve = detail["reserve"] if reserve is None else int(reserve)
    budget = max(0, cash - int(reserve))
    failed = {tuple(command) for command in failed}
    candidates, rejected = [], []

    def reject(item, reason):
        rejected.append({"identity": identity(item), "index": int(item.get("index", -1)),
                         "price": int(item.get("price", 0)), "reason": reason})

    # R18-K review round: the projection models the DISPLACED item too.  When
    # a14 has auto-equipped a magical or affixed hand weapon, unequipping it also
    # moves _pDamageMod (via _iPLStr) and the arithmetic below would report a
    # gain that the engine will not deliver.  Fail closed: plan nothing, and say
    # so in the catalog rather than buying on a number this law cannot model.
    displaced_unmodelled = current is not None and current["unmodelled"]
    for item in smith_stock:
        if displaced_unmodelled:
            reject(item, "displaced_weapon_unmodelled")
            continue
        if item.get("vendor") != "smith":
            reject(item, "foreign_vendor")
            continue
        if (int(item.get("equip_loc", 0)) != ILOC_ONEHAND
                or int(item.get("item_type", -1)) not in MELEE_WEAPON_ITEM_TYPES):
            reject(item, "not_one_handed_melee_weapon")
            continue
        if not item.get("can_use"):
            reject(item, "cannot_use")
            continue
        if not item.get("can_fit"):
            reject(item, "no_room")
            continue
        if not item.get("durable"):
            reject(item, "not_durable")
            continue
        if int(item.get("quality", 0)) != 0 or item.get("effects_active") is not True:
            reject(item, "unmodelled_quality")
            continue
        if any(int(item.get(name, 0)) for name in DAMAGE_AFFIX_FIELDS):
            reject(item, "unmodelled_affix")
            continue
        if item.get("upgrade") is not True:
            # The native aggregate verdict is the gate, never the arithmetic.
            reject(item, "native_not_an_upgrade")
            continue
        if require_native and item.get("is_upgrade_weapon") is not True:
            # R18-K review round: smith-v1 runs with weapon-purchase-v1 ON, so a
            # missing or false scope verdict means the engine would answer this
            # purchase with unsupported_item. Never spend the microsteps.
            reject(item, "native_scope_missing")
            continue
        if ((require_native or item.get("is_upgrade_weapon") is True)
                and item.get("can_equip") is not True):
            # R18-K2b: with the native scope on, PlanResourceArmor answers for
            # this exact item -- including its own no_damage_gain refusal, which
            # is the check that a14's AGGREGATE utility verdict does not make.
            # Its verdict is the last word: gold is never spent on a weapon the
            # native equip path would refuse or that would not raise the
            # readiness damage this law exists to raise.
            reject(item, "native_cannot_equip")
            continue
        if stock_key(item) in failed:
            reject(item, "previous_failure")
            continue
        price = int(item.get("price", 0))
        if price <= 0 or price > budget:
            reject(item, "over_budget")
            continue
        projected = damage_before - replaced + int(item.get("max_damage", 0))
        gain = projected - damage_before
        if gain < WEAPON_UPGRADE_MIN_GAIN:
            reject(item, "no_projected_damage_gain")
            continue
        candidates.append({"identity": identity(item), "index": int(item["index"]),
                           "price": price, "item_type": int(item["item_type"]),
                           "min_damage": int(item.get("min_damage", 0)),
                           "max_damage": int(item.get("max_damage", 0)),
                           "native_upgrade": True,
                           "damage_before": damage_before,
                           "projected_damage_after": projected,
                           "projected_damage_gain": gain,
                           "stock_key": stock_key(item)})
    candidates.sort(key=lambda c: (-c["projected_damage_gain"], c["price"], c["index"]))
    return {"cash": cash, "reserve": int(reserve), "reserve_detail": detail,
            "budget": budget,
            "damage_before": damage_before,
            "equipped_weapon": deepcopy(current),
            "replaced_max_damage": replaced,
            "candidates": candidates, "rejected": rejected,
            "chosen": deepcopy(candidates[0]) if candidates else None,
            "displaced_unmodelled": displaced_unmodelled,
            "unresolved": [name for name, missing in (
                ("no_budget", budget <= 0),
                ("no_stock", not smith_stock),
                ("displaced_unmodelled", displaced_unmodelled),
                ("no_candidate", not candidates)) if missing]}


def weapon_trigger_plan(raw, smith_stock, *, reserve=None, failed=(), require_native=False):
    """Return (trigger name, plan): the trigger fires when the surplus can buy.

    ``smith_stock`` is the stock the town service already surveyed at Griswold
    THIS trip (same town seed and restock sequence); the leg re-surveys and
    re-plans on fresh quotes before it spends anything.
    """
    if raw is None or raw.get("dead") or raw.get("game_over") or raw.get("is_set_level"):
        return None, None
    if int(raw.get("dungeon_level", -1)) != 0:
        return None, None
    plan = plan_weapon_upgrade(raw, smith_stock, reserve=reserve, failed=failed,
                               require_native=require_native)
    return ("surplus_gold" if plan["chosen"] is not None else None), plan


def weapon_trigger_reason(raw, smith_stock, *, reserve=None, failed=(), require_native=False):
    """The trigger name alone -- the frozen signature every caller already uses."""
    return weapon_trigger_plan(raw, smith_stock, reserve=reserve, failed=failed,
                               require_native=require_native)[0]


def book_weapon_purchase(town_service, weapon_service, active_service, command, receipt):
    """Book an accepted in-trip weapon exactly as R18-F books the witch scroll.

    R18-K review round: this IS the production booking block; ``OptionsEnv``
    calls nothing else, so the ledger test drives the shipped code instead of
    re-implementing it.  The town trip's cash reconciliation
    (``SustainLootService._seal_trip``) subtracts ``gold_spent`` from the trip's
    opening wallet, so a purchase made by a nested service that is not booked
    here raises "Completed loot trip has unexplained cash movement".

    Returns the gold booked (0 when the guard refuses), so a caller can assert on
    it.  Every guard term is the one the R18-F scroll block uses.
    """
    if weapon_service is None or active_service is not weapon_service:
        return 0
    if not command or command[0] != "buy":
        return 0
    if not receipt.get("accepted"):
        return 0
    if not getattr(town_service, "active", False):
        return 0
    price = int(receipt.get("price", 0) or 0)
    town_service.purchases += 1
    town_service.gold_spent += price
    return price


# ----------------------------------------------------------------- the service
@dataclass
class WeaponUpgradeService:
    """One scripted buy+equip leg per town trip.  Never ends the episode."""
    protocol: str = "smith-v1"
    seams: list = field(default_factory=list)
    attempted: bool = False
    active: bool = False
    phase: str = "idle"
    reason: str | None = None
    trigger: str | None = None
    start_steps: int = 0
    steps: int = 0
    trips_served: int = 0
    weapon_upgrades: int = 0
    weapon_gold_spent: int = 0
    purchases: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    phase_steps: dict = field(default_factory=dict)
    _last_phase: str = "idle"
    _last_phase_step: int = 0
    _served_trip: int | None = None
    _plan: dict | None = None
    # R18-K2b review round (2026-09-07): the plan the TRIGGER law built, kept so
    # the reserve is auditable at every seam the law evaluated, not only at the
    # seams that won a leg (see telemetry).
    _seam_plan: dict | None = None
    _pending: dict | None = None
    _shop_failures: set = field(default_factory=set)
    _equip_failures: set = field(default_factory=set)
    _talk_attempts: int = 0
    _walk_rejections: int = 0
    _give_up: str | None = None

    # ---- the shape the RESUPPLY window loop expects of a service ----
    @property
    def service_microstep_cap(self):
        return WEAPON_UPGRADE_SERVICE_CAP

    @property
    def command_microstep_deadline(self):
        return self.start_steps + WEAPON_UPGRADE_SERVICE_CAP

    def record_steps(self, absolute_steps, phase=None):
        elapsed = int(absolute_steps) - self._last_phase_step
        charged = self._last_phase if phase is None else phase
        if elapsed > 0:
            self.phase_steps[charged] = self.phase_steps.get(charged, 0) + elapsed
        self._last_phase_step = int(absolute_steps)
        self.steps = int(absolute_steps) - self.start_steps

    # ---- the manager-side law ----
    def trip_trigger_reason(self, raw, smith_stock, trip):
        """One leg per town trip, and only while a purchase is actually possible."""
        if self.protocol != "smith-v1" or self.active or self._served_trip == int(trip):
            return None
        # R18-K review round: the executable arm demands the native verdicts.
        # R18-K2b review round: keep the plan this evaluation built. The reserve
        # law binds at EVERY seam it is asked about, but only a seam that wins
        # reaches _command, so a seam refused for over_budget -- exactly the
        # seam where the reserve may have bound -- used to leave no record.
        reason, plan = weapon_trigger_plan(raw, smith_stock, failed=self._shop_failures,
                                           require_native=True)
        if plan is not None:
            self._seam_plan = deepcopy(plan)
        return reason

    def observe_seam(self, raw, smith_stock, trip, steps):
        """dry-v1: record what smith-v1 WOULD do here, and issue nothing.

        ``smith_stock`` is the survey the town service already holds for THIS
        trip, so this consumes no native call, no command and no microstep: the
        arm's rows stay bit-identical to the control arm.
        """
        if self.protocol != "dry-v1" or self._served_trip == int(trip):
            return None
        self._served_trip = int(trip)
        self.trips_served += 1
        plan = plan_weapon_upgrade(raw, smith_stock)
        self._plan = deepcopy(plan)
        chosen = plan["chosen"]
        seam = {"trip": int(trip), "beat": int(steps), "gold": plan["cash"],
                "budget": plan["budget"], "stock_seen": len(smith_stock),
                "weapon_damage_before": plan["damage_before"],
                "would_fire": chosen is not None,
                "would_price": None if chosen is None else chosen["price"],
                "would_damage_after": None if chosen is None else chosen["projected_damage_after"],
                "would_damage_gain": None if chosen is None else chosen["projected_damage_gain"],
                "candidates": len(plan["candidates"]),
                "reject_reasons": sorted({r["reason"] for r in plan["rejected"]}),
                "unresolved": list(plan["unresolved"])}
        self.seams.append(seam)
        return seam

    # ---- window start ----
    def start(self, raw, steps, trigger, trip):
        if self.active:
            raise RuntimeError("weapon upgrade leg already active")
        if int(raw.get("dungeon_level", -1)) != 0:
            raise RuntimeError("the weapon upgrade leg runs in town")
        validate_native_weapon_upgrade(raw, "smith-v1")
        self.attempted = self.active = True
        self.phase = self._last_phase = "shop"
        self.reason = None
        self.trigger = trigger
        self.start_steps = self._last_phase_step = int(steps)
        self.steps = 0
        self._served_trip = int(trip)
        self.trips_served += 1
        self._plan = None
        self._pending = None
        self._talk_attempts = 0
        self._walk_rejections = 0
        self._give_up = None
        self.attempts.append({
            "trip": int(trip), "trigger": trigger, "beat0": int(steps),
            "gold0": int(raw.get("gold", 0)),
            "weapon_damage_before": int(_readiness(raw).get("damage", 0)),
            "weapon_damage_after": None, "outcome": None, "reason": None,
            "beat1": None, "gold_spent": 0, "price": None, "identity": None})

    # ---- the worker-side hands ----
    def command(self, env, bridge):
        self.record_steps(env._steps)
        command = self._command(env, bridge)
        self._last_phase = self.phase
        return command

    def _command(self, env, bridge):
        raw = env._raw
        if raw.get("is_set_level") or int(raw["dungeon_level"]) != 0:
            return self._end(env, "weapon_unexpected_scene", failed=True)
        if raw.get("dead") or raw.get("game_over") or int(raw.get("hp", 0)) <= 0:
            return self._end(env, "weapon_died", failed=True)
        if self._give_up is not None:
            return self._end(env, self._give_up, failed=True)
        if self.steps >= self.service_microstep_cap:
            return self._end(env, "weapon_cap", failed=True)
        town = _state(raw).get("town", {})
        if self.phase == "equip":
            return self._equip(env, raw)
        command = self._visit(env, "smith", town)
        if command is not None:
            return command
        # Fresh, real quotes at the shop -- never spend on the remembered survey.
        stock = [item for item in town.get("stock", []) if item.get("vendor") == "smith"]
        plan = plan_weapon_upgrade(raw, stock, failed=self._shop_failures,
                                   require_native=True)
        self._plan = deepcopy(plan)
        chosen = plan["chosen"]
        if chosen is None:
            return self._end(env, "weapon_no_candidate", failed=False)
        self._pending = deepcopy(chosen)
        return ("buy", *chosen["stock_key"])

    def _equip(self, env, raw):
        pending = self._pending
        if pending is None:
            return self._end(env, "weapon_equip_lost", failed=True)
        wanted = tuple(pending["identity"])
        if any(item.get("present") and identity(item) == wanted
               for item in raw.get("equipped_items", ())):
            return self._settle(env, raw)
        # inventory_items is armor-only; inventory_equipment carries every
        # equippable carried item with its live index.
        carried = next((item for item in _state(raw).get("inventory_equipment", [])
                        if identity(item) == wanted), None)
        if carried is None:
            return self._end(env, "weapon_purchase_not_carried", failed=True)
        if (int(carried["index"]), *wanted) in self._equip_failures:
            return self._end(env, "weapon_equip_rejected", failed=True)
        return ("equip", int(carried["index"]), *wanted)

    def _settle(self, env, raw):
        pending = self._pending
        after = int(_readiness(raw).get("damage", 0))
        record = {"identity": list(pending["identity"]), "price": pending["price"],
                  "item_type": pending["item_type"],
                  "weapon_damage_before": pending["damage_before"],
                  "weapon_damage_after": after,
                  "projected_damage_after": pending["projected_damage_after"],
                  "projection_exact": after == pending["projected_damage_after"],
                  "beat": int(env._steps)}
        self.purchases.append(record)
        self.weapon_upgrades += 1
        if self.attempts:
            self.attempts[-1].update(weapon_damage_after=after,
                                     identity=list(pending["identity"]),
                                     price=pending["price"])
        self._pending = None
        return self._end(env, "weapon_equipped", failed=False)

    # ---- receipts ----
    def receipt(self, command, receipt):
        kind = command[0]
        if kind in ("walk", "open"):
            if receipt.get("accepted"):
                self._walk_rejections = 0
            else:
                self._walk_rejections += 1
                if self._walk_rejections >= WEAPON_UPGRADE_WALK_REJECTIONS:
                    self._give_up = "weapon_walk_rejected"
        elif kind == "buy":
            if receipt.get("accepted"):
                paid = int(receipt.get("price", 0) or 0)
                self.weapon_gold_spent += paid
                if self.attempts:
                    self.attempts[-1]["gold_spent"] += paid
                self.phase = "equip"
            else:
                self._shop_failures.add(tuple(command[1:]))
                self._pending = None
                self._give_up = "weapon_purchase_rejected"
        elif kind == "equip" and not receipt.get("accepted"):
            self._equip_failures.add(tuple(command[1:]))

    def _end(self, env, reason, *, failed):
        self.active = False
        self.phase = "done"
        self.reason = reason
        if self.attempts:
            record = self.attempts[-1]
            record["outcome"] = "failed" if failed else "completed"
            record["reason"] = reason
            record["beat1"] = int(env._steps)
        return ("complete",)

    def telemetry(self):
        fired = [s for s in self.seams if s["would_fire"]]
        return {"protocol": self.protocol, "attempted": self.attempted,
                "seams": deepcopy(self.seams),
                "seams_observed": len(self.seams),
                "seams_would_fire": len(fired),
                "would_gold_spent": sum(s["would_price"] for s in fired),
                "would_damage_gain": [s["would_damage_gain"] for s in fired],
                "active": self.active, "phase": self.phase, "reason": self.reason,
                "trigger": self.trigger, "steps": self.steps,
                "trips_served": self.trips_served,
                "weapon_upgrades": self.weapon_upgrades,
                "weapon_gold_spent": self.weapon_gold_spent,
                "weapon_damage_before": [p["weapon_damage_before"] for p in self.purchases],
                "weapon_damage_after": [p["weapon_damage_after"] for p in self.purchases],
                # R18-K2b review round (2026-09-07): fall back to the plan the
                # trigger law built, so an episode whose leg never fired still
                # reports the reserve that was evaluated at its last seam.
                "reserve": (self._plan or self._seam_plan
                            or {}).get("reserve", WEAPON_UPGRADE_RESERVE),
                "base_reserve": WEAPON_UPGRADE_RESERVE,
                "reserve_detail": deepcopy((self._plan or self._seam_plan
                                            or {}).get("reserve_detail")),
                "reserve_from": ("command" if self._plan
                                 else ("seam" if self._seam_plan else None)),
                "purchases": deepcopy(self.purchases),
                "attempts": [dict(a) for a in self.attempts],
                "last_plan": deepcopy(self._plan),
                "phase_steps": dict(self.phase_steps)}

    # ---- shop navigation (the ordinary town-service pattern) ----
    def _visit(self, env, vendor, town):
        if town.get("active_vendor") == vendor and not town.get("dialog_active"):
            return None
        if town.get("dialog_active") or town.get("active_vendor") not in (None, "none", vendor):
            return ("dismiss",)
        npc = next((n for n in town.get("npcs", []) if n.get("type") == vendor), None)
        if npc is None:
            self._give_up = "weapon_npc_missing"
            return ("wait",)
        raw = env._raw
        px, py = int(raw["player_x"]), int(raw["player_y"])
        if max(abs(int(npc["x"]) - px), abs(int(npc["y"]) - py)) <= 1:
            self._talk_attempts += 1
            if self._talk_attempts > WEAPON_UPGRADE_TALK_ATTEMPTS:
                self._give_up = "weapon_vendor_unavailable"
                return ("wait",)
            return ("talk", int(npc["id"]))
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            command = self._walk(env, int(npc["x"]) + dx, int(npc["y"]) + dy)
            if command is not None:
                return command
        self._give_up = "weapon_npc_unreachable"
        return ("wait",)

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
