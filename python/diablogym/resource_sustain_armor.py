"""Dormant v5 ordinary-armor planner using observed native proposals only.

This compares a single normal armor mutation, repairs of retained equipment,
and the minimum quoted medicine reserve. It is a bounded candidate catalog,
not a proof of game-wide affordability. Already observed Smith identities are
reprojected by the engine against the current character before each joint
decision. Every real mutation still causes a live native replan, and
only the engine's transition guard can authorize descent.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .resource_protocol import ResourceService, native_readiness
from .resource_sustain import GATED_SLOTS, item_identity, medicine_candidates
from .resource_sustain_gold import SustainGoldExtendedService


def plan_ordinary_armor_basket(raw, smith_stock, repair_quotes, healer_stock, failed=()):
    """Prefer effective blocking among affordable fully quoted candidates.

    This shopping preference is not a native readiness requirement. If no
    affordable candidate has observed native blocking, fall back to the
    cheapest complete basket and report the missing evidence or constraint.

    Normal can_equip is not an improvement score. Consequently this planner
    has no unconditional free-equip fallback that could alternate two items.
    Prospective repairs of a newly equipped damaged item need a new real
    quote, so such multi-stage candidates are deliberately outside this catalog.
    """
    ready = native_readiness(raw)
    state = raw["resource_state"]
    if state.get("ordinary_armor_scope") is not True:
        raise RuntimeError("sustain-v5 requires native ordinary_armor_scope=True")
    cash = int(raw.get("gold", 0))
    failed = {tuple(command) for command in failed}
    equipment = {slot: item for slot, item in enumerate(raw["equipped_items"])
                 if slot in GATED_SLOTS and item.get("present")}
    deficit = max(0, int(ready["required_belt_heals"]) - int(ready["belt_heals"]))
    medicines = [item for item in medicine_candidates(healer_stock)
                 if ("buy", *ResourceService._stock_key(item)) not in failed]
    med_cost = deficit * int(medicines[0]["price"]) if medicines else (None if deficit else 0)
    belt_room = deficit <= int(ready["belt_free_slots"])
    alternatives, rejected = [], []

    def reject(kind, reason, item=None):
        entry = {"kind": kind, "reason": reason}
        if item is not None:
            entry["identity"] = item_identity(item)
        rejected.append(entry)

    def projection_usable(projection):
        # Only already observed potions and finite-durability deficits are
        # addressed by this basket. Preserve all other native verdicts.
        return (isinstance(projection, dict)
                and isinstance(projection.get("ready"), bool)
                and isinstance(projection.get("failures"), (list, tuple))
                and not set(projection["failures"]) - {"potions", "durability"})

    def repairs_without(removed):
        commands, cost = [], 0
        for slot, item in equipment.items():
            if slot in removed or int(item["max_durability"]) == 255 or int(item["durability"]) >= 15:
                continue
            if int(item["max_durability"]) < 15:
                return None, "retained_item_max_durability"
            quote = next((q for q in repair_quotes if int(q["slot"]) == slot
                          and item_identity(q) == item_identity(item)
                          and int(q["durability"]) == int(item["durability"])
                          and q.get("repair_needed_for_gate")), None)
            if quote is None:
                return None, "missing_current_repair_quote"
            command = ("repair", slot, *item_identity(quote), int(quote["durability"]), int(quote["price"]))
            if command in failed:
                return None, "repair_previously_rejected"
            commands.append(command)
            cost += int(quote["price"])
        commands.sort(key=lambda command: (command[-1], command[1]))
        return (commands, cost), None

    def add(kind, first, price, removed, projected, item=None, target=None):
        if first in failed:
            reject(kind, "command_previously_rejected", item)
            return
        if not projection_usable(projected):
            reject(kind, "native_projection_unresolved_or_missing", item)
            return
        repairs, reason = repairs_without(removed)
        if repairs is None:
            reject(kind, reason, item)
            return
        commands, repair_cost = repairs
        gear_cost = price + repair_cost
        total = None if med_cost is None else gear_cost + med_cost
        alternative = {"kind": kind, "gear_cost": gear_cost,
                       "commands": ([first] if first else []) + commands,
                       "total_cost": total,
                       "affordable": total is not None and total <= cash and belt_room,
                       "target_slot": target, "replaced_slots": sorted(removed),
                       "projected_readiness": deepcopy(projected)}
        if item is not None:
            alternative["identity"] = item_identity(item)
        alternatives.append(alternative)

    retained_projection = deepcopy(ready)
    for key in ("block_enabled", "block_chance"):
        # This reports current native blocking while keeping the equipment.
        # It does not invent a repair-after state for an ineffective shield.
        retained_projection.pop(key, None)
        if key in raw:
            retained_projection[key] = raw[key]
    add("retain_and_repair", None, 0, set(), retained_projection)

    def armor_candidate(item, buying):
        kind = "buy_ordinary_armor" if buying else "equip_owned_ordinary_armor"
        if buying and item.get("vendor") != "smith":
            return
        if not item.get("is_ordinary_armor"):
            return
        if not item.get("can_use") or not item.get("can_equip") or (buying and not item.get("can_fit")):
            reject(kind, "native_use_or_capacity_rejected", item)
            return
        # This also excludes an owned worn item without a prospective repair
        # quote. A normal item's purchase price never stands in for repair cost.
        if not item.get("meets_armor_gate"):
            reject(kind, "native_armor_gate_rejected", item)
            return
        target = int(item["target_slot"])
        removed_list = [int(slot) for slot in item["replaced_slots"]]
        removed = set(removed_list)
        if (target not in GATED_SLOTS or len(removed) != len(removed_list)
                or not removed.issubset(equipment)
                or (target in equipment and target not in removed)):
            reject(kind, "native_replacement_slots_stale_or_invalid", item)
            return
        if buying:
            first, price = ("buy", *ResourceService._stock_key(item)), int(item["price"])
        else:
            first, price = ("equip", int(item["index"]), *item_identity(item)), 0
        add(kind, first, price, removed, item.get("projected_readiness"), item, target)

    for item in smith_stock:
        armor_candidate(item, True)
    for item in state.get("inventory_items", []):
        armor_candidate(item, False)
    for item in state.get("unequip_candidates", []):
        slot = int(item["slot"])
        if (not item.get("can_unequip") or slot not in equipment
                or item_identity(item) != item_identity(equipment[slot])
                or int(item["durability"]) != int(equipment[slot]["durability"])
                or int(item["max_durability"]) == 255 or int(item["durability"]) >= 15):
            continue
        add("unequip_failing_armor", ("unequip", slot, *item_identity(item)), 0, {slot},
            item.get("projected_readiness"), item, slot)

    alternatives.sort(key=lambda candidate: (
        candidate["total_cost"] is None,
        candidate["total_cost"] if candidate["total_cost"] is not None else candidate["gear_cost"],
        len(candidate["commands"]), candidate["kind"], candidate.get("identity", ())))
    for candidate in alternatives:
        block = candidate["projected_readiness"].get("block_enabled")
        candidate["projected_block_enabled"] = block if isinstance(block, bool) else None
    effective = [candidate for candidate in alternatives if candidate["projected_block_enabled"] is True]
    effective_totals = [candidate["total_cost"] for candidate in effective if candidate["total_cost"] is not None]
    effective_minimum = min(effective_totals) if effective_totals else None
    affordable_effective = [candidate for candidate in effective if candidate["affordable"]]
    affordable = [candidate for candidate in alternatives if candidate["affordable"]]
    chosen = next(iter(affordable_effective or affordable), None)
    if affordable_effective:
        block_reason = "effective_blocking_preserved_or_restored"
    elif not affordable:
        block_reason = "no_fully_quoted_affordable_plan"
    elif effective:
        block_reason = "effective_blocking_plan_not_affordable"
    elif any(candidate["projected_block_enabled"] is None for candidate in alternatives):
        block_reason = "blocking_projection_unknown"
    else:
        block_reason = "no_observed_effective_blocking_plan"
    totals = [candidate["total_cost"] for candidate in alternatives if candidate["total_cost"] is not None]
    minimum = min(totals) if totals else None
    return {"cash": cash, "medicine_deficit": deficit, "medicine_cost": med_cost,
            "belt_has_room": belt_room, "minimum_observed_total": minimum,
            "shortfall": None if minimum is None else max(0, minimum - cash),
            "alternatives": alternatives, "chosen": chosen, "zero_improvements": [],
            "blocking_preference": {
                "policy": "effective_blocking_if_complete_affordable_then_lowest_total",
                "is_native_readiness_gate": False,
                "selected_block_enabled": None if chosen is None else chosen["projected_block_enabled"],
                "reason": block_reason,
                "minimum_observed_effective_total": effective_minimum,
                "effective_shortfall": None if effective_minimum is None else max(0, effective_minimum - cash),
                "known_effective_alternatives": len(effective),
                "affordable_effective_alternatives": len(affordable_effective),
                "unknown_alternatives": sum(candidate["projected_block_enabled"] is None for candidate in alternatives),
            },
            "rejected_candidates": rejected,
            "scope": "observed_single_ordinary_armor_retained_repairs_minimum_medicine",
            "unresolved": [name for name, missing in (
                ("level", "level" in ready["failures"]),
                ("health", "health" in ready["failures"]),
                ("medicine_stock", med_cost is None), ("belt_capacity", not belt_room),
                ("gear_solution", not alternatives),
                ("cash", minimum is not None and minimum > cash)) if missing]}


@dataclass
class SustainOrdinaryArmorService(SustainGoldExtendedService):
    """V5 broadens the quoted armor catalog; v4 clocks and routes are inherited."""
    _projection_bridge: object | None = field(default=None, init=False, repr=False)
    _projection_refreshes: int = field(default=0, init=False)
    _stale_projection_items: list = field(default_factory=list, init=False)

    def _command(self, env, bridge):
        # Bind the very same bridge supplied to the shared service. This adds
        # no new movement, wait, native observation, or phase transition.
        self._projection_bridge = bridge
        try:
            return super()._command(env, bridge)
        finally:
            self._projection_bridge = None

    def _refresh_smith_projections(self, raw, smith_stock):
        if not smith_stock:
            return []
        if any(item.get("vendor") != "smith" for item in smith_stock):
            raise RuntimeError("Known Smith projection request contains a foreign quote")
        identities = [item_identity(item) for item in smith_stock]
        if len(set(identities)) != len(identities):
            raise RuntimeError("Known Smith projection request contains duplicate identities")
        project = getattr(self._projection_bridge, "project_seen_resource_smith_items", None)
        if project is None:
            raise RuntimeError("sustain-v5 requires native known-Smith projection support")
        # Native rejects unknown identities rather than revealing unseen stock.
        # Errors deliberately propagate as engineering failures, never no-cash.
        responses = project(identities)
        if not isinstance(responses, list) or len(responses) != len(identities):
            raise RuntimeError("Known Smith projection response count is invalid")
        state = raw["resource_state"]
        refreshed, stale = [], []
        for expected, response in zip(identities, responses):
            if not isinstance(response, dict):
                raise RuntimeError("Known Smith projection response is not an object")
            try:
                identity = item_identity(response)
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("Known Smith projection response identity is invalid") from error
            if (identity != expected
                    or response.get("projection_origin") != "known-smith-live-player"
                    or response.get("town_seed") != state["town_seed"]
                    or response.get("town_restock_sequence") != state["town_restock_sequence"]):
                raise RuntimeError("Known Smith projection response identity/origin/town changed")
            if response.get("status") == "stale_item":
                stale.append({"identity": identity, "reason": "native_seen_item_no_longer_in_stock"})
            elif response.get("status") == "projected" and response.get("vendor") == "smith":
                refreshed.append(deepcopy(response))
            else:
                raise RuntimeError("Known Smith projection response status/vendor is invalid")
        # Commit only after the whole response validates. Never retain a stale
        # projection as a fallback, including a pre-healing HP projection.
        self._smith_stock = refreshed
        self._stale_projection_items.extend(stale)
        self._projection_refreshes += 1
        return refreshed

    def _plan_basket(self, raw, smith_stock, repair_quotes, healer_stock, failed=()):
        refreshed = self._refresh_smith_projections(raw, smith_stock)
        return plan_ordinary_armor_basket(raw, refreshed, repair_quotes, healer_stock, failed)

    def telemetry(self):
        info = super().telemetry()
        info["policy"] = "sustain-v5"
        info["smith_projection_refreshes"] = self._projection_refreshes
        info["stale_smith_projection_items"] = deepcopy(self._stale_projection_items)
        return info


@dataclass
class SustainEquipmentReadinessService(SustainOrdinaryArmorService):
    """V6 uses the exact v5 service with native a14 equipment preservation."""

    def telemetry(self):
        info = super().telemetry()
        info["policy"] = "sustain-v6"
        return info
