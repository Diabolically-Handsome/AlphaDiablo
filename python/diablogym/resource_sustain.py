"""Explicit R20 service candidate; legacy-v1 retains its historical recipe.

Planning uses observed shop quotes and native item projections. It grants no
descent permit: the engine re-evaluates the real character at the transition.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .resource_navigation import ResourceNavigationRecovery
from .resource_protocol import ResourceService, native_readiness

RESOURCE_SERVICE_POLICIES = ("legacy-v1", "sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1")
SUSTAIN_SERVICE_CAP = 1500
IDENTITY_FIELDS = ("seed_hi", "seed_lo", "create_info", "base_id")
GATED_SLOTS = (0, 4, 5, 6)


def validate_service_policy(protocol, mode, policy):
    if policy not in RESOURCE_SERVICE_POLICIES:
        raise ValueError(f"Unknown resource_service_policy: {policy!r}")
    if policy in ("sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1") and (protocol != "l2-town-v1" or mode != "full"):
        raise ValueError(f"{policy} requires l2-town-v1 and full purchase mode")
    return policy


def item_identity(item):
    return tuple(int(item[k]) for k in IDENTITY_FIELDS)


def medicine_candidates(stock):
    # Instant kinds only. Scrolls and non-healing stock cannot fill this gate.
    return sorted((item for item in stock
        if item.get("vendor") == "healer" and int(item.get("heal_kind", 0)) in (1, 2, 3, 4)
        and item.get("can_use") and item.get("can_fit")),
        key=lambda item: (int(item["price"]), int(item["index"])))


def plan_basket(raw, smith_stock, repair_quotes, healer_stock, failed=()):
    """Compare retain/one armor change plus repairs and the minimum belt stock.

    These are executable *next* actions, not an exhaustive inventory optimizer.
    After each real mutation the caller observes and plans again. In particular
    a native unequip preview must never be reused as the final readiness gate.
    """
    ready = native_readiness(raw)
    state = raw["resource_state"]
    cash = int(raw.get("gold", 0))
    deficit = max(0, int(ready["required_belt_heals"]) - int(ready["belt_heals"]))
    medicines = medicine_candidates(healer_stock)
    med_cost = deficit * int(medicines[0]["price"]) if medicines else (0 if not deficit else None)
    belt_room = deficit <= int(ready["belt_free_slots"])
    equipment = {i: item for i, item in enumerate(raw["equipped_items"])
                 if i in GATED_SLOTS and item.get("present")}
    failed = set(failed)
    alternatives = []
    zero_improvements = []

    def repairs_without(removed):
        commands, cost = [], 0
        for slot, item in equipment.items():
            if slot == removed or int(item["max_durability"]) == 255 or int(item["durability"]) >= 15:
                continue
            if int(item["max_durability"]) < 15:
                return None
            quote = next((q for q in repair_quotes if int(q["slot"]) == slot
                          and item_identity(q) == item_identity(item)
                          and int(q["durability"]) == int(item["durability"])
                          and q.get("repair_needed_for_gate")), None)
            if quote is None:
                return None
            command = ("repair", slot, *item_identity(quote), int(quote["durability"]), int(quote["price"]))
            if command in failed:
                return None
            commands.append(command)
            cost += int(quote["price"])
        commands.sort(key=lambda c: (c[-1], c[1]))
        return commands, cost

    def add(name, first, price, removed=None):
        repairs = repairs_without(removed)
        if repairs is None or first in failed:
            return
        commands, cost = repairs
        actions = ([first] if first else []) + commands
        alternatives.append({"kind": name, "gear_cost": price + cost,
                             "commands": actions})

    if not set(ready["failures"]) & {"armor", "damage", "weapon"}:
        add("retain_and_repair", None, 0)
    for item in smith_stock:
        if (item.get("vendor") == "smith" and item.get("can_use")
                and item.get("can_fit") and item.get("meets_armor_gate")):
            add("buy_chest", ("buy", *ResourceService._stock_key(item)), int(item["price"]), 6)
    for item in state.get("inventory_items", []):
        if item.get("can_use") and item.get("meets_armor_gate"):
            command = ("equip", int(item["index"]), *item_identity(item))
            add("equip_owned_chest", command, 0, 6)
            if command not in failed and ("armor" in ready["failures"] or "durability" in ready["failures"]):
                zero_improvements.append(command)
    for item in state.get("unequip_candidates", []):
        slot = int(item["slot"])
        projected = item.get("projected_readiness", {})
        # Only removing an actual failing finite piece is a useful free action.
        # Native projections handle AC, damage and weapon/stat cascades.
        if (not item.get("can_unequip") or slot not in equipment
                or int(item["max_durability"]) == 255 or int(item["durability"]) >= 15
                or set(projected.get("failures", ("armor",))) & {"armor", "damage", "weapon", "health"}):
            continue
        command = ("unequip", slot, *item_identity(item))
        add("unequip_failing_armor", command, 0, slot)
        if command not in failed:
            zero_improvements.append(command)
    for alternative in alternatives:
        total = None if med_cost is None else alternative["gear_cost"] + med_cost
        alternative.update(total_cost=total, affordable=total is not None and total <= cash and belt_room)
    alternatives.sort(key=lambda a: (a["gear_cost"], len(a["commands"]), a["kind"]))
    chosen = next((a for a in alternatives if a["affordable"]), None)
    affordable_total = [a["total_cost"] for a in alternatives if a["total_cost"] is not None]
    lower = min(affordable_total) if affordable_total else None
    return {"cash": cash, "medicine_deficit": deficit, "medicine_cost": med_cost,
            "belt_has_room": belt_room, "minimum_observed_total": lower,
            "shortfall": None if lower is None else max(0, lower - cash),
            "alternatives": alternatives, "chosen": chosen,
            "zero_improvements": zero_improvements,
            "unresolved": [name for name, missing in (
                ("level", "level" in ready["failures"]),
                ("medicine_stock", med_cost is None), ("belt_capacity", not belt_room),
                ("gear_solution", not alternatives), ("cash", lower is not None and lower > cash)) if missing]}


@dataclass
class SustainResourceService(ResourceService):
    """One real trip with joint quotes, minimum medicine reserve and recovery."""
    navigation: ResourceNavigationRecovery = field(default_factory=ResourceNavigationRecovery)
    _smith_stock: list = field(default_factory=list)
    _repair_quotes: list = field(default_factory=list)
    _healer_stock: list = field(default_factory=list)
    _failed_commands: set = field(default_factory=set)
    _last_plan: dict | None = None
    _town_sequence: tuple | None = None
    _gear_mutations: int = 0
    unequips: int = 0

    def __post_init__(self):
        super().__post_init__()
        validate_service_policy("l2-town-v1", self.mode, "sustain-v2")

    @property
    def service_microstep_cap(self):
        return SUSTAIN_SERVICE_CAP if self.calibration is None else self.calibration.service_microstep_cap

    def telemetry(self):
        info = super().telemetry()
        info.update(policy="sustain-v2", service_microstep_cap=self.service_microstep_cap,
                    unequips=self.unequips, basket=deepcopy(self._last_plan),
                    navigation=self.navigation.telemetry())
        return info

    def _plan_basket(self, raw, smith_stock, repair_quotes, healer_stock, failed=()):
        return plan_basket(raw, smith_stock, repair_quotes, healer_stock, failed)

    def _command(self, env, bridge):
        raw = env._raw
        depth = int(raw["dungeon_level"])
        # Reuse the proven real gold, scene, cap, return and reward settlement.
        # Intercept the town entry before the legacy armor-first branch.
        if self.phase == "outbound" and depth == 0:
            self.phase = "survey_smith"
        if self.phase not in ("survey_smith", "survey_healer", "joint_gear", "minimum_potions"):
            return super()._command(env, bridge)
        if raw.get("is_set_level") or depth != 0:
            return self._finish("resource_unexpected_scene")
        if self.steps >= self.service_microstep_cap:
            return self._finish("resource_service_cap")
        state = raw["resource_state"]
        town = state["town"]
        sequence = (int(state["town_seed"]), int(state["town_restock_sequence"]))
        if self._town_sequence is None:
            self._town_sequence = sequence
        elif self._town_sequence != sequence:
            raise RuntimeError("Town stock generation changed inside one sustain trip")
        if self.phase == "survey_smith":
            command = self._visit(env, "smith", town)
            if command:
                return command
            self._smith_stock = deepcopy(town.get("stock", []))
            self._repair_quotes = deepcopy(town.get("repair_quotes", []))
            self.phase = "survey_healer"
        if self.phase == "survey_healer":
            command = self._visit(env, "healer", town)
            if command:
                return command
            if int(raw["hp"]) < int(raw["max_hp"]):
                self._talk_attempts += 1
                if self._talk_attempts > 8:
                    return self._finish("resource_healer_unavailable")
                return self._talk(env, "healer", town)
            self._town_healed = True
            self._healer_stock = deepcopy(town.get("stock", []))
            self.phase = "joint_gear"
        if self.phase == "joint_gear":
            if self._armor_pending is not None:
                pending = self._armor_pending
                self._armor_pending = None
                for item in state.get("inventory_items", []):
                    if item_identity(item) == pending:
                        return ("equip", int(item["index"]), *pending)
                if not any(i.get("present") and item_identity(i) == pending for i in raw["equipped_items"]):
                    raise RuntimeError("Accepted armor purchase absent from real equipment and inventory")
            if self._gear_mutations >= 12:
                return self._finish("resource_gear_no_convergence")
            # Current native proposals and repair costs are refreshed on-site.
            # The two surveys supply the first decision without a blind detour.
            if town.get("active_vendor") == "smith" and not town.get("dialog_active"):
                self._smith_stock = deepcopy(town.get("stock", []))
                self._repair_quotes = deepcopy(town.get("repair_quotes", []))
            plan = self._plan_basket(raw, self._smith_stock, self._repair_quotes,
                               self._healer_stock, self._failed_commands)
            self._last_plan = plan
            chosen = plan["chosen"]
            commands = chosen["commands"] if chosen else plan["zero_improvements"]
            if commands:
                command = commands[0]
                # Re-enter Smith before all gear decisions (even free ones), so
                # each subsequent plan consumes fresh real quotes at this shop.
                visit = self._visit(env, "smith", town)
                if visit:
                    return visit
                self._gear_mutations += 1
                return command
            self.phase = "minimum_potions"
        if self.phase == "minimum_potions":
            ready = native_readiness(raw)
            if int(ready["belt_heals"]) < int(ready["required_belt_heals"]) and int(ready["belt_free_slots"]) > 0:
                command = self._visit(env, "healer", town)
                if command:
                    return command
                self._healer_stock = deepcopy(town.get("stock", []))
                candidates = [item for item in medicine_candidates(self._healer_stock)
                    if int(item["price"]) <= int(raw.get("gold", 0))
                    and ("buy", *self._stock_key(item)) not in self._failed_commands]
                if candidates:
                    return self._buy(candidates[0])
            self.phase = "return"
            return super()._command(env, bridge)
        raise RuntimeError(f"Unhandled sustain phase: {self.phase}")

    def receipt(self, command, receipt):
        super().receipt(command, receipt)
        if command[0] in ("buy", "repair", "equip", "unequip") and not receipt.get("accepted"):
            self._failed_commands.add(tuple(command))
        if command[0] == "unequip" and receipt.get("accepted"):
            self.unequips += 1

    def _walk(self, env, x, y):
        raw = env._raw
        path = env._plan_descend_path(raw, x, y, avoid_monsters=True)
        decision = self.navigation.next_command(raw, (x, y), now=int(env._steps),
            service_deadline=self.start_steps + self.service_microstep_cap,
            path_available=bool(path))
        if not path:
            if decision.command is not None:
                # The legacy gold-route guard rejects the same walk twice.
                # Repeated real combat/wait ticks instead have the recovery's
                # own absolute and no-progress limits and must be allowed.
                self._gold_navigation_seen.clear()
            return decision.command
        nx, ny, door = path[0]
        if door:
            from . import bridge
            if not bool(bridge.probe_tile(int(nx), int(ny))["walkable"]):
                return ("open", int(nx), int(ny))
        return ("walk", int(nx), int(ny))
