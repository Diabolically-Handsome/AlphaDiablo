"""Opt-in bounded, observed equipment combinations with real service actions.

Every sequence is projected by the native engine, including purchase capacity,
retained-item repairs and attribute cascades. This is a limited candidate
catalog, not a proof that unlisted equipment combinations are impossible.
The legacy completion service and its factory remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations, permutations, zip_longest
import json

from .resource_protocol import ResourceService, native_readiness
from .resource_sustain import GATED_SLOTS, item_identity, medicine_candidates
from .resource_sustain_armor import plan_ordinary_armor_basket
from .resource_sustain_completion import SustainCompletionService

COMBINATION_POLICY = "observed-equipment-combinations-v1"
BATCH_LIMIT = 256
STOCK_LIMIT = 16
OWNED_LIMIT = 12
REPAIR_RETAINED = ("repair_retained",)


def equipment_sequences(raw, smith_stock):
    """Enumerate ordinary items and actual removable armor, without fake gates.

    Individual can_equip/meets_armor_gate/target_slot fields cannot reject a
    combination: an earlier normal mutation may change capacity or attributes.
    All source identities were actually observed; native remains authoritative.
    """
    stock = sorted((i for i in smith_stock if i.get("vendor") == "smith"
                    and i.get("is_ordinary_armor")),
                   key=lambda i: (int(i["price"]), item_identity(i)))
    owned = sorted((i for i in raw["resource_state"].get("inventory_items", [])
                    if i.get("is_ordinary_armor")), key=item_identity)
    gear = [("buy_equip", *item_identity(i)) for i in stock[:STOCK_LIMIT]]
    gear += [("equip", int(i["index"]), *item_identity(i)) for i in owned[:OWNED_LIMIT]]
    removes = []
    for item in raw["resource_state"].get("unequip_candidates", []):
        slot = int(item["slot"])
        if slot not in GATED_SLOTS or slot >= len(raw["equipped_items"]):
            continue
        actual = raw["equipped_items"][slot]
        if actual.get("present") and item_identity(actual) == item_identity(item):
            removes.append(("unequip", slot, *item_identity(item)))
    removes = sorted(set(removes))
    categories = [[()], [(i,) for i in gear], [], [], []]
    for count in (1, 2):
        for group in combinations(removes, count):
            categories[0].extend(permutations(group))
    for item in gear:
        for remove in removes:
            categories[2].extend(((item, remove), (remove, item)))
        for group in combinations(removes, 2):
            categories[4].extend(permutations((item, *group)))
    for pair in combinations(gear, 2):
        if tuple(pair[0][-4:]) != tuple(pair[1][-4:]):
            categories[3].extend(permutations(pair))
    # Round-robin categories prevent a large class of proposals monopolizing
    # the bounded batch. Within each class, quoted price/identity is stable.
    ordered, seen = [], set()
    for row in zip_longest(*categories):
        for sequence in row:
            if sequence is None:
                continue
            sequence = tuple(sequence) + (REPAIR_RETAINED,)
            if sequence not in seen:
                seen.add(sequence)
                ordered.append(sequence)
    return ordered[:BATCH_LIMIT], {
        "batch_limit": BATCH_LIMIT, "generated_sequences": len(ordered),
        "submitted_sequences": min(BATCH_LIMIT, len(ordered)),
        "omitted_stock_items": max(0, len(stock) - STOCK_LIMIT),
        "omitted_owned_items": max(0, len(owned) - OWNED_LIMIT),
        "truncated": len(ordered) > BATCH_LIMIT or len(stock) > STOCK_LIMIT or len(owned) > OWNED_LIMIT,
        "scope": "up_to_two_items_or_one_item_two_removals_then_retained_repairs",
    }


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_preview(raw, sequences, budget, response):
    if (not isinstance(response, dict) or response.get("version") != COMBINATION_POLICY
            or response.get("max_gold_cost") != budget
            or response.get("batch_limit") != BATCH_LIMIT or response.get("step_limit") != 8
            or response.get("truncated") is not False):
        raise RuntimeError("Invalid native equipment combination envelope")
    source = response.get("source")
    state = raw["resource_state"]
    if (not isinstance(source, dict) or source.get("gold") != int(raw["gold"])
            or any(source.get(k) != state[k] for k in ("town_seed", "town_restock_sequence"))):
        raise RuntimeError("Native combination source wallet/town is stale")
    current = native_readiness(raw)
    if any(source.get("readiness", {}).get(k) != v for k, v in current.items()):
        raise RuntimeError("Native combination source readiness is stale")
    equipment = source.get("equipment")
    if not isinstance(equipment, list) or len(equipment) != len(raw["equipped_items"]):
        raise RuntimeError("Native combination source equipment is incomplete")
    for slot, (expected, actual) in enumerate(zip(raw["equipped_items"], equipment)):
        if actual.get("index") != slot or actual.get("empty") is not (not expected.get("present")):
            raise RuntimeError("Native combination source equipment is stale")
        if expected.get("present") and (tuple(actual.get("identity", ())) != item_identity(expected)
                or actual.get("durability") != expected["durability"]
                or actual.get("max_durability") != expected["max_durability"]):
            raise RuntimeError("Native combination source equipment identity/durability changed")
    results = response.get("results")
    if not isinstance(results, list) or len(results) != len(sequences):
        raise RuntimeError("Native combination result count changed")
    for index, result in enumerate(results):
        if result.get("index") != index or type(result.get("valid")) is not bool:
            raise RuntimeError("Native combination ordering/verdict is invalid")
        if not result["valid"]:
            if result.get("first_command") is not None or result.get("projected_readiness") is not None:
                raise RuntimeError("Rejected native combination exposed an executable projection")
            continue
        commands = result.get("expanded_commands")
        projection = result.get("projected_readiness")
        cost = result.get("gold_cost")
        if (not isinstance(commands, list) or type(cost) is not int or not 0 <= cost <= budget
                or result.get("gold_remaining") != int(raw["gold"]) - cost
                or not isinstance(projection, dict) or type(projection.get("ready")) is not bool
                or not isinstance(projection.get("failures"), (list, tuple))
                or result.get("failed_step") is not None
                or _json(result.get("first_command")) != _json(commands[0] if commands else None)):
            raise RuntimeError("Invalid complete native combination projection")
        if any(not c or c[0] not in ("buy", "equip", "unequip", "repair") for c in commands):
            raise RuntimeError("Native combination expanded an unsupported real command")
    return sha256(_json(source).encode()).hexdigest()


def plan_combination_basket(raw, smith_stock, repair_quotes, healer_stock,
                            sequences, preview, failed=()):
    """Compare full native combinations with the unchanged single-item fallback."""
    plan = plan_ordinary_armor_basket(raw, smith_stock, repair_quotes, healer_stock, failed)
    medicine_cost = plan["medicine_cost"]
    budget = max(0, int(raw["gold"]) - (medicine_cost or 0))
    source_hash = _validate_preview(raw, sequences, budget, preview)
    rejected = []
    failed = {tuple(c) for c in failed}
    for sequence, result in zip(sequences, preview["results"]):
        if not result["valid"]:
            rejected.append({"index": result["index"], "reason": result.get("reason"),
                             "failed_step": result.get("failed_step")})
            continue
        projected = result["projected_readiness"]
        if set(projected["failures"]) - {"potions"}:
            rejected.append({"index": result["index"], "reason": "native_final_readiness_unresolved"})
            continue
        commands = [tuple(c) for c in result["expanded_commands"]]
        if any(c in failed for c in commands):
            rejected.append({"index": result["index"], "reason": "real_command_previously_rejected"})
            continue
        # Equipment transactions do not add medicine. The native projection
        # must still have actual belt space for the known minimum reserve.
        deficit = max(0, int(projected["required_belt_heals"]) - int(projected["belt_heals"]))
        if deficit != plan["medicine_deficit"]:
            raise RuntimeError("Equipment combination unexpectedly changed the medicine deficit")
        total = None if medicine_cost is None else result["gold_cost"] + medicine_cost
        block = projected.get("block_enabled")
        plan["alternatives"].append({
            "kind": "equipment_combination", "gear_cost": result["gold_cost"],
            "commands": commands, "sequence": tuple(tuple(i) for i in sequence),
            "total_cost": total,
            "affordable": total is not None and total <= plan["cash"]
                and deficit <= int(projected["belt_free_slots"]),
            "projected_readiness": deepcopy(projected),
            "projected_block_enabled": block if type(block) is bool else None,
            "native_source_sha256": source_hash, "native_result_index": result["index"],
        })
    plan["alternatives"].sort(key=lambda c: (
        c["total_cost"] is None, c["total_cost"] if c["total_cost"] is not None else c["gear_cost"],
        len(c["commands"]), c["kind"], _json(c.get("sequence", c.get("identity", ())))))
    affordable = [c for c in plan["alternatives"] if c["affordable"]]
    effective = [c for c in affordable if c["projected_block_enabled"] is True]
    plan["chosen"] = next(iter(effective or affordable), None)
    totals = [c["total_cost"] for c in plan["alternatives"] if c["total_cost"] is not None]
    minimum = min(totals) if totals else None
    plan.update(minimum_observed_total=minimum,
                shortfall=None if minimum is None else max(0, minimum-plan["cash"]),
                scope="bounded_observed_equipment_combinations_with_single_item_fallback",
                combination_preview={"source_sha256": source_hash, "results": len(sequences),
                                     "rejected": rejected},
                blocking_preference={"policy": "effective_blocking_if_complete_affordable_then_lowest_total",
                    "is_native_readiness_gate": False, "affordable_effective_alternatives": len(effective),
                    "selected_block_enabled": None if plan["chosen"] is None else plan["chosen"]["projected_block_enabled"]})
    plan["unresolved"] = [name for name in plan["unresolved"] if name not in ("gear_solution", "cash")]
    if not plan["alternatives"]:
        plan["unresolved"].append("gear_solution")
    if minimum is not None and minimum > plan["cash"]:
        plan["unresolved"].append("cash")
    return plan


class _RemainderInvalid(RuntimeError):
    pass


@dataclass
class SustainCombinationService(SustainCompletionService):
    """Execute one checked action at a time; acquired assets are never forgotten."""
    reserve_belt_target: int = field(default=0, kw_only=True)
    _combination_bridge: object = field(default=None, init=False, repr=False)
    _remaining: list | None = field(default=None, init=False)
    _invested: bool = field(default=False, init=False)
    _combination_queries: int = field(default=0, init=False)
    _combination_history: list = field(default_factory=list, init=False)
    _catalog: dict | None = field(default=None, init=False)
    _reserve_purchases: int = field(default=0, init=False)
    _reserve_last_command: tuple | None = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()
        if type(self.reserve_belt_target) is not int or self.reserve_belt_target not in (0, 6, 8):
            raise ValueError("reserve_belt_target must be 0 (disabled), 6, or 8")

    def _preview(self, raw, sequences, medicine_cost):
        budget = max(0, int(raw["gold"]) - (medicine_cost or 0))
        project = getattr(self._combination_bridge, "preview_resource_equipment_combinations", None)
        if project is None:
            raise RuntimeError("Combination service requires native equipment preview support")
        response = project(sequences, budget)
        _validate_preview(raw, sequences, budget, response)
        self._combination_queries += 1
        return response

    def _normalize_remaining(self, raw):
        equipped = {item_identity(i) for i in raw["equipped_items"] if i.get("present")}
        inventory = {item_identity(i): int(i["index"])
                     for i in raw["resource_state"].get("inventory_items", [])}
        normalized = []
        for intent in self._remaining or []:
            if intent[0] == "equip":
                identity = tuple(intent[-4:])
                if identity in equipped:
                    continue  # Real normal StoreAutoPlace may already equip.
                if identity not in inventory:
                    raise _RemainderInvalid("committed_owned_item_missing")
                intent = ("equip", inventory[identity], *identity)
            normalized.append(tuple(intent))
        self._remaining = normalized
        return tuple(normalized)

    def _committed_plan(self, raw, smith_stock, repair_quotes, healer_stock, failed):
        sequence = self._normalize_remaining(raw)
        baseline = plan_ordinary_armor_basket(raw, smith_stock, repair_quotes, healer_stock, failed)
        response = self._preview(raw, [sequence], baseline["medicine_cost"])
        plan = plan_combination_basket(raw, smith_stock, repair_quotes, healer_stock,
                                      [sequence], response, failed)
        candidate = next((c for c in plan["alternatives"] if c["kind"] == "equipment_combination"
                          and c["affordable"]), None)
        if candidate is None:
            raise _RemainderInvalid("committed_native_projection_no_longer_complete")
        plan["chosen"] = candidate
        plan["committed_remainder"] = True
        if not candidate["commands"]:
            self._remaining = None
            self._invested = False
        return plan

    def _plan_basket(self, raw, smith_stock, repair_quotes, healer_stock, failed=()):
        refreshed = self._refresh_smith_projections(raw, smith_stock)
        if self._remaining is not None:
            try:
                return self._committed_plan(raw, refreshed, repair_quotes, healer_stock, failed)
            except _RemainderInvalid:
                if self._invested:
                    raise
                self._remaining = None
        baseline = plan_ordinary_armor_basket(raw, refreshed, repair_quotes, healer_stock, failed)
        sequences, self._catalog = equipment_sequences(raw, refreshed)
        response = self._preview(raw, sequences, baseline["medicine_cost"])
        plan = plan_combination_basket(raw, refreshed, repair_quotes, healer_stock,
                                      sequences, response, failed)
        plan = self._select_initial_plan(raw, refreshed, repair_quotes, healer_stock, failed, plan)
        chosen = plan["chosen"]
        if chosen and chosen["kind"] == "equipment_combination" and chosen["commands"]:
            self._remaining = list(chosen["sequence"])
            self._invested = False
            self._combination_history.append({"event": "selected", "sequence": deepcopy(self._remaining),
                "source_sha256": chosen["native_source_sha256"], "total_cost": chosen["total_cost"]})
        return plan

    def _select_initial_plan(self, raw, smith_stock, repair_quotes, healer_stock, failed, plan):
        """Optional initial selection hook; committed remainders never use it."""
        return plan

    def _reserve_command(self, env):
        raw = env._raw
        ready = native_readiness(raw)
        if (not self.reserve_belt_target or not ready["ready"]
                or int(ready["belt_heals"]) >= self.reserve_belt_target
                or int(ready["belt_free_slots"]) <= 0):
            return None
        # Do not make an optional detour on an unknown or unaffordable quote.
        if not any(int(i["price"]) <= int(raw["gold"])
                   and ("buy", *self._stock_key(i)) not in self._failed_commands
                   for i in medicine_candidates(self._healer_stock)):
            return None
        town = raw["resource_state"]["town"]
        visit = self._visit(env, "healer", town)
        if visit:
            return visit
        self._healer_stock = deepcopy(town.get("stock", []))
        candidates = [i for i in medicine_candidates(self._healer_stock)
                      if int(i["price"]) <= int(raw["gold"])
                      and ("buy", *self._stock_key(i)) not in self._failed_commands]
        if not candidates:
            return None
        self._reserve_last_command = self._buy(candidates[0])
        return self._reserve_last_command

    def _command(self, env, bridge):
        self._combination_bridge = bridge
        try:
            raw = env._raw
            town_phase = self.phase in ("joint_gear", "minimum_potions")
            if town_phase and (raw.get("is_set_level") or int(raw["dungeon_level"]) != 0
                               or self.steps >= self.service_microstep_cap):
                return super()._command(env, bridge)  # Inherited normal finish.
            if self.phase == "joint_gear":
                town = raw["resource_state"]["town"]
                if town.get("dialog_active"):
                    return ("dismiss",)
                if self._remaining is not None and self._armor_pending is not None:
                    # Parent's pending equip precedes its planning hook. Recheck
                    # the *whole remainder* before allowing that normal action.
                    checked = self._committed_plan(raw, self._smith_stock, self._repair_quotes,
                                                   self._healer_stock, self._failed_commands)
                    self._last_plan = checked
                    pending = tuple(self._armor_pending)
                    inventory = next((i for i in raw["resource_state"].get("inventory_items", [])
                                      if item_identity(i) == pending), None)
                    if inventory is not None:
                        expected = ("equip", int(inventory["index"]), *pending)
                        if checked["chosen"]["commands"][:1] != [expected]:
                            raise _RemainderInvalid("pending_equip_is_not_checked_first_action")
            if self.phase == "minimum_potions":
                extra = self._reserve_command(env)
                if extra:
                    return extra
            command = super()._command(env, bridge)
            # The original core completes minimum medicine and starts returning
            # in one call. Intercept only its unsubmitted dismiss command; no
            # movement/path query has happened and every future beat still uses
            # the same native settlement and the total 3000 service budget.
            if (town_phase and self.phase == "return" and command == ("dismiss",)
                    and self.reserve_belt_target):
                extra = self._reserve_command(env)
                if extra:
                    self.phase = "minimum_potions"
                    return extra
            return command
        except _RemainderInvalid as error:
            self._combination_history.append({"event": "stopped", "reason": str(error),
                                              "after_real_mutation": self._invested})
            return self._finish("resource_combination_remainder_invalid")
        finally:
            self._combination_bridge = None

    def receipt(self, command, receipt):
        super().receipt(command, receipt)
        command = tuple(command)
        if command == self._reserve_last_command:
            self._reserve_purchases += int(receipt.get("accepted") is True)
            self._reserve_last_command = None
        if self._remaining is None or command[0] not in ("buy", "equip", "unequip", "repair"):
            return
        if not receipt.get("accepted"):
            # Leave the queue in place: next native plan sees this exact failed
            # command and stops after investment instead of cycling purchases.
            return
        self._invested = True
        first = self._remaining[0] if self._remaining else None
        if command[0] == "buy" and command[1] == "smith":
            if not first or first[0] != "buy_equip" or tuple(first[1:]) != command[3:]:
                raise RuntimeError("Accepted purchase does not match committed combination")
            self._remaining[0] = ("equip", -1, *command[3:])
        elif command[0] in ("equip", "unequip"):
            if not first or first[0] != command[0] or tuple(first[-4:]) != command[-4:]:
                raise RuntimeError("Accepted equipment mutation does not match committed combination")
            self._remaining.pop(0)
        elif command[0] == "repair":
            if first != REPAIR_RETAINED:
                raise RuntimeError("Accepted repair does not match committed combination")
        self._combination_history.append({"event": "accepted", "command": command,
            "receipt": deepcopy(receipt), "remaining": deepcopy(self._remaining)})

    def telemetry(self):
        info = super().telemetry()
        info.update(policy=COMBINATION_POLICY, reserve_belt_target=self.reserve_belt_target,
                    reserve_purchases=self._reserve_purchases,
                    combination_queries=self._combination_queries,
                    combination_catalog=deepcopy(self._catalog),
                    combination_remaining=deepcopy(self._remaining),
                    combination_history=deepcopy(self._combination_history))
        return info
