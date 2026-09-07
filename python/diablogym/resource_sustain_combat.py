"""Opt-in combat-aware ranking of already validated native combinations.

No extra native query or game action is performed here. The reference is the
old affordable single-item choice, mapped to its complete native projection
(including actual retained repairs). Only initial selection changes; a paid
combination's remainder retains the parent's live validation and commitment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math

from .resource_sustain_armor import plan_ordinary_armor_basket
from .resource_sustain_combinations import REPAIR_RETAINED, SustainCombinationService

COMBAT_POLICY = "observed-equipment-combat-v1"
_NONREGRESSION_FIELDS = ("damage", "block_chance", "hp_fixed", "max_hp_fixed", "min_finite_durability")


def _stable(candidate):
    return json.dumps(candidate.get("sequence", candidate.get("identity", ())),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cost_order(candidate):
    return (candidate["total_cost"], len(candidate["commands"]), _stable(candidate))


def _projection(candidate):
    projected = candidate.get("projected_readiness", {})
    values, missing = {}, []
    for key in ("armor_class", *_NONREGRESSION_FIELDS):
        value = projected.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (key == "min_finite_durability" and (type(value) is not int or not 0 <= value <= 255))):
            missing.append(key)
        else:
            # 255 is the native sentinel for no finite durability. None/missing
            # is absent evidence, not an indestructibility claim.
            values[key] = math.inf if key == "min_finite_durability" and value == 255 else value
    if type(projected.get("block_enabled")) is not bool:
        missing.append("block_enabled")
    else:
        values["block_enabled"] = projected["block_enabled"]
    return values, missing


def _legacy_sequence(chosen):
    if chosen is None:
        return None
    commands = chosen.get("commands", [])
    first = next((tuple(c) for c in commands if c[0] != "repair"), None)
    if first is None:
        return (REPAIR_RETAINED,)
    if first[0] == "buy" and first[1] == "smith":
        first = ("buy_equip", *first[3:])
    elif first[0] not in ("equip", "unequip"):
        return None
    return (first, REPAIR_RETAINED)


def _brief(candidate):
    if candidate is None:
        return None
    return {key: deepcopy(candidate.get(key)) for key in (
        "kind", "total_cost", "commands", "sequence", "projected_readiness", "native_source_sha256")}


def select_combat_basket(raw, smith_stock, repair_quotes, healer_stock, failed, plan):
    """Protect a fully quoted reference, then improve AC without regressions.

    Equal AC preserves the reference unless another plan strictly improves a
    recorded combat field (or effective blocking), costs no more, and regresses
    none. Saving gold alone is not a combat improvement. Missing evidence never
    establishes an improvement; the already selected reference is retained.
    """
    result = deepcopy(plan)
    audit = {"version": COMBAT_POLICY, "is_native_readiness_gate": False,
             "reference_origin": None, "reference": None, "selected": None,
             "rejected": [], "reason": None}
    result["combat_selection"] = audit
    source = plan.get("combination_preview", {}).get("source_sha256")
    full = [c for c in result["alternatives"]
            if c.get("kind") == "equipment_combination" and c.get("affordable") is True
            and source is not None and c.get("native_source_sha256") == source]
    for candidate in full:
        projected = candidate.get("projected_readiness", {})
        if (not isinstance(projected.get("failures"), (list, tuple))
                or set(projected["failures"]) - {"potions"}):
            raise RuntimeError("Combat selection received an incomplete native combination")
    legacy = plan_ordinary_armor_basket(raw, smith_stock, repair_quotes, healer_stock, failed)
    sequence = _legacy_sequence(legacy["chosen"])
    if legacy["chosen"] is not None:
        matches = [c for c in full if tuple(tuple(i) for i in c["sequence"]) == sequence]
        if not matches:
            # The bounded batch may omit a single-item quote. Preserve the old
            # actual fallback rather than inventing its repaired combat values.
            result["chosen"] = deepcopy(legacy["chosen"])
            audit.update(reference_origin="old_single_item_unmapped", reference=_brief(legacy["chosen"]),
                         selected=_brief(result["chosen"]), reason="reference_full_native_projection_unavailable")
            return result
        reference = min(matches, key=_cost_order)
        audit["reference_origin"] = "old_single_item_complete_native_projection"
    else:
        effective = [c for c in full if c["projected_readiness"].get("block_enabled") is True]
        reference = min(effective or full, key=_cost_order) if full else None
        audit["reference_origin"] = "lowest_cost_complete_native_combination"
    audit["reference"] = _brief(reference)
    if reference is None:
        audit.update(reason="no_complete_affordable_native_combination", selected=_brief(result["chosen"]))
        return result
    result["chosen"] = reference
    ref, missing = _projection(reference)
    if missing:
        audit.update(reason="reference_combat_evidence_incomplete", missing_reference_fields=missing,
                     selected=_brief(reference))
        return result
    eligible = []
    for candidate in full:
        values, missing = _projection(candidate)
        reasons = [f"{key}_evidence_missing_or_invalid" for key in missing]
        if not missing:
            reasons += [f"{key}_decreased" for key in _NONREGRESSION_FIELDS if values[key] < ref[key]]
            if ref["block_enabled"] and not values["block_enabled"]:
                reasons.append("effective_blocking_lost")
            if values["armor_class"] < ref["armor_class"]:
                reasons.append("armor_class_decreased")
            if values["armor_class"] == ref["armor_class"] and candidate is not reference:
                strict = (any(values[k] > ref[k] for k in _NONREGRESSION_FIELDS)
                          or (values["block_enabled"] and not ref["block_enabled"]))
                if not strict:
                    reasons.append("equal_armor_without_strict_combat_improvement")
                if candidate["total_cost"] > reference["total_cost"]:
                    reasons.append("equal_armor_strict_improvement_costs_more")
        if reasons:
            audit["rejected"].append({"sequence": deepcopy(candidate.get("sequence")), "reasons": reasons})
        else:
            eligible.append(candidate)
    effective = [c for c in eligible if c["projected_readiness"]["block_enabled"]]
    pool = effective or eligible
    if pool:
        result["chosen"] = min(pool, key=lambda c: (-c["projected_readiness"]["armor_class"], *_cost_order(c)))
    audit.update(reason="reference_preserved" if result["chosen"] is reference else "strict_native_combat_improvement",
                 pool="nonregressing_effective_blocking" if effective else "nonregressing_fallback",
                 selected=_brief(result["chosen"]))
    return result


@dataclass
class SustainCombatService(SustainCombinationService):
    _combat_selection: dict | None = field(default=None, init=False)
    _combat_selection_count: int = field(default=0, init=False)

    def _select_initial_plan(self, raw, smith_stock, repair_quotes, healer_stock, failed, plan):
        result = select_combat_basket(raw, smith_stock, repair_quotes, healer_stock, failed, plan)
        self._combat_selection = deepcopy(result["combat_selection"])
        self._combat_selection_count += 1
        return result

    def telemetry(self):
        info = super().telemetry()
        info.update(policy=COMBAT_POLICY, combat_selection=deepcopy(self._combat_selection),
                    combat_selection_count=self._combat_selection_count)
        return info
