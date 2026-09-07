"""Explicit, unregistered protection preference over the native v6 catalog.

This is a service-planning diagnostic, not a new readiness gate or learned
economic policy. All quoted medicine, repair, capacity and blocking rules are
owned by the original planner. Only the choice among its affordable candidates
changes; no native armor formula or prospective repair price is reconstructed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite

from .resource_sustain_armor import (
    SustainEquipmentReadinessService,
    plan_ordinary_armor_basket,
)


PROTECTION_POLICY = "diagnostic-protection-v1"


def _native_armor_class(candidate):
    if candidate is None:
        return None
    value = candidate["projected_readiness"].get("armor_class")
    # A missing or malformed native value cannot establish an improvement.
    return value if type(value) is int else None


def _selection(candidate):
    if candidate is None:
        return None
    return {
        "kind": candidate["kind"],
        "identity": candidate.get("identity"),
        "target_slot": candidate["target_slot"],
        "total_cost": candidate["total_cost"],
        "command_count": len(candidate["commands"]),
        "native_armor_class": _native_armor_class(candidate),
        "projected_block_enabled": candidate["projected_block_enabled"],
    }


def _nonregression_reasons(original, candidate):
    """Compare existing native facts; never infer repair-after durability."""
    before = original["projected_readiness"]
    after = candidate["projected_readiness"]
    keys = ["damage", "block_chance"]
    if "max_hp" in before and "max_hp" in after:
        keys.append("max_hp")
    reasons = []
    for key in keys:
        values = (before.get(key), after.get(key))
        if not all(type(value) in (int, float) and isfinite(value) for value in values):
            reasons.append(f"{key}_evidence_missing_or_nonfinite")
        elif values[1] < values[0]:
            reasons.append(f"{key}_decreased")
    return reasons


def plan_protection_basket(raw, smith_stock, repair_quotes, healer_stock, failed=()):
    """Prefer strictly higher native AC inside the original eligible pool.

    Replanning after every real transaction remains the caller's responsibility.
    Direct callers must supply current native projections; the service below
    refreshes already-seen Smith identities through the existing native API.
    Equal AC, unknown AC or no affordable basket preserves the original choice.
    Switching also requires known nondecreasing damage and block chance, and
    nondecreasing max HP when both projections expose it. These preferences do
    not assert a repair-after durability value or a guarantee of survival.
    """
    result = plan_ordinary_armor_basket(
        raw, smith_stock, repair_quotes, healer_stock, failed)
    original = result["chosen"]
    affordable = [item for item in result["alternatives"] if item["affordable"]]
    effective = [item for item in affordable
                 if item["projected_block_enabled"] is True]
    pool = effective or affordable
    known = [item for item in pool if _native_armor_class(item) is not None]
    selected = original
    excluded = []
    reason = "no_fully_quoted_affordable_plan"
    original_ac = _native_armor_class(original)
    if original is not None:
        reason = "original_native_armor_class_unknown"
        if original_ac is not None:
            better = [item for item in known if _native_armor_class(item) > original_ac]
            reason = "no_strict_native_armor_improvement"
            if better:
                nonregressing = []
                for item in better:
                    reasons = _nonregression_reasons(original, item)
                    if reasons:
                        excluded.append({"candidate": _selection(item), "reasons": reasons})
                    else:
                        nonregressing.append(item)
                reason = "higher_armor_candidates_regress_or_lack_native_evidence"
                if nonregressing:
                    selected = min(nonregressing, key=lambda item: (
                        -_native_armor_class(item), item["total_cost"],
                        len(item["commands"]), item["kind"], item.get("identity", ())))
                    reason = "higher_native_armor_within_original_affordable_pool"
    result["chosen"] = selected
    # Retain the original affordability/blocking evidence, but do not label the
    # new chosen item as the original cheapest effective-blocking selection.
    result["blocking_preference"]["policy"] = (
        "effective_blocking_if_complete_affordable_then_native_ac_then_lowest_total")
    result["blocking_preference"]["selected_block_enabled"] = (
        None if selected is None else selected["projected_block_enabled"])
    result["protection_preference"] = {
        "policy": PROTECTION_POLICY,
        "is_native_readiness_gate": False,
        "reason": reason,
        "pool": "affordable_effective_blocking" if effective else "affordable_fallback",
        "pool_size": len(pool),
        "unknown_native_armor_class_candidates": len(pool) - len(known),
        "nonregression_required": ["damage", "block_chance"],
        "nonregression_when_both_present": ["max_hp"],
        "excluded_candidates": excluded,
        "original_choice": _selection(original),
        "selected_choice": _selection(selected),
        "changed_from_original": selected is not original,
    }
    return result


@dataclass
class SustainProtectionService(SustainEquipmentReadinessService):
    """Dormant opt-in service; all routes, budgets and native commands inherited."""
    _protection_selection: dict | None = field(default=None, init=False, repr=False)

    def _plan_basket(self, raw, smith_stock, repair_quotes, healer_stock, failed=()):
        refreshed = self._refresh_smith_projections(raw, smith_stock)
        result = plan_protection_basket(raw, refreshed, repair_quotes, healer_stock, failed)
        self._protection_selection = deepcopy(result["protection_preference"])
        return result

    def telemetry(self):
        result = super().telemetry()
        result["policy"] = PROTECTION_POLICY
        result["protection_preference"] = deepcopy(self._protection_selection)
        return result
