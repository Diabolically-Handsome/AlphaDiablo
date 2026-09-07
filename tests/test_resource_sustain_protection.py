"""Synthetic native quotes only: no model, game or optimizer execution."""
from copy import deepcopy
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from diablogym.resource_sustain import RESOURCE_SERVICE_POLICIES, item_identity
from diablogym.resource_sustain_armor import (
    SustainEquipmentReadinessService, plan_ordinary_armor_basket,
)
from diablogym.resource_sustain_protection import (
    PROTECTION_POLICY, SustainProtectionService, plan_protection_basket,
)


def readiness(ac=9, heals=4, failures=(), block=True):
    return dict(ready=not failures, failures=list(failures), armor_class=ac,
                damage=7, weapon_equipped=True, hp=86, max_hp=86,
                min_finite_durability=16, required_belt_heals=4,
                belt_heals=heals, belt_free_slots=8-heals,
                block_enabled=block, block_chance=56)


def item(slot=4, **changes):
    result = dict(present=True, slot=slot, seed_hi=7, seed_lo=slot,
                  create_info=3, base_id=slot+50, durability=20, max_durability=24)
    result.update(changes)
    return result


def state(cash=359, heals=5, ac=9, failures=()):
    equipped = [dict(present=False) for _ in range(7)]
    equipped[4], equipped[5] = item(4), item(5)
    return dict(gold=cash, hp=86, max_hp=86, block_enabled=True, block_chance=56,
                dungeon_level=0, is_set_level=False, player_x=10, player_y=10,
                equipped_items=equipped,
                resource_state=dict(enabled=True, ordinary_armor_scope=True,
                    preserve_equipment_readiness=True, town_seed=123,
                    town_restock_sequence=1, readiness=readiness(ac, heals, failures),
                    inventory_items=[], unequip_candidates=[],
                    town=dict(active_vendor="smith", dialog_active=False,
                              stock=[], repair_quotes=[])))


def armor(ac=15, price=75, slot=6, identity=57, **changes):
    result = dict(item(slot, seed_lo=identity, base_id=identity), vendor="smith",
                  index=0, price=price, is_ordinary_armor=True, can_equip=True,
                  can_use=True, can_fit=True, meets_armor_gate=True,
                  target_slot=slot, replaced_slots=[], projected_readiness=readiness(ac))
    result.update(changes)
    return result


def medicine():
    return dict(item(0), vendor="healer", index=0, price=50, heal_kind=1,
                can_use=True, can_fit=True)


def plan(raw, stock=(), quotes=(), healer=None, failed=()):
    return plan_protection_basket(raw, stock, quotes,
                                 [medicine()] if healer is None else healer, failed)


def projected(offer, raw, status="projected"):
    return dict(deepcopy(offer), status=status,
                projection_origin="known-smith-live-player",
                town_seed=raw["resource_state"]["town_seed"],
                town_restock_sequence=raw["resource_state"]["town_restock_sequence"])


def test_ready_character_can_spend_real_surplus_on_highest_native_ac():
    raw = state()
    result = plan(raw, [armor(), armor(ac=20, price=300, identity=59)])
    assert result["chosen"]["identity"] == (7, 59, 3, 59)
    assert result["chosen"]["total_cost"] == 300
    assert result["chosen"]["commands"] == [("buy", "smith", 0, 7, 59, 3, 59)]
    assert result["medicine_cost"] == 0
    assert result["protection_preference"]["original_choice"]["total_cost"] == 0


@pytest.mark.parametrize("cash,expected", [(199, "retain_and_repair"),
                                          (200, "buy_ordinary_armor")])
def test_four_potion_reserve_is_not_spent_on_armor(cash, expected):
    result = plan(state(cash, heals=2, failures=("potions",)), [armor(price=100)])
    assert result["medicine_cost"] == 100
    assert result["chosen"]["kind"] == expected


def test_retained_repairs_remain_in_total_budget():
    raw = state(107, heals=2, failures=("potions", "durability"))
    worn = item(5, durability=13)
    raw["equipped_items"][5] = worn
    quote = dict(worn, price=7, repair_needed_for_gate=True)
    result = plan(raw, [armor(price=1)], [quote])
    assert result["chosen"]["total_cost"] == 107
    assert result["chosen"]["kind"] == "retain_and_repair"
    assert result["chosen"]["commands"][0][0] == "repair"


def test_higher_ac_cannot_displace_affordable_effective_blocking():
    no_block = armor(ac=99, price=1, projected_readiness=readiness(99, block=False))
    result = plan(state(), [no_block, armor(ac=15)])
    assert result["chosen"]["projected_readiness"]["armor_class"] == 15
    assert result["chosen"]["projected_block_enabled"] is True
    assert result["protection_preference"]["pool"] == "affordable_effective_blocking"


def test_original_fallback_pool_is_preserved_when_no_effective_plan_exists():
    raw = state()
    raw["block_enabled"] = False
    offer = armor(projected_readiness=readiness(15, block=False))
    result = plan(raw, [offer])
    assert result["chosen"]["projected_readiness"]["armor_class"] == 15
    assert result["protection_preference"]["pool"] == "affordable_fallback"


@pytest.mark.parametrize("change", [dict(meets_armor_gate=False, durability=6, max_durability=6),
    dict(can_equip=False), dict(can_fit=False),
    dict(projected_readiness=readiness(99, failures=("damage",))),
    dict(projected_readiness=readiness(99, failures=("weapon",))),
    dict(projected_readiness=readiness(99, failures=("health",)))])
def test_native_ineligibility_is_not_overridden_by_ac(change):
    assert plan(state(), [armor(ac=99, **change)])["chosen"]["kind"] == "retain_and_repair"


def test_equal_ac_keeps_original_and_free_owned_items_do_not_oscillate():
    raw = state()
    raw["resource_state"]["inventory_items"] = [armor(ac=9, price=0)]
    for unused in range(3):
        result = plan(raw, [armor(ac=9)])
        assert result["chosen"]["kind"] == "retain_and_repair"
        assert result["chosen"]["commands"] == []
        assert not result["protection_preference"]["changed_from_original"]


def test_after_real_state_change_replan_retains_purchase_without_buying_or_equipping_back():
    raw = state()
    old = armor(ac=9, price=0, slot=6, identity=54)
    raw["equipped_items"][6] = old
    upgrade = armor(ac=20, price=300, identity=59, replaced_slots=[6])
    equal = armor(ac=20, price=25, identity=57, replaced_slots=[6])
    first = plan(raw, [upgrade])
    assert first["chosen"]["identity"][-1] == 59
    raw["gold"] -= first["chosen"]["total_cost"]
    raw["equipped_items"][6] = deepcopy(upgrade)
    raw["resource_state"]["readiness"] = readiness(ac=20, heals=5)
    raw["resource_state"]["inventory_items"] = [dict(old, index=2,
        projected_readiness=readiness(ac=9, heals=5), replaced_slots=[6])]
    # Purchased identity is removed from stock; remaining quote is native AC20.
    second = plan(raw, [equal])
    assert second["chosen"]["kind"] == "retain_and_repair"
    assert second["chosen"]["commands"] == []
    assert not second["protection_preference"]["changed_from_original"]


def test_same_ac_uses_cost_then_stable_identity_independent_of_stock_order():
    offers = [armor(ac=20, price=100, identity=60),
              armor(ac=20, price=75, identity=59), armor(ac=20, price=75, identity=57)]
    for stock in (offers, list(reversed(offers))):
        selected = plan(state(), stock)["chosen"]
        assert selected["total_cost"] == 75 and selected["identity"][-1] == 57


def test_same_cost_and_ac_prefers_fewer_original_commands():
    raw = state(failures=("durability",))
    worn = item(0, durability=5)
    raw["equipped_items"][0] = worn
    quote = dict(worn, price=10, repair_needed_for_gate=True)
    result = plan(raw, [armor(ac=20, price=15),
                       armor(ac=20, price=25, slot=0, identity=49, replaced_slots=[0])], [quote])
    assert result["chosen"]["target_slot"] == 0
    assert result["chosen"]["total_cost"] == 25 and len(result["chosen"]["commands"]) == 1


@pytest.mark.parametrize("value", [None, True, 99.0, float("nan")])
def test_unknown_or_noninteger_native_ac_does_not_invent_an_improvement(value):
    offer = armor()
    offer["projected_readiness"]["armor_class"] = value
    assert plan(state(), [offer])["chosen"]["kind"] == "retain_and_repair"


def test_missing_original_ac_keeps_original_choice():
    raw = state()
    del raw["resource_state"]["readiness"]["armor_class"]
    result = plan(raw, [armor()])
    assert result["chosen"]["kind"] == "retain_and_repair"
    assert result["protection_preference"]["reason"] == "original_native_armor_class_unknown"


@pytest.mark.parametrize("key,value,reason", [
    ("damage", 6, "damage_decreased"),
    ("block_chance", 55, "block_chance_decreased"),
    ("max_hp", 85, "max_hp_decreased"),
    ("damage", None, "damage_evidence_missing_or_nonfinite"),
    ("block_chance", float("inf"), "block_chance_evidence_missing_or_nonfinite"),
    ("max_hp", float("nan"), "max_hp_evidence_missing_or_nonfinite"),
])
def test_more_ac_cannot_lower_or_invent_native_combat_evidence(key, value, reason):
    offer = armor(ac=99)
    offer["projected_readiness"][key] = value
    result = plan(state(), [offer])
    assert result["chosen"]["kind"] == "retain_and_repair"
    assert reason in result["protection_preference"]["excluded_candidates"][0]["reasons"]


@pytest.mark.parametrize("key", ["damage", "block_chance"])
def test_missing_required_field_on_original_or_candidate_preserves_original(key):
    for location in ("original", "candidate"):
        raw, offer = state(), armor(ac=99)
        if location == "candidate":
            del offer["projected_readiness"][key]
        elif key == "block_chance":
            del raw[key]
        else:
            del raw["resource_state"]["readiness"][key]
        result = plan(raw, [offer])
        assert result["chosen"]["kind"] == "retain_and_repair"
        assert f"{key}_evidence_missing_or_nonfinite" in result["protection_preference"]["excluded_candidates"][0]["reasons"]


def test_optional_max_hp_absence_and_unprojected_repair_margin_do_not_invent_new_gate():
    offer = armor()
    del offer["projected_readiness"]["max_hp"]
    offer["projected_readiness"]["min_finite_durability"] = 15
    result = plan(state(), [offer])
    assert result["chosen"]["kind"] == "buy_ordinary_armor"


def test_selects_lower_ac_nonregressing_offer_when_highest_ac_loses_damage():
    bad, good = armor(ac=30, price=25, identity=60), armor(ac=20, price=75, identity=59)
    bad["projected_readiness"]["damage"] = 6
    result = plan(state(), [bad, good])
    assert result["chosen"]["identity"][-1] == 59
    assert result["protection_preference"]["excluded_candidates"][0]["reasons"] == ["damage_decreased"]


def test_failed_purchase_and_incomplete_medicine_or_capacity_cannot_be_bypassed():
    offer = armor()
    command = ("buy", "smith", 0, *item_identity(offer))
    assert plan(state(), [offer], failed=[command])["chosen"]["kind"] == "retain_and_repair"
    assert plan(state(heals=2, failures=("potions",)), [offer], healer=[])["chosen"] is None
    raw = state(heals=2, failures=("potions",))
    raw["resource_state"]["readiness"]["belt_free_slots"] = 1
    assert plan(raw, [offer])["chosen"] is None


def test_inputs_rng_and_original_planner_output_are_unchanged():
    args = (state(), [armor(), armor(ac=20, price=300)], [], [medicine()], ())
    before = deepcopy(args)
    legacy = plan_ordinary_armor_basket(*args)
    rng, np_rng = random.getstate(), np.random.get_state()
    result = plan_protection_basket(*args)
    assert args == before and random.getstate() == rng
    np_after = np.random.get_state()
    assert np.array_equal(np_after[1], np_rng[1]) and np_after[2:] == np_rng[2:]
    assert plan_ordinary_armor_basket(*args) == legacy
    assert result["alternatives"] == legacy["alternatives"]
    assert legacy["chosen"]["kind"] == "retain_and_repair"
    assert "protection_preference" not in legacy


def test_service_refreshes_once_and_uses_fresh_projection_not_cached_ac():
    raw, old = state(), armor(ac=99)
    fresh = projected(armor(ac=15), raw)
    service = SustainProtectionService()
    service._projection_bridge = SimpleNamespace(project_seen_resource_smith_items=Mock(return_value=[fresh]))
    result = service._plan_basket(raw, [old], [], [medicine()])
    assert result["chosen"]["projected_readiness"]["armor_class"] == 15
    service._projection_bridge.project_seen_resource_smith_items.assert_called_once_with([item_identity(old)])
    assert service.steps == 0 and service.telemetry()["smith_projection_refreshes"] == 1
    audit = service.telemetry()["protection_preference"]
    audit["reason"] = "tampered"
    assert service.telemetry()["protection_preference"]["reason"] != "tampered"


def test_service_uses_original_buy_then_exact_inventory_equip_path():
    raw, offer = state(), armor(ac=20, price=300, identity=59)
    raw["resource_state"]["town"]["stock"] = [offer]
    service = SustainProtectionService(active=True, attempted=True, phase="joint_gear")
    service._healer_stock = [medicine()]
    native = SimpleNamespace(project_seen_resource_smith_items=Mock(
        return_value=[projected(offer, raw)]))
    env = SimpleNamespace(_raw=raw, _steps=1)
    command = service.command(env, native)
    assert command == ("buy", "smith", 0, 7, 59, 3, 59)
    service.receipt(command, {"accepted": True, "price": 300})
    raw["gold"] -= 300
    raw["resource_state"]["inventory_items"] = [dict(offer, index=2)]
    assert service.command(env, native) == ("equip", 2, 7, 59, 3, 59)
    assert service.gold_spent == 300 and service.purchases == 1
    native.project_seen_resource_smith_items.assert_called_once_with([item_identity(offer)])


def test_stale_identity_is_dropped_without_cached_fallback_and_error_propagates():
    raw, old = state(), armor()
    stale = projected({k: old[k] for k in ("seed_hi", "seed_lo", "create_info", "base_id")}, raw, "stale_item")
    service = SustainProtectionService()
    service._projection_bridge = SimpleNamespace(project_seen_resource_smith_items=Mock(return_value=[stale]))
    assert service._plan_basket(raw, [old], [], [medicine()])["chosen"]["kind"] == "retain_and_repair"
    assert service._smith_stock == [] and service.telemetry()["stale_smith_projection_items"]
    service._projection_bridge.project_seen_resource_smith_items.side_effect = ValueError("unknown native identity")
    with pytest.raises(ValueError, match="unknown native identity"):
        service._plan_basket(raw, [old], [], [medicine()])


def test_service_remains_unregistered_and_inherits_routes_budget_and_live_execution():
    service = SustainProtectionService()
    assert PROTECTION_POLICY not in RESOURCE_SERVICE_POLICIES
    assert service.collect_microstep_cap == 450 and service.service_microstep_cap == 1500
    assert SustainProtectionService._command is SustainEquipmentReadinessService._command
    assert SustainProtectionService._walk is SustainEquipmentReadinessService._walk
    assert SustainProtectionService.receipt is SustainEquipmentReadinessService.receipt
    assert service.telemetry()["policy"] == PROTECTION_POLICY
    assert SustainEquipmentReadinessService().telemetry()["policy"] == "sustain-v6"
