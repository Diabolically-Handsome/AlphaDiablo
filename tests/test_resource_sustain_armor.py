"""V5 plans normal armor from native projections, with no game execution."""
from copy import deepcopy
import random
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from diablogym.resource_sustain import SustainResourceService
from diablogym.resource_sustain_armor import SustainOrdinaryArmorService, plan_ordinary_armor_basket
from diablogym.resource_sustain_gold import SustainGoldExtendedService


def item(slot=6, durability=30, max_durability=30, **changes):
    result = dict(present=True, slot=slot, seed_hi=7, seed_lo=slot,
                  create_info=3, base_id=54 + slot, durability=durability,
                  max_durability=max_durability)
    result.update(changes)
    return result


def readiness(failures=(), heals=4, **changes):
    result = dict(ready=not failures, failures=list(failures), armor_class=10,
                  damage=6, weapon_equipped=True, hp=70, max_hp=70,
                  min_finite_durability=30, required_belt_heals=4,
                  belt_heals=heals, belt_free_slots=8-heals,
                  block_enabled=True, block_chance=35)
    result.update(changes)
    return result


def raw(gold=500, failures=("armor",), heals=4):
    equipment = [dict(present=False) for _ in range(7)]
    equipment[4], equipment[6] = item(4), item(6)
    return dict(dungeon_level=0, is_set_level=False, gold=gold, hp=70, max_hp=70,
                block_enabled=True, block_chance=35,
                player_x=10, player_y=10, equipped_items=equipment,
                resource_state=dict(enabled=True, ordinary_armor_scope=True,
                    town_seed=123, town_restock_sequence=1,
                    readiness=readiness(failures, heals), inventory_items=[], unequip_candidates=[],
                    town=dict(active_vendor="smith", dialog_active=False, stock=[], repair_quotes=[])))


def armor(slot=0, price=25, replaced=(), **changes):
    result = item(slot, seed_lo=100+slot, base_id=48 if slot == 0 else 71,
                  index=0, vendor="smith", price=price, can_use=True, can_fit=True,
                  can_equip=True, equip_reason="ready", is_ordinary_armor=True,
                  target_slot=slot, replaced_slots=list(replaced), meets_armor_gate=True,
                  projected_readiness=readiness())
    result.update(changes)
    return result


def medicine(price=50, index=0):
    return dict(item(0), vendor="healer", index=index, price=price, heal_kind=1,
                can_use=True, can_fit=True)


def projected(item, state, status="projected"):
    return dict(deepcopy(item), status=status, projection_origin="known-smith-live-player",
                town_seed=state["resource_state"]["town_seed"],
                town_restock_sequence=state["resource_state"]["town_restock_sequence"])


def quote(equipment, price):
    return dict(equipment, price=price, repair_needed_for_gate=True)


def plan(state, stock=(), quotes=(), medicine_stock=None, failed=()):
    return plan_ordinary_armor_basket(state, stock, quotes,
        [medicine()] if medicine_stock is None else medicine_stock, failed)


@pytest.mark.parametrize("scope", [None, False, "true"])
def test_native_opt_in_is_required(scope):
    state = raw()
    state["resource_state"]["ordinary_armor_scope"] = scope
    with pytest.raises(RuntimeError, match="ordinary_armor_scope=True"):
        plan(state, [armor()])


@pytest.mark.parametrize("slot", [0, 4, 5, 6])
def test_actual_target_slot_replaces_only_its_own_repair(slot):
    state = raw(gold=82, failures=("armor", "durability"), heals=3)
    state["equipped_items"][slot] = item(slot, durability=2)
    other = 6 if slot != 6 else 0
    state["equipped_items"][other] = item(other, durability=3)
    quotes = [quote(state["equipped_items"][slot], 100), quote(state["equipped_items"][other], 7)]
    chosen = plan(state, [armor(slot, replaced=[slot])], quotes)["chosen"]
    assert chosen["total_cost"] == 25 + 7 + 50
    assert chosen["target_slot"] == slot and chosen["replaced_slots"] == [slot]
    assert [command[1] for command in chosen["commands"] if command[0] == "repair"] == [other]


def test_empty_head_can_be_filled_without_removing_unrepairable_chest():
    state = raw(failures=("armor", "durability"))
    state["equipped_items"][6] = item(6, durability=2, max_durability=6)
    result = plan(state, [armor()])
    assert not result["alternatives"]
    assert "retained_item_max_durability" in {entry["reason"] for entry in result["rejected_candidates"]}


def test_multislot_native_plan_excludes_every_displaced_repair():
    state = raw(gold=35, failures=("armor", "durability"))
    state["equipped_items"][4] = item(4, durability=2, max_durability=6)
    state["equipped_items"][5] = item(5, durability=2, max_durability=6)
    state["equipped_items"][6] = item(6, durability=2)
    chosen = plan(state, [armor(5, replaced=[4, 5])], [quote(state["equipped_items"][6], 10)])["chosen"]
    assert chosen["total_cost"] == 35 and chosen["replaced_slots"] == [4, 5]
    assert [command[1] for command in chosen["commands"] if command[0] == "repair"] == [6]


@pytest.mark.parametrize("failure", ["weapon", "damage", "health", "armor", "level", "unknown_future_gate"])
def test_complete_native_projection_rejects_unresolved_cascades(failure):
    candidate = armor(5, replaced=[4], projected_readiness=readiness([failure]))
    # Even an apparently favorable AC and legacy meets flag cannot override
    # losing the two-handed weapon, life, or another canonical requirement.
    candidate["projected_readiness"]["armor_class"] = 99
    assert plan(raw(), [candidate])["chosen"] is None


@pytest.mark.parametrize("replacement", [[1], [0], [4, 4], []])
def test_stale_or_invalid_replacement_slots_fail_closed(replacement):
    assert plan(raw(), [armor(4, replaced=replacement)])["chosen"] is None


def test_native_blocking_projection_is_preserved_without_adding_a_block_gate():
    for enabled, chance in [(False, 0), (True, 35)]:
        candidate = armor(projected_readiness=readiness(block_enabled=enabled, block_chance=chance))
        chosen = plan(raw(), [candidate])["chosen"]
        assert chosen["projected_readiness"]["block_enabled"] is enabled
        assert chosen["projected_readiness"]["block_chance"] == chance
        chosen["projected_readiness"]["armor_class"] = -100
        assert candidate["projected_readiness"]["armor_class"] == 10


def test_cheapest_complete_repair_plan_reserves_four_potions_before_armor():
    state = raw(gold=105, failures=("durability", "potions"), heals=2)
    state["equipped_items"][6] = item(6, durability=2)
    result = plan(state, [armor(6, replaced=[6], price=10)], [quote(state["equipped_items"][6], 5)])
    assert result["chosen"]["kind"] == "retain_and_repair"
    assert result["chosen"]["total_cost"] == 105
    state["gold"] = 104
    result = plan(state, [armor(6, replaced=[6], price=10)], [quote(state["equipped_items"][6], 5)])
    assert result["chosen"] is None and result["shortfall"] == 1
    assert not result["zero_improvements"]


def test_replacement_can_be_cheaper_than_repair_and_keeps_other_repairs():
    state = raw(failures=("durability",))
    state["equipped_items"][0] = item(0, durability=2)
    state["equipped_items"][6] = item(6, durability=2)
    quotes = [quote(state["equipped_items"][0], 40), quote(state["equipped_items"][6], 5)]
    chosen = plan(state, [armor(0, price=25, replaced=[0])], quotes)["chosen"]
    assert chosen["kind"] == "buy_ordinary_armor" and chosen["total_cost"] == 30
    assert len(chosen["commands"]) == 2


def test_stale_repair_identity_current_durability_and_rejected_commands():
    state = raw(failures=("durability",))
    state["equipped_items"][6] = item(6, durability=2)
    valid = quote(state["equipped_items"][6], 5)
    for change in ({"seed_hi": 999}, {"seed_lo": 999}, {"create_info": 999},
                   {"base_id": 999}, {"durability": 3}):
        stale = dict(valid, **change)
        assert not plan(state, quotes=[stale])["alternatives"]
    command = plan(state, quotes=[valid])["chosen"]["commands"][0]
    assert not plan(state, quotes=[valid], failed=[command])["alternatives"]


def test_missing_medicine_capacity_and_rejected_potion_have_no_complete_basket():
    state = raw(heals=2)
    result = plan(state, [armor()], medicine_stock=[])
    assert result["chosen"] is None and "medicine_stock" in result["unresolved"]
    state["resource_state"]["readiness"]["belt_free_slots"] = 1
    result = plan(state, [armor()])
    assert result["chosen"] is None and "belt_capacity" in result["unresolved"]
    state["resource_state"]["readiness"]["belt_free_slots"] = 6
    rejected = ("buy", "healer", 0, 7, 0, 3, 54)
    result = plan(state, [armor()], failed=[rejected])
    assert result["chosen"] is None and result["medicine_cost"] is None


@pytest.mark.parametrize("changes", [dict(can_use=False), dict(can_fit=False), dict(can_equip=False),
                                    dict(is_ordinary_armor=False), dict(meets_armor_gate=False)])
def test_native_purchase_permission_is_required(changes):
    assert plan(raw(), [armor(**changes)])["chosen"] is None


def test_owned_item_needs_native_equip_capacity_not_new_purchase_space():
    state = raw()
    state["resource_state"]["inventory_items"] = [armor(can_fit=False)]
    chosen = plan(state)["chosen"]
    assert chosen["kind"] == "equip_owned_ordinary_armor" and chosen["total_cost"] == 0
    command = chosen["commands"][0]
    assert command == ("equip", 0, 7, 100, 3, 48)
    assert plan(state, failed=[command])["chosen"] is None


def test_worn_owned_item_without_prospective_repair_quote_is_outside_catalog():
    state = raw()
    state["resource_state"]["inventory_items"] = [armor(durability=2, max_durability=20, meets_armor_gate=False)]
    assert not plan(state)["alternatives"]
    assert not plan(state)["zero_improvements"]


def test_live_ready_retention_wins_over_pointless_free_equips():
    state = raw(failures=())
    state["resource_state"]["inventory_items"] = [armor()]
    chosen = plan(state)["chosen"]
    assert chosen["kind"] == "retain_and_repair" and chosen["commands"] == []


def test_native_legal_lower_ac_replacement_can_solve_durability_gate():
    state = raw(gold=15, failures=("durability",))
    state["resource_state"]["readiness"]["armor_class"] = 11
    state["equipped_items"][0] = item(0, durability=14, max_durability=20)
    candidate = armor(0, price=15, replaced=[0], durability=15, max_durability=15,
                      projected_readiness=readiness(armor_class=10))
    # The native ordinary-scope legality fix must permit this proposal. Python
    # neither reconstructs the old utility score nor prefers a higher AC.
    chosen = plan(state, [candidate])["chosen"]
    assert chosen["kind"] == "buy_ordinary_armor" and chosen["total_cost"] == 15
    assert chosen["projected_readiness"]["armor_class"] == 10
    candidate.update(can_equip=False, equip_reason="not_upgrade", projected_readiness={})
    assert plan(state, [candidate])["chosen"] is None


def test_normal_unequip_compares_with_purchase_and_validates_exact_identity():
    state = raw(failures=("durability",))
    state["equipped_items"][6] = item(6, durability=2, max_durability=6)
    candidate = dict(state["equipped_items"][6], can_unequip=True, projected_readiness=readiness())
    state["resource_state"]["unequip_candidates"] = [candidate]
    chosen = plan(state, [armor(6, replaced=[6])])["chosen"]
    assert chosen["kind"] == "unequip_failing_armor" and chosen["total_cost"] == 0
    candidate["seed_lo"] = 999
    assert plan(state)["chosen"] is None


def test_planning_does_not_modify_inputs_or_rng():
    state, stock = raw(), [armor()]
    before = deepcopy((state, stock))
    rng, np_rng = random.getstate(), np.random.get_state()
    assert plan(state, stock)["chosen"] is not None
    assert (state, stock) == before and random.getstate() == rng
    after = np.random.get_state()
    assert after[0] == np_rng[0] and np.array_equal(after[1], np_rng[1]) and after[2:] == np_rng[2:]


def shield_repair_case(cash=184):
    state = raw(gold=cash, failures=("durability", "potions"), heals=2)
    state["equipped_items"][5] = item(5, durability=12, max_durability=16)
    state["resource_state"]["unequip_candidates"] = [dict(
        state["equipped_items"][5], can_unequip=True,
        projected_readiness=readiness(["potions"], heals=2, armor_class=11,
                                      block_enabled=False, block_chance=35))]
    return state, [quote(state["equipped_items"][5], 3)]


def test_three_gold_shield_repair_plus_two_potions_beats_free_shield_removal():
    state, quotes = shield_repair_case()
    result = plan(state, quotes=quotes)
    assert result["chosen"]["kind"] == "retain_and_repair"
    assert result["chosen"]["total_cost"] == 103
    assert result["minimum_observed_total"] == 100  # Cheapest old-gate basket still reported.
    assert result["chosen"]["commands"] == [("repair", 5, 7, 5, 3, 59, 12, 3)]
    preference = result["blocking_preference"]
    assert preference["selected_block_enabled"] is True
    assert preference["reason"] == "effective_blocking_preserved_or_restored"
    assert preference["minimum_observed_effective_total"] == 103
    assert preference["is_native_readiness_gate"] is False


def test_unaffordable_full_shield_plan_falls_back_without_spending_potion_reserve():
    state, quotes = shield_repair_case(cash=102)
    result = plan(state, quotes=quotes)
    assert result["chosen"]["kind"] == "unequip_failing_armor"
    assert result["chosen"]["total_cost"] == 100
    assert result["blocking_preference"]["reason"] == "effective_blocking_plan_not_affordable"
    assert result["blocking_preference"]["effective_shortfall"] == 1
    state["gold"] = 99
    result = plan(state, quotes=quotes)
    assert result["chosen"] is None
    assert result["blocking_preference"]["reason"] == "no_fully_quoted_affordable_plan"


def test_missing_block_projection_is_unknown_not_an_invented_false_or_true():
    state, quotes = shield_repair_case()
    del state["block_enabled"]; del state["block_chance"]
    projected = state["resource_state"]["unequip_candidates"][0]["projected_readiness"]
    del projected["block_enabled"]; del projected["block_chance"]
    result = plan(state, quotes=quotes)
    assert result["chosen"]["kind"] == "unequip_failing_armor"
    preference = result["blocking_preference"]
    assert preference["reason"] == "blocking_projection_unknown"
    assert preference["selected_block_enabled"] is None and preference["unknown_alternatives"] == 2


def test_repair_does_not_invent_blocking_restoration_from_false_current_state():
    state, quotes = shield_repair_case()
    state["block_enabled"] = False
    result = plan(state, quotes=quotes)
    assert result["chosen"]["kind"] == "unequip_failing_armor"
    assert result["blocking_preference"]["known_effective_alternatives"] == 0
    assert result["blocking_preference"]["reason"] == "no_observed_effective_blocking_plan"


def test_shield_blocking_preference_never_overrides_native_weapon_or_damage_failure():
    for failure in ("weapon", "damage"):
        candidate = armor(5, price=1, replaced=[4],
            projected_readiness=readiness([failure], block_enabled=True))
        result = plan(raw(), [candidate])
        assert result["chosen"] is None
        assert result["blocking_preference"]["known_effective_alternatives"] == 0


def test_legacy_hook_forwards_exact_existing_arguments_and_return():
    arguments, sentinel = ({}, [], [], [], set()), object()
    with patch("diablogym.resource_sustain.plan_basket", return_value=sentinel) as original:
        assert SustainResourceService()._plan_basket(*arguments) is sentinel
    original.assert_called_once_with(*arguments)


def test_v5_dispatch_buy_receipt_then_equips_exact_ordinary_inventory_item():
    state, candidate = raw(), armor(5, replaced=[])
    state["resource_state"]["town"]["stock"] = [candidate]
    service = SustainOrdinaryArmorService(active=True, attempted=True, phase="joint_gear")
    service._healer_stock = [medicine()]
    env = SimpleNamespace(_raw=state, _steps=1)
    native = SimpleNamespace(project_seen_resource_smith_items=Mock(return_value=[projected(candidate, state)]))
    command = service.command(env, native)
    assert command == ("buy", "smith", 0, 7, 105, 3, 71)
    service.receipt(command, {"accepted": True, "price": 25})
    state["resource_state"]["inventory_items"] = [dict(candidate, index=3)]
    assert service.command(env, native) == ("equip", 3, 7, 105, 3, 71)
    native.project_seen_resource_smith_items.assert_called_once_with([(7, 105, 3, 71)])
    assert service.gold_spent == 25 and service.purchases == 1


def test_v5_keeps_v4_routes_clocks_and_handoff_semantics():
    service = SustainOrdinaryArmorService()
    assert service.collect_microstep_cap == 450 and service.service_microstep_cap == 1500
    assert SustainOrdinaryArmorService._walk is SustainGoldExtendedService._walk
    assert SustainOrdinaryArmorService._leave_collect is SustainGoldExtendedService._leave_collect
    telemetry = service.telemetry()
    assert telemetry["policy"] == "sustain-v5"
    assert telemetry["collect_command_window_microsteps"] == 450
    assert "collect_microstep_cap" not in telemetry and telemetry["collect_handoff"] is None


def test_v5_command_binds_original_bridge_once_and_clears_even_after_exception():
    service, env, native = SustainOrdinaryArmorService(), object(), object()
    def delegated(given_env, given_native):
        assert given_env is env and given_native is native
        assert service._projection_bridge is native
        return ("wait",)
    with patch.object(SustainGoldExtendedService, "_command", side_effect=delegated) as original:
        assert service._command(env, native) == ("wait",)
        original.assert_called_once_with(env, native)
    assert service._projection_bridge is None
    with patch.object(SustainGoldExtendedService, "_command", side_effect=ValueError("native-error")):
        with pytest.raises(ValueError, match="native-error"):
            service._command(env, native)
    assert service._projection_bridge is None


def test_live_known_quote_projection_repairs_stale_pre_heal_health_without_advancing():
    state = raw()
    old = armor(projected_readiness=readiness(["health"], hp=30))
    fresh = projected(armor(), state)
    service = SustainOrdinaryArmorService(active=True, attempted=True, phase="joint_gear")
    service._smith_stock = [old]
    native = SimpleNamespace(project_seen_resource_smith_items=Mock(return_value=[fresh]))
    service._projection_bridge = native
    before = deepcopy((state, old)); clock = service.steps
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    result = service._plan_basket(state, service._smith_stock, [], [medicine()])
    assert result["chosen"]["projected_readiness"]["hp"] == 70
    assert result["chosen"]["kind"] == "buy_ordinary_armor"
    native.project_seen_resource_smith_items.assert_called_once_with([(7, 100, 3, 48)])
    assert (state, old) == before and service.steps == clock
    assert random.getstate() == python_rng
    after = np.random.get_state()
    assert np.array_equal(after[1], numpy_rng[1]) and after[2:] == numpy_rng[2:]
    assert service.telemetry()["smith_projection_refreshes"] == 1


def test_stale_seen_item_is_dropped_and_logged_with_no_old_projection_fallback():
    state, old = raw(), armor()
    stale = projected({key: old[key] for key in ("seed_hi", "seed_lo", "create_info", "base_id")},
                      state, "stale_item")
    service = SustainOrdinaryArmorService()
    service._smith_stock = [old]
    service._projection_bridge = SimpleNamespace(project_seen_resource_smith_items=Mock(return_value=[stale]))
    result = service._plan_basket(state, service._smith_stock, [], [medicine()])
    assert result["chosen"] is None and service._smith_stock == []
    assert service.telemetry()["stale_smith_projection_items"] == [dict(
        identity=(7, 100, 3, 48), reason="native_seen_item_no_longer_in_stock")]


@pytest.mark.parametrize("change", [{"seed_hi": 999}, {"projection_origin": "unseen-stock"},
    {"town_seed": 999}, {"town_restock_sequence": 999}, {"status": "unavailable"},
    {"vendor": "healer"}])
def test_projection_response_drift_is_engineering_error_and_keeps_cache(change):
    state, old = raw(), armor()
    response = dict(projected(old, state), **change)
    service = SustainOrdinaryArmorService()
    service._smith_stock = [old]
    service._projection_bridge = SimpleNamespace(project_seen_resource_smith_items=Mock(return_value=[response]))
    with pytest.raises(RuntimeError, match="projection response"):
        service._plan_basket(state, service._smith_stock, [], [medicine()])
    assert service._smith_stock == [old] and service._projection_refreshes == 0
    assert service._stale_projection_items == []


def test_no_known_smith_quotes_means_no_native_query_and_failures_are_not_no_cash():
    service, state = SustainOrdinaryArmorService(), raw()
    native = SimpleNamespace(project_seen_resource_smith_items=Mock(side_effect=ValueError("unknown identity")))
    service._projection_bridge = native
    assert service._plan_basket(state, [], [], [medicine()])["chosen"] is None
    native.project_seen_resource_smith_items.assert_not_called()
    with pytest.raises(ValueError, match="unknown identity"):
        service._plan_basket(state, [armor()], [], [medicine()])
    assert service._projection_refreshes == 0


def test_duplicate_requests_or_partial_responses_fail_closed():
    state, old = raw(), armor()
    service = SustainOrdinaryArmorService()
    native = SimpleNamespace(project_seen_resource_smith_items=Mock(return_value=[]))
    service._projection_bridge = native
    with pytest.raises(RuntimeError, match="duplicate identities"):
        service._plan_basket(state, [old, old], [], [medicine()])
    native.project_seen_resource_smith_items.assert_not_called()
    with pytest.raises(RuntimeError, match="response count"):
        service._plan_basket(state, [old], [], [medicine()])
