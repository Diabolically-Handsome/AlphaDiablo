"""Synthetic native verdicts and real Python service dispatch; no game/model."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import random

import numpy as np
import pytest

from diablogym.resource_sustain import item_identity, RESOURCE_SERVICE_POLICIES
from diablogym.resource_sustain_armor import plan_ordinary_armor_basket
from diablogym.resource_sustain_combinations import (
    BATCH_LIMIT, COMBINATION_POLICY, REPAIR_RETAINED, SustainCombinationService,
    equipment_sequences, plan_combination_basket,
)


def ready(failures=(), heals=4, block=True):
    return dict(ready=not failures, failures=list(failures), armor_class=12, damage=7,
                weapon_equipped=True, hp=86, max_hp=86, min_finite_durability=22,
                required_belt_heals=4, belt_heals=heals, belt_free_slots=8-heals,
                block_enabled=block, block_chance=56)


def item(slot, number=None, **kw):
    result = dict(present=True, slot=slot, seed_hi=12, seed_lo=slot if number is None else number,
                  create_info=1030, base_id=50+slot, durability=22, max_durability=24)
    result.update(kw)
    return result


def state(cash=367, heals=0, failures=("durability", "potions")):
    eq = [dict(present=False) for _ in range(7)]
    eq[4], eq[5] = item(4), item(5, durability=14, max_durability=16)
    eq[6] = item(6, durability=4, max_durability=6)
    return dict(dungeon_level=0, is_set_level=False, gold=cash, hp=86, max_hp=86,
                player_x=10, player_y=10, block_enabled=True, block_chance=56,
                equipped_items=eq, resource_state=dict(enabled=True, ordinary_armor_scope=True,
                    preserve_equipment_readiness=True, town_seed=123, town_restock_sequence=1,
                    readiness=ready(failures, heals), inventory_items=[],
                    unequip_candidates=[dict(eq[6], can_unequip=False,
                                            projected_readiness=ready(["armor", "potions"], heals))],
                    town=dict(active_vendor="smith", dialog_active=False, stock=[], repair_quotes=[])))


def offer(number=19, price=90, **kw):
    result = dict(item(5, number), vendor="smith", index=3, price=price,
                  is_ordinary_armor=True, can_equip=False, can_use=True, can_fit=True,
                  meets_armor_gate=False, target_slot=-1, replaced_slots=[],
                  projected_readiness=ready(["durability", "potions"], 0))
    result.update(kw)
    return result


def medicine(price=50):
    return dict(item(0, 99), vendor="healer", index=0, price=price, heal_kind=1,
                can_use=True, can_fit=True)


def native_source(raw):
    eq = []
    for index, entry in enumerate(raw["equipped_items"]):
        eq.append(dict(deepcopy(entry), index=index, empty=not entry.get("present"),
                       **({"identity": item_identity(entry)} if entry.get("present") else {})))
    return dict(gold=raw["gold"], equipment=eq, inventory_count=0, inventory=[], belt=[],
                inventory_grid=[], readiness=deepcopy(raw["resource_state"]["readiness"]),
                town_seed=123, town_restock_sequence=1, episode_generation=1)


def response(raw, sequences, budget, valid=None):
    """The test supplies verdicts; no equipment formula is reproduced here."""
    result = dict(version=COMBINATION_POLICY, max_gold_cost=budget, batch_limit=256,
                  step_limit=8, truncated=False, source=native_source(raw), results=[])
    for index, seq in enumerate(sequences):
        answer = valid(tuple(tuple(i) for i in seq)) if valid else None
        if answer is None:
            result["results"].append(dict(index=index, valid=False, reason="retained_item_max_durability",
                failed_step=0, gold_cost=0, gold_remaining=raw["gold"], expanded_commands=[],
                first_command=None, projected_readiness=None))
        else:
            commands, cost, projection = answer
            result["results"].append(dict(index=index, valid=True, reason="ready", failed_step=None,
                gold_cost=cost, gold_remaining=raw["gold"]-cost, expanded_commands=list(commands),
                first_command=commands[0] if commands else None, projected_readiness=deepcopy(projection)))
    return result


def sequence_for(raw, stock):
    return (("buy_equip", *item_identity(stock)),
            ("unequip", 6, *item_identity(raw["equipped_items"][6])), REPAIR_RETAINED)


def expanded(raw, stock):
    return [("buy", "smith", stock["index"], *item_identity(stock)),
            ("equip", 2, *item_identity(stock)),
            ("unequip", 6, *item_identity(raw["equipped_items"][6]))]


def pure_plan(raw, stock, sequences, preview, failed=(), meds=None):
    return plan_combination_basket(raw, stock, [], [medicine()] if meds is None else meds,
                                   sequences, preview, failed)


def test_catalog_covers_buy_then_remove_without_single_item_gate_or_seed_constants():
    raw, stock = state(), offer(can_equip=False, can_fit=False, can_use=False,
                                meets_armor_gate=False, target_slot=-1)
    seqs, info = equipment_sequences(raw, [stock])
    expected = sequence_for(raw, stock)
    assert expected in seqs
    assert (expected[1], expected[0], REPAIR_RETAINED) in seqs
    assert not info["truncated"]
    # A second completely different identity receives identical capability.
    stock["seed_hi"], stock["base_id"] = 32000, 71
    assert sequence_for(raw, stock) in equipment_sequences(raw, [stock])[0]


def test_unrepairable_single_chest_does_not_delete_valid_complete_combination():
    raw, stock = state(), offer()
    seqs, unused = equipment_sequences(raw, [stock])
    target = sequence_for(raw, stock)
    preview = response(raw, seqs, 167, lambda seq: (expanded(raw, stock), 90,
                       ready(["potions"], 0)) if seq == target else None)
    legacy = plan_ordinary_armor_basket(raw, [stock], [], [medicine()])
    assert legacy["chosen"] is None
    plan = pure_plan(raw, [stock], seqs, preview)
    assert plan["chosen"]["sequence"] == target
    assert plan["chosen"]["total_cost"] == 290
    assert plan["medicine_cost"] == 200 and plan["shortfall"] == 0
    assert not set(plan["unresolved"]) & {"cash", "gear_solution"}


def test_catalog_is_stable_bounded_and_declares_omissions():
    raw = state()
    stock = [offer(i, price=i) for i in range(40)]
    raw["resource_state"]["inventory_items"] = [offer(100+i, index=i) for i in range(20)]
    raw["resource_state"]["unequip_candidates"].append(dict(raw["equipped_items"][5], can_unequip=False))
    seqs, info = equipment_sequences(raw, stock)
    again, other = equipment_sequences(raw, list(reversed(stock)))
    assert seqs == again and info == other and len(seqs) == BATCH_LIMIT
    assert info["truncated"] and info["omitted_stock_items"] == 24 and info["omitted_owned_items"] == 8
    assert len(set(seqs)) == len(seqs) and all(len(s) <= 8 for s in seqs)
    assert any(sum(i[0] in ("equip", "buy_equip") for i in s) == 2 for s in seqs)


@pytest.mark.parametrize("failure", ["armor", "weapon", "damage", "health", "level", "durability", "future_gate"])
def test_native_complete_failures_are_not_reconstructed_or_ignored(failure):
    raw, stock = state(), offer()
    seqs = [sequence_for(raw, stock)]
    r = response(raw, seqs, 167, lambda seq: (expanded(raw, stock), 90, ready([failure, "potions"], 0)))
    assert pure_plan(raw, [stock], seqs, r)["chosen"] is None


def test_failed_partial_projection_never_spends_money_or_executes_partial_commands():
    raw, stock = state(), offer()
    seqs = [sequence_for(raw, stock)]
    r = response(raw, seqs, 167)
    r["results"][0].update(gold_cost=90, expanded_commands=expanded(raw, stock)[:2])
    assert pure_plan(raw, [stock], seqs, r)["chosen"] is None


@pytest.mark.parametrize("change", ["count", "ordering", "town", "gold", "durability", "readiness", "first", "cost"])
def test_stale_or_malformed_native_response_fails_loud(change):
    raw, stock = state(), offer()
    seqs = [sequence_for(raw, stock)]
    r = response(raw, seqs, 167, lambda seq: (expanded(raw, stock), 90, ready(["potions"], 0)))
    if change == "count": r["results"] = []
    elif change == "ordering": r["results"][0]["index"] = 3
    elif change == "town": r["source"]["town_seed"] = 9
    elif change == "gold": r["source"]["gold"] = 999
    elif change == "durability": r["source"]["equipment"][6]["durability"] = 99
    elif change == "readiness": r["source"]["readiness"]["armor_class"] = 999
    elif change == "first": r["results"][0]["first_command"] = ("wait",)
    else: r["results"][0]["gold_cost"] = 168
    with pytest.raises(RuntimeError): pure_plan(raw, [stock], seqs, r)


def test_effective_blocking_pool_and_minimum_medicine_remain_authoritative():
    raw, stock = state(), [offer(19, 90), offer(20, 10)]
    seqs = [sequence_for(raw, i) for i in stock]
    r = response(raw, seqs, 167, lambda seq: (expanded(raw, stock[0]), 90, ready(["potions"], 0))
        if seq == seqs[0] else (expanded(raw, stock[1]), 10, ready(["potions"], 0, block=False)))
    plan = pure_plan(raw, stock, seqs, r)
    assert plan["chosen"]["total_cost"] == 290 and plan["chosen"]["projected_block_enabled"] is True
    raw["gold"] = 289
    r = response(raw, seqs, 89, lambda seq: None if seq == seqs[0]
        else (expanded(raw, stock[1]), 10, ready(["potions"], 0, block=False)))
    assert pure_plan(raw, stock, seqs, r)["chosen"]["total_cost"] == 210


def test_missing_medicine_or_full_belt_does_not_create_complete_plan():
    raw, stock = state(), offer()
    seqs = [sequence_for(raw, stock)]
    r = response(raw, seqs, 367, lambda seq: (expanded(raw, stock), 90, ready(["potions"], 0)))
    assert pure_plan(raw, [stock], seqs, r, meds=[])["chosen"] is None
    p = ready(["potions"], 0); p["belt_free_slots"] = 3
    r = response(raw, seqs, 167, lambda seq: (expanded(raw, stock), 90, p))
    assert pure_plan(raw, [stock], seqs, r)["chosen"] is None


def test_failed_real_command_and_legacy_output_and_inputs_rng_are_preserved():
    raw, stock = state(), offer()
    seqs = [sequence_for(raw, stock)]
    r = response(raw, seqs, 167, lambda seq: (expanded(raw, stock), 90, ready(["potions"], 0)))
    original = deepcopy((raw, stock, seqs, r))
    legacy = plan_ordinary_armor_basket(raw, [stock], [], [medicine()])
    rng, nrng = random.getstate(), np.random.get_state()
    assert pure_plan(raw, [stock], seqs, r, failed=[expanded(raw, stock)[0]])["chosen"] is None
    assert original == (raw, stock, seqs, r) and random.getstate() == rng
    after = np.random.get_state(); assert np.array_equal(after[1], nrng[1]) and after[2:] == nrng[2:]
    assert legacy == plan_ordinary_armor_basket(raw, [stock], [], [medicine()])
    assert COMBINATION_POLICY not in RESOURCE_SERVICE_POLICIES


def service_fixture():
    raw, stock = state(), offer()
    service = SustainCombinationService()
    service.phase = "joint_gear"; service.active = True
    service._smith_stock = [stock]; service._healer_stock = [medicine()]
    raw["resource_state"]["town"]["stock"] = [stock]
    env = SimpleNamespace(_raw=raw)
    bridge = SimpleNamespace()
    bridge.project_seen_resource_smith_items = lambda ids: [dict(stock,
        status="projected", projection_origin="known-smith-live-player", town_seed=123,
        town_restock_sequence=1)]
    target = sequence_for(raw, stock)
    expected_commands = expanded(raw, stock)
    def preview(seqs, budget):
        def verdict(seq):
            if seq == target:
                return expected_commands, 90, ready(["potions"], 0)
            if seq == (("equip", 2, *item_identity(stock)), target[1], REPAIR_RETAINED):
                return expected_commands[1:], 0, ready(["potions"], 0)
            if seq == (target[1], REPAIR_RETAINED):
                if item_identity(env._raw["equipped_items"][5]) == item_identity(stock):
                    return expected_commands[2:], 0, ready(["potions"], 0)
            if seq == (REPAIR_RETAINED,):
                if not env._raw["equipped_items"][6].get("present"):
                    return [], 0, ready(["potions"], 0)
            return None
        return response(env._raw, seqs, budget, verdict)
    bridge.preview_resource_equipment_combinations = Mock(side_effect=preview)
    return env, bridge, service, stock, expected_commands


def test_real_dispatch_purchase_pending_equip_unequip_reprojects_and_retains_spent_plan():
    env, bridge, s, stock, commands = service_fixture()
    assert s._command(env, bridge) == commands[0]
    assert s._remaining and s._combination_queries == 1
    s.receipt(commands[0], dict(accepted=True, price=90))
    env._raw["gold"] -= 90
    env._raw["resource_state"]["inventory_items"] = [dict(stock, index=2)]
    assert s._armor_pending == item_identity(stock)
    assert s._command(env, bridge) == commands[1]
    assert s._combination_queries == 2  # Before inherited pending-equip path.
    s.receipt(commands[1], dict(accepted=True))
    old = env._raw["equipped_items"][5]
    env._raw["equipped_items"][5] = deepcopy(stock)
    env._raw["resource_state"]["inventory_items"] = [dict(old, index=2, is_ordinary_armor=True)]
    assert s._command(env, bridge) == commands[2]
    s.receipt(commands[2], dict(accepted=True, inventory_item=dict(env._raw["equipped_items"][6], index=3)))
    env._raw["equipped_items"][6] = dict(present=False)
    env._raw["resource_state"]["readiness"] = ready(["potions"], 0)
    # No shop movement is mocked into a transaction. The next original action
    # is a normal dismiss before going to Pepin, not another armor purchase.
    assert s._command(env, bridge) == ("dismiss",)
    assert s.phase == "minimum_potions" and s._remaining is None
    assert s.purchases == 1 and s.gold_spent == 90 and s.unequips == 1
    assert s.telemetry()["collect_command_window_microsteps"] == 900
    assert s.service_microstep_cap == 3000


def test_pending_auto_equipped_purchase_is_not_equipped_twice():
    env, bridge, s, stock, commands = service_fixture()
    assert s._command(env, bridge) == commands[0]
    s.receipt(commands[0], dict(accepted=True, price=90))
    env._raw["gold"] -= 90
    env._raw["equipped_items"][5] = deepcopy(stock)
    assert s._command(env, bridge) == commands[2]
    assert s._armor_pending is None


def test_after_purchase_invalid_remainder_stops_before_original_pending_equip():
    env, bridge, s, stock, commands = service_fixture()
    assert s._command(env, bridge) == commands[0]
    s.receipt(commands[0], dict(accepted=True, price=90))
    env._raw["gold"] -= 90
    env._raw["resource_state"]["inventory_items"] = [dict(stock, index=2)]
    bridge.preview_resource_equipment_combinations.side_effect = lambda seqs, budget: response(env._raw, seqs, budget)
    before = deepcopy(env._raw)
    assert s._command(env, bridge) == ("finish", "resource_combination_remainder_invalid")
    assert env._raw == before and s.gold_spent == 90 and s._armor_pending == item_identity(stock)
    assert s.telemetry()["combination_history"][-1]["after_real_mutation"] is True


def test_inventory_index_is_live_after_gold_compaction_and_missing_item_cannot_rebuy():
    env, bridge, s, stock, commands = service_fixture()
    s._command(env, bridge); s.receipt(commands[0], dict(accepted=True, price=90))
    env._raw["gold"] -= 90
    env._raw["resource_state"]["inventory_items"] = [dict(stock, index=5)]
    assert s._normalize_remaining(env._raw)[0] == ("equip", 5, *item_identity(stock))
    env._raw["resource_state"]["inventory_items"] = []
    assert s._command(env, bridge) == ("finish", "resource_combination_remainder_invalid")
    assert s.purchases == 1


def test_retained_repairs_expand_again_after_actual_repair_receipt():
    env, bridge, s, stock, commands = service_fixture()
    s._remaining = [REPAIR_RETAINED]; s._invested = True
    worn = env._raw["equipped_items"][5]
    repair = ("repair", 5, *item_identity(worn), worn["durability"], 3)
    env._raw["equipped_items"][6] = dict(present=False)
    bridge.preview_resource_equipment_combinations.side_effect = lambda seqs, budget: response(
        env._raw, seqs, budget, lambda seq: ([repair], 3, ready(["potions"], 0))
        if worn["durability"] == 14 else ([], 0, ready(["potions"], 0)))
    assert s._command(env, bridge) == repair
    s.receipt(repair, dict(accepted=True, price=3))
    assert s._remaining == [REPAIR_RETAINED]  # No stale repair queue reuse.
    env._raw["gold"] -= 3; worn["durability"] = 16
    env._raw["resource_state"]["readiness"] = ready(["potions"], 0)
    assert s._command(env, bridge) == ("dismiss",)
    assert s._remaining is None and s.repairs == 1 and s.repair_gold_spent == 3


def test_explicit_rejected_pending_equip_ends_without_repurchase():
    env, bridge, s, stock, commands = service_fixture()
    s._command(env, bridge); s.receipt(commands[0], dict(accepted=True, price=90))
    env._raw["gold"] -= 90
    env._raw["resource_state"]["inventory_items"] = [dict(stock, index=2)]
    assert s._command(env, bridge) == commands[1]
    s.receipt(commands[1], dict(accepted=False, reason="no_room"))
    assert s._command(env, bridge) == ("finish", "resource_combination_remainder_invalid")
    assert s.purchases == 1 and s.gold_spent == 90


def test_native_query_exception_is_engineering_error_not_no_money():
    env, bridge, s, stock, commands = service_fixture()
    bridge.preview_resource_equipment_combinations.side_effect = ValueError("unknown identity")
    with pytest.raises(ValueError, match="unknown identity"):
        s._command(env, bridge)
    assert s._combination_bridge is None and s.purchases == 0


def test_dialog_and_service_cap_prevent_new_combination_queries():
    env, bridge, s, stock, commands = service_fixture()
    env._raw["resource_state"]["town"]["dialog_active"] = True
    assert s._command(env, bridge) == ("dismiss",)
    assert not bridge.preview_resource_equipment_combinations.called
    s.steps = 3000
    assert s._command(env, bridge) == ("finish", "resource_service_cap")
    assert not bridge.preview_resource_equipment_combinations.called


def reserve_fixture(target, heals=4, cash=77):
    raw = state(cash, heals, ())
    raw["resource_state"]["town"].update(active_vendor="healer", stock=[medicine()])
    s = SustainCombinationService(reserve_belt_target=target)
    s.phase = "minimum_potions"; s._healer_stock = [medicine()]
    return SimpleNamespace(_raw=raw), SimpleNamespace(), s


@pytest.mark.parametrize("target", [0, 6, 8])
def test_reserve_only_uses_actual_post_minimum_surplus_and_empty_belt(target):
    env, bridge, s = reserve_fixture(target)
    command = s._command(env, bridge)
    if target == 0:
        assert command == ("dismiss",) and s.phase == "return"
        return
    assert command == ("buy", "healer", 0, *item_identity(medicine()))
    s.receipt(command, dict(accepted=True, price=50))
    env._raw["gold"] = 27
    env._raw["resource_state"]["readiness"] = ready((), 5)
    assert s._command(env, bridge) == ("dismiss",)
    assert s._reserve_purchases == 1 and s.gold_spent == 50


@pytest.mark.parametrize("heals,free,failures", [(8, 0, ()), (4, 0, ()), (4, 4, ("health",))])
def test_reserve_never_ignores_capacity_or_real_readiness(heals, free, failures):
    env, bridge, s = reserve_fixture(8, heals)
    env._raw["resource_state"]["readiness"].update(belt_free_slots=free, failures=list(failures), ready=not failures)
    assert s._command(env, bridge) == ("dismiss",)
    assert not s._reserve_purchases


@pytest.mark.parametrize("target", [6, 8])
def test_reserve_stops_exactly_at_target_after_actual_belt_and_wallet_changes(target):
    env, bridge, s = reserve_fixture(target, cash=500)
    for heals in range(4, target):
        env._raw["resource_state"]["readiness"] = ready((), heals)
        command = s._command(env, bridge)
        assert command[0:2] == ("buy", "healer")
        s.receipt(command, dict(accepted=True, price=50))
        env._raw["gold"] -= 50
    env._raw["resource_state"]["readiness"] = ready((), target)
    assert s._command(env, bridge) == ("dismiss",)
    assert s._reserve_purchases == target-4 and s.gold_spent == (target-4)*50


@pytest.mark.parametrize("value", [True, -1, 4, 7, None, 8.0])
def test_reserve_config_is_explicit_and_does_not_change_old_factory(value):
    with pytest.raises(ValueError): SustainCombinationService(reserve_belt_target=value)
