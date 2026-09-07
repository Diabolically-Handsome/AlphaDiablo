"""Joint real-price budgets, native projections and explicit service wiring."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym

from diablogym.resource_protocol import ResourceCalibration, ResourceService
from diablogym.resource_sustain import SustainResourceService, plan_basket
from diablogym.options_env import OptionsEnv


def item(slot=6, durability=2, max_durability=6, price=0, **kw):
    return dict(present=True, slot=slot, seed_hi=1, seed_lo=slot,
                create_info=3, base_id=4, durability=durability,
                max_durability=max_durability, price=price, **kw)


def raw(gold=111, failures=("durability",), heals=4):
    equipment = [dict(present=False) for _ in range(7)]
    equipment[6] = item()
    return dict(dungeon_level=0, is_set_level=False, hp=70, max_hp=70,
                player_x=10, player_y=10, gold=gold, equipped_items=equipment,
                resource_state=dict(enabled=True, town_seed=123, town_restock_sequence=1,
                    readiness=dict(ready=not failures, failures=list(failures),
                        required_belt_heals=4, belt_heals=heals,
                        belt_free_slots=8-heals, armor_class=14, damage=6),
                    inventory_items=[], unequip_candidates=[], town=dict(
                        active_vendor="smith", dialog_active=False, stock=[], repair_quotes=[])))


def medicine(price=50):
    return dict(item(slot=0, price=price), vendor="healer", index=0,
                heal_kind=1, can_use=True, can_fit=True)


def chest(price=200):
    return dict(item(price=price, durability=30, max_durability=30),
                vendor="smith", index=0, can_use=True, can_fit=True, meets_armor_gate=True)


def unequip(slot=6, **kw):
    return dict(item(slot=slot), can_fit=True, can_unequip=True,
                projected_readiness=dict(failures=[], armor_class=10, damage=6), **kw)


def test_normal_unequip_can_solve_unrepairable_chest_without_buying_replacement():
    state = raw()
    state["resource_state"]["unequip_candidates"] = [unequip()]
    before = deepcopy(state)
    plan = plan_basket(state, [chest()], [], [medicine()])
    assert plan["chosen"]["kind"] == "unequip_failing_armor"
    assert plan["chosen"]["total_cost"] == 0
    assert plan["chosen"]["commands"] == [("unequip", 6, 1, 6, 3, 4)]
    assert state == before


def test_projection_cannot_remove_armor_if_capacity_or_other_gate_is_lost():
    for changes in ({"can_unequip": False}, {"projected_readiness": {"failures": ["armor"]}},
                    {"projected_readiness": {"failures": ["damage"]}},
                    {"projected_readiness": {"failures": ["health"]}}):
        state = raw()
        candidate = unequip()
        candidate.update(changes)
        state["resource_state"]["unequip_candidates"] = [candidate]
        plan = plan_basket(state, [chest()], [], [medicine()])
        assert plan["chosen"] is None
        assert not plan["zero_improvements"]


def test_cheaper_repair_preserves_minimum_medicine_cash_and_uses_real_quote():
    state = raw(gold=105, heals=2)
    state["equipped_items"][6]["max_durability"] = 30
    quote = item(price=5, max_durability=30, repair_needed_for_gate=True)
    plan = plan_basket(state, [chest(100)], [quote], [medicine()])
    assert plan["chosen"]["kind"] == "retain_and_repair"
    assert plan["chosen"]["total_cost"] == 105
    assert plan["chosen"]["commands"] == [("repair", 6, 1, 6, 3, 4, 2, 5)]
    state["gold"] = 104
    below = plan_basket(state, [chest(100)], [quote], [medicine()])
    assert below["chosen"] is None and below["shortfall"] == 1


def test_stale_repair_identity_or_durability_is_never_a_plan():
    state = raw()
    state["equipped_items"][6]["max_durability"] = 30
    for mutation in ({"seed_lo": 999}, {"durability": 3}):
        quote = item(price=5, max_durability=30, repair_needed_for_gate=True)
        quote.update(mutation)
        assert not plan_basket(state, [], [quote], [medicine()])["alternatives"]


def test_replacement_still_accounts_for_other_equipped_repairs():
    state = raw(gold=254, heals=3)
    state["equipped_items"][0] = item(slot=0, durability=4, max_durability=20)
    quote = item(slot=0, durability=4, max_durability=20, price=5, repair_needed_for_gate=True)
    plan = plan_basket(state, [chest()], [quote], [medicine()])
    assert plan["minimum_observed_total"] == 255
    assert plan["shortfall"] == 1
    assert plan["chosen"] is None


def test_belt_capacity_and_missing_observed_medicine_prices_fail_closed():
    state = raw(gold=999, heals=0)
    assert plan_basket(state, [chest()], [], [])["chosen"] is None
    state["resource_state"]["readiness"]["belt_free_slots"] = 3
    plan = plan_basket(state, [chest()], [], [medicine()])
    assert not plan["belt_has_room"] and plan["chosen"] is None


def test_rejected_mutation_is_not_retried_and_cannot_fake_an_equipped_chest():
    state = raw()
    state["resource_state"]["unequip_candidates"] = [unequip()]
    command = ("unequip", 6, 1, 6, 3, 4)
    plan = plan_basket(state, [chest()], [], [medicine()], failed=[command])
    assert not plan["zero_improvements"] and plan["chosen"] is None
    service = SustainResourceService(active=True, attempted=True, phase="joint_gear")
    service._armor_pending = (8, 8, 8, 8)
    env = SimpleNamespace(_raw=state, _steps=1)
    import pytest
    with pytest.raises(RuntimeError, match="absent"):
        service.command(env, SimpleNamespace())


def test_two_shop_surveys_precede_payment_and_pepin_dialogue_is_closed():
    state = raw()
    service = SustainResourceService(active=True, attempted=True, phase="survey_smith")
    env = SimpleNamespace(_raw=state, _steps=1)
    assert service.command(env, SimpleNamespace()) == ("dismiss",)
    assert service.phase == "survey_healer"
    state["resource_state"]["town"].update(active_vendor="healer", dialog_active=True)
    assert service.command(env, SimpleNamespace()) == ("dismiss",)
    assert service.purchases == service.gold_spent == 0


def test_exact_cap_return_settles_but_late_return_fails():
    for beat, result in ((1500, ("complete",)), (1501, ("finish", "resource_service_cap"))):
        state = raw(failures=())
        state["dungeon_level"] = 1
        service = SustainResourceService(active=True, attempted=True, phase="return")
        env = SimpleNamespace(_raw=state, _steps=beat)
        assert service.command(env, SimpleNamespace(configure_town_service=lambda v: None)) == result


def test_opt_in_changes_cap_not_model_wire_or_farm_denominator():
    for policy, cls, cap in (("legacy-v1", ResourceService, 600), ("sustain-v2", SustainResourceService, 1500)):
        base = SimpleNamespace(observation_space=gym.spaces.Box(-1, 1, shape=(295,)))
        with patch("diablogym.options_env.DiabloGymEnv", return_value=base) as factory:
            options = OptionsEnv(max_steps=6000, resource_protocol="l2-town-v1", resource_service_policy=policy)
        assert type(options.resource_service) is cls
        assert options.resource_service.service_microstep_cap == cap
        assert options.max_steps == 6000 and options.farm_scene_cap == 3600
        assert "resource_service_policy" not in factory.call_args.kwargs
        assert not hasattr(base, "_resource_calibration")


def test_policy_and_calibration_have_separate_explicit_identities():
    import pytest
    for args in ({}, {"resource_protocol": "l2-town-v1", "resource_purchase_mode": "armor"}):
        with pytest.raises(ValueError, match="requires"):
            OptionsEnv(resource_service_policy="sustain-v2", **args)
    calibration = ResourceCalibration(calibration_id="diagnostic", service_microstep_cap=1700)
    service = SustainResourceService(calibration=calibration)
    assert service.service_microstep_cap == 1700
    assert service.telemetry()["calibration"]["formal_metric_eligible"] is False
