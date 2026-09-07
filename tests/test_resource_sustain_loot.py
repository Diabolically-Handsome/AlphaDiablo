"""Pure service tests: no package __init__, native import, model or game."""
from copy import deepcopy
import importlib
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PACKAGE = "_r21_loot_service_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/diablogym")]
    sys.modules[PACKAGE] = package
module = importlib.import_module(PACKAGE + ".resource_sustain_loot")
Service = module.SustainLootService
Original = module.SustainCompletionService


def item(number=1, **extra):
    value = dict(empty=False, index=0, seed_hi=7, seed_lo=number, create_info=257,
                 base_id=3, sellable=True, is_quest=False, upgrade=False,
                 reserved_upgrade=False, sell_price=5, price=5, can_fit=True,
                 active_id=number, x=10, y=10, durability=20, max_durability=20)
    value.update(extra)
    return value


def identity(i):
    return tuple(i[k] for k in ("seed_hi", "seed_lo", "create_info", "base_id"))


def raw(*, gold=100, depth=1, failures=("armor",), owned=(), loot=(), **extra):
    value = dict(gold=gold, dungeon_level=depth, is_set_level=False, dead=False,
                 player_mode=0, player_x=10, player_y=10, hp=70, max_hp=70,
                 xp=100, char_level=3, equipped_items=[],
                 monsters=[dict(hp=10, type=1)],
                 resource_state=dict(enabled=True, ordinary_armor_scope=True,
                     preserve_equipment_readiness=True, loot_economy=True,
                     town_seed=23, town_restock_sequence=1,
                     readiness=dict(ready=not failures, failures=list(failures),
                         hp_fixed=4480, max_hp_fixed=4480, hp=70, max_hp=70,
                         clvl=3, belt_heals=4, required_belt_heals=4, belt_free_slots=4),
                     inventory_state=dict(items=list(owned), inventory_count=len(owned),
                         grid=[0]*40, free_cells=40, held_empty=True),
                     loot_items=list(loot), gold_items=[], inventory_items=[],
                     town=dict(active_vendor="smith", dialog_active=False, stock=[],
                               sell_quotes=[], repair_quotes=[])))
    value.update(extra)
    return value


def env(value, clock):
    return SimpleNamespace(_raw=value, _steps=clock,
                           _resource_actual_microsteps=clock,
                           controller_action_context=lambda: ([0]*15, None))


def start(value=None):
    service = Service()
    value = raw() if value is None else value
    assert service.maybe_start(value, 3600, farm_scene_steps=3600, cleared=False)
    return service, value


def request(service, before, command, clock=3601):
    with patch.object(Original, "command", return_value=command):
        return service.command(env(before, clock), Mock())


def transact(service, before, after, command, receipt, clock=3601):
    request(service, before, command, clock)
    service.observe(after, clock+1)
    service.receipt(command, receipt)
    service.record_steps(clock+1)


def native_receipt(i, before, after, *, accepted=True, received=0, **extra):
    value = dict(zip(("seed_hi", "seed_lo", "create_info", "base_id"), identity(i)))
    value.update(accepted=accepted, received=received, price=0,
                 gold_before=before, gold_after=after, index=i["index"], reason="native")
    value.update(extra)
    return value


def return_trip(service, value=None, clock=5000):
    value = raw(gold=service._latest["gold"]) if value is None else value
    town = deepcopy(value); town["dungeon_level"] = 0
    service.observe(town, clock-1)
    service.phase = "return"
    command = service.command(env(value, clock), Mock())
    assert command == ("complete",)
    service.receipt(command, dict(accepted=False, price=0, reason="complete"))
    service.record_steps(clock)
    return value


class TriggerTests(unittest.TestCase):
    def test_remaining_memory_allows_search_without_claiming_income_or_querying(self):
        s, _ = start(raw(loot=[item(x=30, y=30, can_fit=False)]))
        a = raw(monsters=[])
        e = env(a, 5000); e.controller_action_context = Mock(side_effect=AssertionError("extra query"))
        self.assertEqual(s._finish_or_farm(e, "resource_unreachable"), ("complete",))
        self.assertEqual(s.reason, "growth_remaining")
        self.assertEqual(s.gold_collected, 0)
        self.assertFalse(s._remaining_resource_evidence["memory_is_income_or_route_proof"])

    def test_second_trip_can_move_previously_known_but_now_fitting_loot(self):
        known = item(can_fit=False)
        s, _ = start(raw(loot=[known]))
        a = raw(loot=[known]); return_trip(s, a)
        known["can_fit"] = True
        self.assertTrue(s.maybe_start(a, 5001, farm_scene_steps=5001, cleared=False))
        self.assertIn("known_remaining_resource_currently_collectible", s._trigger_changes)
        self.assertNotIn("new_observed_saleable_loot", s._trigger_changes)

    def test_current_gold_reopens_only_present_collectible_not_offscreen_or_gone(self):
        gold = dict(active_id=8, x=10, y=10, seed_hi=3, seed_lo=8, create_info=257, base_id=0, value=12)
        offscreen = dict(gold, seed_lo=9, active_id=9, x=30)
        gone = dict(gold, seed_lo=10, active_id=10)
        s, _ = start()
        a = raw(); a["resource_state"]["gold_items"] = [gold, offscreen, gone]
        s.observe(a, 3700)
        for i in [gold, offscreen, gone]: s.gold_memory.mark_attempted(i, "route_unavailable", 3700)
        empty = raw(); s.gold_memory.mark_gone(gone, empty, 3701)
        a["resource_state"]["gold_items"] = [gold]
        return_trip(s, a)
        self.assertTrue(s.maybe_start(a, 5001, farm_scene_steps=5001, cleared=False))
        self.assertFalse(s.gold_memory._items[identity(gold)]["attempted"])
        self.assertTrue(s.gold_memory._items[identity(offscreen)]["attempted"])
        self.assertTrue(s.gold_memory._items[identity(gone)]["gone"])
        self.assertEqual(len(s._reopened_gold), 1)

    def test_stale_or_current_full_inventory_remaining_resource_cannot_restart(self):
        for condition in ["offscreen", "no_room", "held"]:
            with self.subTest(condition=condition):
                known = item(can_fit=False)
                s, _ = start(raw(loot=[known])); a = raw(loot=[known]); return_trip(s, a)
                if condition == "offscreen": a["resource_state"]["loot_items"] = []
                elif condition == "held":
                    known["can_fit"] = True
                    a["resource_state"]["inventory_state"]["held_empty"] = False
                self.assertFalse(s.maybe_start(a, 5001, farm_scene_steps=5001, cleared=True))

    def test_gold_without_known_empty_cell_does_not_claim_stacking_capacity(self):
        gold = dict(active_id=8, x=10, y=10, seed_hi=3, seed_lo=8, create_info=257, base_id=0, value=12)
        s, _ = start(); a = raw(); a["resource_state"]["gold_items"] = [gold]
        return_trip(s, a)
        a["resource_state"]["inventory_state"]["free_cells"] = 0
        self.assertFalse(s.maybe_start(a, 5001, farm_scene_steps=5001, cleared=True))

    def test_remaining_memory_cannot_authorize_third_town_trip(self):
        s, _ = start(raw(loot=[item(x=30)])); s.trip_count = 2
        a = raw(monsters=[]); e = env(a, 7000)
        self.assertEqual(s._finish_or_farm(e, "resource_unreachable"),
                         ("finish", "resource_trip_limit_remaining_resources"))
        self.assertEqual(s.reason, "resource_trip_limit_remaining_resources")

    def test_first_trigger_and_native_deficit(self):
        s = Service()
        self.assertFalse(s.maybe_start(raw(), 3599, farm_scene_steps=3599, cleared=False))
        self.assertFalse(s.maybe_start(raw(failures=()), 3600, farm_scene_steps=3600, cleared=False))
        self.assertTrue(s.maybe_start(raw(), 3600, farm_scene_steps=3600, cleared=False))
        self.assertEqual(s.trigger, "farm_cap")
        self.assertFalse(s.maybe_start(raw(), 3601, farm_scene_steps=4000, cleared=True))

    def test_cleared_can_start_without_saturated_farm(self):
        s = Service()
        self.assertTrue(s.maybe_start(raw(), 90, farm_scene_steps=90, cleared=True))
        self.assertEqual(s.trigger, "cleared")

    def test_bad_scope_scene_or_nonidle_never_starts(self):
        for change in [dict(depth=0), dict(depth=2), dict(is_set_level=True),
                       dict(player_mode=1), dict(dead=True)]:
            with self.subTest(change=change):
                self.assertFalse(Service().maybe_start(raw(**change), 4000,
                    farm_scene_steps=4000, cleared=True))
        with self.assertRaises(ValueError): Service(mode="potions")
        with self.assertRaises(ValueError): Service(calibration=object())
        bad = raw(); bad["resource_state"]["loot_economy"] = False
        with self.assertRaisesRegex(RuntimeError, "loot_economy"):
            Service().maybe_start(bad, 4000, farm_scene_steps=4000, cleared=True)

    def test_no_direct_start_or_idle_retrigger(self):
        s, _ = start()
        baseline = return_trip(s)
        for tick in [5000, 5001, 8000]:
            moved = deepcopy(baseline); moved["player_x"] = 99
            self.assertFalse(s.maybe_start(moved, tick, farm_scene_steps=9000, cleared=True))
        with self.assertRaisesRegex(RuntimeError, "maybe_start"):
            s.start(baseline, 8001, "farm_cap")

    def test_second_trip_requires_postreturn_change_not_first_trip_cash(self):
        s, a = start()
        b = raw(gold=120)
        transact(s, a, b, ("gold", 1, 7, 1, 257, 0), dict(accepted=True, price=0))
        baseline = return_trip(s, b)
        self.assertFalse(s.maybe_start(baseline, 5001, farm_scene_steps=9000, cleared=True))
        b = deepcopy(baseline); b["gold"] += 1
        self.assertTrue(s.maybe_start(b, 5002, farm_scene_steps=9000, cleared=False))
        self.assertEqual(s.trip_count, 2)
        self.assertEqual(s._trigger_changes, ["gold"])
        self.assertEqual(s.start_steps, 5002)
        self.assertEqual(s.command_microstep_deadline, 5902)
        self.assertEqual(s.steps, 0)
        return_trip(s, b, 6500)
        b["gold"] += 100
        self.assertFalse(s.maybe_start(b, 6501, farm_scene_steps=9999, cleared=True))
        self.assertEqual(s.telemetry()["last_start_denial"], "trip_limit")

    def test_real_health_consumption_and_growth_can_trigger_second(self):
        for change, expected in [("health", "actual_health_consumption"),
                                 ("xp", "experience"), ("loot", "new_observed_saleable_loot"),
                                 ("belt", "belt_heals")]:
            with self.subTest(change=change):
                s, _ = start(); b = return_trip(s)
                if change == "health":
                    b["resource_state"]["readiness"].update(failures=["health"], hp_fixed=3400)
                elif change == "xp": b["xp"] += 1
                elif change == "loot": b["resource_state"]["loot_items"] = [item()]
                else: b["resource_state"]["readiness"]["belt_heals"] = 3
                self.assertTrue(s.maybe_start(b, 5100, farm_scene_steps=4000, cleared=False))
                self.assertIn(expected, s._trigger_changes)

    def test_healing_or_health_drop_without_native_health_failure_does_not_trigger(self):
        for amount in [4600, 4400]:
            s, _ = start(); b = return_trip(s)
            b["resource_state"]["readiness"]["hp_fixed"] = amount
            self.assertFalse(s.maybe_start(b, 5100, farm_scene_steps=5000, cleared=True))

    def test_quote_failure_navigation_and_plan_state_reset_per_trip(self):
        s, _ = start(); b = return_trip(s)
        s._smith_stock = [{"old": True}]; s._failed_commands.add(("old",))
        s._shop_failures.add(("old",)); s._last_plan = {"old": True}
        s._town_sequence = (1, 1); s._loot_attempted.add((7, 1, 257, 3))
        b["gold"] += 5
        self.assertTrue(s.maybe_start(b, 5100, farm_scene_steps=5000, cleared=False))
        self.assertEqual(s._smith_stock, []); self.assertEqual(s._failed_commands, set())
        self.assertEqual(s._shop_failures, set()); self.assertIsNone(s._last_plan)
        self.assertIsNone(s._town_sequence); self.assertEqual(s._loot_attempted, set())


class CommandTests(unittest.TestCase):
    def test_memory_detached_rng_unchanged_and_cross_window_persistent(self):
        a = raw(loot=[item()]); original = deepcopy(a); rng = random.getstate()
        s = Service(); s.observe(a, 0); self.assertEqual(a, original)
        a["resource_state"]["loot_items"][0]["sell_price"] = 999
        self.assertEqual(s._loot_memory[identity(item())]["sell_price"], 5)
        s.observe(raw(depth=0), 1); self.assertEqual(len(s._loot_memory), 1)
        self.assertEqual(random.getstate(), rng)
        with self.assertRaisesRegex(RuntimeError, "backwards"): s.observe(raw(), 0)

    def test_current_safe_loot_pickup_command_and_no_extra_bridge_calls(self):
        s, a = start(raw(loot=[item()])); bridge = Mock()
        command = s.command(env(a, 3601), bridge)
        self.assertEqual(command, ("loot", 1, 10, 10, 7, 1, 257, 3))
        self.assertEqual(bridge.mock_calls, [])

    def test_quest_upgrade_reserved_full_inventory_are_not_selected(self):
        for flag in [dict(is_quest=True), dict(upgrade=True),
                     dict(reserved_upgrade=True), dict(can_fit=False), dict(sellable=False)]:
            with self.subTest(flag=flag):
                s, a = start(raw(loot=[item(**flag)]))
                self.assertIsNone(s._collect_loot(env(a, 3601)))

    def test_loot_target_locked_despite_new_nearer_item(self):
        far = item(x=20, y=20)
        s, a = start(raw(loot=[far])); s._walk = Mock(return_value=("walk", 11, 11))
        self.assertEqual(s._collect_loot(env(a, 3601)), ("walk", 11, 11))
        b = raw(loot=[far, item(2, x=11, y=10)], player_x=11, player_y=11)
        s.observe(b, 3602); s._walk.return_value = ("walk", 12, 12)
        self.assertEqual(s._collect_loot(env(b, 3602)), ("walk", 12, 12))
        self.assertEqual(identity(s._loot_target), identity(far))

    def test_absence_only_disqualifies_current_selected_target_and_no_cash_credit(self):
        s, _ = start(raw(loot=[item()])); s.observe(raw(), 3601)
        self.assertIsNone(s._collect_loot(env(raw(), 3601)))
        self.assertEqual(s.loot_collected, 0); self.assertEqual(s.gold_sold, 0)
        self.assertEqual(s.loot_failures[-1]["reason"], "no_current_safe_loot_at_target")

    def test_loot_respects_existing_bounded_recovery_retry_guard(self):
        s, a = start(raw(loot=[item(x=20, y=20)]))
        def bounded_recovery(*args):
            s._gold_navigation_seen.clear()
            return ("attack", 11, 10)
        s._walk = Mock(side_effect=bounded_recovery)
        for clock in [3601, 3602]:
            self.assertEqual(s._collect_loot(env(a, clock)), ("attack", 11, 10))
        self.assertEqual(s.loot_failures, [])

    def test_existing_gold_target_has_priority_and_collect_cap_inherited(self):
        s, a = start(raw(loot=[item()])); s._gold_target = {"locked": True}
        self.assertIsNone(s._collect_loot(env(a, 3601)))
        s.steps = 900
        with patch.object(Original, "_command", return_value=("parent",)) as original:
            self.assertEqual(s._command(env(a, 4500), Mock()), ("parent",))
            original.assert_called_once()
        self.assertEqual(s.collect_microstep_cap, 900)
        self.assertEqual(s.service_microstep_cap, 3000)

    def test_sell_before_original_shopping_and_real_live_quote(self):
        owned = item(); a = raw(depth=0, owned=[owned])
        a["resource_state"]["town"]["sell_quotes"] = [dict(owned, vendor="smith", price=5)]
        s, _ = start(); s.phase = "outbound"; s.observe(a, 3700); s._visit = Mock(return_value=None)
        self.assertEqual(s._command(env(a, 3700), Mock()), ("sell", "smith", 0, 7, 1, 257, 3, 5))
        self.assertEqual(s.phase, "sell_idle")
        b = raw(depth=0); s.observe(b, 3701)
        with patch.object(Original, "_command", return_value=("original-shopping",)):
            self.assertEqual(s._command(env(b, 3701), Mock()), ("original-shopping",))
            self.assertEqual(s.phase, "survey_smith")

    def test_equipped_protected_pending_or_unquoted_inventory_never_sold(self):
        for protection in ["equipped", "reserved", "pending", "quest", "unquoted"]:
            with self.subTest(protection=protection):
                owned = item(); a = raw(depth=0, owned=[owned])
                a["resource_state"]["town"]["sell_quotes"] = [dict(owned, vendor="smith")]
                s, _ = start(); s.phase = "sell_idle"; s._visit = Mock(return_value=None)
                if protection == "equipped": a["equipped_items"] = [dict(owned, present=True)]
                if protection == "reserved": owned["reserved_upgrade"] = True
                if protection == "pending": s._purchase_pending.add(identity(owned))
                if protection == "quest": owned["is_quest"] = True
                if protection == "unquoted": a["resource_state"]["town"]["sell_quotes"] = []
                s.observe(a, 3700)
                with patch.object(Original, "_command", return_value=("original",)):
                    self.assertEqual(s._command(env(a, 3700), Mock()), ("original",))

    def test_live_quote_index_or_town_generation_drift_refused(self):
        i = item(); a = raw(depth=0, owned=[i]); a["resource_state"]["town"]["sell_quotes"] = [dict(i, vendor="smith", index=2)]
        s, _ = start(); s.phase = "sell_idle"; s._visit = Mock(return_value=None); s.observe(a, 3700)
        with self.assertRaisesRegex(RuntimeError, "index"): s._command(env(a, 3700), Mock())
        a["resource_state"]["town_restock_sequence"] = 2; s.observe(a, 3701)
        with self.assertRaisesRegex(RuntimeError, "generation"): s._command(env(a, 3701), Mock())


class ReceiptTests(unittest.TestCase):
    def test_sale_cash_is_separate_and_cannot_be_counted_again_as_gold(self):
        i = item(); s, _ = start(); a = raw(depth=0, owned=[i]); b = raw(depth=0, gold=105)
        c = ("sell", "smith", 0, *identity(i), 5)
        r = native_receipt(i, 100, 105, received=5, quoted_price=5)
        transact(s, a, b, c, r)
        self.assertEqual((s.gold_sold, s.gold_collected, s.gold_spent), (5, 0, 0))
        with patch.object(Original, "_command", return_value=("parent",)):
            s._command(env(b, 3700), Mock())
        self.assertEqual(s.gold_collected, 0)
        return_trip(s, raw(gold=105))
        self.assertEqual(s.telemetry()["trips"][0]["cash_residual"], 0)
        self.assertEqual(s.telemetry()["cumulative"]["gold_sold"], 5)

    def test_sale_received_wallet_identity_and_failed_no_effect_are_strict(self):
        for mutation in ["amount", "wallet", "identity", "still_owned", "failed_cash", "paid"]:
            with self.subTest(mutation=mutation):
                i = item(); s, _ = start(); a = raw(depth=0, owned=[i]); b = raw(depth=0, gold=105)
                c = ("sell", "smith", 0, *identity(i), 5)
                r = native_receipt(i, 100, 105, received=5, quoted_price=5)
                if mutation == "amount": r["received"] = 6
                if mutation == "wallet": r["gold_before"] = 99
                if mutation == "identity": r["seed_lo"] = 2
                if mutation == "still_owned": b["resource_state"]["inventory_state"]["items"] = [i]
                if mutation == "failed_cash": r.update(accepted=False, received=0)
                if mutation == "paid": r["price"] = 5
                with self.assertRaises(RuntimeError): transact(s, a, b, c, r)

    def test_failed_sale_keeps_item_and_records_reason_without_retry(self):
        i = item(); s, _ = start(); a = raw(depth=0, owned=[i]); c = ("sell", "smith", 0, *identity(i), 5)
        r = native_receipt(i, 100, 100, accepted=False, quoted_price=5, reason="no_room")
        transact(s, a, deepcopy(a), c, r)
        self.assertEqual(s.sales, 0); self.assertEqual(s.sale_failures[-1]["reason"], "no_room")
        self.assertIn(identity(i), s._sale_attempted)

    def test_pickup_uses_actual_pregen_cleared_identity_without_cash(self):
        source = item(create_info=257); actual = item(create_info=1)
        s, a = start(raw(loot=[source])); b = raw(owned=[actual])
        c = ("loot", 1, 10, 10, *identity(source))
        r = native_receipt(source, 100, 100, inventory_item=actual)
        transact(s, a, b, c, r)
        self.assertEqual(s.loot_collected, 1); self.assertEqual(s.gold_collected, 0)
        self.assertNotIn(identity(source), s._loot_memory)

    def test_pickup_accept_without_actual_item_or_with_cash_rejected(self):
        for mutation in ["absent", "cash", "preexisting"]:
            source = item(); actual = item(create_info=1)
            s, a = start(raw(loot=[source])); b = raw(owned=[actual])
            r = native_receipt(source, 100, 100, inventory_item=actual)
            if mutation == "absent": b["resource_state"]["inventory_state"]["items"] = []
            if mutation == "cash": b["gold"] = 105; r["gold_after"] = 105
            if mutation == "preexisting": a["resource_state"]["inventory_state"]["items"] = [actual]
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                transact(s, a, b, ("loot", 1, 10, 10, *identity(source)), r)

    def test_purchase_and_repair_use_receipt_expenditure(self):
        s, a = start(); b = raw(gold=80)
        transact(s, a, b, ("buy", "smith", 0, 7, 2, 1030, 55), dict(accepted=True, price=20))
        self.assertIn((7, 2, 1030, 55), s._purchase_pending)
        c = raw(gold=77)
        transact(s, b, c, ("repair", 4, 7, 3, 0, 1, 14, 3), dict(accepted=True, price=3), 3603)
        self.assertEqual((s.gold_spent, s.repair_gold_spent, s.purchases, s.repairs), (23, 3, 1, 1))

    def test_gold_receipts_positive_and_rejection_no_credit(self):
        for accepted, gold in [(True, 99), (False, 105)]:
            s, a = start()
            with self.subTest(accepted=accepted), self.assertRaises(RuntimeError):
                transact(s, a, raw(gold=gold), ("gold", 1, 7, 1, 257, 0), dict(accepted=accepted, price=0))

    def test_accepted_queued_gold_without_cash_is_not_fake_income(self):
        s, a = start()
        transact(s, a, deepcopy(a), ("gold", 1, 7, 1, 257, 0), dict(accepted=True, price=0))
        self.assertEqual(s.gold_collected, 0)

    def test_former_upgrade_is_not_permanently_reserved(self):
        i = item(upgrade=True, reserved_upgrade=True)
        s = Service(); s.observe(raw(owned=[i]), 1)
        self.assertIn(identity(i), s._protected)
        i.update(upgrade=False, reserved_upgrade=False)
        s.observe(raw(owned=[i]), 2)
        self.assertNotIn(identity(i), s._protected)
        s._purchase_pending.add(identity(i)); s.observe(raw(owned=[i]), 3)
        self.assertIn(identity(i), s._purchase_pending)

    def test_synchronous_loot_then_death_does_not_turn_success_into_engine_error(self):
        source, actual = item(), item(create_info=1)
        s, a = start(raw(loot=[source])); b = raw(gold=50, dead=True)
        c = ("loot", 1, 10, 10, *identity(source))
        r = native_receipt(source, 100, 100, inventory_item=actual)
        transact(s, a, b, c, r)
        self.assertEqual(s.loot_collected, 1); self.assertEqual(s.gold_collected, 0)
        self.assertEqual(s.death_cash_change, -50)
        s.finish_episode(b, 3602, "death")
        report = s.telemetry()["trips"][0]
        self.assertEqual(report["cash_residual"], 0)
        self.assertEqual(report["reason"], "death")
        self.assertTrue(report["gold_collected_complete"])

    def test_queued_gold_and_same_tick_death_amount_is_unknown_not_invented(self):
        s, a = start(); b = raw(gold=55, dead=True)
        transact(s, a, b, ("gold", 1, 7, 1, 257, 0), dict(accepted=True, price=0))
        s.finish_episode(b, 3602, "death")
        report = s.telemetry()["trips"][0]
        self.assertEqual(s.gold_collected, 0)
        self.assertFalse(report["gold_collected_complete"])
        self.assertEqual(report["cash_reconciliation"], "death_pickup_amount_unknown")
        self.assertEqual(report["cash_residual"], -45)
        self.assertEqual(report["terminal_cash_events"][0]["net_observed_change"], -45)

    def test_finish_episode_archives_actual_timeout_once_and_preserves_raw(self):
        s, a = start(); before = deepcopy(a)
        s.finish_episode(a, 12000, "episode_truncated")
        s.finish_episode(a, 12000, "episode_truncated")
        self.assertEqual(a, before); self.assertEqual(len(s.telemetry()["trips"]), 1)
        self.assertEqual(s.reason, "episode_truncated")
        self.assertFalse(s.active)
        self.assertIsNone(s._return_baseline)
        with self.assertRaises(ValueError): s.finish_episode(a, 12000, "")

    def test_missing_mismatched_duplicate_receipt_or_concurrent_command_refused(self):
        s, a = start(); c = ("walk", 11, 10)
        with self.assertRaises(RuntimeError): s.receipt(c, dict(accepted=True))
        request(s, a, c)
        with self.assertRaises(RuntimeError): s.command(env(a, 3601), Mock())
        with self.assertRaises(RuntimeError): s.receipt(("walk", 9, 10), dict(accepted=True))
        s.receipt(c, dict(accepted=True))
        with self.assertRaises(RuntimeError): s.receipt(c, dict(accepted=True))

    def test_cumulative_counters_are_detached_and_not_doubled_after_second_trip(self):
        s, a = start(); b = raw(gold=110)
        transact(s, a, b, ("gold", 1, 7, 1, 257, 0), dict(accepted=True, price=0))
        return_trip(s, b); b["gold"] = 115
        self.assertTrue(s.maybe_start(b, 5100, farm_scene_steps=5100, cleared=False))
        c = raw(gold=122)
        transact(s, b, c, ("gold", 1, 7, 2, 257, 0), dict(accepted=True, price=0), 5101)
        self.assertEqual(s.telemetry()["cumulative"]["gold_collected"], 17)
        return_trip(s, c, 6500)
        info = s.telemetry(); self.assertEqual(info["cumulative"]["gold_collected"], 17)
        self.assertEqual([t["start_gold"] for t in info["trips"]], [100, 115])
        info["trips"][0]["gold_collected"] = 999
        self.assertEqual(s.telemetry()["trips"][0]["gold_collected"], 10)

    def test_legacy_service_policy_caps_and_one_trip_behavior_unchanged(self):
        old = Original(); old.start(raw(), 1, "farm_cap"); old.start(raw(gold=999), 5, "again")
        self.assertEqual(old.start_steps, 1)
        old_info = old.telemetry()
        self.assertEqual(old_info["policy"], "sustain-v6")
        self.assertNotIn("gold_sold", old_info); self.assertNotIn("trips", old_info)
        self.assertEqual((old.collect_microstep_cap, old.service_microstep_cap), (900, 3000))


if __name__ == "__main__":
    unittest.main()
