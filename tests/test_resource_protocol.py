"""Hard-gate resource protocol: censoring, canonical receipts and service accounting.

These tests use deterministic raw-state doubles. Native shopping / stairs must
also pass the separate rebuilt-engine smoke before the 48-seed effects pilot.
"""
from __future__ import annotations
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import eval_contract
import probe_resource_protocol as pilot

spec = importlib.util.spec_from_file_location(
    "resource_protocol_under_test", ROOT / "python/diablogym/resource_protocol.py")
resource = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = resource
spec.loader.exec_module(resource)


def raw(depth=1, ready=False, sequence=0, accepted=False):
    return {"dungeon_level": depth, "is_set_level": False,
            "player_x": 10, "player_y": 10, "hp": 70, "max_hp": 70,
            "gold": 100, "belt_heals": 4, "monster_kill_total": 0,
            "resource_state": {"enabled": True, "protocol": "l2-town-v1",
                               "readiness": {"ready": ready,
                                             "failures": [] if ready else ["armor"]},
                               "transition": {"sequence": sequence,
                                              "accepted": accepted,
                                              "pretransition_ready": ready}}}


class ReadinessTests(unittest.TestCase):
    def test_native_receipt_is_authoritative_even_if_python_panel_disagrees(self):
        state = raw(ready=False)
        state.update(char_level=99, armor_class=999, item_max_damage=999)
        self.assertFalse(resource.native_readiness(state)["ready"])
        self.assertFalse(resource.progression_allowed(state))

    def test_receipt_missing_or_nonboolean_is_a_hard_error(self):
        for state in ({}, {"resource_state": {}}, raw()):
            state = deepcopy(state)
            if "readiness" in state.get("resource_state", {}):
                state["resource_state"]["readiness"]["ready"] = 1
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                resource.native_readiness(state)

    def test_ready_main_descent_allowed_but_l3_not_assumed_calibrated(self):
        self.assertTrue(resource.progression_allowed(raw(1, True)))
        self.assertFalse(resource.progression_allowed(raw(2, True)))

    def test_quest_and_return_are_not_mistaken_for_deeper_descent(self):
        state = raw(ready=False)
        state["is_set_level"] = True
        self.assertTrue(resource.progression_allowed(state))
        state["is_set_level"] = False
        state["progression_targets"] = [{"kind": "diablo_switch"}]
        self.assertTrue(resource.progression_allowed(state))

    def test_exhausted_cannot_force_an_unready_descent_and_cap_routes_to_service(self):
        import numpy as np
        from diablogym import bridge
        from diablogym.options_env import OptionsEnv, FARM, DIVE, RESUPPLY
        state = raw()
        state["triggers"] = [{"msg": bridge.WM_DIABNEXTLVL, "x": 20, "y": 20}]
        state["monsters"] = [{"hp": 5, "type": 1}]
        env = OptionsEnv.__new__(OptionsEnv)
        env.env = SimpleNamespace(_raw=state, _steps=0,
                                  _ensure_active=lambda **kwargs: None)
        env._last_base_obs = np.zeros(295, dtype=np.float32)
        env._sync_farm_scene = lambda scene: None
        env._controller_action_context = lambda: (np.ones(15, dtype=bool), None)
        env.exhausted = True
        env.resource_protocol = "l2-town-v1"
        env.resource_service = resource.ResourceService()
        env.farm_scene_steps = 140
        env.dive_live_sovereignty = True
        env._win = {"opt": DIVE}
        masks = env.action_masks()
        self.assertFalse(masks[DIVE])
        self.assertTrue(masks[FARM])
        env.farm_scene_steps = 3600
        masks = env.action_masks()
        self.assertEqual(list(masks), [False, False, True])
        self.assertEqual(env.resource_option_choice(masks), RESUPPLY)

    def test_paid_repair_recipe_is_explicit_only_in_full_and_off_stays_unchanged(self):
        off = eval_contract.make_protocol([2114001])
        self.assertNotIn("resource_service_recipe", off)
        on = eval_contract.make_protocol([2114001], r16_environment={
            "resource_protocol": "l2-town-v1"})
        recipe = on["resource_service_recipe"]
        self.assertEqual(recipe["version"], "l2-town-paid-repair-v2")
        self.assertEqual(recipe["service_microstep_cap"], 600)
        stages = recipe["stages"]
        self.assertLess(stages.index("armor_if_needed"), stages.index("normal_paid_repair"))
        self.assertLess(stages.index("normal_paid_repair"), stages.index("native_heal"))
        self.assertLess(stages.index("native_heal"), stages.index("potions_to_capacity"))
        for mode in ("none", "heal", "potions", "armor"):
            self.assertNotIn("normal_paid_repair",
                eval_contract.resource_service_recipe("l2-town-v1", mode)["stages"])

    def test_protocol_and_ablation_are_explicit_in_eval_identity(self):
        for arm in pilot.ARMS:
            payload = {"resource_protocol": "l2-town-v1"}
            if arm != "full":
                payload["resource_purchase_mode"] = arm
            self.assertEqual(eval_contract.validate_r16_environment(payload), payload)
        for invalid in ({"resource_protocol": "off"},
                        {"resource_purchase_mode": "armor"},
                        {"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full"}):
            with self.subTest(invalid=invalid), self.assertRaises(eval_contract.EvalContractError):
                eval_contract.validate_r16_environment(invalid)


class ExposureTests(unittest.TestCase):
    def test_alive_nonarrival_is_not_success(self):
        result = pilot.followup_outcome(None, 6000, died=False)
        self.assertFalse(result["joint_success"])
        self.assertIsNone(result["alive_at_followup"])

    def test_alive_early_stop_is_censored_never_a_success(self):
        result = pilot.followup_outcome(5800, 6000, died=False)
        self.assertTrue(result["followup_censored"])
        self.assertIsNone(result["alive_at_followup"])
        self.assertFalse(result["joint_success"])

    def test_full_exposure_and_death_at_boundary(self):
        self.assertTrue(pilot.followup_outcome(6000, 7800, died=False)["joint_success"])
        self.assertFalse(pilot.followup_outcome(6000, 7800, died=True,
                                               death_beat=7800)["joint_success"])
        self.assertTrue(pilot.followup_outcome(100, 2000, died=True,
                                              death_beat=2000)["joint_success"])

    def test_native_arrival_extends_deadline_without_town_reset(self):
        env = SimpleNamespace(max_steps=4)
        ledger = pilot.ExposureLedger(arrival_budget=4, followup=3)
        ledger(env, raw(0), 1)
        ledger(env, raw(1), 2)
        self.assertIsNone(ledger.first_arrival)
        ledger(env, raw(2, True, 1, True), 3)
        self.assertEqual(env.max_steps, 6)
        for beat in (4, 5, 6):
            ledger(env, raw(2, True, 1, True), beat)
        result = ledger.finish(raw(2, True), 6)
        self.assertTrue(result["joint_success"])
        self.assertEqual(result["first_l2_beat"], 3)
        self.assertEqual(result["l2_microsteps"], 3)
        self.assertEqual(sum(result["beats_by_depth"].values()), 6)
        self.assertEqual(len(result["transitions"]), 1)

    def test_mask_bypass_is_caught_from_pretransition_receipt(self):
        ledger = pilot.ExposureLedger()
        with self.assertRaisesRegex(RuntimeError, "pre-transition"):
            ledger(SimpleNamespace(max_steps=6000), raw(2, False, 1, True), 1)

    def test_microtick_gaps_and_python_native_disagreement_fail_closed(self):
        ledger = pilot.ExposureLedger()
        env = SimpleNamespace(max_steps=6000)
        with self.assertRaisesRegex(RuntimeError, "gap/repeat"):
            ledger(env, raw(), 2)
        ledger(env, raw(), 1)
        with self.assertRaisesRegex(RuntimeError, "accounting"):
            ledger.finish(raw(), 2)

    def test_shop_and_phase_changes_are_logged_without_extra_observations(self):
        env = SimpleNamespace(max_steps=6000)
        ledger = pilot.ExposureLedger()
        phase = {"phase": "outbound"}
        ledger.service_state_provider = lambda: dict(phase)
        state = raw(0)
        state["resource_state"]["town"] = {"active_vendor": "smith",
            "stock": [{"index": 0, "seed_hi": 1, "seed_lo": 2, "price": 40}]}
        ledger(env, state, 1)
        ledger(env, deepcopy(state), 2)
        self.assertEqual(len(ledger.shop_stock_snapshots), 1)
        self.assertEqual(len(ledger.town_entry_snapshots), 1)
        phase["phase"] = "smith"
        state["resource_state"]["town"]["stock"][0]["price"] = 50
        ledger(env, state, 3)
        self.assertEqual([p["phase"] for p in ledger.phase_changes], ["outbound", "smith"])
        self.assertEqual(len(ledger.shop_stock_snapshots), 2)
        self.assertIsNone(ledger.shop_stock_snapshots[-1]["rng"])

    def test_smoke_cannot_pass_without_real_successful_roundtrips(self):
        unexercised = [{"completed_town_trips": 0, "service": {"reason": None}}] * 16
        self.assertFalse(pilot.smoke_coverage(unexercised, unexercised, unexercised)["pass"])
        exercised = [*unexercised[:-1], {"completed_town_trips": 1,
                                      "service": {"reason": "complete"}}]
        self.assertFalse(pilot.smoke_coverage(exercised, unexercised, exercised)["pass"])
        self.assertFalse(pilot.smoke_coverage(exercised, exercised[:2], exercised)["pass"])
        self.assertTrue(pilot.smoke_coverage(exercised, exercised, exercised)["pass"])

    def test_old_r17_probe_censor_fix_does_not_require_native_execution(self):
        # Execute only the pure reporting helper, not the legacy probe module.
        path = ROOT / "train/runs/r10-staging/probe_r17_deployment.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "followup_status")
        namespace = {"ALIVE_AFTER_FIRST_DESCENT_BEATS": 1800}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        status = namespace["followup_status"]
        self.assertEqual(status(5800, 6000, False), (None, True))
        self.assertEqual(status(5800, 7600, False), (True, False))
        self.assertEqual(status(5800, 7600, True), (False, False))
        self.assertEqual(status(None, 6000, False), (None, False))


class ServiceAccountingTests(unittest.TestCase):
    def test_zero_tick_service_finish_is_not_a_policy_wait_or_stall(self):
        from diablogym.options_env import _validated_action_effect_audit
        for command in (("finish", "resource_unreachable"), ("complete",)):
            info = {"resource_action_audit": {
                        "source": "resupply-script", "command": command,
                        "micro_steps": 0, "accepted": False, "reason": command[0], "price": 0},
                    "action_effect_audit": {
                        "requested_action": 0, "native_attempts": 0, "native_accepts": 0,
                        "request_executed": False, "material_effect": False,
                        "effect_reasons": (), "same_scene": True, "stall_cost_applied": False}}
            audit = _validated_action_effect_audit(info, 0)
            self.assertFalse(audit["stall_cost_applied"])
            self.assertFalse(audit["request_executed"])
            info["action_effect_audit"]["stall_cost_applied"] = True
            with self.assertRaises(RuntimeError):
                _validated_action_effect_audit(info, 0)

    def test_repair_accounting_is_separate_from_item_purchases(self):
        service = resource.ResourceService(mode="full")
        command = ("repair", 3, 1, 2, 3, 4, 6, 10)
        service.receipt(command, {"accepted": True, "price": 10})
        self.assertEqual((service.repairs, service.repair_gold_spent,
                          service.gold_spent, service.purchases), (1, 10, 10, 0))
        service.receipt(command, {"accepted": False, "price": 99, "reason": "no_money"})
        self.assertEqual((service.repairs, service.repair_gold_spent,
                          service.gold_spent, service.purchases), (1, 10, 10, 0))
        self.assertIn("no_money", service.repair_failures)

    def test_full_repairs_cheapest_gate_gap_then_dispatches_to_healer(self):
        service = resource.ResourceService(mode="full")
        state = raw(0)
        state["gold"] = 15
        state["hp"] = 30
        state["resource_state"]["readiness"]["repair_service_needed"] = True
        quote = {"slot": 3, "seed_hi": 1, "seed_lo": 2, "create_info": 3,
                 "base_id": 4, "durability": 6, "price": 10, "repair_needed_for_gate": True}
        state["resource_state"]["town"] = {"active_vendor": "smith", "dialog_active": False,
            "stock": [], "repair_quotes": [
                {**quote, "slot": 1, "price": 100},
                {**quote, "slot": 2, "price": 1, "repair_needed_for_gate": False}, quote]}
        service.start(state, 0, "farm_cap")
        service.phase = "repair"
        env = SimpleNamespace(_raw=state, _steps=0)
        command = service.command(env, SimpleNamespace())
        self.assertEqual(command, ("repair", 3, 1, 2, 3, 4, 6, 10))
        service.receipt(command, {"accepted": True, "price": 10})
        state["gold"] = 5
        self.assertEqual(service.command(env, SimpleNamespace()), ("dismiss",))
        self.assertEqual(service.phase, "healer")
        self.assertIn("unaffordable_repair", service.repair_failures)
        self.assertIsNone(service.reason)

    def test_full_skips_smith_when_native_reports_no_repair_gap(self):
        for phase in ("smith", "repair"):
            with self.subTest(phase=phase):
                service = resource.ResourceService(mode="full")
                state = raw(0)
                state["hp"] = 30
                state["resource_state"]["readiness"].update(
                    armor_service_needed=False, repair_service_needed=False)
                service.start(state, 0, "farm_cap")
                service.phase = phase
                vendors = []
                def visit(env, vendor, town):
                    vendors.append(vendor)
                    return ("walk", 20, 30)
                service._visit = visit
                self.assertEqual(service.command(SimpleNamespace(_raw=state, _steps=0),
                                                 SimpleNamespace()), ("walk", 20, 30))
                self.assertEqual(vendors, ["healer"])
                self.assertEqual(service.phase, "healer")
                self.assertFalse(service._repair_attempted)

    def test_armor_arm_skips_repair_and_reaches_healer(self):
        service = resource.ResourceService(mode="armor")
        state = raw(0)
        state["hp"] = 30
        state["resource_state"]["readiness"]["armor_service_needed"] = False
        state["resource_state"]["town"] = {"active_vendor": "smith", "dialog_active": False,
            "stock": [], "repair_quotes": [{"slot": 3, "seed_hi": 1, "seed_lo": 2,
                "create_info": 3, "base_id": 4, "durability": 6, "price": 1,
                "repair_needed_for_gate": True}]}
        service.start(state, 0, "farm_cap")
        service.phase = "smith"
        command = service.command(SimpleNamespace(_raw=state, _steps=0), SimpleNamespace())
        self.assertEqual(command, ("dismiss",))
        self.assertEqual(service.phase, "healer")
        self.assertEqual(service.repairs, 0)
        self.assertFalse(service._repair_attempted)

    def test_only_accepted_native_purchase_spends_and_counts(self):
        service = resource.ResourceService()
        command = ("buy", "griswold", 0, 1, 2, 3, 4)
        service.receipt(command, {"accepted": False, "price": 100})
        self.assertEqual((service.purchases, service.gold_spent), (0, 0))
        service.receipt(command, {"accepted": True, "price": 40})
        service.receipt(("talk", "pepin"), {"accepted": True, "price": 500})
        self.assertEqual((service.purchases, service.gold_spent), (1, 40))

    def test_resource_operation_reaches_impact_before_cancel_and_respects_deadline(self):
        from unittest.mock import patch
        from diablogym import bridge
        from diablogym.env import DiabloGymEnv
        for remaining, expected_beats in ((20, 3), (1, 1)):
            with self.subTest(remaining=remaining):
                env = DiabloGymEnv.__new__(DiabloGymEnv)
                env._raw = {**raw(), "player_mode": bridge.PM_STAND}
                env.max_steps = 6000
                env._resource_actual_microsteps = 10
                env._resource_service_deadline = 10 + remaining
                modes = iter((4, 4, bridge.PM_STAND))
                def step_native():
                    env._resource_actual_microsteps += 1
                    return {**env._raw, "player_mode": next(modes)}
                env._step_native = step_native
                with patch("diablogym.bridge.act_controller_operate", return_value=1), \
                     patch("diablogym.bridge.act_wait") as cancel, \
                     patch("diablogym.bridge.probe_tile", side_effect=[
                         {"walkable": False}, {"walkable": False}, {"walkable": True}]):
                    observed, beats, receipt = env._execute_resource_command(("open", 62, 81))
                self.assertEqual(beats, expected_beats)
                self.assertEqual(env._resource_actual_microsteps, 10 + expected_beats)
                self.assertTrue(receipt["accepted"])
                cancel.assert_not_called()
                if remaining > 1:
                    self.assertEqual(observed["player_mode"], bridge.PM_STAND)

    def test_replanned_open_door_is_walked_through_not_toggled_shut(self):
        from unittest.mock import patch
        from diablogym.resource_protocol import ResourceService
        env = SimpleNamespace(_raw=raw(),
                              _plan_descend_path=lambda *args, **kwargs: [(62, 81, True)])
        with patch("diablogym.bridge.probe_tile", side_effect=[
                {"walkable": False}, {"walkable": True}]) as probe:
            self.assertEqual(ResourceService._walk(env, 80, 90), ("open", 62, 81))
            self.assertEqual(ResourceService._walk(env, 80, 90), ("walk", 62, 81))
            self.assertEqual(probe.call_count, 2)

    def test_gold_target_persists_across_visibility_and_cycles_are_bounded(self):
        service = resource.ResourceService(mode="full")
        state = raw()
        first = {"active_id": 1, "seed_hi": 1, "seed_lo": 2,
                 "create_info": 3, "base_id": 4, "x": 20, "y": 10}
        second = {"active_id": 2, "seed_hi": 1, "seed_lo": 3,
                  "create_info": 3, "base_id": 4, "x": 11, "y": 12}
        state["resource_state"]["gold_items"] = [first]
        service.start(state, 0, "farm_cap")
        service._walk = lambda env, x, y: ("walk", x, y)
        env = SimpleNamespace(_raw=state, _steps=0)
        self.assertEqual(service.command(env, SimpleNamespace()), ("walk", 20, 10))
        state["player_x"] = 11
        env._steps = 1
        state["resource_state"]["gold_items"] = [first, second]
        self.assertEqual(service.command(env, SimpleNamespace()), ("walk", 20, 10))
        state["player_x"] = 12
        env._steps = 2
        state["resource_state"]["gold_items"] = [second]
        self.assertEqual(service.command(env, SimpleNamespace()), ("walk", 20, 10))
        state["player_x"] = 10
        env._steps = 3
        self.assertEqual(service.command(env, SimpleNamespace()), ("walk", 11, 12))
        self.assertEqual(service.gold_navigation_failures, 1)
        self.assertIn(service._gold_key(first), service._gold_seen)
        self.assertEqual(service.telemetry()["phase_steps"], {"collect": 3})

    def test_bought_armor_is_equipped_by_full_identity_once(self):
        service = resource.ResourceService(mode="armor")
        state = raw(0)
        state["gold"] = 300
        item = {"vendor": "smith", "index": 2, "seed_hi": 11,
                "seed_lo": 12, "create_info": 13, "base_id": 14,
                "can_use": True, "can_fit": True, "meets_armor_gate": True,
                "price": 40}
        state["resource_state"]["town"] = {
            "active_vendor": "smith", "dialog_active": False, "stock": [item]}
        service.start(state, 0, "farm_cap")
        service.phase = "smith"
        env = SimpleNamespace(_raw=state, _steps=0)
        command = service.command(env, SimpleNamespace())
        self.assertEqual(command, ("buy", "smith", 2, 11, 12, 13, 14))
        service.receipt(command, {"accepted": True, "price": 40})
        state["resource_state"]["inventory_items"] = [
            {**item, "index": 1, "seed_lo": 999}, {**item, "index": 6}]
        equip = service.command(env, SimpleNamespace())
        self.assertEqual(equip, ("equip", 6, 11, 12, 13, 14))
        service.receipt(equip, {"accepted": True})
        following = service.command(env, SimpleNamespace())
        self.assertEqual(following, ("dismiss",))
        self.assertEqual(service.purchases, 1)
        self.assertEqual(service.gold_spent, 40)
        self.assertIsNone(service._armor_pending)

    def test_service_cap_is_counted_as_failure_not_downward_permission(self):
        service = resource.ResourceService()
        state = raw()
        service.start(state, 100, "cap")
        env = SimpleNamespace(_steps=700, _raw=state)
        command = service.command(env, SimpleNamespace())
        self.assertEqual(command, ("finish", "resource_service_cap"))
        self.assertFalse(service.active)
        self.assertFalse(resource.progression_allowed(state))
        self.assertEqual(service.telemetry()["steps"], 600)

    def test_service_cannot_reopen_and_farm_idle_does_not_reset_cost(self):
        service = resource.ResourceService()
        service.start(raw(), 100, "cleared")
        service._finish("resource_unresolved")
        service.start(raw(), 2000, "idle_clock")
        self.assertEqual(service.start_steps, 100)
        self.assertFalse(service.active)
        self.assertEqual(service.trigger, "cleared")


if __name__ == "__main__":
    unittest.main()
