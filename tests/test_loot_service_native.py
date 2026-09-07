"""R21 engineering fixture for the real service-to-native sale/purchase chain.

Run only on the independently built R21 bridge selected by the normal candidate
loader.  No model, worker callback, training, or natural-start effect evaluation
is involved.  Fixture preparation deliberately supplies an improved sword,
immunity, 20 inventory gold and three belt heals, then approaches Smith through
the existing native test helpers.  Preparation is outside the measured command
segment; this is not evidence about the cost or success of a normal town trip.
"""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import test_loot_resource_native as loot_fixtures
from test_resource_native import bridge, identity
from diablogym.options_env import (
    OptionsEnv, WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
)
from diablogym.resource_sustain_loot import SustainLootService


class LootServiceNativeTests(unittest.TestCase):
    # Reuse existing real geometry, gear-retention and store-opening fixtures.
    # Do not inherit their TestCase: that would silently run their entire suite.
    stand_on = loot_fixtures.LootResourceNativeTests.stand_on
    floor = loot_fixtures.LootResourceNativeTests.floor
    gear = loot_fixtures.LootResourceNativeTests.gear
    strong_sword = loot_fixtures.LootResourceNativeTests.strong_sword
    enter_service = loot_fixtures.LootResourceNativeTests.enter_service
    approach = loot_fixtures.LootResourceNativeTests.approach
    open_vendor = loot_fixtures.LootResourceNativeTests.open_vendor

    def setUp(self):
        for name in ("act_sell_inventory_item", "act_pickup_loot_at",
                     "probe_loot_inventory_value", "probe_resource_set_belt_heals"):
            if not hasattr(bridge, name):
                raise RuntimeError(f"This integration test requires the isolated R21 bridge: {name}")
        self.owner = OptionsEnv(max_steps=6000, resource_protocol="l2-town-v1",
            resource_purchase_mode="full", resource_service_policy="sustain-loot-v1",
            worker_time_protocol="completion-l2-v1",
            worker_observation_view=WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
        self.addCleanup(self.close_owner)
        self.owner.reset(seed=424719)
        self.env = self.owner.env
        self.service = self.owner.resource_service
        self.assertIsInstance(self.service, SustainLootService)
        self.assertTrue(self.env._raw["resource_state"]["loot_economy"])
        self.assertEqual(self.env._raw["dungeon_level"], 1)

    def close_owner(self):
        try:
            self.owner.close()
        finally:
            bridge.end_game()
            bridge.configure_resource_protocol(False)

    def prepare_at_smith(self):
        bridge.probe_invincible(True)
        old_sword, _ = self.strong_sword()
        inventory = bridge.observe()["resource_state"]["inventory_state"]["items"]
        retained = next(i for i in inventory if identity(i) == identity(old_sword))
        self.assertTrue(retained["retained_from_a14"])
        clubs = [i for i in inventory if i["base_id"] == int(bridge.IDI_WARRCLUB)]
        self.assertEqual(len(clubs), 1, "Use the existing Club, not an extra saleable fixture item")
        club = clubs[0]
        currency = [i for i in inventory if i["item_type"] == 11]
        self.assertEqual(len(currency), 1)
        bridge.probe_loot_inventory_value(currency[0]["index"], 20)
        bridge.probe_resource_set_belt_heals(3)
        raw = bridge.observe()
        self.assertEqual(raw["gold"], 20)
        self.assertEqual(raw["resource_state"]["readiness"]["belt_heals"], 3)
        self.assertFalse(raw["resource_state"]["readiness"]["ready"])

        # This synthetic FARM threshold opens the real guarded service. All
        # mutation probes above precede its opening cash baseline.
        self.env._raw = raw
        self.assertTrue(self.service.maybe_start(raw, self.env._steps,
            farm_scene_steps=3600, cleared=False))
        self.enter_service()
        raw, _ = self.open_vendor("smith")
        self.assertEqual(raw["dungeon_level"], 0)
        self.assertTrue(raw["resource_state"]["service_trip"])
        self.assertEqual(raw["resource_state"]["service_trips_started"], 1)
        quotes = {identity(q): q for q in raw["resource_state"]["town"]["sell_quotes"]}
        expected = {identity(old_sword): 30, identity(club): 5}
        self.assertEqual({k: quotes[k]["price"] for k in expected}, expected)
        self.assertEqual(set(quotes), set(expected), "No third idle item may hide extra sale income")

        # Install the already prepared scene at the command boundary. The old
        # fixture helpers step the engine directly, so their preparation ticks
        # are expressly not claimed as wrapper-accounted service ticks.
        self.env._raw = raw
        self.service.observe(raw, self.env._steps)
        self.service.phase = "outbound"
        self.service._last_phase = "outbound"
        return expected

    def test_native_sale35_then_heal50_and_complete_service_cash_ledger(self):
        expected_sales = self.prepare_at_smith()
        starting_native_clock = self.env._resource_actual_microsteps
        original_callback = self.env.resource_observation_callback
        self.assertTrue(callable(original_callback))
        observed_ticks = []

        def observe_once(environment, raw, beat):
            # Same existing callback, called once with its original arguments.
            original_callback(environment, raw, beat)
            observed_ticks.append((int(beat), int(raw["gold"])))

        self.env.resource_observation_callback = observe_once
        commands, sales, purchases = [], [], []
        settled_microsteps = 0
        try:
            # This wraps, but does not replace, the real native dispatch path.
            with patch.object(self.env, "_execute_resource_command",
                              wraps=self.env._execute_resource_command) as executed:
                for _ in range(1800):
                    if not self.service.active:
                        break
                    before_clock = self.env._resource_actual_microsteps
                    command = self.service.command(self.env, bridge)
                    self.env._resource_service_deadline = self.service.command_microstep_deadline
                    before_dispatches = executed.call_count
                    _, _, terminated, truncated, info = self.env.step_resource(command)
                    self.assertEqual(executed.call_count, before_dispatches + 1)
                    self.assertEqual(executed.call_args.args, (command,))
                    audit = deepcopy(info["resource_action_audit"])
                    self.service.receipt(command, audit)
                    self.service.record_steps(self.env._steps, self.service.phase)
                    commands.append(tuple(command))
                    settled_microsteps += self.env._resource_actual_microsteps - before_clock
                    self.assertEqual(self.service._observed_step, self.env._resource_actual_microsteps)
                    self.assertEqual(self.env._steps, self.env._resource_actual_microsteps)
                    self.assertEqual(self.owner._completion_clock.state.micro_step, self.env._steps)
                    self.assertFalse(terminated or truncated, (command, info))
                    self.assertLessEqual(self.service.steps, 3000)
                    self.assertEqual(self.service.gold_collected, 0, "Sale cash must never become pickup income")
                    if command[0] == "sell":
                        key = tuple(command[3:7])
                        self.assertIn(key, expected_sales)
                        self.assertTrue(audit["accepted"], audit)
                        self.assertEqual(audit["received"], expected_sales[key])
                        self.assertEqual(audit["price"], 0)
                        self.assertEqual(audit["gold_after"] - audit["gold_before"], expected_sales[key])
                        self.assertEqual(self.env._raw["gold"], audit["gold_after"])
                        self.assertNotIn(key, {identity(i) for i in self.env._raw["resource_state"]["inventory_state"]["items"]})
                        sales.append(audit)
                        self.assertEqual(self.service.gold_sold, sum(r["received"] for r in sales))
                        if len(sales) == 2:
                            self.assertEqual(self.env._raw["gold"], 55)
                    elif command[0] == "buy":
                        self.assertEqual(len(sales), 2)
                        self.assertEqual(command[1], "healer")
                        self.assertTrue(audit["accepted"], audit)
                        self.assertEqual(audit["price"], 50)
                        self.assertEqual(self.env._raw["gold"], 5)
                        self.assertEqual(self.env._raw["resource_state"]["readiness"]["belt_heals"], 4)
                        purchases.append(audit)
                else:
                    self.fail("Engineering service did not finish within the explicit command bound")
        finally:
            self.env.resource_observation_callback = original_callback

        self.assertFalse(self.service.active)
        self.assertEqual(self.env._raw["dungeon_level"], 1)
        self.assertEqual(commands[-1], ("complete",))
        self.assertIn(self.service.reason, ("complete", "growth_remaining"))
        self.assertEqual(len(sales), 2)
        self.assertEqual(len(purchases), 1)
        self.assertEqual(self.service.gold_sold, 35)
        self.assertEqual(self.service.gold_spent, 50)
        self.assertEqual(self.service.gold_collected, 0)
        self.assertEqual(self.service.repair_gold_spent, 0)
        self.assertEqual(self.env._raw["gold"], 5)
        self.assertEqual(self.env._resource_actual_microsteps - starting_native_clock, settled_microsteps)
        self.assertEqual([beat for beat, _ in observed_ticks],
                         list(range(starting_native_clock + 1, self.env._resource_actual_microsteps + 1)))
        report = self.service.telemetry()
        self.assertEqual(report["trip_count"], 1)
        self.assertEqual(len(report["trips"]), 1)
        trip = report["trips"][0]
        self.assertEqual((trip["start_gold"], trip["gold_collected"], trip["gold_sold"],
                          trip["gold_spent"], trip["end_gold"]), (20, 0, 35, 50, 5))
        self.assertEqual(trip["cash_residual"], 0)
        self.assertEqual(trip["cash_reconciliation"], "complete")
        self.assertTrue(trip["gold_collected_complete"])
        self.assertEqual(report["cumulative"]["gold_sold"], 35)
        self.assertEqual(report["cumulative"]["gold_collected"], 0)

    def test_pending_sale_at_zero_deadline_is_fully_accounted_without_native_execution(self):
        self.prepare_at_smith()
        command = self.service.command(self.env, bridge)
        self.assertEqual(command[0], "sell")
        before = deepcopy(self.env._raw)
        clock = self.env._resource_actual_microsteps
        self.env._resource_service_deadline = clock
        # Exercise the real precheck directly: zero remaining time must not
        # call a native action, wait, query, or observation callback.
        with patch.object(bridge, "act_sell_inventory_item", side_effect=AssertionError("native sale after deadline")), \
             patch.object(bridge, "act_wait", side_effect=AssertionError("wait after deadline")), \
             patch.object(self.env, "_step_native", side_effect=AssertionError("tick after deadline")):
            result, microsteps, audit = self.env._execute_resource_command(command)
        self.assertIs(result, self.env._raw)
        self.assertEqual(microsteps, 0)
        self.assertEqual(self.env._raw, before)
        self.assertEqual(self.env._resource_actual_microsteps, clock)
        self.assertEqual(audit["source"], "environment-precheck")
        self.assertFalse(audit["native_executed"])
        self.assertEqual(audit["reason"], "resource_command_deadline")
        self.assertEqual(audit["received"], 0)
        self.service.receipt(command, audit)
        self.service.record_steps(self.env._steps, self.service.phase)
        self.service.finish_episode(self.env._raw, self.env._steps, "resource_service_cap")
        self.assertEqual(self.service.gold_sold, 0)
        self.assertEqual(self.service.gold_collected, 0)
        self.assertEqual(self.service.telemetry()["trips"][0]["cash_residual"], 0)
        self.assertEqual(self.service.reason, "resource_service_cap")


if __name__ == "__main__":
    unittest.main()
