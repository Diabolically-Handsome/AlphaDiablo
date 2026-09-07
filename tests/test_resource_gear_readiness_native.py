"""Native equipment-preservation fixtures, never model or progression evidence.

The five observed low-durability swaps are reproduced as item-property fixtures,
not replayed policy episodes. Only an explicitly selected isolated bridge may run
these tests. Importing this module does not initialize the engine.
"""
from __future__ import annotations

import unittest

import test_resource_native as native_fixtures
from test_resource_native import bridge, identity
from diablogym import DiabloGymEnv


EQUIPMENT_FAILURES = frozenset({"armor", "damage", "weapon", "durability"})
# Observed seed, candidate base, AC, current/max durability, occupied old slot.
# Provenance: R20 PASSIVE-OPPORTUNITY-RESULTS.json; these are engineering data.
OBSERVED_SWAPS = (
    (2129513, 48, 3, 9, 15, "empty_head"),
    (2129514, 55, 5, 3, 6, "empty_chest"),
    (2129519, 55, 6, 3, 6, "empty_chest"),
    (2129528, 55, 6, 4, 6, "chest"),
    (2129540, 71, 5, 12, 16, "shield"),
)


class EquipmentReadinessNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        signature = getattr(bridge.configure_resource_protocol, "__doc__", "") or ""
        if "preserve_equipment_readiness" not in signature:
            raise unittest.SkipTest("requires isolated sustain-v6 native bridge")
        cls.engine_owner = DiabloGymEnv()

    @classmethod
    def tearDownClass(cls):
        bridge.end_game()
        bridge.configure_resource_protocol(False)
        cls.engine_owner.close()

    def tearDown(self):
        bridge.end_game()
        bridge.configure_resource_protocol(False)

    def reset(self, *, preserve, layout="empty_chest", level=True):
        bridge.end_game()
        bridge.configure_resource_protocol(True, ordinary_armor_scope=True,
                                           preserve_equipment_readiness=preserve)
        bridge.reset(424301)
        # Remain in normal town: no monsters or injected combat immunity can
        # cause durability/HP changes during the two-tile pickup geometry.
        if level:
            bridge.probe_add_experience(2000)
            bridge.step(1)
        if layout != "empty_head":
            # Real cap identity with AC 2. Stable through CalcPlrInv; do not
            # use probe_bonus_ac, whose temporary bonus is recomputed away.
            bridge.probe_resource_recreate_equipment(0, 48, 8883, 62872, 257, 15)
        if layout in ("empty_head", "chest", "armored"):
            bridge.probe_resource_recreate_equipment(6, 57, 32542, 8876, 1030, 24)
        bridge.probe_resource_set_belt_heals(4)
        bridge.probe_resource_set_durability(15)
        bridge.probe_resource_set_hp_fixed(
            bridge.observe()["resource_state"]["readiness"]["max_hp_fixed"])
        bridge.probe_resource_process_game_packets()
        raw = bridge.observe()
        self.assertFalse(EQUIPMENT_FAILURES.intersection(raw["resource_state"]["readiness"]["failures"]), raw)
        if level:
            self.assertTrue(raw["resource_state"]["readiness"]["ready"], raw)
        else:
            self.assertEqual(raw["resource_state"]["readiness"]["failures"], ["level"])
        return raw

    def snapshot(self):
        return (bridge.observe(), bridge.probe_resource_inventory_snapshot(),
                bridge.probe_resource_transition_snapshot())

    def spawn(self, base, armor, durability, maximum, *, damage=(0, 0)):
        return bridge.probe_spawn_test_gear(base, *damage, armor_class=armor,
            durability=durability, max_durability=maximum)

    def floor(self, item):
        return next(entry for entry in bridge.observe()["floor_items"]
                    if entry["active_id"] == item["active_id"])

    def stand_on(self, item):
        target = (item["x"], item["y"])
        raw = bridge.observe()
        for _ in range(128):
            if ((raw["player_x"], raw["player_y"]) == target
                    and (raw["future_x"], raw["future_y"]) == target
                    and raw["player_mode"] == bridge.PM_STAND):
                bridge.probe_resource_process_game_packets()
                return
            if raw["player_mode"] == bridge.PM_STAND:
                bridge.act_walk(*target)
            raw = bridge.step(1)
        self.fail("Could not stand on the engineered item within 128 logic ticks")

    def commit(self, item):
        return bridge.act_pickup_gear_at(item["active_id"], item["x"], item["y"], *identity(item))

    def assert_rejection_is_atomic(self, item):
        self.stand_on(item)
        before = self.snapshot()
        # Repeated candidate queries cannot consume the target, alter inventory,
        # move gold, advance the engine RNG/ticks, or create delayed packets.
        for _ in range(3):
            self.assertFalse(self.floor(item)["gear"])
            self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.commit(item), 0)
        self.assertEqual(self.snapshot(), before)
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.snapshot(), before)

    def test_five_observed_properties_off_allow_on_reject_floor_and_commit(self):
        for observed_seed, base, ac, dur, maximum, layout in OBSERVED_SWAPS:
            for preserve in (False, True):
                with self.subTest(observed_seed=observed_seed, preserve=preserve):
                    self.reset(preserve=preserve, layout=layout)
                    item = self.spawn(base, ac, dur, maximum)
                    floor = self.floor(item)
                    self.assertEqual((floor["durability"], floor["max_durability"]), (dur, maximum))
                    if preserve:
                        self.assert_rejection_is_atomic(item)
                    else:
                        self.assertTrue(floor["gear"], floor)
                        self.stand_on(item)
                        utility = bridge.observe()["gear_combat_utility"]
                        self.assertEqual(self.commit(item), 1)
                        raw = bridge.observe()
                        self.assertGreater(raw["gear_combat_utility"], utility)
                        self.assertIn("durability", raw["resource_state"]["readiness"]["failures"])
                        self.assertLess(raw["resource_state"]["readiness"]["min_finite_durability"], 15)

    def test_configuration_default_marker_and_live_toggle_fail_closed(self):
        self.reset(preserve=True)
        self.assertIs(bridge.observe()["resource_state"]["preserve_equipment_readiness"], True)
        before = self.snapshot()
        bridge.configure_resource_protocol(True, ordinary_armor_scope=True,
                                           preserve_equipment_readiness=True)
        self.assertEqual(self.snapshot(), before)
        with self.assertRaises(RuntimeError):
            bridge.configure_resource_protocol(True, ordinary_armor_scope=True)
        self.assertEqual(self.snapshot(), before)
        bridge.end_game()
        with self.assertRaises(ValueError):
            bridge.configure_resource_protocol(False, preserve_equipment_readiness=True)
        self.reset(preserve=False)
        self.assertNotIn("preserve_equipment_readiness", bridge.observe()["resource_state"])
        bridge.end_game()
        bridge.configure_resource_protocol(True)  # The legacy one-argument ABI.
        raw = bridge.reset(424301)
        self.assertNotIn("preserve_equipment_readiness", raw["resource_state"])
        self.assertNotIn("ordinary_armor_scope", raw["resource_state"])

    def test_stale_growth_plan_is_rechecked_after_current_equipment_recovers(self):
        self.reset(preserve=True)
        bridge.probe_resource_set_durability(14)
        item = self.spawn(55, 6, 3, 6)
        self.assertTrue(self.floor(item)["gear"], "early growth retains legacy utility rules")
        self.stand_on(item)
        bridge.probe_resource_set_durability(15)
        self.assertTrue(bridge.observe()["resource_state"]["readiness"]["ready"])
        self.assert_rejection_is_atomic(item)  # Same ID/coordinates/identity; live plan must change.

    def test_health_potions_and_level_failures_do_not_disable_equipment_protection(self):
        for failure in ("health", "potions", "level"):
            with self.subTest(failure=failure):
                self.reset(preserve=True, level=failure != "level")
                if failure == "health":
                    bridge.probe_resource_set_hp_fixed(1 << 6)
                elif failure == "potions":
                    bridge.probe_resource_set_belt_heals(3)
                ready = bridge.observe()["resource_state"]["readiness"]
                self.assertIn(failure, ready["failures"])
                self.assertFalse(EQUIPMENT_FAILURES.intersection(ready["failures"]))
                self.assert_rejection_is_atomic(self.spawn(55, 6, 3, 6))

    def test_current_equipment_unready_still_allows_early_growth(self):
        self.reset(preserve=True)
        bridge.probe_resource_set_durability(14)
        item = self.spawn(55, 6, 3, 6)
        self.assertTrue(self.floor(item)["gear"])
        self.stand_on(item)
        self.assertEqual(self.commit(item), 1)
        self.assertIn("durability", bridge.observe()["resource_state"]["readiness"]["failures"])

    def test_exact_fifteen_boundary_and_indestructible_upgrade_remain_legal(self):
        for base, ac, dur, maximum, damage in ((49, 4, 15, 20, (0, 0)),
                (1, 0, 255, 255, (20, 30))):
            with self.subTest(base=base, durability=dur):
                self.reset(preserve=True)
                item = self.spawn(base, ac, dur, maximum, damage=damage)
                self.assertTrue(self.floor(item)["gear"])
                self.stand_on(item)
                self.assertEqual(self.commit(item), 1)
                ready = bridge.observe()["resource_state"]["readiness"]
                self.assertTrue(ready["ready"], ready)
                self.assertGreaterEqual(ready["min_finite_durability"], 15)

    def test_empty_optional_slot_is_not_itself_an_equipment_failure(self):
        raw = self.reset(preserve=True, layout="empty_head")
        self.assertFalse(raw["equipped_items"][0]["present"])
        self.assertTrue(raw["resource_state"]["readiness"]["ready"])
        self.assert_rejection_is_atomic(self.spawn(48, 3, 14, 15))

    def test_two_hand_replacement_uses_final_ac_and_clears_both_hands(self):
        for preserve, layout, allowed in ((False, "empty_chest", True),
                (True, "empty_chest", False), (True, "armored", True)):
            with self.subTest(preserve=preserve, layout=layout):
                self.reset(preserve=preserve, layout=layout)
                item = self.spawn(int(bridge.IDI_CLEAVER), 0, 15, 15, damage=(200, 255))
                if not allowed:
                    self.assert_rejection_is_atomic(item)
                    continue
                self.assertTrue(self.floor(item)["gear"])
                self.stand_on(item)
                self.assertEqual(self.commit(item), 1)
                raw = bridge.observe()
                self.assertFalse(raw["equipped_items"][5]["present"])
                self.assertTrue(raw["equipped_items"][4]["present"])
                self.assertFalse(raw["block_enabled"])
                if preserve:
                    self.assertTrue(raw["resource_state"]["readiness"]["ready"])
                else:
                    self.assertIn("armor", raw["resource_state"]["readiness"]["failures"])

    def test_existing_stat_cascade_rejection_remains_atomic_off_and_on(self):
        for preserve in (False, True):
            with self.subTest(preserve=preserve):
                self.reset(preserve=preserve)
                bridge.probe_resource_recreate_equipment(0, 48, 8883, 62872, 257, 15, 20, 0)
                bridge.probe_resource_recreate_equipment(4, 122, 3, 4, 0, 36)
                raw = bridge.observe()
                self.assertTrue(raw["equipped_items"][4]["stat_usable"])
                self.assertFalse(EQUIPMENT_FAILURES.intersection(raw["resource_state"]["readiness"]["failures"]))
                # A normal skull cap removes +20 STR, disabling the 35-STR
                # sword. The existing scalar utility rule already rejects
                # this swap. It is a calc/atomicity compatibility regression,
                # not evidence that the new guard uniquely prevents it.
                index = bridge.probe_resource_armor_item(
                    "inventory", 49, 0xA503, 354, 0, 20, 0, 4)
                projected = next(item for item in bridge.observe()["resource_state"]["inventory_items"]
                                 if item["index"] == index)
                self.assertFalse(projected["projected_readiness"]["weapon_equipped"])
                self.assertIn("weapon", projected["projected_readiness"]["failures"])
                plan = bridge.probe_resource_armor_plan(index)
                self.assertFalse(plan["old_upgrade_valid"])
                self.assertLess(plan["simulated_candidate_profile"]["physical_max"],
                                plan["live_profile"]["physical_max"])
                self.assertLess(plan["simulated_candidate_profile"]["utility"],
                                plan["live_profile"]["utility"])
                self.assert_rejection_is_atomic(self.spawn(49, 4, 20, 20))


if __name__ == "__main__":
    unittest.main()
