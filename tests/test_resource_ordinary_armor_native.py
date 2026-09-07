"""Native engineering tests for the explicit sustain-v5 ordinary armor scope.

These injected, recreated item fixtures are transaction evidence only. They are
never policy outcomes. No test executes merely by importing this module.
"""
from __future__ import annotations

import unittest

import test_resource_native as native_fixtures
from test_resource_native import bridge, identity, nav
from diablogym import DiabloGymEnv


class OrdinaryArmorNativeTests(unittest.TestCase):
    scope = True
    walk_to_town = native_fixtures.ResourceNativeTests.walk_to_town
    approach = native_fixtures.ResourceNativeTests.approach
    open_vendor = native_fixtures.ResourceNativeTests.open_vendor
    buy = native_fixtures.ResourceNativeTests.buy

    @classmethod
    def setUpClass(cls):
        if not hasattr(bridge, "probe_resource_armor_item"):
            raise unittest.SkipTest("requires isolated ordinary-armor native build")
        cls.engine_owner = DiabloGymEnv()

    @classmethod
    def tearDownClass(cls):
        bridge.end_game()
        bridge.configure_resource_protocol(False)
        cls.engine_owner.close()

    def tearDown(self):
        bridge.end_game()
        bridge.configure_resource_protocol(False)
        self.scope = True

    def reset(self, seed=8001, *, dungeon=True):
        bridge.end_game()
        bridge.configure_resource_protocol(True, ordinary_armor_scope=self.scope)
        raw = bridge.reset(seed)
        if dungeon:
            raw = nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            bridge.probe_invincible(True)
        return raw

    def snapshot(self):
        return (bridge.observe(), bridge.probe_resource_inventory_snapshot(),
                bridge.probe_resource_transition_snapshot())

    def item(self, destination, base, durability, *, quality=0, seed=123, armor_class=-1):
        index = bridge.probe_resource_armor_item(
            destination, base, 0xA501, seed, 0, durability, quality, armor_class)
        raw = bridge.observe()
        items = (raw["resource_state"]["town"]["stock"] if destination == "smith"
                 else raw["resource_state"]["inventory_equipment"])
        return next(i for i in items if i["index"] == index)

    def carried(self, item):
        return next(i for i in bridge.observe()["resource_state"]["inventory_items"]
                    if identity(i) == identity(item))

    def equip(self, item):
        return bridge.act_equip_inventory_item(item["index"], *identity(item))

    def assert_atomic_rejection(self, action, reasons):
        before = self.snapshot()
        result = action()
        self.assertFalse(result["accepted"], result)
        self.assertIn(result["reason"], reasons)
        self.assertEqual(result["price"], 0)
        self.assertEqual(self.snapshot(), before)
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.snapshot(), before)

    def test_scope_is_explicit_and_cannot_change_mid_episode(self):
        self.reset(dungeon=False)
        self.assertIs(bridge.observe()["resource_state"]["ordinary_armor_scope"], True)
        before = self.snapshot()
        bridge.configure_resource_protocol(True, ordinary_armor_scope=True)
        self.assertEqual(before, self.snapshot())
        with self.assertRaises(RuntimeError):
            bridge.configure_resource_protocol(True)
        self.assertEqual(before, self.snapshot())
        bridge.end_game()
        with self.assertRaises(ValueError):
            bridge.configure_resource_protocol(False, ordinary_armor_scope=True)
        bridge.configure_resource_protocol(True)
        raw = bridge.reset(8001)
        self.assertNotIn("ordinary_armor_scope", raw["resource_state"])

    def test_legacy_stock_schema_and_helmet_rejection_stay_chest_only(self):
        self.scope = False
        self.walk_to_town()
        bridge.probe_resource_add_gold(5000)
        self.open_vendor("smith")
        item = self.item("smith", 48, 15)
        self.assertFalse(item["is_armor"])
        for field in ("ordinary_armor_scope", "is_ordinary_armor", "target_slot",
                      "replaced_slots", "can_equip", "projected_readiness", "durability"):
            self.assertNotIn(field, item)
        self.assert_atomic_rejection(lambda: self.buy(item), {"unsupported_item"})
        own = self.item("inventory", 48, 15, seed=124)
        self.assertNotIn(identity(own), {identity(i) for i in
                         bridge.observe()["resource_state"]["inventory_items"]})
        self.assert_atomic_rejection(lambda: self.equip(own), {"cannot_use"})

    def test_shop_projection_is_readonly_complete_and_not_remote(self):
        self.walk_to_town()
        self.assertEqual(bridge.observe()["resource_state"]["town"]["stock"], [])
        self.open_vendor("smith")
        self.item("smith", 48, 15)
        before = self.snapshot()
        for _ in range(4):
            self.assertEqual(self.snapshot(), before)
        item = next(i for i in before[0]["resource_state"]["town"]["stock"]
                    if i["seed_hi"] == 0xA501)
        self.assertTrue(item["is_ordinary_armor"] and item["can_equip"])
        self.assertEqual(item["target_slot"], 0)
        self.assertEqual(item["replaced_slots"], [])
        for field in ("item_type", "equip_loc", "quality", "durability", "max_durability",
                      "min_strength", "min_magic", "min_dexterity", "effect_flags"):
            self.assertIn(field, item)
        self.assertEqual((item["durability"], item["max_durability"]), (15, 15))
        self.assertIn("block_enabled", item["projected_readiness"])
        self.assertIn("block_chance", item["projected_readiness"])

    def test_normal_helmet_and_shield_purchase_debit_and_deliver_real_items(self):
        for base, durability in ((48, 15), (72, 24)):
            with self.subTest(base=base):
                self.walk_to_town()
                bridge.probe_resource_add_gold(5000)
                self.open_vendor("smith")
                item = self.item("smith", base, durability)
                before = self.snapshot()
                receipt = self.buy(item)
                self.assertEqual(receipt, {"accepted": True, "reason": "purchased", "price": item["price"]})
                immediate = self.snapshot()
                self.assertEqual(immediate[0]["gold"], before[0]["gold"] - item["price"])
                self.assertEqual(immediate[2], before[2], "purchase must not advance ticks or RNG")
                raw = bridge.step(1)
                delivered = [i for i in raw["resource_state"]["inventory_equipment"]
                             + raw["equipped_items"] if i.get("present", True) and identity(i) == identity(item)]
                self.assertEqual(len(delivered), 1)
                self.assertEqual(delivered[0]["durability"], durability)
                self.assertNotIn(identity(item), {identity(i) for i in raw["resource_state"]["town"]["stock"]})

    def test_head_and_shield_swap_preserve_displaced_items_and_actual_stats(self):
        for old_slot, old_base, old_dur, new_base, new_dur in (
                (0, 48, 1, 49, 20), (5, 2, 1, 72, 24)):
            with self.subTest(slot=old_slot):
                self.walk_to_town()
                bridge.probe_resource_recreate_equipment(old_slot, old_base, 1, 2, 0, old_dur)
                item = self.item("inventory", new_base, new_dur)
                candidate = self.carried(item)
                self.assertTrue(candidate["can_equip"], candidate)
                self.assertEqual(candidate["target_slot"], old_slot)
                self.assertEqual(candidate["replaced_slots"], [old_slot])
                before = self.snapshot()
                displaced = before[0]["equipped_items"][old_slot]
                result = self.equip(candidate)
                self.assertTrue(result["accepted"], result)
                self.assertEqual(result["target_slot"], old_slot)
                self.assertEqual(result["replaced_slots"], [old_slot])
                immediate = self.snapshot()
                self.assertEqual(immediate[0]["gold"], before[0]["gold"])
                self.assertEqual(immediate[2]["rng"], before[2]["rng"])
                raw = bridge.step(1)
                actual = raw["equipped_items"][old_slot]
                self.assertEqual(identity(actual), identity(candidate))
                kept = next(i for i in raw["resource_state"]["inventory_equipment"]
                            if identity(i) == identity(displaced))
                self.assertEqual({k: v for k, v in kept.items() if k != "index"},
                                 {k: v for k, v in displaced.items() if k != "present"})
                for field, value in raw["resource_state"]["readiness"].items():
                    self.assertEqual(candidate["projected_readiness"][field], value, field)
                self.assertEqual(candidate["projected_readiness"]["block_enabled"], raw["block_enabled"])
                self.assertEqual(candidate["projected_readiness"]["block_chance"], raw["block_chance"])
                after = self.snapshot()
                bridge.probe_resource_process_game_packets()
                self.assertEqual(self.snapshot(), after)

    def test_stale_wrong_vendor_and_nonordinary_items_reject_atomically(self):
        self.walk_to_town()
        bridge.probe_resource_add_gold(5000)
        self.open_vendor("smith")
        item = self.item("smith", 48, 15)
        stale = list(identity(item)); stale[1] ^= 1
        self.assert_atomic_rejection(lambda: bridge.act_buy_store_item(
            "smith", item["index"], *stale), {"stale_item"})
        self.assert_atomic_rejection(lambda: bridge.act_buy_store_item(
            "healer", item["index"], *identity(item)), {"wrong_vendor"})
        for base, durability, quality in ((48, 15, 1), (1, 24, 0)):
            unsupported = self.item("smith", base, durability, quality=quality, seed=base)
            self.assertFalse(unsupported["is_ordinary_armor"])
            self.assert_atomic_rejection(lambda: self.buy(unsupported), {"unsupported_item"})
        own = self.item("inventory", 48, 15)
        self.assert_atomic_rejection(lambda: bridge.act_equip_inventory_item(
            own["index"], *stale), {"stale_item"})

    def test_full_inventory_keeps_larger_displaced_shield_on_rejection(self):
        self.walk_to_town()
        # Inactive 40-STR large shield occupies 2x3; the usable incoming
        # buckler frees only 2x2 in an otherwise full inventory.
        bridge.probe_resource_recreate_equipment(5, 73, 1, 2, 0, 32)
        item = self.item("inventory", 71, 16)
        self.assertTrue(self.carried(item)["can_equip"])
        bridge.probe_resource_fill_inventory()
        candidate = self.carried(item)
        self.assertFalse(candidate["can_equip"])
        self.assertEqual(candidate["equip_reason"], "no_room_for_replaced_items")
        self.assert_atomic_rejection(lambda: self.equip(candidate), {"no_room_for_replaced_items"})

    def test_full_inventory_smith_purchase_rejects_before_debit_or_stock_change(self):
        self.walk_to_town()
        bridge.probe_resource_add_gold(5000)
        self.open_vendor("smith")
        item = self.item("smith", 72, 24)
        bridge.probe_resource_fill_inventory()
        current = next(i for i in bridge.observe()["resource_state"]["town"]["stock"]
                       if identity(i) == identity(item))
        self.assertFalse(current["can_fit"])
        self.assert_atomic_rejection(lambda: self.buy(current), {"no_room"})

    def test_finite_durability_gate_uses_actual_current_and_max(self):
        self.walk_to_town()
        self.open_vendor("smith")
        bridge.probe_bonus_ac(9)
        for base, durability, maximum in ((48, 14, 15), (54, 12, 12), (55, 6, 6)):
            with self.subTest(base=base):
                item = self.item("smith", base, durability, seed=base)
                self.assertEqual((item["durability"], item["max_durability"]), (durability, maximum))
                self.assertFalse(item["meets_armor_gate"])

    def test_two_hand_displacement_is_legal_but_readiness_exposes_missing_weapon(self):
        self.walk_to_town()
        # An unusable two-handed sword plus offhand is an engineering fixture
        # to cover both displaced identities. Normal placement is legal;
        # readiness must still reject a resulting shield-only loadout.
        bridge.probe_resource_recreate_equipment(4, 128, 1, 2, 0, 75)
        item = self.item("inventory", 72, 24)
        candidate = self.carried(item)
        self.assertTrue(candidate["can_equip"], candidate)
        self.assertFalse(candidate["meets_armor_gate"])
        self.assertEqual(candidate["target_slot"], 4)
        self.assertEqual(candidate["replaced_slots"], [4, 5])
        self.assertFalse(candidate["projected_readiness"]["ready"])
        self.assertFalse(candidate["projected_readiness"]["weapon_equipped"])
        self.assertIn("weapon", candidate["projected_readiness"]["failures"])
        before = bridge.observe()
        receipt = self.equip(candidate)
        self.assertTrue(receipt["accepted"], receipt)
        raw = bridge.step(1)
        self.assertFalse(raw["resource_state"]["readiness"]["ready"])
        kept = {identity(i) for i in raw["resource_state"]["inventory_equipment"]}
        self.assertTrue({identity(before["equipped_items"][slot]) for slot in (4, 5)} <= kept)

    def test_legal_ac_decrease_repairs_durability_without_utility_upgrade(self):
        self.walk_to_town()
        bridge.probe_add_experience(2000)
        bridge.step(1)
        bridge.probe_resource_set_belt_heals(4)
        bridge.probe_resource_set_hp_fixed(
            bridge.observe()["resource_state"]["readiness"]["max_hp_fixed"])
        # Both AC values are within these real base items' normal ranges.
        # Exact AC is an explicit engineering fixture, never evaluation data.
        old = self.item("inventory", 49, 14, seed=140, armor_class=4)
        self.assertTrue(self.equip(self.carried(old))["accepted"])
        raw = bridge.step(1)
        self.assertEqual(raw["armor_class"], 11)
        self.assertEqual(raw["resource_state"]["readiness"]["failures"], ["durability"])
        new = self.item("inventory", 48, 15, seed=150, armor_class=3)
        candidate = self.carried(new)
        self.assertTrue(candidate["can_equip"], candidate)
        self.assertTrue(candidate["meets_armor_gate"])
        self.assertEqual(candidate["projected_armor_class"], 10)
        self.assertTrue(candidate["projected_readiness"]["ready"])
        before = self.snapshot()
        before_utility = bridge.probe_gear_combat_profile()["utility"]
        result = self.equip(candidate)
        self.assertTrue(result["accepted"], result)
        immediate = self.snapshot()
        self.assertEqual(immediate[0]["gold"], before[0]["gold"])
        self.assertEqual(immediate[2], before[2])
        raw = bridge.step(1)
        self.assertEqual(raw["armor_class"], 10)
        self.assertEqual(bridge.probe_gear_combat_profile()["utility"] - before_utility, -1008)
        self.assertEqual(raw["equipped_items"][0]["durability"], 15)
        self.assertTrue(raw["resource_state"]["readiness"]["ready"])
        self.assertIn(identity(old), {identity(i) for i in raw["resource_state"]["inventory_equipment"]})

    def test_scope_only_unequip_projection_reports_actual_block(self):
        self.walk_to_town()
        raw = bridge.observe()
        self.assertTrue(raw["block_enabled"])
        shield = next(i for i in raw["resource_state"]["unequip_candidates"] if i["slot"] == 5)
        self.assertFalse(shield["projected_readiness"]["block_enabled"])
        self.assertIn("block_chance", shield["projected_readiness"])
        self.scope = False
        self.walk_to_town()
        for item in bridge.observe()["resource_state"]["unequip_candidates"]:
            self.assertNotIn("block_enabled", item["projected_readiness"])
            self.assertNotIn("block_chance", item["projected_readiness"])

    def test_seen_smith_projection_uses_healed_player_without_world_changes(self):
        self.walk_to_town()
        bridge.probe_resource_set_hp_fixed(1 << 6)
        self.open_vendor("smith")
        item = self.item("smith", 48, 15)
        self.assertIn("health", item["projected_readiness"]["failures"])
        self.open_vendor("healer")
        before = self.snapshot()
        self.assertEqual(before[0]["hp"], before[0]["max_hp"])
        for _ in range(4):
            projected = bridge.project_seen_resource_smith_items([identity(item)])
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(len(projected), 1)
            entry = projected[0]
            self.assertEqual(identity(entry), identity(item))
            self.assertEqual(entry["status"], "projected")
            self.assertEqual(entry["projection_origin"], "known-smith-live-player")
            self.assertEqual(entry["vendor"], "smith")
            self.assertEqual(entry["price"], item["price"])
            self.assertEqual(entry["town_seed"], before[0]["resource_state"]["town_seed"])
            self.assertEqual(entry["town_restock_sequence"], before[0]["resource_state"]["town_restock_sequence"])
            self.assertNotIn("health", entry["projected_readiness"]["failures"])
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.snapshot(), before)

    def test_smith_projection_rejects_unseen_duplicates_and_reset_identity(self):
        self.walk_to_town()
        self.open_vendor("smith")
        known = self.item("smith", 48, 15)
        self.open_vendor("healer")
        # Real stock inserted by an engineering fixture while the player is
        # at Pepin. It exists, but has never been exported in Smith's raw view.
        bridge.probe_resource_armor_item("smith", 49, 0xA501, 777, 0, 20)
        unseen = (0xA501, 777, 0, 49)
        for requested in ([identity(known), unseen], [identity(known), identity(known)]):
            before = self.snapshot()
            with self.assertRaises(ValueError):
                bridge.project_seen_resource_smith_items(requested)
            bridge.probe_resource_process_game_packets()
            self.assertEqual(self.snapshot(), before)
        self.walk_to_town()
        self.open_vendor("healer")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            bridge.project_seen_resource_smith_items([identity(known)])
        self.assertEqual(self.snapshot(), before)
        self.scope = False
        self.walk_to_town()
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            bridge.project_seen_resource_smith_items([identity(known)])
        self.assertEqual(self.snapshot(), before)

    def test_sold_known_item_returns_no_stale_projection(self):
        self.walk_to_town()
        bridge.probe_resource_add_gold(5000)
        self.open_vendor("smith")
        item = self.item("smith", 48, 15)
        self.assertTrue(self.buy(item)["accepted"])
        bridge.step(1)
        before = self.snapshot()
        result = bridge.project_seen_resource_smith_items([identity(item)])
        self.assertEqual(len(result), 1)
        self.assertEqual(identity(result[0]), identity(item))
        self.assertEqual(result[0]["status"], "stale_item")
        self.assertNotIn("projected_readiness", result[0])
        self.assertNotIn("price", result[0])
        self.assertEqual(result[0]["projection_origin"], "known-smith-live-player")
        self.assertEqual(result[0]["town_seed"], before[0]["resource_state"]["town_seed"])
        self.assertEqual(self.snapshot(), before)

    def test_stat_cascade_refreshes_carried_requirement_flags(self):
        self.walk_to_town()
        bridge.probe_resource_recreate_equipment(0, 48, 1, 2, 0, 15, 20, 0)
        carried_weapon = self.item("inventory", 122, 36, seed=351)
        self.assertTrue(carried_weapon["stat_usable"])
        replacement = self.item("inventory", 49, 20, seed=352)
        candidate = self.carried(replacement)
        self.assertTrue(candidate["can_equip"])
        self.assertTrue(self.equip(candidate)["accepted"])
        raw = bridge.observe()
        self.assertEqual(raw["strength"], 30)
        carried = next(i for i in raw["resource_state"]["inventory_equipment"]
                       if identity(i) == identity(carried_weapon))
        self.assertFalse(carried["stat_usable"])
        bridge.step(1)
        self.assertFalse(next(i for i in bridge.observe()["resource_state"]["inventory_equipment"]
                              if identity(i) == identity(carried_weapon))["stat_usable"])

    def test_stat_dependency_and_fatal_hp_replacement_never_bypass_native_plan(self):
        for strength_bonus, hp_bonus, low_hp in ((20, 0, False), (0, 100 << 6, True)):
            with self.subTest(strength_bonus=strength_bonus):
                self.walk_to_town()
                bridge.probe_resource_recreate_equipment(0, 48, 1, 2, 0, 15, strength_bonus, hp_bonus)
                if strength_bonus:
                    bridge.probe_resource_recreate_equipment(4, 122, 3, 4, 0, 36)
                if low_hp:
                    bridge.probe_resource_set_hp_fixed(1 << 6)
                item = self.item("inventory", 49, 20)
                candidate = self.carried(item)
                if candidate["can_equip"]:
                    self.assertFalse(candidate["meets_armor_gate"])
                    self.assertFalse(candidate["projected_readiness"]["weapon_equipped"])
                else:
                    self.assert_atomic_rejection(lambda: self.equip(candidate), {"cannot_use_after_swap", "unsafe_life"})


if __name__ == "__main__":
    unittest.main()
