"""Native engineering tests for observed ordinary-equipment combinations.

Only an isolated, explicitly selected bridge may run these tests. The recreated
items, experience, belt, gold and invincibility below are engineering fixtures,
never policy samples or evidence of an economically successful episode. Importing
this module does not initialize a game. Root owns compilation and real execution.
"""
from __future__ import annotations

import unittest

import test_resource_native as native_fixtures
import test_resource_ordinary_armor_native as armor_fixtures
from test_resource_native import bridge, identity, nav
from diablogym import DiabloGymEnv


VERSION = "observed-equipment-combinations-v1"
SOURCE_ONLY = frozenset({"episode_generation", "town_seed", "town_restock_sequence"})


class ResourceCombinationsNativeTests(unittest.TestCase):
    scope = True
    walk_to_town = native_fixtures.ResourceNativeTests.walk_to_town
    approach = native_fixtures.ResourceNativeTests.approach
    open_vendor = native_fixtures.ResourceNativeTests.open_vendor
    buy = native_fixtures.ResourceNativeTests.buy
    snapshot = armor_fixtures.OrdinaryArmorNativeTests.snapshot
    item = armor_fixtures.OrdinaryArmorNativeTests.item
    carried = armor_fixtures.OrdinaryArmorNativeTests.carried
    equip = armor_fixtures.OrdinaryArmorNativeTests.equip

    @classmethod
    def setUpClass(cls):
        if not hasattr(bridge, "preview_resource_equipment_combinations"):
            raise unittest.SkipTest("requires the isolated combinations native bridge")
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
        if self.scope:
            bridge.configure_resource_protocol(True, ordinary_armor_scope=True,
                                               preserve_equipment_readiness=True)
        else:
            # Retain the old one-argument ABI and default-off observations.
            bridge.configure_resource_protocol(True)
        raw = bridge.reset(seed)
        if dungeon:
            raw = nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            bridge.probe_invincible(True)
        return raw

    def preview(self, sequences, budget=None):
        if budget is None:
            budget = bridge.observe()["gold"]
        result = bridge.preview_resource_equipment_combinations(sequences, budget)
        self.assertEqual(result["version"], VERSION)
        self.assertEqual(result["max_gold_cost"], budget)
        self.assertEqual(result["batch_limit"], 256)
        self.assertEqual(result["step_limit"], 8)
        self.assertIs(result["truncated"], False)
        self.assertEqual(len(result["results"]), len(sequences))
        self.assertEqual([r["index"] for r in result["results"]], list(range(len(sequences))))
        return result

    def one(self, sequence, budget=None):
        return self.preview([sequence], budget)["results"][0]

    def assert_invalid(self, result, reason, failed_step):
        self.assertIs(result["valid"], False, result)
        self.assertEqual(result["reason"], reason)
        self.assertEqual(result["failed_step"], failed_step)
        # Partial simulated commands/cost never authorize a real transaction.
        for field in ("first_command", "projected_state", "projected_readiness",
                      "projected_equipment", "projected_inventory"):
            self.assertIsNone(result[field], (field, result))

    def assert_unchanged(self, before):
        self.assertEqual(self.snapshot(), before)
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.snapshot(), before)

    def bad_chest_fixture(self):
        """883-type actual item identities; not a replay of seed 2129883."""
        self.walk_to_town()
        bridge.probe_add_experience(2000)
        bridge.step(1)
        bridge.probe_resource_recreate_equipment(4, 1, 50880, 65488, 0, 22)
        bridge.probe_resource_recreate_equipment(5, 2, 18603, 37538, 0, 14)
        bridge.probe_resource_recreate_equipment(6, 55, 8279, 278, 257, 4)
        bridge.probe_resource_set_belt_heals(4)
        bridge.probe_resource_set_hp_fixed(
            bridge.observe()["resource_state"]["readiness"]["max_hp_fixed"])
        gold = bridge.observe()["gold"]
        self.assertLessEqual(gold, 367)
        if gold < 367:
            bridge.probe_resource_add_gold(367 - gold)
        self.open_vendor("smith")
        index = bridge.probe_resource_armor_item("smith", 72, 17525, 649, 1030, 24, 0, 8)
        raw = bridge.observe()  # The offer must actually enter the seen whitelist.
        stock = next(i for i in raw["resource_state"]["town"]["stock"] if i["index"] == index)
        chest = raw["equipped_items"][6]
        self.assertEqual(identity(stock), (17525, 649, 1030, 72))
        self.assertEqual((stock["price"], stock["armor_class"], stock["durability"]), (90, 8, 24))
        self.assertTrue(stock["can_use"])
        self.assertEqual((chest["durability"], chest["max_durability"]), (4, 6))
        self.assertIn("durability", raw["resource_state"]["readiness"]["failures"])
        self.assertEqual(raw["gold"], 367)
        bridge.probe_resource_process_game_packets()
        return stock, chest

    def execute(self, commands):
        """Run only the preview's normal transactions, with real ticks."""
        receipts = []
        for command in commands:
            op, *args = command
            action = {
                "buy": bridge.act_buy_store_item,
                "equip": bridge.act_equip_inventory_item,
                "unequip": bridge.act_unequip_equipped_item,
                "repair": bridge.act_repair_equipped_item,
            }[op]
            receipt = action(*args)
            self.assertIs(receipt["accepted"], True, (command, receipt))
            raw = bridge.step(1)
            receipts.append((tuple(command), receipt, raw))
        return receipts

    def assert_projection_is_actual(self, projected):
        # The source serializer reads the live Player before making any copy.
        # Compare all its fields, including every unused inventory slot and grid.
        actual = self.preview([[]])["source"]
        self.assertEqual(projected["projected_state"],
                         {k: v for k, v in actual.items() if k not in SOURCE_ONLY})
        raw = bridge.observe()
        for key, value in raw["resource_state"]["readiness"].items():
            self.assertEqual(projected["projected_readiness"][key], value, key)
        self.assertEqual(projected["gold_remaining"], raw["gold"])
        self.assertEqual(projected["projected_equipment"], actual["equipment"])
        self.assertEqual(projected["projected_inventory"], actual["inventory"])
        self.assertEqual(actual["inventory_grid"], bridge.probe_resource_inventory_snapshot()["grid"])

    def test_default_scope_and_old_helmet_rejection_are_unchanged(self):
        self.scope = False
        self.walk_to_town()
        self.open_vendor("smith")
        helm = self.item("smith", 48, 15, seed=501)
        raw = bridge.observe()
        for field in ("ordinary_armor_scope", "preserve_equipment_readiness"):
            self.assertNotIn(field, raw["resource_state"])
        self.assertNotIn("projected_readiness", helm)
        self.assertFalse(helm["is_armor"])
        bridge.probe_resource_process_game_packets()
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            bridge.preview_resource_equipment_combinations([[]], 0)
        self.assert_unchanged(before)
        result = self.buy(helm)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "unsupported_item")
        self.assert_unchanged(before)

    def test_preview_requires_actual_town_service(self):
        self.reset(dungeon=False)  # Fresh town is not an authorized resource trip.
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            bridge.preview_resource_equipment_combinations([[]], 0)
        self.assert_unchanged(before)

    def test_seen_shield_and_bad_chest_need_joint_projection(self):
        shield, chest = self.bad_chest_fixture()
        buy = ("buy_equip", *identity(shield))
        remove = ("unequip", 6, *identity(chest))
        results = self.preview([[buy], [buy, remove], [remove, buy]])["results"]
        self.assertTrue(results[0]["valid"])
        self.assertFalse(results[0]["projected_readiness"]["ready"])
        self.assertIn("durability", results[0]["projected_readiness"]["failures"])
        for result in results[1:]:
            self.assertTrue(result["valid"], result)
            self.assertTrue(result["projected_readiness"]["ready"], result)
            self.assertEqual(result["gold_cost"], 90)
            self.assertEqual(result["gold_remaining"], 277)
            self.assertTrue(result["projected_equipment"][6]["empty"])
            self.assertEqual(tuple(result["projected_equipment"][5]["identity"]), identity(shield))
            self.assertEqual(result["projected_readiness"]["armor_class"], 12)
            self.assertTrue(result["projected_readiness"]["block_enabled"])

    def test_preview_is_repeatable_readonly_and_preserves_full_sources(self):
        shield, chest = self.bad_chest_fixture()
        sequences = [[("buy_equip", *identity(shield)), ("unequip", 6, *identity(chest))], []]
        before = self.snapshot()
        first = self.preview(sequences)
        self.assertEqual(len(first["source"]["inventory"]), 40)
        self.assertEqual(len(first["source"]["inventory_grid"]), 40)
        self.assertEqual(len(first["source"]["belt"]), 8)
        self.assertEqual(tuple(first["source"]["equipment"][6]["identity"]), identity(chest))
        self.assertEqual(first["source"]["equipment"][6]["durability"], 4)
        self.assertEqual(first["source"]["gold"], 367)
        for _ in range(3):
            self.assertEqual(self.preview(sequences), first)
            self.assert_unchanged(before)

    def test_normal_buy_equip_unequip_matches_entire_projection(self):
        shield, chest = self.bad_chest_fixture()
        result = self.one([("buy_equip", *identity(shield)), ("unequip", 6, *identity(chest))])
        self.assertTrue(result["valid"], result)
        self.assertEqual([c[0] for c in result["expanded_commands"]], ["buy", "equip", "unequip"])
        self.assertEqual(result["first_command"], result["expanded_commands"][0])
        initial = self.snapshot()
        receipts = self.execute(result["expanded_commands"])
        self.assertEqual(sum(r[1]["price"] for r in receipts), 90)
        # Purchase delivery precedes and is distinct from the equip command.
        bought = receipts[0][2]
        self.assertIn(identity(shield), {identity(i) for i in bought["resource_state"]["inventory_equipment"]})
        self.assertNotEqual(identity(bought["equipped_items"][5]), identity(shield))
        self.assert_projection_is_actual(result)
        final = bridge.observe()["resource_state"]["inventory_equipment"]
        saved = {identity(i) for i in final}
        self.assertIn(identity(chest), saved)
        old_shield = initial[0]["equipped_items"][5]
        self.assertIn(identity(old_shield), saved)

    def test_personal_gold_and_reserved_budget_are_separate_limits(self):
        shield, _ = self.bad_chest_fixture()
        before = self.snapshot()
        self.assert_invalid(self.one([("buy_equip", *identity(shield))], 89), "gold_cost_limit", 0)
        self.assertTrue(self.one([("buy_equip", *identity(shield))], 90)["valid"])
        with self.assertRaises(ValueError):
            bridge.preview_resource_equipment_combinations([[]], 368)
        self.assert_unchanged(before)
        chest = self.item("smith", 59, 35, seed=502, armor_class=12)
        self.assertEqual(chest["price"], 300)
        self.assertTrue(chest["can_use"])
        before = self.snapshot()
        failed = self.one([("buy_equip", *identity(chest)), ("buy_equip", *identity(shield))])
        self.assert_invalid(failed, "no_money", 1)
        self.assertEqual(failed["gold_cost"], 300)  # Partial simulation is not paid.
        self.assertEqual(failed["gold_remaining"], 67)
        self.assert_unchanged(before)

    def test_unknown_mixed_batch_and_reset_identity_fail_closed(self):
        shield, _ = self.bad_chest_fixture()
        unknown = (0xFFFF, 0xFFFF, 0, 72)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.preview([[('buy_equip', *identity(shield))], [('buy_equip', *unknown)]])
        self.assert_unchanged(before)
        self.walk_to_town(seed=8002)
        self.open_vendor("smith")
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.preview([[('buy_equip', *identity(shield))]], 0)
        self.assert_unchanged(before)

    def test_duplicate_purchase_and_sold_seen_item_are_not_executable(self):
        shield, _ = self.bad_chest_fixture()
        buy = ("buy_equip", *identity(shield))
        before = self.snapshot()
        result = self.one([buy, buy])
        self.assert_invalid(result, "duplicate_purchase", 1)
        self.assertEqual(result["gold_cost"], 90)
        self.assert_unchanged(before)
        self.assertTrue(self.buy(shield)["accepted"])
        bridge.step(1)
        bridge.probe_resource_process_game_packets()
        before = self.snapshot()
        self.assert_invalid(self.one([buy]), "stale_item", 0)
        self.assert_unchanged(before)

    def test_full_inventory_rejects_buy_and_storage_of_removed_chest(self):
        shield, chest = self.bad_chest_fixture()
        bridge.probe_resource_fill_inventory()
        before = self.snapshot()
        results = self.preview([[('buy_equip', *identity(shield))],
                                [('unequip', 6, *identity(chest))]])["results"]
        self.assert_invalid(results[0], "no_room", 0)
        self.assert_invalid(results[1], "no_room", 0)
        self.assert_unchanged(before)

    def test_inventory_source_index_survives_real_gold_compaction(self):
        self.bad_chest_fixture()
        chest = self.item("smith", 59, 35, seed=503, armor_class=12)
        own = self.item("inventory", 48, 15, seed=504, armor_class=3)
        source_index = own["index"]
        result = self.one([("buy_equip", *identity(chest)), ("equip", source_index, *identity(own))])
        self.assertTrue(result["valid"], result)
        own_command = next(c for c in result["expanded_commands"] if c[0] == "equip" and tuple(c[2:]) == identity(own))
        self.assertNotEqual(own_command[1], source_index, "fixture must actually compact a paid gold stack")
        self.execute(result["expanded_commands"])
        self.assert_projection_is_actual(result)

    def test_attribute_dependency_and_fixed_hp_loss_use_staged_player(self):
        self.bad_chest_fixture()
        bridge.probe_resource_recreate_equipment(0, 48, 0xB001, 1, 0, 15, bonus_strength=20)
        shield = self.item("smith", 73, 32, seed=505)
        self.assertTrue(shield["can_use"])
        head = bridge.observe()["equipped_items"][0]
        before = self.snapshot()
        failed = self.one([("unequip", 0, *identity(head)), ("buy_equip", *identity(shield))])
        self.assert_invalid(failed, "cannot_use", 1)
        self.assert_unchanged(before)
        bridge.probe_resource_recreate_equipment(0, 48, 0xB002, 1, 0, 15, bonus_hp_fixed=100 << 6)
        bridge.probe_resource_set_hp_fixed(1 << 6)
        head = bridge.observe()["equipped_items"][0]
        before = self.snapshot()
        self.assert_invalid(self.one([("unequip", 0, *identity(head))]), "unsafe_life", 0)
        self.assert_unchanged(before)

    def test_repair_retained_quotes_only_final_equipment_and_matches_commit(self):
        shield, chest = self.bad_chest_fixture()
        before = self.snapshot()
        self.assert_invalid(self.one([("repair_retained",)]), "retained_item_max_durability", 0)
        replacement = self.one([("buy_equip", *identity(shield)),
                                ("unequip", 6, *identity(chest)), ("repair_retained",)])
        self.assertTrue(replacement["valid"], replacement)
        self.assertFalse(any(c[0] == "repair" for c in replacement["expanded_commands"]))
        self.assertEqual(replacement["gold_cost"], 90)
        self.assert_unchanged(before)
        retained = self.one([("unequip", 6, *identity(chest)), ("repair_retained",)])
        self.assertTrue(retained["valid"], retained)
        repair = next(c for c in retained["expanded_commands"] if c[0] == "repair")
        self.assertEqual(repair[1], 5)
        self.assertEqual(tuple(repair[2:6]), (18603, 37538, 0, 2))
        quote = next(q for q in bridge.observe()["resource_state"]["town"]["repair_quotes"] if q["slot"] == 5)
        self.assertEqual((repair[6], repair[7]), (quote["durability"], quote["price"]))
        stale = ("repair", 5, *repair[2:6], repair[6], repair[7] + 1)
        self.assert_invalid(self.one([stale]), "stale_repair_quote", 0)
        self.execute(retained["expanded_commands"])
        self.assert_projection_is_actual(retained)

    def test_input_bounds_and_types_reject_without_truncation_or_effect(self):
        shield, _ = self.bad_chest_fixture()
        buy = ("buy_equip", *identity(shield))
        cases = [([], 0), ([[]] * 257, 0), ([[buy] * 9], 367),
                 ([[('repair_retained',), buy]], 367),
                 ([[('buy_equip', True, 649, 1030, 72)]], 367),
                 ([[('buy_equip', 17525.0, 649, 1030, 72)]], 367),
                 ([[('buy_equip', 65536, 649, 1030, 72)]], 367),
                 ([[('wait',)]], 367), ([[]], -1)]
        before = self.snapshot()
        for sequences, budget in cases:
            with self.subTest(sequences=sequences[:1], budget=budget):
                with self.assertRaises(ValueError):
                    bridge.preview_resource_equipment_combinations(sequences, budget)
                self.assert_unchanged(before)


if __name__ == "__main__":
    unittest.main()
