"""R21 native transaction fixtures, not gameplay/economic-effect evidence.

Only the independently built R21 bridge may execute these tests. Item-property,
value, gold, immunity and inventory-capacity fixtures are explicitly synthetic;
all tested pickups, sales, purchases and town travel use normal engine APIs.
"""
from __future__ import annotations

import unittest

import test_resource_native as fixtures
from test_resource_native import bridge, identity, nav
from diablogym import DiabloGymEnv


class LootResourceNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(bridge, "act_sell_inventory_item"):
            raise RuntimeError("test requires the isolated R21 loot bridge")
        cls.owner = DiabloGymEnv()

    @classmethod
    def tearDownClass(cls):
        bridge.end_game()
        bridge.configure_resource_protocol(False)
        cls.owner.close()

    def tearDown(self):
        bridge.end_game()
        bridge.configure_resource_protocol(False)

    def reset(self, seed=424711, *, loot=True, dungeon=False):
        bridge.end_game()
        bridge.configure_resource_protocol(True, ordinary_armor_scope=True,
            preserve_equipment_readiness=True, loot_economy=loot)
        raw = bridge.reset(seed)
        if dungeon:
            raw = nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            bridge.probe_invincible(True)
        return raw

    approach = fixtures.ResourceNativeTests.approach
    open_vendor = fixtures.ResourceNativeTests.open_vendor

    def snapshot(self):
        return (bridge.observe(), bridge.probe_resource_inventory_snapshot(),
                bridge.probe_resource_transition_snapshot())

    def assert_atomic_rejection(self, invoke, reason):
        bridge.probe_resource_process_game_packets()
        before = self.snapshot()
        receipt = invoke()
        self.assertFalse(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], reason)
        self.assertEqual(receipt["received"], 0)
        self.assertEqual(receipt["gold_before"], receipt["gold_after"])
        self.assertEqual(self.snapshot(), before)
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.snapshot(), before)  # Also catches queued mutation.
        return receipt

    def stand_on(self, item):
        target = item["x"], item["y"]
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
        self.fail("fixture item not reached within bounded normal walk")

    def loot(self, item):
        return bridge.act_pickup_loot_at(item["active_id"], item["x"], item["y"], *identity(item))

    def gear(self, item):
        return bridge.act_pickup_gear_at(item["active_id"], item["x"], item["y"], *identity(item))

    def inventory(self):
        return bridge.observe()["resource_state"]["inventory_state"]["items"]

    def floor(self, item):
        return next(x for x in bridge.observe()["floor_items"] if x["active_id"] == item["active_id"])

    def strong_sword(self):
        raw = bridge.observe()
        old = next(x for x in raw["equipped_items"] if x["present"] and x["item_type"] == 1)
        new = bridge.probe_spawn_test_gear(old["base_id"], 20, 30,
                                          durability=24, max_durability=24)
        self.stand_on(new)
        self.assertTrue(self.floor(new)["gear"])
        self.assertEqual(self.gear(new), 1)
        bridge.probe_resource_process_game_packets()
        return old, new

    def weak_club(self):
        item = bridge.probe_spawn_test_gear(int(bridge.IDI_WARRCLUB), 1, 1,
                                            durability=20, max_durability=20)
        self.stand_on(item)
        return item

    def enter_service(self):
        raw = bridge.observe()
        if raw["engine_level"] == 0:
            raw = nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
        bridge.probe_invincible(True)
        bridge.configure_town_service(True)
        upstairs = next(t for t in raw["triggers"] if t["msg"] == bridge.WM_DIABPREVLVL)
        raw, _ = nav.walk_to(bridge, upstairs["x"], upstairs["y"])
        self.assertEqual(raw["engine_level"], 0)
        self.assertTrue(raw["resource_state"]["service_trip"])
        return raw

    def quote(self, key):
        return next(x for x in bridge.observe()["resource_state"]["town"]["sell_quotes"]
                    if identity(x) == key)

    def sell(self, quote, *, index=None, expected_price=None, key=None):
        return bridge.act_sell_inventory_item("smith",
            quote["index"] if index is None else index,
            *(identity(quote) if key is None else key),
            quote["price"] if expected_price is None else expected_price)

    def test_default_off_raw_legacy_a14_deletion_and_new_api_rejection(self):
        self.reset(loot=False)
        self.assertNotIn("loot_economy", bridge.observe()["resource_state"])
        self.assertNotIn("inventory_state", bridge.observe()["resource_state"])
        old, new = self.strong_sword()
        self.assertNotIn(identity(old), {identity(i) for i in bridge.probe_resource_inventory_snapshot()["items"]})
        weak = self.weak_club()
        self.assert_atomic_rejection(lambda: self.loot(weak), "unavailable")
        self.assert_atomic_rejection(lambda: bridge.act_sell_inventory_item("smith", 0, *identity(old), 30), "unavailable")

    def test_flag_requires_resource_scope_and_no_live_toggle(self):
        raw = self.reset()
        self.assertIs(raw["resource_state"]["loot_economy"], True)
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            bridge.configure_resource_protocol(True, ordinary_armor_scope=True,
                preserve_equipment_readiness=True, loot_economy=False)
        self.assertEqual(self.snapshot(), before)
        bridge.end_game()
        with self.assertRaises(ValueError):
            bridge.configure_resource_protocol(False, loot_economy=True)
        with self.assertRaises(ValueError):
            bridge.configure_resource_protocol(True, loot_economy=True)

    def test_a14_retains_real_old_sword_with_no_gold_payout(self):
        self.reset()
        before = bridge.observe()
        old, new = self.strong_sword()
        raw = bridge.observe()
        retained = next(x for x in self.inventory() if identity(x) == identity(old))
        self.assertTrue(retained["retained_from_a14"])
        self.assertEqual(retained["durability"], old["durability"])
        self.assertEqual(retained["value"], 120)
        self.assertEqual(raw["gold"], before["gold"])
        self.assertNotIn(identity(old), {identity(x) for x in raw["floor_items"]})
        self.assertGreater(raw["gear_combat_utility"], before["gear_combat_utility"])
        receipt = raw["resource_state"]["last_a14_retention"]
        self.assertIn(list(identity(old)), [list(x) for x in receipt["retained_identities"]])
        self.assertEqual(receipt["gold_before"], receipt["gold_after"])

    def test_a14_full_inventory_candidate_and_stale_commit_reject(self):
        self.reset()
        sword = next(x for x in bridge.observe()["equipped_items"] if x["present"] and x["item_type"] == 1)
        candidate = bridge.probe_spawn_test_gear(sword["base_id"], 20, 30)
        self.stand_on(candidate)
        self.assertTrue(self.floor(candidate)["gear"])
        bridge.probe_resource_fill_inventory()
        bridge.probe_resource_process_game_packets()
        before = self.snapshot()
        self.assertFalse(self.floor(candidate)["gear"])
        self.assertEqual(self.gear(candidate), 0)
        self.assertEqual(self.snapshot(), before)
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.snapshot(), before)

    def test_two_hand_upgrade_retains_both_original_hands(self):
        self.reset()
        old = [x for x in bridge.observe()["equipped_items"][4:6] if x["present"]]
        self.assertEqual(len(old), 2)
        candidate = bridge.probe_spawn_test_gear(int(bridge.IDI_CLEAVER), 200, 255)
        self.stand_on(candidate)
        self.assertTrue(self.floor(candidate)["gear"])
        self.assertEqual(self.gear(candidate), 1)
        inventory = {identity(x): x for x in self.inventory()}
        for item in old:
            self.assertIn(identity(item), inventory)
            self.assertEqual(inventory[identity(item)]["durability"], item["durability"])
            self.assertTrue(inventory[identity(item)]["retained_from_a14"])

    def test_loot_pickup_keeps_body_gold_and_unidentified_state(self):
        self.reset()
        self.strong_sword()
        weak = self.weak_club()
        self.assertFalse(self.floor(weak)["gear"])
        before = bridge.observe()
        self.assertIn(weak["active_id"], [x["active_id"] for x in before["resource_state"]["loot_items"]])
        receipt = self.loot(weak)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["received"], 0)
        self.assertEqual(receipt["gold_before"], receipt["gold_after"])
        after = bridge.observe()
        self.assertEqual(after["equipped_items"], before["equipped_items"])
        self.assertEqual(after["gold"], before["gold"])
        carried = receipt["inventory_item"]
        self.assertIn(identity(carried), {identity(x) for x in self.inventory()})
        self.assertEqual(carried["identified"], self.floor_identity_before(weak, before)["identified"])
        self.assertNotIn(weak["active_id"], [x["active_id"] for x in after["floor_items"]])
        stable = bridge.probe_resource_inventory_snapshot()
        bridge.probe_resource_process_game_packets()
        self.assertEqual(bridge.probe_resource_inventory_snapshot(), stable)

    @staticmethod
    def floor_identity_before(item, raw):
        return next(x for x in raw["floor_items"] if x["active_id"] == item["active_id"])

    def test_loot_rejects_stale_identity_upgrade_quest_and_capacity(self):
        self.reset()
        sword = next(x for x in bridge.observe()["equipped_items"] if x["present"] and x["item_type"] == 1)
        upgrade = bridge.probe_spawn_test_gear(sword["base_id"], 20, 30)
        self.stand_on(upgrade)
        self.assert_atomic_rejection(lambda: self.loot(upgrade), "reserved_upgrade")
        quest = bridge.probe_spawn_test_gear(int(bridge.IDI_LAZSTAFF), 1, 1)
        self.stand_on(quest)
        self.assert_atomic_rejection(lambda: self.loot(quest), "not_sellable")
        weak = self.weak_club()
        bad = dict(weak, seed_lo=weak["seed_lo"] ^ 1)
        self.assert_atomic_rejection(lambda: self.loot(bad), "stale_item")
        bridge.probe_resource_fill_inventory()
        self.assert_atomic_rejection(lambda: self.loot(weak), "no_room")

    def prepare_sale_items(self):
        self.reset()
        old, _ = self.strong_sword()
        receipt = self.loot(self.weak_club())
        self.assertTrue(receipt["accepted"])
        club = receipt["inventory_item"]
        self.enter_service()
        self.open_vendor("smith")
        return old, club

    def test_real_sword30_plus_club5_credits_can_purchase_heal50(self):
        old, club = self.prepare_sale_items()
        gold = [x for x in self.inventory() if x["item_type"] == 11]
        self.assertEqual(len(gold), 1)
        bridge.probe_loot_inventory_value(gold[0]["index"], 20)
        self.assertEqual(bridge.observe()["gold"], 20)
        receipts = []
        for item, expected in [(old, 30), (club, 5)]:
            quote = self.quote(identity(item))  # Re-resolve after compaction.
            self.assertEqual(quote["price"], expected)
            before = bridge.observe()["gold"]
            receipt = self.sell(quote)
            self.assertTrue(receipt["accepted"], receipt)
            self.assertEqual(receipt["received"], expected)
            self.assertEqual(receipt["gold_after"] - receipt["gold_before"], expected)
            self.assertEqual(bridge.observe()["gold"], before + expected)
            self.assertNotIn(identity(item), {identity(x) for x in self.inventory()})
            receipts.append(receipt)
            bridge.probe_resource_process_game_packets()
            self.assertEqual(bridge.observe()["gold"], before + expected)
        self.assertEqual(sum(x["received"] for x in receipts), 35)
        self.assertEqual(bridge.observe()["gold"], 55)
        bridge.act_dismiss_dialog()
        bridge.step(1)
        raw, _ = self.open_vendor("healer")
        heal = next(x for x in raw["resource_state"]["town"]["stock"]
                    if x["heal_kind"] and x["price"] == 50)
        before = raw["resource_state"]["readiness"]["belt_heals"]
        result = bridge.act_buy_store_item("healer", heal["index"], *identity(heal))
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["price"], 50)
        self.assertEqual(bridge.observe()["gold"], 5)
        self.assertEqual(bridge.observe()["resource_state"]["readiness"]["belt_heals"], before + 1)

    def test_sell_price_identity_index_vendor_rejections_and_compaction(self):
        old, club = self.prepare_sale_items()
        q = self.quote(identity(old))
        self.assert_atomic_rejection(lambda: self.sell(q, expected_price=q["price"] + 1), "stale_item")
        key = identity(q)
        self.assert_atomic_rejection(lambda: self.sell(q, key=(key[0], key[1] ^ 1, key[2], key[3])), "stale_item")
        self.assert_atomic_rejection(lambda: self.sell(q, index=999), "stale_item")
        self.assert_atomic_rejection(lambda: bridge.act_sell_inventory_item("healer", q["index"], *key, q["price"]), "wrong_vendor")
        stale_club = self.quote(identity(club))
        self.assertTrue(self.sell(q)["accepted"])
        self.assert_atomic_rejection(lambda: self.sell(stale_club), "stale_item")
        fresh = self.quote(identity(club))
        self.assertNotEqual(fresh["index"], stale_club["index"])
        self.assertTrue(self.sell(fresh)["accepted"])
        self.assert_atomic_rejection(lambda: self.sell(fresh), "stale_item")

    def test_town_dungeon_protection_agrees_and_fast_weapon_is_not_sold(self):
        self.reset(dungeon=True)
        old, _ = self.strong_sword()
        fast = bridge.probe_spawn_test_gear(old["base_id"], 20, 30,
            effect_flags=int(bridge.ITEM_EFFECT_FASTEST_ATTACK),
            durability=24, max_durability=24)
        # This is an engineered inventory candidate, never free gameplay loot.
        bridge.probe_loot_carry_floor_item(fast["active_id"])
        keys = {"old": identity(old), "fast": identity(fast)}
        def flags():
            inventory = {identity(x): x for x in self.inventory()}
            return {name: inventory[key]["reserved_upgrade"] for name, key in keys.items()}
        self.assertEqual(flags(), {"old": False, "fast": True})
        self.enter_service()
        self.open_vendor("smith")
        before = self.snapshot()
        self.assertEqual(flags(), {"old": False, "fast": True})
        self.assertEqual(self.quote(keys["old"])["price"], 30)
        self.assertNotIn(keys["fast"], {identity(x) for x in bridge.observe()["resource_state"]["town"]["sell_quotes"]})
        self.assertEqual(self.snapshot(), before)
        protected = next(x for x in self.inventory() if identity(x) == keys["fast"])
        self.assert_atomic_rejection(lambda: self.sell(protected), "reserved_upgrade")

    def test_sell_no_room_is_zero_mutation_not_fake_wallet_credit(self):
        _, club = self.prepare_sale_items()
        current = self.quote(identity(club))
        # Deliberately extreme fixture value exceeds all physical gold capacity;
        # this is a failure-path test, not a claim that this Club exists in play.
        bridge.probe_loot_inventory_value(current["index"], 1000000)
        bridge.probe_resource_fill_inventory()
        quote = self.quote(identity(club))
        self.assertEqual(quote["price"], 250000)
        self.assertEqual(bridge.observe()["resource_state"]["inventory_state"]["free_cells"], 0)
        self.assert_atomic_rejection(lambda: self.sell(quote), "no_room")

    def test_full_inventory_observation_is_complete_and_queries_are_readonly(self):
        self.reset()
        bridge.probe_resource_fill_inventory()
        before = self.snapshot()
        for _ in range(3):
            state = bridge.observe()["resource_state"]
            carried = state["inventory_state"]
            actual = bridge.probe_resource_inventory_snapshot()
            self.assertEqual(carried["inventory_count"], len(actual["items"]))
            self.assertEqual(carried["grid"], actual["grid"])
            self.assertEqual(len(carried["grid"]), 40)
            self.assertEqual(carried["free_cells"], 0)
            self.assertEqual([identity(x) for x in carried["items"]], [identity(x) for x in actual["items"]])
            self.assertGreater(len(carried["items"]), len(state["inventory_equipment"]))
            self.assertEqual(state["town"]["sell_quotes"], [])
            self.assertEqual(self.snapshot(), before)

    def test_two_real_town_roundtrips_then_third_rejected_and_l2_gate_unchanged(self):
        raw = self.reset(dungeon=True)
        for expected in (1, 2):
            self.enter_service()
            state = bridge.observe()["resource_state"]
            self.assertEqual(state["service_trips_started"], expected)
            self.assertEqual(state["max_service_trips"], 2)
            raw = nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            self.assertEqual(raw["engine_level"], 1)
            self.assertFalse(raw["resource_state"]["service_trip"])
            self.assertFalse(raw["resource_state"]["service_authorized"])
        before = self.snapshot()
        with self.assertRaises(RuntimeError):
            bridge.configure_town_service(True)
        self.assertEqual(self.snapshot(), before)
        physical_before = bridge.probe_resource_transition_snapshot()
        bridge.probe_resource_transition(bridge.WM_DIABPREVLVL, 0)
        self.assertEqual(bridge.probe_resource_transition_snapshot(), physical_before)
        self.assertEqual(bridge.observe()["resource_state"]["transition"]["reason"], "town_service_trip_limit")
        bridge.probe_resource_transition(bridge.WM_DIABNEXTLVL, 2)
        self.assertEqual(bridge.probe_resource_transition_snapshot(), physical_before)
        self.assertFalse(bridge.observe()["resource_state"]["transition"]["accepted"])
        self.assertEqual(bridge.observe()["resource_state"]["transition"]["reason"], "not_ready")
        self.reset()
        self.assertEqual(bridge.observe()["resource_state"]["service_trips_started"], 0)
        self.assertEqual(bridge.observe()["resource_state"]["last_a14_retention"]["retained_identities"], [])


if __name__ == "__main__":
    unittest.main()
