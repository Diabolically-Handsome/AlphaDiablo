"""Native engineering fixtures for l2-town-v1; never policy-effect evidence.

Experience, invincibility, belt and real inventory gold are deliberately injected
only here to isolate the admission/transaction rules. Navigation and purchases
use the real headless engine. The published candidate loader selects the build.
"""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import DiabloGymEnv, bridge  # noqa: E402
from diablogym import nav  # noqa: E402


def identity(item):
    return tuple(int(item[key]) for key in
                 ("seed_hi", "seed_lo", "create_info", "base_id"))


class ResourceNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(bridge, "configure_resource_protocol"):
            raise unittest.SkipTest("requires the isolated l2-town-v1 native build")
        cls.engine_owner = DiabloGymEnv()

    @classmethod
    def tearDownClass(cls):
        bridge.end_game()
        bridge.configure_resource_protocol(False)
        cls.engine_owner.close()

    def tearDown(self):
        bridge.end_game()
        bridge.configure_resource_protocol(False)

    def reset(self, seed=8001, *, dungeon=True):
        bridge.end_game()
        bridge.configure_resource_protocol(True)
        raw = bridge.reset(seed)
        if dungeon:
            raw = nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            bridge.probe_invincible(True)  # engineering navigation fixture
        return raw

    def ready_fixture(self):
        self.reset()
        bridge.probe_add_experience(2000)
        raw = bridge.step(1)
        self.assertEqual(raw["char_level"], 2)
        bridge.probe_bonus_ac(9 - raw["armor_class"])
        bridge.probe_resource_set_belt_heals(4)
        bridge.probe_resource_set_durability(15)
        bridge.probe_resource_set_hp_fixed(
            bridge.observe()["resource_state"]["readiness"]["max_hp_fixed"])
        ready = bridge.observe()["resource_state"]["readiness"]
        self.assertTrue(ready["ready"], ready)
        return ready

    def assert_rejected_unchanged(self, mode, target, reason):
        before = bridge.probe_resource_transition_snapshot()
        bridge.probe_resource_transition(mode, target)
        after = bridge.probe_resource_transition_snapshot()
        self.assertEqual(before, after)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], reason)
        return receipt

    def walk_to_town(self, seed=8001):
        raw = self.reset(seed)
        bridge.configure_town_service(True)
        upstairs = next(t for t in raw["triggers"]
                        if t["msg"] == bridge.WM_DIABPREVLVL)
        raw, ticks = nav.walk_to(bridge, upstairs["x"], upstairs["y"])
        self.assertEqual(raw["dungeon_level"], 0, (ticks, raw["player_x"], raw["player_y"]))
        self.assertTrue(raw["resource_state"]["service_trip"])
        return raw

    def approach(self, vendor):
        raw = bridge.observe()
        npc = next(n for n in raw["resource_state"]["town"]["npcs"]
                   if n["type"] == vendor)
        points = [(npc["x"] + dx, npc["y"] + dy)
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if dx or dy]
        points.sort(key=lambda p: max(abs(p[0] - raw["player_x"]),
                                     abs(p[1] - raw["player_y"])))
        for x, y in points:
            if not bridge.probe_tile(x, y)["walkable"]:
                continue
            raw, _ = nav.walk_to(bridge, x, y, max_ticks=4000)
            if max(abs(raw["player_x"] - npc["x"]),
                   abs(raw["player_y"] - npc["y"])) <= 1:
                bridge.act_wait()
                bridge.step(1)
                return npc
        self.fail(f"could not walk adjacent to {vendor}")

    def open_vendor(self, vendor):
        npc = self.approach(vendor)
        dialogs = 0
        for _ in range(6):
            town = bridge.observe()["resource_state"]["town"]
            if town["dialog_active"]:
                self.assertEqual(bridge.act_dismiss_dialog(), 1)
                bridge.step(1)
                dialogs += 1
                continue
            if town["active_vendor"] == vendor:
                return bridge.observe(), dialogs
            self.assertEqual(bridge.act_talk_towner(npc["id"]), 1)
            bridge.step(2)
        self.fail(f"{vendor} did not open after explicit dialog dismissal")

    def buy(self, item):
        return bridge.act_buy_store_item(item["vendor"], item["index"], *identity(item))

    def make_visible_gold_fixture(self):
        # Use the actual monster-death/drop code, rather than creating an item
        # with synthetic type/size fields or writing the player's gold counter.
        for seed in range(8001, 8017):
            raw = self.reset(seed)
            for monster in raw["monsters"]:
                if monster["hp"] <= 0 or not monster["visible"]:
                    continue
                bridge.probe_kill_monster(monster["id"])
                raw = bridge.step(1)
                if raw["resource_state"]["gold_items"]:
                    return raw["resource_state"]["gold_items"][0]
        self.fail("fixture seed pool did not produce a visible ordinary gold drop")

    def test_town_restock_is_reproducible_across_intervening_episodes(self):
        def two_town_visits(seed):
            self.walk_to_town(seed)
            first, _ = self.open_vendor("smith")
            first_state = first["resource_state"]
            self.assertEqual(first_state["town_restock_sequence"], 1)
            self.assertEqual(bridge.act_dismiss_dialog(), 1)
            bridge.step(1)
            raw = nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            bridge.configure_town_service(True)
            upstairs = next(t for t in raw["triggers"]
                            if t["msg"] == bridge.WM_DIABPREVLVL)
            raw, _ = nav.walk_to(bridge, upstairs["x"], upstairs["y"])
            self.assertEqual(raw["dungeon_level"], 0)
            second, _ = self.open_vendor("smith")
            second_state = second["resource_state"]
            self.assertEqual(second_state["town_restock_sequence"], 2)
            self.assertNotEqual(first_state["town_seed"], second_state["town_seed"])
            # Compare actual generated stock identities/stats and projected
            # purchase qualifications, not merely the seed bookkeeping.
            return [(state["town_seed"], state["town"]["stock"])
                    for state in (first_state, second_state)]

        original = two_town_visits(2114001)
        two_town_visits(2114002)
        self.assertEqual(original, two_town_visits(2114001))

    def test_default_off_retains_legacy_unready_descent(self):
        bridge.end_game()
        bridge.configure_resource_protocol(False)
        bridge.reset(8101)
        raw = nav.descend_to_dungeon(bridge)
        self.assertNotIn("resource_state", raw)
        bridge.probe_warp_main_level(2)
        raw = bridge.step(1)
        self.assertEqual(raw["dungeon_level"], 2)

    def test_denied_stair_portal_and_unauthorized_town_are_atomic(self):
        self.reset()
        self.assert_rejected_unchanged(bridge.WM_DIABNEXTLVL, 2, "not_ready")
        self.assert_rejected_unchanged(bridge.WM_DIABPREVLVL, 0, "town_service_not_authorized")
        self.assert_rejected_unchanged(bridge.WM_DIABWARPLVL, 0, "unsupported_portal_or_warp")
        self.assert_rejected_unchanged(bridge.WM_DIABRETOWN, 0, "unsupported_portal_or_warp")
        self.assert_rejected_unchanged(bridge.WM_DIABNEXTLVL, 3, "curriculum_boundary")

    def test_portal_contact_rejects_before_mode_path_or_network_request(self):
        self.reset(dungeon=False)
        bridge.probe_resource_town_portal_fixture()
        before = bridge.probe_resource_transition_snapshot()
        self.assertNotEqual(before["walkpath"][0], 0)
        bridge.probe_resource_process_town_portal()
        self.assertEqual(before, bridge.probe_resource_transition_snapshot())
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "unsupported_portal_or_warp")
        bridge.probe_resource_process_game_packets()
        self.assertEqual(before, bridge.probe_resource_transition_snapshot())
        self.assertEqual(receipt, bridge.observe()["resource_state"]["transition"],
                         "forbidden contact must not enqueue CMD_WARP")
        bridge.act_wait()
        raw = bridge.step(3)
        self.assertEqual(raw["dungeon_level"], 0)
        self.assertFalse(bridge.probe_resource_transition_snapshot()["changing"])
        self.assertNotEqual(bridge.probe_resource_transition_snapshot()["mode"], bridge.PM_NEWLVL)

    def test_restart_message_rejects_before_clearing_death_ui_state(self):
        self.reset()
        bridge.probe_resource_set_death_ui_flag(True)
        before = bridge.probe_resource_transition_snapshot()
        bridge.probe_resource_queue_restart_town()
        bridge.probe_resource_process_game_packets()
        self.assertEqual(before, bridge.probe_resource_transition_snapshot())
        self.assertTrue(bridge.probe_resource_transition_snapshot()["my_player_dead"])
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "unsupported_portal_or_warp")
        bridge.probe_resource_set_death_ui_flag(False)

    def test_rejected_set_level_probe_does_not_change_level_type(self):
        self.reset(dungeon=False)
        before = bridge.probe_resource_transition_snapshot()
        bridge.probe_enter_set_level(4)
        self.assertEqual(before, bridge.probe_resource_transition_snapshot())
        self.assertEqual(bridge.observe()["resource_state"]["transition"]["reason"], "curriculum_boundary")

    def test_readiness_fixed_point_potions_and_durability_boundaries(self):
        ready = self.ready_fixture()
        minimum_hp = math.ceil(4 * ready["max_hp_fixed"] / 5)
        bridge.probe_resource_set_hp_fixed(minimum_hp - 1)
        self.assertIn("health", bridge.observe()["resource_state"]["readiness"]["failures"])
        self.assert_rejected_unchanged(bridge.WM_DIABNEXTLVL, 2, "not_ready")
        bridge.probe_resource_set_hp_fixed(minimum_hp)
        self.assertTrue(bridge.observe()["resource_state"]["readiness"]["ready"])
        bridge.probe_resource_set_belt_heals(3)
        self.assertIn("potions", bridge.observe()["resource_state"]["readiness"]["failures"])
        bridge.probe_resource_set_belt_heals(4)
        bridge.probe_resource_set_durability(14)
        self.assertIn("durability", bridge.observe()["resource_state"]["readiness"]["failures"])
        for durability in (15, 255):
            bridge.probe_resource_set_durability(durability)
            self.assertTrue(bridge.observe()["resource_state"]["readiness"]["ready"])

    def test_allowed_descent_keeps_pretransition_readiness_receipt(self):
        self.ready_fixture()
        bridge.probe_resource_transition(bridge.WM_DIABNEXTLVL, 2)
        raw = bridge.step(1)
        self.assertEqual(raw["dungeon_level"], 2)
        receipt = raw["resource_state"]["transition"]
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["pretransition_ready"])
        self.assertEqual((receipt["source_depth"], receipt["target_depth"]), (1, 2))
        self.assertFalse(receipt["source_is_set"])
        self.assertFalse(receipt["target_is_set"])
        self.assert_rejected_unchanged(bridge.WM_DIABNEXTLVL, 3, "curriculum_boundary")

    def test_real_stair_trigger_uses_the_same_native_readiness_gate(self):
        raw = self.reset()
        downstairs = next(t for t in raw["triggers"] if t["msg"] == bridge.WM_DIABNEXTLVL)
        raw, _ = nav.walk_to(bridge, downstairs["x"], downstairs["y"], max_ticks=12000)
        self.assertEqual(raw["dungeon_level"], 1)
        self.assertEqual((raw["player_x"], raw["player_y"]),
                         (downstairs["x"], downstairs["y"]))
        receipt = raw["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "not_ready")
        self.assertFalse(bridge.probe_resource_transition_snapshot()["changing"])

    def test_episode_reset_clears_service_and_receipt(self):
        self.reset()
        bridge.configure_town_service(True)
        self.assert_rejected_unchanged(bridge.WM_DIABNEXTLVL, 2, "not_ready")
        raw = self.reset(dungeon=False)
        state = raw["resource_state"]
        self.assertFalse(state["service_authorized"])
        self.assertFalse(state["service_trip"])
        self.assertEqual(state["transition"]["sequence"], 0)
        self.assertEqual(state["max_main_depth_reached"], 0)

    def test_pepin_poison_dialog_requires_explicit_dismiss_and_then_heals(self):
        saw_poison_dialog = False
        for seed in range(8001, 8017):
            self.walk_to_town(seed)
            bridge.probe_set_current_hit_points(10)
            npc = self.approach("healer")
            self.assertEqual(bridge.act_talk_towner(npc["id"]), 1)
            raw = bridge.step(2)
            if not raw["resource_state"]["town"]["dialog_active"]:
                continue
            saw_poison_dialog = True
            self.assertEqual(raw["hp"], 10)
            raw = bridge.step(200)
            self.assertTrue(raw["resource_state"]["town"]["dialog_active"])
            self.assertEqual(bridge.act_dismiss_dialog(), 1)
            bridge.step(1)
            self.assertEqual(bridge.act_talk_towner(npc["id"]), 1)
            raw = bridge.step(2)
            self.assertEqual(raw["hp"], raw["max_hp"])
            self.assertEqual(raw["resource_state"]["town"]["active_vendor"], "healer")
            break
        self.assertTrue(saw_poison_dialog, "fixed fixture pool must include the poison-water quest")

    def test_real_store_purchase_failure_atomicity_and_readonly_quotes(self):
        self.walk_to_town()
        raw, _ = self.open_vendor("healer")
        bridge.probe_resource_set_belt_heals(0)
        item = next(s for s in raw["resource_state"]["town"]["stock"] if s["heal_kind"])
        # Empty the small starting wallet through genuine purchases if needed.
        for _ in range(20):
            if bridge.observe()["gold"] < item["price"]:
                break
            self.assertTrue(self.buy(item)["accepted"])
            bridge.step(1)
        before = (bridge.observe(), bridge.probe_resource_transition_snapshot())
        failure = self.buy(item)
        self.assertEqual(failure["reason"], "no_money")
        self.assertEqual(before, (bridge.observe(), bridge.probe_resource_transition_snapshot()))
        bridge.probe_resource_add_gold(5000)
        before = (bridge.observe(), bridge.probe_resource_transition_snapshot())
        wrong = bridge.act_buy_store_item("smith", item["index"], *identity(item))
        self.assertEqual(wrong["reason"], "wrong_vendor")
        stale_id = list(identity(item))
        stale_id[1] ^= 1
        stale = bridge.act_buy_store_item("healer", item["index"], *stale_id)
        self.assertEqual(stale["reason"], "stale_item")
        for _ in range(5):
            bridge.observe()
        self.assertEqual(before, (bridge.observe(), bridge.probe_resource_transition_snapshot()))
        bridge.probe_resource_set_belt_heals(0)
        before = bridge.observe()
        bought = self.buy(item)
        self.assertTrue(bought["accepted"], bought)
        raw = bridge.step(1)
        self.assertEqual(raw["gold"], before["gold"] - item["price"])
        self.assertEqual(raw["belt_heals"], 1)
        first_kinds = raw["belt_heal_kinds"]
        self.assertEqual(sum(bool(k) for k in first_kinds), 1)

    def repair(self, quote, *, seed=None, durability=None, price=None):
        return bridge.act_repair_equipped_item(
            quote["slot"], *(identity(quote) if seed is None else seed),
            quote["durability"] if durability is None else durability,
            quote["price"] if price is None else price)

    def test_equipped_repair_is_paid_readonly_quoted_and_rng_neutral(self):
        self.walk_to_town()
        bridge.probe_resource_set_durability(14)
        bridge.probe_resource_add_gold(5000)
        raw, _ = self.open_vendor("smith")
        state = raw["resource_state"]
        self.assertTrue(state["readiness"]["repair_service_needed"])
        self.assertEqual(state["readiness"]["required_belt_heals"], 4)
        quotes = state["town"]["repair_quotes"]
        self.assertTrue(quotes)
        before = (bridge.observe(), bridge.probe_resource_transition_snapshot())
        for _ in range(5):
            self.assertEqual(bridge.observe(), before[0])
        self.assertEqual(bridge.probe_resource_transition_snapshot(), before[1])
        for quote in quotes:
            self.assertTrue(quote["repair_needed_for_gate"])
            before = bridge.observe()
            old_rng = bridge.probe_resource_transition_snapshot()["rng"]
            receipt = self.repair(quote)
            self.assertEqual(receipt, {"accepted": True, "reason": "repaired", "price": quote["price"]})
            after = bridge.observe()
            self.assertEqual(after["gold"], before["gold"] - quote["price"])
            self.assertEqual(bridge.probe_resource_transition_snapshot()["rng"], old_rng)
            for field in ("hp", "max_hp", "mana", "max_mana", "armor_class", "xp", "char_level", "belt_heal_kinds"):
                self.assertEqual(after[field], before[field], field)
            slot = quote["slot"]
            old_item, repaired = before["equipped_items"][slot], after["equipped_items"][slot]
            self.assertEqual(identity(old_item), identity(repaired))
            self.assertEqual(repaired["durability"], old_item["max_durability"])
            self.assertEqual(repaired["max_durability"], old_item["max_durability"])
            bridge.step(1)
        self.assertFalse(bridge.observe()["resource_state"]["readiness"]["repair_service_needed"])
        self.assertNotIn("durability", bridge.observe()["resource_state"]["readiness"]["failures"])
        self.assertEqual(bridge.observe()["resource_state"]["town"]["repair_quotes"], [])

    def test_equipped_repair_failure_cases_leave_live_state_unchanged(self):
        self.walk_to_town()
        bridge.probe_resource_set_durability(14)
        raw, _ = self.open_vendor("healer")
        potion = next(s for s in raw["resource_state"]["town"]["stock"] if s["heal_kind"])
        bridge.probe_resource_set_belt_heals(0)
        for _ in range(20):
            if bridge.observe()["gold"] < potion["price"]:
                break
            self.assertTrue(self.buy(potion)["accepted"])
            bridge.step(1)
        self.assertEqual(bridge.observe()["gold"], 0)
        raw, _ = self.open_vendor("smith")
        quote = raw["resource_state"]["town"]["repair_quotes"][0]
        before = (bridge.observe(), bridge.probe_resource_transition_snapshot())
        self.assertEqual(self.repair(quote)["reason"], "no_money")
        self.assertEqual(before, (bridge.observe(), bridge.probe_resource_transition_snapshot()))
        bridge.probe_resource_add_gold(5000)
        before = (bridge.observe(), bridge.probe_resource_transition_snapshot())
        changed_identity = list(identity(quote)); changed_identity[1] ^= 1
        for receipt in (self.repair(quote, seed=changed_identity),
                        self.repair(quote, durability=quote["durability"] - 1),
                        self.repair(quote, price=quote["price"] + 1)):
            self.assertEqual(receipt["reason"], "stale_item")
            self.assertEqual(before, (bridge.observe(), bridge.probe_resource_transition_snapshot()))
        raw, _ = self.open_vendor("healer")
        self.assertEqual(raw["resource_state"]["town"]["repair_quotes"], [])
        before = (bridge.observe(), bridge.probe_resource_transition_snapshot())
        self.assertEqual(self.repair(quote)["reason"], "wrong_vendor")
        self.assertEqual(before, (bridge.observe(), bridge.probe_resource_transition_snapshot()))

    def test_bought_armor_is_equipped_and_old_armor_is_preserved(self):
        for seed in range(8001, 8017):
            self.walk_to_town(seed)
            bridge.probe_resource_add_gold(5000)
            raw, _ = self.open_vendor("smith")
            items = sorted((s for s in raw["resource_state"]["town"]["stock"]
                            if s["is_armor"] and s["can_use"] and s["upgrade"]
                            and s["price"] < raw["gold"]),
                           key=lambda s: (s["projected_armor_class"], s["price"]))
            for item in items:
                raw = bridge.observe()
                current = next((s for s in raw["resource_state"]["town"]["stock"]
                                if identity(s) == identity(item) and s["upgrade"]
                                and s["price"] <= raw["gold"]), None)
                if current is None:
                    continue
                previous_equipped = {identity(s) for s in raw["equipped_items"] if s["present"]}
                self.assertTrue(self.buy(current)["accepted"])
                raw = bridge.step(1)
                carried = next((s for s in raw["resource_state"]["inventory_items"]
                                if identity(s) == identity(current)), None)
                if carried is None:  # first armor may auto-equip an empty slot
                    continue
                paid_gold = raw["gold"]
                result = bridge.act_equip_inventory_item(carried["index"], *identity(carried))
                self.assertTrue(result["accepted"], result)
                raw = bridge.step(1)
                equipped = {identity(s) for s in raw["equipped_items"] if s["present"]}
                carried_ids = {identity(s) for s in raw["resource_state"]["inventory_items"]}
                self.assertIn(identity(current), equipped)
                self.assertTrue(previous_equipped - equipped)
                self.assertTrue((previous_equipped - equipped) <= carried_ids)
                self.assertEqual(raw["gold"], paid_gold)
                self.assertEqual(raw["armor_class"], current["projected_armor_class"])
                return
        self.fail("fixture pool did not exercise an actual purchased-armor swap")

    def test_authorized_town_round_trip_preserves_progress_and_depth(self):
        dropped = self.make_visible_gold_fixture()
        raw = bridge.observe()
        kill_total = raw["monster_kill_total"]
        self.assertGreater(kill_total, 0)
        xp = raw["xp"]
        gold = raw["gold"]
        bridge.configure_town_service(True)
        upstairs = next(t for t in raw["triggers"] if t["msg"] == bridge.WM_DIABPREVLVL)
        raw, _ = nav.walk_to(bridge, upstairs["x"], upstairs["y"])
        self.assertEqual(raw["dungeon_level"], 0)
        raw = nav.descend_to_dungeon(bridge)
        self.assertEqual(raw["dungeon_level"], 1)
        self.assertEqual(raw["monster_kill_total"], kill_total)
        self.assertEqual(raw["xp"], xp)
        self.assertEqual(raw["gold"], gold)
        self.assertIn(identity(dropped), {identity(item) for item in raw["floor_items"]})
        self.assertEqual(raw["resource_state"]["max_main_depth_reached"], 1)
        self.assertFalse(raw["resource_state"]["service_trip"])
        self.assertFalse(raw["resource_state"]["service_authorized"])
        receipt = raw["resource_state"]["transition"]
        self.assertEqual(receipt["reason"], "town_service_return")
        self.assertEqual((receipt["source_depth"], receipt["target_depth"]), (0, 1))

    def test_explicit_gold_pickup_checks_identity_and_uses_actual_inventory(self):
        dropped = self.make_visible_gold_fixture()
        raw, _ = nav.walk_to(bridge, dropped["x"], dropped["y"])
        self.assertEqual(raw["dungeon_level"], 1)
        self.assertEqual((raw["future_x"], raw["future_y"]),
                         (dropped["x"], dropped["y"]))
        old_gold = raw["gold"]
        before = (bridge.observe(), bridge.probe_resource_transition_snapshot())
        stale = list(identity(dropped))
        stale[1] ^= 1
        self.assertEqual(bridge.act_pickup_gold_at(dropped["active_id"], *stale), 0)
        self.assertEqual(before, (bridge.observe(), bridge.probe_resource_transition_snapshot()))
        self.assertEqual(bridge.act_pickup_gold_at(dropped["active_id"], *identity(dropped)), 1)
        raw = bridge.step(3)
        self.assertEqual(raw["gold"], old_gold + dropped["value"])
        self.assertNotIn(identity(dropped), {identity(item) for item in raw["floor_items"]})


    def unequip_snapshot(self):
        return (bridge.observe(), bridge.probe_resource_inventory_snapshot(),
                bridge.probe_resource_transition_snapshot())

    def test_unequip_r19_seed2114004_native_fixture_and_item_preservation(self):
        # Engineering fixture: reproduce the recorded identities and durability
        # from R19, not a claim that this setup is an uninjected policy replay.
        self.walk_to_town(2114004)
        bridge.probe_add_experience(5000)
        bridge.step(1)
        self.assertEqual(bridge.observe()["char_level"], 3)
        for slot, base, hi, lo, create, durability in (
            (0, 48, 7380, 20117, 258, 6),
            (5, 2, 33887, 2679, 0, 13),
            (6, 55, 22239, 24979, 257, 2),
        ):
            bridge.probe_resource_recreate_equipment(slot, base, hi, lo, create, durability)
        bridge.probe_resource_set_belt_heals(4)
        ready = bridge.observe()["resource_state"]["readiness"]
        bridge.probe_resource_set_hp_fixed(ready["max_hp_fixed"])
        raw, _ = self.open_vendor("smith")
        self.assertEqual(raw["armor_class"], 14)
        self.assertEqual((raw["equipped_items"][6]["durability"], raw["equipped_items"][6]["max_durability"]), (2, 6))
        repair_quotes = [q for q in raw["resource_state"]["town"]["repair_quotes"] if q["repair_needed_for_gate"]]
        self.assertEqual(sorted(q["price"] for q in repair_quotes), [2, 4])
        for quote in repair_quotes:
            self.assertTrue(self.repair(quote)["accepted"])
            bridge.step(1)
        raw = bridge.observe()
        candidate = next(q for q in raw["resource_state"]["unequip_candidates"] if q["slot"] == 6)
        self.assertTrue(candidate["can_fit"] and candidate["can_unequip"])
        self.assertTrue(candidate["removes_durability_failure"])
        self.assertTrue(candidate["projected_readiness"]["ready"])
        self.assertEqual(candidate["projected_readiness"]["armor_class"], 10)
        before = self.unequip_snapshot()
        for _ in range(4):
            self.assertEqual(self.unequip_snapshot(), before)
        receipt = bridge.act_unequip_equipped_item(6, *identity(candidate))
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["price"], 0)
        after = bridge.observe()
        self.assertEqual(after["resource_state"]["readiness"], candidate["projected_readiness"])
        self.assertEqual(receipt["readiness_after"], after["resource_state"]["readiness"])
        self.assertFalse(after["equipped_items"][6]["present"])
        carried = next(q for q in after["resource_state"]["inventory_equipment"] if identity(q) == identity(candidate))
        expected = dict(before[0]["equipped_items"][6]); expected.pop("present")
        actual = dict(carried); actual.pop("index")
        self.assertEqual(actual, expected)
        self.assertEqual(after["gold"], before[0]["gold"])
        self.assertEqual(bridge.probe_resource_transition_snapshot()["rng"], before[2]["rng"])
        self.assertEqual(len(bridge.probe_resource_inventory_snapshot()["items"]), len(before[1]["items"]) + 1)
        self.assertEqual(carried, receipt["inventory_item"])
        snapshot = self.unequip_snapshot()
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.unequip_snapshot(), snapshot)
        bridge.step(1)
        self.assertTrue(bridge.observe()["resource_state"]["readiness"]["ready"])
        self.assertEqual(bridge.observe()["resource_state"]["curriculum_max_depth"], 2)

    def test_unequip_rejects_stale_unsupported_and_full_inventory_atomically(self):
        self.walk_to_town()
        raw = bridge.observe()
        shield = raw["equipped_items"][5]
        before = self.unequip_snapshot()
        stale = list(identity(shield)); stale[1] ^= 1
        self.assertEqual(bridge.act_unequip_equipped_item(5, *stale)["reason"], "stale_item")
        weapon = raw["equipped_items"][4]
        self.assertEqual(bridge.act_unequip_equipped_item(4, *identity(weapon))["reason"], "invalid_item")
        self.assertEqual(self.unequip_snapshot(), before)
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.unequip_snapshot(), before)
        bridge.probe_resource_fill_inventory()
        candidate = next(q for q in bridge.observe()["resource_state"]["unequip_candidates"] if q["slot"] == 5)
        self.assertFalse(candidate["can_fit"])
        self.assertFalse(candidate["can_unequip"])
        before = self.unequip_snapshot()
        self.assertEqual(bridge.act_unequip_equipped_item(5, *identity(shield))["reason"], "no_room")
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.unequip_snapshot(), before)

    def test_unequip_native_stat_dependency_and_unsafe_life(self):
        self.walk_to_town()
        # Native requirement dependency: an ordinary Claymore requires 35 STR;
        # the fixture adds a +20 STR cap to the starting 30-STR warrior.
        bridge.probe_resource_recreate_equipment(0, 48, 1, 2, 0, 15, 20, 0)
        bridge.probe_resource_recreate_equipment(4, 122, 3, 4, 0, 36)
        raw = bridge.observe()
        self.assertTrue(raw["equipped_items"][4]["stat_usable"])
        head = next(q for q in raw["resource_state"]["unequip_candidates"] if q["slot"] == 0)
        self.assertTrue(head["can_unequip"])
        self.assertFalse(head["projected_readiness"]["weapon_equipped"])
        self.assertTrue(bridge.act_unequip_equipped_item(0, *identity(head))["accepted"])
        after = bridge.observe()
        self.assertFalse(after["equipped_items"][4]["stat_usable"])
        self.assertEqual(after["resource_state"]["readiness"], head["projected_readiness"])
        # A second isolated fixture proves that removing an HP bonus cannot
        # commit a fatal operation or publish network messages on rejection.
        self.walk_to_town()
        bridge.probe_resource_recreate_equipment(0, 48, 1, 2, 0, 15, 0, 100 << 6)
        bridge.probe_resource_set_hp_fixed(1 << 6)
        head = next(q for q in bridge.observe()["resource_state"]["unequip_candidates"] if q["slot"] == 0)
        self.assertEqual(head["reason"], "unsafe_life")
        self.assertFalse(head["can_unequip"])
        before = self.unequip_snapshot()
        self.assertEqual(bridge.act_unequip_equipped_item(0, *identity(head))["reason"], "unsafe_life")
        bridge.probe_resource_process_game_packets()
        self.assertEqual(self.unequip_snapshot(), before)

    def test_unequip_candidates_absent_in_dungeon_and_default_off(self):
        raw = self.reset()
        self.assertEqual(raw["resource_state"]["unequip_candidates"], [])
        before = self.unequip_snapshot()
        self.assertEqual(bridge.act_unequip_equipped_item(5, *identity(raw["equipped_items"][5]))["reason"], "unavailable")
        self.assertEqual(self.unequip_snapshot(), before)
        bridge.end_game()
        bridge.configure_resource_protocol(False)
        raw = bridge.reset(8001)
        self.assertNotIn("resource_state", raw)
        before = self.unequip_snapshot()
        self.assertEqual(bridge.act_unequip_equipped_item(5, *identity(raw["equipped_items"][5]))["reason"], "unavailable")
        self.assertEqual(self.unequip_snapshot(), before)

    def test_unequip_head_and_shield_are_preserved_even_when_gate_worsens(self):
        self.walk_to_town()
        bridge.probe_resource_recreate_equipment(0, 48, 1, 2, 0, 15)
        for slot in (0, 5):
            raw = bridge.observe()
            candidate = next(q for q in raw["resource_state"]["unequip_candidates"] if q["slot"] == slot)
            self.assertTrue(candidate["can_unequip"])
            before_gold = raw["gold"]
            receipt = bridge.act_unequip_equipped_item(slot, *identity(candidate))
            self.assertTrue(receipt["accepted"])
            raw = bridge.observe()
            self.assertFalse(raw["equipped_items"][slot]["present"])
            self.assertEqual(raw["gold"], before_gold)
            self.assertEqual(raw["resource_state"]["readiness"], candidate["projected_readiness"])
            self.assertEqual(sum(identity(q) == identity(candidate) for q in raw["resource_state"]["inventory_equipment"]), 1)
            bridge.step(1)


if __name__ == "__main__":
    unittest.main()
