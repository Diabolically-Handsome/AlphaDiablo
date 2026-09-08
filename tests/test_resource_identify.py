"""R18-H identify-v1 (凯恩鉴定) boundaries.

Two independent suites in one file, in the house shape of
``test_resource_portal.py``:

* ``Identify*Tests`` -- pure Python doubles, no native engine. The modules under
  test are loaded through a private package alias whose ``__path__`` points at
  ``python/diablogym``, so ``diablogym/__init__.py`` (and therefore the .so) is
  never imported.
* ``IdentifyNativeTests`` -- the real headless bridge, skipped unless the module
  actually carries the identify-v1 symbols (i.e. the isolated R18-H build). It
  freezes the DEFAULT-OFF observation and drives one real Cain transaction on a
  real unidentified magic item.
"""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock

PACKAGE = "_r18h_identify_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/diablogym")]
    sys.modules[PACKAGE] = package
if PACKAGE + ".bridge" not in sys.modules:
    # Only ``_walk``'s door branch reaches for the bridge; every path exercised
    # below keeps ``door`` false.
    sys.modules[PACKAGE + ".bridge"] = ModuleType(PACKAGE + ".bridge")

module = importlib.import_module(PACKAGE + ".resource_identify")
loot_module = importlib.import_module(PACKAGE + ".resource_sustain_loot")

IdentifyService = module.IdentifyService
IDENTIFY_PRICE = module.IDENTIFY_PRICE


class FakeBridge:
    """The identify leg grants no native authority; it only spends gold."""

    def __init__(self):
        self.calls = []

    def configure_town_service(self, enabled):
        self.calls.append(bool(enabled))


class FakeEnv:
    """The three attributes and one planner call the service asks of an env."""

    def __init__(self, value, steps=0, path=(), unguarded_path=None):
        self._raw = value
        self._steps = steps
        self.path = list(path)
        self.unguarded_path = self.path if unguarded_path is None else list(unguarded_path)
        self.plan_calls = []

    def _plan_descend_path(self, value, x, y, avoid_monsters=True):
        self.plan_calls.append((x, y, avoid_monsters))
        return list(self.path if avoid_monsters else self.unguarded_path)


def quote(*, index=0, equipped=False, seed_lo=1, base_id=1, value=120,
          identified_value=1860, sale=30, sale_identified=465, price=IDENTIFY_PRICE,
          identified=False, quality=1, is_quest=False, name="Short Sword"):
    return {"equipped": equipped, "index": index, "slot": index if equipped else -1,
            "seed_hi": 0, "seed_lo": seed_lo, "create_info": 204, "base_id": base_id,
            "name": name, "quality": quality, "identified": identified,
            "is_equipment": True, "is_quest": is_quest, "value": value,
            "identified_value": identified_value, "identify_price": price,
            "sale_price": sale, "sale_price_identified": sale_identified}


def town(*, npcs=None, quotes=(), vendor="none", dialog=False, stock=(), repairs=()):
    return {"npcs": [{"id": 3, "type": "storyteller", "x": 62, "y": 71}]
                     if npcs is None else list(npcs),
            "stock": list(stock), "repair_quotes": list(repairs),
            "active_vendor": vendor, "dialog_active": dialog,
            "identify_quotes": [dict(q) for q in quotes]}


def raw(*, depth=0, gold=1000, x=62, y=70, quotes=(), at_cain=True, enabled=True,
        belt_heals=4, required=4, dead=False, is_set_level=False, town_state=None,
        game_over=False, hp=100):
    state = {"enabled": True, "loot_economy": True,
             "identify_enabled": enabled, "identify_price": IDENTIFY_PRICE,
             "identify_at_storyteller": at_cain,
             "readiness": {"ready": True, "failures": [], "belt_heals": belt_heals,
                           "required_belt_heals": required, "belt_free_slots": 8},
             "town": town(quotes=quotes) if town_state is None else town_state}
    if not enabled:
        state.pop("identify_enabled")
        state.pop("identify_price")
        state.pop("identify_at_storyteller")
        state["town"].pop("identify_quotes", None)
    return {"dungeon_level": depth, "is_set_level": is_set_level, "dead": dead,
            "game_over": game_over, "player_x": x, "player_y": y, "gold": gold,
            "hp": hp, "max_hp": 100, "equipped_items": [], "resource_state": state}


def identity(item):
    return tuple(int(item[k]) for k in ("seed_hi", "seed_lo", "create_info", "base_id"))


def native_receipt(command, *, accepted=True, gold_before=1000, name="Short Sword of radiance",
                   price=None, reason="identified", sale_identified=465):
    price = command[-1] if price is None else price
    receipt = {"accepted": accepted, "reason": reason if accepted else "no_money",
               "equipped": bool(command[1]), "index": int(command[2]),
               "seed_hi": command[3], "seed_lo": command[4],
               "create_info": command[5], "base_id": command[6],
               "quoted_price": command[-1], "received": 0,
               "price": price if accepted else 0,
               "gold_before": gold_before,
               "gold_after": gold_before - (price if accepted else 0)}
    if accepted:
        receipt["index_after"] = int(command[2])
        receipt["item"] = dict(quote(index=int(command[2]), equipped=bool(command[1]),
                                     seed_lo=command[4], base_id=command[6],
                                     identified=True, price=None, name=name,
                                     sale=sale_identified, sale_identified=sale_identified))
    return receipt


# ------------------------------------------------------------------ validators
class IdentifyValidatorTests(unittest.TestCase):
    def test_the_protocol_vocabulary_is_closed(self):
        self.assertEqual(module.RESOURCE_IDENTIFY_PROTOCOLS, ("off", "cain-v1"))
        with self.assertRaises(ValueError):
            module.validate_identify_protocol("l2-town-v1", "sustain-loot-v1", "cain-v2")
        with self.assertRaises(ValueError):
            module.validate_identify_env("l2-town-v1", True, "on")

    def test_off_is_legal_under_every_arm(self):
        for protocol in ("off", "l2-town-v1"):
            for policy in ("legacy-v1", "sustain-loot-v1"):
                self.assertEqual(
                    module.validate_identify_protocol(protocol, policy, "off"), "off")
            for loot in (False, True):
                self.assertEqual(module.validate_identify_env(protocol, loot, "off"), "off")

    def test_cain_v1_requires_l2_town_v1_with_sustain_loot_v1(self):
        with self.assertRaises(ValueError):
            module.validate_identify_protocol("off", "sustain-loot-v1", "cain-v1")
        with self.assertRaises(ValueError):
            module.validate_identify_protocol("l2-town-v1", "sustain-v6", "cain-v1")
        self.assertEqual(
            module.validate_identify_protocol("l2-town-v1", "sustain-loot-v1", "cain-v1"),
            "cain-v1")

    def test_the_env_twin_requires_the_loot_economy(self):
        with self.assertRaises(ValueError):
            module.validate_identify_env("off", True, "cain-v1")
        with self.assertRaises(ValueError):
            module.validate_identify_env("l2-town-v1", False, "cain-v1")
        self.assertEqual(module.validate_identify_env("l2-town-v1", True, "cain-v1"), "cain-v1")

    def test_the_native_flag_must_match_the_python_contract(self):
        module.validate_native_identify({}, "off")
        module.validate_native_identify(raw(enabled=False), "off")
        module.validate_native_identify(raw(), "cain-v1")
        with self.assertRaises(RuntimeError):
            module.validate_native_identify({}, "cain-v1")
        with self.assertRaises(RuntimeError):
            module.validate_native_identify(raw(enabled=False), "cain-v1")
        with self.assertRaises(RuntimeError):
            module.validate_native_identify(raw(), "off")


# --------------------------------------------------------------------- reserve
class IdentifyReserveTests(unittest.TestCase):
    def test_the_medicine_floor_is_the_missing_heals_at_the_pinned_price(self):
        reserve = module.identify_reserve(raw(belt_heals=1, required=4))
        self.assertEqual(reserve["medicine_deficit"], 3)
        self.assertEqual(reserve["medicine_floor"], 3 * module.HEALER_MIN_HEAL_PRICE)
        self.assertEqual(reserve["reserve"], 150)
        self.assertIsNone(reserve["observed_minimum_total"])

    def test_a_full_belt_reserves_nothing(self):
        self.assertEqual(module.identify_reserve(raw())["reserve"], 0)

    def test_the_reserve_never_falls_below_the_medicine_floor(self):
        # The joint basket is unobservable at the earliest town phase (no vendor
        # is open yet), which is exactly when this leg runs.
        reserve = module.identify_reserve(raw(belt_heals=0, required=4))
        self.assertEqual(reserve["reserve"], 200)
        self.assertIn(reserve["source"], ("medicine_floor", "medicine_floor_plan_failed"))


# ------------------------------------------------------------------ candidates
class IdentifyCandidateTests(unittest.TestCase):
    def test_only_unidentified_non_quest_magic_items_are_listed(self):
        quotes = (quote(index=0),
                  quote(index=1, identified=True),
                  quote(index=2, quality=0),
                  quote(index=3, is_quest=True),
                  quote(index=4, price=None))
        listed = module.identify_candidates(raw(quotes=quotes))
        self.assertEqual([q["index"] for q in listed], [0])

    def test_candidates_are_ordered_by_expected_sale_gain(self):
        quotes = (quote(index=0, seed_lo=1, sale=30, sale_identified=60),
                  quote(index=1, seed_lo=2, sale=30, sale_identified=465),
                  quote(index=2, seed_lo=3, sale=30, sale_identified=200))
        listed = module.identify_candidates(raw(quotes=quotes))
        self.assertEqual([q["index"] for q in listed], [1, 2, 0])
        self.assertEqual([q["expected_sale_gain"] for q in listed], [435, 170, 30])


# ---------------------------------------------------------------- manager law
class IdentifyTriggerTests(unittest.TestCase):
    def setUp(self):
        self.service = IdentifyService()

    def test_the_leg_fires_once_per_town_trip_in_town(self):
        value = raw(quotes=(quote(),))
        self.assertEqual(self.service.trigger_reason(value, 10, 1), "identify")
        self.service.start(value, 10, "identify", 1)
        self.service._end(FakeEnv(value, steps=20), "identify_complete", failed=False)
        self.assertIsNone(self.service.trigger_reason(value, 30, 1))
        self.assertEqual(self.service.trigger_reason(value, 30, 2), "identify")

    def test_the_leg_never_fires_outside_town_or_when_the_flag_is_off(self):
        self.assertIsNone(self.service.trigger_reason(raw(depth=2, quotes=(quote(),)), 1, 1))
        self.assertIsNone(self.service.trigger_reason(
            raw(quotes=(quote(),), is_set_level=True), 1, 1))
        self.assertIsNone(self.service.trigger_reason(raw(enabled=False), 1, 1))
        self.assertIsNone(self.service.trigger_reason(raw(quotes=(quote(),), dead=True), 1, 1))
        self.assertIsNone(self.service.trigger_reason(raw(quotes=()), 1, 1))

    def test_the_potion_reserve_outranks_the_fee(self):
        # 240 gold with three heals missing: 150 is reserved, 100 would breach it.
        self.assertIsNone(self.service.trigger_reason(
            raw(gold=240, belt_heals=1, quotes=(quote(),)), 1, 1))
        self.assertEqual(self.service.trigger_reason(
            raw(gold=250, belt_heals=1, quotes=(quote(),)), 1, 1), "identify")

    def test_a_failed_leg_takes_a_cooldown(self):
        value = raw(quotes=(quote(),))
        self.service.start(value, 10, "identify", 1)
        self.service._end(FakeEnv(value, steps=20), "identify_cap", failed=True)
        self.assertIsNone(self.service.trigger_reason(value, 30, 2))
        self.assertEqual(
            self.service.trigger_reason(value, 20 + module.IDENTIFY_FAILURE_COOLDOWN, 2),
            "identify")


# ------------------------------------------------------------------- commands
class IdentifyCommandTests(unittest.TestCase):
    def started(self, value, steps=10, trip=1):
        service = IdentifyService()
        service.start(value, steps, "identify", trip)
        return service, FakeBridge()

    def test_an_open_store_page_is_dismissed_before_anything_else(self):
        value = raw(quotes=(quote(),),
                    town_state=town(quotes=(quote(),), vendor="smith"))
        service, bridge = self.started(value)
        self.assertEqual(service.command(FakeEnv(value, steps=11), bridge), ("dismiss",))

    def test_the_script_walks_to_cain_before_paying(self):
        value = raw(quotes=(quote(),), at_cain=False, x=40, y=40)
        service, bridge = self.started(value)
        env = FakeEnv(value, steps=11, path=[(41, 41, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 41, 41))
        self.assertEqual(service.phase, "approach")
        self.assertEqual(env.plan_calls[0][:2], (62, 70))

    def test_standing_at_cain_issues_the_exact_native_quote(self):
        item = quote(index=2, seed_lo=7)
        value = raw(quotes=(item,))
        service, bridge = self.started(value)
        command = service.command(FakeEnv(value, steps=11), bridge)
        self.assertEqual(command, ("identify", False, 2, 0, 7, 204, 1, IDENTIFY_PRICE))
        self.assertEqual(service.phase, "identify")

    def test_equipped_items_are_addressed_by_their_body_slot(self):
        item = quote(index=4, equipped=True, seed_lo=9)
        value = raw(quotes=(item,))
        service, bridge = self.started(value)
        self.assertEqual(service.command(FakeEnv(value, steps=11), bridge),
                         ("identify", True, 4, 0, 9, 204, 1, IDENTIFY_PRICE))

    def test_an_empty_list_hands_back_immediately(self):
        value = raw(quotes=())
        service = IdentifyService()
        service.start(value, 10, "identify", 1)
        self.assertEqual(service.command(FakeEnv(value, steps=11), FakeBridge()), ("complete",))
        self.assertEqual(service.reason, "identify_nothing_to_do")
        self.assertFalse(service.active)

    def test_the_reserve_stops_the_leg_and_records_what_it_refused(self):
        value = raw(gold=240, belt_heals=1, quotes=(quote(),))
        service = IdentifyService()
        service.start(value, 10, "identify", 1)
        self.assertEqual(service.command(FakeEnv(value, steps=11), FakeBridge()), ("complete",))
        self.assertEqual(service.reason, "identify_reserve_exhausted")
        self.assertEqual(len(service.skipped_for_reserve), 1)
        self.assertEqual(service.skipped_for_reserve[0]["identify_price"], IDENTIFY_PRICE)

    def test_no_failure_path_ever_returns_the_town_services_finish_word(self):
        value = raw(quotes=(quote(),))
        for reason, failed in (("identify_cap", True), ("identify_died", True),
                               ("identify_complete", False)):
            service = IdentifyService()
            service.start(value, 10, "identify", 1)
            self.assertEqual(service._end(FakeEnv(value, steps=20), reason, failed=failed),
                             ("complete",))

    def test_a_wrong_scene_ends_the_leg(self):
        value = raw(quotes=(quote(),))
        service, bridge = self.started(value)
        moved = raw(depth=1, quotes=(quote(),))
        self.assertEqual(service.command(FakeEnv(moved, steps=11), bridge), ("complete",))
        self.assertEqual(service.reason, "identify_unexpected_scene")

    def test_repeated_walk_rejections_give_up(self):
        value = raw(quotes=(quote(),), at_cain=False, x=40, y=40)
        service, bridge = self.started(value)
        env = FakeEnv(value, steps=11, path=[(41, 41, False)])
        for _ in range(module.IDENTIFY_WALK_REJECTIONS):
            command = service.command(env, bridge)
            service.receipt(command, {"accepted": False})
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "identify_walk_rejected")


# ------------------------------------------------------------------- receipts
class IdentifyReceiptTests(unittest.TestCase):
    def leg(self, item=None, gold=1000):
        item = quote(index=2, seed_lo=7) if item is None else item
        value = raw(gold=gold, quotes=(item,))
        service = IdentifyService()
        service.start(value, 10, "identify", 1)
        command = service.command(FakeEnv(value, steps=11), FakeBridge())
        return service, command

    def test_an_accepted_transaction_books_the_exact_fee_and_the_item(self):
        service, command = self.leg()
        service.receipt(command, native_receipt(command, gold_before=1000))
        self.assertEqual(service.identified_count, 1)
        self.assertEqual(service.identify_gold_spent, IDENTIFY_PRICE)
        row = service.identified_items[0]
        self.assertEqual(row["price"], IDENTIFY_PRICE)
        self.assertEqual(row["name"], "Short Sword of radiance")
        self.assertEqual(row["sale_price_identified"], 465)
        self.assertEqual(service.attempts[-1]["gold_spent"], IDENTIFY_PRICE)

    def test_a_rejected_transaction_moves_no_cash_and_is_recorded(self):
        service, command = self.leg()
        service.receipt(command, native_receipt(command, accepted=False))
        self.assertEqual(service.identified_count, 0)
        self.assertEqual(service.identify_gold_spent, 0)
        self.assertEqual(service.failures[-1]["reason"], "no_money")

    def test_a_wallet_that_disagrees_with_the_quote_is_fatal(self):
        service, command = self.leg()
        bad = native_receipt(command)
        bad["gold_after"] = bad["gold_before"] - 1
        with self.assertRaises(RuntimeError):
            service.receipt(command, bad)

    def test_a_receipt_for_another_item_is_fatal(self):
        service, command = self.leg()
        bad = native_receipt(command)
        bad["seed_lo"] = 999
        with self.assertRaises(RuntimeError):
            service.receipt(command, bad)

    def test_a_receipt_that_claims_income_is_fatal(self):
        service, command = self.leg()
        bad = native_receipt(command)
        bad["received"] = 5
        with self.assertRaises(RuntimeError):
            service.receipt(command, bad)

    def test_a_receipt_that_left_the_item_unidentified_is_fatal(self):
        service, command = self.leg()
        bad = native_receipt(command)
        bad["item"]["identified"] = False
        with self.assertRaises(RuntimeError):
            service.receipt(command, bad)

    def test_a_rejected_transaction_that_moved_cash_is_fatal(self):
        service, command = self.leg()
        bad = native_receipt(command, accepted=False)
        bad["gold_after"] = bad["gold_before"] - 100
        with self.assertRaises(RuntimeError):
            service.receipt(command, bad)

    def test_the_same_item_is_never_attempted_twice_in_one_leg(self):
        item = quote(index=2, seed_lo=7)
        value = raw(quotes=(item,))
        service = IdentifyService()
        service.start(value, 10, "identify", 1)
        command = service.command(FakeEnv(value, steps=11), FakeBridge())
        service.receipt(command, native_receipt(command, accepted=False))
        # The native list still shows it (the sale never happened), yet the leg
        # will not burn the whole budget retrying the same refusal.
        self.assertEqual(service.command(FakeEnv(value, steps=12), FakeBridge()), ("complete",))
        # R18-H review round: the state here is deterministic (the only receipt
        # was accepted=False), so assert the literal instead of deriving the
        # expectation from the counter the code under test maintains.
        self.assertEqual(service.identified_count, 0)
        self.assertEqual(service.reason, "identify_nothing_to_do")

    def test_later_sales_of_identified_items_are_attributed_to_the_leg(self):
        service, command = self.leg()
        service.receipt(command, native_receipt(command))
        key = tuple(command[3:7])
        self.assertEqual(service.note_sale(key, 465), 465)
        self.assertEqual(service.sale_income_identified, 465)
        self.assertEqual(service.identified_items[0]["sale_income"], 465)
        # An item this leg never touched contributes nothing.
        self.assertEqual(service.note_sale((0, 999, 204, 1), 300), 0)
        self.assertEqual(service.sale_income_identified, 465)


# ------------------------------------------------------------------ contract
class IdentifyServiceContractTests(unittest.TestCase):
    def test_the_service_exposes_the_members_the_resupply_loop_drives(self):
        service = IdentifyService()
        for name in ("active", "phase", "start_steps", "service_microstep_cap",
                     "settle_exempt", "command_microstep_deadline", "record_steps",
                     "command", "receipt", "telemetry", "trigger_reason", "start"):
            self.assertTrue(hasattr(service, name), name)
        self.assertFalse(service.settle_exempt)

    def test_the_command_deadline_is_the_leg_cap(self):
        service = IdentifyService()
        service.start(raw(quotes=(quote(),)), 400, "identify", 1)
        self.assertEqual(service.command_microstep_deadline,
                         400 + module.IDENTIFY_SERVICE_CAP)

    def test_record_steps_charges_the_phase_that_spent_them(self):
        value = raw(quotes=(quote(),))
        service = IdentifyService()
        service.start(value, 10, "identify", 1)
        service.record_steps(15, "approach")
        service.record_steps(20, "identify")
        self.assertEqual(service.phase_steps, {"approach": 5, "identify": 5})

    def test_telemetry_is_json_shaped_and_names_the_protocol(self):
        value = raw(quotes=(quote(index=2, seed_lo=7),))
        service = IdentifyService()
        service.start(value, 10, "identify", 1)
        command = service.command(FakeEnv(value, steps=11), FakeBridge())
        service.receipt(command, native_receipt(command))
        service.note_sale(tuple(command[3:7]), 465)
        telemetry = service.telemetry()
        self.assertEqual(telemetry["protocol"], "cain-v1")
        self.assertEqual(telemetry["price"], IDENTIFY_PRICE)
        for key in ("identified_count", "identify_gold_spent", "identified_items",
                    "sale_income_identified", "skipped_for_reserve", "failures",
                    "last_reserve", "attempts", "trips_served", "phase_steps"):
            self.assertIn(key, telemetry)
        self.assertEqual(telemetry["identified_count"], 1)
        self.assertEqual(telemetry["identify_gold_spent"], IDENTIFY_PRICE)
        self.assertEqual(telemetry["sale_income_identified"], 465)
        json.dumps(telemetry)

    def test_the_engine_price_constant_matches_the_python_one(self):
        self.assertEqual(module.IDENTIFY_PRICE, 100)
        header = (Path(__file__).resolve().parents[1] / "patches"
                  / "0014-headless-storyteller-identify.patch")
        # R18-H review round: a missing engine patch is a delivery defect, not a
        # reason to skip the price cross-check.
        self.assertTrue(header.is_file(), header)
        self.assertIn("StorytellerIdentifyPrice = 100", header.read_text(errors="ignore"))


# ------------------------------------------------- R18-H review round additions
class IdentifyPortalReserveTests(unittest.TestCase):
    """R18-F portal-v1 buys its 200-gold scroll LATER in the same town trip. The
    amount is always measured; whether it binds is the chairman's switch."""

    @staticmethod
    def armed(**kwargs):
        value = raw(**kwargs)
        value["resource_state"].update(portal_enabled=True, portal_scrolls=[],
                                       portals_started=0)
        return value

    def test_an_armed_portal_always_reports_what_the_scroll_would_cost(self):
        reserve = module.identify_reserve(self.armed())
        self.assertEqual(reserve["portal_scroll_reserve"], 200)

    def test_the_delivered_default_does_not_let_it_bind(self):
        self.assertFalse(module.IDENTIFY_RESERVE_PORTAL_SCROLL)
        reserve = module.identify_reserve(self.armed())
        self.assertEqual(reserve["reserve"], 0)
        self.assertFalse(reserve["portal_scroll_reserved"])

    def test_the_switch_makes_it_stack_on_the_medicine_floor(self):
        with mock.patch.object(module, "IDENTIFY_RESERVE_PORTAL_SCROLL", True):
            reserve = module.identify_reserve(self.armed(belt_heals=1, required=4))
            self.assertEqual(reserve["medicine_floor"], 150)
            self.assertEqual(reserve["reserve"], 350)
            self.assertTrue(reserve["portal_scroll_reserved"])

    def test_a_carried_scroll_or_a_spent_budget_reserves_nothing(self):
        carried = self.armed()
        carried["resource_state"]["portal_scrolls"] = [{"index": 0, "spell_from": 1}]
        self.assertEqual(module.identify_reserve(carried)["portal_scroll_reserve"], 0)
        spent = self.armed()
        spent["resource_state"]["portals_started"] = 2
        self.assertEqual(module.identify_reserve(spent)["portal_scroll_reserve"], 0)

    def test_portal_off_measures_nothing(self):
        reserve = module.identify_reserve(raw())
        self.assertEqual(reserve["portal_scroll_reserve"], 0)
        self.assertFalse(reserve["portal_scroll_reserved"])

    def test_under_the_switch_the_leg_refuses_to_eat_the_scroll(self):
        service = IdentifyService()
        with mock.patch.object(module, "IDENTIFY_RESERVE_PORTAL_SCROLL", True):
            # 260 gold, full belt, one candidate: the fee would leave 160 < 200.
            self.assertIsNone(service.trigger_reason(
                self.armed(gold=260, quotes=(quote(),)), 1, 1))
            self.assertEqual(service.trigger_reason(
                self.armed(gold=300, quotes=(quote(),)), 1, 1), "identify")
        # ... and with the delivered default it spends, as the H2 smoke measured.
        self.assertEqual(IdentifyService().trigger_reason(
            self.armed(gold=260, quotes=(quote(),)), 1, 1), "identify")


class IdentifyReserveSourceTests(unittest.TestCase):
    def test_no_open_vendor_page_is_never_reported_as_a_quoted_basket(self):
        reserve = module.identify_reserve(raw())
        self.assertEqual(reserve["source"], "medicine_floor")
        self.assertIsNone(reserve["observed_minimum_total"])

    def test_a_real_quoted_basket_is_still_honoured(self):
        stock = [{"vendor": "healer", "heal_kind": 1, "price": 70, "index": 0,
                  "can_use": True, "can_fit": True}]
        value = raw(belt_heals=2, required=4, town_state=town(stock=stock))
        reserve = module.identify_reserve(value)
        self.assertEqual(reserve["observed_minimum_total"], 140)
        self.assertEqual(reserve["reserve"], 140)
        self.assertEqual(reserve["source"], "joint_basket")


class IdentifyTripBudgetTests(unittest.TestCase):
    """The leg is charged to the town trip's own microstep budget, so it may
    never take the trip past the margin the itinerary still needs."""

    def value(self):
        return raw(quotes=(quote(),))

    def test_a_missing_trip_clock_leaves_the_leg_cap_alone(self):
        self.assertEqual(IdentifyService.leg_budget(None), module.IDENTIFY_SERVICE_CAP)

    def test_the_leg_only_gets_what_the_itinerary_can_lend(self):
        self.assertEqual(IdentifyService.leg_budget(3000), module.IDENTIFY_SERVICE_CAP)
        self.assertEqual(IdentifyService.leg_budget(1900),
                         1900 - module.IDENTIFY_TRIP_MARGIN)
        self.assertEqual(IdentifyService.leg_budget(1000), 0)

    def test_a_starved_trip_defers_the_leg_instead_of_starting_it(self):
        service = IdentifyService()
        self.assertIsNone(service.trigger_reason(self.value(), 10, 1, trip_budget_left=1600))
        self.assertEqual([d["reason"] for d in service.deferrals],
                         ["identify_deferred_trip_budget"])
        self.assertEqual(service.deferrals[0]["trip_budget_left"], 1600)
        self.assertEqual(service.deferrals[0]["leg_budget"], 100)

    def test_a_healthy_trip_still_runs_and_carries_its_narrowed_cap(self):
        service = IdentifyService()
        value = self.value()
        self.assertEqual(service.trigger_reason(value, 10, 1, trip_budget_left=2000),
                         "identify")
        service.start(value, 10, "identify", 1, trip_budget_left=2000)
        self.assertEqual(service.service_microstep_cap, 500)
        self.assertEqual(service.command_microstep_deadline, 510)
        self.assertEqual(service.attempts[-1]["trip_budget_left0"], 2000)

    def test_the_narrowed_cap_actually_ends_the_leg(self):
        service = IdentifyService()
        value = self.value()
        service.start(value, 0, "identify", 1, trip_budget_left=2000)
        service.record_steps(500)
        self.assertEqual(service.command(FakeEnv(value, steps=500), FakeBridge()),
                         ("complete",))
        self.assertEqual(service.reason, "identify_cap")

    def test_start_fails_closed_when_the_trip_cannot_lend_a_leg(self):
        with self.assertRaises(RuntimeError):
            IdentifyService().start(self.value(), 10, "identify", 1, trip_budget_left=1600)

    def test_an_unaffordable_trip_is_recorded_as_a_deferral(self):
        service = IdentifyService()
        service.trigger_reason(raw(gold=80, quotes=(quote(),)), 10, 1, trip_budget_left=3000)
        self.assertEqual([d["reason"] for d in service.deferrals],
                         ["identify_deferred_no_funds"])
        self.assertEqual(service.deferrals[0]["gold"], 80)
        self.assertEqual(service.deferrals[0]["candidates"], 1)
        json.dumps(service.telemetry())

    def test_a_trip_with_nothing_to_identify_is_not_a_deferral(self):
        service = IdentifyService()
        self.assertIsNone(service.trigger_reason(raw(gold=80), 10, 1, trip_budget_left=3000))
        self.assertEqual(service.deferrals, [])

    def test_one_deferral_row_per_town_trip(self):
        service = IdentifyService()
        value = raw(gold=80, quotes=(quote(),))
        for beat in (10, 11, 12):
            service.trigger_reason(value, beat, 1, trip_budget_left=3000)
        self.assertEqual(len(service.deferrals), 1)
        service.trigger_reason(value, 20, 2, trip_budget_left=3000)
        self.assertEqual(len(service.deferrals), 2)


class IdentifyInterceptLawTests(unittest.TestCase):
    """The manager-side law itself (OptionsEnv.step) is not reachable without a
    live engine, so its two load-bearing clauses are asserted on the source."""

    SOURCE = (Path(__file__).resolve().parents[1]
              / "python/diablogym/options_env.py").read_text(encoding="utf-8")

    def test_the_leg_is_gated_on_the_single_pre_sale_phase(self):
        self.assertIn('and getattr(service, "phase", None) == "outbound"\n'
                      "                        and identify.trigger_reason(", self.SOURCE)
        # "sell_idle" is the phase SustainLootService sells in; admitting it
        # would let a deferred leg fire after the first sale.
        self.assertNotIn('in ("outbound", "sell_idle")', self.SOURCE)

    def test_the_leg_is_handed_the_town_trips_own_clock(self):
        self.assertEqual(self.SOURCE.count("trip_budget_left=identify_trip_budget"), 2)
        self.assertIn("identify_trip_budget = (int(getattr(service, \"start_steps\", 0))",
                      self.SOURCE)


class IdentifyReceiptGuardTests(unittest.TestCase):
    def test_a_missing_native_audit_fails_closed(self):
        value = raw(quotes=(quote(),))
        service = IdentifyService()
        service.start(value, 10, "identify", 1)
        command = service.command(FakeEnv(value, steps=11), FakeBridge())
        # options_env hands {} whenever the beat carried no resource_action_audit.
        with self.assertRaises(RuntimeError):
            service.receipt(command, {})
        with self.assertRaises(RuntimeError):
            service.receipt(command, {"accepted": 1})


# -------------------------------------------------------------------- ledger
class IdentifyLedgerTests(unittest.TestCase):
    """The fee must land in the town trip's own ledger, exactly like the R18-F
    scroll purchase, or ``_seal_trip`` raises "unexplained cash movement"."""

    def loot_service(self, gold):
        service = loot_module.SustainLootService()
        service.observe(self.loot_raw(gold), 0)
        return service

    @staticmethod
    def loot_raw(gold):
        return {"gold": gold, "dungeon_level": 1, "is_set_level": False,
                "dead": False, "player_mode": 0, "char_level": 1, "xp": 0,
                "equipped_items": [],
                "resource_state": {
                    "enabled": True, "loot_economy": True,
                    "readiness": {"ready": True, "failures": [], "belt_heals": 4,
                                  "required_belt_heals": 4, "belt_free_slots": 4},
                    "inventory_state": {"items": []},
                    "loot_items": []}}

    def seal(self, *, book_identify):
        service = self.loot_service(1000)
        service.trip_count = 1
        service.start_steps = 0
        service._start_gold = 1000
        if book_identify:
            service.gold_spent += IDENTIFY_PRICE
        service.observe(self.loot_raw(1000 - IDENTIFY_PRICE), 10)
        service._seal_trip(10)
        return service._trips[-1]

    def test_the_booked_fee_reconciles_the_trip(self):
        record = self.seal(book_identify=True)
        self.assertEqual(record["cash_residual"], 0)
        self.assertEqual(record["cash_reconciliation"], "complete")
        self.assertEqual(record["gold_spent"], IDENTIFY_PRICE)

    def test_an_unbooked_fee_is_caught_as_unexplained_cash_movement(self):
        with self.assertRaises(RuntimeError):
            self.seal(book_identify=False)


# --------------------------------------------------------------------- native
def _load_native():
    root = Path(__file__).resolve().parents[1]
    if str(root / "python") not in sys.path:
        sys.path.insert(0, str(root / "python"))
    try:
        from diablogym import DiabloGymEnv, bridge, nav  # noqa: F401
    except Exception:  # pragma: no cover - no built bridge in this tree
        return None
    if not all(hasattr(bridge, name) for name in
               ("act_identify", "probe_resource_identify_fixture")):
        return None
    return DiabloGymEnv, bridge, nav


_NATIVE = _load_native()


@unittest.skipIf(_NATIVE is None, "requires the isolated R18-H identify bridge")
class IdentifyNativeTests(unittest.TestCase):
    """Engineering fixtures against the real headless engine. The default-off
    observation freeze is the house law this feature must not break."""

    @classmethod
    def setUpClass(cls):
        cls.DiabloGymEnv, cls.bridge, cls.nav = _NATIVE
        cls.owner = cls.DiabloGymEnv()

    @classmethod
    def tearDownClass(cls):
        cls.bridge.end_game()
        cls.bridge.configure_resource_protocol(False)
        cls.owner.close()

    def tearDown(self):
        self.bridge.end_game()
        self.bridge.configure_resource_protocol(False)

    def reset(self, seed=8001, *, identify=True):
        bridge = self.bridge
        bridge.end_game()
        bridge.configure_resource_protocol(
            True, ordinary_armor_scope=True, preserve_equipment_readiness=True,
            loot_economy=True, readiness_advisory=True, retreat=True, portal=True,
            identify=identify)
        return bridge.reset(seed)

    def enter_town_trip(self, raw_value):
        """One real L1 -> town service departure, the only way the trip flag is set."""
        bridge, nav = self.bridge, self.nav
        raw_value = nav.descend_to_dungeon(bridge)
        bridge.act_wait()
        raw_value = bridge.step(1)
        bridge.probe_invincible(True)
        bridge.configure_town_service(True)
        upstairs = next(t for t in raw_value["triggers"] if t["msg"] == bridge.WM_DIABPREVLVL)
        raw_value, _ = nav.walk_to(bridge, upstairs["x"], upstairs["y"])
        self.assertEqual(int(raw_value["dungeon_level"]), 0)
        self.assertTrue(raw_value["resource_state"]["service_trip"])
        return raw_value

    def spawn_unidentified_magic(self, raw_value, *, destination="inventory", slot=-1):
        """A REAL engine drop (SetupAllItems, onlygood/uper15) of an equippable
        base the warrior already carries; refused unless it came out magic and
        unidentified."""
        bases = sorted({int(i["base_id"]) for i in raw_value["equipped_items"]
                        if i.get("present")})
        for base in bases:
            for seed in range(1, 400):
                try:
                    return base, seed, self.bridge.probe_resource_identify_fixture(
                        destination, base, (seed >> 16) & 0xFFFF, seed & 0xFFFF, 12, slot)
                except RuntimeError:
                    continue
        self.fail("no seed produced an unidentified magic item")

    def walk_to_cain(self):
        state = self.bridge.observe()["resource_state"]
        cain = next(n for n in state["town"]["npcs"] if n["type"] == "storyteller")
        self.assertEqual((cain["x"], cain["y"]), module.STORYTELLER_POSITION)
        raw_value, _ = self.nav.walk_to(self.bridge, cain["x"], cain["y"] - 1)
        return raw_value, cain

    def test_default_off_hides_every_identify_key_and_closes_the_action(self):
        bridge = self.bridge
        raw_value = self.reset(identify=False)
        state = raw_value["resource_state"]
        for key in ("identify_enabled", "identify_price", "identify_at_storyteller"):
            self.assertNotIn(key, state)
        self.assertNotIn("identify_quotes", state["town"])
        self.assertNotIn("storyteller", [n["type"] for n in state["town"]["npcs"]])
        receipt = bridge.act_identify(False, 0, 0, 1, 204, 1, module.IDENTIFY_PRICE)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "unavailable")
        self.assertEqual(receipt["gold_before"], receipt["gold_after"])

    def test_identify_v1_requires_the_loot_economy(self):
        bridge = self.bridge
        bridge.end_game()
        with self.assertRaises(ValueError):
            bridge.configure_resource_protocol(True, ordinary_armor_scope=True,
                preserve_equipment_readiness=True, loot_economy=False,
                readiness_advisory=True, identify=True)
        with self.assertRaises(ValueError):
            bridge.configure_resource_protocol(False, identify=True)

    def test_the_flag_round_trips_and_publishes_cains_fixed_price(self):
        raw_value = self.reset()
        state = raw_value["resource_state"]
        self.assertIs(state["identify_enabled"], True)
        self.assertEqual(state["identify_price"], module.IDENTIFY_PRICE)
        self.assertIs(state["identify_at_storyteller"], False)
        cain = next(n for n in state["town"]["npcs"] if n["type"] == "storyteller")
        self.assertEqual((cain["x"], cain["y"]), module.STORYTELLER_POSITION)

    def test_a_real_magic_item_is_identified_for_the_fixed_fee(self):
        bridge = self.bridge
        raw_value = self.reset()
        raw_value = self.enter_town_trip(raw_value)
        base, seed, index = self.spawn_unidentified_magic(raw_value)
        bridge.probe_resource_add_gold(1000)
        bridge.step(1)

        before = bridge.observe()["resource_state"]["town"]["identify_quotes"]
        self.assertEqual(len(before), 1, before)
        item = before[0]
        self.assertFalse(item["identified"])
        self.assertNotEqual(item["quality"], 0)
        self.assertEqual(item["identify_price"], module.IDENTIFY_PRICE)
        # The whole economic case: an unidentified magic item sells at _ivalue/4.
        self.assertEqual(item["sale_price"], max(item["value"] // 4, 1))
        self.assertEqual(item["sale_price_identified"], max(item["identified_value"] // 4, 1))
        self.assertGreater(item["sale_price_identified"], item["sale_price"])

        # Not adjacent to Cain yet: the action is closed.
        refused = bridge.act_identify(False, int(item["index"]), int(item["seed_hi"]),
            int(item["seed_lo"]), int(item["create_info"]), int(item["base_id"]),
            int(item["identify_price"]))
        self.assertFalse(refused["accepted"])
        self.assertEqual(refused["reason"], "not_adjacent")
        self.assertEqual(refused["gold_before"], refused["gold_after"])

        self.walk_to_cain()
        state = bridge.observe()["resource_state"]
        self.assertIs(state["identify_at_storyteller"], True)
        quoted = state["town"]["identify_quotes"][0]
        gold_before = bridge.observe()["gold"]

        # A stale quote is refused without moving cash.
        stale = bridge.act_identify(False, int(quoted["index"]), int(quoted["seed_hi"]),
            int(quoted["seed_lo"]), int(quoted["create_info"]), int(quoted["base_id"]),
            int(quoted["identify_price"]) + 1)
        self.assertFalse(stale["accepted"])
        self.assertEqual(stale["reason"], "stale_item")
        self.assertEqual(bridge.observe()["gold"], gold_before)

        receipt = bridge.act_identify(False, int(quoted["index"]), int(quoted["seed_hi"]),
            int(quoted["seed_lo"]), int(quoted["create_info"]), int(quoted["base_id"]),
            int(quoted["identify_price"]))
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "identified")
        self.assertEqual(receipt["price"], module.IDENTIFY_PRICE)
        self.assertEqual(receipt["gold_before"], gold_before)
        self.assertEqual(receipt["gold_after"], gold_before - module.IDENTIFY_PRICE)
        self.assertEqual(bridge.observe()["gold"], gold_before - module.IDENTIFY_PRICE)
        self.assertEqual(receipt["received"], 0)

        identified = receipt["item"]
        self.assertTrue(identified["identified"])
        self.assertEqual(identified["sale_price"], item["sale_price_identified"])
        self.assertGreater(identified["sale_price"], item["sale_price"])
        # The list is now empty: Cain has nothing left to sell us.
        self.assertEqual(bridge.observe()["resource_state"]["town"]["identify_quotes"], [])

        # The engine really flipped _iIdentified on the carried item, not a copy.
        carried = next(i for i in bridge.observe()["resource_state"]["inventory_state"]["items"]
                       if i.get("empty") is not True
                       and (i["seed_hi"], i["seed_lo"], i["create_info"], i["base_id"])
                       == (item["seed_hi"], item["seed_lo"], item["create_info"], item["base_id"]))
        self.assertTrue(carried["identified"])
        self.assertEqual(carried["sell_price"], item["sale_price_identified"])

    def test_a_second_identification_of_the_same_item_is_refused(self):
        bridge = self.bridge
        raw_value = self.enter_town_trip(self.reset())
        self.spawn_unidentified_magic(raw_value)
        bridge.probe_resource_add_gold(1000)
        bridge.step(1)
        self.walk_to_cain()
        item = bridge.observe()["resource_state"]["town"]["identify_quotes"][0]
        key = (False, int(item["index"]), int(item["seed_hi"]), int(item["seed_lo"]),
               int(item["create_info"]), int(item["base_id"]), int(item["identify_price"]))
        self.assertTrue(bridge.act_identify(*key)["accepted"])
        gold = bridge.observe()["gold"]
        again = bridge.act_identify(*key)
        self.assertFalse(again["accepted"])
        self.assertIn(again["reason"], ("not_identifiable", "stale_item"))
        self.assertEqual(bridge.observe()["gold"], gold)

    def test_an_equipped_magic_item_is_identified_in_place(self):
        # R18-H review round: the single case was labelled INVLOC_CHEST but used
        # index 4 (INVLOC_HAND_LEFT). inv_body_loc is HEAD=0, RING_LEFT=1,
        # RING_RIGHT=2, AMULET=3, HAND_LEFT=4, HAND_RIGHT=5, CHEST=6 -- and the
        # UI listing order in ResourceIdentifyBodySlots is deliberately NOT that
        # order, so cover an armour slot and a ring slot as well as the weapon.
        for slot, name in ((6, "INVLOC_CHEST"), (1, "INVLOC_RING_LEFT"),
                           (4, "INVLOC_HAND_LEFT")):
            with self.subTest(slot=name):
                bridge = self.bridge
                raw_value = self.enter_town_trip(self.reset())
                self.spawn_unidentified_magic(raw_value, destination="body", slot=slot)
                bridge.probe_resource_add_gold(1000)
                bridge.step(1)
                self.walk_to_cain()
                quotes = bridge.observe()["resource_state"]["town"]["identify_quotes"]
                worn = next(q for q in quotes if q["equipped"] and q["slot"] == slot)
                self.assertEqual(worn["index"], slot)
                gold = bridge.observe()["gold"]
                receipt = bridge.act_identify(True, int(worn["index"]), int(worn["seed_hi"]),
                    int(worn["seed_lo"]), int(worn["create_info"]), int(worn["base_id"]),
                    int(worn["identify_price"]))
                self.assertTrue(receipt["accepted"], receipt)
                self.assertEqual(bridge.observe()["gold"], gold - module.IDENTIFY_PRICE)
                equipped = bridge.observe()["equipped_items"][slot]
                self.assertTrue(equipped["present"])
                self.assertTrue(equipped["identified"])

    def test_a_refusal_says_which_half_of_the_gate_closed(self):
        """R18-H review round: "not_adjacent" used to cover the law as well."""
        bridge = self.bridge
        raw_value = self.reset()          # in town, but NO authorized service trip
        self.assertIs(raw_value["resource_state"]["service_trip"], False)
        gold = bridge.observe()["gold"]
        refused = bridge.act_identify(False, 0, 0, 0, 0, 0, module.IDENTIFY_PRICE)
        self.assertFalse(refused["accepted"])
        self.assertEqual(refused["reason"], "unavailable")
        self.assertEqual(bridge.observe()["gold"], gold)

    def test_the_fee_is_refused_when_the_purse_cannot_pay_it(self):
        bridge = self.bridge
        raw_value = self.enter_town_trip(self.reset())
        self.spawn_unidentified_magic(raw_value)
        self.walk_to_cain()
        # The warrior starts with exactly Cain's fee; shrink the real gold stack
        # one gold short of it so the engine's own affordability check refuses.
        items = bridge.observe()["resource_state"]["inventory_state"]["items"]
        short = module.IDENTIFY_PRICE - 1
        for entry in items:
            if entry.get("empty") is True:
                continue
            bridge.probe_loot_inventory_value(int(entry["index"]), short)
            if int(bridge.observe()["gold"]) == short:
                break
            bridge.probe_loot_inventory_value(int(entry["index"]), int(entry["value"]))
        gold = int(bridge.observe()["gold"])
        self.assertEqual(gold, short)
        item = bridge.observe()["resource_state"]["town"]["identify_quotes"][0]
        receipt = bridge.act_identify(False, int(item["index"]), int(item["seed_hi"]),
            int(item["seed_lo"]), int(item["create_info"]), int(item["base_id"]),
            int(item["identify_price"]))
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "no_money")
        self.assertEqual(bridge.observe()["gold"], gold)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
