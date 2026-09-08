"""R18-K (2026-09-07) smith-v1 boundaries: pure Python doubles, no native engine.

Same house pattern as ``test_resource_retreat.py``: the modules under test are
loaded through a private package alias whose ``__path__`` points at
``python/diablogym`` plus a stub ``bridge`` submodule, so nothing here imports
the extension, resets the engine or constructs DiabloGymEnv/OptionsEnv.

The synthetic town states below set ``weapon_purchase_enabled`` because the law
fails closed without it.  A bridge older than R18-K2b does NOT set it: Griswold's
counter is armor-only there (``ActBuyStoreItem`` and ``PlanResourceArmor`` both
answer a weapon with ``unsupported_item``), which is exactly what the fail-closed
tests assert.

``NativeWeaponPurchaseTests`` at the bottom is the only class here that touches
the extension: it walks a real seed-8001 town visit on the isolated R18-K2b
bridge and checks the whole native transaction (flag off refuses, flag on buys
and wears a real one-handed weapon and the readiness damage rises, two-handed
weapons stay refused).  It skips itself on any bridge that lacks the new entry
point, so the pure boundaries above still run everywhere.
"""
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

PACKAGE = "_r18k_weapon_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/diablogym")]
    sys.modules[PACKAGE] = package
if PACKAGE + ".bridge" not in sys.modules:
    bridge_stub = ModuleType(PACKAGE + ".bridge")
    bridge_stub.WM_DIABPREVLVL = 1027
    bridge_stub.WM_DIABNEXTLVL = 1026
    sys.modules[PACKAGE + ".bridge"] = bridge_stub

module = importlib.import_module(PACKAGE + ".resource_weapon_upgrade")
sustain = importlib.import_module(PACKAGE + ".resource_sustain")
loot = importlib.import_module(PACKAGE + ".resource_sustain_loot")

WeaponUpgradeService = module.WeaponUpgradeService
plan_weapon_upgrade = module.plan_weapon_upgrade
validate_weapon_upgrade = module.validate_weapon_upgrade
validate_native_weapon_upgrade = module.validate_native_weapon_upgrade
identity = module.identity      # the four-field native item identity

SMITH_ID, HEALER_ID = 3, 5
SMITH_XY, HEALER_XY, PLAYER_XY = (75, 67), (76, 68), (75, 68)


# ------------------------------------------------------------------- fixtures
def stock_item(index, *, base_id, item_type, price, max_damage, min_damage=1,
               equip_loc=1, can_use=True, can_fit=True, durable=True, quality=0,
               upgrade=True, vendor="smith", **extra):
    item = {"vendor": vendor, "index": index, "seed_hi": 100 + index,
            "seed_lo": 200 + index, "create_info": 1030, "base_id": base_id,
            "price": price, "item_type": item_type, "equip_loc": equip_loc,
            "min_damage": min_damage, "max_damage": max_damage,
            "can_use": can_use, "can_fit": can_fit, "durable": durable,
            "quality": quality, "effects_active": True, "upgrade": upgrade,
            "heal_kind": 0, "armor_class": 0, "is_armor": False,
            "is_ordinary_armor": False, "can_equip": False,
            "equip_reason": "unsupported_item", "meets_armor_gate": False,
            "durability": 30, "max_durability": 30}
    item.update({name: 0 for name in module.DAMAGE_AFFIX_FIELDS})
    item.update(extra)
    return item


def native_stock_item(index, **kwargs):
    """A smith row as the R18-K2b bridge exports it while the scope is ON.

    R18-K review round: ``smith-v1`` runs with weapon-purchase-v1 on and now
    DEMANDS both native verdicts (``require_native``), so every executable-arm
    fixture must carry them; a row without them is what an older bridge (or the
    ``dry-v1`` twin) exports and is refused with ``native_scope_missing``.
    """
    kwargs.setdefault("is_upgrade_weapon", True)
    kwargs.setdefault("can_equip", True)
    kwargs.setdefault("equip_reason", "ready")
    return stock_item(index, **kwargs)


def equipped(slot, *, base_id=1, max_damage=6, min_damage=2, present=True,
             item_type=1, quality=0, **affixes):
    item = {"present": present, "seed_hi": 1, "seed_lo": slot, "create_info": 7,
            "base_id": base_id, "durability": 24, "max_durability": 24,
            "min_damage": min_damage, "max_damage": max_damage,
            "item_type": item_type, "quality": quality, "effects_active": True}
    item.update({name: 0 for name in module.DAMAGE_AFFIX_FIELDS})
    item.update(affixes)
    return item


def town_raw(*, gold=400, stock=(), active_vendor=None, damage=7,
             weapon_scope=True, belt_heals=4, failures=("level",)):
    equipment = [{"present": False} for _ in range(7)]
    equipment[4] = equipped(4)                       # the starting short sword
    equipment[5] = equipped(5, base_id=2, max_damage=0, min_damage=0,
                            item_type=5)             # buckler (ItemType::Shield)
    state = {"enabled": True, "protocol": "l2-town-v1", "ordinary_armor_scope": True,
             "town_seed": 11, "town_restock_sequence": 0,
             "readiness": {"ready": False, "failures": list(failures),
                           "required_belt_heals": 4, "belt_heals": belt_heals,
                           "belt_free_slots": 8 - belt_heals, "clvl": 3,
                           "armor_class": 11, "damage": damage,
                           "repair_service_needed": False,
                           "min_finite_durability": 24, "weapon_equipped": True},
             "town": {"active_vendor": active_vendor, "dialog_active": False,
                      "npcs": [{"id": SMITH_ID, "type": "smith",
                                "x": SMITH_XY[0], "y": SMITH_XY[1]},
                               {"id": HEALER_ID, "type": "healer",
                                "x": HEALER_XY[0], "y": HEALER_XY[1]}],
                      "stock": list(stock), "repair_quotes": []},
             "inventory_items": [], "unequip_candidates": [], "gold_items": [],
             "inventory_equipment": []}
    if weapon_scope:
        state["weapon_purchase_enabled"] = True
    return {"gold": gold, "hp": 60, "max_hp": 60, "dungeon_level": 0,
            "is_set_level": False, "dead": False, "game_over": False,
            "player_x": PLAYER_XY[0], "player_y": PLAYER_XY[1],
            "monsters": [], "triggers": [], "missiles": [],
            "equipped_items": equipment, "resource_state": state}


def loot_raw(*, gold=400, depth=1, failures=("armor",)):
    """The shape ``SustainLootService`` needs to open and seal a real trip.

    Copied from ``tests/test_resource_sustain_loot.py`` so the ledger tests below
    drive the shipped trip machinery (``maybe_start`` / ``_seal_trip``) instead of
    a re-implementation of it.
    """
    return dict(gold=gold, dungeon_level=depth, is_set_level=False, dead=False,
                player_mode=0, player_x=10, player_y=10, hp=70, max_hp=70,
                xp=100, char_level=3, equipped_items=[],
                monsters=[dict(hp=10, type=1)],
                resource_state=dict(
                    enabled=True, ordinary_armor_scope=True,
                    preserve_equipment_readiness=True, loot_economy=True,
                    town_seed=23, town_restock_sequence=1,
                    readiness=dict(ready=not failures, failures=list(failures),
                                   hp_fixed=4480, max_hp_fixed=4480, hp=70, max_hp=70,
                                   clvl=3, belt_heals=4, required_belt_heals=4,
                                   belt_free_slots=4),
                    inventory_state=dict(items=[], inventory_count=0, grid=[0] * 40,
                                         free_cells=40, held_empty=True),
                    loot_items=[], gold_items=[], inventory_items=[],
                    town=dict(active_vendor="smith", dialog_active=False, stock=[],
                              sell_quotes=[], repair_quotes=[])))


class FakeEnv:
    """Only the attributes a town-side service actually reads."""

    def __init__(self, raw, steps=1000):
        self._raw = raw
        self._steps = steps

    def _plan_descend_path(self, raw, x, y, avoid_monsters=True):
        raise AssertionError("the fixtures keep every vendor adjacent")


class FakeBridge:
    WM_DIABPREVLVL = 1027
    WM_DIABNEXTLVL = 1026

    def __init__(self):
        self.town_service = []

    def configure_town_service(self, enabled):
        self.town_service.append(bool(enabled))


# ------------------------------------------------------------------ validators
class ValidatorTests(unittest.TestCase):
    def test_off_is_accepted_everywhere(self):
        for protocol in ("off", "l2-town-v1"):
            for policy in ("legacy-v1", "sustain-v6", "sustain-loot-v1"):
                self.assertEqual(
                    validate_weapon_upgrade(protocol, policy, "full", "off"), "off")

    def test_smith_v1_requires_the_loot_itinerary(self):
        self.assertEqual(validate_weapon_upgrade(
            "l2-town-v1", "sustain-loot-v1", "full", "smith-v1"), "smith-v1")
        for protocol, policy, mode in (("off", "sustain-loot-v1", "full"),
                                       ("l2-town-v1", "sustain-v6", "full"),
                                       ("l2-town-v1", "legacy-v1", "full"),
                                       ("l2-town-v1", "sustain-loot-v1", "armor")):
            with self.assertRaises(ValueError):
                validate_weapon_upgrade(protocol, policy, mode, "smith-v1")

    def test_unknown_value_is_refused(self):
        for value in ("smith", "smith-v2", "on", True, None, ""):
            with self.assertRaises(ValueError):
                validate_weapon_upgrade("l2-town-v1", "sustain-loot-v1", "full", value)

    def test_native_scope_fails_closed_in_both_directions(self):
        """Both directions are reachable at runtime (R18-K review round).

        ``gResourceWeaponPurchase`` is a process-wide native flag, so the reverse
        direction -- scope ON while this env's flag is off -- is what proves a
        frozen arm never inherited the scope from another env in the same worker.
        ``DiabloGymEnv._validate_native_resource_flags`` therefore checks it on
        EVERY observation, next to ``validate_native_retreat`` /
        ``validate_native_portal``, not only on smith-v1 episodes.
        """
        without = town_raw(weapon_scope=False)
        validate_native_weapon_upgrade(without, "off")
        with self.assertRaises(RuntimeError):
            validate_native_weapon_upgrade(without, "smith-v1")
        with_scope = town_raw(weapon_scope=True)
        validate_native_weapon_upgrade(with_scope, "smith-v1")
        with self.assertRaises(RuntimeError):
            validate_native_weapon_upgrade(with_scope, "off")

    def test_native_scope_refuses_a_non_boolean_acknowledgement(self):
        raw = town_raw()
        raw["resource_state"]["weapon_purchase_enabled"] = 1
        with self.assertRaises(RuntimeError):
            validate_native_weapon_upgrade(raw, "smith-v1")

    def test_the_r18_bridge_receipt_is_a_refusal(self):
        """The shipped bridge exports no acknowledgement at all."""
        raw = town_raw(weapon_scope=False)
        self.assertNotIn("weapon_purchase_enabled", raw["resource_state"])
        with self.assertRaises(RuntimeError) as error:
            validate_native_weapon_upgrade(raw, "smith-v1")
        self.assertIn("unsupported_item", str(error.exception))


# --------------------------------------------------------------- choice rule
class SelectionTests(unittest.TestCase):
    def plan(self, stock, **kwargs):
        return plan_weapon_upgrade(town_raw(**kwargs.pop("raw", {})), stock, **kwargs)

    def test_maximum_projected_damage_gain_wins(self):
        stock = [stock_item(0, base_id=119, item_type=1, price=120, max_damage=8),
                 stock_item(1, base_id=137, item_type=4, price=300, max_damage=10),
                 stock_item(2, base_id=125, item_type=2, price=200, max_damage=7)]
        plan = plan_weapon_upgrade(town_raw(gold=400), stock)
        self.assertEqual(plan["damage_before"], 7)
        self.assertEqual(plan["replaced_max_damage"], 6)
        self.assertEqual(plan["chosen"]["index"], 1)
        self.assertEqual(plan["chosen"]["projected_damage_after"], 11)
        self.assertEqual(plan["chosen"]["projected_damage_gain"], 4)
        self.assertEqual([c["index"] for c in plan["candidates"]], [1, 0, 2])

    def test_equal_gain_prefers_the_cheaper_then_the_lower_index(self):
        stock = [stock_item(0, base_id=119, item_type=1, price=300, max_damage=9),
                 stock_item(1, base_id=137, item_type=4, price=120, max_damage=9),
                 stock_item(2, base_id=125, item_type=2, price=120, max_damage=9)]
        plan = plan_weapon_upgrade(town_raw(gold=400), stock)
        self.assertEqual([c["index"] for c in plan["candidates"]], [1, 2, 0])
        self.assertEqual(plan["chosen"]["index"], 1)

    def test_budget_is_gold_minus_reserve(self):
        stock = [stock_item(0, base_id=137, item_type=4, price=300, max_damage=10),
                 stock_item(1, base_id=119, item_type=1, price=120, max_damage=8)]
        plan = plan_weapon_upgrade(town_raw(gold=290), stock)
        self.assertEqual(plan["budget"], 290)
        self.assertEqual(plan["chosen"]["index"], 1)
        self.assertIn({"index": 0, "reason": "over_budget"},
                      [{"index": r["index"], "reason": r["reason"]} for r in plan["rejected"]])
        reserved = plan_weapon_upgrade(town_raw(gold=290), stock, reserve=200)
        self.assertEqual(reserved["budget"], 90)
        self.assertIsNone(reserved["chosen"])
        self.assertIn("no_candidate", reserved["unresolved"])

    def test_the_reserve_is_the_potion_floor(self):
        """R18-K2b: the base reserve stays 0, the belt deficit is added on top.

        Same definition and same 50-gold constant as H2's ``identify_reserve``
        medicine floor, so the two legs can never fight over the potion money.
        """
        self.assertEqual(module.WEAPON_UPGRADE_RESERVE, 0)
        self.assertEqual(module.HEALER_MIN_HEAL_PRICE, 50)
        stock = [stock_item(0, base_id=119, item_type=1, price=150, max_damage=8)]
        full = plan_weapon_upgrade(town_raw(gold=150, belt_heals=4), stock)
        self.assertEqual(full["reserve"], 0)              # the normal seam: belt full
        self.assertEqual(full["budget"], 150)
        self.assertIsNotNone(full["chosen"])
        self.assertEqual(full["reserve_detail"]["medicine_deficit"], 0)
        # R18-K2b review round: with a full belt NOTHING is reserved, so the
        # binding term is the base reserve, not the medicine floor. The old
        # ">=" test made this branch unreachable and mislabelled every seam.
        self.assertEqual(full["reserve_detail"]["source"], "base")
        for heals, reserve in ((3, 50), (2, 100), (0, 200)):
            with self.subTest(belt_heals=heals):
                plan = plan_weapon_upgrade(town_raw(gold=250, belt_heals=heals), stock)
                self.assertEqual(plan["reserve"], reserve)
                self.assertEqual(plan["budget"], 250 - reserve)
                self.assertEqual(plan["reserve_detail"]["source"], "medicine_floor")
        starved = plan_weapon_upgrade(town_raw(gold=150, belt_heals=0), stock)
        self.assertEqual(starved["budget"], 0)
        self.assertIsNone(starved["chosen"])
        self.assertIn("no_budget", starved["unresolved"])
        self.assertIn({"index": 0, "reason": "over_budget"},
                      [{"index": r["index"], "reason": r["reason"]} for r in starved["rejected"]])
        # An explicit number still wins, so a caller can pin the reserve.
        self.assertEqual(plan_weapon_upgrade(
            town_raw(gold=150, belt_heals=0), stock, reserve=0)["budget"], 150)

    def test_the_native_equip_verdict_is_a_hard_gate_when_it_is_exported(self):
        """R18-K2b: ``can_equip`` gates only the items that advertise the scope.

        With the native scope ON the smith rows carry ``is_upgrade_weapon`` and
        ``PlanResourceArmor``'s own verdict; that verdict is the last word, so
        gold is never spent on a weapon the equip path would refuse.  ``dry-v1``
        runs with the scope OFF and therefore never sees either field, which is
        why the gate is conditional rather than unconditional.
        """
        refused = stock_item(0, base_id=119, item_type=1, price=120, max_damage=9,
                             is_upgrade_weapon=True, can_equip=False,
                             equip_reason="no_room_for_replaced_items")
        allowed = stock_item(1, base_id=137, item_type=4, price=300, max_damage=10,
                             is_upgrade_weapon=True, can_equip=True,
                             equip_reason="ready")
        plan = plan_weapon_upgrade(town_raw(gold=400), [refused, allowed])
        self.assertEqual([c["index"] for c in plan["candidates"]], [1])
        self.assertIn({"index": 0, "reason": "native_cannot_equip"},
                      [{"index": r["index"], "reason": r["reason"]} for r in plan["rejected"]])
        # Rows without the marker (an older bridge, or the dry twin) are judged
        # on the gates R18-K already had, so the counterfactual is unchanged.
        legacy = plan_weapon_upgrade(town_raw(gold=400), [
            stock_item(0, base_id=119, item_type=1, price=120, max_damage=9)])
        self.assertIsNotNone(legacy["chosen"])

    def test_smith_v1_demands_both_native_verdicts(self):
        """R18-K review round: ``require_native`` is what the executable arm passes.

        With the scope on every smith row carries ``is_upgrade_weapon``; a row
        that does not is a row the engine would answer ``unsupported_item``, and
        a row whose ``can_equip`` is false is one ``PlanResourceArmor`` refuses
        -- including its own ``no_damage_gain`` refusal, the check a14's
        AGGREGATE utility verdict (``upgrade``) does not make.
        """
        bare = stock_item(0, base_id=119, item_type=1, price=120, max_damage=9)
        plan = plan_weapon_upgrade(town_raw(gold=400), [bare], require_native=True)
        self.assertIsNone(plan["chosen"])
        self.assertEqual([r["reason"] for r in plan["rejected"]], ["native_scope_missing"])
        # An in-scope row the engine will not equip is refused even though the
        # aggregate PlanGearUpgrade verdict (`upgrade`) says yes.
        no_gain = native_stock_item(0, base_id=119, item_type=1, price=120,
                                    max_damage=9, upgrade=True, can_equip=False,
                                    equip_reason="no_damage_gain")
        refused = plan_weapon_upgrade(town_raw(gold=400), [no_gain], require_native=True)
        self.assertIsNone(refused["chosen"])
        self.assertEqual([r["reason"] for r in refused["rejected"]], ["native_cannot_equip"])
        ok = plan_weapon_upgrade(town_raw(gold=400), [
            native_stock_item(0, base_id=119, item_type=1, price=120, max_damage=9)],
            require_native=True)
        self.assertIsNotNone(ok["chosen"])

    def test_an_unmodelled_displaced_weapon_stops_the_plan(self):
        """R18-K review round: the projection models the DISPLACED item too.

        ``_pDamageMod`` is ``level * _pStrength / 100`` for the warrior
        (``items.cpp CalcPlrDamageMod``), so unequipping a hand weapon that
        carries ``_iPLStr`` -- or ``_iPLDam`` / ``_iPLDamMod`` -- lowers the
        readiness damage by an amount this arithmetic never subtracts.  Refuse
        to plan instead of reporting a gain the engine will not deliver.
        """
        stock = [native_stock_item(0, base_id=137, item_type=4, price=300, max_damage=10)]
        clean = plan_weapon_upgrade(town_raw(gold=400), stock, require_native=True)
        self.assertIsNotNone(clean["chosen"])
        for label, override in (("magical", {"quality": 1}),
                                ("strength_affix", {"effect_strength": 5}),
                                ("damage_affix", {"effect_damage": 40}),
                                ("damage_mod_affix", {"effect_damage_mod": 3})):
            with self.subTest(displaced=label):
                raw = town_raw(gold=400)
                raw["equipped_items"][4] = equipped(4, **override)
                plan = plan_weapon_upgrade(raw, stock, require_native=True)
                self.assertTrue(plan["displaced_unmodelled"])
                self.assertIsNone(plan["chosen"])
                self.assertIn("displaced_unmodelled", plan["unresolved"])
                self.assertEqual([r["reason"] for r in plan["rejected"]],
                                 ["displaced_weapon_unmodelled"])
                self.assertIsNone(module.weapon_trigger_reason(
                    raw, stock, require_native=True))

    def test_the_unarmed_floor_is_what_a_bare_hand_displaces(self):
        """R18-K review round: ``CalcPlrDamage`` floors _pIMaxDam at 3 (shield) / 1.

        R18-K subtracted 0 for a bare hand, over-stating the gain -- and with it
        the sort key and the counterfactual -- by up to 3 damage.
        """
        stock = [native_stock_item(0, base_id=137, item_type=4, price=300, max_damage=10)]
        with_shield = town_raw(gold=400)
        with_shield["equipped_items"][4] = {"present": False}
        plan = plan_weapon_upgrade(with_shield, stock, require_native=True)
        self.assertTrue(module.holds_shield(with_shield))
        self.assertEqual(module.unarmed_max_damage(with_shield), 3)
        self.assertEqual(plan["replaced_max_damage"], 3)
        self.assertEqual(plan["chosen"]["projected_damage_gain"], 7)   # 10 - 3
        bare = town_raw(gold=400)
        bare["equipped_items"][4] = {"present": False}
        bare["equipped_items"][5] = {"present": False}
        self.assertFalse(module.holds_shield(bare))
        self.assertEqual(module.unarmed_max_damage(bare), 1)
        self.assertEqual(plan_weapon_upgrade(bare, stock, require_native=True)
                         ["chosen"]["projected_damage_gain"], 9)       # 10 - 1

    def test_every_forbidden_shape_is_rejected_with_its_reason(self):
        cases = {
            "not_one_handed_melee_weapon": [
                stock_item(0, base_id=131, item_type=2, price=100, max_damage=12, equip_loc=2),
                stock_item(1, base_id=144, item_type=3, price=100, max_damage=12),
                stock_item(2, base_id=58, item_type=6, price=100, max_damage=0, equip_loc=3),
                stock_item(3, base_id=72, item_type=5, price=100, max_damage=0)],
            "cannot_use": [stock_item(0, base_id=122, item_type=1, price=100,
                                      max_damage=12, can_use=False)],
            "no_room": [stock_item(0, base_id=119, item_type=1, price=100,
                                   max_damage=12, can_fit=False)],
            "not_durable": [stock_item(0, base_id=119, item_type=1, price=100,
                                       max_damage=12, durable=False)],
            "unmodelled_quality": [stock_item(0, base_id=119, item_type=1, price=100,
                                              max_damage=12, quality=1)],
            "unmodelled_affix": [stock_item(0, base_id=119, item_type=1, price=100,
                                            max_damage=12, effect_damage_mod=3)],
            "native_not_an_upgrade": [stock_item(0, base_id=119, item_type=1, price=100,
                                                 max_damage=12, upgrade=False)],
            "no_projected_damage_gain": [stock_item(0, base_id=119, item_type=1,
                                                    price=100, max_damage=6)],
            "foreign_vendor": [stock_item(0, base_id=119, item_type=1, price=100,
                                          max_damage=12, vendor="healer")],
        }
        for reason, stock in cases.items():
            with self.subTest(reason=reason):
                plan = plan_weapon_upgrade(town_raw(gold=1000), stock)
                self.assertIsNone(plan["chosen"], reason)
                self.assertEqual({r["reason"] for r in plan["rejected"]}, {reason})

    def test_a_previously_refused_stock_key_is_not_reproposed(self):
        item = stock_item(0, base_id=119, item_type=1, price=100, max_damage=12)
        plan = plan_weapon_upgrade(town_raw(gold=1000), [item])
        self.assertIsNotNone(plan["chosen"])
        again = plan_weapon_upgrade(town_raw(gold=1000), [item],
                                    failed=[plan["chosen"]["stock_key"]])
        self.assertIsNone(again["chosen"])
        self.assertEqual([r["reason"] for r in again["rejected"]], ["previous_failure"])

    def test_armor_and_shield_decisions_are_never_touched(self):
        stock = [stock_item(0, base_id=58, item_type=6, price=100, max_damage=0,
                            equip_loc=3, is_ordinary_armor=True, meets_armor_gate=True),
                 stock_item(1, base_id=49, item_type=7, price=25, max_damage=0,
                            equip_loc=4, is_ordinary_armor=True, meets_armor_gate=True)]
        plan = plan_weapon_upgrade(town_raw(gold=1000), stock)
        self.assertIsNone(plan["chosen"])
        self.assertEqual(plan["candidates"], [])


# ---------------------------------------------------------------- the leg
class LegTests(unittest.TestCase):
    def stock(self):
        # smith-v1 rows: the executable arm demands both native verdicts.
        return [native_stock_item(0, base_id=119, item_type=1, price=120, max_damage=8),
                native_stock_item(1, base_id=137, item_type=4, price=300, max_damage=10)]

    def test_the_leg_never_fires_on_rows_without_the_native_verdicts(self):
        """R18-K review round: an older bridge's rows cannot start the leg."""
        raw = town_raw(gold=400)
        legacy = [stock_item(0, base_id=137, item_type=4, price=300, max_damage=10)]
        self.assertIsNone(WeaponUpgradeService().trip_trigger_reason(raw, legacy, 1))
        self.assertEqual(
            WeaponUpgradeService().trip_trigger_reason(raw, self.stock(), 1),
            "surplus_gold")

    def test_the_reserve_is_recorded_even_at_a_seam_that_never_fires(self):
        """R18-K2b review round (2026-09-07): the reserve law must be auditable.

        Only a seam that wins reaches ``_command``, so a seam refused for
        ``over_budget`` -- the very place the reserve can bind -- used to leave
        no reserve record at all, and the arm-B rows carried ``reserve_detail``
        only for the episodes that spent.
        """
        fresh = WeaponUpgradeService()
        self.assertEqual(fresh.telemetry()["reserve"], 0)
        self.assertIsNone(fresh.telemetry()["reserve_detail"])
        self.assertIsNone(fresh.telemetry()["reserve_from"])

        service = WeaponUpgradeService()
        raw = town_raw(gold=120, belt_heals=2)      # 100 reserved, 20 to spend
        self.assertIsNone(service.trip_trigger_reason(raw, self.stock(), 1))
        info = service.telemetry()
        self.assertEqual(info["weapon_upgrades"], 0)
        self.assertEqual(info["reserve"], 100)
        self.assertEqual(info["reserve_from"], "seam")
        self.assertEqual(info["reserve_detail"]["medicine_deficit"], 2)
        self.assertEqual(info["reserve_detail"]["source"], "medicine_floor")
        # The seam plan is telemetry only: it never becomes the leg's plan.
        self.assertIsNone(info["last_plan"])

    def test_full_buy_and_equip_leg(self):
        raw = town_raw(gold=400, stock=(), active_vendor=None)
        env, bridge = FakeEnv(raw), FakeBridge()
        service = WeaponUpgradeService()
        trigger = service.trip_trigger_reason(raw, self.stock(), 1)
        self.assertEqual(trigger, "surplus_gold")
        service.start(raw, env._steps, trigger, 1)

        commands = []
        # 1) open Griswold (adjacent, so a single talk).
        commands.append(service.command(env, bridge))
        service.receipt(commands[-1], {"accepted": True})
        raw["resource_state"]["town"]["active_vendor"] = "smith"
        raw["resource_state"]["town"]["stock"] = self.stock()
        env._steps += 1
        # 2) buy the maximum-gain affordable weapon on FRESH quotes.
        commands.append(service.command(env, bridge))
        service.receipt(commands[-1], {"accepted": True, "price": 300, "reason": "purchased"})
        raw["gold"] = 100
        raw["resource_state"]["inventory_equipment"] = [dict(
            self.stock()[1], index=0)]
        env._steps += 1
        # 3) equip it out of the inventory.
        commands.append(service.command(env, bridge))
        service.receipt(commands[-1], {"accepted": True})
        raw["resource_state"]["inventory_equipment"] = []
        raw["equipped_items"][4] = {"present": True, "seed_hi": 101, "seed_lo": 201,
                                    "create_info": 1030, "base_id": 137,
                                    "durability": 30, "max_durability": 30,
                                    "min_damage": 1, "max_damage": 10}
        raw["resource_state"]["readiness"]["damage"] = 11
        env._steps += 1
        # 4) hand the window back.
        commands.append(service.command(env, bridge))

        self.assertEqual(commands[0], ("talk", SMITH_ID))
        self.assertEqual(commands[1], ("buy", "smith", 1, 101, 201, 1030, 137))
        self.assertEqual(commands[2], ("equip", 0, 101, 201, 1030, 137))
        self.assertEqual(commands[3], ("complete",))
        self.assertFalse(service.active)
        self.assertEqual(service.reason, "weapon_equipped")

        info = service.telemetry()
        self.assertEqual(info["weapon_upgrades"], 1)
        self.assertEqual(info["weapon_gold_spent"], 300)
        self.assertEqual(info["weapon_damage_before"], [7])
        self.assertEqual(info["weapon_damage_after"], [11])
        self.assertEqual(info["purchases"][0]["projected_damage_after"], 11)
        self.assertTrue(info["purchases"][0]["projection_exact"])
        self.assertEqual(info["attempts"][-1]["outcome"], "completed")
        self.assertEqual(info["attempts"][-1]["gold_spent"], 300)
        self.assertEqual(info["reserve"], 0)

    def test_one_leg_per_trip_and_no_second_weapon(self):
        raw = town_raw(gold=400)
        service = WeaponUpgradeService()
        service.start(raw, 1000, "surplus_gold", 1)
        service._end(FakeEnv(raw), "weapon_equipped", failed=False)
        self.assertIsNone(service.trip_trigger_reason(raw, self.stock(), 1))
        self.assertEqual(service.trip_trigger_reason(raw, self.stock(), 2), "surplus_gold")

    def test_a_refused_purchase_ends_the_leg_without_retrying(self):
        raw = town_raw(gold=400, stock=self.stock(), active_vendor="smith")
        env, bridge = FakeEnv(raw), FakeBridge()
        service = WeaponUpgradeService()
        service.start(raw, 1000, "surplus_gold", 1)
        command = service.command(env, bridge)
        self.assertEqual(command[0], "buy")
        service.receipt(command, {"accepted": False, "reason": "unsupported_item"})
        env._steps += 1
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "weapon_purchase_rejected")
        self.assertEqual(service.weapon_upgrades, 0)
        self.assertEqual(service.weapon_gold_spent, 0)

    def test_no_candidate_ends_the_leg_without_spending(self):
        poor = town_raw(gold=10, stock=self.stock(), active_vendor="smith")
        env, bridge = FakeEnv(poor), FakeBridge()
        service = WeaponUpgradeService()
        service.start(poor, 1000, "surplus_gold", 1)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "weapon_no_candidate")
        self.assertEqual(service.weapon_gold_spent, 0)

    def test_the_leg_refuses_to_start_without_the_native_scope(self):
        raw = town_raw(gold=400, weapon_scope=False)
        with self.assertRaises(RuntimeError):
            WeaponUpgradeService().start(raw, 1000, "surplus_gold", 1)

    def test_the_leg_refuses_to_start_outside_town(self):
        raw = town_raw(gold=400)
        raw["dungeon_level"] = 1
        with self.assertRaises(RuntimeError):
            WeaponUpgradeService().start(raw, 1000, "surplus_gold", 1)

    def test_the_service_cap_bounds_the_leg(self):
        raw = town_raw(gold=400, stock=self.stock(), active_vendor="smith")
        env, bridge = FakeEnv(raw), FakeBridge()
        service = WeaponUpgradeService()
        service.start(raw, 1000, "surplus_gold", 1)
        env._steps = 1000 + module.WEAPON_UPGRADE_SERVICE_CAP
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "weapon_cap")


# ------------------------------------------------------------- ledger booking
class NativeScopeConfigurationTests(unittest.TestCase):
    """R18-K2b review round (2026-09-07): the scope is written on EVERY configure.

    ``gResourceWeaponPurchase`` is a process-wide native flag and nothing else
    clears it, so an off arm constructed after a smith-v1 arm in the same worker
    used to inherit the leaked ``True`` and raise the handshake mismatch on its
    first observation, with no way to recover.
    """

    class Recorder:
        def __init__(self):
            self.calls = []

        def configure_resource_weapon_purchase(self, enabled):
            self.calls.append(enabled)

    class OlderBridge:
        """A bridge built before R18-K2b: the entry point does not exist."""

    def test_every_configure_writes_the_scope(self):
        for upgrade, expected in (("smith-v1", True), ("dry-v1", False), ("off", False)):
            with self.subTest(resource_weapon_upgrade=upgrade):
                recorder = self.Recorder()
                self.assertIs(
                    module.configure_native_weapon_purchase(recorder, upgrade), expected)
                self.assertEqual(recorder.calls, [expected])

    def test_an_off_arm_clears_the_scope_a_previous_env_switched_on(self):
        recorder = self.Recorder()
        module.configure_native_weapon_purchase(recorder, "smith-v1")
        module.configure_native_weapon_purchase(recorder, "off")
        module.configure_native_weapon_purchase(recorder, "smith-v1")
        self.assertEqual(recorder.calls, [True, False, True])

    def test_an_older_bridge_still_runs_every_arm_but_smith_v1(self):
        older = self.OlderBridge()
        self.assertIs(module.configure_native_weapon_purchase(older, "off"), False)
        self.assertIs(module.configure_native_weapon_purchase(older, "dry-v1"), False)
        with self.assertRaises(RuntimeError) as error:
            module.configure_native_weapon_purchase(older, "smith-v1")
        self.assertIn("isolated R18-K2b rebuild", str(error.exception))


class LedgerTests(unittest.TestCase):
    """R18-K review round: these drive the SHIPPED booking block.

    ``options_env`` no longer carries a copy of the arithmetic; it calls
    ``resource_weapon_upgrade.book_weapon_purchase`` with the same guard terms
    the R18-F scroll block uses, and that function is what these tests call.
    Delete the call in ``options_env`` and a live episode raises
    "Completed loot trip has unexplained cash movement" -- which is exactly what
    ``test_an_unbooked_purchase_breaks_the_trip_reconciliation`` reproduces.
    """

    BUY = ("buy", "smith", 1, 101, 201, 1030, 137)
    RECEIPT = {"accepted": True, "price": 300, "reason": "purchased"}

    def open_trip(self, gold=400):
        town = loot.SustainLootService("full")
        self.assertTrue(town.maybe_start(loot_raw(gold=gold), 3600,
                                         farm_scene_steps=3600, cleared=False))
        self.assertEqual((town.purchases, town.gold_spent), (0, 0))
        self.assertTrue(town.active)
        return town

    def test_the_purchase_is_booked_in_the_town_trip_ledger(self):
        town, weapon = self.open_trip(400), WeaponUpgradeService()
        self.assertEqual(
            module.book_weapon_purchase(town, weapon, weapon, self.BUY, self.RECEIPT), 300)
        self.assertEqual((town.purchases, town.gold_spent), (1, 300))
        town.observe(loot_raw(gold=100), 3700)      # the wallet the engine reports
        town.reason = "complete"
        town._seal_trip(3700)
        trip = town._trips[-1]
        self.assertEqual(trip["start_gold"], 400)
        self.assertEqual(trip["end_gold"], 100)
        self.assertEqual(trip["gold_spent"], 300)
        self.assertEqual(trip["purchases"], 1)
        self.assertEqual(trip["cash_residual"], 0)
        self.assertEqual(trip["cash_reconciliation"], "complete")

    def test_an_unbooked_purchase_breaks_the_trip_reconciliation(self):
        """The negative control the old tautological test could not express."""
        town = self.open_trip(400)
        town.observe(loot_raw(gold=100), 3700)
        town.reason = "complete"
        with self.assertRaisesRegex(RuntimeError, "unexplained cash movement"):
            town._seal_trip(3700)

    def test_a_refused_purchase_books_nothing(self):
        town, weapon = self.open_trip(400), WeaponUpgradeService()
        refused = {"accepted": False, "reason": "unsupported_item", "price": 0}
        self.assertEqual(
            module.book_weapon_purchase(town, weapon, weapon, self.BUY, refused), 0)
        self.assertEqual((town.purchases, town.gold_spent), (0, 0))
        town.observe(loot_raw(gold=400), 3700)
        town.reason = "complete"
        town._seal_trip(3700)                        # nothing moved, nothing booked
        self.assertEqual(town._trips[-1]["cash_residual"], 0)

    def test_every_booking_guard_refuses(self):
        town, weapon = self.open_trip(400), WeaponUpgradeService()
        closed = self.open_trip(400)
        closed.active = False
        cases = {
            "no_weapon_service": (town, None, weapon, self.BUY, self.RECEIPT),
            "another_service_owns_the_command": (town, weapon, town, self.BUY, self.RECEIPT),
            "not_a_purchase": (town, weapon, weapon, ("equip", 0), self.RECEIPT),
            "town_trip_already_closed": (closed, weapon, weapon, self.BUY, self.RECEIPT),
        }
        for label, args in cases.items():
            with self.subTest(guard=label):
                self.assertEqual(module.book_weapon_purchase(*args), 0)
        self.assertEqual((town.purchases, town.gold_spent), (0, 0))
        self.assertEqual((closed.purchases, closed.gold_spent), (0, 0))


# -------------------------------------------------- default-off identity
class DefaultOffIdentityTests(unittest.TestCase):
    """With the flag off nothing new is constructed, passed or emitted."""

    ITINERARY = [("talk", SMITH_ID), ("dismiss",), ("talk", HEALER_ID), ("dismiss",)]

    def drive_itinerary(self, weapon_service):
        """The ordinary sustain itinerary over a synthetic town state.

        The recorded sequence must not depend on whether a weapon service
        object exists: the leg is reachable only through the coach's explicit
        RESUPPLY intercept, which the off arm never constructs.
        """
        raw = town_raw(gold=400, failures=("level",))
        env, bridge = FakeEnv(raw), FakeBridge()
        service = sustain.SustainResourceService("full")
        service.start(raw, env._steps, "test")
        service.phase = "outbound"
        commands = []
        for _ in range(len(self.ITINERARY)):
            command = service.command(env, bridge)
            commands.append(command)
            if command[0] == "talk":
                raw["resource_state"]["town"]["active_vendor"] = (
                    "smith" if command[1] == SMITH_ID else "healer")
                if command[1] == SMITH_ID:
                    raw["resource_state"]["town"]["stock"] = [
                        stock_item(0, base_id=119, item_type=1, price=120, max_damage=8)]
            elif command[0] == "dismiss":
                raw["resource_state"]["town"]["active_vendor"] = None
            env._steps += 1
            if weapon_service is not None:
                # An existing service object still issues nothing on its own.
                self.assertFalse(weapon_service.active)
        return commands, service

    def test_the_town_itinerary_never_consults_the_weapon_service(self):
        """What this actually proves, after the R18-K review round.

        ``SustainResourceService`` has no reference to a ``WeaponUpgradeService``
        -- the leg is reachable only through the ``OptionsEnv`` RESUPPLY seam,
        which the off arm never constructs (``weapon_upgrade_service is None``).
        So this test asserts the ordinary itinerary is untouched and that an
        existing service object issues nothing on its own; it is NOT evidence
        about the seam, which no pure-Python double can drive.  The seam-level
        default-off evidence is ``test_env_kwargs_are_byte_identical_when_off``
        below and, decisively, arm A's rows_sha_v3 in the probe.
        """
        self.assertNotIn("weapon", sustain.__doc__ or "")
        self.assertFalse([name for name in dir(sustain.SustainResourceService)
                          if "weapon_upgrade" in name])
        off_commands, off_service = self.drive_itinerary(None)
        on_commands, on_service = self.drive_itinerary(WeaponUpgradeService())
        self.assertEqual(off_commands, self.ITINERARY)
        self.assertEqual(on_commands, off_commands)
        self.assertEqual(off_service.phase, on_service.phase)
        self.assertEqual(off_service.purchases, 0)
        self.assertEqual(off_service.gold_spent, 0)

    def test_an_unstarted_service_emits_an_inert_telemetry_block(self):
        info = WeaponUpgradeService().telemetry()
        self.assertFalse(info["attempted"])
        self.assertFalse(info["active"])
        self.assertEqual(info["weapon_upgrades"], 0)
        self.assertEqual(info["weapon_gold_spent"], 0)
        self.assertEqual(info["purchases"], [])
        self.assertEqual(info["attempts"], [])

    def test_env_kwargs_are_byte_identical_when_off(self):
        options = self.options_module()
        captured = {}

        class Sentinel(Exception):
            pass

        class Recorder:
            def __init__(self, **kwargs):
                captured.clear()
                captured.update(kwargs)
                raise Sentinel

        base = dict(max_steps=6000, resource_protocol="l2-town-v1",
                    resource_purchase_mode="full",
                    resource_service_policy="sustain-loot-v1",
                    resource_readiness_law="coach-v03",
                    worker_time_protocol="completion-l2-r18c",
                    resource_retreat="retreat-v1", farm_scene_cap=3600,
                    reset_layer_clock_on_window=True, reward_economy="v4",
                    hunt_scope="l1-only", explore_global_hunt=True)
        previous = options.DiabloGymEnv
        options.DiabloGymEnv = Recorder
        try:
            frozen = self.capture(options, Sentinel, captured, base)
            explicit = self.capture(options, Sentinel, captured,
                                    dict(base, resource_weapon_upgrade="off"))
            self.assertEqual(frozen, explicit)
            self.assertNotIn("resource_weapon_upgrade", frozen)
            enabled = self.capture(options, Sentinel, captured,
                                   dict(base, resource_weapon_upgrade="smith-v1"))
            self.assertEqual(enabled.pop("resource_weapon_upgrade"), "smith-v1")
            self.assertEqual(enabled, frozen)
        finally:
            options.DiabloGymEnv = previous

    @staticmethod
    def capture(options, Sentinel, captured, kwargs):
        try:
            options.OptionsEnv(**kwargs)
        except Sentinel:
            return copy.deepcopy(captured)
        raise AssertionError("the recorder was never reached")

    def options_module(self):
        try:
            return importlib.import_module(PACKAGE + ".options_env")
        except Exception as exc:  # pragma: no cover - gym/numpy absent
            raise unittest.SkipTest(f"options_env not importable: {exc!r}")


class DryModeTests(unittest.TestCase):
    """dry-v1 measures the counterfactual without touching the episode."""

    def stock(self):
        return [stock_item(0, base_id=119, item_type=1, price=120, max_damage=8),
                stock_item(1, base_id=137, item_type=4, price=300, max_damage=10)]

    def test_dry_v1_is_a_valid_configuration(self):
        self.assertEqual(validate_weapon_upgrade(
            "l2-town-v1", "sustain-loot-v1", "full", "dry-v1"), "dry-v1")
        with self.assertRaises(ValueError):
            validate_weapon_upgrade("l2-town-v1", "sustain-v6", "full", "dry-v1")

    def test_dry_v1_needs_no_native_scope(self):
        raw = town_raw(weapon_scope=False)
        validate_native_weapon_upgrade(raw, "dry-v1")
        with self.assertRaises(RuntimeError):
            validate_native_weapon_upgrade(town_raw(weapon_scope=True), "dry-v1")

    def test_dry_v1_never_starts_a_leg(self):
        raw = town_raw(gold=400, weapon_scope=False)
        service = WeaponUpgradeService(protocol="dry-v1")
        self.assertIsNone(service.trip_trigger_reason(raw, self.stock(), 1))
        seam = service.observe_seam(raw, self.stock(), 1, 4321)
        self.assertFalse(service.attempted)
        self.assertFalse(service.active)
        self.assertEqual(service.weapon_upgrades, 0)
        self.assertEqual(service.weapon_gold_spent, 0)
        self.assertTrue(seam["would_fire"])
        self.assertEqual(seam["would_price"], 300)
        self.assertEqual(seam["would_damage_gain"], 4)
        self.assertEqual(seam["weapon_damage_before"], 7)

    def test_dry_v1_records_one_seam_per_trip(self):
        raw = town_raw(gold=400, weapon_scope=False)
        service = WeaponUpgradeService(protocol="dry-v1")
        service.observe_seam(raw, self.stock(), 1, 100)
        self.assertIsNone(service.observe_seam(raw, self.stock(), 1, 140))
        service.observe_seam(raw, self.stock(), 2, 900)
        info = service.telemetry()
        self.assertEqual(info["seams_observed"], 2)
        self.assertEqual(info["seams_would_fire"], 2)
        self.assertEqual(info["would_gold_spent"], 600)
        self.assertEqual(info["would_damage_gain"], [4, 4])

    def test_dry_v1_records_the_refusal_reasons_when_it_would_not_fire(self):
        service = WeaponUpgradeService(protocol="dry-v1")
        seam = service.observe_seam(town_raw(gold=10, weapon_scope=False),
                                    self.stock(), 1, 100)
        self.assertFalse(seam["would_fire"])
        self.assertEqual(seam["reject_reasons"], ["over_budget"])
        self.assertEqual(service.telemetry()["would_gold_spent"], 0)

    def test_a_smith_v1_service_ignores_the_dry_seam_hook(self):
        service = WeaponUpgradeService()
        self.assertIsNone(service.observe_seam(
            town_raw(gold=400), self.stock(), 1, 100))
        self.assertEqual(service.telemetry()["seams_observed"], 0)


class NativeWeaponPurchaseTests(unittest.TestCase):
    """R18-K2b (2026-09-07): the live bridge, a real seed-8001 town visit.

    Engineering fixtures (invincibility for the walk, a funded wallet) exist only
    so a refusal can never be confused with ``no_money`` or a death; every
    purchase, equip and refusal below is the real native transaction.
    """

    SEED = 8001
    MELEE = (1, 2, 4, 10)          # Sword / Axe / Mace / Staff, Bow excluded

    @classmethod
    def setUpClass(cls):
        try:
            from diablogym import DiabloGymEnv, bridge, nav
        except Exception as exc:                        # pragma: no cover
            raise unittest.SkipTest(f"native bridge unavailable: {exc}")
        if not hasattr(bridge, "configure_resource_weapon_purchase"):
            raise unittest.SkipTest(
                "requires the isolated R18-K2b weapon-purchase bridge")
        cls.bridge, cls.nav = bridge, nav
        cls.owner = DiabloGymEnv()

    @classmethod
    def tearDownClass(cls):
        cls.reset_native(cls)
        cls.owner.close()

    def reset_native(self):
        self.bridge.end_game()
        self.bridge.configure_resource_weapon_purchase(False)
        self.bridge.configure_resource_protocol(False)

    def tearDown(self):
        self.reset_native()

    # ---- the same walk the R18-K blocker receipt used ----
    def approach(self, vendor):
        raw = self.bridge.observe()
        npc = next(n for n in raw["resource_state"]["town"]["npcs"]
                   if n["type"] == vendor)
        points = [(npc["x"] + dx, npc["y"] + dy)
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        points.sort(key=lambda p: max(abs(p[0] - raw["player_x"]),
                                      abs(p[1] - raw["player_y"])))
        for x, y in points:
            if not self.bridge.probe_tile(x, y)["walkable"]:
                continue
            raw, _ = self.nav.walk_to(self.bridge, x, y, max_ticks=4000)
            if max(abs(raw["player_x"] - npc["x"]),
                   abs(raw["player_y"] - npc["y"])) <= 1:
                self.bridge.act_wait()
                self.bridge.step(1)
                return npc
        self.fail(f"could not walk adjacent to {vendor}")

    def free_cells(self):
        return sum(1 for cell in
                   self.bridge.probe_resource_inventory_snapshot()["grid"] if cell == 0)

    def visit_smith(self, weapon_flag, *, gold=5000, loot_economy=False):
        self.bridge.end_game()
        if loot_economy:
            # The loot economy is what arms a14's retention-capacity filter
            # (ConsiderGearUpgradePlan), i.e. the regime the review round found.
            self.bridge.configure_resource_protocol(
                True, ordinary_armor_scope=True, preserve_equipment_readiness=True,
                loot_economy=True, readiness_advisory=True)
        else:
            self.bridge.configure_resource_protocol(True, ordinary_armor_scope=True)
        self.bridge.configure_resource_weapon_purchase(weapon_flag)
        raw = self.bridge.reset(self.SEED)
        raw = self.nav.descend_to_dungeon(self.bridge)
        self.bridge.act_wait()
        raw = self.bridge.step(1)
        self.bridge.probe_invincible(True)
        self.bridge.configure_town_service(True)
        upstairs = next(t for t in raw["triggers"]
                        if t["msg"] == self.bridge.WM_DIABPREVLVL)
        raw, _ = self.nav.walk_to(self.bridge, upstairs["x"], upstairs["y"])
        self.assertEqual(raw["dungeon_level"], 0)
        self.bridge.probe_resource_add_gold(gold)
        npc = self.approach("smith")
        for _ in range(6):
            town = self.bridge.observe()["resource_state"]["town"]
            if town["dialog_active"]:
                self.bridge.act_dismiss_dialog()
                self.bridge.step(1)
                continue
            if town["active_vendor"] == "smith":
                return self.bridge.observe()
            self.bridge.act_talk_towner(npc["id"])
            self.bridge.step(2)
        self.fail("Griswold did not open")

    def stock(self, raw):
        return raw["resource_state"]["town"]["stock"]

    def one_handed(self, raw):
        return [i for i in self.stock(raw)
                if int(i["equip_loc"]) == 1 and int(i["item_type"]) in self.MELEE
                and i["can_use"] and int(i["quality"]) == 0]

    def two_handed(self, raw):
        return [i for i in self.stock(raw) if int(i["equip_loc"]) == 2
                and int(i["item_type"]) in (1, 2, 3, 4, 10)]

    # ---- the tests ----
    def test_flag_off_keeps_the_counter_armor_only(self):
        raw = self.visit_smith(False)
        state = raw["resource_state"]
        self.assertIs(state["weapon_purchase_enabled"], False)
        for item in self.stock(raw):
            self.assertNotIn("is_upgrade_weapon", item)
        weapons = self.one_handed(raw)
        self.assertTrue(weapons, "seed 8001 must quote a usable one-handed weapon")
        gold_before = raw["gold"]
        for item in weapons:
            with self.subTest(index=item["index"]):
                self.assertFalse(item["can_equip"])
                self.assertEqual(item["equip_reason"], "unsupported_item")
                receipt = self.bridge.act_buy_store_item(
                    "smith", item["index"], *identity(item))
                self.assertFalse(receipt["accepted"], receipt)
                self.assertEqual(receipt["reason"], "unsupported_item")
                self.assertEqual(receipt["price"], 0)
        self.assertEqual(self.bridge.observe()["gold"], gold_before)

    def test_flag_on_buys_and_wears_a_real_one_handed_weapon(self):
        raw = self.visit_smith(True)
        state = raw["resource_state"]
        self.assertIs(state["weapon_purchase_enabled"], True)
        damage_before = state["readiness"]["damage"]
        gold_before = raw["gold"]
        shield_before = raw["equipped_items"][5]
        weapons = [i for i in self.one_handed(raw) if i["can_equip"]]
        self.assertTrue(weapons, "seed 8001 must quote an equippable one-handed weapon")
        item = max(weapons, key=lambda i: int(i["max_damage"]))
        self.assertIs(item["is_upgrade_weapon"], True)
        self.assertEqual(item["equip_reason"], "ready")
        self.assertIn(item["target_slot"], (4, 5))       # INVLOC_HAND_LEFT/RIGHT

        receipt = self.bridge.act_buy_store_item("smith", item["index"], *identity(item))
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "purchased")
        self.assertEqual(receipt["price"], item["price"])
        self.bridge.step(1)
        self.assertEqual(self.bridge.observe()["gold"], gold_before - item["price"])

        wanted = identity(item)
        carried = next(i for i in self.bridge.observe()["resource_state"]["inventory_equipment"]
                       if identity(i) == wanted)
        equipped_receipt = self.bridge.act_equip_inventory_item(
            int(carried["index"]), *wanted)
        self.assertTrue(equipped_receipt["accepted"], equipped_receipt)
        self.assertEqual(equipped_receipt["reason"], "equipped")
        self.bridge.step(1)
        after = self.bridge.observe()
        body = after["equipped_items"][int(equipped_receipt["target_slot"])]
        self.assertTrue(body["present"])
        self.assertEqual(identity(body), wanted)
        self.assertGreater(after["resource_state"]["readiness"]["damage"], damage_before)
        # v1 never touches the shield: it is an armor decision.
        self.assertEqual(after["equipped_items"][5]["present"], shield_before["present"])
        if shield_before["present"]:
            self.assertEqual(identity(after["equipped_items"][5]), identity(shield_before))
        # Gold moved exactly once, and only by the quoted price.
        self.assertEqual(after["gold"], gold_before - item["price"])

    def test_two_handed_weapons_stay_refused_with_the_flag_on(self):
        raw = self.visit_smith(True)
        pairs = self.two_handed(raw)
        self.assertTrue(pairs, "seed 8001 must quote a two-handed weapon")
        gold_before = raw["gold"]
        usable = [i for i in pairs if i["can_use"]]
        self.assertTrue(usable, "seed 8001 must quote a WIELDABLE two-handed weapon")
        for item in pairs:
            with self.subTest(index=item["index"], can_use=item["can_use"]):
                self.assertIs(item["is_upgrade_weapon"], False)
                self.assertFalse(item["can_equip"])
                self.assertEqual(item["equip_reason"], "unsupported_item")
                receipt = self.bridge.act_buy_store_item(
                    "smith", item["index"], *identity(item))
                self.assertFalse(receipt["accepted"], receipt)
                # A two-hander the warrior could actually wield is refused by
                # the v1 SCOPE; one he cannot wield is refused one gate earlier
                # by the unchanged CanUseItem check.
                self.assertEqual(receipt["reason"],
                                 "unsupported_item" if item["can_use"] else "cannot_use")
        self.assertEqual(self.bridge.observe()["gold"], gold_before)

    def test_the_python_law_chooses_the_native_candidate(self):
        raw = self.visit_smith(True)
        plan = plan_weapon_upgrade(raw, self.stock(raw))
        self.assertIsNotNone(plan["chosen"], plan["unresolved"])
        chosen = plan["chosen"]
        row = next(i for i in self.stock(raw) if int(i["index"]) == chosen["index"])
        self.assertIs(row["is_upgrade_weapon"], True)
        self.assertTrue(row["can_equip"])
        self.assertLessEqual(chosen["price"], plan["budget"])
        receipt = self.bridge.act_buy_store_item("smith", row["index"], *identity(row))
        self.assertTrue(receipt["accepted"], receipt)
        self.bridge.step(1)
        wanted = identity(row)
        carried = next(i for i in self.bridge.observe()["resource_state"]["inventory_equipment"]
                       if identity(i) == wanted)
        self.assertTrue(self.bridge.act_equip_inventory_item(
            int(carried["index"]), *wanted)["accepted"])
        self.bridge.step(1)
        after = self.bridge.observe()["resource_state"]["readiness"]["damage"]
        # The Python projection is the native metric _pIMaxDam + _pDamageMod.
        self.assertEqual(after, chosen["projected_damage_after"])

    def test_a_tight_pack_agrees_between_the_counter_and_the_hand(self):
        """R18-K2b review round (2026-09-07): the counter IS the equip verdict.

        The counter answers ``PlanResourceArmor`` with ``inventoryIndex = -1``,
        before the pack holds the weapon; the hand answers it with the real
        index, which the plan removes before staging the displaced item.  Those
        are the same question only while a14's own retention-capacity filter is
        not ALSO applied to the live pack: with it, the hand demanded room for
        the displaced weapon WITHOUT giving back the cells the just-purchased
        one occupies, so a tight-pack purchase passed the counter and was then
        refused with ``not_an_upgrade`` -- gold spent, weapon unworn.

        At seed 8001 Griswold quotes a 2-cell mace that displaces a 2-cell hand
        weapon, so ``leave_free = 3`` is exactly the regime the pre-fix hand
        refused (one free cell left after the purchase); ``leave_free = 0`` is
        the honest refusal, which must name capacity at the counter and cost
        nothing.
        """
        bought = refused = 0
        for leave_free in (0, 3):
            with self.subTest(leave_free=leave_free):
                raw = self.visit_smith(True, loot_economy=True)
                self.bridge.probe_resource_fill_inventory(leave_free)
                raw = self.bridge.observe()
                self.assertEqual(self.free_cells(), leave_free)
                state = raw["resource_state"]
                damage_before = state["readiness"]["damage"]
                gold_before = raw["gold"]
                rows = [i for i in self.one_handed(raw) if i.get("is_upgrade_weapon")]
                self.assertTrue(rows, "seed 8001 must quote a one-handed weapon")
                ready = [i for i in rows if i["can_equip"]]
                if not ready:
                    refused += 1
                    for item in rows:
                        # never the aggregate-upgrade verdict standing in for
                        # capacity: the refusal must name the real cause.
                        self.assertIn(item["equip_reason"],
                                      ("no_room_for_replaced_items", "no_damage_gain"))
                        receipt = self.bridge.act_buy_store_item(
                            "smith", item["index"], *identity(item))
                        self.assertFalse(receipt["accepted"], receipt)
                        self.assertEqual(receipt["reason"], item["equip_reason"])
                    self.assertEqual(self.bridge.observe()["gold"], gold_before)
                    continue
                bought += 1
                item = max(ready, key=lambda i: int(i["max_damage"]))
                self.assertEqual(item["equip_reason"], "ready")
                free_before = self.free_cells()
                receipt = self.bridge.act_buy_store_item(
                    "smith", item["index"], *identity(item))
                self.assertTrue(receipt["accepted"], receipt)
                self.bridge.step(1)
                free_after_buy = self.free_cells()
                wanted = identity(item)
                carried = next(i for i in self.bridge.observe()["resource_state"]
                               ["inventory_equipment"] if identity(i) == wanted)
                equipped_receipt = self.bridge.act_equip_inventory_item(
                    int(carried["index"]), *wanted)
                # The whole law: accepted at the counter implies worn.
                self.assertTrue(equipped_receipt["accepted"], equipped_receipt)
                self.bridge.step(1)
                after = self.bridge.observe()
                # The pack loses the weapon and gains the displaced one, so the
                # displaced footprint is measurable -- and it must be LARGER
                # than the room left after the purchase, or this seed no longer
                # exercises the regime the pre-fix hand refused.
                displaced_cells = free_before - self.free_cells()
                self.assertGreater(displaced_cells, free_after_buy)
                self.assertGreater(
                    after["resource_state"]["readiness"]["damage"], damage_before)
                self.assertEqual(after["gold"], gold_before - item["price"])
        self.assertEqual((bought, refused), (1, 1))

    def test_the_switch_is_scoped_and_between_episodes_only(self):
        self.bridge.end_game()
        self.bridge.configure_resource_protocol(False)
        with self.assertRaises(ValueError):
            self.bridge.configure_resource_weapon_purchase(True)
        self.bridge.configure_resource_protocol(True)          # no armor scope
        with self.assertRaises(ValueError):
            self.bridge.configure_resource_weapon_purchase(True)
        self.bridge.configure_resource_protocol(True, ordinary_armor_scope=True)
        self.bridge.configure_resource_weapon_purchase(True)
        self.bridge.reset(self.SEED)
        with self.assertRaises(RuntimeError):
            self.bridge.configure_resource_weapon_purchase(False)
        self.assertIs(self.bridge.observe()["resource_state"]["weapon_purchase_enabled"], True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
