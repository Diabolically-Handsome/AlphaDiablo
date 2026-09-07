"""R18-F portal-v1 boundaries.

Two independent suites in one file:

* ``Portal*Tests`` -- pure Python doubles, no native engine. The modules under
  test are loaded through a private package alias whose ``__path__`` points at
  ``python/diablogym`` (the house pattern of ``test_resource_retreat.py``, which
  keeps ``diablogym/__init__.py`` and therefore the .so out of the way), plus a
  stub ``bridge`` submodule. Nothing here resets the engine, plans a native
  path, probes a tile or constructs DiabloGymEnv/OptionsEnv.
* ``PortalNativeTests`` -- the real headless bridge, skipped unless the module
  actually carries the portal-v1 symbols (i.e. the isolated R18-F build). It
  freezes the DEFAULT-OFF verdict for all four portal/warp messages in both
  scenes, which is the house law this feature has to keep.
"""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

PACKAGE = "_r18f_portal_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/diablogym")]
    sys.modules[PACKAGE] = package
if PACKAGE + ".bridge" not in sys.modules:
    # Only ``_walk``'s door branch reaches for the bridge; every path exercised
    # below keeps ``door`` false.
    sys.modules[PACKAGE + ".bridge"] = ModuleType(PACKAGE + ".bridge")

module = importlib.import_module(PACKAGE + ".resource_portal")
protocol = importlib.import_module(PACKAGE + ".resource_protocol")

PortalPolicy = module.PortalPolicy
PortalService = module.PortalService
TOWN_PORTAL_MISSILE = module.TOWN_PORTAL_MISSILE


class FakeBridge:
    """Records the portal authority the script grants and revokes."""

    def __init__(self, revoke_error=None):
        self.calls = []
        self.town_service = []
        self.revoke_error = revoke_error

    def configure_resource_portal(self, enabled):
        self.calls.append(bool(enabled))
        if not enabled and self.revoke_error is not None:
            raise RuntimeError(self.revoke_error)
        return bool(enabled)

    def configure_town_service(self, enabled):
        self.town_service.append(bool(enabled))


class FakeEnv:
    """The three attributes and one planner call the service asks of an env."""

    def __init__(self, value, steps=0, path=(), unguarded_path=None, ticks_per_step=4):
        self._raw = value
        self._steps = steps
        self.ticks_per_step = ticks_per_step
        self.path = list(path)
        self.unguarded_path = self.path if unguarded_path is None else list(unguarded_path)
        self.plan_calls = []

    def _plan_descend_path(self, value, x, y, avoid_monsters=True):
        self.plan_calls.append((x, y, avoid_monsters))
        return list(self.path if avoid_monsters else self.unguarded_path)


def scroll(spell_from=47, container="belt", index=0):
    return {"seed_hi": 1, "seed_lo": 2, "create_info": 3,
            "base_id": module.PORTAL_SCROLL_BASE_ID,
            "container": container, "index": index, "spell_from": spell_from}


def witch_stock(index=module.WITCH_PORTAL_INDEX, price=module.PORTAL_SCROLL_PRICE,
                base_id=module.PORTAL_SCROLL_BASE_ID, can_fit=True):
    return {"vendor": "witch", "index": index, "seed_hi": 7, "seed_lo": 8,
            "create_info": 9, "base_id": base_id, "price": price,
            "can_use": True, "can_fit": can_fit}


def monster(x=11, y=10, hp=10, kind=1, invalid=False):
    return {"x": x, "y": y, "hp": hp, "type": kind, "is_invalid": invalid}


def raw(*, depth=2, hp=100, max_hp=100, belt=(1, 1, 1, 1), x=10, y=10,
        monsters=(), missiles=(), scrolls=None, started=0, gold=1000,
        dead=False, game_over=False, is_set_level=False, portal_enabled=True,
        open_portal=False, portal_lvl=0, portal_xy=(0, 0), town=None,
        required_belt_heals=4):
    state = {"portal_enabled": portal_enabled, "portals_started": started,
             "portal_authorized": False, "max_portals": module.PORTAL_MAX_PER_EPISODE,
             "portal_open": open_portal, "portal_level": portal_lvl,
             "portal_x": portal_xy[0], "portal_y": portal_xy[1],
             "portal_scrolls": [] if scrolls is None else list(scrolls),
             "readiness": {"belt_heals": sum(int(v) in (1, 2, 3, 4) for v in (belt or ())),
                           "required_belt_heals": required_belt_heals},
             "town": {"npcs": [], "stock": [], "active_vendor": "none",
                      "dialog_active": False} if town is None else town}
    value = {"dungeon_level": depth, "is_set_level": is_set_level, "dead": dead,
             "game_over": game_over, "player_x": x, "player_y": y,
             "hp": hp, "max_hp": max_hp, "gold": gold,
             "monsters": [dict(m) for m in monsters],
             "missiles": [dict(m) for m in missiles],
             "resource_state": state}
    if belt is not None:
        value["belt_heal_kinds"] = list(belt)
    return value


def portal_missile(dx=3, dy=0, deleted=False):
    return {"type": TOWN_PORTAL_MISSILE, "tile_dx": dx, "tile_dy": dy, "deleted": deleted}


def town_state(*, npcs=(), stock=(), vendor="none", dialog=False):
    return {"npcs": list(npcs), "stock": list(stock),
            "active_vendor": vendor, "dialog_active": dialog}


ADRIA = {"id": 4, "type": "witch", "x": 80, "y": 20}


def started(trigger="low_hp", value=None, steps=100, bridge=None, policy=None):
    service = PortalService(policy=PortalPolicy() if policy is None else policy)
    bridge = FakeBridge() if bridge is None else bridge
    service.start(raw(scrolls=[scroll()]) if value is None else value,
                  steps, bridge, trigger)
    return service, bridge


# ------------------------------------------------------------------ validators
class PortalProtocolValidationTests(unittest.TestCase):
    def test_protocol_identity_is_versioned_and_reexported(self):
        self.assertEqual(protocol.RESOURCE_PORTAL_PROTOCOLS, ("off", "portal-v1"))
        self.assertIs(module.RESOURCE_PORTAL_PROTOCOLS, protocol.RESOURCE_PORTAL_PROTOCOLS)
        self.assertIs(module.validate_portal_protocol, protocol.validate_portal_protocol)
        self.assertIs(module.validate_native_portal, protocol.validate_native_portal)

    def test_off_is_accepted_under_every_protocol_law_and_retreat(self):
        for name in ("off", "l2-town-v1"):
            for law in ("veto-v1", "coach-v03"):
                for retreat in ("off", "retreat-v1"):
                    with self.subTest(protocol=name, law=law, retreat=retreat):
                        self.assertEqual(
                            protocol.validate_portal_protocol(name, law, retreat, "off"), "off")

    def test_portal_v1_requires_town_protocol_coach_law_and_retreat(self):
        self.assertEqual(
            protocol.validate_portal_protocol("l2-town-v1", "coach-v03", "retreat-v1", "portal-v1"),
            "portal-v1")

    def test_portal_v1_rejects_every_weaker_combination(self):
        for name, law, retreat, needle in (
                ("off", "coach-v03", "retreat-v1", "l2-town-v1"),
                ("l2-town-v1", "veto-v1", "retreat-v1", "l2-town-v1"),
                ("l2-town-v1", "coach-v03", "off", "retreat-v1")):
            with self.subTest(protocol=name, law=law, retreat=retreat):
                with self.assertRaises(ValueError) as caught:
                    protocol.validate_portal_protocol(name, law, retreat, "portal-v1")
                self.assertIn(needle, str(caught.exception))

    def test_unknown_portal_value_is_rejected_before_any_law_check(self):
        for value in ("portal-v2", "on", True, None, ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    protocol.validate_portal_protocol("l2-town-v1", "coach-v03", "retreat-v1", value)
                self.assertIn("Unknown resource_portal", str(caught.exception))

    def test_native_portal_identity_must_match_the_python_contract(self):
        protocol.validate_native_portal({"resource_state": {"portal_enabled": True}}, "portal-v1")
        protocol.validate_native_portal({"resource_state": {"portal_enabled": False}}, "off")
        protocol.validate_native_portal({"resource_state": {}}, "off")
        protocol.validate_native_portal({}, "off")
        for payload, expected in (
                ({"resource_state": {"portal_enabled": False}}, "portal-v1"),
                ({"resource_state": {}}, "portal-v1"),
                ({}, "portal-v1"),
                ({"resource_state": {"portal_enabled": True}}, "off"),
                ({"resource_state": {"portal_enabled": 1}}, "portal-v1")):
            with self.subTest(payload=payload, expected=expected):
                with self.assertRaises(RuntimeError) as caught:
                    protocol.validate_native_portal(payload, expected)
                self.assertIn("isolated rebuild required", str(caught.exception))

    def test_missing_key_is_treated_as_off_not_as_absent_state(self):
        # A bridge that has resource_state but no portal_enabled is a STALE
        # bridge, not an off arm: only a fully empty state may be forgiven.
        with self.assertRaises(RuntimeError):
            protocol.validate_native_portal(
                {"resource_state": {"enabled": True, "retreat_enabled": True}}, "portal-v1")


class PortalPolicyTests(unittest.TestCase):
    def test_defaults_mirror_the_retreat_law(self):
        retreat = importlib.import_module(PACKAGE + ".resource_retreat").RetreatPolicy()
        portal = PortalPolicy()
        for name in ("hp_fraction", "empty_belt_hp_fraction", "pressure_radius",
                     "pressure_count", "drink_hp_fraction"):
            self.assertEqual(getattr(portal, name), getattr(retreat, name), name)

    def test_fractions_and_counts_are_validated(self):
        for kwargs in ({"hp_fraction": 1.5}, {"empty_belt_hp_fraction": -0.1},
                       {"drink_hp_fraction": True}, {"pressure_radius": -1},
                       {"pressure_count": 1.0}, {"cast_clear_radius": -2},
                       {"scroll_budget": -1}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    PortalPolicy(**kwargs)

    def test_as_dict_is_serialisable_and_complete(self):
        keys = set(PortalPolicy().as_dict())
        self.assertEqual(keys, {"hp_fraction", "empty_belt_hp_fraction",
                                "pressure_radius", "pressure_count",
                                "drink_hp_fraction", "cast_clear_radius",
                                "scroll_budget"})


# ------------------------------------------------------------------ the law
class PortalTriggerTests(unittest.TestCase):
    def test_the_outbound_law_is_the_retreat_law_plus_a_carried_scroll(self):
        service = PortalService()
        with_scroll = raw(hp=40, scrolls=[scroll()])
        self.assertEqual(service.trigger_reason(with_scroll, 0), "low_hp")
        self.assertIsNone(service.trigger_reason(raw(hp=40), 0))

    def test_empty_belt_and_pressure_triggers(self):
        service = PortalService()
        self.assertEqual(service.trigger_reason(
            raw(hp=70, belt=(0, 0, 0, 0), scrolls=[scroll()]), 0), "empty_belt")
        pressure = PortalService(policy=PortalPolicy(pressure_count=2))
        self.assertEqual(pressure.trigger_reason(
            raw(hp=100, scrolls=[scroll()], monsters=(monster(), monster(12, 12))), 0),
            "pressure")

    def test_no_trigger_above_the_curriculum_floor_or_in_a_quest_scene(self):
        service = PortalService()
        for value in (raw(depth=1, hp=10, scrolls=[scroll()]),
                      raw(depth=0, hp=10, scrolls=[scroll()], gold=0),
                      raw(hp=10, scrolls=[scroll()], is_set_level=True),
                      raw(hp=0, scrolls=[scroll()]),
                      raw(hp=10, scrolls=[scroll()], dead=True),
                      raw(hp=10, scrolls=[scroll()], game_over=True)):
            with self.subTest(value=value["dungeon_level"]):
                self.assertIsNone(service.trigger_reason(value, 0))

    def test_engine_budget_and_failure_cooldown_close_the_law(self):
        service = PortalService()
        self.assertIsNone(service.trigger_reason(
            raw(hp=10, scrolls=[scroll()], started=module.PORTAL_MAX_PER_EPISODE), 0))
        service._cooldown_until = 500
        self.assertIsNone(service.trigger_reason(raw(hp=10, scrolls=[scroll()]), 499))
        self.assertEqual(service.trigger_reason(raw(hp=10, scrolls=[scroll()]), 500), "low_hp")

    def test_a_standing_portal_is_crossed_instead_of_recast(self):
        service = PortalService()
        value = raw(hp=100, scrolls=[scroll()], open_portal=True, portal_lvl=2)
        self.assertEqual(service.trigger_reason(value, 0), "portal_standing")

    def test_town_buys_the_scroll_only_once_the_belt_law_is_satisfied(self):
        service = PortalService()
        ok = raw(depth=0, gold=500, belt=(1, 1, 1, 1))
        self.assertEqual(service.trigger_reason(ok, 0), "buy_scroll")
        self.assertIsNone(service.trigger_reason(
            raw(depth=0, gold=500, belt=(1, 1, 0, 0)), 0))
        self.assertIsNone(service.trigger_reason(
            raw(depth=0, gold=100, belt=(1, 1, 1, 1)), 0))
        self.assertIsNone(service.trigger_reason(
            raw(depth=0, gold=500, belt=(1, 1, 1, 1), scrolls=[scroll()]), 0))
        service.scrolls_bought = PortalPolicy().scroll_budget
        self.assertIsNone(service.trigger_reason(ok, 0))

    def test_town_prefers_the_return_leg_over_another_purchase(self):
        service = PortalService()
        service.awaiting_return = True
        value = raw(depth=0, gold=500, open_portal=True, portal_lvl=3)
        self.assertEqual(service.trigger_reason(value, 0), "portal_return")
        # Awaiting a return with no portal left is not a shopping opportunity.
        self.assertIsNone(service.trigger_reason(raw(depth=0, gold=500), 0))


class PortalStartTests(unittest.TestCase):
    def test_outbound_authorizes_the_engine_and_starts_at_the_cast(self):
        service, bridge = started()
        self.assertTrue(service.active)
        self.assertEqual((service.mission, service.phase), ("outbound", "cast"))
        self.assertEqual(bridge.calls, [True])

    def test_a_standing_portal_starts_directly_at_the_walk_in(self):
        service, bridge = started(
            trigger="portal_standing",
            value=raw(open_portal=True, portal_lvl=2, scrolls=[scroll()]))
        self.assertEqual(service.phase, "enter")
        self.assertEqual(bridge.calls, [True])

    def test_the_witch_leg_never_touches_the_transition_authority(self):
        service, bridge = started(trigger="buy_scroll",
                                  value=raw(depth=0, gold=500))
        self.assertEqual((service.mission, service.phase), ("restock", "shop"))
        self.assertEqual(bridge.calls, [])

    def test_the_return_leg_authorizes_from_town(self):
        service, bridge = started(
            trigger="portal_return",
            value=raw(depth=0, open_portal=True, portal_lvl=4))
        self.assertEqual((service.mission, service.phase), ("return", "enter"))
        self.assertEqual(bridge.calls, [True])

    def test_scene_preconditions_are_enforced(self):
        with self.assertRaises(RuntimeError):
            PortalService().start(raw(depth=1, scrolls=[scroll()]), 0, FakeBridge(), "low_hp")
        with self.assertRaises(RuntimeError):
            PortalService().start(raw(is_set_level=True), 0, FakeBridge(), "low_hp")
        with self.assertRaises(RuntimeError):
            PortalService().start(raw(depth=2), 0, FakeBridge(), "buy_scroll")
        with self.assertRaises(RuntimeError):
            PortalService().start(raw(depth=2), 0, FakeBridge(), "portal_return")
        service, bridge = started()
        with self.assertRaises(RuntimeError):
            service.start(raw(scrolls=[scroll()]), 0, bridge, "low_hp")


# ------------------------------------------------------------------ outbound
class PortalOutboundTests(unittest.TestCase):
    def test_the_cast_uses_the_exported_spell_from_and_never_use_inv_item(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll(spell_from=49)]), steps=100)
        self.assertEqual(service.command(env, bridge), ("cast_portal", 49))

    def test_a_missing_scroll_hands_back_without_ending_the_episode(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[]), steps=100)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_scroll_missing")
        self.assertFalse(service.active)

    def test_a_cast_that_places_no_portal_is_a_burned_scroll_not_a_retry_loop(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()]), steps=100)
        self.assertEqual(service.command(env, bridge)[0], "cast_portal")
        service.receipt(("cast_portal", 47), {"accepted": True})
        for beat in range(1, module.PORTAL_CAST_BEATS):
            env._steps = 100 + beat
            # in flight: hold (a wait would cancel the queued spell), settle exempt
            self.assertEqual(service.command(env, bridge), ("hold",))
            self.assertTrue(service.settle_exempt)
        env._steps = 100 + module.PORTAL_CAST_BEATS
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_cast_failed")
        self.assertEqual(bridge.calls, [True, False])  # authority revoked

    def test_a_rejected_cast_costs_nothing_and_gives_up_this_leg(self):
        service, bridge = started()
        service.receipt(("cast_portal", 47), {"accepted": False, "reason": "wrong_scene"})
        env = FakeEnv(raw(scrolls=[scroll()]), steps=100)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_cast_rejected")
        self.assertEqual(service.casts, 0)

    def test_the_missile_switches_the_leg_to_the_walk_in(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()], missiles=[portal_missile(dx=3)]),
                      steps=100, path=[(11, 10, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        self.assertEqual(service.phase, "enter")
        self.assertEqual(env.plan_calls[0][:2], (13, 10))

    def test_a_deleted_missile_is_not_a_portal(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()],
                          missiles=[portal_missile(dx=3, deleted=True)]), steps=100)
        self.assertEqual(service.command(env, bridge)[0], "cast_portal")

    def test_standing_on_the_portal_waits_for_the_engine_transition(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()], missiles=[portal_missile(dx=0, dy=0)]),
                      steps=100)
        self.assertEqual(service.command(env, bridge), ("wait",))

    def test_the_hundred_tick_lifetime_is_a_real_deadline(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()], missiles=[portal_missile(dx=3)]),
                      steps=100, path=[(11, 10, False)], ticks_per_step=4)
        service.command(env, bridge)
        self.assertEqual(service._enter_deadline, 125)  # 100 ticks / 4 per microstep
        env._steps = 126
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_expired")

    def test_a_vanished_portal_is_reported_as_expired(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()], missiles=[portal_missile(dx=3)]),
                      steps=100, path=[(11, 10, False)])
        service.command(env, bridge)
        env._raw = raw(scrolls=[scroll()])
        env._steps = 101
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_expired")

    def test_an_unreachable_portal_hands_back(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()], missiles=[portal_missile(dx=3)]),
                      steps=100, path=[], unguarded_path=[])
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_unreachable")

    def test_arrival_in_town_hands_back_and_arms_the_return(self):
        service, bridge = started()
        env = FakeEnv(raw(depth=0), steps=140)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_to_town")
        self.assertTrue(service.awaiting_return)
        self.assertFalse(service.active)
        self.assertEqual(service.attempts[-1]["outcome"], "completed")

    def test_a_failed_outbound_leg_disarms_the_return(self):
        service, bridge = started()
        env = FakeEnv(raw(depth=5, scrolls=[scroll()]), steps=140)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_unexpected_scene")
        self.assertFalse(service.awaiting_return)

    def test_drinking_outranks_walking_while_the_leg_runs(self):
        service, bridge = started()
        env = FakeEnv(raw(hp=20, scrolls=[scroll()], missiles=[portal_missile(dx=3)]),
                      steps=100, path=[(11, 10, False)])
        self.assertEqual(service.command(env, bridge), ("drink",))
        service.receipt(("drink",), {"accepted": True})
        self.assertEqual(service.drinks, 1)
        env._steps = 101
        self.assertEqual(service.command(env, bridge)[0], "walk")  # spacing holds

    def test_the_service_cap_and_death_both_hand_back(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()]), steps=100 + module.PORTAL_SERVICE_CAP)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_cap")
        service2, bridge2 = started()
        env2 = FakeEnv(raw(hp=0, dead=True), steps=110)
        self.assertEqual(service2.command(env2, bridge2), ("complete",))
        self.assertEqual(service2.reason, "portal_died")

    def test_a_revoke_failure_after_death_is_recorded_not_raised(self):
        bridge = FakeBridge(revoke_error="engine is not in a game")
        service, _ = started(bridge=bridge)
        env = FakeEnv(raw(dead=True, hp=0), steps=110)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertIn("revoke_error", service.attempts[-1])

    def test_walk_rejections_give_up_instead_of_spinning(self):
        service, bridge = started()
        env = FakeEnv(raw(scrolls=[scroll()], missiles=[portal_missile(dx=3)]),
                      steps=100, path=[(11, 10, False)])
        for _ in range(module.PORTAL_WALK_REJECTIONS):
            service.receipt(("walk", 11, 10), {"accepted": False})
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_walk_rejected")

    def test_the_clear_cast_radius_option_waits_for_the_crowd_to_thin(self):
        policy = PortalPolicy(cast_clear_radius=4)
        service, bridge = started(policy=policy)
        env = FakeEnv(raw(scrolls=[scroll()], monsters=(monster(),)), steps=100)
        self.assertEqual(service.command(env, bridge), ("wait",))
        env._raw = raw(scrolls=[scroll()])
        self.assertEqual(service.command(env, bridge)[0], "cast_portal")


# ------------------------------------------------------------------ return
class PortalReturnTests(unittest.TestCase):
    def start_return(self, **kwargs):
        value = raw(depth=0, open_portal=True, portal_lvl=3, **kwargs)
        return started(trigger="portal_return", value=value)

    def test_the_town_side_portal_is_found_from_the_missile_list(self):
        service, bridge = self.start_return()
        env = FakeEnv(raw(depth=0, open_portal=True, portal_lvl=3,
                          x=50, y=40, missiles=[portal_missile(dx=7, dy=0)]),
                      steps=10, path=[(51, 40, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 51, 40))
        self.assertEqual(env.plan_calls[0][:2], (57, 40))

    def test_without_a_missile_the_engine_town_anchor_is_used(self):
        service, bridge = self.start_return()
        env = FakeEnv(raw(depth=0, open_portal=True, portal_lvl=3, x=50, y=40),
                      steps=10, path=[(51, 40, False)])
        service.command(env, bridge)
        self.assertEqual(env.plan_calls[0][:2], module.TOWN_PORTAL_POSITION)

    def test_a_store_page_is_dismissed_before_walking(self):
        service, bridge = self.start_return()
        env = FakeEnv(raw(depth=0, open_portal=True, portal_lvl=3,
                          town=town_state(vendor="healer")), steps=10)
        self.assertEqual(service.command(env, bridge), ("dismiss",))

    def test_arrival_back_in_the_dungeon_completes_the_round_trip(self):
        service, bridge = self.start_return()
        service.awaiting_return = True
        env = FakeEnv(raw(depth=3, started=1), steps=40)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_return")
        self.assertFalse(service.awaiting_return)
        self.assertEqual(bridge.calls, [True, False])

    def test_a_closed_portal_ends_the_leg_and_disarms_the_return(self):
        service, bridge = self.start_return()
        service.awaiting_return = True
        env = FakeEnv(raw(depth=0), steps=40)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_closed")
        self.assertFalse(service.awaiting_return)


# ------------------------------------------------------------------ the witch
class PortalRestockTests(unittest.TestCase):
    def start_shop(self, **kwargs):
        value = raw(depth=0, gold=500, **kwargs)
        return started(trigger="buy_scroll", value=value)

    def test_the_leg_walks_to_adria_then_talks_then_buys(self):
        service, bridge = self.start_shop()
        env = FakeEnv(raw(depth=0, gold=500, x=70, y=20,
                          town=town_state(npcs=[ADRIA])), steps=10,
                      path=[(71, 20, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 71, 20))
        env._raw = raw(depth=0, gold=500, x=80, y=19, town=town_state(npcs=[ADRIA]))
        self.assertEqual(service.command(env, bridge), ("talk", 4))
        env._raw = raw(depth=0, gold=500, x=80, y=19,
                       town=town_state(npcs=[ADRIA], vendor="witch",
                                       stock=[witch_stock()]))
        self.assertEqual(service.command(env, bridge),
                         ("buy", "witch", 2, 7, 8, 9, module.PORTAL_SCROLL_BASE_ID))

    def test_only_the_pinned_portal_scroll_is_ever_bought(self):
        service, bridge = self.start_shop()
        env = FakeEnv(raw(depth=0, gold=500, x=80, y=19,
                          town=town_state(npcs=[ADRIA], vendor="witch",
                                          stock=[witch_stock(index=0, base_id=25),
                                                 witch_stock(index=5, base_id=99)])),
                      steps=10)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_scroll_unavailable")

    def test_a_quest_speech_is_dismissed_like_pepins(self):
        service, bridge = self.start_shop()
        env = FakeEnv(raw(depth=0, gold=500, x=80, y=19,
                          town=town_state(npcs=[ADRIA], dialog=True)), steps=10)
        self.assertEqual(service.command(env, bridge), ("dismiss",))

    def test_an_endless_talk_loop_gives_up(self):
        service, bridge = self.start_shop()
        env = FakeEnv(raw(depth=0, gold=500, x=80, y=19,
                          town=town_state(npcs=[ADRIA])), steps=10)
        for _ in range(module.PORTAL_TALK_ATTEMPTS):
            self.assertEqual(service.command(env, bridge), ("talk", 4))
        self.assertEqual(service.command(env, bridge), ("wait",))
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_vendor_unavailable")

    def test_a_purchase_receipt_is_accounted(self):
        service, bridge = self.start_shop()
        service.receipt(("buy", "witch", 2, 7, 8, 9, 27), {"accepted": True, "price": 200})
        self.assertEqual((service.scrolls_bought, service.gold_spent), (1, 200))
        self.assertEqual(service.attempts[-1]["gold_spent"], 200)

    def test_the_engine_belt_reservation_stops_the_leg_immediately(self):
        service, bridge = self.start_shop()
        service.receipt(("buy", "witch", 2, 7, 8, 9, 27),
                        {"accepted": False, "reason": "belt_reserved_for_heals"})
        env = FakeEnv(raw(depth=0, gold=500, x=80, y=19,
                          town=town_state(npcs=[ADRIA], vendor="witch",
                                          stock=[witch_stock()])), steps=10)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_belt_reserved")

    def test_a_scroll_that_arrived_some_other_way_finishes_the_leg(self):
        service, bridge = self.start_shop()
        env = FakeEnv(raw(depth=0, gold=500, scrolls=[scroll()]), steps=10)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "portal_scroll_carried")
        self.assertEqual(service.attempts[-1]["outcome"], "completed")

    def test_the_witch_leg_never_revokes_transition_authority(self):
        service, bridge = self.start_shop()
        env = FakeEnv(raw(depth=0, gold=0), steps=10)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(bridge.calls, [])


# ------------------------------------------------------------------ contract
class PortalServiceContractTests(unittest.TestCase):
    def test_the_service_exposes_the_members_the_resupply_loop_drives(self):
        retreat_module = importlib.import_module(PACKAGE + ".resource_retreat")
        for name in ("service_microstep_cap", "command_microstep_deadline",
                     "record_steps", "trigger_reason", "start", "command",
                     "receipt", "telemetry", "active", "phase", "start_steps"):
            self.assertTrue(hasattr(PortalService(), name), name)
            self.assertTrue(hasattr(retreat_module.RetreatService(), name), name)

    def test_no_failure_path_ever_returns_the_town_services_finish_word(self):
        for reason, value, steps in (
                ("portal_cap", raw(scrolls=[scroll()]), 100 + module.PORTAL_SERVICE_CAP),
                ("portal_died", raw(dead=True, hp=0), 110),
                ("portal_unexpected_scene", raw(is_set_level=True), 110)):
            with self.subTest(reason=reason):
                service, bridge = started()
                command = service.command(FakeEnv(value, steps=steps), bridge)
                self.assertEqual(command, ("complete",))
                self.assertEqual(service.reason, reason)

    def test_record_steps_charges_the_phase_that_spent_them(self):
        service, bridge = started(steps=100)
        env = FakeEnv(raw(scrolls=[scroll()]), steps=100)
        service.command(env, bridge)
        env._steps = 112
        service.record_steps(env._steps, service.phase)
        self.assertEqual(service.phase_steps.get("cast"), 12)
        self.assertEqual(service.steps, 12)

    def test_the_command_deadline_is_the_leg_cap(self):
        service, _ = started(steps=100)
        self.assertEqual(service.service_microstep_cap, module.PORTAL_SERVICE_CAP)
        self.assertEqual(service.command_microstep_deadline,
                         100 + module.PORTAL_SERVICE_CAP)

    def test_telemetry_is_json_shaped_and_names_the_protocol(self):
        service, bridge = started()
        service.command(FakeEnv(raw(depth=0), steps=140), bridge)
        telemetry = service.telemetry()
        self.assertEqual(telemetry["protocol"], "portal-v1")
        for key in ("policy", "attempted", "active", "mission", "phase", "reason",
                    "trigger", "steps", "drinks", "casts", "scrolls_bought",
                    "gold_spent", "awaiting_return", "attempts", "to_town",
                    "returned", "failed", "phase_steps"):
            self.assertIn(key, telemetry)
        self.assertEqual(telemetry["to_town"], 1)
        import json
        json.dumps(telemetry)

    def test_the_engine_budget_constant_matches_the_python_one(self):
        self.assertEqual(module.PORTAL_MAX_PER_EPISODE, 2)


# ------------------------------------------------------------------ native
def _load_native():
    root = Path(__file__).resolve().parents[1]
    if str(root / "python") not in sys.path:
        sys.path.insert(0, str(root / "python"))
    try:
        from diablogym import DiabloGymEnv, bridge, nav  # noqa: F401
    except Exception:  # pragma: no cover - no built bridge in this tree
        return None
    if not all(hasattr(bridge, name) for name in
               ("configure_resource_portal", "act_cast_town_portal",
                "probe_resource_portal_fixture")):
        return None
    return DiabloGymEnv, bridge, nav


_NATIVE = _load_native()
PORTAL_MESSAGES = ("WM_DIABWARPLVL", "WM_DIABRETOWN", "WM_DIABTWARPUP", "WM_DIABTOWNWARP")


@unittest.skipIf(_NATIVE is None, "requires the isolated R18-F portal bridge")
class PortalNativeTests(unittest.TestCase):
    """Engineering fixtures against the real headless engine. The default-off
    verdict freeze is the house law this feature must not break."""

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

    def reset(self, seed=8001, *, portal=False, dungeon=False):
        bridge = self.bridge
        bridge.end_game()
        bridge.configure_resource_protocol(
            True, ordinary_armor_scope=True, preserve_equipment_readiness=True,
            loot_economy=True, readiness_advisory=True, retreat=True, portal=portal)
        raw = bridge.reset(seed)
        if dungeon:
            raw = self.nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            bridge.probe_invincible(True)
        return raw

    # ---- flags ----
    def test_the_portal_flag_round_trips_and_hides_its_keys_when_off(self):
        state = self.reset(portal=True)["resource_state"]
        self.assertIs(state["portal_enabled"], True)
        self.assertEqual(state["portal_authorized"], False)
        self.assertEqual(state["portals_started"], 0)
        self.assertEqual(state["max_portals"], module.PORTAL_MAX_PER_EPISODE)
        for key in ("portal_open", "portal_level", "portal_x", "portal_y",
                    "portal_on_level", "portal_scrolls"):
            self.assertIn(key, state)
        off = self.reset(portal=False)["resource_state"]
        self.assertIs(off["portal_enabled"], False)
        for key in ("portal_authorized", "portals_started", "max_portals",
                    "portal_open", "portal_level", "portal_x", "portal_y",
                    "portal_scrolls"):
            self.assertNotIn(key, off)

    def test_portal_v1_requires_the_coach_law(self):
        self.bridge.end_game()
        with self.assertRaises(ValueError):
            self.bridge.configure_resource_protocol(True, portal=True)
        with self.assertRaises(ValueError):
            self.bridge.configure_resource_protocol(False, portal=True)

    # ---- the default-off freeze (house law) ----
    def test_off_rejects_all_four_portal_and_warp_messages_in_both_scenes(self):
        bridge = self.bridge
        for dungeon in (False, True):
            self.reset(portal=False, dungeon=dungeon)
            # An OPEN portal must not change the off verdict either.
            bridge.probe_resource_portal_fixture(2, 40, 40)
            for name in PORTAL_MESSAGES:
                with self.subTest(dungeon=dungeon, message=name):
                    before = bridge.probe_resource_transition_snapshot()
                    bridge.probe_resource_transition(getattr(bridge, name), -1)
                    receipt = bridge.observe()["resource_state"]["transition"]
                    self.assertFalse(receipt["accepted"], receipt)
                    self.assertEqual(receipt["reason"], "unsupported_portal_or_warp")
                    self.assertEqual(receipt["message"], getattr(bridge, name))
                    if name == "WM_DIABWARPLVL":
                        # The portal branch must not even OVERWRITE the target
                        # the engine passed (-1) while the feature is off.
                        # RestartTownLvl passes 0 and StartNewLvl passes ours.
                        self.assertEqual(receipt["target_depth"], -1)
                    self.assertEqual(bridge.probe_resource_transition_snapshot(), before)

    def test_on_keeps_the_old_verdict_for_the_other_three_messages(self):
        bridge = self.bridge
        for dungeon in (False, True):
            self.reset(portal=True, dungeon=dungeon)
            bridge.probe_resource_portal_fixture(2, 40, 40)
            for name in PORTAL_MESSAGES[1:]:
                with self.subTest(dungeon=dungeon, message=name):
                    bridge.probe_resource_transition(getattr(bridge, name), -1)
                    receipt = bridge.observe()["resource_state"]["transition"]
                    self.assertFalse(receipt["accepted"], receipt)
                    self.assertEqual(receipt["reason"], "unsupported_portal_or_warp")

    def test_on_still_rejects_the_warp_when_the_geometry_is_wrong(self):
        bridge = self.bridge
        # Town with no open portal.
        self.reset(portal=True)
        bridge.probe_resource_portal_fixture(0)
        bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "unsupported_portal_or_warp")
        # Main L1 is below the curriculum floor for a portal departure.
        self.reset(portal=True, dungeon=True)
        bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "unsupported_portal_or_warp")

    def test_an_open_portal_needs_authorization_and_publishes_its_floor(self):
        bridge = self.bridge
        self.reset(portal=True)
        bridge.probe_resource_portal_fixture(4, 40, 40)
        state = bridge.observe()["resource_state"]
        self.assertTrue(state["portal_open"])
        self.assertEqual(state["portal_level"], 4)
        bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "portal_not_authorized")
        self.assertEqual(receipt["target_depth"], 4)

    def test_authorized_return_is_accepted_and_names_the_portal_floor(self):
        bridge = self.bridge
        self.reset(portal=True)
        bridge.probe_resource_portal_fixture(3, 40, 40)
        bridge.configure_resource_portal(True)
        self.assertTrue(bridge.observe()["resource_state"]["portal_authorized"])
        bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "portal_return")
        self.assertEqual(receipt["target_depth"], 3)

    # ---- authorization preconditions ----
    def test_authorization_is_refused_off_the_sanctioned_geometry(self):
        bridge = self.bridge
        self.reset(portal=True)
        with self.assertRaises(RuntimeError):
            bridge.configure_resource_portal(True)      # town, no portal
        bridge.probe_resource_portal_fixture(1, 40, 40)
        with self.assertRaises(RuntimeError):
            bridge.configure_resource_portal(True)      # portal to L1
        self.reset(portal=True, dungeon=True)
        with self.assertRaises(RuntimeError):
            bridge.configure_resource_portal(True)      # main L1
        self.reset(portal=False)
        with self.assertRaises(RuntimeError):
            bridge.configure_resource_portal(True)      # feature off

    def test_revocation_is_always_legal_and_resets_per_episode(self):
        bridge = self.bridge
        self.reset(portal=True)
        bridge.configure_resource_portal(False)
        bridge.probe_resource_portal_fixture(2, 40, 40)
        bridge.configure_resource_portal(True)
        self.assertTrue(bridge.observe()["resource_state"]["portal_authorized"])
        state = self.reset(portal=True)["resource_state"]
        self.assertFalse(state["portal_authorized"])
        self.assertEqual(state["portals_started"], 0)

    # ---- the witch ----
    def test_adria_is_exported_only_under_portal_v1(self):
        on = self.reset(portal=True)["resource_state"]["town"]["npcs"]
        self.assertIn("witch", {npc["type"] for npc in on})
        witch = next(npc for npc in on if npc["type"] == "witch")
        self.assertEqual((witch["x"], witch["y"]), module.WITCH_POSITION)
        off = self.reset(portal=False)["resource_state"]["town"]["npcs"]
        self.assertEqual({npc["type"] for npc in off}, {"smith", "healer"})

    def test_the_witch_purchase_api_is_closed_when_the_feature_is_off(self):
        self.reset(portal=False)
        receipt = self.bridge.act_buy_store_item("witch", 2, 0, 0, 0, 0)
        self.assertFalse(receipt["accepted"])
        self.assertIn(receipt["reason"], ("unavailable", "wrong_vendor"))

    # ---- the cast ----
    def test_the_cast_action_is_closed_off_and_refuses_town(self):
        self.reset(portal=False)
        receipt = self.bridge.act_cast_town_portal(47)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "unavailable")
        self.reset(portal=True)
        receipt = self.bridge.act_cast_town_portal(47)
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["reason"], "wrong_scene")

    def test_the_cast_action_refuses_an_out_of_range_spell_slot(self):
        self.reset(portal=True, dungeon=True)
        for slot in (-1, 55, 999):
            receipt = self.bridge.act_cast_town_portal(slot)
            self.assertFalse(receipt["accepted"])
            self.assertIn(receipt["reason"], ("invalid_item", "wrong_scene"))

    # ---- the two real transitions ----
    def settle(self, depth, *, budget=80):
        raw = None
        for _ in range(budget):
            raw = self.bridge.step(1)
            if (int(raw["dungeon_level"]) == depth
                    and raw["player_mode"] == self.bridge.PM_STAND):
                return raw
        self.fail(f"level {depth} not reached within {budget} ticks")

    def reach_l2(self, seed=8001):
        """Main L2 through the ordinary descent messages (coach-v03 is advisory,
        so an unready L1 -> L2 is recorded, not vetoed)."""
        bridge = self.bridge
        self.reset(seed, portal=True, dungeon=True)
        bridge.probe_resource_transition(bridge.WM_DIABNEXTLVL, 2)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertTrue(receipt["accepted"], receipt)
        return self.settle(2)

    def test_the_outbound_leg_needs_main_l2_and_an_explicit_authorization(self):
        bridge = self.bridge
        self.reach_l2()
        bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertFalse(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "portal_not_authorized")
        self.assertEqual(receipt["target_depth"], 0)
        bridge.configure_resource_portal(True)
        bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "portal_to_town")
        self.assertEqual(receipt["target_depth"], 0)
        self.settle(0)
        state = bridge.observe()["resource_state"]
        # The arrival opens the shops WITHOUT spending a loot-economy town trip,
        # consumes the authorization, and does not yet count a round trip.
        self.assertTrue(state["service_trip"])
        self.assertFalse(state["portal_authorized"])
        self.assertEqual(state["portals_started"], 0)
        self.assertIn("witch", {npc["type"] for npc in state["town"]["npcs"]})

    def test_a_round_trip_is_counted_on_arrival_and_capped_at_max_portals(self):
        bridge = self.bridge
        self.reset(portal=True)
        for trip in (1, 2):
            bridge.probe_resource_portal_fixture(2, 40, 40)
            bridge.configure_resource_portal(True)
            bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
            receipt = bridge.observe()["resource_state"]["transition"]
            self.assertTrue(receipt["accepted"], receipt)
            self.assertEqual(receipt["reason"], "portal_return")
            self.settle(2)
            state = bridge.observe()["resource_state"]
            self.assertEqual(state["portals_started"], trip)
            self.assertFalse(state["portal_authorized"])
            # A portal return skips the depth == 1 branch that normally clears
            # the town-trip flags, so the return branch has to clear them itself.
            self.assertFalse(state["service_trip"])
            self.assertFalse(state["service_authorized"])
            if trip == 1:
                bridge.configure_resource_portal(True)
                bridge.probe_resource_transition(bridge.WM_DIABWARPLVL, -1)
                self.settle(0)
                self.assertTrue(bridge.observe()["resource_state"]["service_trip"])
        with self.assertRaises(RuntimeError) as caught:
            bridge.configure_resource_portal(True)
        self.assertIn("at most two", str(caught.exception))

    # ---- the whole feature, once, through the real engine ----
    def walk_onto(self, tile, *, depth_change_to=None, budget=200):
        """Ordinary pathed walk; stops when the tile is reached or the scene
        changes. No fixture teleports."""
        bridge = self.bridge
        raw = bridge.observe()
        for _ in range(budget):
            if depth_change_to is not None and int(raw["dungeon_level"]) == depth_change_to:
                return raw
            if (raw["player_mode"] == bridge.PM_STAND
                    and (raw["player_x"], raw["player_y"]) != tuple(tile)):
                bridge.act_walk(int(tile[0]), int(tile[1]))
            raw = bridge.step(1)
        return raw

    def portal_missile_tile(self, raw):
        return next(((raw["player_x"] + m["tile_dx"], raw["player_y"] + m["tile_dy"])
                     for m in raw["missiles"]
                     if m["type"] == module.TOWN_PORTAL_MISSILE and not m["deleted"]), None)

    def test_a_real_scroll_read_on_l2_places_a_portal_and_crosses_to_town(self):
        bridge = self.bridge
        raw = self.reach_l2()
        bridge.probe_resource_add_portal_scroll(False)
        scrolls = bridge.observe()["resource_state"]["portal_scrolls"]
        self.assertEqual(len(scrolls), 1)
        self.assertEqual(scrolls[0]["base_id"], module.PORTAL_SCROLL_BASE_ID)
        self.assertEqual(scrolls[0]["container"], "inventory")

        receipt = bridge.act_cast_town_portal(int(scrolls[0]["spell_from"]))
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "cast_requested")
        for _ in range(30):
            raw = bridge.step(4)
            if bridge.observe()["resource_state"]["portal_open"]:
                break
        state = bridge.observe()["resource_state"]
        self.assertTrue(state["portal_open"], state)
        self.assertEqual(state["portal_level"], 2)
        self.assertEqual(len(state["portal_scrolls"]), 0)   # the scroll is spent
        self.assertEqual(self.portal_missile_tile(raw),
                         (state["portal_x"], state["portal_y"]))
        # A second read on the same floor would only replace the first portal.
        for _ in range(20):
            if raw["player_mode"] == bridge.PM_STAND:
                break
            raw = bridge.step(1)
        self.assertEqual(bridge.act_cast_town_portal(9)["reason"], "portal_already_open")

        bridge.configure_resource_portal(True)
        raw = self.walk_onto((state["portal_x"], state["portal_y"]), depth_change_to=0)
        self.assertEqual(int(raw["dungeon_level"]), 0)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "portal_to_town")
        state = bridge.observe()["resource_state"]
        self.assertTrue(state["service_trip"])          # the shops are open
        self.assertFalse(state["portal_authorized"])    # consumed on arrival
        self.assertEqual(state["portals_started"], 0)   # a round trip is not done yet
        self.assertEqual(self.portal_missile_tile(raw), module.TOWN_PORTAL_POSITION)

    def test_the_return_leg_lands_on_the_tile_the_scroll_was_read_from(self):
        bridge = self.bridge
        raw = self.reach_l2()
        bridge.probe_resource_add_portal_scroll(False)
        scrolls = bridge.observe()["resource_state"]["portal_scrolls"]
        bridge.act_cast_town_portal(int(scrolls[0]["spell_from"]))
        for _ in range(30):
            raw = bridge.step(4)
            if bridge.observe()["resource_state"]["portal_open"]:
                break
        cast_tile = tuple(bridge.observe()["resource_state"][k] for k in ("portal_x", "portal_y"))
        bridge.configure_resource_portal(True)
        raw = self.walk_onto(cast_tile, depth_change_to=0)
        self.assertEqual(int(raw["dungeon_level"]), 0)

        town_tile = self.portal_missile_tile(raw)
        bridge.configure_resource_portal(True)
        raw = self.walk_onto(town_tile, depth_change_to=2)
        receipt = bridge.observe()["resource_state"]["transition"]
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "portal_return")
        self.assertEqual(receipt["target_depth"], 2)
        self.assertEqual(int(raw["dungeon_level"]), 2)
        # The known hazard, frozen as a fact: the pair reappears on the exact
        # tile the scroll was read on, among whatever survived the fight.
        self.assertEqual((raw["player_x"], raw["player_y"]), cast_tile)
        state = bridge.observe()["resource_state"]
        self.assertEqual(state["portals_started"], 1)
        self.assertFalse(state["portal_authorized"])
        self.assertFalse(state["portal_open"])          # one scroll = one round trip
        self.assertFalse(state["service_trip"])
        self.assertFalse(state["service_authorized"])

    def arrive_in_town_by_portal(self):
        """Reach town the way the feature intends to: read a scroll on L2 and
        walk through. This is also what opens the shops on that trip."""
        bridge = self.bridge
        raw = self.reach_l2()
        bridge.probe_resource_add_portal_scroll(False)
        scrolls = bridge.observe()["resource_state"]["portal_scrolls"]
        bridge.act_cast_town_portal(int(scrolls[0]["spell_from"]))
        for _ in range(30):
            raw = bridge.step(4)
            if bridge.observe()["resource_state"]["portal_open"]:
                break
        state = bridge.observe()["resource_state"]
        bridge.configure_resource_portal(True)
        raw = self.walk_onto((state["portal_x"], state["portal_y"]), depth_change_to=0)
        self.assertEqual(int(raw["dungeon_level"]), 0)
        self.assertTrue(bridge.observe()["resource_state"]["service_trip"])
        return raw

    def test_the_witch_sells_only_the_pinned_scroll_and_reserves_the_belt(self):
        bridge = self.bridge
        raw = self.arrive_in_town_by_portal()
        bridge.probe_resource_add_gold(500)
        raw = bridge.step(1)
        adria = next(npc for npc in bridge.observe()["resource_state"]["town"]["npcs"]
                     if npc["type"] == "witch")
        self.assertEqual((adria["x"], adria["y"]), module.WITCH_POSITION)
        raw = self.walk_onto((adria["x"], adria["y"] - 1), budget=900)
        self.assertEqual((raw["player_x"], raw["player_y"]), (adria["x"], adria["y"] - 1))
        self.assertEqual(bridge.act_talk_towner(int(adria["id"])), 1)
        for _ in range(20):
            raw = bridge.step(1)
            if bridge.observe()["resource_state"]["town"]["active_vendor"] == "witch":
                break
        state = bridge.observe()["resource_state"]
        self.assertEqual(state["town"]["active_vendor"], "witch")
        stock = [item for item in state["town"]["stock"] if item["vendor"] == "witch"]
        self.assertTrue(stock)
        pinned = next(item for item in stock
                      if int(item["base_id"]) == module.PORTAL_SCROLL_BASE_ID)
        # SortVendor(WitchItems, 3) never moves the three pinned rows, so the
        # Scroll of Town Portal is always WitchItems[2] at 200 gold.
        self.assertEqual(int(pinned["index"]), module.WITCH_PORTAL_INDEX)
        self.assertEqual(int(pinned["price"]), module.PORTAL_SCROLL_PRICE)
        key = tuple(int(pinned[k]) for k in
                    ("index", "seed_hi", "seed_lo", "create_info", "base_id"))

        # The four-heal readiness law owns the belt before the scroll does.
        bridge.probe_resource_set_belt_heals(2)
        bridge.step(1)
        receipt = bridge.act_buy_store_item("witch", *key)
        self.assertFalse(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "belt_reserved_for_heals")

        bridge.probe_resource_set_belt_heals(4)
        bridge.step(1)
        gold_before = int(bridge.observe()["gold"])
        receipt = bridge.act_buy_store_item("witch", *key)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "purchased")
        self.assertEqual(int(receipt["price"]), module.PORTAL_SCROLL_PRICE)
        bridge.step(1)
        state = bridge.observe()["resource_state"]
        self.assertEqual(int(bridge.observe()["gold"]),
                         gold_before - module.PORTAL_SCROLL_PRICE)
        self.assertEqual(len(state["portal_scrolls"]), 1)
        self.assertEqual(state["readiness"]["belt_heals"], 4)  # the heals survive

        # The rest of Adria's catalogue is never reachable from this API.
        other = next(item for item in stock
                     if int(item["base_id"]) != module.PORTAL_SCROLL_BASE_ID)
        okey = tuple(int(other[k]) for k in
                     ("index", "seed_hi", "seed_lo", "create_info", "base_id"))
        receipt = bridge.act_buy_store_item("witch", *okey)
        self.assertFalse(receipt["accepted"], receipt)
        self.assertEqual(receipt["reason"], "unsupported_item")


if __name__ == "__main__":
    unittest.main()
