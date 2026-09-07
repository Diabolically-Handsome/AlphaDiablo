"""R18-A retreat-v1 boundaries: pure Python doubles, no native engine.

The bridge extension is never imported.  The modules under test are loaded
through a private package alias whose ``__path__`` points at
``python/diablogym`` (the house pattern of ``test_resource_sustain_loot.py``,
which keeps ``diablogym/__init__.py`` and therefore the .so out of the way),
plus a stub ``bridge`` submodule so the ``worker_env`` constructor guard can be
exercised without the extension.  Nothing here resets the engine, plans a
native path, probes a tile or constructs DiabloGymEnv/OptionsEnv.
"""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

WM_DIABPREVLVL = 1027  # the engine trigger id for up stairs; a fake bridge value

PACKAGE = "_r18a_retreat_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/diablogym")]
    sys.modules[PACKAGE] = package
if PACKAGE + ".bridge" not in sys.modules:
    # Only ``_walk``'s door branch and worker_env's transitive imports reach
    # for the bridge; every path exercised below keeps ``door`` false.
    bridge_stub = ModuleType(PACKAGE + ".bridge")
    bridge_stub.WM_DIABPREVLVL = WM_DIABPREVLVL
    sys.modules[PACKAGE + ".bridge"] = bridge_stub

module = importlib.import_module(PACKAGE + ".resource_retreat")
protocol = importlib.import_module(PACKAGE + ".resource_protocol")
loot = importlib.import_module(PACKAGE + ".resource_sustain_loot")
try:
    worker_env = importlib.import_module(PACKAGE + ".worker_env")
    WORKER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - only when gym/numpy are absent
    worker_env, WORKER_IMPORT_ERROR = None, exc

RetreatPolicy = module.RetreatPolicy
RetreatService = module.RetreatService


class FakeBridge:
    """Records the retreat authority the script grants and revokes."""

    WM_DIABPREVLVL = WM_DIABPREVLVL

    def __init__(self, revoke_error=None):
        self.calls = []
        self.revoke_error = revoke_error

    def configure_retreat(self, enabled):
        self.calls.append(bool(enabled))
        if not enabled and self.revoke_error is not None:
            raise RuntimeError(self.revoke_error)
        return bool(enabled)


class FakeEnv:
    """The two attributes and one planner call the service asks of an env."""

    def __init__(self, value, steps=0, path=(), unguarded_path=None):
        self._raw = value
        self._steps = steps
        self.path = list(path)
        self.unguarded_path = self.path if unguarded_path is None else list(unguarded_path)
        self.plan_calls = []

    def _plan_descend_path(self, value, x, y, avoid_monsters=True):
        self.plan_calls.append((x, y, avoid_monsters))
        return list(self.path if avoid_monsters else self.unguarded_path)


def monster(x=11, y=10, hp=10, kind=1, invalid=False):
    return {"x": x, "y": y, "hp": hp, "type": kind, "is_invalid": invalid}


def raw(*, depth=2, hp=100, max_hp=100, belt=(1, 1, 0, 0), readiness_belt=None,
        x=10, y=10, stairs=(20, 20), extra_stairs=(), monsters=(), started=0,
        dead=False, game_over=False, is_set_level=False, retreat_enabled=True):
    """Native payload double for main L2 with the up stairs in the trigger list."""
    state = {"retreat_enabled": retreat_enabled, "retreats_started": started}
    if readiness_belt is not None:
        state["readiness"] = {"belt_heals": readiness_belt}
    triggers = [{"msg": WM_DIABPREVLVL + 1, "x": x, "y": y}]  # decoy: down stairs
    if stairs is not None:
        triggers.append({"msg": WM_DIABPREVLVL, "x": stairs[0], "y": stairs[1]})
    for sx, sy in extra_stairs:
        triggers.append({"msg": WM_DIABPREVLVL, "x": sx, "y": sy})
    value = {"dungeon_level": depth, "is_set_level": is_set_level, "dead": dead,
             "game_over": game_over, "player_x": x, "player_y": y,
             "hp": hp, "max_hp": max_hp,
             "monsters": [dict(m) for m in monsters],
             "triggers": triggers, "resource_state": state}
    if belt is not None:
        value["belt_heal_kinds"] = list(belt)
    return value


def started(policy=None, value=None, steps=100, bridge=None, trigger="low_hp"):
    service = RetreatService(policy=RetreatPolicy() if policy is None else policy)
    bridge = FakeBridge() if bridge is None else bridge
    service.start(raw() if value is None else value, steps, bridge, trigger)
    return service, bridge


class RetreatProtocolValidationTests(unittest.TestCase):
    def test_protocol_identity_is_versioned_and_reexported(self):
        self.assertEqual(protocol.RESOURCE_RETREAT_PROTOCOLS, ("off", "retreat-v1"))
        self.assertIs(module.RESOURCE_RETREAT_PROTOCOLS,
                      protocol.RESOURCE_RETREAT_PROTOCOLS)
        self.assertIs(module.validate_retreat_protocol,
                      protocol.validate_retreat_protocol)
        self.assertIs(module.validate_native_retreat, protocol.validate_native_retreat)

    def test_off_is_accepted_under_every_protocol_and_law(self):
        for name in ("off", "l2-town-v1"):
            for law in ("veto-v1", "coach-v03"):
                with self.subTest(protocol=name, law=law):
                    self.assertEqual(
                        protocol.validate_retreat_protocol(name, law, "off"), "off")

    def test_retreat_v1_requires_the_town_protocol_under_the_coach_law(self):
        self.assertEqual(
            protocol.validate_retreat_protocol("l2-town-v1", "coach-v03", "retreat-v1"),
            "retreat-v1")

    def test_retreat_v1_rejects_every_other_protocol_or_law(self):
        for name, law in (("off", "coach-v03"), ("l2-town-v1", "veto-v1"),
                          ("off", "veto-v1")):
            with self.subTest(protocol=name, law=law):
                with self.assertRaises(ValueError) as caught:
                    protocol.validate_retreat_protocol(name, law, "retreat-v1")
                self.assertIn("l2-town-v1", str(caught.exception))

    def test_unknown_retreat_value_is_rejected_before_any_law_check(self):
        for value in ("retreat-v2", "on", True, None, ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    protocol.validate_retreat_protocol("l2-town-v1", "coach-v03", value)
                self.assertIn("Unknown resource_retreat", str(caught.exception))

    def test_constants_match_the_published_interface(self):
        self.assertEqual(module.RETREAT_SERVICE_CAP, 900)
        self.assertEqual(module.RETREAT_MAX_PER_EPISODE, 3)
        self.assertEqual(module.RETREAT_FAILURE_COOLDOWN, 300)
        self.assertEqual(module.RETREAT_DRINK_SPACING, 20)
        self.assertEqual(module.RETREAT_WALK_REJECTIONS, 8)


class NativeRetreatIdentityTests(unittest.TestCase):
    def test_absent_resource_state_passes_only_when_retreat_is_off(self):
        for value in ({}, {"dungeon_level": 2}, {"resource_state": {}}, None, []):
            with self.subTest(raw=value):
                self.assertIsNone(protocol.validate_native_retreat(value, "off"))
                with self.assertRaises(RuntimeError):
                    protocol.validate_native_retreat(value, "retreat-v1")

    def test_missing_or_false_flag_fails_closed_against_retreat_v1(self):
        for state in ({"enabled": True}, {"retreat_enabled": False}):
            with self.subTest(state=state):
                with self.assertRaises(RuntimeError) as caught:
                    protocol.validate_native_retreat({"resource_state": state},
                                                     "retreat-v1")
                self.assertIn("identity mismatch", str(caught.exception))

    def test_native_flag_on_while_python_is_off_fails_closed(self):
        with self.assertRaises(RuntimeError):
            protocol.validate_native_retreat(
                {"resource_state": {"retreat_enabled": True}}, "off")

    def test_matching_identities_pass_in_both_directions(self):
        self.assertIsNone(protocol.validate_native_retreat(
            {"resource_state": {"retreat_enabled": True}}, "retreat-v1"))
        self.assertIsNone(protocol.validate_native_retreat(
            {"resource_state": {"retreat_enabled": False, "enabled": True}}, "off"))

    def test_non_boolean_flag_is_not_accepted_as_truth(self):
        for observed in (1, 0, "true", None):
            with self.subTest(observed=observed):
                with self.assertRaises(RuntimeError):
                    protocol.validate_native_retreat(
                        {"resource_state": {"retreat_enabled": observed}}, "retreat-v1")

    def test_default_expectation_is_off(self):
        self.assertIsNone(protocol.validate_native_retreat({"resource_state": {}}))
        with self.assertRaises(RuntimeError):
            protocol.validate_native_retreat(
                {"resource_state": {"retreat_enabled": True}})


class RetreatPolicyTests(unittest.TestCase):
    def test_defaults_are_the_published_law(self):
        policy = RetreatPolicy()
        self.assertEqual(policy.as_dict(), {
            "hp_fraction": 0.5, "empty_belt_hp_fraction": 0.75,
            "pressure_radius": 6, "pressure_count": 0, "drink_hp_fraction": 0.4})

    def test_as_dict_reflects_explicit_values_and_is_a_plain_copy(self):
        policy = RetreatPolicy(hp_fraction=0.25, empty_belt_hp_fraction=1.0,
                               pressure_radius=3, pressure_count=4,
                               drink_hp_fraction=0.0)
        value = policy.as_dict()
        self.assertEqual(value, {"hp_fraction": 0.25, "empty_belt_hp_fraction": 1.0,
                                 "pressure_radius": 3, "pressure_count": 4,
                                 "drink_hp_fraction": 0.0})
        value["hp_fraction"] = 9
        self.assertEqual(policy.hp_fraction, 0.25)

    def test_fractions_must_lie_in_the_unit_interval(self):
        for name in ("hp_fraction", "empty_belt_hp_fraction", "drink_hp_fraction"):
            for value in (-0.01, 1.01, 2, -1):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError) as caught:
                        RetreatPolicy(**{name: value})
                    self.assertIn(name, str(caught.exception))

    def test_fraction_boundaries_are_inclusive(self):
        self.assertEqual(RetreatPolicy(hp_fraction=0.0).hp_fraction, 0.0)
        self.assertEqual(RetreatPolicy(hp_fraction=1).hp_fraction, 1)

    def test_boolean_and_non_numeric_fractions_are_rejected(self):
        for value in (True, False, "0.5", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RetreatPolicy(hp_fraction=value)

    def test_pressure_fields_must_be_non_negative_plain_ints(self):
        for name in ("pressure_radius", "pressure_count"):
            for value in (-1, 2.0, True, "3", None):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError) as caught:
                        RetreatPolicy(**{name: value})
                    self.assertIn(name, str(caught.exception))
        self.assertEqual(RetreatPolicy(pressure_radius=0, pressure_count=0).as_dict()[
            "pressure_radius"], 0)

    def test_policy_is_frozen(self):
        policy = RetreatPolicy()
        with self.assertRaises(Exception):
            policy.hp_fraction = 0.9


class BeltAndMonsterHelperTests(unittest.TestCase):
    def test_belt_heal_kinds_counts_only_healing_potions(self):
        self.assertEqual(module.belt_heals({"belt_heal_kinds": [1, 1, 0, 0]}), 2)
        self.assertEqual(module.belt_heals({"belt_heal_kinds": [1, 2, 3, 4]}), 4)
        self.assertEqual(module.belt_heals({"belt_heal_kinds": [0, 0]}), 0)
        self.assertEqual(module.belt_heals({"belt_heal_kinds": []}), 0)
        self.assertEqual(module.belt_heals({"belt_heal_kinds": [5, 9, 0]}), 0)

    def test_belt_falls_back_to_the_native_readiness_block(self):
        self.assertEqual(module.belt_heals(
            {"resource_state": {"readiness": {"belt_heals": 3}}}), 3)
        self.assertEqual(module.belt_heals({"resource_state": {}}), 0)
        self.assertEqual(module.belt_heals({}), 0)

    def test_alive_monsters_within_uses_chebyshev_radius(self):
        value = raw(x=10, y=10, monsters=[monster(x=16, y=10), monster(x=17, y=10),
                                          monster(x=14, y=14)])
        self.assertEqual(module.alive_monsters_within(value, 6), 2)
        self.assertEqual(module.alive_monsters_within(value, 4), 1)
        self.assertEqual(module.alive_monsters_within(value, 0), 0)

    def test_dead_invalid_and_excluded_types_never_count_as_pressure(self):
        value = raw(monsters=[monster(hp=0), monster(kind=109), monster(invalid=True),
                              monster(x=11, y=11)])
        self.assertEqual(module.alive_monsters_within(value, 6), 1)

    def test_missing_monster_list_is_no_pressure(self):
        self.assertEqual(module.alive_monsters_within(
            {"player_x": 1, "player_y": 1}, 6), 0)

    def test_retreats_started_reads_the_engine_counter(self):
        self.assertEqual(module.retreats_started({}), 0)
        self.assertEqual(module.retreats_started({"resource_state": {}}), 0)
        self.assertEqual(module.retreats_started(
            {"resource_state": {"retreats_started": None}}), 0)
        self.assertEqual(module.retreats_started(
            {"resource_state": {"retreats_started": 2}}), 2)


class TriggerLawTests(unittest.TestCase):
    def setUp(self):
        self.service = RetreatService()

    def test_town_and_l1_never_retreat(self):
        for depth in (0, 1):
            with self.subTest(depth=depth):
                self.assertIsNone(self.service.trigger_reason(raw(depth=depth, hp=10), 0))

    def test_death_game_over_set_level_and_missing_raw_never_retreat(self):
        self.assertIsNone(self.service.trigger_reason(raw(hp=10, dead=True), 0))
        self.assertIsNone(self.service.trigger_reason(raw(hp=10, game_over=True), 0))
        self.assertIsNone(self.service.trigger_reason(raw(hp=10, is_set_level=True), 0))
        self.assertIsNone(self.service.trigger_reason(raw(hp=0), 0))
        self.assertIsNone(self.service.trigger_reason(None, 0))

    def test_an_active_retreat_does_not_retrigger(self):
        service, _ = started()
        self.assertTrue(service.active)
        self.assertIsNone(service.trigger_reason(raw(hp=10), 200))

    def test_failure_cooldown_suppresses_the_law_until_it_elapses(self):
        self.service._cooldown_until = 500
        self.assertIsNone(self.service.trigger_reason(raw(hp=10), 499))
        self.assertEqual(self.service.trigger_reason(raw(hp=10), 500), "low_hp")

    def test_episode_budget_of_three_retreats_is_final(self):
        self.assertEqual(self.service.trigger_reason(raw(hp=10, started=2), 0), "low_hp")
        for count in (3, 4, 9):
            with self.subTest(started=count):
                self.assertIsNone(
                    self.service.trigger_reason(raw(hp=10, started=count), 0))

    def test_low_hp_fires_at_or_below_half(self):
        self.assertEqual(self.service.trigger_reason(raw(hp=50, max_hp=100), 0), "low_hp")
        self.assertEqual(self.service.trigger_reason(raw(hp=49, max_hp=100), 0), "low_hp")
        self.assertIsNone(self.service.trigger_reason(raw(hp=51, max_hp=100), 0))

    def test_empty_belt_raises_the_threshold_to_three_quarters(self):
        for belt in ([], [0, 0], [5, 9]):
            with self.subTest(belt=belt):
                self.assertEqual(self.service.trigger_reason(
                    raw(hp=75, max_hp=100, belt=belt), 0), "empty_belt")
                self.assertIsNone(self.service.trigger_reason(
                    raw(hp=76, max_hp=100, belt=belt), 0))

    def test_a_stocked_belt_keeps_the_half_threshold(self):
        self.assertIsNone(self.service.trigger_reason(
            raw(hp=70, max_hp=100, belt=[1, 1, 0, 0]), 0))
        self.assertIsNone(self.service.trigger_reason(
            raw(hp=70, max_hp=100, belt=None, readiness_belt=4), 0))
        self.assertEqual(self.service.trigger_reason(
            raw(hp=70, max_hp=100, belt=None, readiness_belt=0), 0), "empty_belt")

    def test_pressure_is_off_by_default_even_in_a_crowd(self):
        crowd = [monster(x=11, y=10), monster(x=12, y=10), monster(x=13, y=10)]
        self.assertIsNone(self.service.trigger_reason(
            raw(hp=100, max_hp=100, monsters=crowd), 0))

    def test_pressure_fires_only_at_the_configured_count_and_radius(self):
        service = RetreatService(policy=RetreatPolicy(pressure_count=2,
                                                      pressure_radius=6))
        two = [monster(x=11, y=10), monster(x=12, y=10)]
        self.assertEqual(service.trigger_reason(raw(monsters=two), 0), "pressure")
        self.assertIsNone(service.trigger_reason(raw(monsters=two[:1]), 0))
        far = [monster(x=30, y=30), monster(x=31, y=31)]
        self.assertIsNone(service.trigger_reason(raw(monsters=far), 0))

    def test_pressure_ignores_corpses_and_the_excluded_type(self):
        service = RetreatService(policy=RetreatPolicy(pressure_count=2))
        mixed = [monster(x=11, y=10), monster(x=12, y=10, hp=0),
                 monster(x=13, y=10, kind=109), monster(x=14, y=10, invalid=True)]
        self.assertIsNone(service.trigger_reason(raw(monsters=mixed), 0))

    def test_hp_trigger_outranks_the_belt_and_pressure_triggers(self):
        service = RetreatService(policy=RetreatPolicy(pressure_count=1))
        value = raw(hp=10, max_hp=100, belt=[], monsters=[monster()])
        self.assertEqual(service.trigger_reason(value, 0), "low_hp")


class RetreatStartTests(unittest.TestCase):
    def test_start_grants_native_authority_and_opens_an_attempt(self):
        bridge = FakeBridge()
        service = RetreatService()
        value = raw(depth=3, hp=40, max_hp=100, belt=[1, 0, 0, 0],
                    monsters=[monster()])
        service.start(value, 250, bridge, "low_hp")
        self.assertEqual(bridge.calls, [True])
        self.assertTrue(service.attempted)
        self.assertTrue(service.active)
        self.assertEqual(service.phase, "ascend")
        self.assertEqual(service.trigger, "low_hp")
        self.assertIsNone(service.reason)
        self.assertEqual(service.start_steps, 250)
        self.assertEqual(service.start_depth, 3)
        self.assertEqual(service.steps, 0)
        self.assertEqual(len(service.attempts), 1)
        attempt = service.attempts[0]
        self.assertEqual(attempt["trigger"], "low_hp")
        self.assertEqual(attempt["beat0"], 250)
        self.assertEqual(attempt["depth0"], 3)
        self.assertEqual(attempt["hp0"], 40)
        self.assertEqual(attempt["max_hp"], 100)
        self.assertEqual(attempt["belt0"], 1)
        self.assertEqual(attempt["monsters_near0"], 1)
        self.assertIsNone(attempt["outcome"])
        self.assertEqual(attempt["drinks"], 0)

    def test_start_refuses_a_second_concurrent_retreat(self):
        service, bridge = started()
        with self.assertRaises(RuntimeError) as caught:
            service.start(raw(), 200, bridge, "low_hp")
        self.assertIn("already active", str(caught.exception))
        self.assertEqual(bridge.calls, [True])
        self.assertEqual(len(service.attempts), 1)

    def test_start_refuses_town_l1_and_set_levels(self):
        for value in (raw(depth=0), raw(depth=1), raw(is_set_level=True)):
            with self.subTest(depth=value["dungeon_level"],
                              set_level=value["is_set_level"]):
                bridge = FakeBridge()
                service = RetreatService()
                with self.assertRaises(RuntimeError):
                    service.start(value, 0, bridge, "low_hp")
                self.assertEqual(bridge.calls, [])
                self.assertFalse(service.attempted)
                self.assertFalse(service.active)
                self.assertEqual(service.attempts, [])


class RetreatCommandTests(unittest.TestCase):
    def test_walks_the_first_planner_step_towards_the_nearest_up_stairs(self):
        service, bridge = started()
        env = FakeEnv(raw(), steps=101, path=[(11, 10, False), (12, 10, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        self.assertEqual(env.plan_calls, [(20, 20, True)])
        self.assertTrue(service.active)
        self.assertEqual(service.phase, "ascend")

    def test_the_nearest_of_several_up_stairs_is_targeted(self):
        service, bridge = started()
        value = raw(x=10, y=10, stairs=(40, 40), extra_stairs=[(13, 12)])
        env = FakeEnv(value, steps=101, path=[(11, 10, False)])
        service.command(env, bridge)
        self.assertEqual(env.plan_calls, [(13, 12, True)])

    def test_standing_on_the_stairs_waits_for_the_engine_ascent(self):
        service, bridge = started()
        env = FakeEnv(raw(x=20, y=20, stairs=(20, 20)), steps=101,
                      path=[(21, 20, False)])
        self.assertEqual(service.command(env, bridge), ("wait",))
        self.assertEqual(env.plan_calls, [])
        self.assertTrue(service.active)

    def test_missing_up_stairs_hands_control_back_and_arms_the_cooldown(self):
        service, bridge = started(steps=100)
        env = FakeEnv(raw(stairs=None), steps=140)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "retreat_stairs_missing")
        self.assertFalse(service.active)
        self.assertEqual(service.phase, "done")
        self.assertEqual(bridge.calls, [True, False])
        attempt = service.attempts[-1]
        self.assertEqual(attempt["outcome"], "failed")
        self.assertEqual(attempt["reason"], "retreat_stairs_missing")
        self.assertEqual(attempt["beat1"], 140)
        self.assertEqual(attempt["hp1"], 100)
        self.assertIsNone(service.trigger_reason(raw(hp=10), 140 + 299))
        self.assertEqual(service.trigger_reason(raw(hp=10), 140 + 300), "low_hp")

    def test_an_unreachable_stairs_tile_tries_both_planner_modes_then_gives_up(self):
        service, bridge = started()
        env = FakeEnv(raw(), steps=110, path=[], unguarded_path=[])
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "retreat_stairs_unreachable")
        self.assertEqual(env.plan_calls, [(20, 20, True), (20, 20, False)])
        self.assertFalse(service.active)
        self.assertEqual(bridge.calls, [True, False])
        self.assertEqual(service.attempts[-1]["outcome"], "failed")

    def test_the_monster_blind_fallback_path_is_used_when_the_guarded_one_fails(self):
        service, bridge = started()
        env = FakeEnv(raw(), steps=110, path=[], unguarded_path=[(11, 10, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        self.assertEqual(env.plan_calls, [(20, 20, True), (20, 20, False)])
        self.assertTrue(service.active)

    def test_arriving_one_floor_up_completes_without_revoking_authority(self):
        service, bridge = started(value=raw(depth=3), steps=100)
        env = FakeEnv(raw(depth=2, hp=60), steps=180)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "retreat_ascended")
        self.assertFalse(service.active)
        self.assertEqual(bridge.calls, [True])
        attempt = service.attempts[-1]
        self.assertEqual(attempt["outcome"], "ascended")
        self.assertEqual(attempt["reason"], "retreat_ascended")
        self.assertEqual(attempt["beat1"], 180)
        self.assertEqual(attempt["hp1"], 60)
        self.assertEqual(service._cooldown_until, 0)

    def test_an_unexpected_scene_or_death_fails_the_attempt(self):
        for value, reason in ((raw(depth=5), "retreat_unexpected_scene"),
                              (raw(is_set_level=True), "retreat_unexpected_scene"),
                              (raw(dead=True), "retreat_died"),
                              (raw(game_over=True), "retreat_died"),
                              (raw(hp=0), "retreat_died")):
            with self.subTest(reason=reason):
                service, bridge = started()
                env = FakeEnv(value, steps=150)
                self.assertEqual(service.command(env, bridge), ("complete",))
                self.assertEqual(service.reason, reason)
                self.assertEqual(bridge.calls, [True, False])

    def test_drinking_is_spaced_by_twenty_microsteps(self):
        service, bridge = started()
        low = raw(hp=30, max_hp=100, belt=[1, 1, 0, 0])
        env = FakeEnv(low, steps=110, path=[(11, 10, False)])
        self.assertEqual(service.command(env, bridge), ("drink",))
        self.assertEqual(env.plan_calls, [])
        env._steps = 129
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        env._steps = 130
        self.assertEqual(service.command(env, bridge), ("drink",))

    def test_no_drink_without_a_belt_potion_or_above_the_drink_threshold(self):
        service, bridge = started()
        env = FakeEnv(raw(hp=30, max_hp=100, belt=[]), steps=110,
                      path=[(11, 10, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        env._raw = raw(hp=41, max_hp=100, belt=[1, 1, 0, 0])
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        env._raw = raw(hp=40, max_hp=100, belt=[1, 1, 0, 0])
        self.assertEqual(service.command(env, bridge), ("drink",))

    def test_the_service_cap_ends_the_attempt(self):
        service, bridge = started(steps=100)
        env = FakeEnv(raw(), steps=100 + 899, path=[(11, 10, False)])
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        self.assertEqual(service.steps, 899)
        env._steps = 100 + 900
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "retreat_cap")
        self.assertEqual(service.attempts[-1]["outcome"], "failed")
        self.assertEqual(bridge.calls, [True, False])

    def test_clocks_expose_the_cap_and_the_absolute_deadline(self):
        service, _ = started(steps=250)
        self.assertEqual(service.service_microstep_cap, 900)
        self.assertEqual(service.command_microstep_deadline, 1150)

    def test_eight_consecutive_walk_rejections_end_the_attempt(self):
        service, bridge = started()
        env = FakeEnv(raw(), steps=110, path=[(11, 10, False)])
        for index in range(module.RETREAT_WALK_REJECTIONS):
            command = service.command(env, bridge)
            self.assertEqual(command, ("walk", 11, 10), index)
            service.receipt(command, {"accepted": False})
            env._steps += 1
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "retreat_walk_rejected")
        self.assertFalse(service.active)
        self.assertEqual(service.attempts[-1]["outcome"], "failed")
        self.assertEqual(bridge.calls, [True, False])

    def test_an_accepted_walk_clears_the_rejection_streak(self):
        service, bridge = started()
        env = FakeEnv(raw(), steps=110, path=[(11, 10, False)])
        for accepted in [False] * 7 + [True] + [False] * 7:
            command = service.command(env, bridge)
            self.assertEqual(command, ("walk", 11, 10))
            service.receipt(command, {"accepted": accepted})
        self.assertTrue(service.active)
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))

    def test_accepted_drinks_are_counted_on_the_service_and_the_attempt(self):
        service, _ = started()
        service.receipt(("drink",), {"accepted": True})
        service.receipt(("drink",), {"accepted": True})
        service.receipt(("drink",), {"accepted": False})
        self.assertEqual(service.drinks, 2)
        self.assertEqual(service.attempts[-1]["drinks"], 2)

    def test_receipts_for_other_commands_do_not_disturb_the_counters(self):
        service, _ = started()
        service.receipt(("wait",), {"accepted": False})
        service.receipt(("complete",), {"accepted": True})
        self.assertEqual(service.drinks, 0)
        self.assertEqual(service._walk_rejections, 0)

    def test_a_revoke_failure_after_death_is_recorded_not_raised(self):
        bridge = FakeBridge(revoke_error="engine is out of game")
        service, _ = started(bridge=bridge)
        env = FakeEnv(raw(dead=True), steps=150)
        self.assertEqual(service.command(env, bridge), ("complete",))
        self.assertEqual(service.reason, "retreat_died")
        self.assertEqual(service.attempts[-1]["revoke_error"], "engine is out of game")

    def test_no_bridge_or_planner_call_after_the_attempt_is_finished(self):
        service, bridge = started()
        env = FakeEnv(raw(stairs=None), steps=140)
        service.command(env, bridge)
        self.assertEqual(bridge.calls, [True, False])
        self.assertEqual(env.plan_calls, [])

    def test_phase_steps_charge_the_ascend_phase(self):
        service, bridge = started(steps=100)
        env = FakeEnv(raw(), steps=140, path=[(11, 10, False)])
        service.command(env, bridge)
        self.assertEqual(service.phase_steps, {"ascend": 40})
        self.assertEqual(service.steps, 40)


class RetreatTelemetryTests(unittest.TestCase):
    KEYS = ("protocol", "attempted", "active", "phase", "reason", "trigger",
            "attempts", "ascended", "failed")

    def test_idle_telemetry_reports_the_protocol_without_an_attempt(self):
        telemetry = RetreatService().telemetry()
        for key in self.KEYS:
            self.assertIn(key, telemetry)
        self.assertEqual(telemetry["protocol"], "retreat-v1")
        self.assertFalse(telemetry["attempted"])
        self.assertFalse(telemetry["active"])
        self.assertEqual(telemetry["phase"], "idle")
        self.assertIsNone(telemetry["reason"])
        self.assertIsNone(telemetry["trigger"])
        self.assertEqual(telemetry["attempts"], [])
        self.assertEqual((telemetry["ascended"], telemetry["failed"]), (0, 0))
        self.assertEqual(telemetry["policy"], RetreatPolicy().as_dict())

    def test_active_telemetry_names_the_trigger(self):
        service, _ = started(trigger="empty_belt")
        telemetry = service.telemetry()
        self.assertTrue(telemetry["attempted"])
        self.assertTrue(telemetry["active"])
        self.assertEqual(telemetry["phase"], "ascend")
        self.assertEqual(telemetry["trigger"], "empty_belt")
        self.assertEqual((telemetry["ascended"], telemetry["failed"]), (0, 0))

    def test_telemetry_tallies_one_ascent_and_one_failure(self):
        service, bridge = started(value=raw(depth=3), steps=100)
        service.command(FakeEnv(raw(depth=2), steps=180), bridge)
        service.active = False
        service.start(raw(depth=3), 900, bridge, "empty_belt")
        service.command(FakeEnv(raw(depth=3, stairs=None), steps=950), bridge)
        telemetry = service.telemetry()
        self.assertEqual(telemetry["ascended"], 1)
        self.assertEqual(telemetry["failed"], 1)
        self.assertEqual(telemetry["reason"], "retreat_stairs_missing")
        self.assertEqual(len(telemetry["attempts"]), 2)
        telemetry["attempts"][0]["outcome"] = "tampered"
        self.assertEqual(service.attempts[0]["outcome"], "ascended")


class LootTripLimitTests(unittest.TestCase):
    def test_two_trips_without_any_retreat(self):
        self.assertEqual(loot.SustainLootService.trip_limit({}), 2)
        self.assertEqual(loot.SustainLootService.trip_limit({"resource_state": {}}), 2)
        self.assertEqual(loot.SustainLootService.trip_limit(None), 2)

    def test_each_engine_counted_retreat_earns_one_extra_trip(self):
        self.assertEqual(loot.SustainLootService.trip_limit(
            {"resource_state": {"retreats_started": 2}}), 4)
        self.assertEqual(loot.SustainLootService.trip_limit(
            {"resource_state": {"retreats_started": 1}}), 3)
        self.assertEqual(loot.SustainLootService.trip_limit(
            {"resource_state": {"retreats_started": None}}), 2)

    def test_trip_limit_is_a_static_method(self):
        self.assertIsInstance(
            loot.SustainLootService.__dict__["trip_limit"], staticmethod)


@unittest.skipIf(worker_env is None,
                 f"worker_env import unavailable: {WORKER_IMPORT_ERROR!r}")
class WorkerConstructorBoundaryTests(unittest.TestCase):
    """The R18-A probe scope: the Worker training wrapper refuses retreat-v1."""

    class StubOptions:
        """Enough of OptionsEnv for the constructor; no engine, no reset."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.env = SimpleNamespace(
                observation_space=SimpleNamespace(shape=(295,)),
                max_steps=kwargs.get("max_steps", 3000))

    def _build(self, **overrides):
        arguments = dict(manager_npz=None, manager_heuristic="readiness-v1")
        arguments.update(overrides)
        with patch.object(worker_env, "OptionsEnv",
                          side_effect=lambda **kw: self.StubOptions(**kw)) as options:
            return worker_env.WorkerWindowEnv(**arguments), options

    def test_retreat_v1_without_the_coach_law_is_rejected_before_the_options_constructor(self):
        # R18-B (2026-09-07): the wrapper now forwards retreat-v1, but only under
        # l2-town-v1 + coach-v03; the default (protocol off) still rejects it.
        with patch.object(worker_env, "OptionsEnv") as options:
            with self.assertRaises(ValueError) as caught:
                worker_env.WorkerWindowEnv(manager_npz=None,
                                           manager_heuristic="readiness-v1",
                                           resource_retreat="retreat-v1")
        self.assertIn("retreat-v1 requires l2-town-v1 under coach-v03", str(caught.exception))
        options.assert_not_called()

    def test_retreat_v1_under_the_coach_law_is_forwarded(self):
        env, options = self._build(resource_protocol="l2-town-v1",
                                   resource_readiness_law="coach-v03",
                                   resource_retreat="retreat-v1")
        self.assertEqual(env.resource_retreat, "retreat-v1")
        options.assert_called_once()
        self.assertEqual(env.oe.kwargs.get("resource_retreat"), "retreat-v1")

    def test_unknown_retreat_values_are_rejected_too(self):
        for value in ("retreat-v2", "on", True):
            with self.subTest(value=value):
                with patch.object(worker_env, "OptionsEnv") as options:
                    with self.assertRaises(ValueError):
                        worker_env.WorkerWindowEnv(manager_npz=None,
                                                   manager_heuristic="readiness-v1",
                                                   resource_retreat=value)
                options.assert_not_called()

    def test_the_default_off_path_builds_and_leaves_kwargs_untouched(self):
        env, options = self._build()
        self.assertEqual(env.resource_retreat, "off")
        options.assert_called_once()
        self.assertNotIn("resource_retreat", env.oe.kwargs)

    def test_explicit_off_is_identical_to_the_default(self):
        env, _ = self._build(resource_retreat="off")
        self.assertEqual(env.resource_retreat, "off")
        self.assertNotIn("resource_retreat", env.oe.kwargs)


if __name__ == "__main__":
    unittest.main()
