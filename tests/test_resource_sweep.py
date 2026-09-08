"""R18-H (2026-09-07) sweep-v1 boundaries: the chest/barrel sweep on main L1.

Two independent suites in one file (the house pattern of
``test_resource_portal.py`` / ``test_resource_retreat.py``):

* ``Sweep*Tests`` -- pure Python doubles, no native engine. The modules under
  test are loaded through a private package alias whose ``__path__`` points at
  ``python/diablogym``, which keeps ``diablogym/__init__.py`` and therefore the
  .so out of the way, plus a stub ``bridge`` submodule.
* ``SweepNativeTests`` -- the real headless bridge, skipped unless the module
  carries the R18-H symbol ``configure_object_observation`` (i.e. the isolated
  H1 build). It freezes the DEFAULT-OFF observation dict (the house law this
  feature must not break) and then really opens a chest and smashes a barrel
  on a real main L1.
"""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest

PACKAGE = "_r18h_sweep_pure"
if PACKAGE not in sys.modules:
    package = ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "python/diablogym")]
    sys.modules[PACKAGE] = package
if PACKAGE + ".bridge" not in sys.modules:
    # Only ``_walk``'s door branch reaches for the bridge; every pure path
    # exercised below keeps ``door`` false.
    sys.modules[PACKAGE + ".bridge"] = ModuleType(PACKAGE + ".bridge")

module = importlib.import_module(PACKAGE + ".resource_sweep")
protocol = importlib.import_module(PACKAGE + ".resource_protocol")

SweepService = module.SweepService
BUDGET = module.SWEEP_MICROSTEP_BUDGET
MAX_TARGETS = module.SWEEP_MAX_TARGETS
ABORT_RADIUS = module.SWEEP_MONSTER_ABORT_RADIUS
FARM_TRIGGER = 3600


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


def obj(kind="chest", x=12, y=10, interactable=True, solid=True, visible=True):
    # ``visible`` is the R18-H review-round lighting tag (IsTileLit), the same
    # 口径 the floor-item channel publishes.
    return {"kind": kind, "x": x, "y": y, "visible": visible,
            "interactable": interactable, "solid": solid}


def monster(x=30, y=30, hp=10, kind=1, invalid=False):
    return {"x": x, "y": y, "hp": hp, "type": kind, "is_invalid": invalid}


def raw(*, depth=1, x=10, y=10, objects=None, monsters=(), floor_items=(),
        gold=100, dead=False, game_over=False, is_set_level=False,
        player_mode=0, hp=100):
    payload = {"player_x": x, "player_y": y, "dungeon_level": depth,
               "dead": dead, "game_over": game_over, "is_set_level": is_set_level,
               "player_mode": player_mode, "hp": hp, "max_hp": 100, "gold": gold,
               "monsters": list(monsters), "floor_items": list(floor_items)}
    if objects is not None:
        payload["objects"] = list(objects)
    return payload


def fire(service, payload, steps=0, *, cleared=True, farm_scene_steps=0,
         town_trip_active=False, loot_trip_slots_left=1):
    return service.trigger_reason(
        payload, steps, farm_scene_steps=farm_scene_steps, cleared=cleared,
        farm_trigger=FARM_TRIGGER, town_trip_active=town_trip_active,
        loot_trip_slots_left=loot_trip_slots_left)


# ------------------------------------------------------------------ validators
class SweepValidatorTests(unittest.TestCase):

    def test_the_protocol_names_are_exactly_off_and_sweep_v1(self):
        self.assertEqual(protocol.RESOURCE_SWEEP_PROTOCOLS, ("off", "sweep-v1"))
        self.assertIs(module.RESOURCE_SWEEP_PROTOCOLS, protocol.RESOURCE_SWEEP_PROTOCOLS)
        self.assertIs(module.validate_sweep_protocol, protocol.validate_sweep_protocol)
        self.assertIs(module.validate_native_sweep, protocol.validate_native_sweep)

    def test_an_unknown_name_is_rejected(self):
        for name in ("sweep", "sweep-v2", "on", True, None, 1):
            with self.subTest(name=name), self.assertRaises(ValueError):
                protocol.validate_sweep_protocol("l2-town-v1", "sustain-loot-v1", name)

    def test_sweep_v1_requires_l2_town_v1_with_sustain_loot_v1(self):
        self.assertEqual(
            protocol.validate_sweep_protocol("l2-town-v1", "sustain-loot-v1", "sweep-v1"),
            "sweep-v1")
        with self.assertRaises(ValueError):
            protocol.validate_sweep_protocol("off", "sustain-loot-v1", "sweep-v1")
        for policy in ("legacy-v1", "sustain-v3", "sustain-v6"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                protocol.validate_sweep_protocol("l2-town-v1", policy, "sweep-v1")

    def test_off_is_legal_under_every_combination(self):
        for base in ("off", "l2-town-v1"):
            for policy in ("legacy-v1", "sustain-loot-v1"):
                with self.subTest(base=base, policy=policy):
                    self.assertEqual(
                        protocol.validate_sweep_protocol(base, policy, "off"), "off")

    def test_the_native_identity_is_the_objects_key_and_fails_closed(self):
        protocol.validate_native_sweep({"objects": []}, "sweep-v1")
        protocol.validate_native_sweep({"objects": [obj()]}, "sweep-v1")
        with self.assertRaises(RuntimeError):
            # a bridge older than the review-round lighting tag
            protocol.validate_native_sweep(
                {"objects": [{"x": 1, "y": 1, "kind": "chest",
                              "interactable": True, "solid": True}]}, "sweep-v1")
        protocol.validate_native_sweep({}, "off")
        protocol.validate_native_sweep({"triggers": []}, "off")
        with self.assertRaises(RuntimeError):
            protocol.validate_native_sweep({}, "sweep-v1")
        with self.assertRaises(RuntimeError):
            protocol.validate_native_sweep({"objects": "nope"}, "sweep-v1")
        with self.assertRaises(RuntimeError):
            protocol.validate_native_sweep({"objects": []}, "off")

    def test_sweep_objects_never_invents_a_floor(self):
        self.assertEqual(module.sweep_objects({}), ())
        self.assertEqual(module.sweep_objects({"objects": None}), ())
        self.assertEqual(module.sweep_objects(None), ())
        self.assertEqual(len(module.sweep_objects({"objects": [obj(), obj()]})), 2)


# ------------------------------------------------------------------ the law
class SweepLawTests(unittest.TestCase):

    def test_the_trigger_fires_on_a_cleared_main_l1_floor(self):
        service = SweepService()
        self.assertEqual(fire(service, raw(objects=[obj()])), "cleared")

    def test_the_trigger_fires_at_the_farm_cap_without_a_clear(self):
        service = SweepService()
        self.assertIsNone(fire(service, raw(objects=[obj()]), cleared=False,
                               farm_scene_steps=FARM_TRIGGER - 1))
        self.assertEqual(fire(service, raw(objects=[obj()]), cleared=False,
                              farm_scene_steps=FARM_TRIGGER), "farm_cap")

    def test_the_sweep_never_runs_in_town_or_below_l1_or_on_a_set_level(self):
        service = SweepService()
        for depth in (0, 2, 3, 16):
            with self.subTest(depth=depth):
                self.assertIsNone(fire(service, raw(depth=depth, objects=[obj()])))
        self.assertIsNone(fire(service, raw(objects=[obj()], is_set_level=True)))

    def test_the_sweep_never_runs_during_a_town_trip(self):
        service = SweepService()
        self.assertIsNone(fire(service, raw(objects=[obj()]), town_trip_active=True))

    def test_a_dead_or_finished_or_busy_player_never_starts_a_sweep(self):
        service = SweepService()
        self.assertIsNone(fire(service, raw(objects=[obj()], dead=True)))
        self.assertIsNone(fire(service, raw(objects=[obj()], game_over=True)))
        self.assertIsNone(fire(service, raw(objects=[obj()], player_mode=6)))

    def test_a_monster_inside_the_abort_radius_blocks_the_trigger(self):
        service = SweepService()
        close = monster(x=10 + ABORT_RADIUS, y=10)
        far = monster(x=10 + ABORT_RADIUS + 1, y=10)
        self.assertIsNone(fire(service, raw(objects=[obj()], monsters=[close])))
        self.assertEqual(fire(service, raw(objects=[obj()], monsters=[far])), "cleared")

    def test_dead_and_invalid_monsters_are_not_monsters(self):
        service = SweepService()
        for ghost in (monster(x=10, y=10, hp=0), monster(x=10, y=10, kind=109),
                      monster(x=10, y=10, invalid=True)):
            with self.subTest(ghost=ghost):
                self.assertEqual(fire(service, raw(objects=[obj()], monsters=[ghost])),
                                 "cleared")

    def test_without_the_native_channel_the_sweep_never_fires(self):
        service = SweepService()
        self.assertIsNone(fire(service, raw()))
        self.assertIsNone(fire(service, raw(objects=[])))

    def test_only_chests_and_barrels_are_targets(self):
        service = SweepService()
        for kind in module.SWEEP_SKIPPED_KINDS:
            with self.subTest(kind=kind):
                self.assertIsNone(fire(SweepService(), raw(objects=[obj(kind=kind)])))
        for kind in module.SWEEP_TAKEN_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(fire(SweepService(), raw(objects=[obj(kind=kind)])),
                                 "cleared")
        self.assertIsNone(fire(service, raw(objects=[obj(interactable=False)])))

    def test_the_sweep_never_opens_once_the_loot_trip_limit_is_spent(self):
        """Review round 2026-09-07: with no trip left, ``maybe_start`` denies for
        the rest of the episode and nothing would ever collect the drops."""
        service = SweepService()
        self.assertEqual(fire(service, raw(objects=[obj()]), loot_trip_slots_left=1),
                         "cleared")
        self.assertIsNone(fire(service, raw(objects=[obj()]), loot_trip_slots_left=0))

    def test_an_object_never_seen_lit_is_not_a_candidate(self):
        """Partial observability, the floor-item 口径: an unlit chest on an
        unexplored side of the floor is not a legal target, and one the pair HAS
        seen stays a target after the light moves on."""
        service = SweepService()
        dark = raw(objects=[obj(x=40, y=40, visible=False)])
        self.assertIsNone(fire(service, dark))
        lit = raw(objects=[obj(x=40, y=40, visible=True)])
        self.assertEqual(fire(service, lit), "cleared")
        self.assertIsNone(fire(SweepService(), dark))
        # remembered: the same object unlit again is still a candidate
        self.assertEqual(fire(service, dark), "cleared")

    def test_observe_records_the_lighting_memory_without_a_window(self):
        service = SweepService()
        service.observe(raw(gold=140, objects=[obj(x=40, y=40, visible=True)]))
        self.assertEqual(service.gold_after, 140)
        self.assertIsNotNone(fire(service, raw(objects=[obj(x=40, y=40,
                                                           visible=False)])))

    def test_an_active_sweep_never_triggers_a_second_time(self):
        service = SweepService()
        payload = raw(objects=[obj()])
        service.start(payload, 0, "cleared")
        self.assertIsNone(fire(service, payload))
        with self.assertRaises(RuntimeError):
            service.start(payload, 0, "cleared")

    def test_start_refuses_any_floor_but_main_l1(self):
        for payload in (raw(depth=0, objects=[obj()]), raw(depth=2, objects=[obj()]),
                        raw(objects=[obj()], is_set_level=True)):
            with self.subTest(payload=payload), self.assertRaises(RuntimeError):
                SweepService().start(payload, 0, "cleared")

    def test_the_episode_budget_and_target_cap_close_the_law(self):
        service = SweepService()
        service.microsteps_used = BUDGET
        self.assertIsNone(fire(service, raw(objects=[obj()])))
        service.microsteps_used = 0
        service.targets_attempted = MAX_TARGETS
        self.assertIsNone(fire(service, raw(objects=[obj()])))

    def test_the_window_cooldown_keeps_the_manager_from_looping(self):
        service = SweepService()
        env = FakeEnv(raw(objects=[obj(x=10, y=10)]), steps=100)
        service.start(env._raw, 100, "cleared")
        service._end(env, "sweep_monster_near")
        self.assertIsNone(fire(service, env._raw, 100))
        self.assertEqual(fire(service, raw(objects=[obj(x=10, y=10)]),
                              100 + module.SWEEP_WINDOW_COOLDOWN), "cleared")


# ------------------------------------------------------------------ the hands
class SweepCommandTests(unittest.TestCase):

    def start(self, payload, *, path=(), steps=0):
        service = SweepService()
        service.start(payload, steps, "cleared")
        env = FakeEnv(payload, steps=steps, path=path)
        return service, env

    def test_an_adjacent_target_is_operated_not_walked_to(self):
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)])
        service, env = self.start(payload)
        self.assertEqual(service.command(env, None), ("open", 11, 10))
        self.assertEqual(service.open_commands, 1)
        self.assertEqual(env.plan_calls, [])

    def test_a_distant_target_is_walked_toward_with_the_monster_avoiding_bfs(self):
        # Reachable = the BFS lands ADJACENT to the object; (15, 10) is.
        payload = raw(x=10, y=10, objects=[obj(x=16, y=10)])
        service, env = self.start(
            payload, path=[(11, 10, False), (12, 10, False), (13, 10, False),
                           (14, 10, False), (15, 10, False)])
        self.assertEqual(service.command(env, None), ("walk", 11, 10))
        self.assertIn((16, 10, True), env.plan_calls)
        self.assertEqual(service.walk_commands, 1)

    def test_an_object_the_bfs_cannot_reach_is_never_a_target(self):
        # The planner stops five tiles short: not adjacent, so not reachable.
        payload = raw(x=10, y=10, objects=[obj(x=60, y=60)])
        service, env = self.start(payload, path=[(11, 10, False), (12, 10, False)])
        self.assertEqual(service.command(env, None), ("complete",))
        self.assertEqual(service.reason, "sweep_no_reachable_target")
        self.assertEqual(service.targets_attempted, 0)

    def test_a_target_that_becomes_unreachable_is_abandoned_without_closing_the_window(self):
        payload = raw(x=10, y=10, objects=[obj(x=16, y=10)])
        service, env = self.start(
            payload, path=[(11, 10, False), (12, 10, False), (13, 10, False),
                           (14, 10, False), (15, 10, False)])
        self.assertEqual(service.command(env, None), ("walk", 11, 10))
        self.assertEqual(service.targets_attempted, 1)
        env.path = env.unguarded_path = []          # the corridor closed behind us
        self.assertEqual(service.command(env, None), ("wait",))
        self.assertTrue(service.active)             # one object, not the sweep
        self.assertEqual([f["reason"] for f in service.failures],
                         ["sweep_target_unreachable"])
        # A chest that IS adjacent is still swept on the next beat: abandoning
        # one object never abandons the window.
        env._raw = raw(x=10, y=10, objects=[obj(x=16, y=10), obj(x=11, y=11)])
        self.assertEqual(service.command(env, None), ("open", 11, 11))
        self.assertEqual(service.targets_attempted, 2)

    def test_a_trapped_chest_a_sarcophagus_and_an_explosive_barrel_are_only_recorded(self):
        payload = raw(x=10, y=10, objects=[
            obj(kind="chest_trapped", x=11, y=10),
            obj(kind="sarcophagus", x=10, y=11),
            obj(kind="barrel_explosive", x=9, y=10),
            obj(kind="chest", x=11, y=11)])
        service, env = self.start(payload)
        self.assertEqual(service.command(env, None), ("open", 11, 11))
        self.assertEqual(service.targets_skipped_by_kind,
                         {"chest_trapped": 1, "sarcophagus": 1, "barrel_explosive": 1})

    def test_a_skipped_object_is_counted_once_not_once_per_beat(self):
        payload = raw(x=10, y=10, objects=[obj(kind="sarcophagus", x=11, y=10),
                                           obj(kind="chest", x=11, y=11)])
        service, env = self.start(payload)
        for _ in range(4):
            service.command(env, None)
        self.assertEqual(service.targets_skipped_by_kind, {"sarcophagus": 1})

    def test_an_opened_chest_is_booked_when_the_engine_flips_interactable(self):
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)])
        service, env = self.start(payload)
        self.assertEqual(service.command(env, None), ("open", 11, 10))
        service.receipt(("open", 11, 10), {"accepted": True})
        env._raw = raw(x=10, y=10, gold=100,
                       objects=[obj(x=11, y=10, interactable=False)],
                       floor_items=[{"x": 11, "y": 11, "seed_hi": 1, "seed_lo": 2,
                                     "create_info": 3, "base_id": 0}])
        env._steps = 4
        self.assertEqual(service.command(env, None), ("complete",))
        self.assertEqual(service.chests_opened, 1)
        self.assertEqual(service.barrels_smashed, 0)
        self.assertEqual(service.items_dropped_seen, 1)

    def test_a_smashed_barrel_is_booked_as_a_barrel(self):
        payload = raw(x=10, y=10, objects=[obj(kind="barrel", x=11, y=10)])
        service, env = self.start(payload)
        command = service.command(env, None)
        service.receipt(command, {"accepted": True})
        env._raw = raw(x=10, y=10, objects=[
            obj(kind="barrel", x=11, y=10, interactable=False, solid=False)])
        service.command(env, None)
        self.assertEqual((service.chests_opened, service.barrels_smashed), (0, 1))
        self.assertEqual((service.open_accepted, service.open_rejected), (1, 0))

    def test_a_rejected_open_books_nothing_when_something_else_breaks_it(self):
        """Review round 2026-09-07: ``act_controller_operate`` fails closed on
        ``MyPlayer->position.future``, so a mid-tile player is refused and
        nothing reaches the engine. The worker's own attack may break that
        barrel later; the sweep must not claim it."""
        payload = raw(x=10, y=10, objects=[obj(kind="barrel", x=11, y=10)])
        service, env = self.start(payload)
        command = service.command(env, None)
        self.assertEqual(command, ("open", 11, 10))
        service.receipt(command, {"accepted": False})
        self.assertEqual(service._operated_tiles, set())
        env._raw = raw(x=10, y=10, objects=[
            obj(kind="barrel", x=11, y=10, interactable=False, solid=False)],
            floor_items=[{"x": 11, "y": 10, "seed_hi": 9, "seed_lo": 9,
                          "create_info": 9, "base_id": 9}])
        service.command(env, None)
        self.assertEqual((service.chests_opened, service.barrels_smashed), (0, 0))
        self.assertEqual(service.items_dropped_seen, 0)
        self.assertEqual((service.open_accepted, service.open_rejected), (0, 1))

    def test_a_blocking_barrel_smashed_on_the_way_is_booked_too(self):
        # The planner's softwall channel opens a blocking barrel EN ROUTE. That
        # is still a barrel this sweep smashed and still loot for the collect.
        payload = raw(x=10, y=10,
                      objects=[obj(x=16, y=10), obj(kind="barrel", x=11, y=10)])
        service, env = self.start(
            payload, path=[(11, 10, True), (12, 10, False), (13, 10, False),
                           (14, 10, False), (15, 10, False)])
        # This barrel is already retired as a TARGET (the cap/budget retired it),
        # so the only way it can break is the navigation channel.
        service._done_targets.add(("barrel", 11, 10))
        bridge_stub = sys.modules[PACKAGE + ".bridge"]
        bridge_stub.probe_tile = lambda x, y: {"walkable": False}
        try:
            self.assertEqual(service.command(env, None), ("open", 11, 10))
            self.assertEqual(service.path_open_commands, 1)
            self.assertEqual(service.open_commands, 0)   # not the chosen target
            service.receipt(("open", 11, 10), {"accepted": True})
        finally:
            del bridge_stub.probe_tile
        env._raw = raw(x=10, y=10, objects=[
            obj(x=16, y=10),
            obj(kind="barrel", x=11, y=10, interactable=False, solid=False)])
        env.path = env.unguarded_path = []
        service.command(env, None)
        self.assertEqual(service.barrels_smashed, 1)
        self.assertEqual(service.chests_opened, 0)

    def test_a_door_opened_on_the_way_is_navigation_not_loot(self):
        service = SweepService()
        service.start(raw(x=10, y=10, objects=[obj(x=11, y=10)]), 0, "cleared")
        service._operated_tiles.add((11, 10))
        service._book_operated(raw(x=10, y=10, objects=[
            {"kind": "door", "x": 11, "y": 10, "interactable": False, "solid": False}]))
        self.assertEqual((service.chests_opened, service.barrels_smashed), (0, 0))

    def test_an_object_that_this_script_never_operated_is_never_booked(self):
        payload = raw(x=10, y=10, objects=[obj(x=16, y=10)])
        service, env = self.start(payload, path=[(11, 10, False)])
        service.command(env, None)                      # walk, no operate
        env._raw = raw(x=10, y=10, objects=[obj(x=16, y=10, interactable=False)])
        service.command(env, None)
        self.assertEqual((service.chests_opened, service.barrels_smashed), (0, 0))

    def test_items_already_lying_beside_the_object_are_not_our_drops(self):
        """Review round 2026-09-07: a monster that died beside the chest during
        FARM leaves its drop within radius 1. Only identities that appear AFTER
        the open was issued are this sweep's loot."""
        old = {"x": 11, "y": 11, "seed_hi": 7, "seed_lo": 7,
               "create_info": 7, "base_id": 7}
        new = {"x": 11, "y": 10, "seed_hi": 1, "seed_lo": 2,
               "create_info": 3, "base_id": 4}
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)], floor_items=[old])
        service, env = self.start(payload)
        command = service.command(env, None)
        service.receipt(command, {"accepted": True})
        env._raw = raw(x=10, y=10, objects=[obj(x=11, y=10, interactable=False)],
                       floor_items=[old, new])
        service.command(env, None)
        self.assertEqual(service.chests_opened, 1)
        self.assertEqual(service.items_dropped_seen, 1)

    def test_the_same_drop_identity_is_counted_once(self):
        drop = {"x": 11, "y": 10, "seed_hi": 1, "seed_lo": 2,
                "create_info": 3, "base_id": 5}
        service = SweepService()
        service._count_drops({"floor_items": [drop, dict(drop)]}, 11, 10)
        service._count_drops({"floor_items": [dict(drop)]}, 11, 10)
        self.assertEqual(service.items_dropped_seen, 1)

    def test_an_idle_monster_in_the_corridor_does_not_make_a_target_unreachable(self):
        """Review round 2026-09-07: ``_reachable_targets`` used the
        monster-avoiding BFS only, while ``_walk`` falls back to the plain one.
        An object the walker CAN reach must not be classified unreachable."""
        payload = raw(x=10, y=10, objects=[obj(x=16, y=10)])
        service, env = self.start(payload, path=[])
        env.unguarded_path = [(11, 10, False), (12, 10, False), (13, 10, False),
                              (14, 10, False), (15, 10, False)]
        self.assertEqual(service.command(env, None), ("walk", 11, 10))
        self.assertIn((16, 10, True), env.plan_calls)
        self.assertIn((16, 10, False), env.plan_calls)

    def test_an_empty_window_retires_no_candidate_and_is_bounded_by_a_counter(self):
        """Review round 2026-09-07: one ``sweep_no_reachable_target`` used to
        retire EVERY chest and barrel on the floor for the rest of the episode.
        Nothing is retired now; the cost is bounded by a counter instead."""
        payload = raw(x=10, y=10, objects=[obj(x=60, y=60), obj(x=61, y=61)])
        service = SweepService()
        steps = 0
        for window in range(module.SWEEP_EMPTY_WINDOWS):
            with self.subTest(window=window):
                self.assertEqual(fire(service, payload, steps), "cleared")
                service.start(payload, steps, "cleared")
                env = FakeEnv(payload, steps=steps, path=[])
                self.assertEqual(service.command(env, None), ("complete",))
                self.assertEqual(service.reason, "sweep_no_reachable_target")
                steps += module.SWEEP_WINDOW_COOLDOWN
        self.assertEqual(service._done_targets, set())
        self.assertEqual(service.empty_windows, module.SWEEP_EMPTY_WINDOWS)
        self.assertIsNone(fire(service, payload, steps))
        # ... and a window that DID reach a target clears the counter.
        service.empty_windows = module.SWEEP_EMPTY_WINDOWS - 1
        near = raw(x=10, y=10, objects=[obj(x=11, y=10)])
        service.start(near, steps, "cleared")
        env = FakeEnv(near, steps=steps)
        service.command(env, None)
        service._end(env, "sweep_monster_near")
        self.assertEqual(service.empty_windows, 0)

    def test_a_low_hp_player_is_handed_back_to_the_retreat_law(self):
        """Review round 2026-09-07: the sweep owned the body for up to the whole
        budget with no HP floor. It hands back at RetreatPolicy.hp_fraction."""
        floor = int(module.SWEEP_HP_ABORT_FRACTION * 100)
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)])
        service, env = self.start(payload)
        env._raw = raw(x=10, y=10, objects=[obj(x=11, y=10)], hp=floor)
        self.assertEqual(service.command(env, None), ("complete",))
        self.assertEqual(service.reason, "sweep_low_hp")
        self.assertIsNone(fire(SweepService(), raw(objects=[obj()], hp=floor)))
        self.assertEqual(fire(SweepService(), raw(objects=[obj()], hp=floor + 1)),
                         "cleared")

    def test_a_monster_inside_the_abort_radius_hands_control_back(self):
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)],
                      monsters=[monster(x=10 + ABORT_RADIUS, y=10)])
        service, env = self.start(payload)
        self.assertEqual(service.command(env, None), ("complete",))
        self.assertEqual(service.reason, "sweep_monster_near")
        self.assertFalse(service.active)

    def test_death_a_scene_change_and_the_budget_all_hand_control_back(self):
        cases = {"sweep_died": raw(x=10, y=10, objects=[obj(x=11, y=10)], dead=True),
                 "sweep_unexpected_scene": raw(x=10, y=10, depth=2,
                                               objects=[obj(x=11, y=10)])}
        for reason, payload in cases.items():
            with self.subTest(reason=reason):
                service = SweepService()
                service.start(raw(x=10, y=10, objects=[obj(x=11, y=10)]), 0, "cleared")
                env = FakeEnv(payload)
                self.assertEqual(service.command(env, None), ("complete",))
                self.assertEqual(service.reason, reason)
        service, env = self.start(raw(x=10, y=10, objects=[obj(x=11, y=10)]))
        service.microsteps_used = BUDGET
        self.assertEqual(service.command(env, None), ("complete",))
        self.assertEqual(service.reason, "sweep_budget")

    def test_repeated_rejected_walks_abandon_the_sweep(self):
        payload = raw(x=10, y=10, objects=[obj(x=16, y=10)])
        service, env = self.start(payload, path=[(11, 10, False)])
        for _ in range(module.SWEEP_WALK_REJECTIONS):
            service.receipt(("walk", 11, 10), {"accepted": False})
        self.assertEqual(service.command(env, None), ("complete",))
        self.assertEqual(service.reason, "sweep_walk_rejected")

    def test_an_accepted_command_clears_the_rejection_run(self):
        service = SweepService()
        for _ in range(module.SWEEP_WALK_REJECTIONS - 1):
            service.receipt(("walk", 1, 1), {"accepted": False})
        service.receipt(("walk", 1, 1), {"accepted": True})
        self.assertEqual(service._walk_rejections, 0)
        self.assertIsNone(service._give_up)

    def test_an_object_that_refuses_to_open_is_abandoned_after_the_attempt_cap(self):
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)])
        service, env = self.start(payload)
        for _ in range(module.SWEEP_OPEN_ATTEMPTS):
            self.assertEqual(service.command(env, None), ("open", 11, 10))
        self.assertEqual(service.command(env, None), ("wait",))
        self.assertEqual([f["reason"] for f in service.failures], ["sweep_open_exhausted"])

    def test_the_target_cap_closes_the_window(self):
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)])
        service, env = self.start(payload)
        service.targets_attempted = MAX_TARGETS
        self.assertEqual(service.command(env, None), ("complete",))
        self.assertEqual(service.reason, "sweep_target_cap")

    def test_the_window_cap_is_frozen_for_the_life_of_the_window(self):
        """Review round 2026-09-07 (blocking): the RESUPPLY loop recomputes
        ``env._resource_service_deadline = start_steps + service_microstep_cap``
        on EVERY iteration, so a cap that decayed with the wall clock walked the
        deadline backwards and expired at half the budget -- which cancelled
        env.py's 12-beat accepted-``open`` follow-through from the middle of
        every window on. The cap is a constant; ``budget_remaining`` is the
        separate decaying episode ledger."""
        service = SweepService()
        env = FakeEnv(raw(objects=[obj()]), steps=500)
        service.start(env._raw, 500, "cleared")
        service.command(env, None)                     # baselines the window
        self.assertEqual(service.service_microstep_cap, BUDGET)
        self.assertEqual(service.command_microstep_deadline, 500 + BUDGET)
        for used in (10, BUDGET // 2, BUDGET - 1):
            with self.subTest(used=used):
                service.microsteps_used = used
                self.assertEqual(service.service_microstep_cap, BUDGET)
                deadline = service.start_steps + service.service_microstep_cap
                wall = service.start_steps + used
                self.assertGreater(deadline, wall)
        self.assertEqual(service.budget_remaining, 1)

    def test_a_second_window_freezes_the_cap_at_the_budget_that_is_left(self):
        service = SweepService()
        first = FakeEnv(raw(objects=[obj()]), steps=0)
        service.start(first._raw, 0, "cleared")
        service.command(first, None)
        service.microsteps_used = 300
        service._end(first, "sweep_monster_near")
        second = FakeEnv(raw(objects=[obj()]), steps=1000)
        service.start(second._raw, 1000, "cleared")
        second._steps = 1000
        service.command(second, None)
        self.assertEqual(service.service_microstep_cap, BUDGET - 300)
        self.assertEqual(service.command_microstep_deadline, 1000 + BUDGET - 300)

    def test_the_first_command_rebaselines_a_window_opened_mid_worker_window(self):
        """``action_masks()`` also runs inside the frozen worker's dual
        observation build, so a window can open several beats before the
        RESUPPLY loop asks for a command. Those beats belong to the worker, not
        to the sweep budget."""
        service = SweepService()
        payload = raw(objects=[obj(x=11, y=10)])
        service.start(payload, 1000, "cleared")        # trigger beat
        env = FakeEnv(payload, steps=1400)             # 400 worker beats later
        service.command(env, None)
        self.assertEqual(service.microsteps_used, 0)
        self.assertEqual(service.start_steps, 1400)
        self.assertEqual(service.telemetry()["windows"][-1]["beat0"], 1400)
        self.assertEqual(service.command_microstep_deadline, 1400 + BUDGET)

    def test_record_steps_charges_the_episode_budget(self):
        service = SweepService()
        service.start(raw(objects=[obj()]), 100, "cleared")
        service.record_steps(140)
        self.assertEqual(service.microsteps_used, 40)
        self.assertEqual(service.steps, 40)
        self.assertEqual(service.phase_steps, {"sweep": 40})

    def test_the_sweep_is_not_settle_exempt(self):
        # Only the portal's queued spell needed the settle fence lifted.
        self.assertFalse(getattr(SweepService(), "settle_exempt", False))


# ------------------------------------------------------------------ telemetry
class SweepTelemetryTests(unittest.TestCase):

    def test_a_fresh_service_reports_an_untouched_json_serialisable_ledger(self):
        telemetry = SweepService().telemetry()
        json.dumps(telemetry)
        self.assertEqual(telemetry["protocol"], "sweep-v1")
        for key in ("chests_opened", "barrels_smashed", "targets_skipped_by_kind",
                    "sweep_microsteps", "gold_before", "gold_after",
                    "items_dropped_seen"):
            self.assertIn(key, telemetry)
        self.assertEqual(telemetry["chests_opened"], 0)
        self.assertEqual(telemetry["barrels_smashed"], 0)
        self.assertEqual(telemetry["targets_skipped_by_kind"], {})
        self.assertIsNone(telemetry["gold_before"])
        self.assertIsNone(telemetry["gold_delta"])
        self.assertFalse(telemetry["attempted"])

    def test_the_gold_pair_spans_the_sweep_and_the_collect_that_follows(self):
        service = SweepService()
        payload = raw(x=10, y=10, gold=100, objects=[obj(x=11, y=10)])
        service.start(payload, 0, "cleared")
        env = FakeEnv(payload)
        service.command(env, None)
        env._raw = raw(x=10, y=10, gold=100,
                       objects=[obj(x=11, y=10, interactable=False)])
        service.command(env, None)
        self.assertFalse(service.active)
        service.observe_gold(raw(gold=460))     # the loot trip banked the drops
        telemetry = service.telemetry()
        self.assertEqual(telemetry["gold_before"], 100)
        self.assertEqual(telemetry["gold_after"], 460)
        self.assertEqual(telemetry["gold_delta"], 360)
        self.assertEqual(telemetry["gold_peak_after_first_sweep"], 460)
        self.assertEqual(telemetry["gold_peak_delta"], 360)

    def test_the_peak_survives_the_town_trip_spending_the_collected_gold(self):
        service = SweepService()
        payload = raw(x=10, y=10, gold=100, objects=[obj(x=11, y=10)])
        service.start(payload, 0, "cleared")
        service.observe_gold(raw(gold=460))     # collected
        service.observe_gold(raw(gold=20))      # then spent at the smith
        telemetry = service.telemetry()
        self.assertEqual(telemetry["gold_after"], 20)
        self.assertEqual(telemetry["gold_delta"], -80)
        self.assertEqual(telemetry["gold_peak_after_first_sweep"], 460)
        self.assertEqual(telemetry["gold_peak_delta"], 360)

    def test_a_second_window_keeps_the_first_windows_gold_baseline(self):
        service = SweepService()
        first = raw(x=10, y=10, gold=100, objects=[obj(x=11, y=10)])
        service.start(first, 0, "cleared")
        service._end(FakeEnv(first, steps=10), "sweep_monster_near")
        second = raw(x=10, y=10, gold=250, objects=[obj(x=11, y=10)])
        service.start(second, 200, "cleared")
        self.assertEqual(service.gold_before, 100)
        self.assertEqual(len(service.telemetry()["windows"]), 2)

    def test_the_window_ledger_records_the_outcome_and_the_reason(self):
        payload = raw(x=10, y=10, objects=[obj(x=11, y=10)])
        service = SweepService()
        service.start(payload, 0, "cleared")
        service._end(FakeEnv(payload, steps=30), "sweep_monster_near")
        window = service.telemetry()["windows"][-1]
        self.assertEqual(window["outcome"], "handed_back")
        self.assertEqual(window["reason"], "sweep_monster_near")
        self.assertEqual(window["beat1"], 30)
        json.dumps(service.telemetry())

    def test_the_constants_the_probe_reports_are_the_module_constants(self):
        telemetry = SweepService().telemetry()
        self.assertEqual(telemetry["microstep_budget"], BUDGET)
        self.assertEqual(telemetry["max_targets"], MAX_TARGETS)
        self.assertEqual(telemetry["monster_abort_radius"], ABORT_RADIUS)
        self.assertEqual(telemetry["hp_abort_fraction"], module.SWEEP_HP_ABORT_FRACTION)
        for key in ("open_accepted", "open_rejected", "empty_windows",
                    "objects_seen_lit", "window_microstep_cap"):
            self.assertIn(key, telemetry)


# ------------------------------------------------------------------ wiring
def _load_wiring():
    root = Path(__file__).resolve().parents[1]
    if str(root / "python") not in sys.path:
        sys.path.insert(0, str(root / "python"))
    try:
        from diablogym import DiabloGymEnv, OptionsEnv, bridge, nav  # noqa: F401
        from diablogym.completion_clock import COMPLETION_L2_V1
        from diablogym.resource_sweep import SweepService as NativeSweepService
    except Exception:  # pragma: no cover - no built bridge in this tree
        return None
    if not hasattr(bridge, "configure_object_observation"):
        return None
    return (DiabloGymEnv, OptionsEnv, bridge, nav, NativeSweepService,
            COMPLETION_L2_V1)


_NATIVE = _load_wiring()
LOOT_ARM = dict(resource_protocol="l2-town-v1", resource_purchase_mode="full",
                resource_service_policy="sustain-loot-v1",
                resource_readiness_law="coach-v03",
                worker_time_protocol="completion-l2-v1",
                max_steps=(6000 if _NATIVE is None else _NATIVE[5].actor_denominator))


@unittest.skipIf(_NATIVE is None, "requires the isolated R18-H sweep bridge")
class SweepWiringTests(unittest.TestCase):
    """OptionsEnv/DiabloGymEnv wiring, including the default-off kwarg freeze."""

    @classmethod
    def setUpClass(cls):
        (cls.DiabloGymEnv, cls.OptionsEnv, cls.bridge, cls.nav,
         cls.SweepService, cls.recipe) = _NATIVE

    def test_the_default_is_off_and_passes_no_new_env_kwarg(self):
        seen = {}
        real_init = self.DiabloGymEnv.__init__

        def spy(inner, **kwargs):
            seen.update(kwargs)
            return real_init(inner, **kwargs)

        self.DiabloGymEnv.__init__ = spy
        try:
            env = self.OptionsEnv(**LOOT_ARM)
        finally:
            self.DiabloGymEnv.__init__ = real_init
        try:
            self.assertEqual(env.resource_sweep, "off")
            self.assertNotIn("resource_sweep", seen)
            self.assertIsNone(getattr(env, "sweep_service", "missing"))
        finally:
            env.close()

    def test_sweep_v1_passes_exactly_one_new_env_kwarg(self):
        seen = {}
        real_init = self.DiabloGymEnv.__init__

        def spy(inner, **kwargs):
            seen.update(kwargs)
            return real_init(inner, **kwargs)

        self.DiabloGymEnv.__init__ = spy
        try:
            env = self.OptionsEnv(resource_sweep="sweep-v1", **LOOT_ARM)
        finally:
            self.DiabloGymEnv.__init__ = real_init
        try:
            self.assertEqual(seen.get("resource_sweep"), "sweep-v1")
            self.assertEqual(env.resource_sweep, "sweep-v1")
            self.assertIsNotNone(env.sweep_service)
        finally:
            env.close()

    def test_sweep_v1_is_refused_without_the_loot_economy(self):
        arm = dict(LOOT_ARM, resource_service_policy="sustain-v6")
        arm.pop("worker_time_protocol")
        with self.assertRaises(ValueError):
            self.OptionsEnv(resource_sweep="sweep-v1", **arm)


# ------------------------------------------------------------------ native
# R18-H review round (2026-09-07): nothing is excluded from the
# default-off observation freeze.
_FREEZE_UNSTABLE_KEYS = ()


@unittest.skipIf(_NATIVE is None, "requires the isolated R18-H sweep bridge")
class SweepNativeTests(unittest.TestCase):
    """The real headless engine. The default-off observation freeze is the
    house law this feature must not break; the smoke then opens a real chest
    and smashes a real barrel on a real main L1."""

    CHEST_SEED = 2133009
    BARREL_SEED = 2133016

    @classmethod
    def setUpClass(cls):
        (cls.DiabloGymEnv, cls.OptionsEnv, cls.bridge, cls.nav,
         cls.SweepService, cls.recipe) = _NATIVE
        cls.owner = cls.DiabloGymEnv()

    @classmethod
    def tearDownClass(cls):
        cls.bridge.end_game()
        cls.bridge.configure_object_observation(False)
        cls.bridge.configure_resource_protocol(False)
        cls.owner.close()

    def tearDown(self):
        self.bridge.end_game()
        self.bridge.configure_object_observation(False)
        self.bridge.configure_resource_protocol(False)

    def reset(self, seed, *, objects, dungeon=True):
        bridge = self.bridge
        bridge.end_game()
        bridge.configure_object_observation(objects)
        bridge.configure_resource_protocol(
            True, ordinary_armor_scope=True, preserve_equipment_readiness=True,
            loot_economy=True, readiness_advisory=True)
        raw = bridge.reset(seed)
        if dungeon:
            raw = self.nav.descend_to_dungeon(bridge)
            bridge.act_wait()
            raw = bridge.step(1)
            bridge.probe_invincible(True)
        return raw

    # ---- the default-off freeze (house law) ----
    def test_off_leaves_the_observation_dict_byte_identical(self):
        off = self.reset(self.CHEST_SEED, objects=False)
        self.assertNotIn("objects", off)
        on = self.reset(self.CHEST_SEED, objects=True)
        self.assertIn("objects", on)
        self.assertEqual(set(on) - set(off), {"objects"})
        self.assertEqual(set(off) - set(on), set())
        # Review round 2026-09-07: the original test skipped "missiles" with no
        # recorded reason. Re-measured on this build, the two resets produce an
        # identical missile list, so nothing is excluded any more and EVERY key
        # of the frozen observation dict is compared.
        self.assertEqual(_FREEZE_UNSTABLE_KEYS, ())
        for key in set(off):
            if key in _FREEZE_UNSTABLE_KEYS:  # pragma: no cover - empty by law
                continue
            with self.subTest(key=key):
                self.assertEqual(off[key], on[key])

    def test_the_channel_may_not_flip_inside_an_episode(self):
        self.reset(self.CHEST_SEED, objects=True, dungeon=False)
        with self.assertRaises(RuntimeError):
            self.bridge.configure_object_observation(False)
        self.bridge.configure_object_observation(True)  # a no-op re-assert is legal

    def test_the_channel_reports_only_the_loot_classes_with_sane_fields(self):
        raw = self.reset(self.CHEST_SEED, objects=True)
        self.assertTrue(raw["objects"])
        kinds = set()
        for entry in raw["objects"]:
            self.assertEqual(set(entry),
                             {"x", "y", "kind", "visible", "interactable", "solid"})
            self.assertIsInstance(entry["interactable"], bool)
            self.assertIsInstance(entry["visible"], bool)
            self.assertIsInstance(entry["solid"], bool)
            self.assertTrue(0 <= int(entry["x"]) < 112)
            self.assertTrue(0 <= int(entry["y"]) < 112)
            kinds.add(entry["kind"])
        self.assertTrue(kinds <= set(module.SWEEP_TAKEN_KINDS)
                        | set(module.SWEEP_SKIPPED_KINDS), kinds)
        self.assertIn("chest", kinds)
        self.assertIn("barrel", kinds)

    # ---- the real hands ----
    def _sweep_env(self, seed):
        env = self.DiabloGymEnv(
            resource_protocol="l2-town-v1", resource_ordinary_armor_scope=True,
            resource_preserve_equipment_readiness=True, resource_loot_economy=True,
            resource_readiness_law="coach-v03", resource_sweep="sweep-v1")
        env.reset(seed=seed)
        self.nav.descend_to_dungeon(self.bridge)
        self.bridge.act_wait()
        env._raw = self.bridge.step(1)
        self.bridge.probe_invincible(True)
        env._resource_service_deadline = env.max_steps
        return env

    def _walk_to_and_operate(self, env, kind):
        """Walk to the nearest reachable object of ``kind`` and operate it."""
        service = self.SweepService()
        # This test is about the HANDS. The lit-object law is covered by the
        # pure tests and by test_the_lit_gate_really_narrows_the_pool below;
        # here we hand the service the whole floor so it can pick any object.
        service._seen_objects.update(
            service._key(o) for o in env._raw["objects"])
        candidates = [row for row in service._reachable_targets(env._raw, env)
                      if row[2]["kind"] == kind]
        self.assertTrue(candidates, f"no reachable {kind} on this seed")
        target = candidates[0][2]
        tx, ty = int(target["x"]), int(target["y"])
        for _ in range(400):
            px, py = env._raw["player_x"], env._raw["player_y"]
            if max(abs(tx - px), abs(ty - py)) <= 1:
                break
            command = self.SweepService._walk(env, tx, ty)
            self.assertIsNotNone(command)
            env._raw, micro, _ = env._execute_resource_command(command)
            env._steps += micro
        px, py = env._raw["player_x"], env._raw["player_y"]
        self.assertLessEqual(max(abs(tx - px), abs(ty - py)), 1)
        live = [o for o in env._raw["objects"] if (o["x"], o["y"]) == (tx, ty)][0]
        self.assertTrue(live["interactable"])
        for _ in range(module.SWEEP_OPEN_ATTEMPTS):
            env._raw, micro, receipt = env._execute_resource_command(("open", tx, ty))
            env._steps += micro
            self.assertTrue(receipt.get("accepted"), receipt)
            live = [o for o in env._raw["objects"] if (o["x"], o["y"]) == (tx, ty)][0]
            if not live["interactable"]:
                break
        return tx, ty, live

    def test_a_real_chest_is_opened_and_stops_being_interactable(self):
        env = self._sweep_env(self.CHEST_SEED)
        try:
            tx, ty, live = self._walk_to_and_operate(env, "chest")
            self.assertFalse(live["interactable"],
                             f"chest at {(tx, ty)} stayed interactable")
        finally:
            self.bridge.end_game()
            env.close()

    def test_a_real_barrel_is_smashed_and_stops_being_solid(self):
        env = self._sweep_env(self.BARREL_SEED)
        try:
            tx, ty, live = self._walk_to_and_operate(env, "barrel")
            self.assertFalse(live["interactable"],
                             f"barrel at {(tx, ty)} stayed interactable")
            # BreakBarrel clears _oSolidFlag; a mere "cursor moved" would not.
            self.assertFalse(live["solid"])
        finally:
            self.bridge.end_game()
            env.close()

    def test_the_lit_gate_really_narrows_the_pool(self):
        """The channel is level-wide; the service is not. At reset only the
        objects in the player's light are candidates."""
        raw_now = self.reset(self.CHEST_SEED, objects=True)
        service = self.SweepService()
        service._observe_objects(raw_now)
        loot = [o for o in raw_now["objects"]
                if o["kind"] in module.SWEEP_TAKEN_KINDS and o["interactable"]]
        self.assertTrue(loot, "no loot objects on this seed")
        candidates = list(service._candidates(raw_now))
        self.assertLess(len(candidates), len(loot),
                        "the lit gate excluded nothing at all")
        for entry in candidates:
            self.assertTrue(entry["visible"])

    def test_the_scripted_sweep_drives_itself_and_books_real_objects(self):
        env = self._sweep_env(self.CHEST_SEED)
        service = self.SweepService()
        try:
            service._seen_objects.update(
                service._key(o) for o in env._raw["objects"])
            service.start(env._raw, env._steps, "cleared")
            for _ in range(900):
                if not service.active:
                    break
                command = service.command(env, self.bridge)
                if command[0] == "complete":
                    break
                env._raw, micro, receipt = env._execute_resource_command(command)
                env._steps += micro
                service.receipt(command, receipt)
            telemetry = service.telemetry()
            json.dumps(telemetry)
            self.assertGreaterEqual(
                telemetry["chests_opened"] + telemetry["barrels_smashed"], 1,
                telemetry)
            self.assertTrue(telemetry["targets_skipped_by_kind"], telemetry)
            self.assertLessEqual(telemetry["sweep_microsteps"], BUDGET)
            self.assertFalse(service.active)
        finally:
            self.bridge.end_game()
            env.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
