"""R18-D aggro cap + R18-E engagement selection: pure Python, no native engine.

Nothing here resets the engine, plans a native path or constructs a real
DiabloGymEnv.  ``aggro_cap`` and ``engagement`` are leaf modules, so they are
imported directly; the two env-side surfaces under test (the trailing
``_ControllerMonster`` fields and ``_engage_candidate_for_action9``) are
exercised against hand-built rows and a ``SimpleNamespace`` snapshot, in the
house style of ``test_env_v4_semantics.py`` (which already calls the unbound
``DiabloGymEnv._canonical_engage_candidate``).
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from types import SimpleNamespace
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym.aggro_cap import (  # noqa: E402
    AGGRO_CAP_PROTOCOLS,
    AggroCapPolicy,
    cap_fires,
    validate_aggro_cap,
    visible_alive_monsters_within,
)
from diablogym.engagement import (  # noqa: E402
    ENGAGEMENT_PROTOCOLS,
    RANGED_AI,
    THREAT_BY_AI,
    choose_engage_candidate,
    engage_sort_key,
    threat_weight,
    validate_engagement_priority,
)
from diablogym.env import DiabloGymEnv, _ControllerMonster  # noqa: E402


PLAYER_X, PLAYER_Y = 10, 10

_MISSING = object()

# Exactly the constructor keywords that existed before R18-E added
# ai/monster_level/max_damage.  Kept as a literal so the test fails loudly if a
# later refactor makes one of the new fields mandatory again.
LEGACY_MONSTER_KEYWORDS = {
    "monster_id": 1,
    "generation_key": (1, 0x1111, 0x2222),
    "monster_type": 0,
    "x": 11,
    "y": 10,
    "future_x": 11,
    "future_y": 10,
    "hp": 100,
    "max_hp": 100,
    "ledger_low": 0,
    "ledger_max": 1,
    "blocked": False,
    "visible": True,
    "native_reachable": True,
    "locally_engageable": True,
    "dynamic_quantities": (),
    "combat_flags": 0,
}


def row(monster_id, *, hp=100, ai=0, max_damage=0, monster_level=0,
        future_x=None, future_y=None, blocked=False, x=None, y=None):
    """One controller candidate row; only the selector-relevant knobs vary."""
    future_x = PLAYER_X + 1 if future_x is None else future_x
    future_y = PLAYER_Y if future_y is None else future_y
    return _ControllerMonster(
        monster_id=int(monster_id),
        generation_key=(int(monster_id), 0x1000 + int(monster_id), 0x2000),
        monster_type=0,
        x=future_x if x is None else x,
        y=future_y if y is None else y,
        future_x=int(future_x),
        future_y=int(future_y),
        hp=int(hp),
        max_hp=100,
        ledger_low=0,
        ledger_max=1,
        blocked=bool(blocked),
        visible=True,
        native_reachable=True,
        locally_engageable=True,
        dynamic_quantities=(),
        combat_flags=0,
        ai=int(ai),
        monster_level=int(monster_level),
        max_damage=int(max_damage),
    )


def snapshot(*candidates, player_x=PLAYER_X, player_y=PLAYER_Y):
    """The three snapshot attributes the two selectors actually read."""
    return SimpleNamespace(
        candidates=tuple(candidates),
        player_future_x=player_x,
        player_future_y=player_y,
    )


def fake_env(priority, *, dungeon_level=2, is_set_level=False, raw=_MISSING):
    """A stand-in ``self`` carrying only what ``_engage_candidate_for_action9``
    touches: the flag, ``_raw`` (the v1 main-L2+ scope gate), the two counters
    and the (static) canonical selector.  Defaults sit inside the scope."""
    if raw is _MISSING:
        raw = {"dungeon_level": dungeon_level, "is_set_level": is_set_level}
    return SimpleNamespace(
        engagement_priority=priority,
        _raw=raw,
        _engagement_decisions=0,
        _engagement_reordered=0,
        _canonical_engage_candidate=DiabloGymEnv._canonical_engage_candidate,
    )


def monster_raw(**updates):
    """One raw wire monster dict as ``visible_alive_monsters_within`` reads it."""
    result = {
        "hp": 50,
        "type": 0,
        "x": PLAYER_X,
        "y": PLAYER_Y,
        "visible": True,
        "is_invalid": False,
    }
    result.update(updates)
    return result


def raw(monsters=(), *, player_x=PLAYER_X, player_y=PLAYER_Y):
    return {"player_x": player_x, "player_y": player_y,
            "monsters": list(monsters)}


class ValidatorTests(unittest.TestCase):

    def test_aggro_cap_protocol_tuple(self):
        self.assertEqual(AGGRO_CAP_PROTOCOLS, ("off", "hold-v1"))

    def test_engagement_protocol_tuple(self):
        self.assertEqual(ENGAGEMENT_PROTOCOLS, ("off", "threat-v1"))

    def test_validate_aggro_cap_accepts_and_returns(self):
        for value in AGGRO_CAP_PROTOCOLS:
            self.assertEqual(validate_aggro_cap(value), value)

    def test_validate_aggro_cap_rejects(self):
        for value in ("on", "hold", "HOLD-V1", "", None, 1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_aggro_cap(value)

    def test_validate_engagement_priority_accepts_and_returns(self):
        for value in ENGAGEMENT_PROTOCOLS:
            self.assertEqual(validate_engagement_priority(value), value)

    def test_validate_engagement_priority_rejects(self):
        for value in ("threat", "threat-v2", "THREAT-V1", "", None, 0):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_engagement_priority(value)

    def test_rejection_messages_name_the_protocol_set(self):
        with self.assertRaises(ValueError) as caught:
            validate_aggro_cap("on")
        self.assertIn("hold-v1", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            validate_engagement_priority("threat")
        self.assertIn("threat-v1", str(caught.exception))

    def test_env_constructor_rejects_unknown_protocols_before_any_engine_work(self):
        # The kwargs are validated by the same leaf functions, so an unknown
        # value must be a ValueError and never reach the bridge.
        with self.assertRaises(ValueError):
            validate_aggro_cap("hold-v2")
        with self.assertRaises(ValueError):
            validate_engagement_priority("off-v1")


class AggroCapPolicyTests(unittest.TestCase):

    def test_chairman_defaults(self):
        policy = AggroCapPolicy()
        self.assertEqual(policy.count, 5)
        self.assertEqual(policy.radius, 6)

    def test_as_dict(self):
        self.assertEqual(AggroCapPolicy().as_dict(), {"count": 5, "radius": 6})
        self.assertEqual(
            AggroCapPolicy(count=3, radius=4).as_dict(),
            {"count": 3, "radius": 4},
        )

    def test_frozen(self):
        policy = AggroCapPolicy()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            policy.count = 9

    def test_rejects_non_positive_and_non_int(self):
        for kwargs in ({"count": 0}, {"count": -1}, {"count": 5.0},
                       {"count": True}, {"count": "5"},
                       {"radius": 0}, {"radius": -3}, {"radius": 6.0},
                       {"radius": False}, {"radius": None}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    AggroCapPolicy(**kwargs)

    def test_accepts_the_chairman_lower_bound(self):
        self.assertEqual(AggroCapPolicy(count=3).count, 3)
        self.assertEqual(AggroCapPolicy(count=1, radius=1).as_dict(),
                         {"count": 1, "radius": 1})


class VisibleAliveMonstersWithinTests(unittest.TestCase):

    def test_counts_plain_neighbours(self):
        state = raw([monster_raw(x=11, y=10), monster_raw(x=10, y=12)])
        self.assertEqual(visible_alive_monsters_within(state, 6), 2)

    def test_ignores_dead(self):
        state = raw([monster_raw(hp=0), monster_raw(hp=-5), monster_raw(hp=1)])
        self.assertEqual(visible_alive_monsters_within(state, 6), 1)

    def test_ignores_golem_type_109(self):
        state = raw([monster_raw(type=109), monster_raw(type=108)])
        self.assertEqual(visible_alive_monsters_within(state, 6), 1)

    def test_ignores_invalid_rows(self):
        state = raw([monster_raw(is_invalid=True), monster_raw()])
        self.assertEqual(visible_alive_monsters_within(state, 6), 1)

    def test_ignores_invisible_rows(self):
        state = raw([monster_raw(visible=False), monster_raw(visible=True)])
        self.assertEqual(visible_alive_monsters_within(state, 6), 1)

    def test_missing_visible_key_counts_as_visible(self):
        row_without_flag = monster_raw()
        row_without_flag.pop("visible")
        self.assertEqual(
            visible_alive_monsters_within(raw([row_without_flag]), 6), 1)

    def test_ignores_rows_missing_coordinates(self):
        no_x = monster_raw()
        no_x.pop("x")
        no_y = monster_raw()
        no_y.pop("y")
        state = raw([no_x, no_y, monster_raw()])
        self.assertEqual(visible_alive_monsters_within(state, 6), 1)

    def test_chebyshev_radius_boundary(self):
        inside = monster_raw(x=PLAYER_X + 6, y=PLAYER_Y + 6)   # Chebyshev 6
        outside = monster_raw(x=PLAYER_X + 7, y=PLAYER_Y)      # Chebyshev 7
        self.assertEqual(visible_alive_monsters_within(raw([inside]), 6), 1)
        self.assertEqual(visible_alive_monsters_within(raw([outside]), 6), 0)
        # Euclidean would have excluded the diagonal one (8.49 tiles away).
        self.assertEqual(
            visible_alive_monsters_within(raw([inside, outside]), 7), 2)

    def test_negative_offsets_use_absolute_distance(self):
        state = raw([monster_raw(x=PLAYER_X - 6, y=PLAYER_Y - 6),
                     monster_raw(x=PLAYER_X - 7, y=PLAYER_Y)])
        self.assertEqual(visible_alive_monsters_within(state, 6), 1)

    def test_empty_and_missing_monster_list(self):
        self.assertEqual(visible_alive_monsters_within(raw([]), 6), 0)
        self.assertEqual(
            visible_alive_monsters_within(
                {"player_x": PLAYER_X, "player_y": PLAYER_Y}, 6),
            0,
        )

    def test_crowd_of_mixed_rows(self):
        crowd = [monster_raw(x=PLAYER_X + offset) for offset in range(1, 8)]
        crowd.append(monster_raw(type=109, x=PLAYER_X))
        crowd.append(monster_raw(hp=0, x=PLAYER_X))
        # offsets 1..6 are inside radius 6; offset 7, the golem and the corpse
        # are not.
        self.assertEqual(visible_alive_monsters_within(raw(crowd), 6), 6)


class CapFiresTests(unittest.TestCase):

    def setUp(self):
        self.policy = AggroCapPolicy()
        self.crowd = raw([monster_raw(x=PLAYER_X + offset) for offset in range(1, 7)])

    def test_fires_when_crowded_and_action9_available(self):
        self.assertEqual(visible_alive_monsters_within(self.crowd, 6), 6)
        self.assertIs(cap_fires(self.crowd, self.policy, True), True)

    def test_never_fires_when_mask9_false_even_if_crowded(self):
        self.assertIs(cap_fires(self.crowd, self.policy, False), False)
        self.assertIs(cap_fires(self.crowd, self.policy, 0), False)
        self.assertIs(cap_fires(self.crowd, self.policy, None), False)

    def test_boundary_is_greater_or_equal_count(self):
        five = raw([monster_raw(x=PLAYER_X + offset) for offset in range(1, 6)])
        four = raw([monster_raw(x=PLAYER_X + offset) for offset in range(1, 5)])
        self.assertIs(cap_fires(five, self.policy, True), True)
        self.assertIs(cap_fires(four, self.policy, True), False)

    def test_respects_the_policy_radius(self):
        far = raw([monster_raw(x=PLAYER_X + 8) for _ in range(6)])
        self.assertIs(cap_fires(far, AggroCapPolicy(radius=6), True), False)
        self.assertIs(cap_fires(far, AggroCapPolicy(radius=9), True), True)

    def test_returns_a_bool(self):
        self.assertIsInstance(cap_fires(self.crowd, self.policy, True), bool)


class ThreatWeightTests(unittest.TestCase):

    def test_known_ai_weights(self):
        self.assertEqual(threat_weight(0), 1.0)     # Zombie baseline
        self.assertEqual(threat_weight(8), 0.8)     # Fallen: below baseline
        self.assertEqual(threat_weight(13), 3.0)    # Butcher
        self.assertEqual(threat_weight(3), 3.0)     # SkeletonRanged

    def test_unknown_ai_degrades_to_one(self):
        for ai in (18, 26, 36, 99, -1, 1000):
            with self.subTest(ai=ai):
                self.assertNotIn(ai, THREAT_BY_AI)
                self.assertEqual(threat_weight(ai), 1.0)

    def test_accepts_non_int_ai_values(self):
        self.assertEqual(threat_weight(13.0), 3.0)
        self.assertEqual(threat_weight(True), THREAT_BY_AI[1])

    def test_ranged_table_and_threat_table_are_consistent(self):
        # Every ranged id that also carries a weight must be at least baseline;
        # the ranged bit is the primary key, the weight only the tie-breaker.
        for ai in sorted(RANGED_AI & set(THREAT_BY_AI)):
            with self.subTest(ai=ai):
                self.assertGreaterEqual(THREAT_BY_AI[ai], 1.0)


class EngageSortKeyTests(unittest.TestCase):

    def key(self, candidate):
        return engage_sort_key(candidate, PLAYER_X, PLAYER_Y)

    def assert_wins(self, winner, loser):
        self.assertLess(self.key(winner), self.key(loser))
        self.assertIs(min((loser, winner), key=self.key), winner)
        self.assertIs(min((winner, loser), key=self.key), winner)

    def test_ranged_first_beats_a_heavier_melee(self):
        # ai 36 is ranged but carries no weight entry (1.0); ai 13 (Butcher) is
        # melee at 3.0 and also wins damage, hp, distance and id.  The ranged
        # bit still decides.
        ranged = row(9, ai=36, hp=100, max_damage=0, future_x=PLAYER_X + 6)
        melee = row(1, ai=13, hp=1, max_damage=60, future_x=PLAYER_X + 1)
        self.assertIn(36, RANGED_AI)
        self.assertNotIn(13, RANGED_AI)
        self.assertGreater(threat_weight(13), threat_weight(36))
        self.assert_wins(ranged, melee)
        self.assertEqual(self.key(ranged)[0], 0)
        self.assertEqual(self.key(melee)[0], 1)

    def test_skeleton_archer_beats_equal_weight_melee(self):
        archer = row(7, ai=3)               # SkeletonRanged, weight 3.0
        butcher = row(2, ai=13)             # Butcher, weight 3.0, lower id
        self.assert_wins(archer, butcher)

    def test_heavier_threat_weight_wins_among_melee(self):
        butcher = row(8, ai=13)             # 3.0, worse id
        zombie = row(1, ai=0)               # 1.0
        self.assert_wins(butcher, zombie)
        self.assertEqual(self.key(butcher)[1], -3.0)

    def test_heavier_max_damage_breaks_the_threat_tie(self):
        hard = row(8, ai=0, max_damage=30)
        soft = row(1, ai=0, max_damage=5)
        self.assert_wins(hard, soft)

    def test_lower_hp_breaks_the_damage_tie(self):
        nearly_dead = row(8, ai=0, max_damage=10, hp=3)
        healthy = row(1, ai=0, max_damage=10, hp=90)
        self.assert_wins(nearly_dead, healthy)

    def test_nearer_breaks_the_hp_tie(self):
        near = row(8, future_x=PLAYER_X + 1)
        far = row(1, future_x=PLAYER_X + 5)
        self.assert_wins(near, far)

    def test_lower_id_is_the_final_tie_break(self):
        first = row(1)
        second = row(2)
        self.assert_wins(first, second)
        self.assertEqual(self.key(first)[:-1], self.key(second)[:-1])

    def test_distance_is_chebyshev_over_future_tiles(self):
        # Standing far away now, arriving adjacent: the key must read the
        # future tile the macro and the a9 approach reward both use.
        arriving = row(1, x=PLAYER_X + 9, y=PLAYER_Y + 9,
                       future_x=PLAYER_X + 2, future_y=PLAYER_Y + 2)
        self.assertEqual(self.key(arriving)[4], 2)
        leaving = row(2, x=PLAYER_X, y=PLAYER_Y,
                      future_x=PLAYER_X - 4, future_y=PLAYER_Y + 1)
        self.assertEqual(self.key(leaving)[4], 4)

    def test_key_shape_and_defaults(self):
        key = self.key(row(3))
        self.assertEqual(len(key), 6)
        self.assertEqual(key, (1, -1.0, 0, 100, 1, 3))

    def test_missing_optional_attributes_fall_back(self):
        # engage_sort_key reads ai/max_damage through getattr defaults, so a row
        # built with the pre-R18-E keyword set still ranks.
        legacy = _ControllerMonster(**LEGACY_MONSTER_KEYWORDS)
        self.assertEqual(self.key(legacy), (1, -1.0, 0, 100, 1, 1))

    def test_key_is_a_total_order_over_a_mixed_pack(self):
        pack = [
            row(4, ai=0, hp=90),
            row(3, ai=13, hp=90),
            row(2, ai=3, hp=90),
            row(1, ai=8, hp=90),
        ]
        ordered = [c.monster_id for c in sorted(pack, key=self.key)]
        # archer (ranged) -> butcher (3.0) -> zombie (1.0) -> fallen (0.8)
        self.assertEqual(ordered, [2, 3, 4, 1])


class ChooseEngageCandidateTests(unittest.TestCase):

    def test_returns_none_on_empty(self):
        self.assertIsNone(choose_engage_candidate(snapshot()))

    def test_single_candidate(self):
        only = row(5)
        self.assertIs(choose_engage_candidate(snapshot(only)), only)

    def test_prefers_unblocked_even_when_the_blocked_row_ranks_higher(self):
        blocked_archer = row(1, ai=3, blocked=True)
        unblocked_zombie = row(9, ai=0, hp=90, blocked=False)
        chosen = choose_engage_candidate(
            snapshot(blocked_archer, unblocked_zombie))
        self.assertIs(chosen, unblocked_zombie)

    def test_falls_back_to_the_ranked_blocked_pool_when_all_blocked(self):
        weak = row(1, ai=0, blocked=True)
        archer = row(9, ai=3, blocked=True)
        chosen = choose_engage_candidate(snapshot(weak, archer))
        self.assertIs(chosen, archer)

    def test_picks_the_best_of_several_unblocked(self):
        candidates = (
            row(1, ai=0, hp=100),
            row(2, ai=13, hp=100),
            row(3, ai=7, hp=100),      # GoatRanged
            row(4, ai=3, hp=40),       # SkeletonRanged, lower hp
        )
        chosen = choose_engage_candidate(snapshot(*candidates))
        self.assertEqual(chosen.monster_id, 4)

    def test_uses_the_player_future_tile_from_the_snapshot(self):
        west = row(1, future_x=PLAYER_X - 3)
        east = row(2, future_x=PLAYER_X + 1)
        self.assertIs(
            choose_engage_candidate(snapshot(west, east)), east)
        self.assertIs(
            choose_engage_candidate(
                snapshot(west, east, player_x=PLAYER_X - 4)),
            west,
        )

    def test_accepts_any_iterable_candidate_container(self):
        rows = [row(2, ai=3), row(1, ai=0)]
        chosen = choose_engage_candidate(
            SimpleNamespace(candidates=rows,
                            player_future_x=PLAYER_X,
                            player_future_y=PLAYER_Y))
        self.assertEqual(chosen.monster_id, 2)


class EngageCandidateForAction9Tests(unittest.TestCase):
    """``DiabloGymEnv._engage_candidate_for_action9`` called unbound on a fake
    ``self`` -- the real method, none of the constructor."""

    def call(self, env, snap):
        return DiabloGymEnv._engage_candidate_for_action9(env, snap)

    def mixed_snapshot(self):
        # Wire-canonical order (nearest, then id) puts the zombie first; the
        # threat ranking wants the archer.
        zombie = row(1, ai=0, hp=100, future_x=PLAYER_X + 1)
        archer = row(5, ai=3, hp=40, future_x=PLAYER_X + 4)
        return snapshot(zombie, archer), zombie, archer

    def test_off_returns_the_canonical_first_unblocked_row(self):
        snap, zombie, _archer = self.mixed_snapshot()
        env = fake_env("off")
        chosen = self.call(env, snap)
        self.assertIs(chosen, zombie)
        self.assertIs(chosen, DiabloGymEnv._canonical_engage_candidate(snap))

    def test_off_never_touches_the_counters(self):
        snap, _zombie, _archer = self.mixed_snapshot()
        env = fake_env("off")
        for _ in range(3):
            self.call(env, snap)
        self.assertEqual(env._engagement_decisions, 0)
        self.assertEqual(env._engagement_reordered, 0)

    def test_off_skips_blocked_rows_like_the_canonical_selector(self):
        blocked = row(1, blocked=True)
        live = row(2, blocked=False)
        env = fake_env("off")
        self.assertIs(self.call(env, snapshot(blocked, live)), live)

    def test_off_falls_back_to_the_first_row_when_all_blocked(self):
        first = row(4, ai=0, blocked=True)
        stronger = row(5, ai=3, blocked=True)
        env = fake_env("off")
        self.assertIs(self.call(env, snapshot(first, stronger)), first)

    def test_threat_v1_returns_the_ranked_row_and_counts_a_reorder(self):
        snap, _zombie, archer = self.mixed_snapshot()
        env = fake_env("threat-v1")
        chosen = self.call(env, snap)
        self.assertIs(chosen, archer)
        self.assertEqual(env._engagement_decisions, 1)
        self.assertEqual(env._engagement_reordered, 1)

    def test_threat_v1_counts_a_decision_without_a_reorder_when_ids_agree(self):
        archer = row(1, ai=3, future_x=PLAYER_X + 1)
        zombie = row(2, ai=0, future_x=PLAYER_X + 2)
        env = fake_env("threat-v1")
        chosen = self.call(env, snapshot(archer, zombie))
        self.assertIs(chosen, archer)
        self.assertEqual(env._engagement_decisions, 1)
        self.assertEqual(env._engagement_reordered, 0)

    def test_threat_v1_counters_accumulate_across_calls(self):
        reorder_snap, _zombie, archer = self.mixed_snapshot()
        agree_snap = snapshot(row(1, ai=3), row(2, ai=0, future_x=PLAYER_X + 2))
        env = fake_env("threat-v1")
        self.assertIs(self.call(env, reorder_snap), archer)
        self.call(env, agree_snap)
        self.assertIs(self.call(env, reorder_snap), archer)
        self.assertEqual(env._engagement_decisions, 3)
        self.assertEqual(env._engagement_reordered, 2)

    def test_threat_v1_on_an_empty_snapshot_returns_none(self):
        env = fake_env("threat-v1")
        self.assertIsNone(self.call(env, snapshot()))
        self.assertEqual(env._engagement_decisions, 1)
        self.assertEqual(env._engagement_reordered, 0)

    def test_off_on_an_empty_snapshot_returns_none(self):
        env = fake_env("off")
        self.assertIsNone(self.call(env, snapshot()))

    def test_missing_flag_attribute_defaults_to_off(self):
        snap, zombie, _archer = self.mixed_snapshot()
        env = SimpleNamespace(
            _raw={"dungeon_level": 4},
            _canonical_engage_candidate=DiabloGymEnv._canonical_engage_candidate)
        self.assertIs(self.call(env, snap), zombie)

    def test_v1_scope_gate_holds_the_canonical_target_on_level_1(self):
        snap, zombie, _archer = self.mixed_snapshot()
        env = fake_env("threat-v1", dungeon_level=1)
        self.assertIs(self.call(env, snap), zombie)
        self.assertEqual(env._engagement_decisions, 0)
        self.assertEqual(env._engagement_reordered, 0)

    def test_v1_scope_gate_holds_the_canonical_target_on_set_levels(self):
        snap, zombie, _archer = self.mixed_snapshot()
        env = fake_env("threat-v1", dungeon_level=9, is_set_level=True)
        self.assertIs(self.call(env, snap), zombie)
        self.assertEqual(env._engagement_decisions, 0)

    def test_v1_scope_gate_opens_at_main_level_2(self):
        snap, _zombie, archer = self.mixed_snapshot()
        for level in (2, 3, 16):
            with self.subTest(dungeon_level=level):
                env = fake_env("threat-v1", dungeon_level=level)
                self.assertIs(self.call(env, snap), archer)
                self.assertEqual(env._engagement_decisions, 1)

    def test_absent_or_empty_raw_stays_gated(self):
        snap, zombie, _archer = self.mixed_snapshot()
        for raw_value in ({}, None, {"is_set_level": False}):
            with self.subTest(raw=raw_value):
                env = fake_env("threat-v1", raw=raw_value)
                self.assertIs(self.call(env, snap), zombie)
                self.assertEqual(env._engagement_decisions, 0)

    def test_off_ignores_the_scope_gate_entirely(self):
        snap, zombie, _archer = self.mixed_snapshot()
        for level in (1, 2, 16):
            with self.subTest(dungeon_level=level):
                env = fake_env("off", dungeon_level=level)
                self.assertIs(self.call(env, snap), zombie)
                self.assertEqual(env._engagement_decisions, 0)

    def test_threat_v1_agrees_with_the_standalone_selector(self):
        snap, _zombie, _archer = self.mixed_snapshot()
        env = fake_env("threat-v1")
        self.assertIs(self.call(env, snap), choose_engage_candidate(snap))

    def test_threat_v1_prefers_unblocked_and_still_counts_the_reorder(self):
        blocked_archer = row(1, ai=3, blocked=True)
        canonical = row(2, ai=0, blocked=False, future_x=PLAYER_X + 1)
        stronger = row(3, ai=13, blocked=False, future_x=PLAYER_X + 3)
        snap = snapshot(blocked_archer, canonical, stronger)
        self.assertIs(DiabloGymEnv._canonical_engage_candidate(snap), canonical)
        env = fake_env("threat-v1")
        self.assertIs(self.call(env, snap), stronger)
        self.assertEqual(env._engagement_reordered, 1)


class ControllerMonsterDefaultTests(unittest.TestCase):

    def test_legacy_keyword_set_still_constructs(self):
        legacy = _ControllerMonster(**LEGACY_MONSTER_KEYWORDS)
        self.assertEqual(legacy.ai, 0)
        self.assertEqual(legacy.monster_level, 0)
        self.assertEqual(legacy.max_damage, 0)
        self.assertEqual(legacy.monster_id, 1)

    def test_new_fields_are_trailing_and_defaulted(self):
        names = [f.name for f in dataclasses.fields(_ControllerMonster)]
        self.assertEqual(names[-3:], ["ai", "monster_level", "max_damage"])
        for field in dataclasses.fields(_ControllerMonster):
            with self.subTest(field=field.name):
                if field.name in {"ai", "monster_level", "max_damage"}:
                    self.assertEqual(field.default, 0)
                else:
                    self.assertIs(field.default, dataclasses.MISSING)

    def test_positional_construction_matches_the_legacy_order(self):
        positional = _ControllerMonster(
            *[LEGACY_MONSTER_KEYWORDS[f.name]
              for f in dataclasses.fields(_ControllerMonster)[:17]])
        self.assertEqual(positional, _ControllerMonster(**LEGACY_MONSTER_KEYWORDS))

    def test_row_stays_frozen_and_hashable_by_replace(self):
        base = _ControllerMonster(**LEGACY_MONSTER_KEYWORDS)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            base.ai = 3
        promoted = dataclasses.replace(base, ai=3, max_damage=12)
        self.assertEqual(promoted.ai, 3)
        self.assertEqual(promoted.max_damage, 12)
        self.assertEqual(base.ai, 0)


if __name__ == "__main__":
    unittest.main()
