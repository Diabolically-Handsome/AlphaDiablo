from __future__ import annotations

from dataclasses import replace
import pathlib
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import diablogym.env as env_module  # noqa: E402
from diablogym.env import (  # noqa: E402
    CONTROLLER_SNAPSHOT_BELT_DIM,
    CONTROLLER_SNAPSHOT_COMBAT_DIM,
    CONTROLLER_SNAPSHOT_EXACT_DIM,
    CONTROLLER_SNAPSHOT_HEAL_TARGET_DIM,
    CONTROLLER_SNAPSHOT_MAP_DIM,
    CONTROLLER_SNAPSHOT_MISSILE_DIM,
    CONTROLLER_SNAPSHOT_MISSILE_FIELDS,
    CONTROLLER_SNAPSHOT_MISSILE_LIMIT,
    CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_DIM,
    CONTROLLER_SNAPSHOT_MONSTER_FIELDS,
    CONTROLLER_SNAPSHOT_MONSTER_LIMIT,
    CONTROLLER_SNAPSHOT_MONSTER_DIM,
    CONTROLLER_SNAPSHOT_VECTOR_DIM,
    DiabloGymEnv,
    GEAR_COMBAT_UTILITY_REWARD_CAP,
    TERMINAL_DEATH_REWARD_SPEC,
    gear_upgrade_reward_component,
    terminal_death_reward_component,
)
from diablogym.options_env import (  # noqa: E402
    BeatOutcome,
    MANAGER_OBSERVATION_VIEW_LEGACY_V3,
    MANAGER_OBSERVATION_VIEW_RAW_V4,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    OptionsEnv,
)
from diablogym.worker_env import (  # noqa: E402
    legacy_worker_policy_observation_view,
)


def monster(mid: int, hp: int, *, max_hp: int = 100,
            visible: bool = True, reachable: bool = True,
            generation_seed: int | None = None) -> dict:
    if generation_seed is None:
        generation_seed = mid + 1
    result = {
        "id": mid,
        "type": 0,
        "x": 11 + mid,
        "y": 10,
        "future_x": 11 + mid,
        "future_y": 10,
        "hp": hp,
        "max_hp": max_hp,
        "visible": visible,
        "reachable": reachable,
        "rnd_item_seed_hi": (int(generation_seed) >> 16) & 0xFFFF,
        "rnd_item_seed_lo": int(generation_seed) & 0xFFFF,
    }
    result.update({
        field: 0
        for field in env_module.CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS
    })
    result.update({
        field: 0
        for field in env_module.CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS
    })
    result.update({
        "anim_ticks_per_frame": 1,
        "anim_num_frames": 1,
        "unique_type": 255,
        "hp_fixed_hi": ((int(hp) << 6) >> 16) & 0xFFFF,
        "hp_fixed_lo": (int(hp) << 6) & 0xFFFF,
        "max_hp_fixed_hi": ((int(max_hp) << 6) >> 16) & 0xFFFF,
        "max_hp_fixed_lo": (int(max_hp) << 6) & 0xFFFF,
        "combat_flags": 0,
    })
    return result


def missile(**updates) -> dict:
    result = {
        field: 0
        for field in (
            env_module.CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS
            + env_module.CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS
        )
    }
    result.update(
        visible=True,
        source_visible=True,
        start_visible=True,
        tile_dx=1,
        tile_dy=0,
        start_dx=2,
        start_dy=0,
        duration=10,
        source_id=0,
        source_type=0,
        hostile=True,
    )
    result.update(updates)
    return result


def raw(*, monsters=(), floor_items=(), hp=70, max_hp=70,
        belt_heals=0, belt_free_slots=8, missiles=()) -> dict:
    normalized_items = []
    gear_defaults = {
        field: 0
        for field in env_module.CONTROLLER_SNAPSHOT_GEAR_FIELDS
    }
    for index, source in enumerate(floor_items):
        item = dict(source)
        item.setdefault("active_id", index)
        item.setdefault("heal_kind", 1 if item.get("heal") else 0)
        item.setdefault("legacy_heal", bool(item.get("heal")))
        for field, value in gear_defaults.items():
            item.setdefault(field, value)
        item["active_id"] = int(item.get("active_id", index))
        item.setdefault("effect_flags", 0)
        item.setdefault("effect_dam_ac_flags", 0)
        item.setdefault("seed_hi", 0)
        item.setdefault("seed_lo", index + 1)
        item.setdefault("create_info", 0)
        item.setdefault("base_id", 0)
        normalized_items.append(item)
    heals = min(8, max(0, int(belt_heals)))
    free = min(8 - heals, max(0, int(belt_free_slots)))
    belt_slot_kinds = [2] * heals + [0] * free
    belt_slot_kinds += [1] * (8 - len(belt_slot_kinds))
    state = {
        "dungeon_level": 1,
        "engine_level": 1,
        "level_type": 1,
        "is_set_level": False,
        "set_level_id": 0,
        "xp": 0,
        "gold": 0,
        "char_level": 1,
        "armor_class": 0,
        "gear_combat_utility": 0,
        "monsters": list(monsters),
        "floor_items": normalized_items,
        "triggers": [],
        "progression_targets": [],
        "player_x": 10,
        "player_y": 10,
        "hp": hp,
        "max_hp": max_hp,
        "mana": 0,
        "max_mana": 0,
        "hp_fixed_hi": ((int(hp) << 6) >> 16) & 0xFFFF,
        "hp_fixed_lo": (int(hp) << 6) & 0xFFFF,
        "max_hp_fixed_hi": ((int(max_hp) << 6) >> 16) & 0xFFFF,
        "max_hp_fixed_lo": (int(max_hp) << 6) & 0xFFFF,
        "mana_fixed_hi": 0,
        "mana_fixed_lo": 0,
        "max_mana_fixed_hi": 0,
        "max_mana_fixed_lo": 0,
        "belt_heals": belt_heals,
        "legacy_belt_heals": belt_heals,
        "belt_free_slots": belt_free_slots,
        "belt_heal_kinds": (
            [1] * min(8, belt_heals)
            + [0] * (8 - min(8, belt_heals))
        ),
        "belt_slot_kinds": belt_slot_kinds,
        "betrayer_quest_active": 0,
        "betrayer_quest_stage": 0,
        "betrayer_portal_stage": 0,
        "monotonic_quest_turn_in_used": False,
        "player_mode": env_module.bridge.PM_STAND,
        "walkpath0": env_module.bridge.WALK_NONE,
        "future_x": 10,
        "future_y": 10,
        "dest_action": env_module.bridge.ACTION_NONE,
        "dead": False,
        "game_over": False,
        "victory": False,
        "monster_kill_total": 0,
        "missiles": [dict(entry) for entry in missiles],
    }
    state.update({
        field: 0
        for field in env_module.CONTROLLER_SNAPSHOT_COMBAT_FIELDS
    })
    state["block_enabled"] = False
    state["item_effect_flags"] = 0
    state["item_dam_ac_flags"] = 0
    state["equipped_items"] = [
        dict(
            {
                "present": False,
                "effect_flags": 0,
                "effect_dam_ac_flags": 0,
            },
            **gear_defaults,
        )
        for _ in range(env_module.CONTROLLER_SNAPSHOT_EQUIPPED_SLOTS)
    ]
    return state


def controller_map(*, walkable: int = 1) -> dict:
    cells = env_module.CONTROLLER_SNAPSHOT_CELLS
    return {
        "walkable": [walkable] * cells,
        "monster": [0] * cells,
        "door": [0] * cells,
        "closed_door": [0] * cells,
        "hazard": [0] * cells,
        "explosive_softwall": [0] * cells,
    }


def controller_fixture(
        state: dict,
        *,
        engage_blocked=(),
        explore_blocked=(),
        visited=((10, 10),),
        sticky=None,
        ledger=None,
        local_map=None):
    env = DiabloGymEnv.__new__(DiabloGymEnv)
    env._raw = state
    env._steps = 0
    env._visited = set(visited)
    env._explore_target = sticky
    env._explore_blocked_targets = set(explore_blocked)
    by_id = {
        int(entry["id"]): DiabloGymEnv._monster_generation_key(entry)
        for entry in state.get("monsters", ())
    }
    env._engage_blocked_keys = {
        (
            blocked
            if isinstance(blocked, tuple)
            else by_id.get(int(blocked), (int(blocked), 0, 0))
        )
        for blocked in engage_blocked
    }
    env._combat_hp_floor = (
        {
            (
                key
                if isinstance(key, tuple)
                else by_id.get(int(key), (int(key), 0, 0))
            ): value
            for key, value in ledger.items()
        }
        if ledger is not None
        else {
            DiabloGymEnv._monster_generation_key(entry): (
                int(entry["hp"]), max(1, int(entry["max_hp"])))
            for entry in state.get("monsters", ())
        }
    )
    env._ensure_active = lambda **_kwargs: None
    with mock.patch.object(
            env_module.bridge, "local_map",
            return_value=(local_map or controller_map())):
        env._controller_snapshot = env._capture_controller_snapshot(state)
    return env


def execution_step_fixture(
        state: dict,
) -> tuple[DiabloGymEnv, mock.Mock]:
    """Pure base-step shell for causal execution-receipt tests."""
    env = controller_fixture(state)
    env.action_space = env_module.gym.spaces.Discrete(15)
    env.max_steps = 100
    env.ticks_per_step = 4
    env.start_in_dungeon = False
    env._controller_snapshot_enabled = False
    env._episode_ended = False
    env._ep_kills = 0
    env._ep_start_xp = 0
    env._exploration_progress = 0
    env._softwalls_opened = 0
    env._settle_to_idle = lambda current, beats, **_kwargs: (
        current, beats)
    env._info = lambda _current: {}
    env._vectorize = lambda _current: np.zeros(295, dtype=np.float32)
    reward = mock.Mock(return_value=0.0)
    env._reward = reward
    return env, reward


class ProtocolV4MaskTests(unittest.TestCase):
    @staticmethod
    def env_with(state: dict) -> DiabloGymEnv:
        return controller_fixture(state)

    def test_policy_helpers_reject_omniscient_or_unreachable_entities(self):
        state = raw(
            monsters=[
                monster(1, 100, visible=False),
                monster(2, 100, reachable=False),
                monster(3, 100),
            ],
            floor_items=[
                {"x": 1, "y": 1, "heal": True, "gear": False,
                 "visible": False, "reachable": True},
                {"x": 2, "y": 2, "heal": True, "gear": False,
                 "visible": True, "reachable": False},
                {"x": 3, "y": 3, "heal": True, "gear": False,
                 "visible": True, "reachable": True},
            ],
        )
        self.assertEqual(
            [m["id"] for m in DiabloGymEnv._policy_monsters(state)], [3])
        self.assertEqual(
            [(it["x"], it["y"])
             for it in DiabloGymEnv._policy_floor_items(state, "heal")],
            [(3, 3)],
        )

    def test_masks_encode_exact_resource_and_target_preconditions(self):
        heal = {
            "x": 12, "y": 10, "heal": True, "gear": False,
            "visible": True, "reachable": True,
        }
        gear = {
            "x": 13, "y": 10, "heal": False, "gear": True,
            "visible": True, "reachable": True,
        }
        state = raw(
            monsters=[monster(1, 100)],
            floor_items=[heal, gear],
            hp=40,
            max_hp=70,
            belt_heals=2,
            belt_free_slots=1,
        )
        mask = self.env_with(state).action_masks()
        self.assertTrue(mask[9])
        self.assertTrue(mask[12])
        self.assertTrue(mask[13])
        self.assertTrue(mask[14])

        state["hp"] = state["max_hp"]
        self.assertFalse(self.env_with(state).action_masks()[12])
        state["hp"] = 40
        state["belt_free_slots"] = 0
        self.assertFalse(self.env_with(state).action_masks()[13])
        state["belt_free_slots"] = 1
        state["floor_items"][0]["reachable"] = False
        # The executable controller path, not the coarse native global hint,
        # is the legality source: a closed door can make native reachable
        # false while action13 can safely open it.  Seal the local map to make
        # the exact target genuinely unavailable.
        self.assertTrue(self.env_with(state).action_masks()[13])
        self.assertFalse(controller_fixture(
            state, local_map=controller_map(walkable=0),
        ).action_masks()[13])
        state["floor_items"][1]["visible"] = False
        self.assertFalse(self.env_with(state).action_masks()[14])
        state["monsters"][0]["visible"] = False
        self.assertFalse(self.env_with(state).action_masks()[9])

        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (15,))

    def test_public_masks_never_advertise_radius_thirteen_targets(self):
        outside_monster = monster(1, 100)
        outside_monster.update(
            x=23, y=10, future_x=23, future_y=10)
        state = raw(
            monsters=[outside_monster],
            floor_items=[
                {
                    "x": 23, "y": 10, "heal": True, "gear": False,
                    "visible": True, "reachable": True,
                },
                {
                    "x": 23, "y": 11, "heal": False, "gear": True,
                    "visible": True, "reachable": True,
                },
            ],
            belt_free_slots=8,
        )
        self.assertTrue(DiabloGymEnv._policy_monsters(state))
        self.assertTrue(DiabloGymEnv._policy_floor_items(state, "heal"))
        self.assertTrue(DiabloGymEnv._policy_floor_items(state, "gear"))

        mask, nearest = controller_fixture(
            state).controller_action_context()
        self.assertFalse(mask[9])
        self.assertFalse(mask[13])
        self.assertFalse(mask[14])
        self.assertIsNone(nearest)

    def test_belt_scalar_losslessly_exposes_free_slots_for_pickup_mask(self):
        heal = {
            "x": 12, "y": 10, "heal": True, "gear": False,
            "visible": True, "reachable": True,
        }
        full = raw(
            floor_items=[dict(heal)], belt_heals=2, belt_free_slots=0)
        open_slots = raw(
            floor_items=[dict(heal)], belt_heals=2, belt_free_slots=3)
        cells = (2 * env_module._MAP_RADIUS + 1) ** 2
        local_map = {
            "walkable": [1] * cells,
            "monster": [0] * cells,
            "door": [0] * cells,
        }
        with mock.patch.object(
                env_module.bridge, "local_map", return_value=local_map):
            full_obs = DiabloGymEnv._vectorize(full)
            open_obs = DiabloGymEnv._vectorize(open_slots)

        belt_index = 12 + env_module._K_MONSTERS * 4 + 2 * cells
        self.assertEqual(belt_index, 286)
        self.assertAlmostEqual(float(full_obs[belt_index]), 2 / 8)
        self.assertAlmostEqual(
            float(open_obs[belt_index]), 2 / 8 + 3 / 128)
        different = np.flatnonzero(full_obs != open_obs)
        self.assertTrue(np.array_equal(different, [belt_index]), different)

        # 两个整数域可从单 scalar 唯一恢复；因此 exact free-slot mask 不再
        # 是 critic 看不见的 raw-only 前置条件。
        encoded = float(open_obs[belt_index])
        decoded_heals = int(np.floor(encoded * 8 + 1e-6))
        decoded_free = int(round(
            (encoded - decoded_heals / 8.0) * 128))
        self.assertEqual((decoded_heals, decoded_free), (2, 3))
        codebook = {
            DiabloGymEnv._belt_observation_scalar(
                {"belt_heals": heals, "belt_free_slots": free})
            for heals in range(9)
            for free in range(9 - heals)
        }
        self.assertEqual(len(codebook), sum(range(1, 10)))
        self.assertFalse(self.env_with(full).action_masks()[13])
        self.assertTrue(self.env_with(open_slots).action_masks()[13])

    def test_engage_tie_rotates_away_from_persistently_failed_target(self):
        left = monster(47, 1)
        left.update(x=11, y=9, future_x=11, future_y=9)
        right = monster(96, 1)
        right.update(x=11, y=11, future_x=11, future_y=11)
        state = raw(monsters=[left, right])
        state.update(future_x=10, future_y=10)
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._engage_blocked_keys = set()
        left_key = env._monster_generation_key(left)

        first = env._select_engage_target(
            state, allow_blocked_cycle=True)
        self.assertEqual(first["id"], 47)
        env._engage_blocked_keys.add(left_key)
        second = env._select_engage_target(
            state, allow_blocked_cycle=True)
        self.assertEqual(second["id"], 96)
        self.assertIn(left_key, env._engage_blocked_keys)

        # 单目标时完成一整轮后允许重试，不能永久把 action9 变成空拍。
        state["monsters"] = [left]
        retried = env._select_engage_target(
            state, allow_blocked_cycle=True)
        self.assertEqual(retried["id"], 47)
        self.assertNotIn(left_key, env._engage_blocked_keys)

    def test_canonical_a9_window_summarizes_above_fixed_capacity(self):
        monsters = []
        for index in range(1, CONTROLLER_SNAPSHOT_MONSTER_LIMIT + 3):
            candidate = monster(index, 100)
            candidate.update(
                x=11,
                y=10,
                future_x=11,
                future_y=10,
            )
            monsters.append(candidate)
        state = raw(monsters=monsters)
        state.update(future_x=10, future_y=10)
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._steps = 0
        env._visited = {(10, 10)}
        env._explore_target = None
        env._explore_blocked_targets = set()
        env._combat_hp_floor = {
            env._monster_generation_key(entry): (
                entry["hp"], entry["max_hp"])
            for entry in monsters
        }
        env._engage_blocked_keys = set()
        with mock.patch.object(
                env_module.bridge, "local_map",
                return_value=controller_map()):
            snapshot = env._capture_controller_snapshot(state)
        self.assertEqual(
            len(snapshot.monsters), CONTROLLER_SNAPSHOT_MONSTER_LIMIT)
        self.assertEqual(
            len(snapshot.candidates), CONTROLLER_SNAPSHOT_MONSTER_LIMIT)
        self.assertAlmostEqual(
            snapshot.monster_overflow_quantities[1], 2 / 256)

    def test_engage_returns_control_at_first_emergency_drink_boundary(self):
        target = monster(7, 100)
        target.update(x=11, y=10, future_x=11, future_y=10)
        before = raw(
            monsters=[target], hp=60, max_hp=100,
            belt_heals=2, belt_free_slots=6)
        after = {
            **before,
            "monsters": [dict(target)],
            "hp": 49,
        }
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = before
        env.ticks_per_step = 4
        env._engage_blocked_keys = set()
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)

        with (
            mock.patch.object(
                env_module.bridge, "local_map",
                return_value=controller_map()),
            mock.patch.object(
                env_module.bridge, "act_controller_attack_monster",
                return_value=1, create=True) as attack,
            mock.patch.object(
                env_module.bridge, "step", return_value=after) as step,
        ):
            result, beats = env._macro_engage(max_beats=10)

        self.assertIs(result, after)
        self.assertEqual(beats, 1)
        attack.assert_called_once_with(7, 10, 10, 12)
        step.assert_called_once_with(ticks=4)

    def test_every_multibeat_navigation_macro_hands_off_at_reflex_boundary(self):
        def prepare(env, before):
            env._raw = before
            env.ticks_per_step = 4
            env._record_visit = lambda _position: False
            env._finish_macro = lambda final_raw, beats, _scene: (
                final_raw, beats)

        def injured(before):
            after = dict(before)
            after["hp"] = 34
            after["max_hp"] = 70
            after["belt_heals"] = 1
            return after

        progression = {
            "kind": "diablo_switch",
            "action": "operate",
            "x": 11,
            "y": 10,
            "goal_x": 11,
            "goal_y": 10,
            "exact": False,
        }
        before = raw(hp=36, max_hp=70, belt_heals=1)
        before["progression_targets"] = [dict(progression)]
        after = injured(before)
        after["progression_targets"] = [dict(progression)]
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        prepare(env, before)
        with (
            mock.patch.object(
                env_module.bridge, "act_controller_operate",
                return_value=1,
            ) as operate,
            mock.patch.object(
                env_module.bridge, "step", return_value=after) as step,
        ):
            result, beats = env._macro_progression(max_beats=12)
        self.assertIs(result, after)
        self.assertEqual(beats, 1)
        operate.assert_called_once_with(
            11, 10, 10, 10, env._DESCEND_RADIUS)
        step.assert_called_once_with(ticks=4)

        before = raw(hp=36, max_hp=70, belt_heals=1)
        before["triggers"] = [{
            "x": 15,
            "y": 10,
            "msg": env_module.bridge.WM_DIABNEXTLVL,
        }]
        after = injured(before)
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        prepare(env, before)
        env._plan_descend_path = lambda *_args, **_kwargs: [
            (11, 10, False), (12, 10, False)]
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=1,
            ) as walk,
            mock.patch.object(
                env_module.bridge, "step", return_value=after) as step,
        ):
            result, beats = env._macro_descend(max_beats=12)
        self.assertIs(result, after)
        self.assertEqual(beats, 1)
        walk.assert_called_once_with(
            11, 10, (), 10, 10, env._DESCEND_RADIUS)
        step.assert_called_once_with(ticks=4)

        before = raw(hp=36, max_hp=70, belt_heals=1)
        env = controller_fixture(before)
        prepare(env, before)
        env._plan_explore_step = lambda *_args, **_kwargs: (
            "frontier", 12, 10)
        env._plan_controller_path = lambda *_args, **_kwargs: [
            (11, 10, False), (12, 10, False)]
        after = injured(before)
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk") as walk,
            mock.patch.object(
                env_module.bridge, "step", return_value=after) as step,
        ):
            result, beats = env._macro_explore(
                max_beats=12,
                controller_snapshot=env._controller_snapshot,
            )
        self.assertIs(result, after)
        self.assertEqual(beats, 1)
        walk.assert_called_once()
        step.assert_called_once_with(ticks=4)

        before = raw(
            hp=36,
            max_hp=70,
            belt_heals=1,
            belt_free_slots=7,
            floor_items=[{
                "x": 11,
                "y": 10,
                "heal": True,
                "gear": False,
                "visible": True,
                "reachable": True,
            }],
        )
        env = controller_fixture(before)
        prepare(env, before)
        after = injured(before)
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=1,
            ) as walk,
            mock.patch.object(
                env_module.bridge, "act_pickup_at",
                return_value=1, create=True) as pickup,
            mock.patch.object(
                env_module.bridge, "step", return_value=after) as step,
        ):
            result, beats = env._macro_pickup(
                "heal",
                max_beats=12,
                controller_snapshot=env._controller_snapshot,
            )
        self.assertIs(result, after)
        self.assertEqual(beats, 1)
        walk.assert_called_once()
        pickup.assert_not_called()
        step.assert_called_once_with(ticks=4)

        no_belt = injured(raw(belt_heals=0, belt_free_slots=8))
        no_belt["belt_heals"] = 0
        self.assertFalse(env._reflex_eligible(no_belt))
        exactly_half = raw(
            hp=35, max_hp=70, belt_heals=1, belt_free_slots=7)
        self.assertFalse(env._reflex_eligible(exactly_half))

    def test_engage_generation_key_stops_same_slot_respawn_alias(self):
        original = monster(7, 100)
        original.update(x=11, y=10, future_x=11, future_y=10)
        respawned = dict(original)
        respawned["rnd_item_seed_lo"] += 1
        before = raw(monsters=[original])
        after = raw(monsters=[respawned])

        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = before
        env.ticks_per_step = 4
        env._engage_blocked_keys = set()
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)

        with (
            mock.patch.object(
                env_module.bridge, "local_map",
                return_value=controller_map()),
            mock.patch.object(
                env_module.bridge, "act_controller_attack_monster",
                return_value=1, create=True) as attack,
            mock.patch.object(
                env_module.bridge, "step", return_value=after) as step,
        ):
            result, beats = env._macro_engage(max_beats=10)

        self.assertIs(result, after)
        self.assertEqual(beats, 1)
        attack.assert_called_once_with(7, 10, 10, 12)
        step.assert_called_once_with(ticks=4)
        self.assertNotEqual(
            env._monster_generation_key(original),
            env._monster_generation_key(respawned),
        )

        # The drop seed is controller-private identity, never an actor feature;
        # stale blocked memory therefore must be generation-aware internally.
        old_key = env._monster_generation_key(original)
        old_fixture = controller_fixture(
            before, engage_blocked={old_key})
        new_fixture = controller_fixture(
            after, engage_blocked={old_key})
        self.assertTrue(old_fixture._controller_snapshot.monsters[0].blocked)
        self.assertFalse(new_fixture._controller_snapshot.monsters[0].blocked)
        np.testing.assert_array_equal(
            controller_fixture(before).controller_snapshot_vector(),
            controller_fixture(after).controller_snapshot_vector(),
        )

    def test_action12_audit_does_not_turn_native_rejection_into_execution(self):
        before = raw(
            hp=40, max_hp=100, belt_heals=2, belt_free_slots=6)
        idle = {
            **before,
            "future_x": before["player_x"],
            "future_y": before["player_y"],
            "player_mode": env_module.bridge.PM_STAND,
            "walkpath0": env_module.bridge.WALK_NONE,
            "dest_action": env_module.bridge.ACTION_NONE,
        }
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = before
        env._ensure_active = lambda **_kwargs: None
        env.action_space = env_module.gym.spaces.Discrete(15)
        env.max_steps = 100
        env._steps = 0
        env.ticks_per_step = 4
        env.start_in_dungeon = False
        env._visited = set()
        env._exploration_progress = 0
        env._explore_target = None
        env._explore_blocked_targets = set()
        env._engage_blocked_keys = set()
        env._ep_kills = 0
        env._episode_ended = False
        env._settle_to_idle = lambda state, beats, **_kwargs: (
            state, beats)
        env._reward = lambda *_args, **_kwargs: 0.0
        env._info = lambda _state: {}
        env._vectorize = lambda _state: np.zeros(295, dtype=np.float32)

        with (
            mock.patch.object(env_module.bridge, "act_wait", return_value=1),
            mock.patch.object(env_module.bridge, "act_drink", return_value=0),
            mock.patch.object(env_module.bridge, "step", return_value=idle),
            mock.patch.object(
                env_module.bridge, "local_map",
                return_value=controller_map()),
        ):
            _obs, _reward, terminated, truncated, info = env.step(12)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            info["action12_audit"],
            {
                "accepted": False,
                "accepted_belt_before": 0,
                "belt_before": 2,
                "consumed": False,
                "belt_after": 2,
            },
        )

    def test_native_execution_receipt_rejects_lossy_zero_or_one_coercions(self):
        for invalid in (
            -1,
            2,
            0.0,
            1.0,
            1.5,
            "1",
            None,
        ):
            with self.subTest(invalid=invalid):
                audit = {"attempts": 0, "accepts": 0}
                with self.assertRaisesRegex(RuntimeError, "必须是.*0/1"):
                    DiabloGymEnv._record_native_execution(
                        audit, invalid, "strict receipt")
                self.assertEqual(audit, {"attempts": 0, "accepts": 0})

        audit = {"attempts": 0, "accepts": 0}
        for accepted_type in (0, False, np.int64(0)):
            self.assertFalse(DiabloGymEnv._record_native_execution(
                audit, accepted_type, "strict receipt"))
        for accepted_type in (1, True, np.int64(1)):
            self.assertTrue(DiabloGymEnv._record_native_execution(
                audit, accepted_type, "strict receipt"))
        self.assertEqual(audit, {"attempts": 6, "accepts": 3})

    def test_rejected_direction_cannot_use_exogenous_motion_to_forge_execution(self):
        before = raw()
        after = raw()
        # This endpoint is deliberately indistinguishable from a successful
        # eastward key if one looks only at state deltas.  The native command
        # itself rejected, so the movement is external/previous-state evidence
        # and must not certify this request.
        after.update(
            player_x=11,
            player_y=10,
            future_x=11,
            future_y=10,
        )
        env, reward = execution_step_fixture(before)

        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk", return_value=0),
            mock.patch.object(
                env_module.bridge, "step", return_value=after),
        ):
            _obs, _r, terminated, truncated, info = env.step(3)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        receipt = info["action_effect_audit"]
        self.assertEqual(receipt["native_attempts"], 1)
        self.assertEqual(receipt["native_accepts"], 0)
        self.assertTrue(receipt["material_effect"])
        self.assertIn("move", receipt["effect_reasons"])
        self.assertFalse(receipt["request_executed"])
        self.assertTrue(receipt["stall_cost_applied"])
        reward.assert_called_once_with(
            before,
            after,
            requested_action=3,
            engage_target_generation_key=None,
            action_executed=False,
            action14_utility_delta=None,
        )

    def test_accepted_direction_without_landing_is_executed_and_not_stalled(self):
        state = raw()
        env, reward = execution_step_fixture(state)

        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk", return_value=1),
            mock.patch.object(
                env_module.bridge, "step", return_value=state),
        ):
            _obs, _r, terminated, truncated, info = env.step(3)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        receipt = info["action_effect_audit"]
        self.assertEqual(receipt["native_attempts"], 1)
        self.assertEqual(receipt["native_accepts"], 1)
        self.assertTrue(receipt["request_executed"])
        self.assertFalse(receipt["material_effect"])
        self.assertFalse(receipt["stall_cost_applied"])
        reward.assert_called_once_with(
            state,
            state,
            requested_action=3,
            engage_target_generation_key=None,
            action_executed=True,
            action14_utility_delta=None,
        )

        # The published stall ledger and the numeric reward must agree.  An
        # accepted edge interrupted before landing is not a wall-bump reject.
        reward_env = DiabloGymEnv.__new__(DiabloGymEnv)
        reward_env._combat_hp_floor = {}
        reward_env._reset_combat_ledger(state)
        reward_env.descend_ladder = True
        reward_env.death_ladder = True
        self.assertEqual(
            reward_env._reward(
                state,
                state,
                requested_action=3,
                action_executed=True,
            ),
            0.0,
        )

    def test_accepted_action9_miss_is_executed_and_not_stalled(self):
        state = raw()
        env, reward = execution_step_fixture(state)
        snapshot = mock.Mock()
        snapshot.candidates = ()
        env._capture_controller_snapshot = mock.Mock(
            return_value=snapshot)

        def accepted_miss(
                max_beats, *, controller_snapshot, execution_audit):
            self.assertEqual(max_beats, 10)
            self.assertIs(controller_snapshot, snapshot)
            DiabloGymEnv._record_native_execution(
                execution_audit, 1, "action9 accepted miss")
            return state, 1

        env._macro_engage = mock.Mock(side_effect=accepted_miss)
        _obs, _r, terminated, truncated, info = env.step(9)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        receipt = info["action_effect_audit"]
        self.assertEqual(receipt["native_attempts"], 1)
        self.assertEqual(receipt["native_accepts"], 1)
        self.assertTrue(receipt["request_executed"])
        self.assertFalse(receipt["material_effect"])
        self.assertFalse(receipt["stall_cost_applied"])
        reward.assert_called_once_with(
            state,
            state,
            requested_action=9,
            engage_target_generation_key=None,
            action_executed=True,
            action14_utility_delta=None,
        )

    def test_action14_partial_execution_and_commit_have_separate_receipts(self):
        state = raw()
        snapshot = mock.Mock()
        snapshot.candidates = ()

        partial_env, partial_reward = execution_step_fixture(state)
        partial_env._capture_controller_snapshot = mock.Mock(
            return_value=snapshot)

        def accepted_partial(
                kind, max_beats, *, controller_snapshot,
                action14_audit, execution_audit):
            self.assertEqual(kind, "gear")
            self.assertEqual(max_beats, 12)
            self.assertIs(controller_snapshot, snapshot)
            DiabloGymEnv._record_native_execution(
                execution_audit, 1, "action14 safe walk")
            return state, 1

        partial_env._macro_pickup = mock.Mock(
            side_effect=accepted_partial)
        _obs, _r, terminated, truncated, partial_info = (
            partial_env.step(14))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        partial = partial_info["action_effect_audit"]
        self.assertTrue(partial["request_executed"])
        self.assertFalse(partial["material_effect"])
        self.assertFalse(partial["stall_cost_applied"])
        self.assertEqual(
            partial_info["action14_audit"],
            {
                "accepted": False,
                "commit_attempts": 0,
                "utility_before": 0,
                "utility_after": 0,
                "utility_delta": 0,
            },
        )
        partial_reward.assert_called_once_with(
            state,
            state,
            requested_action=14,
            engage_target_generation_key=None,
            action_executed=True,
            action14_utility_delta=0,
        )

        committed_state = raw()
        committed_state["gear_combat_utility"] = 37
        commit_env, commit_reward = execution_step_fixture(state)
        commit_env._capture_controller_snapshot = mock.Mock(
            return_value=snapshot)

        def accepted_commit(
                kind, max_beats, *, controller_snapshot,
                action14_audit, execution_audit):
            DiabloGymEnv._record_native_execution(
                execution_audit, 1, "action14 gear commit")
            action14_audit.update({
                "accepted": True,
                "commit_attempts": 1,
                "utility_before": 0,
                "utility_after": 37,
                "utility_delta": 37,
            })
            return committed_state, 1

        commit_env._macro_pickup = mock.Mock(
            side_effect=accepted_commit)
        _obs, _r, terminated, truncated, commit_info = (
            commit_env.step(14))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        committed = commit_info["action_effect_audit"]
        self.assertTrue(committed["request_executed"])
        self.assertTrue(committed["material_effect"])
        self.assertIn("gear_commit", committed["effect_reasons"])
        self.assertFalse(committed["stall_cost_applied"])
        self.assertTrue(commit_info["action14_audit"]["accepted"])
        self.assertEqual(
            commit_info["action14_audit"]["utility_delta"], 37)
        commit_reward.assert_called_once_with(
            state,
            committed_state,
            requested_action=14,
            engage_target_generation_key=None,
            action_executed=True,
            action14_utility_delta=37,
        )

    def test_action14_partial_execution_is_not_counted_as_native_commit(self):
        base = mock.Mock()
        base.action_space = env_module.gym.spaces.Discrete(15)
        base._raw = raw()
        base._ep_kills = 0
        options = OptionsEnv.__new__(OptionsEnv)
        options.env = base
        options._worker_masks = lambda: np.ones(15, dtype=bool)
        options._drain = lambda: None

        def run(gear_audit):
            effect_audit = {
                "requested_action": 14,
                "native_attempts": 1,
                "native_accepts": 1,
                "request_executed": True,
                "material_effect": bool(gear_audit["accepted"]),
                "effect_reasons": (
                    ("gear_commit",) if gear_audit["accepted"] else ()),
                "same_scene": True,
                "stall_cost_applied": False,
            }
            options._win = {
                "W": 0.0,
                "worker_wage": 0.0,
                "worker_kills": 0,
                "worker_action14_requests": 0,
                "worker_action14_native_successes": 0,
                "worker_action14_gear_utility_delta": 0,
                "worker_no_effect_requests": 0,
                "gear_grace_pending": True,
                "gear_grace_consumed": False,
                "gear_grace_decisions": 0,
            }
            options._win_beat = lambda action: BeatOutcome(
                reason=None,
                requested_action=int(action),
                executed_action=14,
                fuse_tripped=False,
                action14_audit=dict(gear_audit),
                action_effect_audit=effect_audit,
            )
            outcome = options._win_step_worker(14)
            return outcome, dict(options._win)

        partial_outcome, partial_window = run({
            "accepted": False,
            "commit_attempts": 0,
            "utility_before": 0,
            "utility_after": 0,
            "utility_delta": 0,
        })
        self.assertEqual(partial_outcome.executed_action, 14)
        self.assertEqual(partial_window["worker_action14_requests"], 1)
        self.assertEqual(
            partial_window["worker_action14_native_successes"], 0)
        self.assertEqual(
            partial_window["worker_action14_gear_utility_delta"], 0)
        self.assertFalse(partial_window["gear_grace_consumed"])

        commit_outcome, commit_window = run({
            "accepted": True,
            "commit_attempts": 1,
            "utility_before": 0,
            "utility_after": 37,
            "utility_delta": 37,
        })
        self.assertEqual(commit_outcome.executed_action, 14)
        self.assertEqual(commit_window["worker_action14_requests"], 1)
        self.assertEqual(
            commit_window["worker_action14_native_successes"], 1)
        self.assertEqual(
            commit_window["worker_action14_gear_utility_delta"], 37)
        self.assertTrue(commit_window["gear_grace_consumed"])

    def test_option_controller_memories_never_change_flat_action_legality(self):
        """宏内部调度可改变 a9/a10 的目标，但不能制造同观测异 mask。"""
        state = raw(monsters=[monster(1, 100)])
        env = self.env_with(state)
        env._visited = {(10, 10)}
        env._explore_target = None
        env._explore_blocked_targets = set()
        env._engage_blocked_keys = set()
        clean = env.action_masks()

        env._visited = {(x, 10) for x in range(1, 20)}
        env._explore_target = (18, 10)
        env._explore_blocked_targets = {(15, 10), (16, 10)}
        env._engage_blocked_keys = {
            env._monster_generation_key(state["monsters"][0])}
        remembered = env.action_masks()
        self.assertTrue(np.array_equal(clean, remembered))
        self.assertTrue(remembered[9] and remembered[10])

    def test_walk_and_attack_animation_are_not_classified_as_idle(self):
        state = raw()
        state.update(
            future_x=9,
            future_y=10,
            player_mode=env_module.bridge.PM_WALK_NORTHWARDS,
            walkpath0=env_module.bridge.WALK_NONE,
            dest_action=env_module.bridge.ACTION_NONE,
        )
        self.assertTrue(DiabloGymEnv._movement_engine_busy(state))
        state.update(
            future_x=10,
            player_mode=env_module.bridge.PM_ATTACK,
            dest_action=env_module.bridge.ACTION_NONE,
        )
        self.assertTrue(DiabloGymEnv._engage_engine_busy(state))
        state["player_mode"] = 0
        self.assertFalse(DiabloGymEnv._engage_engine_busy(state))

    def test_worker_direction_uses_native_one_step_authority_guard(self):
        state = raw()
        state["triggers"] = [{
            "x": 11,
            "y": 10,
            "msg": env_module.bridge.WM_DIABNEXTLVL,
        }]
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = state
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=0,
            ) as guarded,
            mock.patch.object(env_module.bridge, "act_walk") as unrestricted,
        ):
            env._apply_action(3, worker_authority=True)

        guarded.assert_called_once_with(
            11, 10, [(11, 10)], 10, 10, 1)
        unrestricted.assert_not_called()

    def test_settle_charges_hidden_animation_and_respects_total_budget(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env.ticks_per_step = 4
        env._visited = {(10, 10)}
        env._exploration_progress = 0
        scene = (1, False, 0)

        busy = raw()
        busy.update(
            future_x=11, future_y=10,
            player_mode=env_module.bridge.PM_WALK_SIDEWAYS,
            walkpath0=env_module.bridge.WALK_NONE,
            dest_action=env_module.bridge.ACTION_NONE,
        )
        idle = dict(busy)
        idle.update(
            player_x=11, player_y=10,
            future_x=11, future_y=10,
            player_mode=env_module.bridge.PM_STAND,
        )
        self.assertFalse(env._decision_idle(busy))
        self.assertTrue(env._decision_idle(idle))

        with (
            mock.patch.object(
                env_module.bridge, "act_wait", return_value=1) as wait,
            mock.patch.object(
                env_module.bridge, "step", side_effect=[busy, idle]) as step,
        ):
            settled, beats = env._settle_to_idle(
                busy, 1, max_beats=4, start_scene=scene)
        self.assertIs(settled, idle)
        self.assertEqual(beats, 3)
        self.assertEqual(step.call_count, 2)
        wait.assert_called_once_with()
        self.assertEqual(env.exploration_progress, 1)

        # 只剩一拍时必须计费后截断，不能越预算偷偷推进到 idle。
        env._visited = {(10, 10)}
        env._exploration_progress = 0
        with (
            mock.patch.object(
                env_module.bridge, "act_wait", return_value=1),
            mock.patch.object(
                env_module.bridge, "step", return_value=busy) as step,
        ):
            unsettled, beats = env._settle_to_idle(
                busy, 1, max_beats=2, start_scene=scene)
        self.assertIs(unsettled, busy)
        self.assertEqual(beats, 2)
        step.assert_called_once_with(ticks=4)

    def test_budget_cut_mid_animation_is_not_a_bootstrapable_truncation(self):
        """295 维未编码的 busy terminal observation 必须 fail-closed。"""
        # 真实引擎覆盖放在 smoke_random_agent；这里锁死边界分类的最小
        # 语义：idle 上限可 TimeLimit bootstrap，busy 上限不可。
        idle = raw()
        idle.update(
            future_x=10, future_y=10,
            player_mode=env_module.bridge.PM_STAND,
            walkpath0=env_module.bridge.WALK_NONE,
            dest_action=env_module.bridge.ACTION_NONE,
        )
        busy = dict(idle)
        busy.update(
            future_x=11,
            player_mode=env_module.bridge.PM_WALK_SIDEWAYS,
        )
        self.assertTrue(DiabloGymEnv._decision_idle(idle))
        self.assertFalse(DiabloGymEnv._decision_idle(busy))
        self.assertEqual(
            DiabloGymEnv._episode_boundary(idle, 1, 1),
            (False, True, True, True, False),
        )
        self.assertEqual(
            DiabloGymEnv._episode_boundary(busy, 1, 1),
            (True, False, True, False, True),
        )
        # 真死亡优先于预算中断，且 terminated/truncated 仍互斥。
        busy["dead"] = True
        self.assertEqual(
            DiabloGymEnv._episode_boundary(busy, 1, 1),
            (True, False, True, False, False),
        )

    def test_all_actions_share_first_visit_progress_accounting(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._visited = {(10, 10)}
        env._exploration_progress = 0

        self.assertFalse(env._record_visit((10, 10)))
        self.assertEqual(env.exploration_progress, 0)
        self.assertTrue(env._record_visit((11, 10)))
        self.assertEqual(env.exploration_progress, 1)
        self.assertFalse(env._record_visit((11, 10)))
        self.assertEqual(env.exploration_progress, 1)

    def test_explore_frontier_target_is_sticky_across_macro_boundaries(self):
        """复现 seed7002 的旧 A↔B 抖动：重选会回头，粘性目标会续走。"""
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._visited = {(20, 64)}
        env._explore_target = None
        env._explore_blocked_targets = set()
        state = raw()
        state.update(player_x=20, player_y=64)

        # A=(20,69) 起初是唯一达到五格阈值的边疆；一段宏后玩家到
        # (23,68)，若按新位置重新贪心则 B=(18,64) 反而最近并导致回头。
        walkable = {
            (20, 64), (20, 65), (20, 66), (20, 67), (20, 68),
            (20, 69), (21, 68), (22, 68), (23, 68),
            (19, 64), (18, 64), (17, 64), (16, 64),
        }
        radius = env._EXPLORE_RADIUS
        side = 2 * radius + 1

        def local_map(*, radius):
            self.assertEqual(radius, env._EXPLORE_RADIUS)
            px, py = state["player_x"], state["player_y"]
            cells = side * side
            result = {
                "walkable": [0] * cells,
                "monster": [0] * cells,
                "door": [0] * cells,
                "closed_door": [0] * cells,
                "hazard": [0] * cells,
                "explosive_softwall": [0] * cells,
            }
            for x, y in walkable:
                if abs(x - px) <= radius and abs(y - py) <= radius:
                    i = (y - py + radius) * side + (x - px + radius)
                    result["walkable"][i] = 1
            return result

        with mock.patch.object(
                env_module.bridge, "local_map", side_effect=local_map):
            first = env._plan_explore_step(state)
            self.assertEqual(first, ("frontier", 20, 69))
            self.assertEqual(env._explore_target, (20, 69))

            env._visited.update({
                (20, 65), (20, 66), (20, 67), (20, 68),
                (21, 68), (22, 68), (23, 68),
            })
            state.update(player_x=23, player_y=68)
            self.assertEqual(
                env._plan_explore_step(state), ("frontier", 20, 69))

            # 证明该断言确实覆盖旧 bug：丢掉跨宏目标后，同一几何会回选 B。
            env._explore_target = None
            self.assertEqual(
                env._plan_explore_step(state), ("frontier", 18, 64))

    def test_explore_failed_frontiers_rotate_but_never_become_permanent_blacklist(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._visited = {(10, 10)}
        env._explore_target = None
        env._explore_blocked_targets = {(15, 10)}
        state = raw()

        radius = env._EXPLORE_RADIUS
        side = 2 * radius + 1
        cells = side * side
        local_map = {
            "walkable": [0] * cells,
            "monster": [0] * cells,
            "door": [0] * cells,
            "closed_door": [0] * cells,
            "hazard": [0] * cells,
            "explosive_softwall": [0] * cells,
        }
        for x in range(10, 16):
            i = radius * side + (x - 10 + radius)
            local_map["walkable"][i] = 1

        with mock.patch.object(
                env_module.bridge, "local_map", return_value=local_map):
            # 旧实现会因为唯一 frontier 曾失败而从此每拍返回 None；
            # 动态阻挡已经消失时必须开始下一轮，并保持相同 action10 语义。
            self.assertEqual(
                env._plan_explore_step(state), ("frontier", 15, 10))
        self.assertNotIn((15, 10), env._explore_blocked_targets)

    def test_explore_without_connected_frontier_waits_and_never_descends(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = raw()
        env._raw["progression_targets"] = []
        env._raw["triggers"] = [{"msg": 123, "x": 90, "y": 90}]
        env._visited = {(10, 10)}
        waited = object()
        env._wait_step = lambda: (waited, 1)
        env._macro_descend = lambda **_kwargs: self.fail(
            "普通 explore 无 frontier 时越权调用 descend")

        radius = env._EXPLORE_RADIUS
        cells = (2 * radius + 1) ** 2
        local_map = {
            "walkable": [0] * cells,
            "monster": [0] * cells,
            "door": [0] * cells,
            "closed_door": [0] * cells,
            "hazard": [0] * cells,
            "explosive_softwall": [0] * cells,
        }
        local_map["walkable"][cells // 2] = 1
        with mock.patch.object(env_module.bridge, "local_map",
                               return_value=local_map):
            result, beats = env._macro_explore(max_beats=12)
        self.assertIs(result, waited)
        self.assertEqual(beats, 1)

    def test_explore_plans_softwall_but_protects_triggers_and_progression(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._visited = {(10, 10)}
        state = raw()
        radius = env._EXPLORE_RADIUS
        side = 2 * radius + 1
        cells = side * side

        def index(x, y):
            return (y - 10 + radius) * side + (x - 10 + radius)

        closed = {
            "walkable": [0] * cells,
            "monster": [0] * cells,
            "door": [0] * cells,
            "hazard": [0] * cells,
            "explosive_softwall": [0] * cells,
        }
        closed["walkable"][index(10, 10)] = 1
        closed["door"][index(11, 10)] = 1
        for x in range(12, 17):
            closed["walkable"][index(x, 10)] = 1

        with mock.patch.object(
                env_module.bridge, "local_map", return_value=closed):
            self.assertEqual(
                env._plan_explore_step(state), ("open", 11, 10))

            protected_trigger = dict(state)
            protected_trigger["triggers"] = [
                {"x": 11, "y": 10, "msg": 999},
            ]
            self.assertIsNone(env._plan_explore_step(protected_trigger))

            protected_story = dict(state)
            protected_story["progression_targets"] = [{
                "x": 11, "y": 10,
                "goal_x": 12, "goal_y": 10,
            }]
            self.assertIsNone(env._plan_explore_step(protected_story))

        # 即使楼梯后方有足够远的未访地板，trigger 格也不能被 BFS 当作
        # 通路，更不能成为 act_walk 的相邻安全步。
        stairs = {
            "walkable": [0] * cells,
            "monster": [0] * cells,
            "door": [0] * cells,
        }
        for x in range(10, 17):
            stairs["walkable"][index(x, 10)] = 1
        state_with_stairs = dict(state)
        state_with_stairs["triggers"] = [
            {"x": 11, "y": 10, "msg": 999},
        ]
        with mock.patch.object(
                env_module.bridge, "local_map", return_value=stairs):
            self.assertIsNone(
                env._plan_explore_step(state_with_stairs))

    def test_explore_opens_softwall_then_requires_a_new_snapshot(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        state = raw()
        state.update(
            future_x=10, future_y=10, player_mode=0,
            walkpath0=env_module.bridge.WALK_NONE,
            dest_action=env_module.bridge.ACTION_NONE,
        )
        env._raw = state
        env.ticks_per_step = 4
        env._visited = {(10, 10)}
        env._exploration_progress = 0
        env._softwalls_opened = 0
        env._wait_step = lambda: self.fail(
            "有普通闭门通往新区域时 action10 错误返回 wait")
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)

        radius = env._EXPLORE_RADIUS
        side = 2 * radius + 1
        cells = side * side

        def index(x, y):
            return (y - 10 + radius) * side + (x - 10 + radius)

        closed = {
            "walkable": [0] * cells,
            "monster": [0] * cells,
            "door": [0] * cells,
            "hazard": [0] * cells,
            "explosive_softwall": [0] * cells,
        }
        closed["walkable"][index(10, 10)] = 1
        closed["door"][index(11, 10)] = 1
        for x in range(12, 17):
            closed["walkable"][index(x, 10)] = 1

        after_open = dict(state)

        with (
            mock.patch.object(
                env_module.bridge, "local_map",
                return_value=closed) as local_map,
            mock.patch.object(
                env_module.bridge, "act_controller_operate",
                return_value=1, create=True) as operate,
            mock.patch.object(
                env_module.bridge, "act_explore_walk") as walk_to,
            mock.patch.object(
                env_module.bridge, "step",
                return_value=after_open),
            mock.patch.object(
                env_module.bridge, "probe_tile",
                return_value={"walkable": True}),
        ):
            result, beats = env._macro_explore(max_beats=12)

        self.assertIs(result, after_open)
        self.assertEqual(beats, 1)
        self.assertEqual(env.exploration_progress, 1)
        self.assertEqual(env.softwalls_opened, 1)
        operate.assert_called_once_with(11, 10, 10, 10, 12)
        walk_to.assert_not_called()
        # 观测和执行严格共用一次 radius-12 读取；开门后的新几何只能由
        # 下一决策的新快照消费，不能在同一个 action10 内偷读第二张图。
        local_map.assert_called_once_with(radius=env._EXPLORE_RADIUS)

    def test_vector_has_no_hidden_monster_side_channel(self):
        state = raw(monsters=[
            monster(1, 100),
            monster(2, 100, visible=False),
        ])
        side = 11
        cells = side * side
        # 故意让原始 local_map 谎称每格都有怪；向量必须自行按 policy
        # 子集重建怪物通道，不能照抄这个全知碰撞通道。
        local_map = {
            "walkable": [1] * cells,
            "monster": [1] * cells,
            "door": [0] * cells,
        }
        with mock.patch.object(env_module.bridge, "local_map",
                               return_value=local_map):
            vector = DiabloGymEnv._vectorize(state)

        self.assertEqual(vector.shape, (295,))
        self.assertAlmostEqual(float(vector[8]), 1 / 50)
        monster_map_start = 12 + 8 * 4 + cells
        monster_map = vector[monster_map_start:monster_map_start + cells]
        self.assertEqual(float(monster_map.sum()), 1.0)
        # actionable monster(1) 位于玩家东二格：(dy=0, dx=2)。
        self.assertEqual(monster_map[5 * side + 7], 1.0)

    def test_complete_v3_worker_view_is_rebuilt_before_v4_information_loss(self):
        hidden = monster(1, 80, visible=False)
        hidden.update(x=11, y=11)
        visible = monster(2, 60)
        visible.update(x=15, y=10)
        hidden_heal = {
            "x": 11, "y": 10, "heal": False, "legacy_heal": True,
            "gear": False,
            "visible": False, "reachable": True,
        }
        visible_heal = {
            "x": 14, "y": 10, "heal": True, "legacy_heal": True,
            "gear": False,
            "visible": True, "reachable": True,
        }
        state = raw(
            monsters=[hidden, visible],
            floor_items=[hidden_heal, visible_heal],
            belt_heals=2,
            belt_free_slots=3,
        )
        state["legacy_belt_heals"] = 3
        side = 2 * env_module._MAP_RADIUS + 1
        cells = side * side
        physical_monsters = [0] * cells
        physical_monsters[6 * side + 6] = 1
        physical_monsters[5 * side + 10] = 1
        local_map = {
            "walkable": [1] * cells,
            "monster": physical_monsters,
            "door": [0] * cells,
        }
        with mock.patch.object(
                env_module.bridge, "local_map", return_value=local_map):
            current = DiabloGymEnv._vectorize(state)
            legacy = DiabloGymEnv._legacy_policy_vectorize(state)

            base = DiabloGymEnv.__new__(DiabloGymEnv)
            base._raw = state
            base._steps = 17
            base._ep_kills = 0
            options = OptionsEnv.__new__(OptionsEnv)
            options.env = base
            options.max_steps = 3000
            options.manager_observation_view = (
                MANAGER_OBSERVATION_VIEW_LEGACY_V3)
            options._last_base_obs = current
            options._legacy_layer_clock = 7
            options._legacy_exhausted = False
            options._legacy_layer_steps0 = 0
            options.layer_clock = 11
            options._layer_steps0 = 0
            options._layer_kills0 = 0
            options._last_opt = -1
            options._last_tau = 0
            options._win = {
                "t0": 10,
                "voluntary_drinks": 1,
                "drains": 0,
            }
            exact = options._worker_policy_observation(
                WORKER_OBSERVATION_VIEW_LEGACY_V3)
            overlay = options._worker_policy_observation(
                WORKER_OBSERVATION_VIEW_A12_OVERLAY)
            manager = options._mgr_obs(current)
            options.manager_observation_view = (
                MANAGER_OBSERVATION_VIEW_RAW_V4)
            raw_manager = options._mgr_obs(current)

        self.assertEqual(current.shape, legacy.shape, (295,))
        self.assertAlmostEqual(float(current[8]), 1 / 50)
        self.assertAlmostEqual(float(legacy[8]), 2 / 50)
        self.assertAlmostEqual(float(current[9]), 5 / 30)
        self.assertAlmostEqual(float(legacy[9]), 1 / 30)
        # First entity slot and map channel prove that this is not merely the
        # old 286/297 scalar decoder.
        self.assertAlmostEqual(float(current[12]), 5 / 20)
        self.assertAlmostEqual(float(legacy[12]), 1 / 20)
        self.assertAlmostEqual(float(legacy[13]), 1 / 20)
        monster_map_start = 12 + env_module._K_MONSTERS * 4 + cells
        self.assertEqual(
            float(current[
                monster_map_start:monster_map_start + cells].sum()),
            1.0,
        )
        self.assertEqual(
            float(legacy[
                monster_map_start:monster_map_start + cells].sum()),
            2.0,
        )
        self.assertTrue(np.array_equal(exact[:295], legacy))
        self.assertTrue(np.array_equal(manager[:295], legacy))
        self.assertTrue(np.array_equal(raw_manager[:295], current))
        self.assertAlmostEqual(float(exact[286]), 3 / 8)
        self.assertEqual(float(exact[297]), 0.0)
        self.assertAlmostEqual(
            float(overlay[286]), 3 / 8 + 3 / 128)
        self.assertEqual(float(overlay[297]), -1.0)
        self.assertTrue(np.array_equal(
            legacy_worker_policy_observation_view(overlay),
            exact,
        ))

    def test_v3_compatibility_view_rejects_missing_native_legacy_fields(self):
        state = raw(monsters=[], floor_items=[])
        state.pop("legacy_belt_heals", None)
        with self.assertRaisesRegex(RuntimeError, "legacy_belt_heals"):
            DiabloGymEnv._legacy_policy_vectorize(state)

        state["legacy_belt_heals"] = 0
        state["floor_items"] = [{
            "x": 10, "y": 10, "heal": True, "gear": False,
        }]
        with self.assertRaisesRegex(RuntimeError, "legacy_heal"):
            DiabloGymEnv._legacy_policy_vectorize(state)

    def test_controller_wire_exposes_type_ledger_blocked_belt_and_sticky(self):
        target = monster(5, 90, max_hp=100)
        target["type"] = 17
        state = raw(
            monsters=[target],
            floor_items=[
                {
                    "x": 9, "y": 9, "heal": True, "gear": False,
                    "visible": True, "reachable": True,
                },
                {
                    "x": 12, "y": 10, "heal": False, "gear": True,
                    "visible": True, "reachable": True,
                },
            ],
            belt_heals=2,
            belt_free_slots=6,
        )
        state["belt_heal_kinds"] = [1, 4, 0, 0, 0, 0, 0, 0]
        state["belt_slot_kinds"] = [2, 5, 0, 0, 0, 0, 0, 0]
        env = controller_fixture(
            state,
            engage_blocked={5, 999},
            explore_blocked={(11, 10)},
            sticky=(15, 10),
            ledger={5: (40, 100)},
        )
        before_blocked = set(env._engage_blocked_keys)
        first = env.controller_snapshot_vector()
        second = env.controller_snapshot_vector()

        self.assertEqual(CONTROLLER_SNAPSHOT_VECTOR_DIM, 12377)
        self.assertEqual(first.shape, (CONTROLLER_SNAPSHOT_VECTOR_DIM,))
        self.assertTrue(np.isfinite(first).all())
        # 固定尺度只除不裁剪；本 fixture 的登记域应保持在合理幅值内。
        self.assertLessEqual(float(np.abs(first).max()), 1.0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(env._engage_blocked_keys, before_blocked)

        candidate = CONTROLLER_SNAPSHOT_MAP_DIM
        self.assertAlmostEqual(
            float(first[candidate + 2]), 17 / 200)
        self.assertAlmostEqual(
            float(first[candidate + 9]), 40 / 1024)
        self.assertEqual(float(first[candidate + 11]), 1.0)
        self.assertEqual(float(first[candidate + 12]), 1.0)
        self.assertEqual(float(first[candidate + 13]), 1.0)
        self.assertEqual(float(first[candidate + 14]), 1.0)

        belt = (
            CONTROLLER_SNAPSHOT_MAP_DIM
            + CONTROLLER_SNAPSHOT_MONSTER_DIM
            + CONTROLLER_SNAPSHOT_MISSILE_DIM
        )
        np.testing.assert_array_equal(
            first[belt:belt + 12],
            np.asarray(
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                dtype=np.float32,
            ),
        )
        sticky = (
            belt + CONTROLLER_SNAPSHOT_BELT_DIM
            + CONTROLLER_SNAPSHOT_EXACT_DIM
            + CONTROLLER_SNAPSHOT_COMBAT_DIM
        )
        np.testing.assert_allclose(
            first[sticky:sticky + 3],
            np.asarray([1.0, 5 / 112, 0.0], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )
        heal = sticky + 3
        np.testing.assert_allclose(
            first[heal:heal + CONTROLLER_SNAPSHOT_HEAL_TARGET_DIM],
            np.asarray(
                [1.0, -1 / 112, -1 / 112, 0.0, 1 / 4],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=1e-7,
        )
        gear = heal + CONTROLLER_SNAPSHOT_HEAL_TARGET_DIM
        np.testing.assert_allclose(
            first[gear:gear + 3],
            np.asarray([1.0, 2 / 112, 0.0], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )

    def test_controller_wire_distinguishes_each_previous_alias(self):
        base_monster = monster(5, 90, max_hp=100)
        state = raw(monsters=[base_monster], belt_heals=2, belt_free_slots=6)

        def vector(*, monster_type=0, ledger_low=40, blocked=False,
                   belt=(2, 5, 0, 0, 0, 0, 0, 0)):
            current = dict(state)
            current["monsters"] = [dict(base_monster)]
            current["monsters"][0]["type"] = monster_type
            current["belt_slot_kinds"] = list(belt)
            fixture = controller_fixture(
                current,
                engage_blocked={5} if blocked else set(),
                ledger={5: (ledger_low, 100)},
            )
            return fixture.controller_snapshot_vector()

        baseline = vector()
        self.assertFalse(np.array_equal(
            baseline, vector(monster_type=1)))
        self.assertFalse(np.array_equal(
            baseline, vector(ledger_low=39)))
        self.assertFalse(np.array_equal(
            baseline, vector(blocked=True)))
        self.assertFalse(np.array_equal(
            baseline, vector(belt=(5, 2, 0, 0, 0, 0, 0, 0))))

    def test_controller_wire_distinguishes_fixed_hp_monster_phase_and_equipment(self):
        baseline_state = raw(monsters=[monster(5, 90)])
        baseline = controller_fixture(
            baseline_state).controller_snapshot_vector()

        fixed_hp = raw(monsters=[monster(5, 90)])
        fixed_hp["monsters"][0]["hp_fixed_lo"] += 1
        self.assertFalse(np.array_equal(
            baseline,
            controller_fixture(fixed_hp).controller_snapshot_vector(),
        ))

        attack_phase = raw(monsters=[monster(5, 90)])
        attack_phase["monsters"][0]["anim_tick"] = 1
        self.assertFalse(np.array_equal(
            baseline,
            controller_fixture(attack_phase).controller_snapshot_vector(),
        ))

        scavenger_heal_phase = raw(monsters=[monster(5, 90)])
        scavenger_heal_phase["monsters"][0]["goal_var3"] = 1
        self.assertFalse(np.array_equal(
            baseline,
            controller_fixture(
                scavenger_heal_phase
            ).controller_snapshot_vector(),
        ))

        equipped = raw(monsters=[monster(5, 90)])
        equipped["equipped_items"][0].update(
            present=True,
            item_type=1,
            equip_loc=1,
            durability=1,
            max_durability=2,
            min_strength=25,
            stat_usable=True,
            effect_damage=7,
            effect_flags=4,
            combat_utility_lo=4096,
        )
        self.assertFalse(np.array_equal(
            baseline,
            controller_fixture(equipped).controller_snapshot_vector(),
        ))

    def test_every_declared_monster_transition_field_changes_the_wire(self):
        baseline_monster = monster(5, 90)
        baseline = controller_fixture(
            raw(monsters=[baseline_monster])
        ).controller_snapshot_vector()
        for field in (
            env_module.CONTROLLER_SNAPSHOT_MONSTER_DYNAMIC_FIELDS
            + env_module.CONTROLLER_SNAPSHOT_MONSTER_INT32_FIELDS
        ):
            with self.subTest(field=field):
                changed_monster = dict(baseline_monster)
                changed_monster[field] = (
                    0 if int(changed_monster[field]) != 0 else 1)
                changed = controller_fixture(
                    raw(monsters=[changed_monster])
                ).controller_snapshot_vector()
                self.assertFalse(np.array_equal(baseline, changed))

        changed_flags = dict(baseline_monster)
        changed_flags["combat_flags"] = 1
        self.assertFalse(np.array_equal(
            baseline,
            controller_fixture(
                raw(monsters=[changed_flags])
            ).controller_snapshot_vector(),
        ))

    def test_policy_wire_excludes_rng_drop_and_save_bookkeeping(self):
        floor_gear = {
            "x": 11, "y": 10, "heal": False, "gear": True,
            "visible": True, "reachable": True,
            "combat_utility_lo": 4096,
        }
        baseline_state = raw(
            monsters=[monster(5, 90)],
            floor_items=[floor_gear],
            missiles=[missile(damage=9)],
        )
        noisy_state = raw(
            monsters=[monster(5, 90)],
            floor_items=[dict(floor_gear)],
            missiles=[missile(damage=9)],
        )
        noisy_state["monsters"][0].update(
            rnd_item_seed_hi=0x1234,
            rnd_item_seed_lo=0x5678,
            ai_seed_hi=0x9ABC,
            ai_seed_lo=0xDEF0,
        )
        noisy_state["missiles"][0]["light_id"] = 44
        noisy_state["floor_items"][0].update(
            seed_hi=0xABCD,
            seed_lo=0x1234,
            create_info=0x5678,
            charges=33,
        )
        np.testing.assert_array_equal(
            controller_fixture(baseline_state).controller_snapshot_vector(),
            controller_fixture(noisy_state).controller_snapshot_vector(),
        )

    def test_controller_missile_slots_and_overflow_risk_are_observable(self):
        no_projectile = controller_fixture(raw()).controller_snapshot_vector()
        one = controller_fixture(raw(missiles=[
            missile(damage=9, velocity_x=0x12345678),
        ])).controller_snapshot_vector()
        changed_low_word = controller_fixture(raw(missiles=[
            missile(damage=9, velocity_x=0x12345679),
        ])).controller_snapshot_vector()
        self.assertFalse(np.array_equal(no_projectile, one))
        self.assertFalse(np.array_equal(one, changed_low_word))
        for field in ("random", "anim_count", "anim_add"):
            changed = missile(damage=9, velocity_x=0x12345678)
            changed[field] = int(changed[field]) + 1
            self.assertFalse(
                np.array_equal(
                    one,
                    controller_fixture(
                        raw(missiles=[changed])
                    ).controller_snapshot_vector(),
                ),
                field,
            )

        diagnostic_only = missile(damage=9, velocity_x=0x12345678)
        diagnostic_only.update({
            "last_collision_target_hash": 0x1234,
            "var1": 1,
            "var2": 2,
            "var3": 3,
            "var4": 4,
            "var5": 5,
            "var6": 6,
            "var7": 7,
        })
        np.testing.assert_array_equal(
            one,
            controller_fixture(
                raw(missiles=[diagnostic_only])
            ).controller_snapshot_vector(),
        )

        crowded = raw(missiles=[
            missile(
                tile_dx=(index % 5) + 1,
                tile_dy=index % 3,
                damage=index + 1,
                duration=50 - index,
                source_id=index,
            )
            for index in range(CONTROLLER_SNAPSHOT_MISSILE_LIMIT + 3)
        ])
        vector = controller_fixture(crowded).controller_snapshot_vector()
        missile_start = (
            CONTROLLER_SNAPSHOT_MAP_DIM + CONTROLLER_SNAPSHOT_MONSTER_DIM)
        overflow_start = (
            missile_start
            + CONTROLLER_SNAPSHOT_MISSILE_LIMIT
            * CONTROLLER_SNAPSHOT_MISSILE_FIELDS
        )
        overflow = vector[
            overflow_start:
            overflow_start + CONTROLLER_SNAPSHOT_MISSILE_OVERFLOW_DIM
        ]
        self.assertAlmostEqual(
            float(overflow[0]),
            (CONTROLLER_SNAPSHOT_MISSILE_LIMIT + 3) / 256)
        self.assertAlmostEqual(float(overflow[1]), 3 / 256)
        self.assertGreater(float(overflow[4]), 0.0)

    def test_hidden_missiles_and_hidden_source_coordinates_do_not_leak(self):
        no_projectile = controller_fixture(raw()).controller_snapshot_vector()
        hidden = missile(
            visible=False,
            source_visible=False,
            start_visible=False,
            source_id=71,
            start_dx=11,
            start_dy=-9,
        )
        np.testing.assert_array_equal(
            no_projectile,
            controller_fixture(
                raw(missiles=[hidden])
            ).controller_snapshot_vector(),
        )

        first = missile(
            visible=True,
            source_visible=False,
            start_visible=False,
            source_id=71,
            start_dx=11,
            start_dy=-9,
        )
        second = dict(first)
        second.update(source_id=199, start_dx=-77, start_dy=81)
        first_vector = controller_fixture(
            raw(missiles=[first])
        ).controller_snapshot_vector()
        second_vector = controller_fixture(
            raw(missiles=[second])
        ).controller_snapshot_vector()
        np.testing.assert_array_equal(first_vector, second_vector)

        # Two otherwise distinct visible missiles used to be row-sorted by
        # the raw hidden source id before the id was redacted in each row.
        # Swapping those hidden ids therefore leaked through slot order.
        left = missile(
            visible=True,
            source_visible=False,
            source_id=71,
            tile_dx=1,
            damage=3,
        )
        right = missile(
            visible=True,
            source_visible=False,
            source_id=199,
            tile_dx=1,
            damage=9,
        )
        ordered_hidden = controller_fixture(
            raw(missiles=[left, right])
        ).controller_snapshot_vector()
        swapped_left = dict(left)
        swapped_right = dict(right)
        swapped_left["source_id"], swapped_right["source_id"] = (
            swapped_right["source_id"], swapped_left["source_id"])
        swapped_hidden = controller_fixture(
            raw(missiles=[swapped_left, swapped_right])
        ).controller_snapshot_vector()
        np.testing.assert_array_equal(ordered_hidden, swapped_hidden)

        source_revealed = dict(first)
        source_revealed["source_visible"] = True
        start_revealed = dict(first)
        start_revealed["start_visible"] = True
        self.assertFalse(np.array_equal(
            first_vector,
            controller_fixture(
                raw(missiles=[source_revealed])
            ).controller_snapshot_vector(),
        ))
        self.assertFalse(np.array_equal(
            first_vector,
            controller_fixture(
                raw(missiles=[start_revealed])
            ).controller_snapshot_vector(),
        ))

    def test_every_declared_missile_transition_field_changes_the_wire(self):
        baseline_missile = missile()
        baseline = controller_fixture(
            raw(missiles=[baseline_missile])
        ).controller_snapshot_vector()
        for field in (
            env_module.CONTROLLER_SNAPSHOT_MISSILE_DIRECT_FIELDS
            + env_module.CONTROLLER_SNAPSHOT_MISSILE_INT32_FIELDS
        ):
            with self.subTest(field=field):
                changed_missile = dict(baseline_missile)
                changed_missile[field] = (
                    0 if int(changed_missile[field]) != 0 else 1)
                changed = controller_fixture(
                    raw(missiles=[changed_missile])
                ).controller_snapshot_vector()
                self.assertFalse(np.array_equal(baseline, changed))

    def test_controller_candidates_and_items_use_only_local_snapshot_reachability(self):
        locally_open = controller_map()
        target = monster(5, 90, reachable=False)
        target.update(x=12, y=10, future_x=12, future_y=10)
        state = raw(
            monsters=[target],
            floor_items=[{
                "x": 11,
                "y": 10,
                "heal": True,
                "gear": False,
                "visible": True,
                "reachable": False,
            }],
        )
        snapshot = controller_fixture(
            state, local_map=locally_open)._controller_snapshot
        self.assertEqual(
            [entry.monster_id for entry in snapshot.candidates], [5])
        self.assertEqual(
            [entry.monster_id for entry in snapshot.monsters], [5])
        self.assertFalse(snapshot.monsters[0].native_reachable)
        self.assertTrue(snapshot.monsters[0].locally_engageable)
        self.assertIsNotNone(snapshot.heal_target)

        sealed = controller_map(walkable=0)
        far_target = monster(5, 90, reachable=True)
        far_target.update(x=12, y=10, future_x=12, future_y=10)
        outside_item = {
            "x": 23,
            "y": 10,
            "heal": True,
            "gear": False,
            "visible": True,
            "reachable": True,
        }
        blocked = controller_fixture(
            raw(monsters=[far_target], floor_items=[outside_item]),
            local_map=sealed,
        )._controller_snapshot
        self.assertEqual(blocked.candidates, ())
        self.assertEqual(
            [entry.monster_id for entry in blocked.monsters], [5])
        self.assertTrue(blocked.monsters[0].visible)
        self.assertTrue(blocked.monsters[0].native_reachable)
        self.assertFalse(blocked.monsters[0].locally_engageable)
        self.assertIsNone(blocked.heal_target)

    def test_edge_monster_with_future_tile_outside_window_is_not_candidate(self):
        edge = monster(5, 90, reachable=True)
        edge.update(x=22, y=10, future_x=23, future_y=10)
        fixture = controller_fixture(
            raw(monsters=[edge]), local_map=controller_map())
        snapshot = fixture._controller_snapshot

        # Its currently visible tile remains an actor observation row, but
        # action 9 cannot be masked legal when the native command target is
        # already outside the observation-bound radius.
        self.assertEqual(
            [entry.monster_id for entry in snapshot.monsters], [5])
        self.assertFalse(snapshot.monsters[0].locally_engageable)
        self.assertEqual(snapshot.candidates, ())
        self.assertFalse(fixture.action_masks()[9])

    def test_controller_actor_map_and_rows_do_not_leak_hidden_monsters(self):
        visible = monster(1, 90)
        visible.update(x=11, y=10, future_x=11, future_y=10)
        hidden = monster(2, 90, visible=False)
        hidden.update(x=12, y=10, future_x=12, future_y=10)
        local = controller_map()
        radius = env_module.CONTROLLER_SNAPSHOT_RADIUS
        side = env_module.CONTROLLER_SNAPSHOT_SIDE

        def index(x, y):
            return (y - 10 + radius) * side + (x - 10 + radius)

        local["monster"][index(11, 10)] = 1
        local["monster"][index(12, 10)] = 1
        fixture = controller_fixture(
            raw(monsters=[visible, hidden]), local_map=local)
        snapshot = fixture._controller_snapshot
        self.assertEqual(snapshot.physical_monster[index(12, 10)], 1)
        self.assertEqual(snapshot.visible_monster[index(11, 10)], 1)
        self.assertEqual(snapshot.visible_monster[index(12, 10)], 0)
        self.assertEqual(
            [entry.monster_id for entry in snapshot.monsters], [1])

        vector = fixture.controller_snapshot_vector()
        visible_plane = vector[
            env_module.CONTROLLER_SNAPSHOT_CELLS:
            2 * env_module.CONTROLLER_SNAPSHOT_CELLS
        ]
        self.assertEqual(float(visible_plane[index(11, 10)]), 1.0)
        self.assertEqual(float(visible_plane[index(12, 10)]), 0.0)

    def test_controller_monster_capacity_is_deterministic_and_summarized(self):
        crowded_monsters = []
        for monster_id in range(CONTROLLER_SNAPSHOT_MONSTER_LIMIT + 3):
            entry = monster(monster_id, monster_id + 1, max_hp=100)
            entry.update(x=11, y=10, future_x=11, future_y=10)
            entry["max_damage"] = monster_id
            crowded_monsters.append(entry)

        fixture = controller_fixture(raw(monsters=crowded_monsters))
        snapshot = fixture._controller_snapshot
        self.assertEqual(len(snapshot.monsters),
                         CONTROLLER_SNAPSHOT_MONSTER_LIMIT)
        self.assertEqual(len(snapshot.candidates),
                         CONTROLLER_SNAPSHOT_MONSTER_LIMIT)
        self.assertEqual(
            [entry.monster_id for entry in snapshot.monsters],
            list(range(CONTROLLER_SNAPSHOT_MONSTER_LIMIT)),
        )

        vector = fixture.controller_snapshot_vector()
        np.testing.assert_array_equal(
            vector,
            controller_fixture(
                raw(monsters=list(reversed(crowded_monsters)))
            ).controller_snapshot_vector(),
        )
        overflow_start = (
            CONTROLLER_SNAPSHOT_MAP_DIM
            + CONTROLLER_SNAPSHOT_MONSTER_LIMIT
            * CONTROLLER_SNAPSHOT_MONSTER_FIELDS
        )
        overflow = vector[
            overflow_start:
            overflow_start
            + env_module.CONTROLLER_SNAPSHOT_MONSTER_OVERFLOW_DIM
        ]
        self.assertAlmostEqual(
            float(overflow[0]),
            (CONTROLLER_SNAPSHOT_MONSTER_LIMIT + 3) / 256,
        )
        self.assertAlmostEqual(float(overflow[1]), 3 / 256)
        self.assertAlmostEqual(
            float(overflow[2]),
            (CONTROLLER_SNAPSHOT_MONSTER_LIMIT + 3) / 256,
        )
        self.assertAlmostEqual(float(overflow[3]), 3 / 256)
        self.assertGreater(float(overflow[4]), 0.0)
        self.assertAlmostEqual(
            float(overflow[7]),
            (CONTROLLER_SNAPSHOT_MONSTER_LIMIT + 2) / 255,
        )

    def test_controller_path_and_wire_fail_closed_on_hazardous_softwalls(self):
        def corridor(*, hazard=False, explosive=False):
            local = controller_map(walkable=0)
            radius = env_module.CONTROLLER_SNAPSHOT_RADIUS
            side = env_module.CONTROLLER_SNAPSHOT_SIDE

            def index(x, y):
                return (y - 10 + radius) * side + (x - 10 + radius)

            for x in range(10, 14):
                local["walkable"][index(x, 10)] = 1
            local["hazard"][index(11, 10)] = int(hazard)
            local["explosive_softwall"][index(11, 10)] = int(explosive)
            local["door"][index(11, 10)] = int(explosive)
            return local

        clean = controller_fixture(
            raw(), local_map=corridor())._controller_snapshot
        self.assertIsNotNone(
            DiabloGymEnv._plan_controller_path(
                clean, 13, 10, avoid_monsters=True))

        for local in (
            corridor(hazard=True),
            corridor(explosive=True),
        ):
            fixture = controller_fixture(raw(), local_map=local)
            self.assertIsNone(
                DiabloGymEnv._plan_controller_path(
                    fixture._controller_snapshot,
                    13,
                    10,
                    avoid_monsters=True,
                )
            )
            self.assertFalse(np.array_equal(
                controller_fixture(
                    raw(), local_map=corridor()
                ).controller_snapshot_vector(),
                fixture.controller_snapshot_vector(),
            ))

    def test_softwall_kind_wire_encodes_all_eight_codes_as_code_over_seven(self):
        cells = env_module.CONTROLLER_SNAPSHOT_CELLS
        center = cells // 2
        channel_start = 2 * cells
        denominator = (
            env_module.CONTROLLER_SNAPSHOT_SOFTWALL_KIND_DENOMINATOR
        )
        self.assertEqual(denominator, 7.0)

        for code in range(8):
            local = controller_map()
            local["door"][center] = code & 1
            local["closed_door"][center] = (code >> 1) & 1
            local["explosive_softwall"][center] = (code >> 2) & 1
            encoded = float(
                controller_fixture(
                    raw(), local_map=local
                ).controller_snapshot_vector()[channel_start + center]
            )
            self.assertAlmostEqual(encoded, code / denominator)
            self.assertEqual(round(encoded * denominator), code)

    def test_descend_path_never_uses_hazard_or_explosive_softwall(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        state = raw()
        radius = env._DESCEND_RADIUS
        side = 2 * radius + 1
        cells = side * side

        def index(x, y):
            return (
                (y - state["player_y"] + radius) * side
                + (x - state["player_x"] + radius)
            )

        def local_map(kind, *, detour):
            result = {
                "walkable": [0] * cells,
                "monster": [0] * cells,
                "door": [0] * cells,
                "hazard": [0] * cells,
                "explosive_softwall": [0] * cells,
            }
            for point in ((10, 10), (12, 10), (13, 10)):
                result["walkable"][index(*point)] = 1
            if kind == "hazard":
                result["walkable"][index(11, 10)] = 1
                result["hazard"][index(11, 10)] = 1
            else:
                result["door"][index(11, 10)] = 1
                result["explosive_softwall"][index(11, 10)] = 1
            if detour:
                for point in (
                    (10, 11), (11, 11), (12, 11), (13, 11)
                ):
                    result["walkable"][index(*point)] = 1
            return result

        for kind in ("hazard", "explosive"):
            with self.subTest(kind=kind), mock.patch.object(
                    env_module.bridge, "local_map",
                    return_value=local_map(kind, detour=True)):
                path = env._plan_descend_path(state, 13, 10)
            self.assertIsNotNone(path)
            self.assertEqual(path[-1][:2], (13, 10))
            self.assertNotIn((11, 10), {
                (x, y) for x, y, _softwall in path})

            with self.subTest(kind=kind, only_dangerous=True), \
                    mock.patch.object(
                        env_module.bridge, "local_map",
                        return_value=local_map(kind, detour=False)):
                self.assertIsNone(
                    env._plan_descend_path(state, 13, 10))

    def test_descend_executes_only_adjacent_frozen_path_steps(self):
        before = raw()
        before["triggers"] = [{
            "x": 14,
            "y": 10,
            "msg": env_module.bridge.WM_DIABNEXTLVL,
        }]
        path = [
            (11, 10, False),
            (12, 10, False),
            (13, 10, False),
        ]
        states = []
        for x in (11, 12, 13):
            state = dict(before)
            state.update(
                player_x=x, player_y=10, future_x=x, future_y=10)
            states.append(state)

        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = before
        env.ticks_per_step = 4
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)
        env._plan_descend_path = lambda *_args, **_kwargs: list(path)
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=1,
            ) as walk,
            mock.patch.object(
                env_module.bridge, "step", side_effect=states),
        ):
            result, beats = env._macro_descend(max_beats=3)

        self.assertIs(result, states[-1])
        self.assertEqual(beats, 3)
        self.assertEqual(
            [call.args[:2] for call in walk.call_args_list],
            [(11, 10), (12, 10), (13, 10)],
        )
        for previous, current in zip(
                [(10, 10), (11, 10), (12, 10)],
                [call.args[:2] for call in walk.call_args_list]):
            self.assertEqual(
                abs(current[0] - previous[0])
                + abs(current[1] - previous[1]),
                1,
            )

    def test_navigation_macros_do_not_consume_an_edge_before_landing(self):
        before = raw()
        before["triggers"] = [{
            "x": 14,
            "y": 10,
            "msg": env_module.bridge.WM_DIABNEXTLVL,
        }]
        in_flight = dict(before)
        in_flight.update(
            future_x=11,
            future_y=10,
            player_mode=env_module.bridge.PM_WALK_SIDEWAYS,
        )

        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = before
        env.ticks_per_step = 4
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)
        env._plan_descend_path = lambda *_args, **_kwargs: [
            (11, 10, False), (12, 10, False)]
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=1,
            ) as walk,
            mock.patch.object(
                env_module.bridge, "step",
                side_effect=[in_flight, in_flight]),
        ):
            result, beats = env._macro_descend(max_beats=2)
        self.assertIs(result, in_flight)
        self.assertEqual(beats, 2)
        self.assertEqual(
            [call.args[:2] for call in walk.call_args_list],
            [(11, 10)],
        )

        progression = {
            "kind": "vile_book",
            "action": "operate",
            "x": 12,
            "y": 10,
            "goal_x": 12,
            "goal_y": 10,
            "exact": True,
        }
        before = raw()
        before["progression_targets"] = [dict(progression)]
        in_flight = dict(before)
        in_flight.update(
            future_x=11,
            future_y=10,
            player_mode=env_module.bridge.PM_WALK_SIDEWAYS,
        )
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._raw = before
        env.ticks_per_step = 4
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)
        env._plan_descend_path = lambda *_args, **_kwargs: [
            (11, 10, False), (12, 10, False)]
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=1,
            ) as walk,
            mock.patch.object(
                env_module.bridge, "step",
                side_effect=[in_flight, in_flight]),
        ):
            result, beats = env._macro_progression(max_beats=2)
        self.assertIs(result, in_flight)
        self.assertEqual(beats, 2)
        self.assertEqual(
            [call.args[:2] for call in walk.call_args_list],
            [(11, 10)],
        )

    def test_info_raw_recursively_copies_new_nested_protocol_fields(self):
        state = raw(
            floor_items=[{
                "x": 11, "y": 10, "heal": True, "gear": False,
                "visible": True, "reachable": True,
            }],
            missiles=[missile(damage=17)],
        )
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env.include_raw = True
        env._episode_seed = 7
        info = env._info(state)
        exposed = info["raw"]
        exposed["belt_slot_kinds"][0] = 5
        exposed["equipped_items"][0]["durability"] = 99
        exposed["missiles"][0]["damage"] = 999
        exposed["floor_items"][0]["x"] = 99
        self.assertNotEqual(
            exposed["belt_slot_kinds"], state["belt_slot_kinds"])
        self.assertEqual(state["equipped_items"][0]["durability"], 0)
        self.assertEqual(state["missiles"][0]["damage"], 17)
        self.assertEqual(state["floor_items"][0]["x"], 11)

    def test_snapshot_capture_is_pure_and_non_a9_observation_does_not_prune(self):
        state = raw(monsters=[monster(5, 90)])
        env = controller_fixture(state, engage_blocked={5, 999})
        original = set(env._engage_blocked_keys)
        with mock.patch.object(
                env_module.bridge, "local_map",
                return_value=controller_map()):
            one = env._capture_controller_snapshot(state)
            two = env._capture_controller_snapshot(state)
        self.assertEqual(one, two)
        self.assertEqual(env._engage_blocked_keys, original)
        env._controller_snapshot = two
        env.controller_snapshot_vector()  # 模拟选择任一非 a9 前的重复观测
        self.assertEqual(env._engage_blocked_keys, original)

    def test_pickup_target_tie_and_execution_share_one_snapshot(self):
        # raw 顺序故意把 (11,11) 放在前；canonical snapshot 以
        # (distance,x,y) tie-break 选择 (9,9)，wire 与原生定点拾取必须一致。
        far_lexical = {
            "x": 11, "y": 11, "heal": True, "gear": False,
            "visible": True, "reachable": True,
        }
        chosen = {
            "x": 9, "y": 9, "heal": True, "gear": False,
            "visible": True, "reachable": True,
        }
        state = raw(
            floor_items=[far_lexical, chosen],
            belt_heals=1,
            belt_free_slots=7,
        )
        env = controller_fixture(state)
        snapshot = env._controller_snapshot
        self.assertEqual(
            (snapshot.heal_target.x, snapshot.heal_target.y,
             snapshot.heal_target.active_id),
            (9, 9, 1),
        )

        moved_west = dict(state)
        moved_west.update(
            player_x=9, player_y=10, future_x=9, future_y=10)
        moved_target = dict(state)
        moved_target.update(
            player_x=9, player_y=9, future_x=9, future_y=9)
        after = dict(moved_target)
        after["floor_items"] = []
        env.ticks_per_step = 4
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)
        with (
            mock.patch.object(
                env_module.bridge, "act_pickup_at",
                return_value=1, create=True) as pickup,
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=1,
            ) as walk,
            mock.patch.object(
                env_module.bridge, "step",
                side_effect=[moved_west, moved_target, after]),
        ):
            result, beats = env._macro_pickup(
                "heal", max_beats=12,
                controller_snapshot=snapshot)
        self.assertIs(result, after)
        self.assertEqual(beats, 3)
        self.assertEqual(
            [call.args[:2] for call in walk.call_args_list],
            [(9, 10), (9, 9)],
        )
        pickup.assert_called_once_with(1, 9, 9, 0, 2, 0, 0)

    def test_atomic_gear_pickup_never_bypasses_snapshot_hazard_or_wall(self):
        state = raw(floor_items=[{
            "x": 11, "y": 10,
            "heal": False, "gear": True,
            "visible": True, "reachable": True,
        }])
        env = controller_fixture(state)
        clean = env._controller_snapshot
        self.assertIsNotNone(clean.gear_target)

        radius = env_module.CONTROLLER_SNAPSHOT_RADIUS
        side = env_module.CONTROLLER_SNAPSHOT_SIDE
        target_index = radius * side + radius + 1
        waited = object()
        env._wait_step = lambda: (waited, 1)

        unsafe_snapshots = []
        for field in ("hazard", "explosive_softwall"):
            values = list(getattr(clean, field))
            values[target_index] = 1
            unsafe_snapshots.append(replace(clean, **{field: tuple(values)}))
        walkable = list(clean.walkable)
        walkable[target_index] = 0
        unsafe_snapshots.append(replace(
            clean,
            walkable=tuple(walkable),
            softwall=tuple(0 for _ in clean.softwall),
        ))

        with mock.patch.object(
                env_module.bridge, "act_pickup_gear_at",
                return_value=1, create=True) as pickup:
            for snapshot in unsafe_snapshots:
                result, beats = env._macro_pickup(
                    "gear", max_beats=12,
                    controller_snapshot=snapshot)
                self.assertIs(result, waited)
                self.assertEqual(beats, 1)
        pickup.assert_not_called()

    def test_atomic_gear_pickup_on_player_tile_commits_only_when_tile_safe(self):
        state = raw(floor_items=[{
            "x": 10, "y": 10,
            "heal": False, "gear": True,
            "visible": True, "reachable": True,
        }])
        env = controller_fixture(state)
        snapshot = env._controller_snapshot
        self.assertIsNotNone(snapshot.gear_target)
        # A path planner correctly reports no movement path when the target is
        # the start tile.  That must still be a complete, safe gear path.
        self.assertIsNone(DiabloGymEnv._plan_controller_path(
            snapshot, 10, 10, avoid_monsters=True))

        after = dict(state)
        after["floor_items"] = []
        # Native commit publishes the causal utility increase synchronously.
        # A later endpoint may fall again (for example durability loss);
        # that must not erase the accepted receipt.
        committed = dict(after)
        committed["gear_combat_utility"] = 37
        action14_audit = {
            "accepted": False,
            "commit_attempts": 0,
            "utility_before": 0,
            "utility_after": 0,
            "utility_delta": 0,
        }
        env.ticks_per_step = 4
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)
        with (
            mock.patch.object(
                env_module.bridge, "act_pickup_gear_at",
                return_value=1, create=True) as pickup,
            mock.patch.object(
                env_module.bridge, "observe",
                return_value=committed),
            mock.patch.object(
                env_module.bridge, "step", return_value=after),
        ):
            result, beats = env._macro_pickup(
                "gear", max_beats=12,
                controller_snapshot=snapshot,
                action14_audit=action14_audit)
        self.assertIs(result, after)
        self.assertEqual(beats, 1)
        pickup.assert_called_once_with(0, 10, 10, 0, 1, 0, 0)
        self.assertEqual(
            action14_audit,
            {
                "accepted": True,
                "commit_attempts": 1,
                "utility_before": 0,
                "utility_after": 37,
                "utility_delta": 37,
            },
        )

        center = env_module.CONTROLLER_SNAPSHOT_CELLS // 2
        waited = object()
        env._wait_step = lambda: (waited, 1)
        for field in ("hazard", "explosive_softwall"):
            channel = list(getattr(snapshot, field))
            channel[center] = 1
            unsafe = replace(snapshot, **{field: tuple(channel)})
            with mock.patch.object(
                    env_module.bridge, "act_pickup_gear_at",
                    return_value=1, create=True) as pickup:
                result, beats = env._macro_pickup(
                    "gear", max_beats=12,
                    controller_snapshot=unsafe)
            self.assertIs(result, waited)
            self.assertEqual(beats, 1)
            pickup.assert_not_called()

    def test_action14_endpoint_utility_change_without_receipt_is_not_success(self):
        state = raw(floor_items=[{
            "x": 10, "y": 10,
            "heal": False, "gear": True,
            "visible": True, "reachable": True,
        }])
        env = controller_fixture(state)
        endpoint = dict(state)
        endpoint["floor_items"] = [dict(state["floor_items"][0])]
        endpoint["gear_combat_utility"] = 99
        audit = {
            "accepted": False,
            "commit_attempts": 0,
            "utility_before": 0,
            "utility_after": 0,
            "utility_delta": 0,
        }
        env.ticks_per_step = 4
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)

        with (
            mock.patch.object(
                env_module.bridge, "act_pickup_gear_at",
                return_value=0, create=True),
            mock.patch.object(
                env_module.bridge, "observe") as observe,
            mock.patch.object(
                env_module.bridge, "step", return_value=endpoint),
        ):
            result, beats = env._macro_pickup(
                "gear",
                max_beats=1,
                controller_snapshot=env._controller_snapshot,
                action14_audit=audit,
            )

        self.assertIs(result, endpoint)
        self.assertEqual(beats, 1)
        observe.assert_not_called()
        self.assertEqual(
            audit,
            {
                "accepted": False,
                "commit_attempts": 1,
                "utility_before": 0,
                "utility_after": 0,
                "utility_delta": 0,
            },
        )

    def test_step_publishes_synchronous_action14_audit_at_endpoint(self):
        state = raw(floor_items=[{
            "x": 10, "y": 10,
            "heal": False, "gear": True,
            "visible": True, "reachable": True,
        }])
        state["gear_combat_utility"] = 100
        endpoint = dict(state)
        endpoint["floor_items"] = []
        # The endpoint has already lost the synchronous gain.  The audit
        # passed by the macro remains the causal source of truth.
        endpoint["gear_combat_utility"] = 63
        env = controller_fixture(state)
        env.action_space = env_module.gym.spaces.Discrete(15)
        env.max_steps = 100
        env.ticks_per_step = 4
        env.start_in_dungeon = False
        env._controller_snapshot_enabled = False
        env._episode_ended = False
        env._ep_kills = 0
        env._settle_to_idle = lambda current, beats, **_kwargs: (
            current, beats)
        env._reward = lambda *_args, **_kwargs: 0.0
        env._info = lambda _state: {}
        env._vectorize = lambda _state: np.zeros(295, dtype=np.float32)

        def pickup(_kind, *, action14_audit, **_kwargs):
            action14_audit.update({
                "accepted": True,
                "commit_attempts": 1,
                "utility_before": 100,
                "utility_after": 137,
                "utility_delta": 37,
            })
            return endpoint, 1

        env._macro_pickup = pickup
        with mock.patch.object(
                env_module.bridge,
                "local_map",
                return_value=controller_map()):
            _obs, _reward, terminated, truncated, info = env.step(14)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            info["action14_audit"],
            {
                "accepted": True,
                "commit_attempts": 1,
                "utility_before": 100,
                "utility_after": 137,
                "utility_delta": 37,
            },
        )

    def test_pickup_slot_reuse_cannot_commit_unobserved_replacement(self):
        state = raw(floor_items=[{
            "x": 11, "y": 10,
            "heal": True, "gear": False,
            "visible": True, "reachable": True,
            "seed_hi": 0x1111, "seed_lo": 0x2222,
            "create_info": 0x3333, "base_id": 4,
        }])
        env = controller_fixture(state)
        replacement = raw(floor_items=[{
            "active_id": 0,
            "x": 11, "y": 10,
            "heal": True, "gear": False,
            "visible": True, "reachable": True,
            "seed_hi": 0xAAAA, "seed_lo": 0xBBBB,
            "create_info": 0xCCCC, "base_id": 5,
        }])
        replacement.update(
            player_x=11, player_y=10, future_x=11, future_y=10)
        env.ticks_per_step = 4
        env._record_visit = lambda _position: False
        env._finish_macro = lambda final_raw, beats, _scene: (
            final_raw, beats)
        with (
            mock.patch.object(
                env_module.bridge, "act_explore_walk",
                return_value=1,
            ),
            mock.patch.object(
                env_module.bridge, "act_pickup_at",
                return_value=1,
            ) as pickup,
            mock.patch.object(
                env_module.bridge, "step", return_value=replacement),
        ):
            result, beats = env._macro_pickup(
                "heal",
                max_beats=12,
                controller_snapshot=env._controller_snapshot,
            )
        self.assertIs(result, replacement)
        self.assertEqual(beats, 1)
        pickup.assert_not_called()


class ConservedDamageRewardTests(unittest.TestCase):
    @staticmethod
    def env_with_ledger(state: dict) -> DiabloGymEnv:
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._combat_hp_floor = {}
        env._reset_combat_ledger(state)
        return env

    def test_heal_and_redamage_same_hp_interval_is_not_paid_twice(self):
        full = raw(monsters=[monster(1, 100)])
        half = raw(monsters=[monster(1, 50)])
        env = self.env_with_ledger(full)

        first = env._combat_reward(full, half)
        healed = env._combat_reward(half, full)
        repeated = env._combat_reward(full, half)
        generation_key = env._monster_generation_key(full["monsters"][0])

        self.assertAlmostEqual(first, 0.28125)
        self.assertEqual(healed, 0.0)
        self.assertEqual(repeated, 0.0)
        self.assertEqual(env._combat_hp_floor[generation_key][0], 50)

    def test_damage_shaping_is_bounded_and_kill_is_paid_once(self):
        states = [
            raw(monsters=[monster(1, hp)])
            for hp in (100, 75, 100, 75, 50, 100, 25)
        ]
        dead = raw(monsters=[])
        dead["monster_kill_total"] = 1
        env = self.env_with_ledger(states[0])
        total = 0.0
        for before, after in zip(states, states[1:]):
            total += env._combat_reward(before, after)
        total += env._combat_reward(states[-1], dead)

        damage_component = total - 1.0
        self.assertGreater(damage_component, 0.0)
        self.assertLessEqual(damage_component, 0.75)
        self.assertEqual(env._combat_hp_floor, {})
        self.assertEqual(env._combat_reward(dead, dead), 0.0)

    def test_new_or_reused_id_starts_a_fresh_unpaid_baseline(self):
        empty = raw(monsters=[])
        spawned_damaged = raw(monsters=[
            monster(7, 20, generation_seed=0x11111111)])
        env = self.env_with_ledger(empty)
        self.assertEqual(env._combat_reward(empty, spawned_damaged), 0.0)
        first_key = env._monster_generation_key(
            spawned_damaged["monsters"][0])
        self.assertEqual(env._combat_hp_floor[first_key][0], 20)

        dead = raw(monsters=[])
        dead["monster_kill_total"] = 1
        self.assertGreater(env._combat_reward(spawned_damaged, dead), 1.0)
        reused = raw(monsters=[
            monster(7, 30, generation_seed=0x22222222)])
        reused["monster_kill_total"] = 1
        self.assertEqual(env._combat_reward(dead, reused), 0.0)
        reused_key = env._monster_generation_key(reused["monsters"][0])
        self.assertNotEqual(first_key, reused_key)
        self.assertEqual(env._combat_hp_floor[reused_key][0], 30)

    def test_same_step_slot_reuse_closes_old_lifetime_and_baselines_new(self):
        old = raw(monsters=[
            monster(7, 20, generation_seed=0x11111111)])
        replacement = raw(monsters=[
            monster(7, 70, generation_seed=0x22222222)])
        replacement["monster_kill_total"] = 1
        env = self.env_with_ledger(old)

        reward = env._combat_reward(old, replacement)
        old_key = env._monster_generation_key(old["monsters"][0])
        new_key = env._monster_generation_key(
            replacement["monsters"][0])

        # Remaining 20/100 damage receives 0.145 from the conserved potential,
        # then exactly one terminal kill unit.  The replacement starts at an
        # unpaid 70 HP baseline despite reusing slot 7.
        self.assertAlmostEqual(reward, 1.145)
        self.assertNotIn(old_key, env._combat_hp_floor)
        self.assertEqual(env._combat_hp_floor, {new_key: (70, 100)})
        self.assertEqual(
            env._disappeared_monster_generations(old, replacement), 1)
        self.assertEqual(
            env._disappeared_monster_generations(
                replacement,
                raw(monsters=[
                    monster(7, 50, generation_seed=0x22222222)]),
            ),
            0,
        )

    def test_damage_potential_is_invariant_to_transition_partition(self):
        full = raw(monsters=[monster(1, 100)])
        half = raw(monsters=[monster(1, 50)])
        dead = raw(monsters=[])
        dead["monster_kill_total"] = 1

        one_step = self.env_with_ledger(full)._combat_reward(full, dead)
        split_env = self.env_with_ledger(full)
        split = (
            split_env._combat_reward(full, half)
            + split_env._combat_reward(half, dead)
        )
        self.assertAlmostEqual(one_step, split)
        self.assertAlmostEqual(one_step, 1.625)

    def test_native_kill_delta_counts_spawn_and_death_between_endpoints(self):
        before = raw(monsters=[])
        after = raw(monsters=[])
        after["monster_kill_total"] = 1
        env = self.env_with_ledger(before)

        # No endpoint generation exists to diff, yet the native death event
        # still supplies exactly one terminal reward unit.
        self.assertEqual(
            env._disappeared_monster_generations(before, after), 0)
        self.assertEqual(env._combat_reward(before, after), 1.0)

    def test_successful_stationary_combat_is_never_taxed_as_wall_bumping(self):
        before = raw(monsters=[monster(1, 500, max_hp=500)])
        after = raw(monsters=[monster(1, 499, max_hp=500)])
        env = self.env_with_ledger(before)
        env.descend_ladder = True
        env.death_ladder = True

        reward = env._reward(before, after, requested_action=9)
        self.assertGreater(reward, 0.0)
        self.assertLess(reward, 0.002)

    def test_approach_shaping_tracks_player_not_monster_motion(self):
        target = monster(1, 100)
        target.update(x=12, y=10, future_x=12, future_y=10)
        before = raw(monsters=[target])

        retreating = dict(target)
        retreating.update(x=14, y=10, future_x=14, future_y=10)
        moved_toward = raw(monsters=[retreating])
        moved_toward.update(
            player_x=11, player_y=10, future_x=11, future_y=10)
        self.assertEqual(
            DiabloGymEnv._player_approach_delta(
                before, moved_toward),
            1,
        )
        toward_env = self.env_with_ledger(before)
        toward_env.descend_ladder = True
        toward_env.death_ladder = True
        self.assertAlmostEqual(
            toward_env._reward(
                before, moved_toward, requested_action=3),
            0.005,
        )

        approaching = dict(target)
        approaching.update(x=11, y=10, future_x=11, future_y=10)
        moved_away = raw(monsters=[approaching])
        moved_away.update(
            player_x=9, player_y=10, future_x=9, future_y=10)
        self.assertEqual(
            DiabloGymEnv._player_approach_delta(before, moved_away),
            -1,
        )
        away_env = self.env_with_ledger(before)
        away_env.descend_ladder = True
        away_env.death_ladder = True
        self.assertAlmostEqual(
            away_env._reward(
                before, moved_away, requested_action=7),
            -0.005,
        )

    def test_approach_shaping_does_not_switch_when_fixed_target_disappears(self):
        target = monster(1, 100, generation_seed=0x11111111)
        target.update(x=12, y=10, future_x=12, future_y=10)
        survivor = monster(2, 100, generation_seed=0x22222222)
        survivor.update(x=18, y=10, future_x=18, future_y=10)
        before = raw(monsters=[target, survivor])

        after = raw(monsters=[survivor])
        after.update(
            player_x=11, player_y=10, future_x=11, future_y=10)

        self.assertIsNone(
            DiabloGymEnv._player_approach_delta(before, after))

    def test_approach_shaping_rejects_reused_slot_new_generation(self):
        target = monster(7, 100, generation_seed=0x11111111)
        target.update(x=12, y=10, future_x=12, future_y=10)
        before = raw(monsters=[target])

        replacement = monster(7, 100, generation_seed=0x22222222)
        replacement.update(x=12, y=10, future_x=12, future_y=10)
        after = raw(monsters=[replacement])
        after.update(
            player_x=11, player_y=10, future_x=11, future_y=10)

        self.assertIsNone(
            DiabloGymEnv._player_approach_delta(before, after))

    def test_action9_approach_uses_canonical_target_not_tile_decoy(self):
        # The decoy is closest by current tile but is moving west.  Action 9
        # ranks the fixed controller candidates by future tile, where target
        # wins the stable id tie and lies east.  Re-selecting in reward would
        # therefore punish the exact eastward move the macro requested.
        decoy = monster(1, 100, generation_seed=0x11111111)
        decoy.update(x=9, y=10, future_x=8, future_y=10)
        target = monster(0, 100, generation_seed=0x22222222)
        target.update(x=13, y=10, future_x=12, future_y=10)
        before = raw(monsters=[decoy, target])
        fixture = controller_fixture(before)
        candidate = DiabloGymEnv._canonical_engage_candidate(
            fixture._controller_snapshot)
        self.assertIsNotNone(candidate)
        target_key = DiabloGymEnv._monster_generation_key(target)
        self.assertEqual(candidate.generation_key, target_key)

        moved_decoy = dict(decoy)
        moved_decoy.update(x=8, y=10, future_x=8, future_y=10)
        moved_target = dict(target)
        moved_target.update(x=12, y=10, future_x=12, future_y=10)
        after = raw(monsters=[moved_decoy, moved_target])
        after.update(
            player_x=11, player_y=10, future_x=11, future_y=10)

        self.assertEqual(
            DiabloGymEnv._player_approach_delta(before, after),
            -1,
        )
        self.assertEqual(
            DiabloGymEnv._player_approach_delta(
                before,
                after,
                target_generation_key=target_key,
            ),
            1,
        )
        reward_env = self.env_with_ledger(before)
        reward_env.descend_ladder = True
        reward_env.death_ladder = True
        self.assertAlmostEqual(
            reward_env._reward(
                before,
                after,
                requested_action=9,
                engage_target_generation_key=target_key,
            ),
            0.005,
        )

    def test_action9_without_candidate_cannot_fallback_to_nearest_monster(self):
        target = monster(1, 100)
        target.update(x=12, y=10, future_x=12, future_y=10)
        before = raw(monsters=[target])
        after = raw(monsters=[target])
        after.update(
            player_x=11, player_y=10, future_x=11, future_y=10)
        self.assertEqual(
            DiabloGymEnv._player_approach_delta(before, after),
            1,
        )

        env = self.env_with_ledger(before)
        env.descend_ladder = True
        env.death_ladder = True
        self.assertEqual(
            env._reward(
                before,
                after,
                requested_action=9,
                engage_target_generation_key=None,
            ),
            0.0,
        )

    def test_step_passes_canonical_action9_generation_to_reward(self):
        decoy = monster(1, 100, generation_seed=0x11111111)
        decoy.update(x=9, y=10, future_x=8, future_y=10)
        target = monster(0, 100, generation_seed=0x22222222)
        target.update(x=13, y=10, future_x=12, future_y=10)
        before = raw(monsters=[decoy, target])
        after = dict(before)
        after["monsters"] = [dict(decoy), dict(target)]
        target_key = DiabloGymEnv._monster_generation_key(target)

        env = controller_fixture(before)
        env.action_space = env_module.gym.spaces.Discrete(15)
        env.max_steps = 100
        env.ticks_per_step = 4
        env.start_in_dungeon = False
        env._controller_snapshot_enabled = False
        env._episode_ended = False
        env._ep_kills = 0
        env._settle_to_idle = lambda state, beats, **_kwargs: (
            state, beats)
        env._info = lambda _state: {}
        env._vectorize = lambda _state: np.zeros(295, dtype=np.float32)
        reward = mock.Mock(return_value=0.0)
        env._reward = reward
        env._macro_engage = mock.Mock(return_value=(after, 1))

        with mock.patch.object(
                env_module.bridge,
                "local_map",
                return_value=controller_map()):
            _obs, _reward, terminated, truncated, _info = env.step(9)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        reward.assert_called_once_with(
            before,
            after,
            requested_action=9,
            engage_target_generation_key=target_key,
            action_executed=False,
            action14_utility_delta=None,
        )

    def test_wait_cannot_claim_a_committed_previous_move(self):
        """ActWait 可让旧单格收尾，但该位移不能记成当前 a0 的塑形收益。"""
        before = raw(monsters=[monster(1, 100)])
        after = raw(monsters=[monster(1, 100)])
        after.update(player_x=11, player_y=10)

        def reward(action, *, xp=0):
            env = self.env_with_ledger(before)
            env.descend_ladder = True
            env.death_ladder = True
            current = dict(after)
            current["monsters"] = [dict(m) for m in after["monsters"]]
            current["xp"] = xp
            return env._reward(
                before, current, requested_action=action)

        self.assertAlmostEqual(reward(1), 0.005)
        self.assertAlmostEqual(reward(0), -0.002)
        # 动作归因只剥离接近塑形；真实 XP 等环境账仍须保留。
        self.assertAlmostEqual(reward(0, xp=1), 0.008)

    def test_gear_shaping_uses_shared_utility_and_is_positive_bounded(self):
        previous = raw()
        current = raw()
        previous["gear_combat_utility"] = 100
        current["gear_combat_utility"] = 2148
        self.assertAlmostEqual(
            gear_upgrade_reward_component(previous, current), 0.5)

        current["gear_combat_utility"] = 100 + 100_000
        self.assertEqual(
            gear_upgrade_reward_component(previous, current),
            GEAR_COMBAT_UTILITY_REWARD_CAP,
        )
        previous["gear_combat_utility"] = 500
        current["gear_combat_utility"] = 100
        self.assertEqual(
            gear_upgrade_reward_component(previous, current), 0.0)

        # AC alone is no longer a second, inconsistent source of equipment
        # reward; only the native comparator ledger can credit an upgrade.
        previous = raw()
        current = raw()
        current["armor_class"] = 50
        self.assertEqual(
            gear_upgrade_reward_component(previous, current), 0.0)

        del current["gear_combat_utility"]
        with self.assertRaisesRegex(RuntimeError, "gear_combat_utility"):
            gear_upgrade_reward_component(previous, current)


class TerminalDeathRewardSourceTests(unittest.TestCase):
    def test_shared_pure_component_has_one_immutable_schedule(self):
        self.assertEqual(TERMINAL_DEATH_REWARD_SPEC.flat_cost, 2.0)
        self.assertEqual(
            TERMINAL_DEATH_REWARD_SPEC.ladder_cost_per_depth, 8.0)
        self.assertEqual(
            terminal_death_reward_component(
                dead=False, dungeon_level=999, death_ladder=True),
            0.0,
        )
        self.assertEqual(
            terminal_death_reward_component(
                dead=True, dungeon_level=3, death_ladder=False),
            -2.0,
        )
        self.assertEqual(
            terminal_death_reward_component(
                dead=True, dungeon_level=3, death_ladder=True),
            -24.0,
        )
        for invalid_depth in (-1, 1.5, True, float("nan")):
            with self.assertRaises(ValueError):
                terminal_death_reward_component(
                    dead=True,
                    dungeon_level=invalid_depth,
                    death_ladder=True,
                )

    def test_native_reward_delegates_terminal_component_to_shared_function(self):
        before = raw()
        after = raw()
        after["dead"] = True
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._combat_hp_floor = {}
        env._reset_combat_ledger(before)
        env.descend_ladder = False
        env.death_ladder = True

        with mock.patch.object(
                env_module,
                "terminal_death_reward_component",
                return_value=-37.0) as shared:
            reward = env._reward(before, after, requested_action=0)

        shared.assert_called_once_with(
            dead=True, dungeon_level=1, death_ladder=True)
        # The unrelated stationary-action component remains separate.
        self.assertAlmostEqual(reward, -37.002)


if __name__ == "__main__":
    unittest.main()
