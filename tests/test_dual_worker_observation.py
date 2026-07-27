"""Regression tests for the lossless dual Worker observation boundary."""
from __future__ import annotations

from dataclasses import replace
import pathlib
import sys
import types
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import bridge
from diablogym.controller_wire import (
    _banned_policy_wire_violations,
    _layout_sha256,
    _validate_layout_spec,
    DUAL_WORKER_LAYOUT_FROZEN_SHA256,
)
from diablogym.env import CONTROLLER_SNAPSHOT_VECTOR_DIM
from diablogym.options_env import (
    DUAL_WORKER_ACTION_MASK_SLICE,
    DUAL_WORKER_CONTROLLER_BELT_SLICE,
    DUAL_WORKER_CONTROLLER_COMBAT_SLICE,
    DUAL_WORKER_CONTROLLER_EXACT_SLICE,
    DUAL_WORKER_CONTROLLER_ITEM_TARGET_SLICE,
    DUAL_WORKER_CONTROLLER_MAP_SLICE,
    DUAL_WORKER_CONTROLLER_MISSILE_SLICE,
    DUAL_WORKER_CONTROLLER_MONSTER_SLICE,
    DUAL_WORKER_CONTROLLER_STICKY_SLICE,
    DUAL_WORKER_CURRENT_EXHAUSTED_FEATURE,
    DUAL_WORKER_CURRENT_LAYER_CLOCK_FEATURE,
    DUAL_WORKER_CURRENT_V4_BASE_SLICE,
    DUAL_WORKER_DRINK_LATCH_FEATURE,
    DUAL_WORKER_DRY_FLOOR_REMAINING_FEATURE,
    DUAL_WORKER_FARM_SCENE_FRACTION_FEATURE,
    DUAL_WORKER_FUSE_STREAK_SLICE,
    DUAL_WORKER_LAYER_KILLS_FEATURE,
    DUAL_WORKER_LEGACY_LAYER_TIME_FEATURE,
    DUAL_WORKER_LEGACY_SLICE,
    DUAL_WORKER_LAYOUT,
    DUAL_WORKER_LAYOUT_SHA256,
    DUAL_WORKER_MANAGER_MASK_SLICE,
    DUAL_WORKER_OBSERVATION_DIM,
    DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
    DUAL_WORKER_TIME_REMAINING_FEATURE,
    FARM_SCENE_CAP,
    KILL_PATIENCE,
    OptionsEnv,
    REVISIT_FLOOR,
    TAU_CAP,
    WORKER_OBSERVATION_VIEW_A12_OVERLAY,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
    WORKER_OBSERVATION_VIEW_LEGACY_V3,
    WORKER_OBSERVATION_VIEW_RAW_V4,
    WORKER_ACTION12_ENVIRONMENT_MASK,
    WORKER_ACTION12_PERMANENTLY_MASKED,
    _resolve_worker_drink_sovereignty,
)
from diablogym.worker_env import (
    WorkerWindowEnv,
    legacy_worker_policy_observation_view,
)


class _FakeBaseEnv:
    def __init__(self, raw: dict, current_base: np.ndarray,
                 legacy_base: np.ndarray):
        self._raw = raw
        self._steps = 100
        self._ep_kills = 13
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(295,), dtype=np.float32)
        self.current_base = current_base
        self.legacy_base = legacy_base
        self.base_mask = np.ones(15, dtype=bool)
        self.controller_vector = np.zeros(
            CONTROLLER_SNAPSHOT_VECTOR_DIM, dtype=np.float32)

    def _ensure_active(self, **_kwargs):
        return None

    def action_masks(self):
        return self.base_mask.copy()

    def _legacy_policy_vectorize(self, raw):
        if "legacy_belt_heals" not in raw:
            raise RuntimeError("missing legacy_belt_heals")
        if any("legacy_heal" not in item
               for item in raw.get("floor_items", ())):
            raise RuntimeError("missing legacy_heal")
        return self.legacy_base.copy()

    def controller_snapshot_vector(self):
        return self.controller_vector.copy()


def _raw_state() -> dict:
    return {
        "hp": 60,
        "max_hp": 100,
        "mana": 20,
        "max_mana": 40,
        "xp": 123,
        "gold": 17,
        "char_level": 4,
        "dungeon_level": 2,
        "is_set_level": False,
        "set_level_id": 0,
        "player_x": 10,
        "player_y": 11,
        "future_x": 10,
        "future_y": 11,
        "armor_class": 7,
        "gear_combat_utility": 0,
        "belt_heals": 2,
        "legacy_belt_heals": 2,
        "belt_free_slots": 4,
        "monsters": [{
            "id": 5,
            "x": 12,
            "y": 11,
            "hp": 9,
            "max_hp": 10,
            "visible": True,
            "reachable": True,
            "rnd_item_seed_hi": 0,
            "rnd_item_seed_lo": 6,
        }],
        "floor_items": [{
            "active_id": 8,
            "x": 9,
            "y": 11,
            "combat_utility_hi": 0,
            "combat_utility_lo": 0,
            "heal": True,
            "legacy_heal": True,
            "gear": False,
            "visible": True,
            "reachable": True,
        }],
        "progression_targets": [],
        "triggers": [{
            "msg": bridge.WM_DIABNEXTLVL,
            "x": 30,
            "y": 30,
        }],
    }


def _options_fixture() -> tuple[OptionsEnv, np.ndarray, np.ndarray]:
    raw = _raw_state()
    current = (
        np.arange(295, dtype=np.float32) / np.float32(1000.0))
    legacy = (
        np.arange(295, dtype=np.float32) / np.float32(500.0))
    legacy[286] = np.float32(2 / 8)
    base = _FakeBaseEnv(raw, current, legacy)

    options = OptionsEnv.__new__(OptionsEnv)
    options.env = base
    options.max_steps = 1000
    options.drink_sovereignty = True
    options._last_base_obs = current.copy()
    options._legacy_layer_clock = 28
    options._legacy_exhausted = True
    options._legacy_layer_steps0 = 25
    options.layer_clock = 70
    options.exhausted = False
    options.farm_scene_steps = FARM_SCENE_CAP // 2
    options._farm_scene = (2, False, 0)
    options._layer_kills0 = 3
    options._layer_steps0 = 25
    options._last_tau = 0
    options._last_opt = -1
    options._win = {
        "t0": 90,
        "floor": REVISIT_FLOOR,
        "voluntary_drinks": 1,
        "drains": 0,
    }
    options._fuse_sig = options._sig(9, raw)
    options._fuse = 7
    return options, current, legacy


class DualWorkerObservationTests(unittest.TestCase):
    def test_action12_contract_is_single_source_for_embedded_and_live_masks(self):
        def callback(_observation, _mask):
            return 9

        options, _current, _legacy = _options_fixture()
        options._win["voluntary_drinks"] = 0
        callback.diablogym_worker_observation_view = (
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)

        callback.diablogym_worker_action12_mode = (
            WORKER_ACTION12_ENVIRONMENT_MASK)
        options.drink_sovereignty = True
        options._validate_worker_action12_contract(
            callback, callback.diablogym_worker_observation_view)
        live = options._worker_masks()
        observation = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
        self.assertTrue(live[12])
        self.assertEqual(
            bool(observation[DUAL_WORKER_ACTION_MASK_SLICE][12]),
            bool(live[12]))

        callback.diablogym_worker_action12_mode = (
            WORKER_ACTION12_PERMANENTLY_MASKED)
        options.drink_sovereignty = False
        options._validate_worker_action12_contract(
            callback, callback.diablogym_worker_observation_view)
        live = options._worker_masks()
        observation = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
        self.assertFalse(live[12])
        self.assertEqual(
            bool(observation[DUAL_WORKER_ACTION_MASK_SLICE][12]),
            bool(live[12]))

        options.drink_sovereignty = True
        with self.assertRaisesRegex(RuntimeError, "不一致"):
            options._validate_worker_action12_contract(
                callback, callback.diablogym_worker_observation_view)

    def test_action12_contract_resolution_rejects_split_brain(self):
        def permanent(_observation, _mask):
            return 9

        permanent.diablogym_worker_action12_mode = (
            WORKER_ACTION12_PERMANENTLY_MASKED)

        def environment(_observation, _mask):
            return 9

        environment.diablogym_worker_action12_mode = (
            WORKER_ACTION12_ENVIRONMENT_MASK)

        self.assertFalse(
            _resolve_worker_drink_sovereignty({0: permanent}, None))
        self.assertTrue(
            _resolve_worker_drink_sovereignty({0: environment}, None))
        with self.assertRaisesRegex(ValueError, "不一致"):
            _resolve_worker_drink_sovereignty({0: permanent}, True)
        with self.assertRaisesRegex(ValueError, "多个 Worker"):
            _resolve_worker_drink_sovereignty(
                {0: permanent, 1: environment}, None)
        with self.assertRaisesRegex(ValueError, "无 contract"):
            _resolve_worker_drink_sovereignty(
                {0: permanent, 1: lambda *_args: 9}, None)

    def test_layout_slices_masks_context_and_real_fuse_streak(self):
        options, current, legacy = _options_fixture()
        options.env.controller_vector = np.linspace(
            0.0,
            1.0,
            CONTROLLER_SNAPSHOT_VECTOR_DIM,
            dtype=np.float32,
        )
        worker_mask = options._worker_masks()
        manager_mask = options.action_masks()

        observation = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
            skip_dry_probability=0.375,
        )
        self.assertEqual(
            observation.shape, (DUAL_WORKER_OBSERVATION_DIM,))
        self.assertEqual(DUAL_WORKER_OBSERVATION_DIM, 13012)
        self.assertEqual(CONTROLLER_SNAPSHOT_VECTOR_DIM, 12377)
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.isfinite(observation).all())
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_MAP_SLICE.start, 635)
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_ITEM_TARGET_SLICE.stop,
            DUAL_WORKER_OBSERVATION_DIM,
        )
        self.assertEqual(
            (
                DUAL_WORKER_CONTROLLER_MAP_SLICE.start,
                DUAL_WORKER_CONTROLLER_MAP_SLICE.stop,
                DUAL_WORKER_CONTROLLER_MONSTER_SLICE.stop,
                DUAL_WORKER_CONTROLLER_MISSILE_SLICE.stop,
                DUAL_WORKER_CONTROLLER_BELT_SLICE.stop,
                DUAL_WORKER_CONTROLLER_EXACT_SLICE.stop,
                DUAL_WORKER_CONTROLLER_COMBAT_SLICE.stop,
                DUAL_WORKER_CONTROLLER_STICKY_SLICE.stop,
                DUAL_WORKER_CONTROLLER_ITEM_TARGET_SLICE.stop,
            ),
            (
                635, 5010, 10522, 12227, 12275,
                12296, 12922, 12925, 13012,
            ),
        )
        cursor = 0
        for segment in DUAL_WORKER_LAYOUT.segments:
            self.assertEqual(segment.start, cursor, segment.name)
            self.assertGreater(segment.stop, segment.start, segment.name)
            self.assertTrue(segment.field_names, segment.name)
            self.assertTrue(segment.shape, segment.name)
            self.assertTrue(segment.semantic_tags, segment.name)
            cursor = segment.stop
        self.assertEqual(cursor, DUAL_WORKER_OBSERVATION_DIM)
        self.assertEqual(
            DUAL_WORKER_LAYOUT.schema,
            "diablogym-dual-worker-layout/4",
        )
        map_segment = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "controller_map"
        )
        self.assertEqual(
            map_segment.field_encodings[2],
            "normalized_bitpack_div_7:"
            "softwall+2*closed_door+4*explosive_softwall",
        )
        self.assertEqual(
            DUAL_WORKER_LAYOUT.p_skip_semantic_index,
            DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
        )
        self.assertEqual(
            DUAL_WORKER_LAYOUT.banned_rng_tag_violations, ())
        self.assertEqual(
            DUAL_WORKER_LAYOUT_SHA256,
            "6463990a5732c366f19ae460576f74fc186cf4ad5ddb3015e704140f4c2586c9",
        )
        self.assertEqual(
            DUAL_WORKER_LAYOUT_SHA256,
            DUAL_WORKER_LAYOUT_FROZEN_SHA256,
        )
        combat_segment = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "controller_combat"
        )
        self.assertEqual(
            combat_segment.row_offset,
            len(combat_segment.prefix_field_names),
        )
        self.assertEqual(
            combat_segment.width,
            combat_segment.row_offset
            + combat_segment.row_count * combat_segment.row_width,
        )
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_MONSTER_SLICE.start,
            DUAL_WORKER_CONTROLLER_MAP_SLICE.stop,
        )
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_MISSILE_SLICE.start,
            DUAL_WORKER_CONTROLLER_MONSTER_SLICE.stop,
        )
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_BELT_SLICE.start,
            DUAL_WORKER_CONTROLLER_MISSILE_SLICE.stop,
        )

        self.assertEqual(
            DUAL_WORKER_CONTROLLER_EXACT_SLICE.start,
            DUAL_WORKER_CONTROLLER_BELT_SLICE.stop,
        )
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_COMBAT_SLICE.start,
            DUAL_WORKER_CONTROLLER_EXACT_SLICE.stop,
        )
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_STICKY_SLICE.start,
            DUAL_WORKER_CONTROLLER_COMBAT_SLICE.stop,
        )
        self.assertEqual(
            DUAL_WORKER_CONTROLLER_ITEM_TARGET_SLICE.start,
            DUAL_WORKER_CONTROLLER_STICKY_SLICE.stop,
        )

        expected_legacy = np.concatenate([
            legacy,
            np.asarray([
                10 / TAU_CAP,
                28 / KILL_PATIENCE,
                1.0,
            ], dtype=np.float32),
        ])
        np.testing.assert_array_equal(
            observation[DUAL_WORKER_LEGACY_SLICE],
            expected_legacy,
        )
        np.testing.assert_array_equal(
            observation[DUAL_WORKER_CURRENT_V4_BASE_SLICE],
            current,
        )
        self.assertAlmostEqual(
            float(observation[
                DUAL_WORKER_CURRENT_LAYER_CLOCK_FEATURE]),
            70 / KILL_PATIENCE,
        )
        self.assertEqual(
            float(observation[DUAL_WORKER_CURRENT_EXHAUSTED_FEATURE]),
            0.0,
        )
        self.assertAlmostEqual(
            float(observation[
                DUAL_WORKER_FARM_SCENE_FRACTION_FEATURE]),
            0.5,
        )
        self.assertAlmostEqual(
            float(observation[DUAL_WORKER_TIME_REMAINING_FEATURE]),
            0.9,
        )
        self.assertAlmostEqual(
            float(observation[DUAL_WORKER_LAYER_KILLS_FEATURE]),
            10 / 50,
        )
        self.assertAlmostEqual(
            float(observation[
                DUAL_WORKER_LEGACY_LAYER_TIME_FEATURE]),
            75 / 1500,
        )
        self.assertAlmostEqual(
            float(observation[
                DUAL_WORKER_DRY_FLOOR_REMAINING_FEATURE]),
            (REVISIT_FLOOR - 10) / REVISIT_FLOOR,
        )
        self.assertEqual(
            float(observation[DUAL_WORKER_DRINK_LATCH_FEATURE]),
            1.0,
        )
        self.assertEqual(
            float(observation[
                DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
            0.375,
        )

        fuse = observation[DUAL_WORKER_FUSE_STREAK_SLICE]
        self.assertEqual(int(np.count_nonzero(fuse)), 1)
        self.assertAlmostEqual(float(fuse[9]), 8 / 25)
        np.testing.assert_array_equal(
            observation[DUAL_WORKER_ACTION_MASK_SLICE],
            worker_mask.astype(np.float32),
        )
        np.testing.assert_array_equal(
            observation[DUAL_WORKER_MANAGER_MASK_SLICE],
            manager_mask.astype(np.float32),
        )
        np.testing.assert_array_equal(
            observation[635:],
            options.env.controller_vector,
        )

        # A remembered action is not a current streak after any signature
        # component changes.  In particular, do not fabricate fifteen copies.
        options.env._raw["gold"] += 1
        invalidated = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
        np.testing.assert_array_equal(
            invalidated[DUAL_WORKER_FUSE_STREAK_SLICE],
            np.zeros(15, dtype=np.float32),
        )

    def test_layout_guard_constructively_rejects_banned_wire_field(self):
        combat_segment = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "controller_combat"
        )
        poisoned = replace(
            combat_segment,
            field_names=combat_segment.field_names + ("ai_seed_hi",),
        )
        segments = tuple(
            poisoned if segment is combat_segment else segment
            for segment in DUAL_WORKER_LAYOUT.segments
        )
        self.assertIn(
            "controller_combat:field:ai_seed_hi",
            _banned_policy_wire_violations(segments),
        )
        missile_segment = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "controller_missiles"
        )
        leaked_var = replace(
            missile_segment,
            field_names=missile_segment.field_names + ("var1",),
        )
        leaked_segments = tuple(
            leaked_var if segment is missile_segment else segment
            for segment in DUAL_WORKER_LAYOUT.segments
        )
        self.assertIn(
            "controller_missiles:field:var1",
            _banned_policy_wire_violations(leaked_segments),
        )
        for field in (
            "missile.last_collision_target_hash",
            "missile.var1",
            "missile.var2",
            "missile.var3",
            "missile.var4",
            "missile.var5",
            "missile.var6",
            "missile.var7",
        ):
            self.assertIn(
                field,
                DUAL_WORKER_LAYOUT.excluded_high_entropy_fields,
            )

    def test_layout_hash_binds_scales_and_validator_closes_metadata(self):
        monster = next(
            segment for segment in DUAL_WORKER_LAYOUT.segments
            if segment.name == "controller_monsters"
        )
        changed_encodings = list(monster.field_encodings)
        changed_encodings[1] = "divide_by:201"
        changed_monster = replace(
            monster, field_encodings=tuple(changed_encodings))
        changed_layout = replace(
            DUAL_WORKER_LAYOUT,
            segments=tuple(
                changed_monster if segment is monster else segment
                for segment in DUAL_WORKER_LAYOUT.segments
            ),
        )
        _validate_layout_spec(changed_layout)
        self.assertNotEqual(
            _layout_sha256(changed_layout),
            DUAL_WORKER_LAYOUT_SHA256,
        )

        malformed_monster = replace(
            monster, field_encodings=monster.field_encodings[:-1])
        malformed_layout = replace(
            DUAL_WORKER_LAYOUT,
            segments=tuple(
                malformed_monster if segment is monster else segment
                for segment in DUAL_WORKER_LAYOUT.segments
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "encoding 数量"):
            _validate_layout_spec(malformed_layout)

        malformed_shape = replace(
            monster, shape=(monster.width + 1,))
        malformed_shape_layout = replace(
            DUAL_WORKER_LAYOUT,
            segments=tuple(
                malformed_shape if segment is monster else segment
                for segment in DUAL_WORKER_LAYOUT.segments
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "shape/width"):
            _validate_layout_spec(malformed_shape_layout)

        zero_axis = replace(monster, shape=(0, monster.width))
        zero_axis_layout = replace(
            DUAL_WORKER_LAYOUT,
            segments=tuple(
                zero_axis if segment is monster else segment
                for segment in DUAL_WORKER_LAYOUT.segments
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "shape/width"):
            _validate_layout_spec(zero_axis_layout)

    def test_fuse_unarmed_and_armed_zero_are_not_observation_aliases(self):
        options, _current, _legacy = _options_fixture()
        options._fuse_sig = None
        options._fuse = 0
        unarmed = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)

        options._fuse_sig = options._sig(9, options.env._raw)
        options._fuse = 0
        armed = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)

        self.assertFalse(np.array_equal(unarmed, armed))
        np.testing.assert_array_equal(
            unarmed[DUAL_WORKER_FUSE_STREAK_SLICE],
            np.zeros(15, dtype=np.float32),
        )
        self.assertAlmostEqual(
            float(armed[DUAL_WORKER_FUSE_STREAK_SLICE][9]),
            1 / 25,
        )
        options._fuse = 25
        with self.assertRaisesRegex(RuntimeError, r"\[0,24\]"):
            options._worker_policy_observation(
                WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)

    def test_options_default_skip_probability_is_zero_and_worker_is_dynamic(self):
        options, _current, _legacy = _options_fixture()
        evaluation = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
        self.assertEqual(
            float(evaluation[
                DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
            0.0,
        )

        worker = WorkerWindowEnv.__new__(WorkerWindowEnv)
        worker.oe = options
        worker.policy_observation_view = (
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
        worker.legacy_policy_observation_view = False
        worker.skip_dry = 0.625
        training = worker._policy_observation(options._worker_obs())
        self.assertEqual(
            float(training[
                DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
            0.625,
        )
        worker.set_skip_dry_p(0.2)
        updated = worker._policy_observation(options._worker_obs())
        self.assertAlmostEqual(
            float(updated[
                DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
            0.2,
        )

    def test_historical_298_views_are_unchanged(self):
        options, current, legacy = _options_fixture()
        raw_view = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_RAW_V4)
        exact = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_LEGACY_V3)
        overlay = options._worker_policy_observation(
            WORKER_OBSERVATION_VIEW_A12_OVERLAY)

        np.testing.assert_array_equal(raw_view[:295], current)
        self.assertEqual(raw_view.shape, (298,))
        self.assertEqual(float(raw_view[297]), -2.0)
        np.testing.assert_array_equal(exact[:295], legacy)
        self.assertEqual(float(exact[297]), 1.0)
        self.assertAlmostEqual(float(overlay[286]), 2 / 8 + 4 / 128)
        self.assertEqual(float(overlay[297]), -2.0)
        np.testing.assert_array_equal(
            legacy_worker_policy_observation_view(overlay),
            exact,
        )

        legacy_worker = WorkerWindowEnv.__new__(WorkerWindowEnv)
        legacy_worker.oe = options
        legacy_worker.policy_observation_view = (
            WORKER_OBSERVATION_VIEW_LEGACY_V3)
        legacy_worker.legacy_policy_observation_view = True
        legacy_worker.skip_dry = 1.0
        np.testing.assert_array_equal(
            legacy_worker._policy_observation(options._worker_obs()),
            exact,
        )

    def test_dual_view_fails_closed_without_lossless_legacy_fields(self):
        options, _current, _legacy = _options_fixture()
        del options.env._raw["legacy_belt_heals"]
        with self.assertRaisesRegex(RuntimeError, "legacy_belt_heals"):
            options._worker_policy_observation(
                WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)

        options, _current, _legacy = _options_fixture()
        del options.env._raw["floor_items"][0]["legacy_heal"]
        with self.assertRaisesRegex(RuntimeError, "legacy_heal"):
            options._worker_policy_observation(
                WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)

        options, _current, _legacy = _options_fixture()
        options.env._raw = None
        with self.assertRaisesRegex(RuntimeError, "active native raw"):
            options._worker_policy_observation(
                WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)

    def test_worker_window_requires_explicit_dual_selector(self):
        class FakeManager:
            source_sha256 = "0" * 64

            def __init__(self, *_args, **_kwargs):
                pass

            def require_io_shape(self, *_args):
                pass

        class FakeOptions:
            def __init__(self, *_args, **_kwargs):
                self.env = types.SimpleNamespace(
                    observation_space=gym.spaces.Box(
                        -np.inf, np.inf, shape=(295,), dtype=np.float32))

        module = sys.modules[WorkerWindowEnv.__module__]
        with (
            mock.patch.object(module, "NumpyManager", FakeManager),
            mock.patch.object(module, "OptionsEnv", FakeOptions),
        ):
            default = WorkerWindowEnv("manager.npz")
            legacy = WorkerWindowEnv(
                "manager.npz",
                legacy_policy_observation_view=True,
            )
            dual = WorkerWindowEnv(
                "manager.npz",
                policy_observation_view=(
                    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC),
            )

        self.assertEqual(default.observation_space.shape, (298,))
        self.assertEqual(
            default.policy_observation_view,
            WORKER_OBSERVATION_VIEW_A12_OVERLAY,
        )
        self.assertEqual(legacy.observation_space.shape, (298,))
        self.assertEqual(
            legacy.policy_observation_view,
            WORKER_OBSERVATION_VIEW_LEGACY_V3,
        )
        self.assertEqual(
            dual.observation_space.shape,
            (DUAL_WORKER_OBSERVATION_DIM,),
        )
        self.assertEqual(
            dual.policy_observation_view,
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
        )
        with self.assertRaisesRegex(ValueError, "冲突"):
            WorkerWindowEnv(
                "manager.npz",
                legacy_policy_observation_view=True,
                policy_observation_view=(
                    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC),
            )


if __name__ == "__main__":
    unittest.main()
