from __future__ import annotations

import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym.env import DiabloGymEnv, _scene_identity
from diablogym.options_env import DIVE, OptionsEnv, dispatch
from diablogym.worker_env import WorkerWindowEnv, _MAX_EMPTY_FARM_EPISODES


def raw(*, depth=3, is_set=False, set_id=0, monsters=()):
    return {
        "dungeon_level": depth,
        "is_set_level": is_set,
        "set_level_id": set_id,
        "xp": 0,
        "armor_class": 0,
        "monsters": list(monsters),
        "player_x": 10,
        "player_y": 10,
        "dead": False,
        "victory": False,
    }


class SetLevelSemanticsTests(unittest.TestCase):
    def test_unsupported_hero_class_fails_before_native_init(self):
        with self.assertRaisesRegex(ValueError, "只支持 hero_class=0"):
            DiabloGymEnv(hero_class=1)

    def test_set_level_is_a_new_scene_but_not_a_depth_change(self):
        main = raw(monsters=[{"id": 1, "hp": 10, "max_hp": 10,
                              "x": 11, "y": 10}])
        quest = raw(is_set=True, set_id=1)
        self.assertNotEqual(_scene_identity(main), _scene_identity(quest))

        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env.descend_ladder = True
        env.death_ladder = True
        # The main-map monster disappearing during the map swap is neither a
        # kill nor damage, and equal conceptual depth earns no descent bonus.
        self.assertEqual(env._reward(main, quest), 0.0)

    def test_option_window_returns_control_without_false_descend(self):
        options = OptionsEnv.__new__(OptionsEnv)
        options.env = types.SimpleNamespace(
            _raw=raw(is_set=True, set_id=5, depth=15), _steps=11)
        options._win = {
            "t0": 10,
            "scene0": _scene_identity(raw(depth=15)),
            "dlvl0": 15,
        }
        self.assertEqual(options._win_term(False, False, 0), "scene")

        options.env._raw = raw(depth=16)
        self.assertEqual(options._win_term(False, False, 0), "descend")

    def test_worker_reset_fails_instead_of_rolling_forever_without_farm(self):
        worker = WorkerWindowEnv.__new__(WorkerWindowEnv)
        worker.mgr = types.SimpleNamespace(source_sha256="a" * 64)
        worker.oe = types.SimpleNamespace(_win=None)
        worker._alive = True
        worker._rng = None
        worker.stats = {"reseeds": 0}
        worker.next_window = lambda: None
        worker._new_episode = lambda *args, **kwargs: None
        with self.assertRaisesRegex(RuntimeError, "未产生 FARM"):
            worker.reset()
        self.assertEqual(worker.stats["reseeds"], _MAX_EMPTY_FARM_EPISODES)

    def test_vile_book_requires_exact_circle_and_progression_is_snapshot_isolated(self):
        book = {
            "kind": "vile_book", "action": "operate",
            "x": 26, "y": 45, "goal_x": 26, "goal_y": 46,
            "exact": True,
        }
        state = {"player_x": 26, "player_y": 45}
        self.assertFalse(DiabloGymEnv._progression_ready(state, book))
        state["player_y"] = 46
        self.assertTrue(DiabloGymEnv._progression_ready(state, book))

        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env.include_raw = True
        env._episode_seed = 7
        source = {"progression_targets": [book], "monsters": [],
                  "floor_items": [], "triggers": []}
        snapshot = env._info(source)["raw"]
        snapshot["progression_targets"][0]["kind"] = "tampered"
        self.assertEqual(source["progression_targets"][0]["kind"], "vile_book")

    def test_hierarchical_paths_cannot_mask_or_loop_on_mandatory_progression(self):
        state = raw(monsters=[{"id": 1, "hp": 1, "max_hp": 1,
                               "x": 30, "y": 30}])
        state.update({
            "hp": 70, "max_hp": 70, "belt_heals": 0,
            "floor_items": [], "triggers": [],
            "progression_targets": [{
                "kind": "diablo_switch", "action": "operate",
                "x": 20, "y": 20, "goal_x": 20, "goal_y": 20,
                "exact": False,
            }],
        })
        # 远处/门后的怪仍在全图列表时，FARM 必须先给机关宏一次机会。
        self.assertEqual(dispatch("farm", state, False), 10)
        self.assertEqual(dispatch("dive", state, False), 11)

        options = OptionsEnv.__new__(OptionsEnv)
        options._last_base_obs = object()
        options.env = types.SimpleNamespace(
            _raw=state, _ensure_active=lambda **kwargs: None)
        self.assertTrue(options.action_masks()[DIVE])


if __name__ == "__main__":
    unittest.main()
