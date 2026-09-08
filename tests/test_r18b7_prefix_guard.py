"""R18-B7 (2026-09-07): the earned prefix must never descend under the frozen parent.

The R18-B training arm crashed at 23:25 with "completion L2 arrival before learner handoff":
a DIVE window whose opening verdict was not eligible (coach-v03 opens on the six-condition law,
the handoff needs the seven-condition native verdict) was played by the parent, which pressed
the descend macro. WorkerWindowEnv._prefix_guard_masks withholds a11 and the trigger-tile steps
from the parent inside such windows on main L1 only.
"""
import unittest

import numpy as np

from diablogym import worker_env
from diablogym.env import DiabloGymEnv


def _raw(dlvl=1, set_level=False, px=10, py=10, protected=()):
    return {"dungeon_level": dlvl, "is_set_level": set_level, "player_x": px, "player_y": py,
            "triggers": [{"x": x, "y": y, "msg": 1} for x, y in protected],
            "progression_targets": []}


class PrefixGuardMaskTests(unittest.TestCase):
    def test_a11_is_withheld_on_main_l1(self):
        masks = np.ones(15, dtype=bool)
        guarded = worker_env.WorkerWindowEnv._prefix_guard_masks(_raw(), masks)
        self.assertFalse(guarded[11])
        self.assertTrue(guarded[0])
        self.assertTrue(masks[11], "the caller's mask array is never mutated")

    def test_trigger_tile_steps_are_withheld_exactly_like_the_frozen_walk_law(self):
        raw = _raw()
        protected = DiabloGymEnv._protected_walk_actions(raw)
        masks = np.ones(15, dtype=bool)
        guarded = worker_env.WorkerWindowEnv._prefix_guard_masks(raw, masks)
        for action in protected:
            self.assertFalse(guarded[action])
        for action in range(1, 9):
            if action not in protected:
                self.assertTrue(guarded[action])

    def test_untouched_off_main_l1(self):
        masks = np.ones(15, dtype=bool)
        for raw in (_raw(dlvl=2), _raw(dlvl=0), _raw(dlvl=1, set_level=True)):
            guarded = worker_env.WorkerWindowEnv._prefix_guard_masks(raw, masks)
            self.assertTrue(np.array_equal(guarded, masks))

    def test_wait_stays_legal_when_everything_else_is_masked(self):
        masks = np.zeros(15, dtype=bool)
        masks[11] = True
        guarded = worker_env.WorkerWindowEnv._prefix_guard_masks(_raw(), masks)
        self.assertFalse(guarded[11])
        self.assertTrue(guarded[0])
        self.assertEqual(int(guarded.sum()), 1)

    def test_the_prefix_loop_uses_the_guard(self):
        import inspect
        src = inspect.getsource(worker_env.WorkerWindowEnv._prefix_open_dive)
        self.assertIn("_prefix_guard_masks(self.oe.env._raw, self.oe._worker_masks())", src)


if __name__ == "__main__":
    unittest.main()
