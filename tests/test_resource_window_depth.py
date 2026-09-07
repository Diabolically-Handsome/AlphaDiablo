"""Resource scene returns retain their ledgers and never count as first descent."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from diablogym.env import DiabloGymEnv, REWARD_ECONOMIES
from diablogym.options_env import DIVE, RESUPPLY, OptionsEnv
from diablogym.resource_protocol import ResourceService


def raw(depth, *, is_set=False):
    return dict(dungeon_level=depth, is_set_level=is_set, set_level_id=1,
                char_level=2, xp=0, armor_class=9, gear_combat_utility=0,
                monsters=[], monster_kill_total=0, player_x=10, player_y=10,
                triggers=[], dead=False, victory=False)


class ResourceWindowDepthTests(unittest.TestCase):
    def transition(self, source, target, before, after, *, protocol="l2-town-v1"):
        options = OptionsEnv.__new__(OptionsEnv)
        options.resource_protocol = protocol
        options.action_space = gym.spaces.Discrete(3)
        options.action_masks = lambda: np.ones(3, dtype=bool)
        options._decisions = 0
        options.exhausted = False
        options.env = SimpleNamespace(_raw=source, _steps=10, _ep_kills=0,
                                      _resource_max_main_depth=before)
        options._win_begin(RESUPPLY if source["dungeon_level"] == 0 else DIVE)
        if protocol != "off":
            self.assertEqual(options._win["resource_depth0"], before)
        options.env._raw = target
        options.env._steps = 11
        options.env._resource_max_main_depth = after
        return options._win_term(False, False, 0)

    def reward(self, source, target, before, after, economy):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env.resource_protocol = "l2-town-v1"
        env.reward_economy = REWARD_ECONOMIES[economy]
        env.descend_ladder = env.death_ladder = True
        env._resource_reward_depth_before = before
        env._resource_max_main_depth = after
        env._econ_unvested = 0.0
        env._econ_kills_on_floor = env._econ_prev_epkills = 0
        env._econ_episode_max_depth = after
        env._econ_steps_on_level = env._econ_idle_prev_steps = 0
        env._steps = 1
        paid = env._reward(source, target)
        return paid, env._resource_step_bonus

    def test_town_return_has_no_descent_reason_or_reward(self):
        source, target = raw(0), raw(1)
        self.assertEqual(self.transition(source, target, 1, 1), "scene")
        for economy in ("v1", "v4"):
            with self.subTest(economy=economy):
                self.assertEqual(self.reward(source, target, 1, 1, economy), (0.0, 0.0))

    def test_first_l2_descent_has_reason_and_exact_first_depth_reward(self):
        source, target = raw(1), raw(2)
        self.assertEqual(self.transition(source, target, 1, 2), "descend")
        for economy in ("v1", "v4"):
            with self.subTest(economy=economy):
                unit = REWARD_ECONOMIES[economy].descend_unit
                self.assertEqual(self.reward(source, target, 1, 2, economy), (unit, unit))

    def test_quest_return_is_scene_and_default_off_classification_is_preserved(self):
        self.assertEqual(self.transition(raw(1, is_set=True), raw(1), 1, 1), "scene")
        self.assertEqual(self.transition(raw(0), raw(1), 1, 1, protocol="off"), "descend")


class ResourceFinalMicrostepTests(unittest.TestCase):
    def service_at(self, depth, steps, *, ready=True, growth=False):
        state = raw(depth)
        state.update(hp=70, max_hp=70, gold=0,
                     resource_state={"enabled": True,
                         "readiness": {"ready": ready}})
        service = ResourceService(active=True, attempted=True, phase="return")
        env = SimpleNamespace(_raw=state, _steps=steps,
            controller_action_context=lambda: ([False]*13 + [growth, False], None))
        configured = []
        bridge = SimpleNamespace(configure_town_service=configured.append)
        command = service.command(env, bridge)
        self.assertEqual(env._steps, steps)
        self.assertEqual(service.steps, steps)
        return service, command, configured

    def test_last_allowed_microstep_return_settles_without_another_step(self):
        service, command, configured = self.service_at(1, 600)
        self.assertEqual(command, ("complete",))
        self.assertEqual(service.reason, "complete")
        self.assertFalse(service.active)
        self.assertEqual(configured, [False])

    def test_last_allowed_microstep_still_in_town_is_capped(self):
        service, command, configured = self.service_at(0, 600)
        self.assertEqual(command, ("finish", "resource_service_cap"))
        self.assertFalse(service.active)
        self.assertEqual(configured, [])

    def test_last_allowed_microstep_unready_return_preserves_growth_choice(self):
        service, command, _ = self.service_at(1, 600, ready=False, growth=True)
        self.assertEqual(command, ("complete",))
        self.assertEqual(service.reason, "growth_remaining")
        service, command, _ = self.service_at(1, 600, ready=False)
        self.assertEqual(command, ("finish", "resource_unreachable"))

    def test_an_over_budget_return_cannot_be_reported_as_success(self):
        service, command, configured = self.service_at(1, 601)
        self.assertEqual(command, ("finish", "resource_service_cap"))
        self.assertEqual(configured, [])


if __name__ == "__main__":
    unittest.main()
