"""Diagnostic resource caps have one explicit identity and cannot enter training."""
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from diablogym.env import DiabloGymEnv
from diablogym.options_env import OptionsEnv, RESUPPLY
from diablogym.worker_env import WorkerWindowEnv
from diablogym.resource_protocol import ResourceCalibration, ResourceService


def state(depth=1, *, ready=False):
    return dict(dungeon_level=depth, is_set_level=False, player_x=10, player_y=10,
                hp=70, max_hp=70, gold=0, belt_heals=4,
                monsters=[dict(hp=1, type=1)], monster_kill_total=0,
                resource_state=dict(enabled=True, readiness=dict(ready=ready)))


class CalibrationConfigTests(unittest.TestCase):
    def test_config_is_explicit_immutable_and_rejects_invalid_budgets(self):
        config = ResourceCalibration(calibration_id="prefix")
        self.assertEqual(config.service_microstep_cap, 600)
        self.assertEqual(config.farm_scene_microstep_cap, 3600)
        self.assertFalse(config.metadata()["formal_metric_eligible"])
        with self.assertRaises(FrozenInstanceError):
            config.service_microstep_cap = 2400
        for bad in (True, 0, -1, 1.5, float("inf"), "2400"):
            for field in ("service_microstep_cap", "farm_scene_microstep_cap"):
                with self.subTest(value=bad, field=field), self.assertRaises(ValueError):
                    ResourceCalibration(calibration_id="invalid", **{field: bad})
        with self.assertRaises(ValueError):
            ResourceCalibration(calibration_id=" ")

    def test_options_share_one_config_and_production_keeps_defaults_and_wire(self):
        config = ResourceCalibration(calibration_id="timing", service_microstep_cap=2400,
                                     farm_scene_microstep_cap=1800)
        for calibration, expected in ((None, (600, 3600)), (config, (2400, 1800))):
            base = SimpleNamespace(observation_space=gym.spaces.Box(-1, 1, shape=(295,)))
            with patch("diablogym.options_env.DiabloGymEnv", return_value=base):
                options = OptionsEnv(max_steps=6000, resource_protocol="l2-town-v1",
                                     resource_calibration=calibration)
            self.assertEqual(options.max_steps, 6000)
            self.assertEqual(options.resource_service.service_microstep_cap, expected[0])
            self.assertEqual(options.farm_scene_cap, expected[1])
            self.assertIs(options.resource_service.calibration, calibration)
            if calibration is None:
                self.assertNotIn("calibration", options.resource_service.telemetry())
                self.assertFalse(hasattr(base, "_resource_calibration"))
            else:
                self.assertIs(base._resource_calibration, config)
                options._reset_wrapper_state()
                self.assertIs(options.resource_service.calibration, config)

    def test_calibration_is_rejected_by_off_protocol_and_worker_training(self):
        config = ResourceCalibration(calibration_id="blocked")
        with self.assertRaisesRegex(ValueError, "requires l2-town-v1"):
            OptionsEnv(resource_calibration=config)
        with self.assertRaisesRegex(TypeError, "ResourceCalibration"):
            OptionsEnv(resource_protocol="l2-town-v1", resource_calibration={})
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            WorkerWindowEnv("unused-manager.npz", resource_protocol="l2-town-v1", resource_calibration=config)

    def test_cap_and_threshold_excess_use_actual_start_clock(self):
        config = ResourceCalibration(calibration_id="timing", service_microstep_cap=2400,
                                     farm_scene_microstep_cap=1800)
        service = ResourceService(calibration=config)
        service.start(state(), 1820, "farm_cap", farm_scene_steps=1803)
        info = service.telemetry()
        self.assertEqual(info["start_microstep"], 1820)
        self.assertEqual(info["start_farm_microsteps"], 1803)
        self.assertEqual(info["trigger_threshold_excess"], 3)
        service.phase = "return"
        env = SimpleNamespace(_raw=state(0), _steps=2420)
        service._stairs = lambda *args: ("walk", 11, 10)
        bridge = SimpleNamespace(WM_DIABNEXTLVL=0)
        self.assertEqual(service.command(env, bridge), ("walk", 11, 10))
        env._steps = 4220
        self.assertEqual(service.command(env, bridge), ("finish", "resource_service_cap"))
        self.assertEqual(service.steps, 2400)

    def test_calibration_cap_boundary_never_accepts_a_late_return(self):
        config = ResourceCalibration(calibration_id="timing", service_microstep_cap=2400)
        for elapsed, success in ((2400, True), (2401, False)):
            service = ResourceService(calibration=config, attempted=True, active=True, phase="return")
            env = SimpleNamespace(_raw=state(1, ready=True), _steps=elapsed)
            configured = []
            bridge = SimpleNamespace(configure_town_service=configured.append)
            command = service.command(env, bridge)
            self.assertEqual(command[0], "complete" if success else "finish")
            self.assertEqual(configured, [False] if success else [])

    def test_diagnostic_farm_trigger_routes_at_configured_actual_threshold(self):
        from diablogym import bridge
        raw = state()
        raw["triggers"] = [dict(msg=bridge.WM_DIABNEXTLVL, x=20, y=20)]
        options = OptionsEnv.__new__(OptionsEnv)
        options.env = SimpleNamespace(_raw=raw, _steps=1810, _ensure_active=lambda **kwargs: None)
        options._last_base_obs = np.zeros(295, dtype=np.float32)
        options._sync_farm_scene = lambda scene: None
        options._controller_action_context = lambda: (np.ones(15, dtype=bool), None)
        options.exhausted = False
        options.resource_protocol = "l2-town-v1"
        options.resource_calibration = ResourceCalibration(calibration_id="early", farm_scene_microstep_cap=1800)
        options.resource_service = ResourceService(calibration=options.resource_calibration)
        options.farm_scene_steps = 1799
        options.dive_live_sovereignty = True
        options._win = None
        options.action_masks()
        self.assertFalse(options.resource_service.attempted)
        options.farm_scene_steps = 1802
        self.assertEqual(list(options.action_masks()), [False, False, True])
        self.assertEqual(options.resource_service.start_steps, 1810)
        self.assertEqual(options.resource_service.telemetry()["trigger_threshold_excess"], 2)


class CalibrationActorPrefixTests(unittest.TestCase):
    def options_at(self, trigger, farm_steps):
        # Reuse the complete controller-wire fixture, not a hand-built subset
        # of the actor's scalars, so the full 13012-vector is compared.
        sys.path.insert(0, str(ROOT / "tests"))
        from test_dual_worker_observation import _options_fixture
        from diablogym.options_env import FARM
        options, _, _ = _options_fixture()
        options.resource_protocol = "l2-town-v1"
        options.resource_calibration = ResourceCalibration(
            calibration_id="paired-prefix", farm_scene_microstep_cap=trigger)
        options.resource_service = ResourceService(calibration=options.resource_calibration)
        options.farm_scene_cap = trigger
        options.farm_scene_steps = farm_steps
        options.max_steps = 6000
        options.env._steps = farm_steps + 10
        options.env._raw["dungeon_level"] = 1
        options.env._raw["resource_state"] = dict(enabled=True, readiness=dict(ready=False))
        options._farm_scene = (1, False, 0)
        options._resource_farm_ledgers = {}
        options._win.update(opt=FARM, t0=options.env._steps - 5)
        options.dive_live_sovereignty = True
        return options

    def test_full_actor_observation_is_identical_before_either_trigger(self):
        from diablogym.options_env import (WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
            DUAL_WORKER_FARM_SCENE_FRACTION_FEATURE)
        for farm_steps in (1, 900, 1799):
            with self.subTest(farm_steps=farm_steps):
                early = self.options_at(1800, farm_steps)
                late = self.options_at(3600, farm_steps)
                a = early._worker_policy_observation(WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
                b = late._worker_policy_observation(WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
                self.assertEqual(a.shape, (13012,))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a[DUAL_WORKER_FARM_SCENE_FRACTION_FEATURE],
                                 np.float32(farm_steps / 3600))
                self.assertFalse(early.resource_service.attempted)
                self.assertFalse(late.resource_service.attempted)

    def test_controller_trigger_still_differs_at_actual_1800_boundary(self):
        from diablogym.options_env import FARM
        early, late = self.options_at(1800, 1800), self.options_at(3600, 1800)
        self.assertEqual(list(early.action_masks()), [False, False, True])
        self.assertTrue(late.action_masks()[FARM])
        self.assertTrue(early.resource_service.attempted)
        self.assertFalse(late.resource_service.attempted)
        self.assertEqual(early.resource_calibration.metadata()["worker_farm_scene_denominator"], 3600)
        late.farm_scene_steps = 3600
        self.assertEqual(list(late.action_masks()), [False, False, True])


class ExactDiagnosticStopTests(unittest.TestCase):
    def base(self):
        env = DiabloGymEnv.__new__(DiabloGymEnv)
        env._resource_calibration = ResourceCalibration(calibration_id="stop")
        env._resource_actual_microsteps = 10
        env.max_steps = 10000
        env._decision_idle = lambda raw: False
        env._record_visit = lambda point: None
        return env

    def test_return_callback_deadline_stops_settle_on_exact_native_microtick(self):
        env = self.base()
        calls = []
        def native():
            calls.append("step")
            env._resource_actual_microsteps += 1
            env.max_steps = env._resource_actual_microsteps
            return state(1)
        env._step_native = native
        with patch("diablogym.env.bridge.act_wait", side_effect=lambda: calls.append("wait")):
            raw, beats = env._settle_to_idle(state(0), 1, max_beats=100, start_scene=(0, False, 0))
        self.assertEqual(beats, 2)
        self.assertEqual(raw["dungeon_level"], 1)
        self.assertEqual(calls, ["wait", "step"])
        self.assertEqual(env._resource_actual_microsteps, env.max_steps)

    def test_already_stopped_callback_emits_no_wait_or_extra_microtick(self):
        env = self.base()
        env.max_steps = env._resource_actual_microsteps
        env._step_native = lambda: self.fail("extra native microtick")
        with patch("diablogym.env.bridge.act_wait") as wait:
            raw, beats = env._settle_to_idle(state(), 1, max_beats=100, start_scene=(0, False, 0))
        self.assertEqual(beats, 1)
        wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
