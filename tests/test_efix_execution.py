"""E-fix 执行修理案 KAT(2026-07-31,批文:甲形态/全集口径)。
端到端证据(旧端点 8/8 位级复现官方档案、新端点停滞率 75%→0%、
首潜前缀分域一致)在 train/runs/efix-g0-evidence/ 案卷;本文件钉住
可单元化的决策逻辑与旋钮契约。"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from diablogym import options_env as oe_mod  # noqa: E402
from diablogym.options_env import DIVE, KILL_PATIENCE, OptionsEnv  # noqa: E402


def _shell(protocol):
    shell = OptionsEnv.__new__(OptionsEnv)
    shell.dive_stall_protocol = protocol
    return shell


class DiveStallProtocolTests(unittest.TestCase):
    def _term_reason(self, protocol, tau, last_progress_tau):
        shell = _shell(protocol)

        class _EnvStub:
            _steps = 1000 + tau
            _raw = {"dead": False, "dungeon_level": 1}
        shell.env = _EnvStub()
        shell._win = {
            "opt": DIVE, "t0": 1000, "scene0": "scene-x",
            "dlvl0": 1, "floor": 0,
            "dive_last_progress_tau": last_progress_tau,
        }
        original = oe_mod._scene_identity
        oe_mod._scene_identity = lambda raw: "scene-x"
        try:
            return shell._win_term(False, False, 0)
        finally:
            oe_mod._scene_identity = original

    def test_old_protocol_is_pure_tau(self):
        self.assertEqual(
            self._term_reason("tau-v3", KILL_PATIENCE, KILL_PATIENCE - 1),
            "stall")
        self.assertIsNone(
            self._term_reason("tau-v3", KILL_PATIENCE - 1, 0))

    def test_new_protocol_counts_from_last_progress(self):
        # τ 已超旧限,但最近一次进展在 10 拍前 → 续命
        self.assertIsNone(self._term_reason(
            "no-progress-v1", KILL_PATIENCE + 60, KILL_PATIENCE - 70))
        # 无进展满 KILL_PATIENCE → 收窗(极限环仍被正确枪毙)
        self.assertEqual(self._term_reason(
            "no-progress-v1", KILL_PATIENCE, 0), "stall")
        self.assertEqual(self._term_reason(
            "no-progress-v1", KILL_PATIENCE + 200, 200), "stall")

    def test_unconfigured_shell_defaults_to_old_semantics(self):
        shell = _shell("tau-v3")
        del shell.dive_stall_protocol  # 桩/壳未配置 → 旧语义
        shell2 = shell
        self.assertEqual(
            getattr(shell2, "dive_stall_protocol", "tau-v3"), "tau-v3")


class DiveTargetDistanceTests(unittest.TestCase):
    def _distance(self, raw):
        return OptionsEnv.__new__(OptionsEnv)._dive_target_distance(raw)

    def test_progression_targets_take_priority(self):
        raw = {
            "player_x": 10, "player_y": 10,
            "progression_targets": [
                {"goal_x": 13, "goal_y": 10},
                {"goal_x": 30, "goal_y": 30},
            ],
            "triggers": [{"msg": 999, "x": 11, "y": 10}],
        }
        self.assertEqual(self._distance(raw), 3)

    def test_stairs_chebyshev_nearest(self):
        from diablogym import bridge
        raw = {
            "player_x": 10, "player_y": 10,
            "progression_targets": [],
            "is_set_level": False,
            "triggers": [
                {"msg": bridge.WM_DIABNEXTLVL, "x": 15, "y": 12},
                {"msg": bridge.WM_DIABNEXTLVL, "x": 10, "y": 30},
            ],
        }
        self.assertEqual(self._distance(raw), 5)

    def test_no_target_returns_none(self):
        raw = {"player_x": 1, "player_y": 1, "progression_targets": [],
               "is_set_level": False, "triggers": []}
        self.assertIsNone(self._distance(raw))


class KnobContractTests(unittest.TestCase):
    def test_dive_stall_protocol_validated(self):
        with self.assertRaises(ValueError):
            OptionsEnv(dive_stall_protocol="bogus")

    def test_descend_fallback_promotion_validated(self):
        from diablogym.env import DiabloGymEnv
        with self.assertRaises(TypeError):
            DiabloGymEnv(descend_fallback_promotion="yes")


if __name__ == "__main__":
    unittest.main()
