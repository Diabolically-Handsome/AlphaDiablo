"""真实引擎 + SB3 VecEnv 的时间上限 bootstrap 语义回归测试。"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import types
import unittest

import gymnasium as gym
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym import DiabloGymEnv, WorkerWindowEnv  # noqa: E402
from diablogym.options_env import KILL_PATIENCE  # noqa: E402
from diablogym.worker_env import _AdvanceOutcome  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def _farm_manager(path: pathlib.Path) -> None:
    """写一个确定性选择 FARM 的 303→3 numpy manager。"""
    np.savez_compressed(
        path,
        w0=np.zeros((64, 303), dtype=np.float32),
        b0=np.zeros(64, dtype=np.float32),
        w1=np.zeros((64, 64), dtype=np.float32),
        b1=np.zeros(64, dtype=np.float32),
        wa=np.zeros((3, 64), dtype=np.float32),
        ba=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )


class _ArmOneBeatCap(gym.Wrapper):
    """Worker reset 后把底层预算精确收紧到只剩一个 micro-step。"""

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        cap = int(self.env.oe.env._steps) + 1
        self.env.oe.env.max_steps = cap
        self.env.oe.max_steps = cap
        return obs, info


def _timeout_base_info(*, depth: int = 3) -> dict:
    return {
        "time_limit_bootstrap_safe": True,
        "unsettled_budget_terminal": False,
        "episode_extra": {
            "xp": 0,
            "kills": 0,
            "char_level": 1,
            "depth": depth,
            "died": False,
            "gold": 0,
        },
    }


def _timeout_extra(*, timeout: bool) -> dict:
    return {
        "window_id": 1,
        "reason": "cap",
        "R": 2.5,
        "base_done": True,
        "base_trunc": True,
        "budget_boundary": True,
        "no_progress_micro_steps": KILL_PATIENCE if timeout else 0,
        "timeout_without_progress": timeout,
    }


class _SyntheticWorkerOptions:
    """不启动原生引擎的 Worker 终局边界。"""

    def __init__(self, *, direct_terminal: bool, timeout: bool):
        self._win = {"window_id": 1, "W": 0.0}
        self._direct_terminal = direct_terminal
        self._extra = _timeout_extra(timeout=timeout)
        self._base_info = _timeout_base_info()
        self.env = types.SimpleNamespace(
            death_ladder=True,
            _raw={"dead": False, "dungeon_level": 3},
        )

    def _win_step_worker(self, action):
        self._win["W"] += 2.5
        return types.SimpleNamespace(
            reason="cap" if self._direct_terminal else "levelup",
            requested_action=int(action),
            executed_action=int(action),
            fuse_tripped=False,
            fuse_requested_action=None,
        )

    def _worker_obs(self):
        return np.zeros(298, dtype=np.float32)

    def _win_end(self, reason):
        self._win = None
        if self._direct_terminal:
            return (
                dict(self._extra),
                dict(self._base_info),
                False,
                True,
            )
        return {
            "window_id": 1,
            "reason": reason,
            "R": 2.5,
            "base_done": False,
            "base_trunc": False,
            "timeout_without_progress": False,
        }, {}, False, False


def _synthetic_worker(options, *, additional_cost: float = 3.0):
    worker = object.__new__(WorkerWindowEnv)
    worker.oe = options
    worker.action_space = gym.spaces.Discrete(15)
    worker.skip_dry = 0.0
    worker._pending_skip_dry_probability = None
    worker._pending_skip_dry_remaining_env_steps = 0
    worker._episode_seed = 17
    worker._alive = True
    worker._log = lambda extra, fast_forward: None
    worker.fast_forward_reward_credit = "none"
    worker.additional_terminal_death_cost = additional_cost
    worker.stats = {
        "reasons": {},
        "transition_ff_reward": 0.0,
        "direct_terminal_deaths": 0,
        "transition_ff_terminal_deaths": 0,
        "direct_existing_terminal_death_reward": 0.0,
        "direct_additional_terminal_death_reward": 0.0,
        "transition_ff_terminal_death_reward": 0.0,
        "transition_ff_additional_terminal_death_reward": 0.0,
        "credited_ff_terminal_death_reward": 0.0,
        "additional_terminal_death_reward": 0.0,
    }
    return worker


class TerminalBootstrapTests(unittest.TestCase):
    def test_worker_no_progress_timeout_is_nonbootstrap_failure(self):
        direct = _synthetic_worker(_SyntheticWorkerOptions(
            direct_terminal=True, timeout=True))
        _, direct_reward, terminated, truncated, direct_info = direct.step(9)

        # depth=3 的 ladder 基础死亡等价项为 -24，配置额外项为 -3；
        # 当前动作工资 +2.5，因此策略 transition 总回报为 -24.5。
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(direct_reward, 2.5 - 24.0 - 3.0)
        self.assertTrue(direct_info["worker_no_progress_timeout"])
        self.assertFalse(direct_info["time_limit_bootstrap_safe"])
        self.assertTrue(
            direct_info["terminal_base_info"][
                "time_limit_bootstrap_safe"])
        self.assertEqual(
            direct_info["no_progress_timeout_base_failure_reward"], -24.0)
        self.assertEqual(
            direct_info["no_progress_timeout_additional_failure_reward"], -3.0)
        self.assertEqual(
            direct_info["no_progress_timeout_failure_reward"], -27.0)
        self.assertEqual(direct_info["existing_terminal_death_reward"], 0.0)
        self.assertEqual(direct_info["additional_terminal_death_reward"], 0.0)
        self.assertEqual(
            direct.stats["direct_no_progress_timeouts"], 1)
        self.assertEqual(
            direct.stats["credited_no_progress_timeout_failure_reward"],
            -27.0)

    def test_direct_and_fast_forward_no_progress_timeouts_are_isomorphic(self):
        direct = _synthetic_worker(_SyntheticWorkerOptions(
            direct_terminal=True, timeout=True))
        direct_result = direct.step(9)

        fast_forward = _synthetic_worker(_SyntheticWorkerOptions(
            direct_terminal=False, timeout=False))
        timeout_extra = _timeout_extra(timeout=True)
        fast_forward._advance_to_learning_window = lambda: _AdvanceOutcome(
            np.zeros(298, dtype=np.float32),
            99.0,
            True,
            False,
            (timeout_extra,),
            terminal_base_info=_timeout_base_info(),
            timeout_without_progress=True,
        )
        ff_result = fast_forward.step(9)
        _, ff_reward, terminated, truncated, ff_info = ff_result

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(ff_reward, direct_result[1])
        for key in (
            "worker_no_progress_timeout",
            "no_progress_timeout_base_failure_reward",
            "no_progress_timeout_additional_failure_reward",
            "no_progress_timeout_failure_reward",
            "transition_reward",
        ):
            self.assertEqual(ff_info[key], direct_result[4][key])
        self.assertEqual(ff_info["fast_forward_reward"], 99.0)
        self.assertEqual(ff_info["credited_fast_forward_reward"], -27.0)
        self.assertEqual(
            fast_forward.stats["transition_ff_no_progress_timeouts"], 1)
        self.assertEqual(
            fast_forward.stats[
                "transition_ff_no_progress_timeout_failure_reward"],
            -27.0)

    def test_progressful_safe_time_limit_remains_truncated_without_cost(self):
        progressful = _synthetic_worker(_SyntheticWorkerOptions(
            direct_terminal=True, timeout=False))
        _, reward, terminated, truncated, info = progressful.step(9)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 2.5)
        self.assertFalse(info["worker_no_progress_timeout"])
        self.assertEqual(info["no_progress_timeout_failure_reward"], 0.0)
        self.assertNotIn(
            "direct_no_progress_timeouts", progressful.stats)

    def test_reset_fast_forward_timeout_is_audit_only(self):
        worker = object.__new__(WorkerWindowEnv)
        worker.oe = types.SimpleNamespace(
            _win=None,
            env=types.SimpleNamespace(
                death_ladder=True,
                _raw={"dead": False, "dungeon_level": 3},
            ),
        )
        worker._alive = True
        worker._rng = np.random.default_rng(1)
        worker._episode_seed = 17
        worker.additional_terminal_death_cost = 3.0
        worker.stats = {
            "reset_ff_reward": 0.0,
            "reset_ff_terminal_deaths": 0,
            "reset_ff_terminal_death_reward": 0.0,
            "reset_ff_additional_terminal_death_reward": 0.0,
            "credited_no_progress_timeout_failure_reward": 0.0,
            "reseeds": 0,
        }
        outcomes = iter((
            _AdvanceOutcome(
                np.zeros(298, dtype=np.float32),
                99.0,
                True,
                False,
                (_timeout_extra(timeout=True),),
                terminal_base_info=_timeout_base_info(),
                timeout_without_progress=True,
            ),
            _AdvanceOutcome(
                np.zeros(298, dtype=np.float32),
                4.0,
                False,
                False,
                (),
            ),
        ))

        def advance():
            result = next(outcomes)
            if not result.terminated and not result.truncated:
                worker.oe._win = {"window_id": 2}
            return result

        worker._advance_to_learning_window = advance

        def new_episode(seed=None, options=None):
            worker._alive = True
            worker.oe._win = None

        worker._new_episode = new_episode
        obs, info = worker.reset()

        self.assertEqual(obs.shape, (298,))
        self.assertEqual(info["window_id"], 2)
        self.assertEqual(worker.stats["reset_ff_reward"], 103.0)
        self.assertEqual(worker.stats["reset_ff_no_progress_timeouts"], 1)
        self.assertEqual(
            worker.stats["reset_ff_no_progress_timeout_failure_reward"],
            -27.0)
        self.assertEqual(
            worker.stats["credited_no_progress_timeout_failure_reward"],
            0.0)

    def test_no_progress_marker_rejects_non_safe_boundary(self):
        extra = _timeout_extra(timeout=True)
        base_info = _timeout_base_info()
        base_info["time_limit_bootstrap_safe"] = False
        with self.assertRaisesRegex(
                RuntimeError, "budget_boundary 与底层边界事实不闭合"):
            WorkerWindowEnv._worker_boundary(
                False, True, extra, base_info)

    def test_truncation_cannot_fail_open_when_both_producers_omit_boundary(self):
        extra = _timeout_extra(timeout=False)
        extra["budget_boundary"] = False
        base_info = _timeout_base_info()
        base_info.pop("time_limit_bootstrap_safe")
        base_info.pop("unsettled_budget_terminal")

        with self.assertRaisesRegex(
                RuntimeError, "缺少显式安全 TimeLimit 证明"):
            WorkerWindowEnv._worker_boundary(
                False, True, extra, base_info)

    def test_no_progress_marker_and_clock_must_close_both_directions(self):
        base_info = _timeout_base_info()
        missing = _timeout_extra(timeout=True)
        del missing["timeout_without_progress"]
        with self.assertRaisesRegex(RuntimeError, "缺字段"):
            WorkerWindowEnv._worker_boundary(
                False, True, missing, base_info)

        false_negative = _timeout_extra(timeout=False)
        false_negative["no_progress_micro_steps"] = KILL_PATIENCE
        with self.assertRaisesRegex(RuntimeError, "事实不闭合"):
            WorkerWindowEnv._worker_boundary(
                False, True, false_negative, base_info)

        false_positive = _timeout_extra(timeout=True)
        false_positive["no_progress_micro_steps"] = 0
        with self.assertRaisesRegex(RuntimeError, "事实不闭合"):
            WorkerWindowEnv._worker_boundary(
                False, True, false_positive, base_info)

    def test_zero_reward_manual_fast_forward_is_still_auditable(self):
        worker = object.__new__(WorkerWindowEnv)
        worker.stats = {
            "manual_ff_calls": 0,
            "manual_ff_reward": 0.0,
            "manual_ff_terminal_deaths": 0,
        }
        expected = np.zeros(298, dtype=np.float32)
        worker._advance_to_learning_window = lambda: _AdvanceOutcome(
            expected, 0.0, False, False, ())

        actual = worker.next_window()

        self.assertIs(actual, expected)
        self.assertEqual(worker.stats["manual_ff_calls"], 1)
        self.assertEqual(worker.stats["manual_ff_reward"], 0.0)

    def test_unsettled_no_progress_boundary_stays_terminated(self):
        extra = _timeout_extra(timeout=True)
        extra["base_trunc"] = False
        base_info = _timeout_base_info()
        base_info["time_limit_bootstrap_safe"] = False
        base_info["unsettled_budget_terminal"] = True
        self.assertEqual(
            WorkerWindowEnv._worker_boundary(
                True, False, extra, base_info),
            (True, False, True),
        )

    def _base_vec_step(self, action: int):
        vec = DummyVecEnv([
            lambda: Monitor(DiabloGymEnv(
                start_in_dungeon=True,
                max_steps=1,
                include_raw=False,
            )),
        ])
        try:
            vec.seed(7000)
            obs = vec.reset()
            self.assertEqual(obs.shape, (1, 295))
            _, _, done, infos = vec.step(np.asarray([action]))
            self.assertTrue(done[0])
            self.assertEqual(infos[0]["terminal_observation"].shape, (295,))
            return infos[0]
        finally:
            vec.close()

    def test_sb3_bootstraps_only_idle_base_time_limit(self):
        # seed=0 的地牢出生点上，方向 1 在四 tick 后仍处于已提交走格；
        # action 0 则是 idle 的标准 TimeLimit 边界。
        busy = self._base_vec_step(1)
        self.assertFalse(busy["TimeLimit.truncated"])
        self.assertFalse(busy["decision_idle"])
        self.assertTrue(busy["unsettled_budget_terminal"])
        self.assertFalse(busy["time_limit_bootstrap_safe"])

        idle = self._base_vec_step(0)
        self.assertTrue(idle["TimeLimit.truncated"])
        self.assertTrue(idle["decision_idle"])
        self.assertFalse(idle["unsettled_budget_terminal"])
        self.assertTrue(idle["time_limit_bootstrap_safe"])

    def _worker_vec_step(self, manager_path: pathlib.Path, action: int):
        vec = DummyVecEnv([
            lambda: Monitor(_ArmOneBeatCap(WorkerWindowEnv(
                str(manager_path),
                max_steps=3000,
                rng_seed=101,
                seed_scope="replay",
            ))),
        ])
        try:
            vec.seed(101)
            obs = vec.reset()
            self.assertEqual(obs.shape, (1, 298))
            _, _, done, infos = vec.step(np.asarray([action]))
            self.assertTrue(done[0])
            self.assertEqual(infos[0]["terminal_observation"].shape, (298,))
            return infos[0]
        finally:
            vec.close()

    def test_worker_propagates_unsafe_boundary_without_timelimit_flag(self):
        with tempfile.TemporaryDirectory() as td:
            manager = pathlib.Path(td) / "farm-manager.npz"
            _farm_manager(manager)

            # action10 在出生点提交探索走格；一拍预算不足以结清动画。
            busy = self._worker_vec_step(manager, 10)
            self.assertFalse(busy["TimeLimit.truncated"])
            self.assertTrue(busy["unsettled_budget_terminal"])
            self.assertFalse(busy["time_limit_bootstrap_safe"])
            self.assertTrue(
                busy["terminal_base_info"]["unsettled_budget_terminal"])

            # 显式 wait 在同一预算上限保持 idle，仍是可 bootstrap 的
            # 298 维 TimeLimit terminal_observation。
            idle = self._worker_vec_step(manager, 0)
            self.assertTrue(idle["TimeLimit.truncated"])
            self.assertFalse(idle["unsettled_budget_terminal"])
            self.assertTrue(idle["time_limit_bootstrap_safe"])
            self.assertTrue(
                idle["terminal_base_info"]["time_limit_bootstrap_safe"])


if __name__ == "__main__":
    unittest.main()
