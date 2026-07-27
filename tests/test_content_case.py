"""内容案 E1 之自包含快速回归(PREREG-内容案-课⑤x④乙 E1/E7 对应件;不启动引擎/训练)。

覆盖(E1 施工面,rev3 核认勘正逐字):
- worker_env p_skip:布尔语义保留(1.0/0.0 ≡ 原布尔行为,端点不耗流)、
  setter 校验、专用流逐 env 播种(固定偏移自 episode 种子确定性派生)、
  reset(seed) 派生接线、Monitor __getattr__ 透传;
- 调度器解析(linear/hold 语法、主表 244 项、坏格式 fail-loud);
- 课程回调:腿相对 rollout 序号锚定 (num_timesteps−3,497,984)/2048、
  全局步锚定禁用(腿前/失准/越界一律抛,禁钳位)、逐 rollout 落账实际推送 p
  + 回调内恒等断言(rev3 E1②)、"序号→p"全表暴露;
- 四处条件门"skip_dry ∨ schedule"逐门 + 两处值记录保留 CLI 旗字面值;
- 两旗互斥断言与 CLI 文档化。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

import gymnasium as gym
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

import train_ppo  # noqa: E402
from train_ppo import (  # noqa: E402
    _CONTRACT_REVISION,
    _DRY_CURRICULUM_LEG_START,
    _DRY_CURRICULUM_MAIN_TABLE,
    DryCurriculumCallback,
    _capture_dry_window_demos_sha256,
    _dry_window_mechanism_active,
    _mount_dry_anchor_sentinel,
    _parse_dry_curriculum_schedule,
    _precheck_dry_window_demos,
    _resolve_dry_curriculum_start,
    _training_contract,
)
from diablogym.worker_env import (  # noqa: E402
    _AdvanceOutcome,
    _P_SKIP_SEED_OFFSET,
    WorkerWindowEnv,
    _coerce_p_skip,
    _derive_p_skip_rng,
)
from diablogym.options_env import (  # noqa: E402
    DUAL_WORKER_OBSERVATION_DIM,
    DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE,
    WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC,
)

TRAIN_PPO = ROOT / "train" / "train_ppo.py"
MAIN_TABLE_LITERAL = "linear:1.0:0.5:147,hold:0.5:97"


def _stub_worker_env(p, p_rng=None) -> WorkerWindowEnv:
    """不启动引擎的 WorkerWindowEnv 壳:只挂 p_skip 抽签所需状态。"""
    env = WorkerWindowEnv.__new__(WorkerWindowEnv)
    env.skip_dry = _coerce_p_skip(p)
    env._p_rng = p_rng if p_rng is not None else np.random.default_rng(0)
    return env


def _run_cli(*extra_args):
    return subprocess.run(
        [sys.executable, str(TRAIN_PPO), *extra_args],
        text=True, capture_output=True, check=False)


class PSkipSemanticsTests(unittest.TestCase):
    """E1①:skip_dry bool→float 升格,布尔语义位级保留(1.0/0.0 ≡ 原布尔)。"""

    def test_coerce_preserves_bool_semantics_and_rejects_out_of_range(self):
        self.assertEqual(_coerce_p_skip(True), 1.0)
        self.assertEqual(_coerce_p_skip(False), 0.0)
        self.assertEqual(_coerce_p_skip(0.5), 0.5)
        self.assertEqual(_coerce_p_skip(1), 1.0)
        for bad in (1.5, -0.1, float("nan"), float("inf"), "abc"):
            with self.assertRaises(ValueError):
                _coerce_p_skip(bad)

    def test_endpoints_replicate_bool_and_consume_no_stream(self):
        # p=1.0 恒跳过、p=0.0 恒不跳过,且专用流状态零消耗——端点行为
        # 与原布尔实现位级同构(G0-1 双端点恒等的单测对应件)。
        for p, expected in ((1.0, True), (True, True), (0.0, False),
                            (False, False)):
            env = _stub_worker_env(p, p_rng=np.random.default_rng(99))
            state_before = env._p_rng.bit_generator.state
            for _ in range(8):
                self.assertIs(env._skip_dry_draw(), expected)
            self.assertEqual(env._p_rng.bit_generator.state, state_before)

    def test_midband_draw_is_deterministic_on_dedicated_stream(self):
        env = _stub_worker_env(0.7, p_rng=_derive_p_skip_rng(12345))
        reference = _derive_p_skip_rng(12345)
        drawn = [env._skip_dry_draw() for _ in range(64)]
        expected = [float(reference.random()) < 0.7 for _ in range(64)]
        self.assertEqual(drawn, expected)
        self.assertIn(True, drawn)    # p=0.7 之 64 抽两侧皆应出现
        self.assertIn(False, drawn)

    def test_setter_sets_validates_and_returns(self):
        env = _stub_worker_env(0.0)
        self.assertEqual(env.set_skip_dry_p(0.25), 0.25)
        self.assertEqual(env.skip_dry, 0.25)
        self.assertEqual(env.set_skip_dry_p(True), 1.0)
        for bad in (2.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                env.set_skip_dry_p(bad)
        self.assertEqual(env.skip_dry, 1.0)   # 失败推送不得污染在位值

    def test_scheduled_switch_is_atomic_and_rejects_overlap(self):
        env = _stub_worker_env(1.0)
        receipt = env.schedule_skip_dry_p(0.5, 2)
        self.assertEqual(
            receipt,
            {
                "current_probability": 1.0,
                "pending_probability": 0.5,
                "remaining_env_steps": 2,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "禁止直接覆盖"):
            env.set_skip_dry_p(0.25)
        with self.assertRaisesRegex(RuntimeError, "禁止重叠"):
            env.schedule_skip_dry_p(0.25, 1)
        self.assertFalse(env._tick_skip_dry_schedule())
        self.assertEqual(env.skip_dry, 1.0)
        self.assertEqual(
            env.skip_dry_schedule_state()["remaining_env_steps"], 1)
        self.assertTrue(env._tick_skip_dry_schedule())
        self.assertEqual(env.skip_dry, 0.5)
        self.assertEqual(
            env.skip_dry_schedule_state(),
            {
                "current_probability": 0.5,
                "pending_probability": None,
                "remaining_env_steps": 0,
            },
        )

    def test_schedule_rejects_bad_countdown_without_mutation(self):
        env = _stub_worker_env(0.75)
        for bad in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                env.schedule_skip_dry_p(0.5, bad)
            self.assertEqual(
                env.skip_dry_schedule_state(),
                {
                    "current_probability": 0.75,
                    "pending_probability": None,
                    "remaining_env_steps": 0,
                },
            )

    def test_per_env_seed_derivation_is_fixed_offset_and_decoupled(self):
        # 固定偏移确定性派生:同种子同流,异种子异流;
        # 且派生流 ≠ 训练流 default_rng(seed)(专用流零染训练 RNG)。
        a = [_derive_p_skip_rng(304000).random() for _ in range(8)]
        b = [_derive_p_skip_rng(304000).random() for _ in range(8)]
        c = [_derive_p_skip_rng(304001).random() for _ in range(8)]
        train_stream = [np.random.default_rng(304000).random() for _ in range(8)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, train_stream)
        self.assertEqual(
            _derive_p_skip_rng(0).bit_generator.state,
            np.random.default_rng(_P_SKIP_SEED_OFFSET).bit_generator.state)
        # 偏移移出 [0, 2**32) 训练种子域,专用流不可能与任何 env 训练流同种子
        self.assertGreaterEqual(_P_SKIP_SEED_OFFSET, 2**32)

    def test_reset_with_seed_rederives_both_streams(self):
        env = _stub_worker_env(0.5)
        env.log_windows = False
        env.stats = {"windows": 0, "dry": 0, "fresh": 0, "ff_windows": 0,
                     "ff_dry": 0, "episodes": 0, "reseeds": 0, "reasons": {},
                     "reset_ff_reward": 0.0}
        env.oe = types.SimpleNamespace(_win={"window_id": 1})
        env._alive = False
        env._episode_seed = None
        env._new_episode = lambda seed=None, options=None: None   # 免引擎
        env._advance_to_learning_window = lambda: _AdvanceOutcome(
            np.zeros(1, dtype=np.float32), 0.0, False, False, ())
        WorkerWindowEnv.reset(env, seed=777)
        self.assertEqual(env._p_rng.bit_generator.state,
                         _derive_p_skip_rng(777).bit_generator.state)
        self.assertEqual(env._rng.bit_generator.state,
                         np.random.default_rng(777).bit_generator.state)


class _BoundaryOptions:
    """Native-free OptionsEnv shell for Worker rollout-boundary tests."""

    def __init__(self, mode):
        self.mode = mode
        self.env = types.SimpleNamespace(
            _raw={"dead": False, "dungeon_level": 1},
            death_ladder=False,
        )
        self._next_window_id = 1
        self._win = None
        self.start_window()

    def start_window(self):
        self._win = {
            "window_id": self._next_window_id,
            "W": 0.0,
        }
        self._next_window_id += 1

    def _worker_policy_observation(
            self, view, *, skip_dry_probability=0.0):
        del view
        observation = np.zeros(
            DUAL_WORKER_OBSERVATION_DIM, dtype=np.float32)
        observation[
            DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE
        ] = skip_dry_probability
        return observation

    @staticmethod
    def _worker_obs():
        return np.zeros(298, dtype=np.float32)

    def _win_step_worker(self, action):
        self._win["W"] += 1.0
        reason = {
            "open": None,
            "continue": "levelup",
            "terminal": "end",
        }[self.mode]
        return types.SimpleNamespace(
            reason=reason,
            requested_action=int(action),
            executed_action=int(action),
            fuse_tripped=False,
            fuse_requested_action=None,
        )

    def _win_end(self, reason):
        done = self.mode == "terminal"
        extra = {
            "window_id": int(self._win["window_id"]),
            "reason": reason,
            "R": float(self._win["W"]),
            "base_done": done,
            "base_trunc": False,
            "budget_boundary": False,
            "no_progress_micro_steps": 0,
            "timeout_without_progress": False,
        }
        base_info = (
            {
                "episode_extra": {
                    "xp": 0,
                    "kills": 0,
                    "char_level": 1,
                    "depth": 1,
                    "died": False,
                    "gold": 0,
                },
            }
            if done else {}
        )
        self._win = None
        return extra, base_info, done, False

    def close(self):
        pass


class _BoundaryWorker(WorkerWindowEnv):
    """WorkerWindowEnv.step() with deterministic native-free boundaries."""

    def __init__(self, mode):
        gym.Env.__init__(self)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(DUAL_WORKER_OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(15)
        self.policy_observation_view = (
            WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
        self.legacy_policy_observation_view = False
        self.skip_dry = 1.0
        self._pending_skip_dry_probability = None
        self._pending_skip_dry_remaining_env_steps = 0
        self._p_rng = np.random.default_rng(0)
        self._rng = np.random.default_rng(0)
        self._alive = True
        self._episode_seed = 0
        self.fast_forward_reward_credit = "none"
        self.additional_terminal_death_cost = 0.0
        self.log_windows = False
        self.window_log = []
        self.oe = _BoundaryOptions(mode)
        self.mode = mode
        self.reset_probabilities = []
        self.advance_probabilities = []
        self.stats = {
            "reasons": {},
            "ff_reasons": {},
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

    def reset(self, *, seed=None, options=None):
        del options
        gym.Env.reset(self, seed=seed)
        # Deliberately do not touch the pending countdown.  DummyVecEnv calls
        # this method synchronously inside step() after a terminal.
        self.reset_probabilities.append(float(self.skip_dry))
        self.oe.start_window()
        self._alive = True
        return self._policy_observation(
            self.oe._worker_obs()), {}

    def _advance_to_learning_window(self):
        self.advance_probabilities.append(float(self.skip_dry))
        self.oe.start_window()
        return _AdvanceOutcome(
            self._policy_observation(self.oe._worker_obs()),
            0.0,
            False,
            False,
            (),
        )


class PSkipAtomicBoundaryTests(unittest.TestCase):
    """The next p becomes live inside final Worker step, before VecEnv reset."""

    @staticmethod
    def _vector(mode):
        from stable_baselines3.common.vec_env import DummyVecEnv

        holder = []

        def factory():
            worker = _BoundaryWorker(mode)
            holder.append(worker)
            return worker

        vector = DummyVecEnv([factory])
        vector.reset()
        return vector, holder[0]

    def test_final_terminal_auto_reset_uses_new_probability(self):
        vector, worker = self._vector("terminal")
        try:
            vector.env_method("schedule_skip_dry_p", 0.5, 1)
            observation, _, dones, infos = vector.step(
                np.asarray([9]))
            self.assertTrue(bool(dones[0]))
            self.assertEqual(worker.reset_probabilities, [1.0, 0.5])
            self.assertEqual(worker.skip_dry, 0.5)
            self.assertEqual(
                float(observation[
                    0, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                0.5,
            )
            self.assertEqual(
                float(infos[0]["terminal_observation"][
                    DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                0.5,
            )
            self.assertEqual(
                worker.skip_dry_schedule_state(),
                {
                    "current_probability": 0.5,
                    "pending_probability": None,
                    "remaining_env_steps": 0,
                },
            )
        finally:
            vector.close()

    def test_callback_observes_terminal_auto_reset_after_atomic_switch(self):
        vector, worker = self._vector("terminal")
        try:
            model = types.SimpleNamespace(
                n_steps=1,
                get_env=lambda: vector,
                _last_obs=vector.reset(),
                _total_timesteps=2,
            )
            callback = DryCurriculumCallback((1.0, 0.5), leg_start=0)
            callback.model = model
            callback.num_timesteps = 0
            callback._on_training_start()
            callback._on_rollout_start()

            observation, _, dones, infos = vector.step(np.asarray([9]))
            self.assertTrue(bool(dones[0]))
            callback.num_timesteps = 1
            callback.locals = {
                "n_steps": 0,
                "n_rollout_steps": 1,
                "new_obs": observation,
            }
            self.assertTrue(callback._on_step())
            model._last_obs = observation
            callback._on_rollout_start()

            self.assertEqual(callback._active_index, 1)
            self.assertEqual(callback._active_p, 0.5)
            self.assertTrue(callback.pushed[-1]["boundary_preapplied"])
            self.assertEqual(worker.reset_probabilities[-1], 0.5)
            self.assertEqual(
                float(observation[
                    0, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                0.5,
            )
            self.assertEqual(
                float(infos[0]["terminal_observation"][
                    DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                0.5,
            )
        finally:
            vector.close()

    def test_final_nonterminal_advance_uses_new_probability(self):
        vector, worker = self._vector("continue")
        try:
            vector.env_method("schedule_skip_dry_p", 0.25, 1)
            observation, _, dones, _ = vector.step(np.asarray([9]))
            self.assertFalse(bool(dones[0]))
            self.assertEqual(worker.advance_probabilities, [0.25])
            self.assertEqual(worker.skip_dry, 0.25)
            self.assertEqual(
                float(observation[
                    0, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                0.25,
            )
        finally:
            vector.close()

    def test_nonfinal_reset_preserves_countdown_and_current_probability(self):
        vector, worker = self._vector("open")
        try:
            vector.env_method("schedule_skip_dry_p", 0.5, 2)
            first, _, dones, _ = vector.step(np.asarray([9]))
            self.assertFalse(bool(dones[0]))
            self.assertEqual(worker.skip_dry, 1.0)
            self.assertEqual(
                float(first[
                    0, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                1.0,
            )
            reset_observation = vector.reset()
            self.assertEqual(worker.reset_probabilities, [1.0, 1.0])
            self.assertEqual(
                worker.skip_dry_schedule_state(),
                {
                    "current_probability": 1.0,
                    "pending_probability": 0.5,
                    "remaining_env_steps": 1,
                },
            )
            self.assertEqual(
                float(reset_observation[
                    0, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                1.0,
            )
            final, _, dones, _ = vector.step(np.asarray([9]))
            self.assertFalse(bool(dones[0]))
            self.assertEqual(worker.skip_dry, 0.5)
            self.assertEqual(
                float(final[
                    0, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE]),
                0.5,
            )
        finally:
            vector.close()


class _TinyEnv(gym.Env):
    """Monitor 透传测试用微环境:复用 WorkerWindowEnv 的真 setter。"""

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,),
                                       dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    metadata = {"render_modes": []}
    skip_dry = 0.0
    set_skip_dry_p = WorkerWindowEnv.set_skip_dry_p

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


class MonitorPassthroughTests(unittest.TestCase):
    """E1①:setter 供 VecEnv env_method 经 Monitor __getattr__ 透传推送 p。"""

    def test_env_method_and_get_attr_pass_through_monitor(self):
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec = DummyVecEnv([lambda: Monitor(_TinyEnv())] * 2)
        try:
            self.assertEqual(vec.get_attr("skip_dry"), [0.0, 0.0])
            vec.env_method("set_skip_dry_p", 0.75)
            self.assertEqual(vec.get_attr("skip_dry"), [0.75, 0.75])
        finally:
            vec.close()


class DryCurriculumScheduleParserTests(unittest.TestCase):
    """E1②:--dry-curriculum-schedule 解析(显式可解析,fail-loud)。"""

    def test_main_table_literal_and_shape(self):
        self.assertEqual(_DRY_CURRICULUM_MAIN_TABLE, MAIN_TABLE_LITERAL)
        table = _parse_dry_curriculum_schedule(_DRY_CURRICULUM_MAIN_TABLE)
        self.assertEqual(len(table), 244)                    # 147+97
        self.assertEqual(244 * 2_048, 499_712)               # 恰等腿长(复核)
        self.assertEqual(table[0], 1.0)
        self.assertEqual(table[146], 0.5)                    # 线性段端点含
        self.assertEqual(set(table[147:]), {0.5})            # 97 项持平
        for k in range(147):                                 # 线性内插逐项
            self.assertAlmostEqual(table[k], 1.0 - 0.5 * k / 146, places=12)
        for earlier, later in zip(table, table[1:]):
            self.assertGreaterEqual(earlier, later)          # 单调不增

    def test_segment_grammar_roundtrip(self):
        self.assertEqual(_parse_dry_curriculum_schedule("hold:1.0:3"),
                         (1.0, 1.0, 1.0))
        self.assertEqual(_parse_dry_curriculum_schedule("linear:1.0:0.0:2"),
                         (1.0, 0.0))
        self.assertEqual(
            _parse_dry_curriculum_schedule("linear:1.0:0.5:3,hold:0.25:1"),
            (1.0, 0.75, 0.5, 0.25))

    def test_bad_formats_fail_loud(self):
        for bad in ("", "  ", "bogus", "linear:1.0:0.5", "linear:a:b:2",
                    "linear:1.0:0.5:1", "hold:0.5", "hold:0.5:0",
                    "hold:x:3", "hold:1.5:3", "hold:-0.1:3",
                    "linear:1.2:0.5:4", "linear:1.0:0.5:147;hold:0.5:97"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                _parse_dry_curriculum_schedule(bad)

    def test_native_resume_initial_probability_precedes_first_vec_reset(self):
        table = _parse_dry_curriculum_schedule(
            _DRY_CURRICULUM_MAIN_TABLE)
        index, probability = _resolve_dry_curriculum_start(
            table,
            start_steps=_DRY_CURRICULUM_LEG_START + 37 * 2_048,
            rollout_quantum=2_048,
            total_steps=10 * 2_048,
        )
        self.assertEqual(index, 37)
        self.assertEqual(probability, table[37])
        self.assertNotEqual(probability, table[0])

    def test_resume_start_and_remaining_table_coverage_fail_loud(self):
        table = _parse_dry_curriculum_schedule(
            _DRY_CURRICULUM_MAIN_TABLE)
        for start, total, message in (
            (_DRY_CURRICULUM_LEG_START - 2_048, 2_048, "rollout 边界"),
            (_DRY_CURRICULUM_LEG_START + 1, 2_048, "rollout 边界"),
            (
                _DRY_CURRICULUM_LEG_START + 240 * 2_048,
                5 * 2_048,
                "不足以覆盖",
            ),
        ):
            with self.assertRaisesRegex(ValueError, message):
                _resolve_dry_curriculum_start(
                    table,
                    start_steps=start,
                    rollout_quantum=2_048,
                    total_steps=total,
                )


class _StubVecEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.p = [0.0] * num_envs
        self.pending = [None] * num_envs
        self.remaining = [0] * num_envs
        self.pushes = []

    def env_method(self, name, *method_args):
        if name == "set_skip_dry_p":
            assert all(value is None for value in self.pending)
            self.pushes.append(float(method_args[0]))
            self.p = [float(method_args[0])] * self.num_envs
            return list(self.p)
        if name == "schedule_skip_dry_p":
            next_p, remaining = float(method_args[0]), int(method_args[1])
            assert all(value is None for value in self.pending)
            self.pending = [next_p] * self.num_envs
            self.remaining = [remaining] * self.num_envs
            return self.env_method("skip_dry_schedule_state")
        if name == "skip_dry_schedule_state":
            return [
                {
                    "current_probability": self.p[index],
                    "pending_probability": self.pending[index],
                    "remaining_env_steps": self.remaining[index],
                }
                for index in range(self.num_envs)
            ]
        raise AssertionError(name)

    def get_attr(self, name):
        assert name == "skip_dry", name
        return list(self.p)

    def consume_worker_steps(self, count):
        for _ in range(int(count)):
            for index in range(self.num_envs):
                if self.pending[index] is None:
                    continue
                self.remaining[index] -= 1
                if self.remaining[index] == 0:
                    self.p[index] = self.pending[index]
                    self.pending[index] = None


def _make_callback(table, num_envs=4, run_dir=None):
    cb = DryCurriculumCallback(table, run_dir=run_dir)
    stub = _StubVecEnv(num_envs=num_envs)
    cb.model = types.SimpleNamespace(
        n_steps=512,
        get_env=lambda: stub,
        _last_obs=np.zeros((num_envs, 298), dtype=np.float32),
        _total_timesteps=(
            _DRY_CURRICULUM_LEG_START
            + len(tuple(table)) * 512 * num_envs
        ),
    )
    cb.num_timesteps = _DRY_CURRICULUM_LEG_START
    cb._on_training_start()
    return cb, stub


def _finish_stub_rollout(cb, stub, *, dual=False):
    stub.consume_worker_steps(cb.model.n_steps)
    cb.num_timesteps += cb.quantum
    width = DUAL_WORKER_OBSERVATION_DIM if dual else 298
    new_obs = np.zeros((stub.num_envs, width), dtype=np.float32)
    if dual:
        new_obs[:, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE] = stub.p
    cb.locals = {
        "n_steps": cb.model.n_steps - 1,
        "n_rollout_steps": cb.model.n_steps,
        "new_obs": new_obs,
    }
    cb._on_step()
    cb.model._last_obs = new_obs
    return new_obs


class DryCurriculumCallbackTests(unittest.TestCase):
    """E1②:腿相对锚定、逐 rollout 落账 + 恒等断言、禁钳位。"""

    def test_leg_relative_anchoring_maps_table_by_rollout_index(self):
        table = _parse_dry_curriculum_schedule(_DRY_CURRICULUM_MAIN_TABLE)
        cb, stub = _make_callback(table)
        self.assertEqual(cb.quantum, 2_048)
        sampled = {}
        for index in range(len(table)):
            cb._on_rollout_start()
            if index in {0, 1, 146, 147, 243}:
                sampled[index] = (list(stub.p), cb.pushed[-1]["p"])
            _finish_stub_rollout(cb, stub)
        for index in (0, 1, 146, 147, 243):
            self.assertEqual(sampled[index][0], [table[index]] * 4)
            self.assertEqual(sampled[index][1], table[index])
        self.assertEqual(
            [entry["rollout_index"] for entry in cb.pushed],
            list(range(len(table))),
        )

    def test_global_step_anchoring_is_forbidden(self):
        # 腿前(全局步 0 起算)必须抛——对抗席"全局序号越界钳至表尾恒 0.5"
        # 构造在此关死;失准与越界同拒。
        table = _parse_dry_curriculum_schedule(_DRY_CURRICULUM_MAIN_TABLE)
        cb, _ = _make_callback(table)
        cb.num_timesteps = 0
        with self.assertRaisesRegex(ValueError, "腿相对锚定失义"):
            cb._on_rollout_start()
        cb.num_timesteps = _DRY_CURRICULUM_LEG_START + 1_000
        with self.assertRaisesRegex(ValueError, "边界失准"):
            cb._on_rollout_start()
        cb.num_timesteps = _DRY_CURRICULUM_LEG_START + 244 * 2_048
        with self.assertRaisesRegex(ValueError, "越界"):
            cb._on_rollout_start()

    def test_identity_assertion_fires_on_readback_mismatch(self):
        cb, stub = _make_callback((1.0, 0.5))
        stub.get_attr = lambda name: [1.0, 1.0, 0.5, 1.0]   # env[2] 在位值失配
        with self.assertRaisesRegex(ValueError, "恒等断言失配"):
            cb._on_rollout_start()

    def test_per_rollout_ledger_written_and_matches_table_prefix(self):
        table = _parse_dry_curriculum_schedule(_DRY_CURRICULUM_MAIN_TABLE)
        with tempfile.TemporaryDirectory() as d:
            cb, _ = _make_callback(table, run_dir=pathlib.Path(d))
            for index in range(6):
                cb._on_rollout_start()
                _finish_stub_rollout(cb, cb.model.get_env())
            lines = [json.loads(line) for line in
                     (pathlib.Path(d) / "dry_curriculum.jsonl")
                     .read_text().splitlines()]
        self.assertEqual(lines, cb.pushed)
        self.assertEqual([entry["p"] for entry in lines], list(table[:6]))
        self.assertEqual(
            [entry["num_timesteps"] for entry in lines],
            [_DRY_CURRICULUM_LEG_START + i * 2_048 for i in range(6)])

    def test_rollout_tail_switch_refreshes_bootstrap_and_next_cached_obs(self):
        table = (1.0, 0.5)
        stub = _StubVecEnv(num_envs=4)
        last_obs = np.zeros(
            (4, DUAL_WORKER_OBSERVATION_DIM), dtype=np.float32)
        model = types.SimpleNamespace(
            n_steps=512,
            get_env=lambda: stub,
            _last_obs=last_obs,
            _total_timesteps=_DRY_CURRICULUM_LEG_START + 2 * 2_048,
        )
        cb = DryCurriculumCallback(table)
        cb.model = model
        cb.num_timesteps = _DRY_CURRICULUM_LEG_START
        cb._on_training_start()
        cb._on_rollout_start()
        np.testing.assert_array_equal(
            last_obs[:, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE],
            np.ones(4, dtype=np.float32),
        )
        self.assertEqual(stub.pending, [0.5] * 4)
        self.assertEqual(stub.remaining, [512] * 4)

        new_obs = _finish_stub_rollout(cb, stub, dual=True)
        self.assertEqual(stub.p, [0.5] * 4)
        np.testing.assert_array_equal(
            new_obs[:, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE],
            np.full(4, 0.5, dtype=np.float32),
        )
        cb._on_rollout_start()
        self.assertTrue(cb.pushed[-1]["boundary_preapplied"])
        self.assertTrue(
            cb.pushed[-1]["cached_dual_observation_refreshed"])
        np.testing.assert_array_equal(
            model._last_obs[
                :, DUAL_WORKER_SKIP_DRY_PROBABILITY_FEATURE],
            np.full(4, 0.5, dtype=np.float32),
        )

    def test_schedule_table_exposed_verbatim_for_ledger_event(self):
        table = _parse_dry_curriculum_schedule(_DRY_CURRICULUM_MAIN_TABLE)
        cb = DryCurriculumCallback(table)
        self.assertEqual(cb.schedule_table, tuple(table))   # DRY_CURRICULUM_TABLE 供源
        self.assertEqual(cb.leg_start, 3_497_984)
        with self.assertRaisesRegex(ValueError, "不能为空"):
            DryCurriculumCallback(())
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            DryCurriculumCallback((1.0, 1.5))


def _ns(**kw):
    base = dict(worker=False, skip_dry=False, dry_curriculum_schedule=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


class FourGatePredicateTests(unittest.TestCase):
    """E1③ 四处条件门:谓词一律"干窗机制在位 = skip_dry ∨ schedule"。"""

    def test_predicate_truth_table(self):
        self.assertFalse(_dry_window_mechanism_active(_ns()))
        self.assertTrue(_dry_window_mechanism_active(_ns(skip_dry=True)))
        self.assertTrue(_dry_window_mechanism_active(
            _ns(dry_curriculum_schedule=MAIN_TABLE_LITERAL)))
        self.assertTrue(_dry_window_mechanism_active(
            _ns(skip_dry=True, dry_curriculum_schedule=MAIN_TABLE_LITERAL)))

    def test_gate_dry_cb_mount(self):
        # 门㈣(原 :2101-2104):worker ∧ (skip_dry ∨ schedule)
        self.assertFalse(_mount_dry_anchor_sentinel(_ns(worker=True)))
        self.assertFalse(_mount_dry_anchor_sentinel(_ns(skip_dry=True)))
        self.assertTrue(_mount_dry_anchor_sentinel(
            _ns(worker=True, skip_dry=True)))
        self.assertTrue(_mount_dry_anchor_sentinel(
            _ns(worker=True, dry_curriculum_schedule=MAIN_TABLE_LITERAL)))

    def _patched(self, fn, *args):
        calls = []
        original_report = train_ppo._validate_bc_report
        original_load = train_ppo._load_dry_anchor_demos
        original_capture = train_ppo._capture_file_sha256
        train_ppo._validate_bc_report = (
            lambda path, gate: calls.append(("report", str(path), gate))
            or {"demos_sha256": "d" * 64})
        train_ppo._load_dry_anchor_demos = (
            lambda path, sha: calls.append(("demos", str(path), sha))
            or (None, None, "d" * 64))
        train_ppo._capture_file_sha256 = lambda path, label: "d" * 64
        try:
            result = fn(*args)
        finally:
            train_ppo._validate_bc_report = original_report
            train_ppo._load_dry_anchor_demos = original_load
            train_ppo._capture_file_sha256 = original_capture
        return result, calls

    def test_gate_demos_precheck_fires_for_schedule_without_skip_dry(self):
        # 门㈡(原 :499-505):schedule 单独在位时预检必须跑(否则课程腿上
        # 干层锚哨兵与示范集校验静默不跑——rev3 勘正卷 E1 波及面)。
        _, calls = self._patched(
            _precheck_dry_window_demos,
            _ns(worker=True, dry_curriculum_schedule=MAIN_TABLE_LITERAL))
        self.assertEqual([c[0] for c in calls],
                         ["report", "report", "demos"])
        self.assertEqual(calls[0][2], "data_gate")
        _, calls_off = self._patched(_precheck_dry_window_demos,
                                     _ns(worker=True))
        self.assertEqual(calls_off, [])

    def test_gate_demos_sha256_capture_fires_for_schedule(self):
        # 门㈢(原 :1766-1771):机制在位才捕获 demos_sha256,否则 None。
        sha, calls = self._patched(
            _capture_dry_window_demos_sha256,
            _ns(dry_curriculum_schedule=MAIN_TABLE_LITERAL))
        self.assertEqual(sha, "d" * 64)
        self.assertEqual([c[0] for c in calls],
                         ["report", "report", "demos"])
        sha_off, calls_off = self._patched(
            _capture_dry_window_demos_sha256, _ns())
        self.assertIsNone(sha_off)
        self.assertEqual(calls_off, [])

    def test_gate_callsites_present_in_source(self):
        # 四门助手须真被四处调用点消费(防"助手在、门未接线")。
        src = TRAIN_PPO.read_text()
        self.assertIn("_require(not _dry_window_mechanism_active(args) or args.worker",
                      src)                                        # 门㈠ :445
        self.assertIn("_precheck_dry_window_demos(args)", src)     # 门㈡ :499
        self.assertIn("demos_sha256 = _capture_dry_window_demos_sha256(args)",
                      src)                                        # 门㈢ :1767
        self.assertIn("if _mount_dry_anchor_sentinel(args) else None", src)  # 门㈣


class ValueRecordLiteralTests(unittest.TestCase):
    """E1③ 两处值记录:skip_dry 键保留 CLI 旗字面值,禁被谓词覆写(rev3 勘正)。"""

    @staticmethod
    def _contract(skip_dry, schedule):
        # rev12 契约读取 contextual graft 三旗；本组只验证不在位形制。
        args = types.SimpleNamespace(
            worker=True, options=False, flat_clock=False, arch="mlp",
            max_steps=3000, num_envs=4, n_steps=512, gamma=1.0, lr=3e-4,
            ent_coef=0.005, skip_dry=skip_dry,
            dry_curriculum_schedule=schedule, no_drink_sovereignty=False,
            bc_aux_lambda=0.0, bc_aux_demos=None, bc_aux_graft=False,
            bc_aux_liveness_preflight=False,
            distill_beta=0.0, calib_record_only=False,
            legacy_worker_policy_observation_view=True)
        model = types.SimpleNamespace(
            action_space=types.SimpleNamespace(n=15), device="cpu",
            observation_space=types.SimpleNamespace(shape=(298,)),
            max_grad_norm=0.5)
        return _training_contract(args, model, batch_size=256)

    def test_contract_skip_dry_is_cli_literal_not_predicate(self):
        # L-cur/L-full 形制:--dry-curriculum-schedule 在位、--skip-dry 未携
        # → 契约 skip_dry 必须为 False(跨案取证锚,按字面施工"六门一律改写"
        # 将误记 True,核认 CONFIRMED 勘正)。
        contract = self._contract(skip_dry=False, schedule=MAIN_TABLE_LITERAL)
        self.assertIs(contract["skip_dry"], False)
        self.assertIs(self._contract(True, None)["skip_dry"], True)
        self.assertIs(self._contract(False, None)["skip_dry"], False)

    def test_contract_revision_follows_single_source_now_rev25(self):
        # rev25 binds the strengthened formal on-policy PG /9 audit.
        self.assertEqual(_CONTRACT_REVISION, 25)
        self.assertEqual(
            dict(train_ppo._REGISTERED_DUAL_WORKER_PG_AUDIT_SCHEMAS),
            {
                24: "diablogym-worker-onpolicy-pg/8",
                25: "diablogym-worker-onpolicy-pg/9",
            },
        )
        contract = self._contract(False, MAIN_TABLE_LITERAL)
        self.assertEqual(contract["contract_revision"], 25)
        self.assertEqual(contract["dry_curriculum"],
                         {"schedule": MAIN_TABLE_LITERAL})
        self.assertIs(contract["legacy_policy_observation_view"], True)
        self.assertEqual(contract["actor_migration"], "disabled")

    def test_source_records_cli_literals_and_legacy_print_uses_constant(self):
        src = TRAIN_PPO.read_text()
        self.assertIn('"skip_dry": bool(args.skip_dry),', src)   # :382 契约键
        self.assertIn('"skip_dry": args.skip_dry,', src)          # :1815 config 回执
        # legacy 打印(:404)连带改:修订号取单一真源常量,不再写死 "4"
        self.assertIn("contract_revision {_CONTRACT_REVISION} 契约", src)
        self.assertNotIn("写入 contract_revision 4 契约", src)


class CliGateSubprocessTests(unittest.TestCase):
    """E1 CLI 面:--help 文档化、两旗互斥、仅 worker、p 表覆盖闸(fail-loud)。"""

    def test_help_documents_schedule_flag_and_main_table(self):
        run = _run_cli("--help")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--dry-curriculum-schedule", run.stdout)
        self.assertIn(MAIN_TABLE_LITERAL, run.stdout.replace("\n", ""))

    def test_two_flags_mutually_exclusive(self):
        run = _run_cli("--worker",
                       "--legacy-worker-policy-observation-view", "--skip-dry",
                       "--dry-curriculum-schedule", MAIN_TABLE_LITERAL)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--skip-dry 与 --dry-curriculum-schedule 互斥", run.stderr)

    def test_schedule_requires_worker(self):
        run = _run_cli("--dry-curriculum-schedule", MAIN_TABLE_LITERAL)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("只能与 --worker 同用", run.stderr)

    def test_schedule_must_cover_leg_no_clamping(self):
        run = _run_cli(
            "--worker", "--legacy-worker-policy-observation-view",
            "--dry-curriculum-schedule", "hold:1.0:2")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("不足以覆盖", run.stderr)

    def test_bad_schedule_format_fails_loud(self):
        run = _run_cli(
            "--worker", "--legacy-worker-policy-observation-view",
            "--dry-curriculum-schedule", "bogus:1:2")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("未知段类型", run.stderr)


if __name__ == "__main__":
    unittest.main()
