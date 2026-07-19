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
    _training_contract,
)
from diablogym.worker_env import (  # noqa: E402
    _P_SKIP_SEED_OFFSET,
    WorkerWindowEnv,
    _coerce_p_skip,
    _derive_p_skip_rng,
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
                     "ff_dry": 0, "episodes": 0, "reseeds": 0, "reasons": {}}
        env.oe = types.SimpleNamespace(_win=None)
        env._alive = False
        env._episode_seed = None
        env._new_episode = lambda seed=None, options=None: None   # 免引擎
        env.next_window = lambda: np.zeros(1, dtype=np.float32)
        WorkerWindowEnv.reset(env, seed=777)
        self.assertEqual(env._p_rng.bit_generator.state,
                         _derive_p_skip_rng(777).bit_generator.state)
        self.assertEqual(env._rng.bit_generator.state,
                         np.random.default_rng(777).bit_generator.state)


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


class _StubVecEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.p = [0.0] * num_envs
        self.pushes = []

    def env_method(self, name, *method_args):
        assert name == "set_skip_dry_p", name
        self.pushes.append(float(method_args[0]))
        self.p = [float(method_args[0])] * self.num_envs

    def get_attr(self, name):
        assert name == "skip_dry", name
        return list(self.p)


def _make_callback(table, num_envs=4, run_dir=None):
    cb = DryCurriculumCallback(table, run_dir=run_dir)
    stub = _StubVecEnv(num_envs=num_envs)
    cb.model = types.SimpleNamespace(n_steps=512, get_env=lambda: stub)
    cb.num_timesteps = _DRY_CURRICULUM_LEG_START
    cb._on_training_start()
    return cb, stub


class DryCurriculumCallbackTests(unittest.TestCase):
    """E1②:腿相对锚定、逐 rollout 落账 + 恒等断言、禁钳位。"""

    def test_leg_relative_anchoring_maps_table_by_rollout_index(self):
        table = _parse_dry_curriculum_schedule(_DRY_CURRICULUM_MAIN_TABLE)
        cb, stub = _make_callback(table)
        self.assertEqual(cb.quantum, 2_048)
        for index in (0, 1, 146, 147, 243):
            cb.num_timesteps = _DRY_CURRICULUM_LEG_START + index * 2_048
            cb._on_rollout_start()
            self.assertEqual(stub.p, [table[index]] * 4)
        self.assertEqual([e["rollout_index"] for e in cb.pushed],
                         [0, 1, 146, 147, 243])
        self.assertEqual([e["p"] for e in cb.pushed],
                         [table[i] for i in (0, 1, 146, 147, 243)])

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
                cb.num_timesteps = _DRY_CURRICULUM_LEG_START + index * 2_048
                cb._on_rollout_start()
            lines = [json.loads(line) for line in
                     (pathlib.Path(d) / "dry_curriculum.jsonl")
                     .read_text().splitlines()]
        self.assertEqual(lines, cb.pushed)
        self.assertEqual([entry["p"] for entry in lines], list(table[:6]))
        self.assertEqual(
            [entry["num_timesteps"] for entry in lines],
            [_DRY_CURRICULUM_LEG_START + i * 2_048 for i in range(6)])

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
        train_ppo._validate_bc_report = (
            lambda path, gate: calls.append(("report", str(path), gate))
            or {"demos_sha256": "d" * 64})
        train_ppo._load_dry_anchor_demos = (
            lambda path, sha: calls.append(("demos", str(path), sha))
            or (None, None, "d" * 64))
        try:
            result = fn(*args)
        finally:
            train_ppo._validate_bc_report = original_report
            train_ppo._load_dry_anchor_demos = original_load
        return result, calls

    def test_gate_demos_precheck_fires_for_schedule_without_skip_dry(self):
        # 门㈡(原 :499-505):schedule 单独在位时预检必须跑(否则课程腿上
        # 干层锚哨兵与示范集校验静默不跑——rev3 勘正卷 E1 波及面)。
        _, calls = self._patched(
            _precheck_dry_window_demos,
            _ns(worker=True, dry_curriculum_schedule=MAIN_TABLE_LITERAL))
        self.assertEqual([c[0] for c in calls], ["report", "demos"])
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
        self.assertEqual([c[0] for c in calls], ["report", "demos"])
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
        # E4 改写(相应单测改写而非删除):rev5 契约新读 bc_aux 两旗,
        # 命名空间补默认不在位值(0.0/None);E1 断言面原封。
        args = types.SimpleNamespace(
            worker=True, options=False, flat_clock=False, arch="mlp",
            max_steps=3000, num_envs=4, n_steps=512, gamma=1.0, lr=3e-4,
            ent_coef=0.005, skip_dry=skip_dry,
            dry_curriculum_schedule=schedule, no_drink_sovereignty=False,
            bc_aux_lambda=0.0, bc_aux_demos=None)
        model = types.SimpleNamespace(
            action_space=types.SimpleNamespace(n=15), device="cpu",
            observation_space=types.SimpleNamespace(shape=(298,)))
        return _training_contract(args, model, batch_size=256)

    def test_contract_skip_dry_is_cli_literal_not_predicate(self):
        # L-cur/L-full 形制:--dry-curriculum-schedule 在位、--skip-dry 未携
        # → 契约 skip_dry 必须为 False(跨案取证锚,按字面施工"六门一律改写"
        # 将误记 True,核认 CONFIRMED 勘正)。
        contract = self._contract(skip_dry=False, schedule=MAIN_TABLE_LITERAL)
        self.assertIs(contract["skip_dry"], False)
        self.assertIs(self._contract(True, None)["skip_dry"], True)
        self.assertIs(self._contract(False, None)["skip_dry"], False)

    def test_contract_revision_follows_single_source_now_rev5(self):
        # E4 施工后改写(相应单测改写而非删除,PREREG-内容案 E4/R8):
        # E1 时代"仍为 4"围栏由 E4 条款接替——修订号单一真源升 5,
        # rev5 键 dry_curriculum 随 schedule 在位携载荷(形制断言细面在
        # tests/test_content_case_infra.py)。
        self.assertEqual(_CONTRACT_REVISION, 5)
        contract = self._contract(False, MAIN_TABLE_LITERAL)
        self.assertEqual(contract["contract_revision"], 5)
        self.assertEqual(contract["dry_curriculum"],
                         {"schedule": MAIN_TABLE_LITERAL})

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
        run = _run_cli("--worker", "--skip-dry",
                       "--dry-curriculum-schedule", MAIN_TABLE_LITERAL)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--skip-dry 与 --dry-curriculum-schedule 互斥", run.stderr)

    def test_schedule_requires_worker(self):
        run = _run_cli("--dry-curriculum-schedule", MAIN_TABLE_LITERAL)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("只能与 --worker 同用", run.stderr)

    def test_schedule_must_cover_leg_no_clamping(self):
        run = _run_cli("--worker", "--dry-curriculum-schedule", "hold:1.0:2")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("不足以覆盖", run.stderr)

    def test_bad_schedule_format_fails_loud(self):
        run = _run_cli("--worker", "--dry-curriculum-schedule", "bogus:1:2")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("未知段类型", run.stderr)


if __name__ == "__main__":
    unittest.main()
