"""内容案 E4/E5/E6/E8 基建面之自包含快速回归(PREREG-内容案-课⑤x④乙 对应件;
不启动引擎/训练产物,循 tests/test_b1_infra.py 与 tests/test_content_case.py 风格)。

覆盖:
- E4 契约 4→5:修订号单一真源、三腿形制(L-base 双键 disabled / L-cur
  dry_curriculum 载荷 / L-full 双键载荷)、skip_dry 键仍 CLI 旗字面值、
  契约与 config 回执同构增键、旧 rev4 检查点非 legacy 续训拒绝路径(R8);
- E5 三仪表:① 干/鲜 distill_ce 分列离线探针(schema/公式面/零触训练路径)、
  ② 干窗行为仪表(干态熵与 a 分布/干鲜窗工资与宽度聚合/区间清零/fail-closed)、
  ③ 金丝雀 a12/局 统计函数+记录器(封闭 schema,fail-loud);
  三仪表不在位 = 代码路径与 HEAD 等价(旋钮默认 0/挂载守卫/RNG 与参数零触);
- E6 探针集钉死:BC-v1 demos 字节 ≡ 冻结常量(真件必过、伪字节必炸、
  两处加载门接线、探针构造内断言);
- E8 launcher:bash -n 语法 + PID 簿记修复面关键逻辑(B1 判决 OPS 段
  "查错进程号"误报之修复:caffeinate 断言改进程树内;行为面不变)。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest

import gymnasium as gym
import numpy as np
import torch as th

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

import leashed_ppo  # noqa: E402
import train_ppo  # noqa: E402
from train_ppo import (  # noqa: E402
    _BC_AUX_MAIN_LAMBDA,
    _BC_V1_DEMOS_SHA256,
    _CONTRACT_REVISION,
    _DRY_CURRICULUM_MAIN_TABLE,
    DistillCeProbe,
    DryWindowMetricsCallback,
    _assert_bc_v1_demos_frozen,
    _contract_bc_aux,
    _contract_dry_curriculum,
    _training_contract,
    _validate_resume_contract,
    a12_canary_stats,
    record_a12_canary,
)

TRAIN_PPO = ROOT / "train" / "train_ppo.py"
LAUNCHER = ROOT / "train" / "launch_case.sh"
CANONICAL_DEMOS = ROOT / "train" / "runs" / "bc-worker" / "demos.npz"
MAIN_TABLE_LITERAL = "linear:1.0:0.5:147,hold:0.5:97"


def _run_cli(*extra_args):
    return subprocess.run(
        [sys.executable, str(TRAIN_PPO), *extra_args],
        text=True, capture_output=True, check=False)


def _leg_args(**kw):
    """D3 三腿共用命令形之契约相关面(worker/mppo 形制)缩样命名空间。"""
    base = dict(worker=True, options=False, flat_clock=False, arch="mlp",
                max_steps=3000, num_envs=4, n_steps=512, gamma=1.0, lr=3e-4,
                ent_coef=0.005, skip_dry=False, dry_curriculum_schedule=None,
                no_drink_sovereignty=False, bc_aux_lambda=0.0,
                bc_aux_demos=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _contract_for(args, bc_aux_demos_sha256=None):
    model = types.SimpleNamespace(
        action_space=types.SimpleNamespace(n=15), device="cpu",
        observation_space=types.SimpleNamespace(shape=(298,)))
    return _training_contract(args, model, batch_size=256,
                              bc_aux_demos_sha256=bc_aux_demos_sha256)


class Tiny298MaskedEnv(gym.Env):
    """免引擎 298 维观测 / Discrete(15) 微环境(供探针喂真策略前向)。"""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(298,),
                                       dtype=np.float32)
    action_space = gym.spaces.Discrete(15)
    metadata = {"render_modes": []}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(298, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(298, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self):
        mask = np.ones(15, dtype=bool)
        mask[11] = mask[12] = False
        return mask


def _real_policy(seed=7):
    from sb3_contrib import MaskablePPO

    env = DummyVecEnv([Tiny298MaskedEnv])
    return MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
                       seed=seed, device="cpu", verbose=0)


def _teacher_298(preferred_action=9):
    bias = th.full((15,), -10.0)
    bias[preferred_action] = 10.0
    sd = {
        "mlp_extractor.policy_net.0.weight": th.zeros(64, 298),
        "mlp_extractor.policy_net.0.bias": th.zeros(64),
        "mlp_extractor.policy_net.2.weight": th.zeros(64, 64),
        "mlp_extractor.policy_net.2.bias": th.zeros(64),
        "action_net.weight": th.zeros(15, 64),
        "action_net.bias": bias,
    }
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "teacher.pt"
        th.save(sd, p)
        return leashed_ppo.build_teacher(str(p))


def _fake_demos(directory, n=8):
    """伪字节示范件(形制合法、字节非冻结常量)。"""
    path = pathlib.Path(directory) / "demos.npz"
    x = np.zeros((n, 298), dtype=np.float32)
    x[: n // 2, 297] = 1.0
    y = np.full(n, 9, dtype=np.int64)
    episode_id = np.asarray([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    np.savez_compressed(path, X=x, Y=y, episode_id=episode_id)
    return path


class ContractRev5Tests(unittest.TestCase):
    """E4:契约 4→5,三腿统一;dry_curriculum/bc_aux 双键;skip_dry 字面值。"""

    def test_revision_constant_is_5(self):
        self.assertEqual(_CONTRACT_REVISION, 5)

    def test_l_base_carries_disabled_keys(self):
        # L-base(--skip-dry):双键均 disabled——同案零双版本(圈 7)。
        contract = _contract_for(_leg_args(skip_dry=True))
        self.assertEqual(contract["contract_revision"], 5)
        self.assertIs(contract["skip_dry"], True)          # CLI 旗字面值
        self.assertEqual(contract["dry_curriculum"], "disabled")
        self.assertEqual(contract["bc_aux"], "disabled")

    def test_l_cur_carries_schedule_payload(self):
        contract = _contract_for(
            _leg_args(dry_curriculum_schedule=MAIN_TABLE_LITERAL))
        self.assertIs(contract["skip_dry"], False)          # 字面值,禁谓词覆写
        self.assertEqual(contract["dry_curriculum"],
                         {"schedule": MAIN_TABLE_LITERAL})
        self.assertEqual(contract["bc_aux"], "disabled")

    def test_l_full_carries_both_payloads(self):
        contract = _contract_for(
            _leg_args(dry_curriculum_schedule=MAIN_TABLE_LITERAL,
                      bc_aux_lambda=_BC_AUX_MAIN_LAMBDA,
                      bc_aux_demos="runs/bc-worker-v2/demos.npz"),
            bc_aux_demos_sha256="a" * 64)
        self.assertEqual(contract["dry_curriculum"],
                         {"schedule": MAIN_TABLE_LITERAL})
        self.assertEqual(contract["bc_aux"],
                         {"lambda": _BC_AUX_MAIN_LAMBDA,
                          "demos_sha256": "a" * 64})
        json.dumps(contract)   # stdlib JSON 可序列化(status.json/zip 内嵌)

    def test_main_table_literal_matches_frozen_constant(self):
        self.assertEqual(MAIN_TABLE_LITERAL, _DRY_CURRICULUM_MAIN_TABLE)

    def test_bc_aux_payload_requires_sha_when_active(self):
        args = _leg_args(bc_aux_lambda=_BC_AUX_MAIN_LAMBDA,
                         bc_aux_demos="x.npz")
        with self.assertRaisesRegex(ValueError, "缺 bc-worker-v2 示范集 sha256"):
            _contract_bc_aux(args, None)
        with self.assertRaisesRegex(ValueError, "缺 bc-worker-v2 示范集 sha256"):
            _contract_for(args)   # 契约装配面同炸

    def test_helper_truth_table(self):
        self.assertEqual(_contract_dry_curriculum(_leg_args()), "disabled")
        self.assertEqual(
            _contract_dry_curriculum(_leg_args(
                dry_curriculum_schedule=MAIN_TABLE_LITERAL)),
            {"schedule": MAIN_TABLE_LITERAL})
        # 两旗互不强制:单独任一旗 → 不在位 → disabled(E3 谓词单一真源)
        self.assertEqual(_contract_bc_aux(
            _leg_args(bc_aux_lambda=_BC_AUX_MAIN_LAMBDA), None), "disabled")
        self.assertEqual(_contract_bc_aux(
            _leg_args(bc_aux_demos="x.npz"), None), "disabled")

    def test_config_receipt_mirrors_contract_keys_in_source(self):
        # 契约与 config 回执同构增键:两助手各被两处消费(契约 + 回执);
        # skip_dry 回执键仍 CLI 旗字面值(rev3 勘正原封)。
        src = TRAIN_PPO.read_text()
        self.assertEqual(
            src.count('"dry_curriculum": _contract_dry_curriculum(args),'), 2)
        self.assertEqual(
            src.count('"bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),'),
            2)
        self.assertIn('"skip_dry": bool(args.skip_dry),', src)   # 契约键
        self.assertIn('"skip_dry": args.skip_dry,', src)          # 回执键

    def test_legacy_print_follows_constant(self):
        src = TRAIN_PPO.read_text()
        self.assertIn("contract_revision {_CONTRACT_REVISION} 契约", src)
        self.assertNotIn("写入 contract_revision 4 契约", src)
        self.assertNotIn("写入 contract_revision 5 契约", src)   # 禁写死


class Rev4CheckpointRejectionTests(unittest.TestCase):
    """E4/R8:旧检查点(rev4 契约)非 legacy 续训将拒——已知代价照录。"""

    @staticmethod
    def _rev4_saved():
        # 模拟 E4 之前(rev4)腿存进 zip 的契约:无 dry_curriculum/bc_aux 键。
        contract = _contract_for(_leg_args(skip_dry=True))
        saved = {key: value for key, value in contract.items()
                 if key not in ("dry_curriculum", "bc_aux")}
        saved["contract_revision"] = 4
        return saved

    def test_rev4_resume_rejected_with_drift(self):
        current = _contract_for(_leg_args(skip_dry=True))
        with self.assertRaisesRegex(ValueError, "契约漂移") as ctx:
            _validate_resume_contract(self._rev4_saved(), current)
        message = str(ctx.exception)
        self.assertIn("contract_revision", message)
        self.assertIn("dry_curriculum", message)
        self.assertIn("bc_aux", message)

    def test_legacy_route_unchanged(self):
        # 无契约旧件仍走显式 --allow-legacy-resume 一次性迁移口(R8 原语义)。
        current = _contract_for(_leg_args(skip_dry=True))
        with self.assertRaisesRegex(ValueError, "无 training_contract"):
            _validate_resume_contract(None, current)
        _validate_resume_contract(None, current, allow_legacy_resume=True)

    def test_rev5_self_consistent_resume_passes(self):
        args = _leg_args(dry_curriculum_schedule=MAIN_TABLE_LITERAL)
        _validate_resume_contract(_contract_for(args), _contract_for(args))


class E6FrozenDemosTests(unittest.TestCase):
    """E6:三腿仪表探针一律钉 BC-v1 demos 字节(冻结常量,加载处断言)。"""

    def test_frozen_constant_matches_canonical_bytes(self):
        self.assertEqual(
            _BC_V1_DEMOS_SHA256,
            "3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363")
        actual = hashlib.sha256(CANONICAL_DEMOS.read_bytes()).hexdigest()
        self.assertEqual(actual, _BC_V1_DEMOS_SHA256)
        self.assertEqual(_assert_bc_v1_demos_frozen(CANONICAL_DEMOS),
                         _BC_V1_DEMOS_SHA256)

    def test_fake_bytes_explode(self):
        with tempfile.TemporaryDirectory() as d:
            fake = _fake_demos(d)
            with self.assertRaisesRegex(ValueError, "BC-v1 demos 字节漂移"):
                _assert_bc_v1_demos_frozen(fake)
            with self.assertRaisesRegex(ValueError, "不可读"):
                _assert_bc_v1_demos_frozen(pathlib.Path(d) / "absent.npz")

    def test_probe_constructors_pin_frozen_bytes(self):
        # 伪字节必炸:探针加载即断言(expected = 冻结常量本体)。
        with tempfile.TemporaryDirectory() as d:
            fake = _fake_demos(d)
            with self.assertRaisesRegex(ValueError, "demos SHA 不匹配"):
                DistillCeProbe(pathlib.Path(d), str(fake), every=49_152)
            with self.assertRaisesRegex(ValueError, "demos SHA 不匹配"):
                DryWindowMetricsCallback(pathlib.Path(d), str(fake),
                                         every=49_152)

    def test_dry_window_gates_wired_to_frozen_assert(self):
        # 两处 E1 门(demos/BC 预检、demos_sha256 捕获)接线冻结断言;
        # DryAnchorSentinel 之 expected_sha256 由捕获门供值,传递性钉死。
        src = TRAIN_PPO.read_text()
        self.assertEqual(src.count("_assert_bc_v1_demos_frozen(demos)"), 2)
        self.assertIn("_load_dry_anchor_demos(\n            demos_npz,"
                      " _BC_V1_DEMOS_SHA256)", src.replace("    ", "    "))
        # v2 面不受钉:BC-v2 仅经 --bc-aux-demos 进辅助损失(canonical 不动)
        self.assertNotIn("_assert_bc_v1_demos_frozen(args.bc_aux_demos)", src)


class DistillCeProbeTests(unittest.TestCase):
    """E5①:干/鲜 distill_ce 分列离线探针(孪生 DryAnchorSentinel 形制)。"""

    @classmethod
    def setUpClass(cls):
        cls.model = _real_policy()
        cls.teacher = _teacher_298()

    def _probe(self, run_dir, every=49_152):
        cb = DistillCeProbe(run_dir, str(CANONICAL_DEMOS), every=every)
        cb.model = types.SimpleNamespace(
            policy=self.model.policy, teacher=self.teacher, device="cpu",
            distill_beta=0.015625)
        return cb

    def test_emit_schema_and_group_split(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._probe(run_dir)
            cb.num_timesteps = 3_547_136
            cb._emit(final=False)
            lines = [json.loads(line) for line in
                     (run_dir / "distill_ce_probe.jsonl").read_text()
                     .strip().splitlines()]
            self.assertEqual(len(lines), 1)
            line = lines[0]
            self.assertEqual(
                set(line), {"probe", "step", "dry_ce", "dry_n", "fresh_ce",
                            "fresh_n", "beta", "mask_mode", "demos_sha16"})
            self.assertEqual(line["probe"], "distill-ce")
            self.assertEqual(line["step"], 3_547_136)
            self.assertEqual(line["mask_mode"], "dry-anchor-legacy")
            self.assertEqual(line["demos_sha16"], _BC_V1_DEMOS_SHA256[:16])
            self.assertEqual(line["beta"], 0.015625)
            self.assertGreater(line["dry_n"], 0)
            self.assertGreater(line["fresh_n"], 0)
            for key in ("dry_ce", "fresh_ce"):
                self.assertTrue(np.isfinite(line[key]))
                self.assertGreater(line[key], 0.0)   # 随机学生对尖教师 CE>0

    def test_zero_touch_of_params_and_global_rng(self):
        # 零触训练路径:探针构造+发射不改策略参数、不耗全局 RNG(专用 rng)。
        with tempfile.TemporaryDirectory() as d:
            before_params = [p.detach().clone()
                            for p in self.model.policy.parameters()]
            torch_state = th.get_rng_state().clone()
            np_state = np.random.get_state()
            cb = self._probe(pathlib.Path(d))
            cb.num_timesteps = 3_547_136
            cb._emit(final=False)
            for before, after in zip(before_params,
                                     self.model.policy.parameters()):
                self.assertTrue(th.equal(before, after.detach()))
            self.assertTrue(th.equal(torch_state, th.get_rng_state()))
            after_np = np.random.get_state()
            self.assertEqual(np_state[0], after_np[0])
            self.assertTrue(np.array_equal(np_state[1], after_np[1]))
            self.assertEqual(np_state[2:], after_np[2:])

    def test_missing_teacher_fails_loud(self):
        with tempfile.TemporaryDirectory() as d:
            cb = DistillCeProbe(pathlib.Path(d), str(CANONICAL_DEMOS),
                                every=49_152)
            cb.model = types.SimpleNamespace(
                policy=self.model.policy, teacher=None, device="cpu")
            cb.num_timesteps = 100
            with self.assertRaisesRegex(ValueError, "需教师在位"):
                cb._emit(final=False)

    def test_cadence_aligns_to_next_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            cb = self._probe(pathlib.Path(d))
            cb.num_timesteps = 3_497_984
            cb._on_training_start()
            self.assertEqual(cb.next_at, 3_538_944)   # ((⌊·/49152⌋)+1)×49152
        with self.assertRaisesRegex(ValueError, "间隔必须 > 0"):
            DistillCeProbe(pathlib.Path("."), str(CANONICAL_DEMOS), every=0)

    def test_ce_formula_mirrors_leashed_rubberband(self):
        # 公式面:−Σ t_probs·logp_all 均值(镜像 leashed_ppo train() :356)。
        src = TRAIN_PPO.read_text()
        self.assertIn("-(t_probs * logp_all).sum(dim=-1).mean()", src)
        leashed_src = (ROOT / "train" / "leashed_ppo.py").read_text()
        self.assertIn("-(t_probs * logp_all).sum(dim=-1).mean()", leashed_src)


class DryWindowMetricsTests(unittest.TestCase):
    """E5②:干窗行为仪表(干态熵/a 分布 + 干/鲜窗工资与宽度,只记不裁)。"""

    @classmethod
    def setUpClass(cls):
        cls.model = _real_policy(seed=11)

    def _cb(self, run_dir, every=49_152):
        cb = DryWindowMetricsCallback(run_dir, str(CANONICAL_DEMOS),
                                      every=every)
        cb.model = types.SimpleNamespace(policy=self.model.policy,
                                         device="cpu")
        return cb

    @staticmethod
    def _info(dry, W, tau, dlvl_end):
        return {"option_extra": {"dry": dry, "W": W, "tau": tau,
                                 "dlvl_end": dlvl_end}}

    def test_window_aggregation_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._cb(run_dir)
            cb.num_timesteps = 10_000            # 未到界:只聚合不发射
            cb.locals = {"infos": [
                self._info(True, -0.5, 20, 3), self._info(True, -0.1, 30, 3),
                self._info(False, 2.0, 40, 4), {"overridden": False}]}
            self.assertTrue(cb._on_step())
            self.assertFalse((run_dir / "drywin_metrics.jsonl").exists())
            cb.num_timesteps = 49_152            # 到界发射
            cb.locals = {"infos": []}
            self.assertTrue(cb._on_step())
            line = json.loads(
                (run_dir / "drywin_metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(
                set(line), {"metrics", "step", "dry_state_entropy",
                            "dry_state_n", "dry_state_argmax_hist", "windows",
                            "mask_mode", "demos_sha16"})
            self.assertEqual(line["metrics"], "drywin")
            self.assertEqual(line["step"], 49_152)
            self.assertEqual(line["mask_mode"], "dry-anchor-legacy")
            self.assertEqual(line["demos_sha16"], _BC_V1_DEMOS_SHA256[:16])
            self.assertEqual(line["windows"]["dry"],
                             {"n": 2, "wage_mean": -0.3, "tau_mean": 25.0,
                              "depth_mean": 3.0})
            self.assertEqual(line["windows"]["fresh"],
                             {"n": 1, "wage_mean": 2.0, "tau_mean": 40.0,
                              "depth_mean": 4.0})
            self.assertTrue(np.isfinite(line["dry_state_entropy"]))
            hist = line["dry_state_argmax_hist"]
            self.assertEqual(len(hist), 15)
            self.assertEqual(sum(hist), line["dry_state_n"])
            self.assertEqual(hist[11], 0)   # 旧口径掩码:11/12 恒掩
            self.assertEqual(hist[12], 0)

    def test_interval_reset_and_fail_closed_zero_coverage(self):
        # 区间清零:发射后无新窗 → n=0 组记 n:0、均值 null 不消失。
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._cb(run_dir)
            cb.locals = {"infos": [self._info(True, -0.5, 20, 3)]}
            cb.num_timesteps = 49_152
            cb._on_step()
            cb.locals = {"infos": []}
            cb.num_timesteps = 98_304
            cb._on_step()
            lines = [json.loads(raw) for raw in
                     (run_dir / "drywin_metrics.jsonl").read_text()
                     .strip().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[1]["windows"]["dry"],
                             {"n": 0, "wage_mean": None, "tau_mean": None,
                              "depth_mean": None})
            self.assertEqual(lines[1]["windows"]["fresh"]["n"], 0)

    def test_final_emit_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            cb = self._cb(run_dir)
            cb.num_timesteps = 49_152
            cb.locals = {"infos": []}
            cb._on_step()
            cb._on_training_end()   # 同步点已发射 → 去重不重写
            lines = (run_dir / "drywin_metrics.jsonl").read_text() \
                .strip().splitlines()
            self.assertEqual(len(lines), 1)
            cb.num_timesteps = 60_000
            cb._on_training_end()   # 尾部未对齐点 → final 行
            lines = [json.loads(raw) for raw in
                     (run_dir / "drywin_metrics.jsonl").read_text()
                     .strip().splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertIs(lines[1]["final"], True)


class A12CanaryTests(unittest.TestCase):
    """E5③:金丝雀 a12/局 统计函数 + 记录器(检查点离线序列用,只记不裁)。"""

    def test_stats_correctness(self):
        stats = a12_canary_stats([0, 0, 4, 1])
        self.assertEqual(stats, {"episodes": 4, "a12_total": 5,
                                 "a12_per_episode": 1.25,
                                 "episodes_with_a12": 2, "a12_max": 4})
        zero = a12_canary_stats([0] * 32)
        self.assertEqual(zero["a12_per_episode"], 0.0)   # 零行使如实记 0.0
        self.assertEqual(zero["episodes"], 32)

    def test_stats_fail_loud(self):
        for bad in ([], [1, -1], [1.5], [True], [1, None]):
            with self.assertRaises(ValueError):
                a12_canary_stats(bad)

    def test_recorder_schema_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "a12_canary.jsonl"
            stats = a12_canary_stats([0, 3])
            line = record_a12_canary(out, checkpoint_step=3_596_288,
                                     manager="M29", stats=stats,
                                     tag="lfull-canary")
            written = json.loads(out.read_text().strip())
            self.assertEqual(written, line)
            self.assertEqual(written["canary"], "a12")
            self.assertEqual(written["schema_version"], "a12-canary/1")
            self.assertEqual(written["checkpoint_step"], 3_596_288)
            self.assertEqual(written["manager"], "M29")
            self.assertEqual(written["tag"], "lfull-canary")
            self.assertEqual(written["a12_per_episode"], 1.5)
            # tag 缺省行不携 tag 键(schema 封闭可判)
            second = record_a12_canary(out, checkpoint_step=0,
                                       manager="H", stats=stats)
            self.assertNotIn("tag", second)

    def test_recorder_fail_loud(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "a12_canary.jsonl"
            stats = a12_canary_stats([1])
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=-1, manager="H",
                                  stats=stats)
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=1, manager="",
                                  stats=stats)
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=1, manager="H",
                                  stats={"episodes": 1})
            extra = dict(stats, rogue=1)
            with self.assertRaises(ValueError):
                record_a12_canary(out, checkpoint_step=1, manager="H",
                                  stats=extra)
            self.assertFalse(out.exists())   # fail-loud 不落半行


class E5ZeroIntrusionTests(unittest.TestCase):
    """E5:不在位 = 代码路径与 HEAD 等价(G0-2a 先决;旋钮默认 0+挂载守卫)。"""

    def test_cli_defaults_are_off(self):
        src = TRAIN_PPO.read_text()
        self.assertIn('ap.add_argument("--distill-ce-probe-every", type=int,'
                      " default=0,", src)
        self.assertIn('ap.add_argument("--drywin-metrics-every", type=int,'
                      " default=0,", src)

    def test_mount_guards_gate_on_knobs(self):
        src = TRAIN_PPO.read_text()
        self.assertIn("if (args.worker and args.distill_ce_probe_every > 0)",
                      src)
        self.assertIn("if (args.worker and args.drywin_metrics_every > 0)", src)
        self.assertIn("([distill_ce_cb] if distill_ce_cb else [])", src)
        self.assertIn("([drywin_cb] if drywin_cb else [])", src)

    def test_a12_functions_have_no_training_path_caller(self):
        # ③ 系离线统计件:训练进程内零调用(定义处各恰一现)。
        src = TRAIN_PPO.read_text()
        self.assertEqual(src.count("def a12_canary_stats("), 1)
        self.assertEqual(src.count("a12_canary_stats("), 1)
        self.assertEqual(src.count("def record_a12_canary("), 1)
        self.assertEqual(src.count("record_a12_canary("), 1)

    def test_cli_rejects_bad_knobs_and_documents_them(self):
        help_run = _run_cli("--help")
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--distill-ce-probe-every", help_run.stdout)
        self.assertIn("--drywin-metrics-every", help_run.stdout)
        bad = _run_cli("--total-steps", "2048",
                       "--distill-ce-probe-every", "-1")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("--distill-ce-probe-every 不能为负", bad.stderr)
        nonworker = _run_cli("--total-steps", "2048",
                             "--distill-ce-probe-every", "49152")
        self.assertNotEqual(nonworker.returncode, 0)
        self.assertIn("只适用于 --worker --algo mppo", nonworker.stderr)
        nonworker2 = _run_cli("--total-steps", "2048",
                              "--drywin-metrics-every", "49152")
        self.assertNotEqual(nonworker2.returncode, 0)
        self.assertIn("--drywin-metrics-every 只适用于 --worker",
                      nonworker2.stderr)

    def test_probe_teacher_preflight_wired(self):
        src = TRAIN_PPO.read_text()
        self.assertIn("--distill-ce-probe-every>0 需 Leashed 教师在位", src)


class LauncherFixTests(unittest.TestCase):
    """E8:launch_case.sh PID 簿记误报修复(B1 判决 OPS 段 minor)。"""

    def test_bash_syntax(self):
        run = subprocess.run(["bash", "-n", str(LAUNCHER)],
                             text=True, capture_output=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_tree_assertion_replaces_wrong_pid_check(self):
        src = LAUNCHER.read_text()
        # 修复面:caffeinate 断言改"$PID 进程树内"(本体或子进程双形兼容)。
        self.assertIn('pgrep -P "$PID" -f caffeinate', src)
        self.assertIn('ps -o command= -p "$PID" | grep -q "caffeinate"', src)
        self.assertIn("进程树内无 caffeinate", src)
        self.assertNotIn("不是 caffeinate 进程", src)   # 旧误报文案退场

    def test_behavior_surface_unchanged(self):
        src = LAUNCHER.read_text()
        # 行为面不变:nohup+caffeinate -is+孤儿化+日志+receipt+心跳细则照旧。
        self.assertIn('nohup caffeinate -is "$PY" "$DRIVER" "$@"', src)
        self.assertIn('disown "$PID"', src)
        self.assertIn("launch_receipt.json", src)
        self.assertIn("progress.jsonl", src)
        self.assertIn("exit 70", src)
        self.assertIn("exit 71", src)
        self.assertIn("exit 64", src)
        self.assertIn("set -euo pipefail", src)

    def test_liveness_check_precedes_tree_assertion(self):
        src = LAUNCHER.read_text()
        self.assertLess(src.index('if ! ps -p "$PID"'),
                        src.index('pgrep -P "$PID" -f caffeinate'))


if __name__ == "__main__":
    unittest.main()
