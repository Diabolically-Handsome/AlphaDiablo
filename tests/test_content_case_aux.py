"""内容案 E3 ④乙 辅助示范通路之自包含快速回归(PREREG-内容案-课⑤x④乙 E3/E7
对应件;不启动引擎/训练产物)。

覆盖(E3 施工面逐字):
- CLI 两旗解析与互不强制(单独给任一旗不强制对方;在位谓词 = λ_bc>0 ∧ demos);
- λ_bc=0 之零侵入(不加载/不采样/不进损失图;λ=0 双模型 policy+optimizer
  torch.equal,辅助分支实测零调用);
- v2 demos 专用验证器(镜像断言按世代分别成文:masks 键/形状/dtype、v2 禁 11
  允 12、标签-掩码一致)与 v1 面回归零破坏(v1 常量/验证器/加载器原封);
- 12 类子集过滤与图纸数据面断言(12 类示范对 m[12]=True);
- 辅助 CE 前向消费逐样本 masks(独 12 掩码 → CE 精确 0);
- demo minibatch 专用 rng 流(E1③ 形制固定偏移派生、与 p_skip 流族/训练种子域
  构造性不相交、在位/不在位两跑训练 RNG 状态轨迹逐点相等);
- 12 头梯度非零(小模型 autograd 实测 + 真 train() 指纹:λ>0 行 12 必变、
  λ=0 行 12 位级不变——反 P2 构造性零通路);
- 锚教师/β/锚公式零触碰之源码钉(圈 3/圈 5 双引)。
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch as th

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from leashed_ppo import (  # noqa: E402
    _BC_AUX_SEED_OFFSET,
    LeashedMaskablePPO,
    derive_bc_aux_rng,
)
from train_ppo import (  # noqa: E402
    _BC_AUX_MAIN_LAMBDA,
    _BC_PASS_KEYS,
    _BC_REPORT_SCHEMA_VERSION,
    _WORKER_BC_FORBIDDEN_ACTIONS,
    _WORKER_BC_V2_FORBIDDEN_ACTIONS,
    _bc_aux_active,
    _filter_bc_aux_demo_pairs,
    _load_bc_aux_demos_v2,
    _load_dry_anchor_demos,
)
from diablogym.worker_env import (  # noqa: E402
    _P_SKIP_SEED_OFFSET,
    _derive_p_skip_rng,
)

TRAIN_PPO = ROOT / "train" / "train_ppo.py"
LEASHED_PPO = ROOT / "train" / "leashed_ppo.py"


def _run_cli(*extra_args):
    return subprocess.run(
        [sys.executable, str(TRAIN_PPO), *extra_args],
        text=True, capture_output=True, check=False)


def _ns(**kw):
    import types

    base = dict(bc_aux_lambda=0.0, bc_aux_demos=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _write_v2_npz(path, n=8, labels=None, masks=None, obs_dim=298,
                  episode_ids=None, x_dtype=np.float32):
    x = np.zeros((n, obs_dim), dtype=x_dtype)
    x[: n // 2, min(297, obs_dim - 1)] = 1.0
    y = np.asarray(labels if labels is not None else [9, 12] * (n // 2),
                   dtype=np.int64)
    if masks is None:
        masks = np.ones((n, 15), dtype=bool)
        masks[:, 11] = False
    ep = np.asarray(
        episode_ids if episode_ids is not None
        else [0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64)
    np.savez_compressed(path, X=x, Y=y, episode_id=ep, masks=masks)
    return x, y, ep, masks


class TinyMasked15Env(gym.Env):
    """免引擎 15 动作微环境(Discrete(15) 使"12 头"真实存在)。"""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,),
                                       dtype=np.float32)
    action_space = gym.spaces.Discrete(15)
    metadata = {"render_modes": []}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self):
        mask = np.ones(15, dtype=bool)
        mask[11] = mask[12] = False
        return mask


def _make_model(lam=0.0, seed=7):
    env = DummyVecEnv([TinyMasked15Env])
    return LeashedMaskablePPO(
        "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
        gamma=1.0, ent_coef=0.005, distill_beta=0.0, bc_aux_lambda=lam,
        seed=seed, device="cpu", verbose=0)


def _fill_buffer(model, seed=11):
    # rollout 面 12 恒掩 → PPO 自身对 action_net 第 12 行零梯度(Adam 零梯度
    # 零更新)——12 头位级变动只可能来自 ④乙 辅助通路(反 P2 指纹构造)。
    rng = np.random.default_rng(seed)
    buf = model.rollout_buffer
    buf.reset()
    for index in range(model.n_steps):
        obs = rng.standard_normal((1, 4)).astype(np.float32)
        mask = np.ones((1, 15), dtype=bool)
        mask[:, 11] = mask[:, 12] = False
        action = np.asarray([int(rng.choice([9, 10, 13, 14]))])
        buf.add(obs, action, np.asarray([float(rng.standard_normal())]),
                np.asarray([index % 5 == 0]), th.zeros(1), th.zeros(1),
                action_masks=mask)
    buf.compute_returns_and_advantage(last_values=th.zeros(1),
                                      dones=np.zeros(1))


def _demo_bank(n=16, obs_dim=4, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, obs_dim)).astype(np.float32)
    y = np.full(n, 12, dtype=np.int64)
    masks = np.ones((n, 15), dtype=bool)
    masks[:, 11] = False
    return x, y, masks


def _tree_equal(a, b):
    if isinstance(a, th.Tensor):
        return isinstance(b, th.Tensor) and th.equal(a, b)
    if isinstance(a, dict):
        return (isinstance(b, dict) and set(a) == set(b)
                and all(_tree_equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_tree_equal(x, y) for x, y in zip(a, b)))
    return a == b


class BcAuxActivationPredicateTests(unittest.TestCase):
    """E3⑤:在位谓词 = λ_bc>0 ∧ demos 给定(两旗互不强制)。"""

    def test_predicate_truth_table(self):
        self.assertFalse(_bc_aux_active(_ns()))
        self.assertFalse(_bc_aux_active(_ns(bc_aux_lambda=0.5)))
        self.assertFalse(_bc_aux_active(_ns(bc_aux_demos="demos.npz")))
        self.assertTrue(_bc_aux_active(
            _ns(bc_aux_lambda=0.5, bc_aux_demos="demos.npz")))

    def test_main_lambda_constant_matches_d7(self):
        self.assertEqual(_BC_AUX_MAIN_LAMBDA, 0.015625)   # 与蒸馏锚 β 同量级


class BcAuxDemoValidatorTests(unittest.TestCase):
    """E3②:v2 demos 专用验证器(镜像断言按世代分别成文)。"""

    def _tmp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return pathlib.Path(directory.name)

    def test_good_v2_file_roundtrip_returns_arrays_and_sha(self):
        p = self._tmp() / "demos.npz"
        x, y, _, masks = _write_v2_npz(p)
        lx, ly, lmasks, sha = _load_bc_aux_demos_v2(p)
        self.assertEqual(sha, hashlib.sha256(p.read_bytes()).hexdigest())
        np.testing.assert_array_equal(lx, x)
        np.testing.assert_array_equal(ly, y)
        np.testing.assert_array_equal(lmasks, masks)
        self.assertIn(12, ly)                       # v2 允 12

    def test_missing_masks_key_rejected(self):
        p = self._tmp() / "demos.npz"
        x = np.zeros((6, 298), dtype=np.float32)
        y = np.asarray([9, 12] * 3, dtype=np.int64)
        ep = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        np.savez_compressed(p, X=x, Y=y, episode_id=ep)
        with self.assertRaisesRegex(ValueError, "masks"):
            _load_bc_aux_demos_v2(p)

    def test_generation_conditioned_forbidden_labels(self):
        # v2 禁 11 允 12(守卫面不弱化;v1 禁 (11,12) 系 bc_worker 面原封)
        p = self._tmp() / "demos.npz"
        _write_v2_npz(p, labels=[9, 11] * 4)
        with self.assertRaisesRegex(ValueError, "禁采动作 11"):
            _load_bc_aux_demos_v2(p)

    def test_label_out_of_range_rejected(self):
        p = self._tmp() / "demos.npz"
        _write_v2_npz(p, labels=[9, 15] * 4)
        with self.assertRaisesRegex(ValueError, "标签越界"):
            _load_bc_aux_demos_v2(p)

    def test_masks_shape_and_dtype_rejected(self):
        base = self._tmp()
        p14 = base / "d14.npz"
        _write_v2_npz(p14, masks=np.ones((8, 14), dtype=bool))
        with self.assertRaisesRegex(ValueError, "masks 形状/dtype"):
            _load_bc_aux_demos_v2(p14)
        pint = base / "dint.npz"
        _write_v2_npz(pint, masks=np.ones((8, 15), dtype=np.int8))
        with self.assertRaisesRegex(ValueError, "masks 形状/dtype"):
            _load_bc_aux_demos_v2(pint)

    def test_label_masked_by_own_mask_rejected(self):
        p = self._tmp() / "demos.npz"
        masks = np.ones((8, 15), dtype=bool)
        masks[1, 12] = False                        # 第 1 行标签 12 被自身掩码禁止
        _write_v2_npz(p, masks=masks)
        with self.assertRaisesRegex(ValueError, "on-manifold 破缺"):
            _load_bc_aux_demos_v2(p)

    def test_obs_shape_dtype_and_episode_diversity_mirrored(self):
        base = self._tmp()
        p297 = base / "d297.npz"
        _write_v2_npz(p297, obs_dim=297)
        with self.assertRaisesRegex(ValueError, "数组形状异常"):
            _load_bc_aux_demos_v2(p297)
        p64 = base / "d64.npz"
        _write_v2_npz(p64, x_dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "dtype 异常"):
            _load_bc_aux_demos_v2(p64)
        pep = base / "dep.npz"
        _write_v2_npz(pep, episode_ids=[0] * 8)
        with self.assertRaisesRegex(ValueError, "独立局数"):
            _load_bc_aux_demos_v2(pep)


class BcAuxSubsetFilterTests(unittest.TestCase):
    """E3①:主案限 12 类示范对 + 图纸数据面断言(m[12]=True)。"""

    def test_filter_keeps_only_class12_pairs(self):
        x = np.arange(12, dtype=np.float32).reshape(6, 2)
        y = np.asarray([9, 12, 10, 12, 13, 9], dtype=np.int64)
        masks = np.ones((6, 15), dtype=bool)
        fx, fy, fmasks = _filter_bc_aux_demo_pairs(x, y, masks)
        self.assertEqual(list(fy), [12, 12])
        np.testing.assert_array_equal(fx, x[[1, 3]])
        np.testing.assert_array_equal(fmasks, masks[[1, 3]])

    def test_filter_fails_loud_without_class12(self):
        x = np.zeros((3, 2), dtype=np.float32)
        y = np.asarray([9, 10, 13], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "没有 12 类示范对"):
            _filter_bc_aux_demo_pairs(x, y, np.ones((3, 15), dtype=bool))

    def test_filter_asserts_m12_true_on_class12_pairs(self):
        # 图纸字面(G0-2b/E7):全部 12 类示范对断言 m[12]=True。
        x = np.zeros((2, 2), dtype=np.float32)
        y = np.asarray([12, 12], dtype=np.int64)
        masks = np.ones((2, 15), dtype=bool)
        masks[1, 12] = False
        with self.assertRaisesRegex(ValueError, r"m\[12\]=False"):
            _filter_bc_aux_demo_pairs(x, y, masks)


class BcAuxRngTests(unittest.TestCase):
    """E3③:demo minibatch 专用流,播种规则同 E1③ 形制(固定偏移确定性派生)。"""

    def test_fixed_offset_derivation_is_deterministic(self):
        self.assertEqual(
            derive_bc_aux_rng(0).bit_generator.state,
            np.random.default_rng(_BC_AUX_SEED_OFFSET).bit_generator.state)
        a = [derive_bc_aux_rng(304000).random() for _ in range(8)]
        b = [derive_bc_aux_rng(304000).random() for _ in range(8)]
        c = [derive_bc_aux_rng(304001).random() for _ in range(8)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_stream_disjoint_from_training_and_p_skip_families(self):
        # 偏移移出 [0, 2**32) 训练种子域;与 E1 p_skip 流族 (seed+rank)+2**33+26
        # 之偏移差 ≥ 2**33 >> rank,构造性不相交(种子级同流即同序列,故查种子)。
        self.assertGreaterEqual(_BC_AUX_SEED_OFFSET, 2**32)
        self.assertGreaterEqual(abs(_BC_AUX_SEED_OFFSET - _P_SKIP_SEED_OFFSET),
                                2**33)
        base = derive_bc_aux_rng(304000).bit_generator.state
        self.assertNotEqual(
            base, np.random.default_rng(304000).bit_generator.state)
        for rank in range(64):
            self.assertNotEqual(
                base, _derive_p_skip_rng(304000 + rank).bit_generator.state)

    def test_seed_none_mirrors_worker_env_fallback(self):
        self.assertIsInstance(derive_bc_aux_rng(None), np.random.Generator)


class BcAuxTrainLoopTests(unittest.TestCase):
    """E3①④⑤:辅助 CE 通路(免引擎缩样,TinyMasked15Env + DummyVecEnv)。"""

    def _model(self, lam=0.0, seed=7):
        model = _make_model(lam=lam, seed=seed)
        self.addCleanup(model.env.close)
        return model

    def test_ctor_rejects_bad_lambda(self):
        with self.assertRaisesRegex(ValueError, "bc_aux_lambda"):
            self._model(lam=-0.1)
        with self.assertRaisesRegex(ValueError, "bc_aux_lambda"):
            self._model(lam=float("nan"))

    def test_mount_validates_shapes_masks_and_rng(self):
        model = self._model()
        x, y, masks = _demo_bank()
        bad = masks.copy()
        bad[0, 12] = False                          # 标签 12 被自身掩码禁止
        with self.assertRaisesRegex(ValueError, "掩码禁止"):
            model.mount_bc_aux_demos(x, y, bad, rng=derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "示范观测形状"):
            model.mount_bc_aux_demos(x[:, :3], y, masks, derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "掩码形状"):
            model.mount_bc_aux_demos(x, y, masks[:, :14], derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "标签"):
            model.mount_bc_aux_demos(x, y.astype(np.float32), masks,
                                     derive_bc_aux_rng(4))
        with self.assertRaisesRegex(ValueError, "专用 np.random.Generator"):
            model.mount_bc_aux_demos(x, y, masks, rng=None)

    def test_lambda_positive_without_bank_fails_loud(self):
        # 反 P2:λ>0 而池未挂载不得静默空转(镜像 β>0 无教师条款)。
        model = self._model(lam=0.015625)
        model._setup_learn(total_timesteps=8)
        _fill_buffer(model)
        with self.assertRaisesRegex(RuntimeError, "示范池未挂载"):
            model.train()

    def test_zero_lambda_is_zero_intrusion(self):
        # λ=0:辅助分支零调用、专用流零消耗、policy+optimizer 与"池不在位"
        # 双胞胎逐张量 torch.equal(G0-2a 张量级恒等之免引擎缩样)。
        calls = []
        snapshots = []
        for mounted in (True, False):
            model = self._model(lam=0.0, seed=7)
            if mounted:
                rng = derive_bc_aux_rng(3)
                state_before = rng.bit_generator.state
                model.mount_bc_aux_demos(*_demo_bank(), rng=rng)
                original = model._bc_aux_ce_loss
                model._bc_aux_ce_loss = (
                    lambda: calls.append(1) or original())
            model._setup_learn(total_timesteps=8)
            _fill_buffer(model)
            model.train()
            self.assertIsNone(model._last_bc_aux_ce)
            snapshots.append((model.policy.state_dict(),
                              model.policy.optimizer.state_dict()))
            if mounted:
                self.assertEqual(rng.bit_generator.state, state_before)
        self.assertEqual(calls, [])
        (policy_a, optim_a), (policy_b, optim_b) = snapshots
        self.assertEqual(list(policy_a), list(policy_b))
        for key in policy_a:
            self.assertTrue(th.equal(policy_a[key], policy_b[key]), key)
        self.assertTrue(_tree_equal(optim_a, optim_b))

    def test_12_head_gradient_nonzero_via_autograd(self):
        # G0-2b 单测先立件:autograd 实测辅助 CE 对 action_net 第 12 行梯度非零。
        model = self._model(lam=0.015625)
        model.mount_bc_aux_demos(*_demo_bank(), rng=derive_bc_aux_rng(1))
        ce = model._bc_aux_ce_loss()
        grad_w, grad_b = th.autograd.grad(
            ce, [model.policy.action_net.weight, model.policy.action_net.bias])
        self.assertGreater(float(grad_w[12].abs().sum()), 0.0)
        self.assertNotEqual(float(grad_b[12]), 0.0)

    def test_12_head_train_fingerprint_lambda_gated(self):
        # 真 train() 指纹:rollout 面 12 恒掩时,PPO 自身对 12 头仍有 O(1e-10)
        # 级残留梯度——sb3_contrib MaskableCategorical 之 _original_logits 系
        # 未掩 logsumexp 归一化后逻辑值,掩前归一化项对 raw logit 12 留有
        # softmax_unmasked_12 微通道(施工实测呈报件,详交接单)。故指纹取
        # 数量级分离而非位级冻结:λ>0 之行 12 位移须比 λ=0 残留位移大 ≥1e2 倍。
        deltas = {}
        for lam in (0.015625, 0.0):
            model = self._model(lam=lam)
            model.mount_bc_aux_demos(*_demo_bank(), rng=derive_bc_aux_rng(2))
            model._setup_learn(total_timesteps=8)
            _fill_buffer(model)
            before = model.policy.action_net.weight.detach().clone()
            model.train()
            after = model.policy.action_net.weight.detach()
            deltas[lam] = float((after[12] - before[12]).abs().sum())
            self.assertFalse(th.equal(before[9], after[9]))   # 其余头照常在学
            if lam > 0:
                self.assertIsInstance(model._last_bc_aux_ce, float)
            else:
                self.assertIsNone(model._last_bc_aux_ce)
        self.assertLess(deltas[0.0], 1e-7)          # λ=0:仅掩码归一化残留
        self.assertGreater(deltas[0.015625], 1e-5)  # λ>0:辅助通路真实到达 12 头
        self.assertGreater(deltas[0.015625], 1e2 * deltas[0.0])

    def test_aux_forward_consumes_per_sample_masks(self):
        # E7 masks 消费:独 12 掩码 → π(12)=1 精确 → CE 精确 0;全开掩码之
        # 随机初始网 CE 显著为正——掩码若被忽略此二读数不可分。
        model = self._model(lam=0.015625)
        x, y, _ = _demo_bank()
        only12 = np.zeros((len(y), 15), dtype=bool)
        only12[:, 12] = True
        model.mount_bc_aux_demos(x, y, only12, rng=derive_bc_aux_rng(5))
        ce_masked = float(model._bc_aux_ce_loss().detach())
        model.mount_bc_aux_demos(x, y, np.ones((len(y), 15), dtype=bool),
                                 rng=derive_bc_aux_rng(5))
        ce_open = float(model._bc_aux_ce_loss().detach())
        self.assertLess(abs(ce_masked), 1e-6)
        self.assertGreater(ce_open, 0.01)

    def test_dedicated_rng_stream_zero_contamination(self):
        # E3③/G0-2b 缩样:在位(λ>0+池挂载)/不在位两跑,训练 RNG(torch 全局 +
        # numpy 全局)状态轨迹逐点相等;专用流自身实被消费。
        trajectories = []
        consumed_states = []
        for mounted in (True, False):
            model = self._model(lam=0.015625 if mounted else 0.0, seed=7)
            rng = None
            if mounted:
                rng = derive_bc_aux_rng(304000)
                model.mount_bc_aux_demos(*_demo_bank(), rng=rng)
            model._setup_learn(total_timesteps=8)
            th.manual_seed(1234)
            np.random.seed(1234)
            states = []
            for round_seed in (21, 22):
                _fill_buffer(model, seed=round_seed)
                model.train()
                np_state = np.random.get_state()
                states.append((th.get_rng_state().numpy().tobytes(),
                               np_state[1].tobytes(), int(np_state[2])))
            trajectories.append(states)
            if mounted:
                consumed_states.append(rng.bit_generator.state)
        self.assertEqual(trajectories[0], trajectories[1])
        self.assertNotEqual(consumed_states[0],
                            derive_bc_aux_rng(304000).bit_generator.state)


class BcAuxCliTests(unittest.TestCase):
    """E3②/E7:CLI 旗解析、互不强制、fail-loud(子进程,引擎零点火)。"""

    def _tmp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return pathlib.Path(directory.name)

    def _worker_argv(self, base, *aux):
        manager = base / "manager.npz"
        if not manager.exists():
            manager.write_bytes(b"placeholder")     # 只过 is_file 闸,不被解析
        return ("--worker", "--algo", "mppo", "--gamma", "1.0",
                "--max-steps", "3000", "--manager-npz", str(manager),
                "--seed", "7000", *aux)             # 7000 撞探针段:安全终止哨

    def test_help_documents_both_flags_and_main_lambda(self):
        run = _run_cli("--help")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--bc-aux-lambda", run.stdout)
        self.assertIn("--bc-aux-demos", run.stdout)
        self.assertIn("0.015625", run.stdout.replace("\n", ""))

    def test_negative_lambda_fails_loud(self):
        run = _run_cli("--bc-aux-lambda=-0.5")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("--bc-aux-lambda 必须是有限非负数", run.stderr)

    def test_lambda_alone_does_not_force_demos(self):
        # λ 单独在位 → 通路不在位,无耦合报错;推进至下游种子纪律闸即证
        run = _run_cli(*self._worker_argv(self._tmp(),
                                          "--bc-aux-lambda", "0.015625"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("种子纪律", run.stderr)
        self.assertNotIn("④乙", run.stderr)

    def test_demos_alone_does_not_force_lambda_and_is_not_loaded(self):
        # demos 单独在位(路径故意不存在)→ 不在位即不加载、连存在性都不查(零侵入)
        run = _run_cli(*self._worker_argv(
            self._tmp(), "--bc-aux-demos", "/definitely/missing/demos.npz"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("种子纪律", run.stderr)
        self.assertNotIn("④乙", run.stderr)

    def test_active_path_requires_worker_mppo(self):
        run = _run_cli("--worker", "--bc-aux-lambda", "0.5",
                       "--bc-aux-demos", "/missing/demos.npz")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("只适用于 --worker --algo mppo", run.stderr)

    def test_active_path_missing_file_fails_loud(self):
        run = _run_cli(*self._worker_argv(
            self._tmp(), "--bc-aux-lambda", "0.5",
            "--bc-aux-demos", "/definitely/missing/demos.npz"))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("不存在", run.stderr)

    def test_active_path_rejects_v1_shaped_npz(self):
        base = self._tmp()
        v1 = base / "v1demos.npz"
        x = np.zeros((6, 298), dtype=np.float32)
        np.savez_compressed(
            v1, X=x, Y=np.asarray([9, 12] * 3, dtype=np.int64),
            episode_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
        run = _run_cli(*self._worker_argv(
            base, "--bc-aux-lambda", "0.5", "--bc-aux-demos", str(v1)))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("masks", run.stderr)

    def test_active_path_with_valid_v2_passes_aux_validation(self):
        base = self._tmp()
        v2 = base / "v2demos.npz"
        _write_v2_npz(v2)
        run = _run_cli(*self._worker_argv(
            base, "--bc-aux-lambda", "0.015625", "--bc-aux-demos", str(v2)))
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("种子纪律", run.stderr)        # 辅助面已全过,倒在安全哨
        self.assertNotIn("④乙", run.stderr)


class V1SurfaceRegressionTests(unittest.TestCase):
    """E3②:v1 面(:61 schema 常量/canonical 验证器/加载器)回归零破坏。"""

    def test_v1_constants_untouched(self):
        self.assertEqual(_BC_REPORT_SCHEMA_VERSION, 1)
        self.assertEqual(_WORKER_BC_FORBIDDEN_ACTIONS, (11, 12))
        self.assertEqual(_WORKER_BC_V2_FORBIDDEN_ACTIONS, (11,))
        self.assertEqual(_BC_PASS_KEYS["data_gate"], {
            "schema_version", "pairs", "held_out_top1", "held_out_pairs",
            "held_out_episodes", "class_recalls", "class_weighted_retry",
            "data_gate", "protocol_version", "implementation_sha256",
            "generator_sha256", "manager_npz_sha256", "policy_sha256",
            "demos_sha256",
        })

    def test_v1_loader_accepts_v1_npz_and_v2_validator_rejects_it(self):
        # 世代分账:同一 v1 形制文件——v1 加载器照常受理,v2 验证器 fail-loud。
        with tempfile.TemporaryDirectory() as directory:
            p = pathlib.Path(directory) / "demos.npz"
            x = np.zeros((6, 298), dtype=np.float32)
            x[0, 297] = 1.0
            np.savez_compressed(
                p, X=x, Y=np.asarray([9, 9, 10, 13, 9, 14], dtype=np.int64),
                episode_id=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64))
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            lx, ly, lsha = _load_dry_anchor_demos(p, sha)
            self.assertEqual(lsha, sha)
            self.assertEqual(lx.shape, (6, 298))
            self.assertEqual(len(ly), 6)
            with self.assertRaisesRegex(ValueError, "masks"):
                _load_bc_aux_demos_v2(p)


class AnchorZeroTouchSourcePins(unittest.TestCase):
    """圈 3/圈 5 双引:锚教师 KING_SD 原封、β 冻结、锚公式零触碰之源码钉。"""

    def test_leashed_anchor_formula_untouched_and_aux_guard_present(self):
        src = LEASHED_PPO.read_text()
        self.assertIn("distill_ce = -(t_probs * logp_all).sum(dim=-1).mean()",
                      src)
        self.assertIn("loss = loss + self.distill_beta * distill_ce", src)
        self.assertIn("if self.bc_aux_lambda > 0:", src)
        self.assertIn("loss = loss + self.bc_aux_lambda * bc_aux_ce", src)

    def test_train_ppo_wiring_points_present(self):
        src = TRAIN_PPO.read_text()
        self.assertIn("_load_bc_aux_demos_v2(args.bc_aux_demos)", src)
        self.assertIn("model.mount_bc_aux_demos(*bc_aux_bank, "
                      "rng=derive_bc_aux_rng(args.seed))", src)
        self.assertIn("model.bc_aux_lambda = args.bc_aux_lambda", src)


if __name__ == "__main__":
    unittest.main()
