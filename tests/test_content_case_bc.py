"""内容案 E2 乙1′ 采集件之自包含快速回归(PREREG-内容案-课⑤x④乙 E2/E7;
不启动引擎/训练)。

覆盖(E2 施工面逐字):
- 教师 v2 触发谓词真值表(hp 边界 0.5 闭 / 0.65 开;OC 阈 0.70;belt 前置);
- 迟滞保护:每窗预防饮 ≤1(提案即闩;begin_window 系唯一复位口;
  overridden 剔除拍亦耗闩);反射态 fail-loud(排水失守禁静默采);
- 禁采断言世代条件化镜像:v1 禁 (11,12) 原封 / v2 禁 11 允 12 分别成文;
- v2 demos schema 逐样本 masks(env.action_masks() 现场捕获逐字入档,
  反推口径系第二真源禁用;m[11] 恒 False、标签须在掩码内);
- n₁₂ 闸与 recall 门计算件(分母 = held-out 实标 a12 态;fail-closed
  零覆盖记 0.0 不消失;逐局分解 / 12/13 类占比 / 腰带经济读数);
- v1/v2 回执验证器互斥(v1 验证器对 v2 件必炸,反向亦然)+ v2 篡改矩阵
  (含 rev4 十二附二④ 补铸之 demos 实测字节断言,镜像 policy 侧形制);
- main_v2 回执键集合恰等 + FAIL 拒写权重 + 类加权重试限 v1 面质量闸
  (n₁₂/recall_12 永不触发——类加权系 N12 已除名预案);
- v1 面回归零破坏(canonical 路径 / schema_version=1 / 采集行为原封)。
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

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

import bc_worker  # noqa: E402
import train_ppo  # noqa: E402
from bc_worker import (  # noqa: E402
    _BC_V2_PASS_KEYS,
    _BC_V2_REPORT_NAME,
    _BC_V2_REPORT_SCHEMA_VERSION,
    _N12_GATE_MIN,
    _PREVENTIVE_THRESHOLD_MAIN,
    _PREVENTIVE_THRESHOLD_OC,
    _RECALL12_GATE_MIN,
    _REGISTERED_PREVENTIVE_THRESHOLDS,
    TEACHER_GENERATION_V1,
    TEACHER_GENERATION_V2,
    TeacherV2,
    _n12_readings,
    _recall12_from_model,
    _save_demos_v2,
    _validate_bc_v2_report,
    forbidden_actions_for_generation,
    teacher_v2_preventive_trigger,
)
from eval_contract import PROTOCOL_VERSION  # noqa: E402

BC_WORKER = ROOT / "train" / "bc_worker.py"
BC_WORKER_SHA = hashlib.sha256(BC_WORKER.read_bytes()).hexdigest()


def _raw(hp=100, max_hp=100, belt=0, monsters=(), floor_items=(),
         progression=()):
    """免引擎 raw 字典:dispatch/谓词消费的最小键集。"""
    return {"hp": hp, "max_hp": max_hp, "belt_heals": belt,
            "player_x": 0, "player_y": 0,
            "monsters": list(monsters), "floor_items": list(floor_items),
            "progression_targets": list(progression)}


def _in_band(belt=2, **kw):
    """预防带内态(hp=0.55)+ 贴身怪(dispatch 农期非 12 分支 → 9)。"""
    return _raw(hp=55, max_hp=100, belt=belt,
                monsters=[{"x": 1, "y": 0}], **kw)


def _out_band():
    """带外常态(hp 满、空腰带、空场)→ dispatch 农期兜底 10。"""
    return _raw(hp=100, max_hp=100, belt=0)


def _teacher_env(raw, gear=False):
    """TeacherV2.action 消费的最小 env 壳(免引擎)。"""
    base_masks = np.zeros(15, dtype=bool)
    base_masks[14] = gear
    base = types.SimpleNamespace(_raw=raw, action_masks=lambda: base_masks)
    return types.SimpleNamespace(oe=types.SimpleNamespace(env=base))


def _default_worker_masks():
    m = np.ones(15, dtype=bool)
    m[11] = False   # 11 恒掩归经理(worker 视角)
    return m


class _ScriptedEnv:
    """免引擎 WorkerWindowEnv 替身:按脚本回放窗口拍序列。

    script_for_seed(seed) → [窗1拍列表, 窗2拍列表, ...];每拍 = dict(
    raw=..., overridden=False, mask=None)。窗内末拍 term=True;窗序尽
    next_window → None(绝不滚新局,与真环境同款纪律)。
    """

    def __init__(self, script_for_seed):
        self._script_for_seed = script_for_seed
        self.stats = {"episodes": 0, "reseeds": 0}
        self._windows = None
        self._w = self._b = 0
        self.closed = False
        base = types.SimpleNamespace(
            _raw=None, action_masks=lambda: np.zeros(15, dtype=bool))
        self.oe = types.SimpleNamespace(env=base)

    def _beat(self):
        return self._windows[self._w][self._b]

    def _sync(self):
        self.oe.env._raw = self._beat()["raw"]

    def _obs(self):
        return np.zeros(298, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.stats["episodes"] += 1
        self._windows = self._script_for_seed(seed)
        self._w = self._b = 0
        self._sync()
        return self._obs(), {"episode_seed": seed}

    def action_masks(self):
        mask = self._beat().get("mask")
        return (np.asarray(mask, dtype=bool).copy() if mask is not None
                else _default_worker_masks())

    def step(self, action):
        overridden = bool(self._beat().get("overridden"))
        last = self._b == len(self._windows[self._w]) - 1
        if not last:
            self._b += 1
            self._sync()
        return self._obs(), 0.0, last, False, {"overridden": overridden}

    def next_window(self):
        self._w += 1
        self._b = 0
        if self._w >= len(self._windows):
            return None
        self._sync()
        return self._obs()

    def close(self):
        self.closed = True


class _PatchMixin:
    """unittest 侧 monkeypatch(bc_worker 模块属性,addCleanup 复位)。"""

    def _patch(self, name, value):
        original = getattr(bc_worker, name)
        setattr(bc_worker, name, value)
        self.addCleanup(setattr, bc_worker, name, original)

    def _install_env(self, script_for_seed):
        self._patch("WorkerWindowEnv",
                    lambda npz, max_steps=3000, rng_seed=0:
                    _ScriptedEnv(script_for_seed))


class TriggerPredicateTests(unittest.TestCase):
    """E2①:教师 v2 前置分支谓词真值表(hp 边界 0.5 闭 / 阈开)。"""

    def test_truth_table_main_threshold(self):
        t = teacher_v2_preventive_trigger
        self.assertTrue(t(_raw(hp=50, max_hp=100, belt=1)))    # 下界闭 hp=0.5
        self.assertTrue(t(_raw(hp=64, max_hp=100, belt=3)))    # 带内
        self.assertFalse(t(_raw(hp=49, max_hp=100, belt=1)))   # <0.5 归脑干反射
        self.assertFalse(t(_raw(hp=65, max_hp=100, belt=1)))   # 上界开 hp=0.65
        self.assertFalse(t(_raw(hp=100, max_hp=100, belt=8)))  # 满血
        self.assertFalse(t(_raw(hp=60, max_hp=100, belt=0)))   # belt=0 前置不满足
        self.assertTrue(t(_raw(hp=1, max_hp=2, belt=1)))       # 比例口径 hp/max_hp

    def test_oc_threshold_widens_band_upward_only(self):
        t = teacher_v2_preventive_trigger
        self.assertTrue(t(_raw(hp=65, max_hp=100, belt=1), 0.70))   # 主案外/OC 内
        self.assertTrue(t(_raw(hp=69, max_hp=100, belt=1), 0.70))
        self.assertFalse(t(_raw(hp=70, max_hp=100, belt=1), 0.70))  # 上界开
        self.assertFalse(t(_raw(hp=49, max_hp=100, belt=1), 0.70))  # 下界不动

    def test_registered_constants(self):
        self.assertEqual(bc_worker._PREVENTIVE_HP_LOW, 0.5)
        self.assertEqual(_PREVENTIVE_THRESHOLD_MAIN, 0.65)
        self.assertEqual(_PREVENTIVE_THRESHOLD_OC, 0.70)
        self.assertEqual(_REGISTERED_PREVENTIVE_THRESHOLDS, (0.65, 0.70))
        self.assertEqual(_N12_GATE_MIN, 122)
        self.assertEqual(_RECALL12_GATE_MIN, 0.5)


class TeacherV2HysteresisTests(unittest.TestCase):
    """E2①:每窗预防饮 ≤1 迟滞保护(主案已定条款)+ 前置分支语义。"""

    def test_unregistered_threshold_rejected(self):
        for bad in (0.6, 0.75, 0.5, 1.0):
            with self.assertRaisesRegex(ValueError, "未注册"):
                TeacherV2(bad)
        TeacherV2(0.65)
        TeacherV2(0.70)

    def test_preventive_branch_is_preposed_before_dispatch(self):
        # 带内 + 贴身怪:dispatch 会出 9,前置分支必须先出 12。
        teacher = TeacherV2()
        teacher.begin_window()
        self.assertEqual(teacher.action(_teacher_env(_in_band())), 12)

    def test_at_most_one_preventive_per_window(self):
        teacher = TeacherV2()
        teacher.begin_window()
        env = _teacher_env(_in_band())
        self.assertEqual(teacher.action(env), 12)   # 窗首触发态实标 a12
        self.assertEqual(teacher.action(env), 9)    # 闩在位 → dispatch(贴身怪)
        self.assertEqual(teacher.action(env), 9)
        teacher.begin_window()                       # 新窗复位
        self.assertEqual(teacher.action(env), 12)

    def test_latch_engaged_until_first_begin_window(self):
        # 构造后未开窗即不触发:begin_window 系唯一复位口,漏接线 fail-visible。
        teacher = TeacherV2()
        self.assertEqual(teacher.action(_teacher_env(_in_band())), 9)

    def test_reflex_state_fails_loud(self):
        # hp<0.5∧belt>0 反射态原理上不可见(排水兜底);见到即禁静默采。
        teacher = TeacherV2()
        teacher.begin_window()
        with self.assertRaisesRegex(RuntimeError, "排水失守"):
            teacher.action(_teacher_env(_raw(hp=40, max_hp=100, belt=1)))

    def test_out_of_band_delegates_to_dispatch_verbatim(self):
        teacher = TeacherV2()
        teacher.begin_window()
        self.assertEqual(teacher.action(_teacher_env(_out_band())), 10)
        self.assertEqual(
            teacher.action(_teacher_env(_out_band(), gear=True)), 14)
        heal_floor = _raw(hp=100, max_hp=100, belt=2,
                          floor_items=[{"heal": True}])
        self.assertEqual(teacher.action(_teacher_env(heal_floor)), 13)


class GenerationConditionedForbiddenTests(_PatchMixin, unittest.TestCase):
    """E2③:禁采断言世代条件化(v1 禁 (11,12) 原封;v2 禁 11 允 12)。"""

    def test_generation_table(self):
        self.assertEqual(forbidden_actions_for_generation(1), (11, 12))
        self.assertEqual(forbidden_actions_for_generation(1),
                         tuple(train_ppo._WORKER_BC_FORBIDDEN_ACTIONS))
        self.assertEqual(forbidden_actions_for_generation(2), (11,))
        self.assertEqual(TEACHER_GENERATION_V1, 1)
        self.assertEqual(TEACHER_GENERATION_V2, 2)
        for bad in (0, 3, "1"):
            with self.assertRaisesRegex(ValueError, "未知教师世代"):
                forbidden_actions_for_generation(bad)

    def test_v1_source_assertion_verbatim(self):
        # v1 路径断言原封(源文级镜像,防被"共用助手"静默改写)。
        src = BC_WORKER.read_text()
        self.assertIn("np.isin(labels, _WORKER_BC_FORBIDDEN_ACTIONS).any()", src)
        self.assertIn("示范集含禁采动作 11/12", src)

    def test_v1_collect_rejects_a12_pool(self):
        # v1 教师系 dispatch 直连:反射态 → 12 入池 → 禁采断言必炸。
        self._install_env(lambda seed: [[{"raw": _raw(hp=40, max_hp=100,
                                                      belt=1)}]])
        with self.assertRaisesRegex(RuntimeError, "禁采动作 11/12"):
            bc_worker.collect()

    def test_v2_collect_admits_a12(self):
        self._install_env(
            lambda seed: [[{"raw": _in_band()}, {"raw": _out_band()}]])
        X, labels, groups, masks, belts = bc_worker.collect_v2()
        n = 2 * len(bc_worker.DEMO_SEEDS)
        self.assertEqual(X.shape, (n, 298))
        self.assertEqual(labels.shape, (n,))
        # 每局一窗:窗首带内态实标 12,次拍带外 → 10;真实入池非反事实。
        self.assertEqual(list(labels[:2]), [12, 10])
        self.assertEqual(int((labels == 12).sum()), len(bc_worker.DEMO_SEEDS))
        self.assertTrue(np.array_equal(np.unique(groups),
                                       np.asarray(bc_worker.DEMO_SEEDS)))

    def test_v2_collect_rejects_a11_pool(self):
        # dispatch 农期原生不出 11;以 monkeypatch 注入 11 镜像 v2 禁采面。
        mask_11_open = _default_worker_masks()
        mask_11_open[11] = True   # 先绕过掩码守卫,专测标签级禁采断言
        self._install_env(
            lambda seed: [[{"raw": _out_band(), "mask": mask_11_open}]])
        self._patch("dispatch", lambda mode, raw, gear: 11)
        with self.assertRaisesRegex(RuntimeError, "禁采动作 11"):
            bc_worker.collect_v2()

    def test_v2_hysteresis_consumes_latch_on_overridden_proposal(self):
        # 提案即闩:预防饮拍被保险丝改写(整拍剔除)后,同窗不再补发。
        def script(seed):
            if seed == 100:
                return [[{"raw": _in_band(), "overridden": True},
                         {"raw": _in_band()}, {"raw": _out_band()}]]
            return [[{"raw": _in_band()}, {"raw": _out_band()}]]

        self._install_env(script)
        _, labels, groups, _, _ = bc_worker.collect_v2()
        self.assertEqual(int((labels[groups == 100] == 12).sum()), 0)
        for seed in (101, 150, 227):
            self.assertEqual(int((labels[groups == seed] == 12).sum()), 1)

    def test_v2_latch_resets_per_window(self):
        # 每局两窗、窗内两带内拍:每窗恰 1 预防饮(≤1 且逐窗复位)。
        self._install_env(lambda seed: [
            [{"raw": _in_band()}, {"raw": _in_band()}],
            [{"raw": _in_band()}, {"raw": _in_band()}]])
        _, labels, groups, _, _ = bc_worker.collect_v2()
        for seed in (100, 227):
            per_episode = labels[groups == seed]
            self.assertEqual(list(per_episode), [12, 9, 12, 9])


class MasksSchemaTests(_PatchMixin, unittest.TestCase):
    """E2⑤:v2 demos 逐样本 masks——现场捕获系唯一真源(反推禁用)。"""

    def test_masks_captured_verbatim_from_action_masks(self):
        mask_a = _default_worker_masks()
        mask_b = _default_worker_masks()
        mask_b[[3, 7]] = False   # 可辨识花纹:捕获必须逐字,非自 obs 反推
        self._install_env(lambda seed: [[{"raw": _in_band(), "mask": mask_a},
                                         {"raw": _out_band(), "mask": mask_b}]])
        _, labels, _, masks, _ = bc_worker.collect_v2()
        self.assertEqual(masks.dtype, np.bool_)
        self.assertEqual(masks.shape, (2 * len(bc_worker.DEMO_SEEDS), 15))
        self.assertTrue(np.array_equal(masks[0], mask_a))
        self.assertTrue(np.array_equal(masks[1], mask_b))
        self.assertTrue(masks[labels == 12][:, 12].all())   # a12 对 m[12]=True
        self.assertFalse(masks[:, 11].any())                # 11 恒掩归经理

    def test_collect_v2_rejects_proposal_outside_live_mask(self):
        blocked = _default_worker_masks()
        blocked[12] = False   # 现场掩码不含 12,预防提案即 on-manifold 破坏
        self._install_env(lambda seed: [[{"raw": _in_band(), "mask": blocked}]])
        with self.assertRaisesRegex(RuntimeError, "不在现场掩码内"):
            bc_worker.collect_v2()

    def test_save_demos_v2_schema_and_guards(self):
        n = 4
        X = np.zeros((n, 298), dtype=np.float32)
        labels = np.asarray([12, 9, 10, 13], dtype=np.int64)
        groups = np.asarray([100, 100, 101, 101], dtype=np.int64)
        masks = np.tile(_default_worker_masks(), (n, 1))
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            path = _save_demos_v2(out, X, labels, groups, masks)
            with np.load(path) as z:
                self.assertEqual(set(z.files),
                                 {"X", "Y", "episode_id", "masks"})
                self.assertEqual(z["masks"].shape, (n, 15))
                self.assertEqual(z["masks"].dtype, np.bool_)
                self.assertTrue(np.array_equal(z["Y"], labels))
            bad_label = masks.copy()
            bad_label[0, 12] = False   # a12 样本 m[12] 必须 True
            with self.assertRaisesRegex(RuntimeError, "不在掩码内"):
                _save_demos_v2(out, X, labels, groups, bad_label)
            bad_11 = masks.copy()
            bad_11[2, 11] = True
            with self.assertRaisesRegex(RuntimeError, r"m\[11\] 必须恒 False"):
                _save_demos_v2(out, X, labels, groups, bad_11)
            with self.assertRaisesRegex(RuntimeError, "形状/dtype"):
                _save_demos_v2(out, X, labels, groups,
                               masks.astype(np.int64))
            with self.assertRaisesRegex(RuntimeError, "形状/dtype"):
                _save_demos_v2(out, X, labels, groups, masks[:, :14])


class N12RecallGateTests(unittest.TestCase):
    """E2⑥:n₁₂ 闸与 recall 门计算件(fail-closed;逐局分解;经济读数)。"""

    def test_n12_readings_breakdown_and_shares(self):
        labels = np.asarray([12, 9, 13, 12, 9], dtype=np.int64)
        groups = np.asarray([100, 100, 101, 101, 101], dtype=np.int64)
        belts = np.asarray([3, 0, 5, 2, 1], dtype=np.int64)
        r = _n12_readings(labels, groups, belts)
        self.assertEqual(r["n12"], 2)
        self.assertEqual(r["n12_by_episode"], {"100": 1, "101": 1})
        self.assertEqual(r["class_share_12"], 0.4)
        self.assertEqual(r["class_share_13"], 0.2)
        self.assertEqual(r["belt_economy"]["belt_mean_at_a12"], 2.5)
        self.assertEqual(r["belt_economy"]["belt_mean_overall"], 2.2)
        self.assertEqual(r["belt_economy"]["a13_pairs"], 1)

    def test_n12_zero_fail_closed_readings_do_not_disappear(self):
        labels = np.asarray([9, 9, 10], dtype=np.int64)
        groups = np.asarray([100, 100, 101], dtype=np.int64)
        belts = np.asarray([0, 1, 2], dtype=np.int64)
        r = _n12_readings(labels, groups, belts)
        self.assertEqual(r["n12"], 0)                       # 记 0 不消失
        self.assertEqual(r["n12_by_episode"], {})
        self.assertEqual(r["class_share_12"], 0.0)
        self.assertEqual(r["belt_economy"]["belt_mean_at_a12"], 0.0)
        self.assertLess(r["n12"], _N12_GATE_MIN)            # 闸必拦

    @staticmethod
    def _fixed_model(pred_rows):
        class _M:
            def eval(self):
                pass

            def __call__(self, x):
                logits = torch.full((x.shape[0], 15), -1.0)
                for i, cls in enumerate(pred_rows[:x.shape[0]]):
                    logits[i, cls] = 1.0
                return logits
        return _M()

    def _split(self, groups):
        _, ho, holdout_episodes = bc_worker.split_by_episode(groups)
        return ho, holdout_episodes

    def test_recall12_denominator_is_held_out_labeled_a12_states(self):
        # 8 局各 2 对;held-out 局(确定性 rng(23) 切分)内 1 对实标 a12。
        episodes = np.arange(100, 108, dtype=np.int64)
        groups = np.repeat(episodes, 2)
        _, holdout_episodes = self._split(groups)
        held = int(holdout_episodes[0])
        labels = np.where(
            (groups == held) & (np.arange(len(groups)) % 2 == 0), 12, 9
        ).astype(np.int64)
        X = np.zeros((len(groups), 4), dtype=np.float32)
        ho, _ = self._split(groups)
        # held-out 序:局内首对系 a12 → 命中模型第 0 行出 12 → recall 1.0
        model = self._fixed_model([12] + [9] * (len(ho) - 1))
        recall, denominator = _recall12_from_model(model, X, labels, groups)
        self.assertEqual(denominator, 1)
        self.assertEqual(recall, 1.0)
        # 同分母、argmax 脱靶 → 0.0(度量 = held-out argmax 命中)
        miss = self._fixed_model([9] * len(ho))
        recall_miss, denom_miss = _recall12_from_model(miss, X, labels, groups)
        self.assertEqual(denom_miss, 1)
        self.assertEqual(recall_miss, 0.0)

    def test_recall12_zero_coverage_fail_closed(self):
        # held-out 局零 a12 覆盖:记 0.0 不消失(承 bc_worker 类召回先例),
        # 分母 0 在册 → recall 门必拦,禁 all([]) 式静默 PASS。
        episodes = np.arange(100, 108, dtype=np.int64)
        groups = np.repeat(episodes, 2)
        _, holdout_episodes = self._split(groups)
        held = int(holdout_episodes[0])
        labels = np.where(groups != held, 12, 9).astype(np.int64)
        X = np.zeros((len(groups), 4), dtype=np.float32)
        ho, _ = self._split(groups)
        model = self._fixed_model([12] * len(ho))
        recall, denominator = _recall12_from_model(model, X, labels, groups)
        self.assertEqual(denominator, 0)
        self.assertEqual(recall, 0.0)
        self.assertLess(recall, _RECALL12_GATE_MIN)


def _v2_pass_record(policy_bytes=b"v2-policy", impl_sha="a" * 64, **overrides):
    rec = {
        "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
        "teacher_generation": 2,
        "preventive_threshold": 0.65,
        "pairs": 1000, "held_out_top1": 0.99, "held_out_pairs": 100,
        "held_out_episodes": [100], "class_recalls": {"9": 0.99},
        "class_weighted_retry": False,
        "n12": 244, "n12_gate_min": 122,
        "n12_by_episode": {"178": 120, "187": 124},
        "recall_12": 0.8, "recall_12_denominator": 24,
        "recall_12_gate_min": 0.5,
        "class_share_12": 0.02, "class_share_13": 0.03,
        "belt_economy": {"belt_mean_at_a12": 2.5, "belt_mean_overall": 3.0,
                         "a13_pairs": 30},
        "data_gate": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": impl_sha,
        "generator_sha256": BC_WORKER_SHA,
        "manager_npz_sha256": "c" * 64,
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "demos_sha256": "b" * 64,
    }
    rec.update(overrides)
    return rec


class ValidatorIsolationTests(unittest.TestCase):
    """E2④:回执 schema 隔离——v1/v2 验证器互斥,v2 篡改矩阵 fail-loud。"""

    def test_v2_schema_identifier_is_isolated_from_v1(self):
        self.assertEqual(train_ppo._BC_REPORT_SCHEMA_VERSION, 1)  # v1 一字不动
        self.assertNotEqual(_BC_V2_REPORT_SCHEMA_VERSION, 1)
        self.assertIsInstance(_BC_V2_REPORT_SCHEMA_VERSION, str)  # 非 int:v1
        # 验证器 _is_plain_int 断言对 v2 件双重不相容
        self.assertEqual(_BC_V2_REPORT_NAME, "bc_report_v2.json")
        self.assertNotEqual(_BC_V2_REPORT_NAME, "bc_report.json")
        self.assertNotEqual(set(_BC_V2_PASS_KEYS),
                            train_ppo._BC_PASS_KEYS["data_gate"])

    def test_v2_validator_accepts_canonical_pass_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            policy = pathlib.Path(d) / "policy_sd.pt"
            policy.write_bytes(b"v2-policy")
            # 修①a(rev4 十二附二④):demos 实测字节断言在位——正例须落实件
            (policy.with_name("demos.npz")).write_bytes(b"v2-demos")
            rec = _v2_pass_record(
                demos_sha256=hashlib.sha256(b"v2-demos").hexdigest())
            (policy.with_name(_BC_V2_REPORT_NAME)).write_text(json.dumps(rec))
            out = _validate_bc_v2_report(policy, "a" * 64)
            self.assertEqual(out["teacher_generation"], 2)
            self.assertEqual(out["n12"], 244)

    def test_v1_validator_blows_on_v2_receipt(self):
        # v1 验证器(train_ppo._validate_bc_report)系键集合精确等断言,
        # 对 v2 件必炸——喂 payload 免路径依赖。
        payload = json.dumps(_v2_pass_record()).encode()
        with self.assertRaisesRegex(ValueError, "字段/schema 不匹配"):
            train_ppo._validate_bc_report(
                pathlib.Path("/nonexistent/policy_sd.pt"), "data_gate",
                report_payload=payload, policy_payload=b"x")

    def test_v2_validator_blows_on_v1_receipt(self):
        v1_shaped = {key: 0 for key in train_ppo._BC_PASS_KEYS["data_gate"]}
        v1_shaped["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "互斥"):
            _validate_bc_v2_report(
                pathlib.Path("/nonexistent/policy_sd.pt"), "a" * 64,
                report_payload=json.dumps(v1_shaped).encode(),
                policy_payload=b"x")

    def _tampered(self, **overrides):
        rec = _v2_pass_record(**overrides)
        return json.dumps(rec).encode()

    def test_v2_tamper_matrix_fails_loud(self):
        p = pathlib.Path("/nonexistent/policy_sd.pt")
        cases = (
            (dict(schema_version=1), "schema 标识不符"),
            (dict(teacher_generation=1), "teacher_generation 必须为 2"),
            (dict(preventive_threshold=0.6), "预防阈未注册"),
            (dict(data_gate="FAIL"), "拒绝采信"),
            (dict(data_gate="RUNNING"), "拒绝采信"),
            (dict(n12=121), "n₁₂ 闸不满足"),
            (dict(n12=True), "n₁₂ 闸不满足"),
            (dict(recall_12=0.4), "recall 门不满足"),
            (dict(protocol_version=PROTOCOL_VERSION + 1), "协议过期"),
            (dict(generator_sha256="e" * 64), "生成器已漂移"),
            (dict(implementation_sha256="f" * 64), "身份与当前运行时不一致"),
        )
        for overrides, message in cases:
            with self.assertRaisesRegex(ValueError, message,
                                        msg=repr(overrides)):
                _validate_bc_v2_report(
                    p, "a" * 64, report_payload=self._tampered(**overrides),
                    policy_payload=b"v2-policy")
        # 权重字节篡改 → SHA 绑定必炸
        with self.assertRaisesRegex(ValueError, "SHA 不匹配"):
            _validate_bc_v2_report(p, "a" * 64,
                                   report_payload=self._tampered(),
                                   policy_payload=b"tampered")
        # 修①a(rev4 十二附二④):demos 实测字节断言(镜像 policy 侧形制)
        # ——字节篡改必炸;回执 sha 系"b"*64 字面,demos_payload 实测不合即拦
        with self.assertRaisesRegex(ValueError, "demos 与回执 SHA 不匹配"):
            _validate_bc_v2_report(p, "a" * 64,
                                   report_payload=self._tampered(),
                                   policy_payload=b"v2-policy",
                                   demos_payload=b"tampered-demos")
        # demos 缺失/不可读(无 payload 且路径无实件)→ fail-loud 不静默
        with self.assertRaisesRegex(ValueError, "demos 缺失/不可读"):
            _validate_bc_v2_report(p, "a" * 64,
                                   report_payload=self._tampered(),
                                   policy_payload=b"v2-policy")
        # demos 实测字节恒等 → 全链绿(demos_payload 正例)
        good = _v2_pass_record(
            demos_sha256=hashlib.sha256(b"v2-demos").hexdigest())
        out = _validate_bc_v2_report(p, "a" * 64,
                                     report_payload=json.dumps(good).encode(),
                                     policy_payload=b"v2-policy",
                                     demos_payload=b"v2-demos")
        self.assertEqual(out["n12"], 244)
        # 缺键/多键 → 键集合精确等必炸
        missing = _v2_pass_record()
        missing.pop("demos_sha256")
        extra = _v2_pass_record()
        extra["surprise"] = 1
        for rec in (missing, extra):
            with self.assertRaisesRegex(ValueError, "字段/schema 不匹配"):
                _validate_bc_v2_report(p, "a" * 64,
                                       report_payload=json.dumps(rec).encode(),
                                       policy_payload=b"v2-policy")


class MainV2ReceiptTests(_PatchMixin, unittest.TestCase):
    """E2②④⑥:main_v2 回执键集合恰等、FAIL 拒写权重、重试触发面钉死。"""

    @staticmethod
    def _dataset(n12=140, per_episode=40):
        episodes = np.arange(100, 108, dtype=np.int64)
        groups = np.repeat(episodes, per_episode)
        n = len(groups)
        labels = np.full(n, 9, dtype=np.int64)
        labels[:n12] = 12
        X = np.zeros((n, 4), dtype=np.float32)
        masks = np.tile(_default_worker_masks(), (n, 1))
        belts = np.full(n, 3, dtype=np.int64)
        return X, labels, groups, masks, belts

    def _run(self, tmp, dataset, top1_seq=(0.99,), recall=0.8):
        calls = []
        self.train_calls = calls   # 先挂 self:main_v2 中途抛时调用记录仍可查

        def fake_train_bc(X, Y, groups, class_weights=None):
            calls.append(class_weights)
            top1 = top1_seq[min(len(calls) - 1, len(top1_seq) - 1)]
            return object(), top1, {9: 0.99}

        self._patch("OUT_V2", pathlib.Path(tmp))
        self._patch("collect_v2", lambda thr: dataset)
        self._patch("train_bc", fake_train_bc)
        self._patch("_recall12_from_model", lambda m, X, y, g: (recall, 24))
        self._patch("export_sb3_sd", lambda m: {"w": torch.zeros(1)})
        self._patch("artifact_provenance_v2", lambda thr: {
            "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
            "teacher_generation": 2, "preventive_threshold": thr,
            "protocol_version": PROTOCOL_VERSION,
            "implementation_sha256": "a" * 64,
            "generator_sha256": BC_WORKER_SHA,
            "manager_npz_sha256": "c" * 64})
        bc_worker.main_v2(0.65)
        return calls

    def _report(self, tmp):
        return json.loads((pathlib.Path(tmp) / _BC_V2_REPORT_NAME).read_text())

    def test_pass_receipt_keys_exactly_match_registered_schema(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(d, self._dataset())
            rec = self._report(d)
            self.assertEqual(set(rec), set(_BC_V2_PASS_KEYS))
            self.assertEqual(rec["data_gate"], "PASS")
            self.assertEqual(rec["teacher_generation"], 2)
            self.assertEqual(rec["preventive_threshold"], 0.65)
            self.assertEqual(rec["n12"], 140)
            self.assertEqual(rec["n12_gate_min"], 122)
            self.assertEqual(rec["recall_12"], 0.8)
            policy = pathlib.Path(d) / "policy_sd.pt"
            self.assertTrue(policy.is_file())
            self.assertEqual(
                rec["policy_sha256"],
                hashlib.sha256(policy.read_bytes()).hexdigest())
            # 全环闭合:主权产物过 v2 专用验证器
            out = _validate_bc_v2_report(policy, "a" * 64)
            self.assertEqual(out["n12"], 140)
            # demos 落 v2 目录且携 masks(独立目录,v1 canonical 不涉)
            with np.load(pathlib.Path(d) / "demos.npz") as z:
                self.assertIn("masks", z.files)

    def test_n12_gate_fail_refuses_policy_and_names_registered_oc(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "0.65→0.70 重采一次"):
                self._run(d, self._dataset(n12=121))
            rec = self._report(d)
            self.assertEqual(rec["data_gate"], "FAIL")
            self.assertEqual(rec["n12"], 121)   # FAIL 回执读数在册不消失
            self.assertFalse((pathlib.Path(d) / "policy_sd.pt").exists())

    def test_recall12_fail_never_triggers_class_weighted_retry(self):
        # 类加权系 N12 已除名预案(待亲批):recall_12 失守只 FAIL 不重训。
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "BC-v2 数据闸 FAIL"):
                self._run(d, self._dataset(), recall=0.4)
            self.assertEqual(len(self.train_calls), 1)   # 只训一次,零重试
            rec = self._report(d)
            self.assertIs(rec["class_weighted_retry"], False)
            self.assertFalse((pathlib.Path(d) / "policy_sd.pt").exists())

    def test_v1_quality_gate_still_triggers_single_retry(self):
        with tempfile.TemporaryDirectory() as d:
            calls = self._run(d, self._dataset(), top1_seq=(0.90, 0.99))
            self.assertEqual(len(calls), 2)             # 唯一重试
            self.assertIsNone(calls[0])
            self.assertIsNotNone(calls[1])              # 第二发系类加权
            self.assertIs(self._report(d)["class_weighted_retry"], True)

    def test_unregistered_threshold_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as d:
            self._patch("OUT_V2", pathlib.Path(d))
            with self.assertRaisesRegex(ValueError, "未注册"):
                bc_worker.main_v2(0.6)
            self.assertEqual(list(pathlib.Path(d).iterdir()), [])


class V1SurfaceRegressionTests(_PatchMixin, unittest.TestCase):
    """E2 铁律:v1 面回归零破坏(canonical 路径 / 回执 schema / 采集行为)。"""

    def test_canonical_paths_isolated(self):
        self.assertEqual(bc_worker.OUT,
                         ROOT / "train" / "runs" / "bc-worker")
        self.assertEqual(bc_worker.OUT_V2,
                         ROOT / "train" / "runs" / "bc-worker-v2")
        self.assertNotEqual(bc_worker.OUT, bc_worker.OUT_V2)

    def test_v1_report_surface_verbatim(self):
        src = BC_WORKER.read_text()
        # v1 回执文件名/schema 键源文原封(_previous 归档互斥规避之根据)
        self.assertIn('"schema_version": _BC_REPORT_SCHEMA_VERSION,', src)
        self.assertIn('(OUT / "bc_report.json").write_text', src)
        self.assertIn('tmp.replace(OUT / "bc_report.json")', src)
        self.assertEqual(train_ppo._BC_REPORT_SCHEMA_VERSION, 1)

    def test_v1_collect_behavior_unchanged(self):
        # 干净脚本(dispatch → 10)+ 一发 overridden 剔除:v1 采集面原封。
        def script(seed):
            if seed == 100:
                return [[{"raw": _out_band(), "overridden": True},
                         {"raw": _out_band()}]]
            return [[{"raw": _out_band()}]]

        self._install_env(script)
        X, labels, groups = bc_worker.collect()
        self.assertEqual(X.shape, (len(bc_worker.DEMO_SEEDS), 298))
        self.assertTrue((labels == 10).all())
        self.assertTrue(np.array_equal(np.unique(groups),
                                       np.asarray(bc_worker.DEMO_SEEDS)))

    def test_cli_v1_default_and_flag_guards(self):
        run = subprocess.run(
            [sys.executable, str(BC_WORKER), "--preventive-threshold", "0.65"],
            text=True, capture_output=True, check=False)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("仅与 --v2 同用", run.stderr)
        bad = subprocess.run(
            [sys.executable, str(BC_WORKER), "--v2",
             "--preventive-threshold", "0.8"],
            text=True, capture_output=True, check=False)
        self.assertNotEqual(bad.returncode, 0)   # 未注册阈 argparse choices 拦
        help_run = subprocess.run(
            [sys.executable, str(BC_WORKER), "--help"],
            text=True, capture_output=True, check=False)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--v2", help_run.stdout)


if __name__ == "__main__":
    unittest.main()
