"""内容案 E2 乙1′ 采集件之自包含快速回归(PREREG-内容案-课⑤x④乙 E2/E7;
不启动引擎/训练)。

覆盖(E2 施工面逐字):
- 教师 v2 触发谓词真值表(hp 边界 0.5 闭 / 0.65 开；旧 0.70 OC 禁用;
  belt 前置);
- 教师预防饮目标只依 298 维可见 hp/mask/主动饮位，同观测不受隐藏历史
  影响；overridden 拍剔除后下一可见带内态仍保持 a12；反射态 fail-loud;
- 禁采断言世代条件化镜像:v1 禁 (11,12) 原封 / v2 禁 11 允 12 分别成文;
- v2 demos schema 逐样本 masks(env.action_masks() 现场捕获逐字入档,
  反推口径系第二真源禁用;m[11] 恒 False、标签须在掩码内);
- n₁₂ 闸与 recall 门计算件(分母 = held-out 实标 a12 态;fail-closed
  零覆盖记 0.0 不消失;逐局分解 / 12/13 类占比 / 腰带经济读数);
- v1/v2 回执验证器互斥(v1 验证器对 v2 件必炸,反向亦然)+ v2 篡改矩阵
  (含 rev4 十二附二④ 补铸之 demos 实测字节断言,镜像 policy 侧形制);
- main_v2 回执键集合恰等 + FAIL 拒写权重 + 类加权重试限 v1 面质量闸
  (n₁₂/recall_12 永不触发——类加权系 N12 已除名预案);
- v1/v2 候选选择使用训练局内的独立 validation；篡改 final held-out 标签
  不改变 retry 决策、类权或所选模型；
- v1 面回归零破坏(canonical 路径 / schema_version=1 / 采集行为原封);
- 方案甲(2026-07-19 亲批):v2 采集局数 ×3；当前 v1/v2 active registry 为
  未查看的 2104000..2104127 / 2103000..2103383 固定池，旧 2100000 /
  2101000 池保持 burned + v2 主训类平衡加权 CE
  (w_c = N/(K·n_c) 手算恒等;v1 调用路径不加权)+ 回执新字段
  (collection_episodes / class_weights)与验证器篡改矩阵。
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
from unittest import mock

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

import bc_worker  # noqa: E402
import train_ppo  # noqa: E402
from diablogym.options_env import (  # noqa: E402
    OptionsEnv,
    VOLUNTARY_DRINK_HP_HIGH,
    VOLUNTARY_DRINK_HP_LOW,
    WORKER_DRINK_LATCH_FEATURE,
)
from bc_worker import (  # noqa: E402
    _BC_V2_PASS_KEYS,
    _BC_V2_REPORT_NAME,
    _BC_V2_REPORT_SCHEMA_VERSION,
    _N12_GATE_MIN,
    _PREVENTIVE_THRESHOLD_MAIN,
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


def _visible_obs(raw, prior_window_drink=False):
    obs = np.zeros(298, dtype=np.float32)
    obs[0] = raw["hp"] / max(1, raw["max_hp"])
    obs[WORKER_DRINK_LATCH_FEATURE] = (
        -1.0 if prior_window_drink else 0.0)
    return obs


def _visible_masks(raw, gear=False):
    masks = _default_worker_masks()
    hp = raw["hp"] / max(1, raw["max_hp"])
    masks[9] = bool(raw.get("monsters"))
    masks[12] = bool(raw.get("belt_heals", 0) > 0 and 0.5 <= hp < 0.75)
    masks[13] = bool(raw.get("floor_items"))
    masks[14] = gear
    return masks


def _teacher_env(raw, gear=False, prior_window_drink=False):
    """TeacherV2.action 消费的最小 env 壳(免引擎)。"""
    base_masks = _visible_masks(raw, gear)
    obs = _visible_obs(raw, prior_window_drink)
    base = types.SimpleNamespace(_raw=raw, action_masks=lambda: base_masks)
    oe = types.SimpleNamespace(env=base, _worker_obs=lambda: obs)
    return types.SimpleNamespace(
        oe=oe, action_masks=lambda: base_masks)


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
        self._voluntary_drinks = 0
        self.closed = False
        base = types.SimpleNamespace(
            _raw=None, action_masks=lambda: np.zeros(15, dtype=bool))
        self.oe = types.SimpleNamespace(env=base)

    def _beat(self):
        return self._windows[self._w][self._b]

    def _sync(self):
        self.oe.env._raw = self._beat()["raw"]

    def _obs(self):
        obs = _visible_obs(
            self._beat()["raw"],
            prior_window_drink=self._voluntary_drinks > 0)
        return obs

    def reset(self, *, seed=None, options=None):
        self.stats["episodes"] += 1
        self._windows = self._script_for_seed(seed)
        self._w = self._b = 0
        self._voluntary_drinks = 0
        self._sync()
        return self._obs(), {"episode_seed": seed}

    def action_masks(self):
        mask = self._beat().get("mask")
        return (np.asarray(mask, dtype=bool).copy() if mask is not None
                else _visible_masks(self._beat()["raw"]))

    def step(self, action):
        overridden = bool(self._beat().get("overridden"))
        executed_action = self._beat().get(
            "executed_action",
            None if overridden else int(action),
        )
        if int(action) == 12 and executed_action == 12:
            self._voluntary_drinks += 1
        last = self._b == len(self._windows[self._w]) - 1
        window_id = self._w
        episode_end = last and self._w == len(self._windows) - 1
        if last and not episode_end:
            # v4：自然 FARM 边界 nonterminal，obs2 已是下一窗首态。
            self._w += 1
            self._b = 0
            self._voluntary_drinks = 0
            self._sync()
        elif not last:
            self._b += 1
            self._sync()
        return self._obs(), 0.0, episode_end, False, {
            "overridden": overridden,
            "executed_action": executed_action,
            "farm_window_end": last,
            "farm_window_id": window_id,
        }

    def next_window(self):
        # 新协议只在底层局 terminal 后被 collect_v2 调用；自然窗口转换已经
        # 由 step(nonterminal)+farm_window_end 完成。
        if self._w >= len(self._windows) - 1:
            return None
        self._w += 1
        self._b = 0
        self._voluntary_drinks = 0
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
                    lambda npz, max_steps=3000, rng_seed=0,
                    manager_sha256=None, seed_scope="train", **_kwargs:
                    _ScriptedEnv(script_for_seed))


class TriggerPredicateTests(unittest.TestCase):
    """E2①:教师 v2 可见谓词真值表(hp 边界 + 每窗主动饮位)。"""

    def test_truth_table_main_threshold(self):
        t = teacher_v2_preventive_trigger
        def hit(raw, *, prior=False):
            return t(_visible_obs(raw, prior), _visible_masks(raw))

        self.assertTrue(hit(_raw(hp=50, max_hp=100, belt=1)))   # 下界闭
        self.assertTrue(hit(_raw(hp=64, max_hp=100, belt=3)))   # 带内
        self.assertFalse(hit(_raw(hp=49, max_hp=100, belt=1)))  # 反射域/mask 关
        self.assertFalse(hit(_raw(hp=65, max_hp=100, belt=1)))  # 上界开
        self.assertFalse(hit(_raw(hp=100, max_hp=100, belt=8)))
        self.assertFalse(hit(_raw(hp=60, max_hp=100, belt=0)))  # live m12=False
        self.assertTrue(hit(_raw(hp=1, max_hp=2, belt=1)))      # 比例口径
        self.assertFalse(hit(
            _raw(hp=60, max_hp=100, belt=1), prior=True))

    def test_runtime_safety_envelope_keeps_main_threshold_hard_negatives(self):
        """线上 mask 到 0.75，但唯一注册教师止于 0.65。"""
        fixture = OptionsEnv.__new__(OptionsEnv)
        fixture.drink_sovereignty = True
        fixture._win = {"voluntary_drinks": 0}
        raw = _raw(hp=69, max_hp=100, belt=1)
        base_masks = np.ones(15, dtype=np.bool_)
        fixture.env = types.SimpleNamespace(
            _raw=raw, action_masks=lambda: base_masks)

        # 旧 OC 区仍是线上合法动作，但已经不再是教师正例。
        self.assertFalse(
            teacher_v2_preventive_trigger(
                _visible_obs(raw), _visible_masks(raw),
                _PREVENTIVE_THRESHOLD_MAIN))
        self.assertTrue(fixture._worker_masks()[12])

        # [0.65,0.75) 是真实 m12=True hard-negative 裁量区，而不是
        # “mask 已替策略作答”。
        for hp in (65, 69, 70, 74):
            fixture.env._raw = _raw(hp=hp, max_hp=100, belt=1)
            self.assertFalse(
                teacher_v2_preventive_trigger(
                    _visible_obs(fixture.env._raw),
                    _visible_masks(fixture.env._raw),
                    _PREVENTIVE_THRESHOLD_MAIN))
            self.assertTrue(fixture._worker_masks()[12])
        fixture.env._raw = _raw(hp=75, max_hp=100, belt=1)
        self.assertFalse(fixture._worker_masks()[12])
        self.assertEqual(VOLUNTARY_DRINK_HP_LOW, 0.5)
        self.assertEqual(VOLUNTARY_DRINK_HP_HIGH, 0.75)
        self.assertLess(
            _PREVENTIVE_THRESHOLD_MAIN, VOLUNTARY_DRINK_HP_HIGH)

    def test_registered_constants(self):
        self.assertEqual(bc_worker._PREVENTIVE_HP_LOW, 0.5)
        self.assertEqual(_PREVENTIVE_THRESHOLD_MAIN, 0.65)
        self.assertFalse(hasattr(bc_worker, "_PREVENTIVE_THRESHOLD_OC"))
        self.assertEqual(_REGISTERED_PREVENTIVE_THRESHOLDS, (0.65,))
        self.assertEqual(_N12_GATE_MIN, 122)
        self.assertEqual(_RECALL12_GATE_MIN, 0.5)


class TeacherV2ObservableTargetTests(unittest.TestCase):
    """E2①:预防饮目标必须是当前可见状态的纯函数。"""

    def test_unregistered_threshold_rejected(self):
        for bad in (0.6, 0.70, 0.75, 0.5, 1.0):
            with self.assertRaisesRegex(ValueError, "未注册"):
                TeacherV2(bad)
        TeacherV2(0.65)

    def test_preventive_branch_is_preposed_before_dispatch(self):
        # 带内 + 贴身怪:dispatch 会出 9,前置分支必须先出 12。
        teacher = TeacherV2()
        teacher.begin_window()
        self.assertEqual(teacher.action(_teacher_env(_in_band())), 12)

    def test_repeated_identical_visible_state_keeps_same_label(self):
        teacher = TeacherV2()
        teacher.begin_window()
        env = _teacher_env(_in_band())
        self.assertEqual(teacher.action(env), 12)
        self.assertEqual(teacher.action(env), 12)
        teacher.begin_window()
        self.assertEqual(teacher.action(env), 12)

    def test_visible_prior_drink_bit_changes_target_without_hidden_state(self):
        teacher = TeacherV2()
        self.assertEqual(teacher.action(_teacher_env(_in_band())), 12)
        self.assertEqual(
            teacher.action(
                _teacher_env(_in_band(), prior_window_drink=True)),
            9)

    def test_begin_window_notification_does_not_change_target(self):
        teacher = TeacherV2()
        env = _teacher_env(_in_band())
        self.assertEqual(teacher.action(env), 12)
        teacher.begin_window()
        self.assertEqual(teacher.action(env), 12)

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
        # 方案甲 a:v2 采集环消费 DEMO_SEEDS_V2(×3 延拓)
        n = 2 * len(bc_worker.DEMO_SEEDS_V2)
        self.assertEqual(X.shape, (n, 298))
        self.assertEqual(labels.shape, (n,))
        # 每局一窗:窗首带内态实标 12,次拍带外 → 10;真实入池非反事实。
        self.assertEqual(list(labels[:2]), [12, 10])
        self.assertEqual(int((labels == 12).sum()),
                         len(bc_worker.DEMO_SEEDS_V2))
        self.assertTrue(np.array_equal(np.unique(groups),
                                       np.asarray(bc_worker.DEMO_SEEDS_V2)))

    def test_v2_collect_rejects_a11_pool(self):
        # dispatch 农期原生不出 11;以 monkeypatch 注入 11 镜像 v2 禁采面。
        mask_11_open = _default_worker_masks()
        mask_11_open[11] = True   # 先绕过掩码守卫,专测标签级禁采断言
        self._install_env(
            lambda seed: [[{"raw": _out_band(), "mask": mask_11_open}]])
        self._patch(
            "dispatch",
            lambda mode, raw, gear, action_mask=None: 11,
        )
        with self.assertRaisesRegex(RuntimeError, "禁采动作 11"):
            bc_worker.collect_v2()

    def test_v2_explicit_manager_is_used_and_bound_in_provenance(self):
        seen = []

        def factory(npz, max_steps=3000, rng_seed=0,
                    manager_sha256=None, seed_scope="train", **_kwargs):
            seen.append((npz, manager_sha256))
            return _ScriptedEnv(
                lambda seed: [[{"raw": _out_band()}]])

        self._patch("WorkerWindowEnv", factory)
        self._patch("DEMO_SEEDS_V2", [100, 101])
        with tempfile.TemporaryDirectory() as d:
            manager = pathlib.Path(d) / "alternate-manager.npz"
            manager.write_bytes(b"manager-distribution-B")
            _, _, groups, _, _ = bc_worker.collect_v2(
                manager_npz=manager)
            provenance = bc_worker.artifact_provenance_v2(
                0.65, manager)
        self.assertEqual(
            seen,
            [(str(manager),
              hashlib.sha256(b"manager-distribution-B").hexdigest())])
        np.testing.assert_array_equal(np.unique(groups), [100, 101])
        self.assertEqual(
            provenance["manager_npz_sha256"],
            hashlib.sha256(b"manager-distribution-B").hexdigest())

    def test_v2_overridden_proposal_does_not_poison_next_visible_label(self):
        # 被保险丝拒绝/原生无效果的拍都剔除；同窗下一相同可见态仍须标 a12。
        def script(seed):
            if seed == bc_worker.DEMO_SEEDS_V2[0]:
                return [[{"raw": _in_band(), "overridden": True},
                         {"raw": _in_band(), "executed_action": None},
                         {"raw": _in_band()}, {"raw": _out_band()}]]
            return [[{"raw": _in_band()}, {"raw": _out_band()}]]

        self._install_env(script)
        _, labels, groups, _, _ = bc_worker.collect_v2()
        first = bc_worker.DEMO_SEEDS_V2[0]
        self.assertEqual(int((labels[groups == first] == 12).sum()), 1)
        for seed in (
                bc_worker.DEMO_SEEDS_V2[1],
                bc_worker.DEMO_SEEDS_V2[len(bc_worker.DEMO_SEEDS_V2) // 2],
                bc_worker.DEMO_SEEDS_V2[-1]):
            self.assertEqual(int((labels[groups == seed] == 12).sum()), 1)

    def test_v2_labels_at_most_one_a12_per_visible_window(self):
        self._install_env(lambda seed: [
            [{"raw": _in_band()}, {"raw": _in_band()}],
            [{"raw": _in_band()}, {"raw": _in_band()}]])
        _, labels, groups, _, _ = bc_worker.collect_v2()
        for seed in (
                bc_worker.DEMO_SEEDS_V2[0],
                bc_worker.DEMO_SEEDS_V2[len(bc_worker.DEMO_SEEDS_V2) // 2],
                bc_worker.DEMO_SEEDS_V2[-1]):
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
        self.assertEqual(masks.shape, (2 * len(bc_worker.DEMO_SEEDS_V2), 15))
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
        provenance = bc_worker.artifact_provenance_v2(0.65)
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            path = _save_demos_v2(
                out, X, labels, groups, masks, provenance)
            with np.load(path) as z:
                self.assertEqual(set(z.files),
                                 {"X", "Y", "episode_id", "masks",
                                  "schema_version", "protocol_version",
                                  "implementation_sha256", "generator_sha256",
                                  "manager_npz_sha256", "teacher_generation",
                                  "preventive_threshold"})
                self.assertEqual(z["masks"].shape, (n, 15))
                self.assertEqual(z["masks"].dtype, np.bool_)
                self.assertTrue(np.array_equal(z["Y"], labels))
                self.assertEqual(z["teacher_generation"].item(), 2)
                self.assertEqual(z["preventive_threshold"].item(), 0.65)
            bad_label = masks.copy()
            bad_label[0, 12] = False   # a12 样本 m[12] 必须 True
            with self.assertRaisesRegex(RuntimeError, "不在掩码内"):
                _save_demos_v2(
                    out, X, labels, groups, bad_label, provenance)
            bad_11 = masks.copy()
            bad_11[2, 11] = True
            with self.assertRaisesRegex(RuntimeError, r"m\[11\] 必须恒 False"):
                _save_demos_v2(
                    out, X, labels, groups, bad_11, provenance)
            with self.assertRaisesRegex(RuntimeError, "形状/dtype"):
                _save_demos_v2(out, X, labels, groups,
                               masks.astype(np.int64), provenance)
            with self.assertRaisesRegex(RuntimeError, "形状/dtype"):
                _save_demos_v2(
                    out, X, labels, groups, masks[:, :14], provenance)
            with self.assertRaisesRegex(RuntimeError, "provenance 缺字段"):
                _save_demos_v2(out, X, labels, groups, masks, {})


class HeldoutSelectionIsolationTests(unittest.TestCase):
    """final held-out 只能做最终闸，不能选择 v1/v2 候选。"""

    @staticmethod
    def _selection_fixture():
        groups = np.repeat(np.arange(100, 130, dtype=np.int64), 20)
        X = np.zeros((len(groups), 4), dtype=np.float32)
        fit, validation, _ = bc_worker._split_fit_validation_by_episode(groups)
        _, heldout, _ = bc_worker.split_by_episode(groups)
        labels = np.full(len(groups), 9, dtype=np.int64)
        # 零特征 fit 只见 9、validation 只见 10，稳定触发唯一 retry。
        labels[validation] = 10
        changed = labels.copy()
        changed[heldout] = 14
        return X, labels, changed, groups, fit, validation, heldout

    @staticmethod
    def _state(model):
        return {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    def _assert_same_state(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(
                torch.equal(left[key], right[key]),
                f"候选张量受 final held-out 标签影响:{key}")

    def test_nested_split_is_disjoint_deterministic_and_exhaustive(self):
        _, _, _, groups, fit, validation, heldout = (
            self._selection_fixture())
        fit2, validation2, validation_episodes2 = (
            bc_worker._split_fit_validation_by_episode(groups))
        np.testing.assert_array_equal(fit, fit2)
        np.testing.assert_array_equal(validation, validation2)
        np.testing.assert_array_equal(
            np.unique(groups[validation]), np.sort(validation_episodes2))
        self.assertFalse(np.intersect1d(fit, validation).size)
        self.assertFalse(np.intersect1d(fit, heldout).size)
        self.assertFalse(np.intersect1d(validation, heldout).size)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([fit, validation, heldout])),
            np.arange(len(groups)))

    def test_preselection_coverage_ignores_every_final_domain_value(self):
        """候选冻结前的覆盖诊断只消费 fit/validation 三张量切片。"""
        groups = np.repeat(np.arange(30, dtype=np.int64), 8)
        n = len(groups)
        X = np.zeros((n, 298), dtype=np.float32)
        X[:, 0] = 0.8
        labels = np.full(n, 9, dtype=np.int64)
        masks = np.tile(_default_worker_masks(), (n, 1))
        fit, validation, _ = bc_worker._split_fit_validation_by_episode(
            groups)
        _, final, _ = bc_worker.split_by_episode(groups)

        def add_coverage(indices):
            positive, post_drink = int(indices[0]), int(indices[1])
            labels[positive] = 12
            X[positive, 0] = 0.55
            X[post_drink, 0] = 0.55
            X[post_drink, WORKER_DRINK_LATCH_FEATURE] = -1.5
            masks[post_drink, 12] = False

        add_coverage(fit)
        add_coverage(validation)
        expected = bc_worker._bc_v2_post_drink_coverage(
            X, labels, groups, masks,
            scopes=("fit", "validation"))

        # 分别毒化 final 的 X/Y/masks。若预选诊断读取任一 final 值，
        # 结果会漂移或触发覆盖门；显式 preselection scopes 必须逐次恒等。
        poisoned_x = X.copy()
        poisoned_x[final, WORKER_DRINK_LATCH_FEATURE] = -999.0
        poisoned_y = labels.copy()
        poisoned_y[final] = 12
        poisoned_masks = masks.copy()
        poisoned_masks[final, 12] = False
        for x_arg, y_arg, masks_arg in (
            (poisoned_x, labels, masks),
            (X, poisoned_y, masks),
            (X, labels, poisoned_masks),
            (poisoned_x, poisoned_y, poisoned_masks),
        ):
            self.assertEqual(
                bc_worker._bc_v2_post_drink_coverage(
                    x_arg, y_arg, groups, masks_arg,
                    scopes=("fit", "validation")),
                expected,
            )
        with self.assertRaisesRegex(ValueError, "final 域缺少"):
            bc_worker._bc_v2_post_drink_coverage(
                poisoned_x, poisoned_y, groups, poisoned_masks,
                scopes=("final",))

    def test_recall_gate_uses_full_precision_not_rounded_display(self):
        labels = np.asarray([9] * 2000 + [10] * 8000, dtype=np.int64)
        X = np.zeros((len(labels), 4), dtype=np.float32)
        indices = np.arange(len(labels), dtype=np.int64)

        class FixedPredictions:
            def __init__(self, correct_nine):
                self.pred = np.asarray(
                    [9] * correct_nine
                    + [10] * (2000 - correct_nine)
                    + [10] * 8000,
                    dtype=np.int64)

            def eval(self):
                pass

            def __call__(self, x):
                logits = torch.full((len(x), 15), -1.0)
                logits[torch.arange(len(x)), torch.from_numpy(self.pred)] = 1.0
                return logits

        _, below = bc_worker._score_bc_model_indices(
            FixedPredictions(1699), X, labels, indices,
            eligibility_indices=indices)
        self.assertEqual(below[9], 1699 / 2000)
        self.assertFalse(
            bc_worker._bc_quality_gate_passes(0.9699, below))

        _, exact = bc_worker._score_bc_model_indices(
            FixedPredictions(1700), X, labels, indices,
            eligibility_indices=indices)
        self.assertEqual(exact[9], 0.85)
        self.assertTrue(
            bc_worker._bc_quality_gate_passes(0.97, exact))

    def test_sparse_action14_is_mandatory_recall_class(self):
        labels = np.asarray([9] * 400 + [14] * 64, dtype=np.int64)
        X = np.zeros((len(labels), 4), dtype=np.float32)
        indices = np.arange(len(labels), dtype=np.int64)

        class FixedPredictions:
            def eval(self):
                pass

            def __call__(self, x):
                del x
                logits = torch.full((len(labels), 15), -1.0)
                logits[torch.arange(len(labels)),
                       torch.from_numpy(labels)] = 1.0
                return logits

        _, ordinary = bc_worker._score_bc_model_indices(
            FixedPredictions(), X, labels, indices,
            eligibility_indices=indices)
        self.assertEqual(set(ordinary), {9})

        _, guarded = bc_worker._score_bc_model_indices(
            FixedPredictions(), X, labels, indices,
            eligibility_indices=indices,
            required_recall_actions=(14,))
        self.assertEqual(set(guarded), {9, 14})
        self.assertEqual(guarded[14], 1.0)

    def test_v1_action14_coverage_requires_count_and_episode_breadth(self):
        groups = np.repeat(np.arange(16, dtype=np.int64), 4)
        labels = np.full(len(groups), 14, dtype=np.int64)
        self.assertEqual(
            bc_worker._require_v1_action14_coverage(labels, groups),
            {"labels": 64, "episodes": 16})

        with self.assertRaisesRegex(RuntimeError, "a14 严格升级覆盖不足"):
            bc_worker._require_v1_action14_coverage(
                labels[:-1], groups[:-1])
        with self.assertRaisesRegex(RuntimeError, "a14 严格升级覆盖不足"):
            bc_worker._require_v1_action14_coverage(
                labels, np.repeat(np.arange(15, dtype=np.int64), [5] * 4 + [4] * 10 + [4]))

    def test_v2_a12_uses_specialized_gate_not_generic_class_threshold(self):
        recalls = {9: 0.99, 10: 1.0, 12: 0.69, 13: 0.98}
        self.assertFalse(
            bc_worker._bc_quality_gate_passes(0.98, recalls))
        self.assertTrue(
            bc_worker._bc_v2_quality_gate_passes(0.98, recalls))
        self.assertFalse(
            bc_worker._bc_v2_quality_gate_passes(
                0.98, {**recalls, 13: 0.8499}))

    def test_v1_heldout_label_change_cannot_change_retry_or_selected_model(self):
        X, labels, changed, groups, fit, _, _ = self._selection_fixture()
        first_a, score_a, recalls_a = bc_worker.train_bc(
            X, labels, groups)
        first_b, score_b, recalls_b = bc_worker.train_bc(
            X, changed, groups)
        retry_a = not bc_worker._bc_quality_gate_passes(
            score_a, recalls_a)
        retry_b = not bc_worker._bc_quality_gate_passes(
            score_b, recalls_b)
        self.assertTrue(retry_a)
        self.assertEqual(
            (score_a, recalls_a, retry_a),
            (score_b, recalls_b, retry_b))
        self._assert_same_state(
            self._state(first_a), self._state(first_b))

        def retry_weights(y):
            counts = np.bincount(
                y[fit], minlength=15).astype(np.float64)
            weights = np.where(
                counts > 0,
                counts.sum() / np.maximum(counts, 1), 0.0)
            return weights / weights[weights > 0].mean()

        weights_a = retry_weights(labels)
        weights_b = retry_weights(changed)
        np.testing.assert_array_equal(weights_a, weights_b)
        selected_a, retry_score_a, retry_recalls_a = bc_worker.train_bc(
            X, labels, groups, class_weights=weights_a)
        selected_b, retry_score_b, retry_recalls_b = bc_worker.train_bc(
            X, changed, groups, class_weights=weights_b)
        self.assertEqual(
            (retry_score_a, retry_recalls_a),
            (retry_score_b, retry_recalls_b))
        self._assert_same_state(
            self._state(selected_a), self._state(selected_b))

    def test_v2_heldout_label_change_cannot_change_retry_or_selected_model(self):
        X, labels, changed, groups, fit, _, _ = self._selection_fixture()
        primary_weights_a = bc_worker._balanced_class_weights(labels[fit])
        primary_weights_b = bc_worker._balanced_class_weights(changed[fit])
        np.testing.assert_array_equal(
            primary_weights_a, primary_weights_b)
        selected_a, score_a, recalls_a = bc_worker.train_bc(
            X, labels, groups, class_weights=primary_weights_a)
        selected_b, score_b, recalls_b = bc_worker.train_bc(
            X, changed, groups, class_weights=primary_weights_b)
        retry_a = not bc_worker._bc_quality_gate_passes(
            score_a, recalls_a)
        retry_b = not bc_worker._bc_quality_gate_passes(
            score_b, recalls_b)
        self.assertTrue(retry_a)
        self.assertEqual(
            (score_a, recalls_a, retry_a),
            (score_b, recalls_b, retry_b))
        self._assert_same_state(
            self._state(selected_a), self._state(selected_b))

        counts_a = np.bincount(
            labels[fit], minlength=15).astype(np.float64)
        counts_b = np.bincount(
            changed[fit], minlength=15).astype(np.float64)
        retry_weights_a = np.where(
            counts_a > 0,
            counts_a.sum() / np.maximum(counts_a, 1), 0.0)
        retry_weights_a /= retry_weights_a[retry_weights_a > 0].mean()
        retry_weights_b = np.where(
            counts_b > 0,
            counts_b.sum() / np.maximum(counts_b, 1), 0.0)
        retry_weights_b /= retry_weights_b[retry_weights_b > 0].mean()
        np.testing.assert_array_equal(
            retry_weights_a, retry_weights_b)
        retried_a, retry_score_a, retry_recalls_a = bc_worker.train_bc(
            X, labels, groups, class_weights=retry_weights_a)
        retried_b, retry_score_b, retry_recalls_b = bc_worker.train_bc(
            X, changed, groups, class_weights=retry_weights_b)
        self.assertEqual(
            (retry_score_a, retry_recalls_a),
            (retry_score_b, retry_recalls_b))
        self._assert_same_state(
            self._state(retried_a), self._state(retried_b))


class FinalHoldoutOneShotTests(_PatchMixin, unittest.TestCase):
    """final pool 一旦开始读取，就不能靠归档或重跑重新变成“盲池”。"""

    @staticmethod
    def _provenance():
        return {
            "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
            "teacher_generation": TEACHER_GENERATION_V2,
            "preventive_threshold": _PREVENTIVE_THRESHOLD_MAIN,
            "implementation_sha256": "a" * 64,
            "generator_sha256": "b" * 64,
            "manager_npz_sha256": "c" * 64,
            "protocol_version": PROTOCOL_VERSION,
        }

    def test_stable_exclusive_marker_survives_bundle_move_and_blocks_reread(self):
        seeds = [3000, 3001, 3002]
        with tempfile.TemporaryDirectory() as directory:
            runs = pathlib.Path(directory) / "runs"
            out = runs / "bc-worker-v2"
            out.mkdir(parents=True)
            first = bc_worker._mark_final_holdout_started(
                out, TEACHER_GENERATION_V2, seeds, self._provenance())
            marker, _, pool_sha256 = bc_worker._final_holdout_marker_path(
                out, TEACHER_GENERATION_V2, seeds)
            self.assertTrue(marker.is_file())
            self.assertEqual(first["pool_sha256"], pool_sha256)
            self.assertEqual(
                first["marker_sha256"],
                hashlib.sha256(marker.read_bytes()).hexdigest())
            record = json.loads(marker.read_text())
            self.assertEqual(
                record["schema_version"],
                train_ppo._BC_FINAL_HOLDOUT_POOL_SCHEMA)
            self.assertEqual(
                record["marker_schema_version"],
                train_ppo._BC_FINAL_HOLDOUT_MARKER_SCHEMA)
            self.assertEqual(
                record["consumption_stage"], "before_pool_collection")
            self.assertNotIn("candidate", record)
            self.assertEqual(
                marker.parent, runs / "_bc_final_holdout_registry")

            # canonical 文件移进 _previous 不得带走独立 one-shot marker。
            canonical = out / _BC_V2_REPORT_NAME
            canonical.write_text(json.dumps({"data_gate": "FAIL"}))
            archive = out / "_previous" / "attempt-1"
            archive.mkdir(parents=True)
            canonical.replace(archive / canonical.name)
            self.assertTrue(marker.is_file())

            # 整个 bundle 搬走后，在原 artifact 路径新建空目录也不能把同池
            # 伪装成未使用；registry 是 bundle 的稳定 sibling。
            moved = pathlib.Path(directory) / "archived-bc-worker-v2"
            out.replace(moved)
            out.mkdir()
            marker_after_move, _, _ = bc_worker._final_holdout_marker_path(
                out, TEACHER_GENERATION_V2, seeds)
            self.assertEqual(marker_after_move, marker)
            with self.assertRaisesRegex(
                    RuntimeError, "禁止同池再次采集/评分"):
                bc_worker._assert_final_holdout_unused(
                    out, TEACHER_GENERATION_V2, seeds)
            with self.assertRaisesRegex(RuntimeError, "one-shot marker 已存在"):
                bc_worker._mark_final_holdout_started(
                    out, TEACHER_GENERATION_V2, seeds,
                    self._provenance())

    def test_marker_schema_evolution_cannot_make_same_pool_fresh(self):
        seeds = [3000, 3001, 3002]
        spec, pool_sha256 = train_ppo._bc_final_holdout_marker_identity(
            TEACHER_GENERATION_V2, seeds)
        self.assertEqual(
            spec["schema_version"],
            train_ppo._BC_FINAL_HOLDOUT_POOL_SCHEMA)
        with mock.patch.object(
                train_ppo, "_BC_FINAL_HOLDOUT_MARKER_SCHEMA",
                "bc-final-holdout-consumption/future"):
            future_spec, future_pool_sha256 = (
                train_ppo._bc_final_holdout_marker_identity(
                    TEACHER_GENERATION_V2, seeds))
        self.assertEqual(future_spec, spec)
        self.assertEqual(future_pool_sha256, pool_sha256)

    def test_previous_terminal_receipt_cannot_bypass_direct_cli_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory) / "runs" / "bc-worker-v2"
            self._patch("OUT_V2", out)
            self._patch("DEMO_SEEDS_V2", [3000, 3001])
            archive = out / "_previous" / "attempt-1"
            archive.mkdir(parents=True)
            (archive / _BC_V2_REPORT_NAME).write_text(json.dumps({
                "data_gate": "PASS",
                **self._provenance(),
            }))
            with self.assertRaisesRegex(
                    RuntimeError, "禁止 direct CLI 重试或归档绕过"):
                bc_worker.begin_output_attempt_v2(self._provenance())
            self.assertFalse((out / _BC_V2_REPORT_NAME).exists())

    def test_registry_rejects_partial_subset_superset_and_cross_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory) / "runs" / "bc-worker-v2"
            out.mkdir(parents=True)
            consumed = [4_000, 4_001, 4_002]
            bc_worker._mark_final_holdout_started(
                out, TEACHER_GENERATION_V2, consumed, self._provenance())

            cases = (
                (TEACHER_GENERATION_V2, [4_002, 4_003]),       # partial
                (TEACHER_GENERATION_V2, [4_001]),              # subset
                (TEACHER_GENERATION_V2, [3_999, *consumed]),   # superset
                (TEACHER_GENERATION_V1, [4_002, 5_000]),       # cross-gen
            )
            for generation, seeds in cases:
                with self.subTest(generation=generation, seeds=seeds):
                    with self.assertRaisesRegex(ValueError, "部分/全部重叠"):
                        train_ppo._assert_bc_final_holdout_pool_disjoint(
                            out, generation, seeds)
                    marker, _, _ = bc_worker._final_holdout_marker_path(
                        out, generation, seeds)
                    self.assertFalse(marker.exists())
                    with self.assertRaisesRegex(ValueError, "部分/全部重叠"):
                        bc_worker._mark_final_holdout_started(
                            out, generation, seeds, self._provenance())
                    self.assertFalse(marker.exists())

    def test_registry_malformed_marker_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory) / "runs" / "bc-worker"
            registry = out.parent / "_bc_final_holdout_registry"
            registry.mkdir(parents=True)
            (registry / "broken.json").write_text("{")
            with self.assertRaisesRegex(ValueError, "不可解析"):
                train_ppo._assert_bc_final_holdout_pool_disjoint(
                    out, TEACHER_GENERATION_V1, [6_000, 6_001])

    def test_registry_marker_full_identity_tamper_fails_closed(self):
        mutations = {
            "extra-field": lambda record: record.update({"extra": True}),
            "pool-sha": lambda record: record.update(
                {"pool_sha256": "0" * 64}),
            "marker-schema": lambda record: record.update(
                {"marker_schema_version": "future"}),
            "consumption-stage": lambda record: record.update(
                {"consumption_stage": "after-scoring"}),
            "empty-provenance": lambda record: record.update(
                {"provenance": {}}),
            "split-unit": lambda record: record.update(
                {"split_unit": "row"}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                out = pathlib.Path(directory) / "runs" / "bc-worker-v2"
                out.mkdir(parents=True)
                marker_record = bc_worker._mark_final_holdout_started(
                    out, TEACHER_GENERATION_V2, [7_000, 7_001],
                    self._provenance())
                marker, _, _ = bc_worker._final_holdout_marker_path(
                    out, TEACHER_GENERATION_V2, [7_000, 7_001])
                record = dict(marker_record)
                record.pop("marker_sha256")
                mutate(record)
                marker.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "身份非法"):
                    train_ppo._assert_bc_final_holdout_pool_disjoint(
                        out, TEACHER_GENERATION_V1, [8_000, 8_001])

    def test_registry_unknown_residue_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory) / "runs" / "bc-worker"
            registry = out.parent / "_bc_final_holdout_registry"
            registry.mkdir(parents=True)
            (registry / "orphan.tmp").write_bytes(b"partial marker")
            with self.assertRaisesRegex(ValueError, "未知残件"):
                train_ppo._assert_bc_final_holdout_pool_disjoint(
                    out, TEACHER_GENERATION_V1, [9_000, 9_001])


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
        masks = np.tile(_default_worker_masks(), (len(groups), 1))
        recall, denominator = _recall12_from_model(
            model, X, labels, groups, masks)
        self.assertEqual(denominator, 1)
        self.assertEqual(recall, 1.0)
        # 同分母、argmax 脱靶 → 0.0(度量 = held-out argmax 命中)
        miss = self._fixed_model([9] * len(ho))
        recall_miss, denom_miss = _recall12_from_model(
            miss, X, labels, groups, masks)
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
        masks = np.tile(_default_worker_masks(), (len(groups), 1))
        recall, denominator = _recall12_from_model(
            model, X, labels, groups, masks)
        self.assertEqual(denominator, 0)
        self.assertEqual(recall, 0.0)
        self.assertLess(recall, _RECALL12_GATE_MIN)


class A12CalibrationTests(unittest.TestCase):
    """稀有类校准必须改变实际六张量，并在未看 held-out 时变得可达。"""

    def test_unreserved_random_model_is_rejected_before_destructive_wiring(self):
        model = bc_worker.PiHead(298)
        with self.assertRaisesRegex(RuntimeError, "曾参与普通 CE"):
            bc_worker._wire_a12_teacher_boundary(model, 0.65)

    def test_reserved_path_keeps_every_non12_logit_bitwise_equal(self):
        rng = np.random.default_rng(91)
        groups = np.repeat(np.arange(10, dtype=np.int64), 80)
        X = rng.standard_normal((len(groups), 298)).astype(np.float32)
        labels = np.where(np.arange(len(groups)) % 2, 9, 10).astype(
            np.int64)
        masks = np.tile(_default_worker_masks(), (len(groups), 1))
        model, _, _ = bc_worker.train_bc(
            X, labels, groups, masks=masks, reserve_a12_path=True,
            epochs=1)
        with torch.no_grad():
            before = model(torch.from_numpy(X[:64])).clone()
        bc_worker._wire_a12_teacher_boundary(model, 0.65)
        with torch.no_grad():
            after = model(torch.from_numpy(X[:64]))
        ordinary = [action for action in range(15) if action != 12]
        self.assertTrue(torch.equal(
            before[:, ordinary], after[:, ordinary]))

    def test_exact_upper_boundary_separates_real_neighboring_states(self):
        """真实相邻 HP 格点：.648936 为正，.651163 合法负例概率须安全。"""
        query = np.zeros((2, 298), dtype=np.float32)
        query[:, 0] = np.asarray([0.648936, 0.651163], dtype=np.float32)
        masks = np.tile(_default_worker_masks(), (2, 1))
        self.assertTrue(
            teacher_v2_preventive_trigger(query[0], masks[0], 0.65))
        self.assertFalse(
            teacher_v2_preventive_trigger(query[1], masks[1], 0.65))

        model = bc_worker.PiHead(298)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.bias.fill_(-5.0)
            model.head.bias[9] = 0.0
        bc_worker._wire_a12_teacher_boundary(model, 0.65)

        # 只用正例一侧选一个固定 margin；随后以最终六张量同源前向
        # 检验另一侧，防把旧的 0.652 中心重新带回来。
        with torch.no_grad():
            raw = model(torch.from_numpy(query))
            positive_margin = raw[0, 12] - raw[0, 9]
            model.head.bias[12] = -positive_margin + 0.05
        logits = train_ppo._policy_logits_from_sb3_state_dict(
            bc_worker.export_sb3_sd(model), query)
        masked = torch.where(
            torch.from_numpy(masks), logits,
            torch.full_like(logits, -1e8))
        probabilities = torch.softmax(masked, dim=-1)[:, 12]
        predictions = masked.argmax(dim=-1)
        self.assertEqual(int(predictions[0]), 12)
        self.assertNotEqual(int(predictions[1]), 12)
        self.assertLessEqual(
            float(probabilities[1]),
            train_ppo._A12_LEGAL_NEGATIVE_PROBABILITY_MAX)
        self.assertGreater(
            float(probabilities[0]), float(probabilities[1]) * 1000.0)

    def test_training_only_calibrator_reaches_heldout_behavior_gate(self):
        groups = np.repeat(np.arange(10, dtype=np.int64), 300)
        n = len(groups)
        X = np.zeros((n, 298), dtype=np.float32)
        X[:, 0] = 0.8
        X[:, 295] = 0.5
        labels = np.full(n, 9, dtype=np.int64)
        for episode in range(10):
            index = episode * 300
            X[index, 0] = 0.55
            X[index, 295] = 0.1
            labels[index] = 12
        masks = np.tile(_default_worker_masks(), (n, 1))
        model = bc_worker.PiHead(298)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head.bias.fill_(-5.0)
            model.head.bias[9] = 1.0
        before = train_ppo.bc_aux_behavior_metrics(
            bc_worker.export_sb3_sd(model), X, labels, groups, masks)
        self.assertEqual(
            train_ppo.bc_aux_behavior_gate(before)["verdict"], "FAIL")
        calibration = bc_worker._calibrate_a12_policy(
            model, X, labels, groups, masks, 0.65)
        after = train_ppo.bc_aux_behavior_metrics(
            bc_worker.export_sb3_sd(model), X, labels, groups, masks)
        self.assertEqual(
            calibration["fit_scope"], "nested-fit-episodes-only")
        self.assertEqual(
            train_ppo.bc_aux_behavior_gate(after)["verdict"], "PASS")
        self.assertGreaterEqual(after["recall_12"], 0.25)
        self.assertLessEqual(after["fpr_12"], train_ppo._A12_FPR_MAX)

        # validation 与 final-heldout 标签即使整体改写，也不能改变
        # tau/bias 校准选型或最终六张量；它们只能在候选冻结后的各自 gate
        # 产生不同结论。
        changed_labels = labels.copy()
        _, validation, _ = bc_worker._split_fit_validation_by_episode(groups)
        _, heldout, _ = bc_worker.split_by_episode(groups)
        changed_labels[validation] = 14
        changed_labels[heldout] = 14
        changed_model = bc_worker.PiHead(298)
        with torch.no_grad():
            for parameter in changed_model.parameters():
                parameter.zero_()
            changed_model.head.bias.fill_(-5.0)
            changed_model.head.bias[9] = 1.0
        changed_calibration = bc_worker._calibrate_a12_policy(
            changed_model, X, changed_labels, groups, masks, 0.65)
        self.assertEqual(calibration, changed_calibration)
        original_sd = bc_worker.export_sb3_sd(model)
        changed_sd = bc_worker.export_sb3_sd(changed_model)
        for key in original_sd:
            self.assertTrue(
                torch.equal(original_sd[key], changed_sd[key]),
                f"a12 校准张量受 final held-out 标签影响:{key}")

    def test_calibration_receipt_uses_exact_deployed_forward_and_bias(self):
        for competing_bias in (-19.7, 16.0):
            with self.subTest(competing_bias=competing_bias):
                groups = np.repeat(np.arange(10, dtype=np.int64), 300)
                X = np.zeros((len(groups), 298), dtype=np.float32)
                X[:, 0] = 0.8
                labels = np.full(len(groups), 9, dtype=np.int64)
                for episode in range(10):
                    index = episode * 300
                    X[index, 0] = 0.55
                    labels[index] = 12
                masks = np.tile(
                    _default_worker_masks(), (len(groups), 1))
                model = bc_worker.PiHead(298)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    # -19.7 复现回执/float32 ULP 拒真；16.0 复现旧
                    # margin 捷径把 9/12 tie 错算成 action12。
                    model.head.bias[9] = competing_bias
                calibration = bc_worker._calibrate_a12_policy(
                    model, X, labels, groups, masks, 0.65)
                deployed = float(model.head.bias[12].item())
                self.assertEqual(calibration["bias_12"], deployed)
                policy_sd = bc_worker.export_sb3_sd(model)
                train_ppo._validate_bc_v2_calibration_receipt(
                    {
                        "preventive_threshold": 0.65,
                        "a12_calibration": calibration,
                    },
                    policy_sd, X, labels, groups, masks,
                )


def _v2_pass_record(policy_bytes=b"v2-policy", impl_sha="a" * 64, **overrides):
    behavior = {
        "scope": "heldout", "mask_mode": "bc-v2-recorded",
        "pairs": 10000, "tp": 25, "fp": 0, "fn": 0, "tn": 9975,
        "true_a12": 25, "non_a12": 9975, "all_non_a12": 9975,
        "predicted_a12": 25, "predicted_a12_episodes": 10,
        "predicted_a12_margin_min": 0.10,
        "precision_12": 1.0, "recall_12": 1.0,
        "fpr_12": 0.0, "predicted_share_12": 0.0025,
        "high_hp_non_a12": 5000, "high_hp_false_drinks": 0,
        "high_hp_false_drink_rate": 0.0,
        "eligible_probability_12_min": 0.60,
        "eligible_probability_12_mean": 0.70,
        "eligible_probability_12_max": 0.80,
        "legal_negative_probability_12_mean": 0.0,
        "legal_negative_probability_12_max": 0.0,
        "legal_negative_probability_12_sum": 0.0,
        "predicted_share_13": 0.0, "true_share_13": 0.0,
        "a13_reference": "heldout_label", "a13_reference_share": 0.0,
        "a13_spillover": 0.0,
        "mean_probability_12": 0.70 * 25 / 10000,
        "anchor": None,
    }
    behavior_gate = train_ppo.bc_aux_behavior_gate(
        behavior, require_teacher_recall=False)
    calibration = {
        "schema_version": bc_worker._A12_CALIBRATION_SCHEMA_VERSION,
        "fit_scope": "nested-fit-episodes-only",
        "fit_pairs": 81000, "fit_episodes": 311,
        "validation_pairs_excluded": 9000,
        "validation_episodes_excluded": 35,
        "final_heldout_pairs_excluded": 10000,
        "final_heldout_episodes_excluded": 38,
        "hp_low": 0.5, "hp_high": 0.65,
        "hp_feature": bc_worker._A12_CALIBRATION_HP_FEATURE,
        "drink_latch_feature":
            bc_worker._A12_CALIBRATION_DRINK_LATCH_FEATURE,
        "predicate": bc_worker._A12_CALIBRATION_PREDICATE,
        "bias_12": -5.0,
        "target_recall_12":
            bc_worker._A12_CALIBRATION_TRAIN_RECALL_TARGET,
        "fit_metrics": {
            "tp": 75, "fp": 0, "precision_12": 1.0,
            "recall_12": 0.75, "fpr_12": 0.0,
            "predicted_share_12": 75 / 81000,
            "high_hp_false_drink_rate": 0.0,
            "legal_negative_probability_12_mean": 0.0,
            "legal_negative_probability_12_max": 0.0,
            "a13_spillover": 0.0,
        },
    }
    rec = {
        "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
        "teacher_generation": 2,
        "preventive_threshold": 0.65,
        "pairs": 100000, "held_out_top1": 0.99,
        "held_out_pairs": 10000,
        "held_out_episodes": list(range(100, 138)),
        "class_recalls": {"9": 0.99},
        "class_weighted_retry": False,
        "n12": 244, "n12_gate_min": 122,
        "n12_by_episode": {"178": 120, "187": 124},
        "recall_12": 1.0, "recall_12_denominator": 25,
        "recall_12_gate_min": 0.5,
        "a12_behavior": behavior,
        "a12_behavior_gate": behavior_gate,
        "a12_calibration": calibration,
        "class_share_12": 0.02, "class_share_13": 0.03,
        "belt_economy": {"belt_mean_at_a12": 2.5, "belt_mean_overall": 3.0,
                         "a13_pairs": 30},
        # 方案甲回执新字段(2026-07-19 亲批)
        "collection_episodes": 384,
        "class_weights": {"9": 0.520833, "12": 40.0},
        "data_gate": "PASS",
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": impl_sha,
        "generator_sha256": BC_WORKER_SHA,
        "manager_npz_sha256": "c" * 64,
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "demos_sha256": "b" * 64,
        "final_pool_sha256": "0" * 64,
        "final_holdout_marker_sha256": "0" * 64,
    }
    rec.update(overrides)
    return rec


def _bind_v2_final_holdout_marker(policy: pathlib.Path, report: dict):
    """写生产同构 marker，并把 PASS 回执精确绑定到其稳定字节。"""
    marker, spec, pool_sha256 = train_ppo._bc_final_holdout_marker_path(
        policy.parent, TEACHER_GENERATION_V2, bc_worker.DEMO_SEEDS_V2)
    provenance_keys = {
        "schema_version", "teacher_generation", "preventive_threshold",
        "protocol_version", "implementation_sha256", "generator_sha256",
        "manager_npz_sha256",
    }
    record = {
        **spec,
        "pool_sha256": pool_sha256,
        "marker_schema_version":
            train_ppo._BC_FINAL_HOLDOUT_MARKER_SCHEMA,
        "started_at_ns": 1,
        "provenance": {
            key: report[key] for key in provenance_keys
        },
        "consumption_stage": "before_pool_collection",
    }
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":")).encode()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(payload)
    report["final_pool_sha256"] = pool_sha256
    report["final_holdout_marker_sha256"] = hashlib.sha256(
        payload).hexdigest()
    return marker


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
            policy = (
                pathlib.Path(d) / "runs" / "bc-worker-v2" / "policy_sd.pt")
            policy.parent.mkdir(parents=True)
            policy.write_bytes(b"v2-policy")
            # 修①a(rev4 十二附二④):demos 实测字节断言在位——正例须落实件
            (policy.with_name("demos.npz")).write_bytes(b"v2-demos")
            rec = _v2_pass_record(
                demos_sha256=hashlib.sha256(b"v2-demos").hexdigest())
            _bind_v2_final_holdout_marker(policy, rec)
            (policy.with_name(_BC_V2_REPORT_NAME)).write_text(json.dumps(rec))
            out = _validate_bc_v2_report(policy, "a" * 64)
            self.assertEqual(out["teacher_generation"], 2)
            self.assertEqual(out["n12"], 244)

    def test_v2_validator_binds_expected_manager_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "runs" / "bc-worker-v2" / "policy_sd.pt"
            p.parent.mkdir(parents=True)
            rec = _v2_pass_record(
                demos_sha256=hashlib.sha256(b"v2-demos").hexdigest())
            _bind_v2_final_holdout_marker(p, rec)
            payload = json.dumps(rec).encode()
            out = _validate_bc_v2_report(
                p, "a" * 64, report_payload=payload,
                policy_payload=b"v2-policy", demos_payload=b"v2-demos",
                expected_manager_sha256="c" * 64)
            self.assertEqual(out["manager_npz_sha256"], "c" * 64)
            with self.assertRaisesRegex(
                    ValueError, "经理身份与训练经理不一致"):
                _validate_bc_v2_report(
                    p, "a" * 64, report_payload=payload,
                    policy_payload=b"v2-policy", demos_payload=b"v2-demos",
                    expected_manager_sha256="d" * 64)

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

    def test_v2_tamper_matrix_fails_loud(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "runs" / "bc-worker-v2" / "policy_sd.pt"
            p.parent.mkdir(parents=True)

            def payload(**overrides):
                rec = _v2_pass_record(**overrides)
                _bind_v2_final_holdout_marker(p, rec)
                return json.dumps(rec).encode()

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
                (dict(implementation_sha256="f" * 64),
                 "身份与当前运行时不一致"),
                # 方案甲回执新字段篡改矩阵(2026-07-19 亲批)
                (dict(collection_episodes=0), "collection_episodes 非法"),
                (dict(collection_episodes=True), "collection_episodes 非法"),
                (dict(collection_episodes="384"),
                 "collection_episodes 非法"),
                (dict(class_weights={}), "class_weights 必须是非空对象"),
                (dict(class_weights=[0.5]),
                 "class_weights 必须是非空对象"),
                (dict(class_weights={"16": 1.0}), "class_weights 键非法"),
                (dict(class_weights={"x": 1.0}), "键必须是动作编号"),
                (dict(class_weights={"9": 0}),
                 r"class_weights\['9'\] 非法"),
                (dict(class_weights={"9": True}),
                 r"class_weights\['9'\] 非法"),
            )
            for overrides, message in cases:
                with self.assertRaisesRegex(ValueError, message,
                                            msg=repr(overrides)):
                    _validate_bc_v2_report(
                        p, "a" * 64,
                        report_payload=payload(**overrides),
                        policy_payload=b"v2-policy")

            # PASS 回执必须同时绑定 pool 身份和 marker 的真实字节哈希。
            marker_hash_tamper = _v2_pass_record()
            _bind_v2_final_holdout_marker(p, marker_hash_tamper)
            marker_hash_tamper["final_holdout_marker_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    ValueError, "PASS report 未精确绑定"):
                _validate_bc_v2_report(
                    p, "a" * 64,
                    report_payload=json.dumps(marker_hash_tamper).encode(),
                    policy_payload=b"v2-policy")

            # 权重字节篡改 → SHA 绑定必炸
            with self.assertRaisesRegex(ValueError, "SHA 不匹配"):
                _validate_bc_v2_report(
                    p, "a" * 64, report_payload=payload(),
                    policy_payload=b"tampered")
            # demos 实测字节断言(镜像 policy 侧形制)。
            with self.assertRaisesRegex(
                    ValueError, "demos 与回执 SHA 不匹配"):
                _validate_bc_v2_report(
                    p, "a" * 64, report_payload=payload(),
                    policy_payload=b"v2-policy",
                    demos_payload=b"tampered-demos")
            # demos 缺失/不可读(无 payload 且路径无实件)必须 fail-loud。
            with self.assertRaisesRegex(ValueError, "demos 缺失/不可读"):
                _validate_bc_v2_report(
                    p, "a" * 64, report_payload=payload(),
                    policy_payload=b"v2-policy")
            # demos 实测字节恒等 → 全链绿。
            good = _v2_pass_record(
                demos_sha256=hashlib.sha256(b"v2-demos").hexdigest())
            _bind_v2_final_holdout_marker(p, good)
            out = _validate_bc_v2_report(
                p, "a" * 64, report_payload=json.dumps(good).encode(),
                policy_payload=b"v2-policy", demos_payload=b"v2-demos")
            self.assertEqual(out["n12"], 244)
            # 缺键/多键 → 键集合精确等必炸
            missing = _v2_pass_record()
            missing.pop("demos_sha256")
            extra = _v2_pass_record()
            extra["surprise"] = 1
            for rec in (missing, extra):
                with self.assertRaisesRegex(ValueError, "字段/schema 不匹配"):
                    _validate_bc_v2_report(
                        p, "a" * 64,
                        report_payload=json.dumps(rec).encode(),
                        policy_payload=b"v2-policy")


class MainV2ReceiptTests(_PatchMixin, unittest.TestCase):
    """E2②④⑥:main_v2 回执键集合恰等、FAIL 拒写权重、重试触发面钉死。"""

    @staticmethod
    def _dataset(n12=140, per_episode=1000):
        episodes = np.arange(100, 108, dtype=np.int64)
        groups = np.repeat(episodes, per_episode)
        n = len(groups)
        labels = np.full(n, 9, dtype=np.int64)
        X = np.zeros((n, 298), dtype=np.float32)
        X[:, 0] = 0.8
        fit, validation = train_ppo._bc_v2_fit_validation_indices(groups)
        heldout = train_ppo._bc_v2_holdout_indices(groups)
        domains = (fit, validation, heldout)
        allocations = [
            n12 // len(domains) + (index < n12 % len(domains))
            for index in range(len(domains))
        ]
        for domain, count in zip(domains, allocations):
            positives = domain[:count]
            post_drink = domain[count:2 * count]
            labels[positives] = 12
            X[positives, 0] = 0.55
            X[post_drink, 0] = 0.55
            X[post_drink, 297] = -1.5
        masks = np.tile(_default_worker_masks(), (n, 1))
        for domain, count in zip(domains, allocations):
            masks[domain[count:2 * count], 12] = False
        belts = np.full(n, 3, dtype=np.int64)
        return X, labels, groups, masks, belts

    def _run(self, tmp, dataset, top1_seq=(0.99,), recall=0.8,
             post_calibration_score=(0.99, {9: 0.99})):
        calls = []
        self.train_calls = calls   # 先挂 self:main_v2 中途抛时调用记录仍可查
        self.train_epochs = []
        self.train_masks = []
        self.final_score_calls = 0

        def fake_train_bc(
                X, Y, groups, class_weights=None, *,
                epochs=bc_worker._BC_PRIMARY_EPOCHS, masks=None,
                reserve_a12_path=False):
            calls.append(class_weights)
            self.train_epochs.append(epochs)
            self.train_masks.append(masks)
            self.assertTrue(reserve_a12_path)
            top1 = top1_seq[min(len(calls) - 1, len(top1_seq) - 1)]
            return object(), top1, {9: 0.99}

        self.output_dir = (
            pathlib.Path(tmp) / "runs" / "bc-worker-v2")
        self._patch("OUT_V2", self.output_dir)
        self.collect_saw_marker = False

        def fake_collect(thr, manager_npz, manager_sha256=None):
            del thr, manager_npz, manager_sha256
            marker, _, _ = bc_worker._final_holdout_marker_path(
                self.output_dir, TEACHER_GENERATION_V2,
                bc_worker.DEMO_SEEDS_V2)
            self.assertTrue(
                marker.is_file(),
                "final-pool marker 必须在 collect/首次 reset 前落盘")
            marker_record = json.loads(marker.read_text())
            self.assertEqual(
                marker_record["consumption_stage"],
                "before_pool_collection")
            self.assertNotIn("candidate", marker_record)
            self.collect_saw_marker = True
            return dataset

        self._patch("collect_v2", fake_collect)
        self._patch("train_bc", fake_train_bc)
        self._patch(
            "_score_bc_model_indices",
            lambda m, X, y, score_indices, *,
                   eligibility_indices, masks=None:
            post_calibration_score)

        def final_score(m, X, y, g, masks=None):
            self.final_score_calls += 1
            return (top1_seq[min(len(calls) - 1, len(top1_seq) - 1)],
                    {9: 0.99})

        self._patch("_score_bc_model", final_score)
        _, labels, groups, masks, _ = dataset
        _, ho, heldout_episodes = bc_worker.split_by_episode(groups)
        fit, validation, validation_episodes = (
            bc_worker._split_fit_validation_by_episode(groups))
        calibration = {
            "schema_version": bc_worker._A12_CALIBRATION_SCHEMA_VERSION,
            "fit_scope": "nested-fit-episodes-only",
            "fit_pairs": int(len(fit)),
            "fit_episodes": int(np.unique(groups[fit]).size),
            "validation_pairs_excluded": int(len(validation)),
            "validation_episodes_excluded": int(len(validation_episodes)),
            "final_heldout_pairs_excluded": int(len(ho)),
            "final_heldout_episodes_excluded": int(len(heldout_episodes)),
            "hp_low": 0.5, "hp_high": 0.65,
            "hp_feature": bc_worker._A12_CALIBRATION_HP_FEATURE,
            "drink_latch_feature":
                bc_worker._A12_CALIBRATION_DRINK_LATCH_FEATURE,
            "predicate": bc_worker._A12_CALIBRATION_PREDICATE,
            "bias_12": -5.0,
            "target_recall_12":
                bc_worker._A12_CALIBRATION_TRAIN_RECALL_TARGET,
            "fit_metrics": {
                "tp": 30, "fp": 0, "precision_12": 1.0,
                "recall_12": 0.75, "fpr_12": 0.0,
                "predicted_share_12": 30 / len(fit),
                "high_hp_false_drink_rate": 0.0,
                "legal_negative_probability_12_mean": 0.0,
                "legal_negative_probability_12_max": 0.0,
                "a13_spillover": 0.0,
            },
        }
        self._patch("_calibrate_a12_policy",
                    lambda m, X, y, g, live_masks, thr: calibration)

        def fake_behavior(model, X, y, g, live_masks):
            pairs = int(len(ho))
            true12 = 5
            tp = int(round(recall * true12))
            fp = 0
            predicted = tp + fp
            missed = true12 - tp
            eligible_mean = (
                (0.70 * tp + 0.05 * missed) / true12)
            metrics = {
                "scope": "heldout", "mask_mode": "bc-v2-recorded",
                "pairs": pairs, "tp": tp, "fp": fp,
                "fn": true12 - tp,
                "tn": pairs - true12 - fp,
                "true_a12": true12, "non_a12": pairs - true12,
                "all_non_a12": pairs - true12,
                "predicted_a12": predicted,
                "predicted_a12_episodes": min(predicted, 2),
                "predicted_a12_margin_min":
                    0.10 if predicted else 0.0,
                "precision_12": tp / predicted if predicted else 0.0,
                "recall_12": tp / true12,
                "fpr_12": fp / (pairs - true12),
                "predicted_share_12": predicted / pairs,
                "high_hp_non_a12": max(1, pairs // 2),
                "high_hp_false_drinks": 0,
                "high_hp_false_drink_rate": 0.0,
                "eligible_probability_12_min":
                    0.05 if missed else 0.70,
                "eligible_probability_12_mean": eligible_mean,
                "eligible_probability_12_max": 0.80,
                "legal_negative_probability_12_mean": 0.0,
                "legal_negative_probability_12_max": 0.0,
                "legal_negative_probability_12_sum": 0.0,
                "predicted_share_13": 0.0, "true_share_13": 0.0,
                "a13_reference": "heldout_label",
                "a13_reference_share": 0.0, "a13_spillover": 0.0,
                "mean_probability_12":
                    eligible_mean * true12 / pairs,
                "anchor": None,
            }
            return metrics

        self._patch("_a12_behavior_from_model", fake_behavior)
        self._patch(
            "_a12_behavior_from_indices",
            lambda model, X, y, g, live_masks, indices:
            fake_behavior(
                model, X[indices], y[indices],
                g[indices], live_masks[indices]))
        self._patch("export_sb3_sd", lambda m: {"w": torch.zeros(1)})
        self._patch("artifact_provenance_v2", lambda thr, manager_npz: {
            "schema_version": _BC_V2_REPORT_SCHEMA_VERSION,
            "teacher_generation": 2, "preventive_threshold": thr,
            "protocol_version": PROTOCOL_VERSION,
            "implementation_sha256": "a" * 64,
            "generator_sha256": BC_WORKER_SHA,
            "manager_npz_sha256": "c" * 64})
        bc_worker.main_v2(0.65)
        self.assertTrue(self.collect_saw_marker)
        return calls

    def _report(self, tmp):
        del tmp
        return json.loads(
            (self.output_dir / _BC_V2_REPORT_NAME).read_text())

    def test_pass_receipt_keys_exactly_match_registered_schema(self):
        with tempfile.TemporaryDirectory() as d:
            dataset = self._dataset()
            self._run(d, dataset)
            rec = self._report(d)
            self.assertEqual(set(rec), set(_BC_V2_PASS_KEYS))
            self.assertEqual(rec["data_gate"], "PASS")
            self.assertEqual(rec["teacher_generation"], 2)
            self.assertEqual(rec["preventive_threshold"], 0.65)
            self.assertEqual(rec["n12"], 140)
            self.assertEqual(rec["n12_gate_min"], 122)
            self.assertEqual(rec["recall_12"], 0.8)
            # 方案甲回执新字段:实测采集局数 + 主训类平衡权重逐类摘要
            _, labels, groups, _, _ = dataset
            self.assertEqual(rec["collection_episodes"],
                             int(np.unique(groups).size))
            fit, _, _ = bc_worker._split_fit_validation_by_episode(groups)
            expected_w = bc_worker._balanced_class_weights(labels[fit])
            self.assertEqual(
                rec["class_weights"],
                {str(int(c)): round(float(expected_w[c]), 6)
                 for c in np.flatnonzero(expected_w > 0)})
            policy = self.output_dir / "policy_sd.pt"
            self.assertTrue(policy.is_file())
            self.assertEqual(
                rec["policy_sha256"],
                hashlib.sha256(policy.read_bytes()).hexdigest())
            # 全环闭合:主权产物过 v2 专用验证器
            out = _validate_bc_v2_report(policy, "a" * 64)
            self.assertEqual(out["n12"], 140)
            # demos 落 v2 目录且携 masks(独立目录,v1 canonical 不涉)
            with np.load(self.output_dir / "demos.npz") as z:
                self.assertIn("masks", z.files)

    def test_n12_gate_fail_refuses_policy_and_disables_old_oc(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "独立 fresh pool"):
                self._run(d, self._dataset(n12=121))
            rec = self._report(d)
            self.assertEqual(rec["data_gate"], "FAIL")
            self.assertEqual(rec["n12"], 121)   # FAIL 回执读数在册不消失
            self.assertFalse((self.output_dir / "policy_sd.pt").exists())

    def test_recall12_fail_never_triggers_class_weighted_retry(self):
        # 类加权系 N12 已除名预案(方案甲 2026-07-19 亲批后:主训即类平衡
        # 加权);recall_12 失守仍只 FAIL 不重训。
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(
                    RuntimeError, "post-calibration validation FAIL"):
                self._run(d, self._dataset(), recall=0.4)
            self.assertEqual(len(self.train_calls), 1)   # 只训一次,零重试
            rec = self._report(d)
            self.assertIs(rec["class_weighted_retry"], False)
            self.assertEqual(
                rec["failure_stage"], "post_calibration_selection")
            self.assertEqual(self.final_score_calls, 0)
            self.assertFalse((self.output_dir / "policy_sd.pt").exists())

    def test_v1_quality_gate_still_triggers_single_retry(self):
        with tempfile.TemporaryDirectory() as d:
            dataset = self._dataset()
            calls = self._run(d, dataset, top1_seq=(0.90, 0.99))
            self.assertEqual(len(calls), 2)             # 唯一重试
            # 方案甲 b:v2 主训即类平衡加权(非 None),权重恒等标准平衡式
            _, labels, groups, _, _ = dataset
            fit, _, _ = bc_worker._split_fit_validation_by_episode(groups)
            self.assertIsNotNone(calls[0])
            np.testing.assert_allclose(
                calls[0], bc_worker._balanced_class_weights(labels[fit]))
            self.assertIsNotNone(calls[1])              # 第二发系类加权重试
            self.assertEqual(
                self.train_epochs,
                [bc_worker._BC_PRIMARY_EPOCHS,
                 bc_worker._BC_WEIGHTED_RETRY_EPOCHS])
            self.assertIs(self._report(d)["class_weighted_retry"], True)

    def test_v2_primary_train_call_is_class_balanced(self):
        # 方案甲 b 正例:PASS 路径主训一次即类平衡加权,零重试。
        with tempfile.TemporaryDirectory() as d:
            dataset = self._dataset()
            calls = self._run(d, dataset)
            self.assertEqual(len(calls), 1)
            _, labels, groups, _, _ = dataset
            fit, _, _ = bc_worker._split_fit_validation_by_episode(groups)
            np.testing.assert_allclose(
                calls[0], bc_worker._balanced_class_weights(labels[fit]))
            self.assertEqual(
                self.train_epochs, [bc_worker._BC_PRIMARY_EPOCHS])
            self.assertIs(self.train_masks[0], dataset[3])
            self.assertIs(self._report(d)["class_weighted_retry"], False)

    def test_failed_retry_burns_pool_before_collect_but_skips_final_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(
                    RuntimeError, "candidate validation FAIL"):
                self._run(
                    d, self._dataset(), top1_seq=(0.90, 0.90))
            self.assertEqual(
                self.train_epochs,
                [bc_worker._BC_PRIMARY_EPOCHS,
                 bc_worker._BC_WEIGHTED_RETRY_EPOCHS])
            self.assertEqual(self.final_score_calls, 0)
            rec = self._report(d)
            self.assertEqual(rec["failure_stage"], "candidate_selection")
            self.assertFalse((self.output_dir / "demos.pending.npz").exists())
            marker, _, _ = bc_worker._final_holdout_marker_path(
                self.output_dir, TEACHER_GENERATION_V2,
                bc_worker.DEMO_SEEDS_V2)
            self.assertTrue(marker.is_file())

    def test_post_calibration_validation_precedes_final_holdout(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(
                    RuntimeError, "post-calibration validation FAIL"):
                self._run(
                    d, self._dataset(),
                    post_calibration_score=(0.90, {9: 0.90}))
            self.assertEqual(self.final_score_calls, 0)
            rec = self._report(d)
            self.assertEqual(
                rec["failure_stage"], "post_calibration_selection")

    def test_unregistered_threshold_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as d:
            self._patch("OUT_V2", pathlib.Path(d))
            with self.assertRaisesRegex(ValueError, "未注册"):
                bc_worker.main_v2(0.6)
            self.assertEqual(list(pathlib.Path(d).iterdir()), [])


class PlanAExpansionTests(_PatchMixin, unittest.TestCase):
    """方案甲 a(2026-07-19 亲批):v2 采集局数 ×3 + 种子确定性延拓。"""

    def test_factor_constant_and_seed_extension_rule(self):
        self.assertEqual(bc_worker._V2_COLLECTION_EPISODE_FACTOR, 3)
        self.assertEqual(len(bc_worker.DEMO_SEEDS_V2),
                         3 * len(bc_worker.DEMO_SEEDS))
        # 当前数据污染隔离：v2 使用与全部已打开 predecessor 完全不相交的新池。
        self.assertEqual(bc_worker.DEMO_SEEDS_V2,
                         list(range(2_103_000, 2_103_384)))
        self.assertEqual(bc_worker.DEMO_SEEDS_V2[0], 2_103_000)
        self.assertEqual(bc_worker.DEMO_SEEDS_V2[-1], 2_103_383)
        self.assertFalse(set(bc_worker.DEMO_SEEDS_V2).intersection(
            bc_worker.DEMO_SEEDS))

    def test_v1_episode_count_and_seed_discipline_unaffected(self):
        # v1 局数纪律仍为 128;active registry 换用未查看的 2104000..2104127
        # (2_102 段 2026-07-27 崩溃烧毁,append-only 推进)。
        self.assertEqual(
            list(bc_worker.DEMO_SEEDS),
            list(range(2_104_000, 2_104_128)),
        )
        self.assertEqual(len(bc_worker.DEMO_SEEDS), 128)
        self.assertEqual(tuple(train_ppo._WORKER_BC_DEMO_SEEDS),
                         tuple(range(2_104_000, 2_104_128)))
        # 源文级镜像:v1 采集环仍消费 DEMO_SEEDS,v2 采集环消费 DEMO_SEEDS_V2
        src = BC_WORKER.read_text()
        self.assertIn("for i, seed in enumerate(DEMO_SEEDS):", src)
        self.assertIn("for i, seed in enumerate(DEMO_SEEDS_V2):", src)

    def test_v2_collect_runs_3x_episodes(self):
        self._install_env(lambda seed: [[{"raw": _out_band()}]])
        _, labels, groups, _, _ = bc_worker.collect_v2()
        self.assertEqual(len(labels), 3 * 128)
        self.assertTrue(np.array_equal(np.unique(groups),
                                       np.arange(2_103_000, 2_103_384)))

    def test_v1_collect_still_128_episodes(self):
        self._install_env(lambda seed: [[{"raw": _out_band()}]])
        _, labels, groups = bc_worker.collect()
        self.assertEqual(len(labels), 128)
        self.assertTrue(np.array_equal(np.unique(groups),
                                       np.arange(2_104_000, 2_104_128)))

    def test_active_pool_marker_identity_cannot_alias_burned_predecessors(self):
        active_v1 = train_ppo._bc_final_holdout_marker_identity(
            TEACHER_GENERATION_V1, bc_worker.DEMO_SEEDS)
        burned_v1 = train_ppo._bc_final_holdout_marker_identity(
            TEACHER_GENERATION_V1, range(2_100_000, 2_100_128))
        active_v2 = train_ppo._bc_final_holdout_marker_identity(
            TEACHER_GENERATION_V2, bc_worker.DEMO_SEEDS_V2)
        burned_v2 = train_ppo._bc_final_holdout_marker_identity(
            TEACHER_GENERATION_V2, range(2_101_000, 2_101_384))
        self.assertEqual(
            active_v1[0]["episode_seeds"],
            list(range(2_104_000, 2_104_128)))
        self.assertEqual(
            active_v2[0]["episode_seeds"],
            list(range(2_103_000, 2_103_384)))
        self.assertEqual(
            active_v1[1],
            "62161b134128b7d421462184ae3f4c99d"
            "4e21b374edc2cb2ae2751f7e8590943")
        self.assertEqual(
            active_v2[1],
            "10e33273f96570d6fbad5587f80bde811"
            "02c46803d49db4cd02dae11b9a2dfac")
        self.assertNotEqual(active_v1[1], burned_v1[1])
        self.assertNotEqual(active_v2[1], burned_v2[1])


class PlanAClassWeightTests(unittest.TestCase):
    """方案甲 b(2026-07-19 亲批):类平衡权重 w_c = N/(K·n_c) 计算件。"""

    def test_balanced_formula_hand_identity(self):
        # 手算:N=12, K=3;n_9=6, n_10=4, n_12=2
        # → w_9 = 12/(3·6) = 2/3, w_10 = 12/(3·4) = 1, w_12 = 12/(3·2) = 2
        labels = np.asarray([9] * 6 + [10] * 4 + [12] * 2, dtype=np.int64)
        w = bc_worker._balanced_class_weights(labels)
        self.assertEqual(w.shape, (15,))
        self.assertAlmostEqual(w[9], 12 / (3 * 6))
        self.assertAlmostEqual(w[10], 12 / (3 * 4))
        self.assertAlmostEqual(w[12], 12 / (3 * 2))
        # 类集 = 实际出现类:未出现类记 0.0(占位,CE 不消费)
        for absent in (0, 1, 5, 11, 13, 14):
            self.assertEqual(w[absent], 0.0)

    def test_single_class_weight_is_exactly_one(self):
        w = bc_worker._balanced_class_weights(np.asarray([9] * 7))
        self.assertEqual(float(w[9]), 1.0)   # N/(K·n_c) = 7/(1·7)

    def test_deterministic_pure_arithmetic(self):
        labels = np.asarray([9, 12, 9, 10, 12, 9], dtype=np.int64)
        w1 = bc_worker._balanced_class_weights(labels)
        w2 = bc_worker._balanced_class_weights(labels)
        self.assertTrue(np.array_equal(w1, w2))
        # 与标签顺序无关(纯计数算术)
        w3 = bc_worker._balanced_class_weights(np.sort(labels))
        self.assertTrue(np.array_equal(w1, w3))

    def test_empty_labels_fail_loud(self):
        with self.assertRaisesRegex(RuntimeError, "空标签集"):
            bc_worker._balanced_class_weights(
                np.asarray([], dtype=np.int64))


class V1TrainPathUnweightedTests(_PatchMixin, unittest.TestCase):
    """方案甲 b 铁律面:v1 训练路径零触碰(v1 调用不传权)。"""

    def test_train_bc_shared_function_default_is_none(self):
        import inspect
        sig = inspect.signature(bc_worker.train_bc)
        self.assertIsNone(sig.parameters["class_weights"].default)

    def test_call_sites_verbatim_v1_unweighted_v2_weighted(self):
        src = BC_WORKER.read_text()
        # v1 主训调用零加权原封;v2 主训调用系类平衡加权(方案甲 b)
        self.assertIn(
            "model, top1, recalls = train_bc(\n"
            "        X, Y, groups,\n"
            "        required_recall_actions="
            "_WORKER_BC_REQUIRED_RECALL_ACTIONS)",
            src)
        self.assertIn(
            "class_weights=class_weights_v2,\n"
            "        masks=masks, reserve_a12_path=True)", src)

    def test_v1_main_primary_train_call_passes_no_weights(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            out = pathlib.Path(d) / "runs" / "bc-worker"
            out.mkdir(parents=True)
            collect_saw_marker = False

            def fake_train_bc(
                    X, Y, groups, class_weights=None, *,
                    epochs=bc_worker._BC_PRIMARY_EPOCHS, masks=None,
                    required_recall_actions=()):
                del epochs, masks
                calls.append(class_weights)
                self.assertEqual(required_recall_actions, (14,))
                return object(), 0.99, {9: 0.99, 14: 0.99}

            episodes = np.arange(100, 116, dtype=np.int64)
            groups = np.repeat(episodes, 8)
            n = len(groups)
            labels = np.full(n, 9, dtype=np.int64)
            labels.reshape(len(episodes), 8)[:, :4] = 14
            dataset = (np.zeros((n, 298), dtype=np.float32), labels, groups)

            def fake_collect():
                nonlocal collect_saw_marker
                marker, _, _ = bc_worker._final_holdout_marker_path(
                    out, TEACHER_GENERATION_V1, bc_worker.DEMO_SEEDS)
                self.assertTrue(marker.is_file())
                marker_record = json.loads(marker.read_text())
                self.assertEqual(
                    marker_record["consumption_stage"],
                    "before_pool_collection")
                self.assertNotIn("candidate", marker_record)
                collect_saw_marker = True
                return dataset

            self._patch("OUT", out)
            self._patch("collect", fake_collect)
            self._patch("train_bc", fake_train_bc)
            self._patch("_score_bc_model",
                        lambda m, X, y, g, masks=None,
                               required_recall_actions=():
                        (0.99, {9: 0.99, 14: 0.99}))
            self._patch("export_sb3_sd", lambda m: {"w": torch.zeros(1)})
            self._patch("artifact_provenance", lambda: {
                "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
                "implementation_sha256": "a" * 64,
                "generator_sha256": BC_WORKER_SHA,
                "manager_npz_sha256": "c" * 64})
            bc_worker.main()
            self.assertTrue(collect_saw_marker)
            self.assertEqual(len(calls), 1)     # 主训一次
            self.assertIsNone(calls[0])         # v1 调用不传权(方案甲铁律)
            rec = json.loads(
                (out / "bc_report.json").read_text())
            self.assertIs(rec["class_weighted_retry"], False)
            # v1 回执面原封:方案甲新字段只入 v2 回执,禁漏入 v1
            self.assertNotIn("collection_episodes", rec)
            self.assertNotIn("class_weights", rec)

    def test_v1_failed_retry_skips_final_scoring_after_pool_is_burned(self):
        with tempfile.TemporaryDirectory() as d:
            epochs_seen = []
            final_calls = 0
            out = pathlib.Path(d) / "runs" / "bc-worker"
            out.mkdir(parents=True)

            def fake_train_bc(
                    X, Y, groups, class_weights=None, *,
                    epochs=bc_worker._BC_PRIMARY_EPOCHS, masks=None,
                    required_recall_actions=()):
                del X, Y, groups, class_weights, masks
                self.assertEqual(required_recall_actions, (14,))
                epochs_seen.append(epochs)
                return object(), 0.90, {9: 0.90, 14: 0.90}

            def forbidden_final(*args, **kwargs):
                del args, kwargs
                nonlocal final_calls
                final_calls += 1
                raise AssertionError("selection FAIL 后不得读取 final")

            episodes = np.arange(100, 116, dtype=np.int64)
            groups = np.repeat(episodes, 8)
            n = len(groups)
            labels = np.full(n, 9, dtype=np.int64)
            labels.reshape(len(episodes), 8)[:, :4] = 14
            dataset = (np.zeros((n, 298), dtype=np.float32), labels, groups)

            def fake_collect():
                marker, _, _ = bc_worker._final_holdout_marker_path(
                    out, TEACHER_GENERATION_V1, bc_worker.DEMO_SEEDS)
                self.assertTrue(marker.is_file())
                return dataset

            self._patch("OUT", out)
            self._patch("collect", fake_collect)
            self._patch("train_bc", fake_train_bc)
            self._patch("_score_bc_model", forbidden_final)
            self._patch("artifact_provenance", lambda: {
                "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
                "implementation_sha256": "a" * 64,
                "generator_sha256": BC_WORKER_SHA,
                "manager_npz_sha256": "c" * 64})
            with self.assertRaisesRegex(
                    RuntimeError, "candidate validation FAIL"):
                bc_worker.main()
            self.assertEqual(
                epochs_seen,
                [bc_worker._BC_PRIMARY_EPOCHS,
                 bc_worker._BC_WEIGHTED_RETRY_EPOCHS])
            self.assertEqual(final_calls, 0)
            rec = json.loads(
                (out / "bc_report.json").read_text())
            self.assertEqual(rec["failure_stage"], "candidate_selection")


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
        # 干净脚本(dispatch → 10)+ fuse/no-effect 各一发，二者都不得入池。
        def script(seed):
            if seed == bc_worker.DEMO_SEEDS[0]:
                return [[{"raw": _out_band(), "overridden": True},
                         {"raw": _out_band(), "executed_action": None},
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
        for threshold in ("0.70", "0.8"):
            bad = subprocess.run(
                [sys.executable, str(BC_WORKER), "--v2",
                 "--preventive-threshold", threshold],
                text=True, capture_output=True, check=False)
            self.assertNotEqual(
                bad.returncode, 0,
                f"未注册阈 {threshold} 必须由 argparse choices 拦截")
        help_run = subprocess.run(
            [sys.executable, str(BC_WORKER), "--help"],
            text=True, capture_output=True, check=False)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--v2", help_run.stdout)


if __name__ == "__main__":
    unittest.main()
