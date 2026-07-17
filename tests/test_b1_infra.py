"""B1 捆绑评测基建之自包含快速回归(PREREG-B1 E7;不启动引擎/训练/评测)。

覆盖:三旋钮步集行为(E0)、MS 计算(E1)、签名判别器 v2(E5)、
calib 步表解析(E3)、W-G0 烟测步表窗内保证、腿配方逐字断言、
统计纪律工具(符号检验/去杠杆/临线/Clopper-Pearson)、E8 提取器。
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

import extract_canary_set  # noqa: E402
import probe_composite_signature as pcs  # noqa: E402
import run_b1_infra as b1  # noqa: E402
from train_ppo import (  # noqa: E402
    AtomicRolloutCheckpointCallback,
    DryAnchorSentinel,
    WorkerSentinelCallback,
)


def _row(seed, ret=100.0, depth=1, died=False, kills=10, farm_n=5,
         farm_tau_mean=30.0, farm_tau_sum=150, farm_descend=0, windows=6,
         beats=100, overrides=2, cap=0, mode_seq="FFFFFF"):
    return {"seed": seed, "ret": ret, "depth": depth, "died": died,
            "kills": kills, "farm_n": farm_n, "farm_tau_mean": farm_tau_mean,
            "farm_tau_sum": farm_tau_sum, "farm_descend": farm_descend,
            "windows": windows, "beats": beats, "overrides": overrides,
            "cap": cap, "mode_seq": mode_seq}


class KnobStepSetTests(unittest.TestCase):
    """E0 三旋钮:步集行为与 CLI 校验(旋钮封闭枚举三枚)。"""

    def test_ckpt_every_steps_quantizes_and_aligns_from_resume_start(self):
        cb = AtomicRolloutCheckpointCallback(
            pathlib.Path("/unused"), every_steps=b1.CKPT_EVERY)
        cb.model = types.SimpleNamespace(
            n_steps=512,
            get_env=lambda: types.SimpleNamespace(num_envs=4))
        cb.num_timesteps = b1.KING_STEPS
        cb._on_training_start()
        self.assertEqual(cb.period, 98_304)          # 恰為 48×2048,量子对齐
        self.assertEqual(cb.next_at, 3_596_288)      # 腿内第一个金丝雀点
        # 亚量子输入被抬到量子(回调原逻辑,旋钮不改变语义)
        cb2 = AtomicRolloutCheckpointCallback(
            pathlib.Path("/unused"), every_steps=1)
        cb2.model = cb.model
        cb2.num_timesteps = 0
        cb2._on_training_start()
        self.assertEqual(cb2.period, 2_048)

    def test_sentinel_every_aligns_to_next_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            cb = WorkerSentinelCallback(pathlib.Path(d),
                                        every=b1.SENTINEL_EVERY)
            cb.num_timesteps = b1.KING_STEPS
            cb._on_training_start()
            self.assertEqual(cb.next_at, 3_538_944)  # ((3497984//49152)+1)×49152

    def test_dry_anchor_every_aligns_to_next_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            demos = run_dir / "demos.npz"
            x = np.zeros((10, 298), dtype=np.float32)
            x[:, 297] = 1.0
            y = np.zeros(10, dtype=np.int64)
            episode_id = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
            np.savez(demos, X=x, Y=y, episode_id=episode_id)
            sha = hashlib.sha256(demos.read_bytes()).hexdigest()
            cb = DryAnchorSentinel(run_dir, str(demos), sha,
                                   every=b1.DRY_ANCHOR_EVERY)
            cb.num_timesteps = b1.KING_STEPS
            cb._on_training_start()
            self.assertEqual(cb.next_at, 3_538_944)

    def test_cli_rejects_nonpositive_knobs_and_documents_them(self):
        help_run = subprocess.run(
            [sys.executable, str(ROOT / "train" / "train_ppo.py"), "--help"],
            text=True, capture_output=True, check=False)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        for knob in ("--ckpt-every-steps", "--sentinel-every",
                     "--dry-anchor-every"):
            self.assertIn(knob, help_run.stdout)
        bad = subprocess.run(
            [sys.executable, str(ROOT / "train" / "train_ppo.py"),
             "--total-steps", "2048", "--ckpt-every-steps", "0"],
            text=True, capture_output=True, check=False)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("--ckpt-every-steps 必须 > 0", bad.stderr)


class SmokeStepTableTests(unittest.TestCase):
    """W-G0 烟测专用步表:窗内 ≥1 calib/ckpt/sentinel/dry-anchor 保证。"""

    def test_window_and_quantum(self):
        self.assertEqual(b1.SMOKE_STEPS % 2_048, 0)
        self.assertEqual(b1.SMOKE_END, b1.KING_STEPS + b1.SMOKE_STEPS)
        self.assertEqual(b1.SMOKE_SEED, 305_000)

    def test_at_least_one_of_each_instrument_fires_in_window(self):
        window = range(b1.KING_STEPS + 1, b1.SMOKE_END + 1)
        calib_in = [p for p in b1.SMOKE_CALIB if p in window]
        self.assertGreaterEqual(len(calib_in), 1)
        self.assertTrue(all(p % 2_048 == 0 for p in b1.SMOKE_CALIB))
        first_ckpt = b1.KING_STEPS + b1.CKPT_EVERY
        self.assertIn(first_ckpt, window)
        sentinel_first = ((b1.KING_STEPS // b1.SENTINEL_EVERY) + 1) \
            * b1.SENTINEL_EVERY
        self.assertIn(sentinel_first, window)
        dry_first = ((b1.KING_STEPS // b1.DRY_ANCHOR_EVERY) + 1) \
            * b1.DRY_ANCHOR_EVERY
        self.assertIn(dry_first, window)

    def test_knobs_cmd_differs_from_bare_only_by_instrument_knobs(self):
        bare, knobs = b1._smoke_cmd("bare"), b1._smoke_cmd("knobs")
        bare_renamed = ["b1-smoke-knobs" if x == "b1-smoke-bare" else x
                        for x in bare]
        self.assertEqual(knobs[:len(bare)], bare_renamed)  # 仅 run-name 异
        self.assertEqual(knobs[len(bare):], [
            "--calib-probes", ",".join(str(x) for x in b1.SMOKE_CALIB),
            "--ckpt-every-steps", str(b1.CKPT_EVERY),
            "--sentinel-every", str(b1.SENTINEL_EVERY),
            "--dry-anchor-every", str(b1.DRY_ANCHOR_EVERY)])
        for cmd in (bare, knobs):
            self.assertIn("--seed", cmd)
            self.assertEqual(cmd[cmd.index("--seed") + 1], "305000")
            self.assertNotIn("--board", cmd)


class CalibStepTableTests(unittest.TestCase):
    """E3 十点步表:腿内每 +49,152,k=1..10,量子对齐,预注册逐字。"""

    PREREG_LITERAL = (3547136, 3596288, 3645440, 3694592, 3743744,
                      3792896, 3842048, 3891200, 3940352, 3989504)

    def test_table_matches_prereg_literal(self):
        self.assertEqual(b1.CALIB_PROBES, self.PREREG_LITERAL)
        self.assertEqual(
            b1.CALIB_PROBES,
            tuple(b1.KING_STEPS + 49_152 * k for k in range(1, 11)))
        self.assertTrue(all(p % 2_048 == 0 for p in b1.CALIB_PROBES))

    def test_cli_string_roundtrip_matches_train_ppo_parser(self):
        cli = ",".join(str(x) for x in b1.CALIB_PROBES)
        parsed = [int(x) for x in cli.split(",") if x.strip()]   # train_ppo 同式
        self.assertEqual(tuple(parsed), b1.CALIB_PROBES)

    def test_canary_points_and_leg_account(self):
        self.assertEqual(b1.CANARY_STEPS,
                         tuple(b1.KING_STEPS + 98_304 * k for k in range(1, 5)))
        self.assertEqual(b1.EXTRA_CKPT_STEP, b1.KING_STEPS + 491_520)
        self.assertEqual(b1.NT_TARGET, b1.KING_STEPS + b1.LEG_STEPS)
        self.assertEqual(b1.LEG_STEPS % 2_048, 0)
        self.assertEqual(b1.SEED, 303_000 + 1_000)


class LegRecipeTests(unittest.TestCase):
    """P8 腿:ctrl 配方逐字承继,偏离封闭枚举五处。"""

    def test_leg_cmd_carries_ctrl_recipe_verbatim_plus_five_deviations(self):
        cmd = b1.leg_cmd()

        def val(flag):
            return cmd[cmd.index(flag) + 1]

        for flag, expected in (("--algo", "mppo"), ("--gamma", "1.0"),
                               ("--max-steps", "3000"), ("--n-steps", "512"),
                               ("--num-envs", "4"), ("--lr", "3e-4"),
                               ("--ent-coef", "0.005"),
                               ("--distill-beta", "0.015625"),
                               ("--seed", "304000"),
                               ("--total-steps", "499712"),
                               ("--ckpt-every-steps", "98304"),
                               ("--sentinel-every", "49152"),
                               ("--dry-anchor-every", "49152")):
            self.assertEqual(val(flag), expected, flag)
        for flag in ("--worker", "--skip-dry", "--no-drink-sovereignty",
                     "--allow-legacy-resume", "--calib-record-only"):
            self.assertIn(flag, cmd)
        self.assertNotIn("--board", cmd)
        self.assertTrue(val("--resume-from").endswith(
            "v28-worker-leg1/model_final.zip"))

    def test_exam_tags_closed_enumeration(self):
        self.assertEqual(len(b1.CANARY_TAGS), 8)
        self.assertEqual(len(set(b1.ALL_EXAM_TAGS)), len(b1.ALL_EXAM_TAGS))
        for t in b1.HOLDOUT_TAGS:
            self.assertIn(t, b1.ALL_EXAM_TAGS)


class MsComputationTests(unittest.TestCase):
    """E1/D3:MS(s) = |Δ_H − Δ_M29|,带符号量并列,超线计数。"""

    def test_ms_vectors_hand_computed(self):
        leg_h = {s: _row(s, ret=r) for s, r in
                 {1: 70.0, 2: 100.0, 3: 110.0, 4: 95.0}.items()}
        ref_h = {s: _row(s, ret=100.0) for s in (1, 2, 3, 4)}
        leg_m29 = {s: _row(s, ret=r) for s, r in
                   {1: 105.0, 2: 100.0, 3: 85.0, 4: 105.0}.items()}
        ref_m29 = {s: _row(s, ret=100.0) for s in (1, 2, 3, 4)}
        v = b1.ms_vectors(leg_h, leg_m29, ref_h, ref_m29)
        # Δ_H = {-30, 0, +10, -5};Δ_M29 = {+5, 0, -15, +5}
        self.assertEqual(v["signed_dh_minus_dm"],
                         {1: -35.0, 2: 0.0, 3: 25.0, 4: -10.0})
        self.assertEqual(v["ms_max"], 35.0)
        self.assertEqual(v["ms_median"], 17.5)
        self.assertEqual(v["over_line_seeds"], [1, 3])
        self.assertEqual(v["n_over_line"], 2)
        self.assertEqual(v["flag_line"], 20.0)

    def test_signed_quantity_catches_m29_worse_flip(self):
        # 'M29 侧更差'型:Δ_H=0、Δ_M29=−40 → 带符号 +40(绝对值会吞没方向)
        leg_h = {1: _row(1, ret=100.0)}
        ref_h = {1: _row(1, ret=100.0)}
        leg_m29 = {1: _row(1, ret=60.0)}
        ref_m29 = {1: _row(1, ret=100.0)}
        v = b1.ms_vectors(leg_h, leg_m29, ref_h, ref_m29)
        self.assertEqual(v["signed_dh_minus_dm"][1], 40.0)


class SignatureDiscriminatorTests(unittest.TestCase):
    """E5:复合签名(D3 常量)与判别器 v2(近失判据)。"""

    def test_invariant_class_v2(self):
        a = _row(7000)
        self.assertEqual(pcs.invariant_class_v2(a, dict(a)),
                         "strict_invariant")
        near = dict(a)
        near["windows"] = a["windows"] + 3
        near["mode_seq"] = "FFFFFFF"
        self.assertEqual(pcs.invariant_class_v2(a, near), "near_miss")
        var = dict(a)
        var["ret"] = a["ret"] + 1.0
        self.assertEqual(pcs.invariant_class_v2(a, var), "variable")

    def test_composite_signature_conjunction(self):
        ref = {7001: _row(7001, depth=2), 7002: _row(7002, depth=3),
               7003: _row(7003, depth=1), 7004: _row(7004, depth=2)}
        leg = {7001: _row(7001, mode_seq="FFFF"),          # D=0
               7002: _row(7002, mode_seq="FDFF"),          # D=1 → (ii) 失守
               7003: _row(7003, mode_seq="FFFF"),
               7004: _row(7004, mode_seq="FFFF")}          # D=0 但 τ 出带
        tau = {7001: 26.0, 7002: 25.0, 7004: 55.0}
        sig = pcs.composite_signature(ref, leg, tau)
        self.assertEqual(sig["ref_depth2_seeds"], [7001, 7002, 7004])
        self.assertEqual(sig["hits"], [7001])
        self.assertTrue(sig["per_seed"][7002]["cond_iii_tau_floor"])
        self.assertFalse(sig["per_seed"][7002]["cond_ii_leg_d_windows_zero"])
        self.assertFalse(sig["per_seed"][7004]["cond_iii_tau_floor"])
        # τ 地板闭区间端点
        self.assertTrue(pcs.TAU_FLOOR_LO <= 25.0 <= pcs.TAU_FLOOR_HI)
        self.assertEqual((pcs.TAU_FLOOR_LO, pcs.TAU_FLOOR_HI), (25.0, 40.0))

    def test_triage_variable_is_flock_and_invariant_is_worker_damage(self):
        ref = {7001: _row(7001, depth=2), 7002: _row(7002, depth=2)}
        leg_h = {7001: _row(7001, ret=50.0, mode_seq="FFFF"),
                 7002: _row(7002, ret=40.0, mode_seq="FFFF")}
        leg_m29 = {7001: _row(7001, ret=150.0, mode_seq="FDFF"),  # 轨迹可变
                   7002: _row(7002, ret=40.0, mode_seq="FFFF")}   # 严格不变
        sig = pcs.composite_signature(ref, leg_h, {7001: 30.0, 7002: 30.0})
        tri = pcs.triage(sig, leg_h, leg_m29)
        self.assertEqual(tri["per_seed"][7001]["verdict"], "F-lock 型")
        self.assertEqual(tri["n_flock_type"], 1)
        self.assertEqual(tri["n_worker_damage_candidate"], 1)

    def test_d_window_count_counts_death_marked_windows(self):
        self.assertEqual(pcs.d_window_count(_row(1, mode_seq="FD†FD")), 2)


class StatsDisciplineTests(unittest.TestCase):
    """D3 统计纪律:符号检验/去杠杆/临线/CP 区间/RB.4 判决格。"""

    def test_sign_test_and_deleveraged(self):
        st = b1.sign_test([-1.0, -2.0, 3.0, -4.0, 0.0])
        self.assertEqual((st["neg"], st["pos"], st["ties"]), (3, 1, 1))
        self.assertAlmostEqual(st["p_one_sided"], 0.3125)
        dl = b1.deleveraged_mean({1: -1.0, 2: -2.0, 3: 30.0})
        self.assertEqual(dl["dropped_seed"], 3)
        self.assertAlmostEqual(dl["mean"], -1.5)

    def test_band_judge_borderline_clauses(self):
        j = b1.band_judge(-2.1, -45.0, -2.0)      # 距边界 0.1 ≤ 0.05×43
        self.assertTrue(j["in_band"])
        self.assertIn("borderline_note", j)
        j2 = b1.band_judge(-20.0, -45.0, -2.0)
        self.assertNotIn("borderline_note", j2)
        j3 = b1.band_judge(3, 2, 14, integer=True)   # 计数量距边界 1
        self.assertIn("borderline_note", j3)

    def test_clopper_pearson_known_values(self):
        lo, hi = b1.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** 0.1, places=3)
        lo, hi = b1.clopper_pearson(10, 10)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.025 ** 0.1, places=3)

    def test_rb4_grid_five_cells(self):
        strong_sign = {"neg": 25, "pos": 7, "ties": 0, "p_one_sided": 0.001}
        weak_sign = {"neg": 10, "pos": 22, "ties": 0, "p_one_sided": 0.9}
        delev_in = {"dropped_seed": 1, "dropped_value": -1.0, "mean": -20.0}
        delev_out = {"dropped_seed": 1, "dropped_value": -1.0, "mean": 0.0}
        self.assertEqual(
            b1.rb4_grid(-20.0, -8.0, strong_sign, delev_in)["cell"], "复现")
        self.assertEqual(
            b1.rb4_grid(-20.0, -1.0, weak_sign, delev_out)["cell"],
            "杠杆驱动候选")
        self.assertEqual(
            b1.rb4_grid(1.0, -1.0, weak_sign, delev_out)["cell"], "不复现")
        self.assertEqual(
            b1.rb4_grid(1.0, -6.0, weak_sign, delev_out)["cell"],
            "均值掩蔽候选")
        self.assertEqual(
            b1.rb4_grid(-50.0, -30.0, strong_sign,
                        {"dropped_seed": 1, "dropped_value": -200.0,
                         "mean": -50.0})["cell"], "放大")


class ExtractorTests(unittest.TestCase):
    """E8:W-C depth≥2 机器提取(depth2_count 先例口径)。"""

    def test_depth2_extraction_and_control_guards(self):
        rows = [_row(7000, depth=1), _row(7001, depth=2), _row(7002, depth=3),
                _row(7003, depth=1)]
        self.assertEqual(extract_canary_set.depth2_seeds(rows), [7001, 7002])
        with tempfile.TemporaryDirectory() as d:
            arch = pathlib.Path(d) / "ref.json"
            arch.write_text(json.dumps({"rows": rows}))
            info = extract_canary_set.extract(arch, (7003,))
            self.assertEqual(info["depth2_seeds"], [7001, 7002])
            self.assertEqual(info["n_D"], 2)
            self.assertEqual(info["C"], [7001, 7002, 7003])
            with self.assertRaisesRegex(ValueError, "阴性对照失义"):
                extract_canary_set.extract(arch, (7001,))
            with self.assertRaisesRegex(ValueError, "不在档案种子面内"):
                extract_canary_set.extract(arch, (9999,))

    def test_duplicate_seed_rejected(self):
        rows = [_row(7001, depth=2), _row(7001, depth=2)]
        with self.assertRaisesRegex(ValueError, "重复 seed"):
            extract_canary_set.depth2_seeds(rows)


class NullIntrusionComparatorTests(unittest.TestCase):
    """W-G0 张量级判据工具:torch.equal 树比对与状态摘要。"""

    def test_tree_equal_detects_tensor_and_scalar_diffs(self):
        a = {"w": torch.zeros(3), "g": [{"lr": 3e-4, "step": 1}]}
        b_same = {"w": torch.zeros(3), "g": [{"lr": 3e-4, "step": 1}]}
        diffs = []
        b1._tree_equal(a, b_same, "root", diffs)
        self.assertEqual(diffs, [])
        b_tensor = {"w": torch.ones(3), "g": [{"lr": 3e-4, "step": 1}]}
        diffs = []
        b1._tree_equal(a, b_tensor, "root", diffs)
        self.assertEqual(diffs, ["root.w"])
        b_scalar = {"w": torch.zeros(3), "g": [{"lr": 3e-4, "step": 2}]}
        diffs = []
        b1._tree_equal(a, b_scalar, "root", diffs)
        self.assertEqual(diffs, ["root.g[0].step"])

    def test_state_digest_is_content_sensitive(self):
        pol = {"w": torch.zeros(2)}
        opt = {"state": {0: {"exp_avg": torch.zeros(2)}}}
        d1 = b1._state_digest(pol, opt)
        d2 = b1._state_digest({"w": torch.zeros(2)},
                              {"state": {0: {"exp_avg": torch.zeros(2)}}})
        self.assertEqual(d1, d2)
        d3 = b1._state_digest({"w": torch.full((2,), 1e-7)}, opt)
        self.assertNotEqual(d1, d3)


if __name__ == "__main__":
    unittest.main()
