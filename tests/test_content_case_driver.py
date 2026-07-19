"""v33 内容案驱动器之自包含快速回归(PREREG-内容案-课⑤x④乙 对应件;
不启动引擎/训练/评测,循 tests/test_b1_infra.py 与 tests/test_content_case.py
风格;一切写台账路径经 monkeypatch 捕获,正账零触碰)。

覆盖(施工任务清单逐项):
- LEGS 表逐字断言(三腿完整 CLI 逐项:共用命令形 + D3 逐腿附加项);
- 退出码表(D5 集中常量化);
- W-LAUNCH 发车令闸(无 LAUNCH_ORDER 必 exit 9;非失败态);
- 阶段序防后见(后阶段在前阶段未 done 时拒跑;N12 PASS 语义谓词);
- 304000/308000 对账断言函数(合成台账正反例)+ 评测池段守卫;
- 金丝雀补评截止断言(该腿 s16 FIRING_START 前);
- 额度计数(P2 评测发 2 / P3 腿点火 2);
- 附:退火主表与腿终复核、REF_BITEQ 纯比对、烟测命令形、W-PIN 冻结常量
  对实件抽验、判据纯函数(资格/胜者/课⑤/④乙档位)、MS 计算;
- rev4 十二附二补铸对应件(修①-⑤):N12_GATE 落定值硬取键 + L-full 点火前
  demos 字节链(④);G0-2a sentinel/dry-anchor 双行型分计(⑤);
  DRY_CURRICULUM_TABLE 全精度落账(⑥);G0-2b episode 种子序列恒等之
  可见面等价物(③);共用命令形 E5①② 两旋钮 + DRYWIN_METRICS 转录件(①)。
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import run_v33_content as v33  # noqa: E402
from eval_contract import OperationalFailure  # noqa: E402

MAIN_TABLE_LITERAL = "linear:1.0:0.5:147,hold:0.5:97"
CALIB_LITERAL = ("3547136,3596288,3645440,3694592,3743744,"
                 "3792896,3842048,3891200,3940352,3989504")


def _staged_events(upto: str | None = None) -> list[dict]:
    """合成台账:STAGE_SEQUENCE 依序全事件(upto 为止,含 upto)。"""
    seq = [
        {"event": "G0_BASELINE"},
        {"event": "BC_REGEN"},
        {"event": "N12_GATE", "gate": "PASS"},
        {"event": "G0_ENDPOINT"},
        {"event": "G0_NULLINTRUSION", "verdict": "PASS"},
        {"event": "G0_SMOKE", "verdict": "PASS"},
        {"event": "G0_GHOST"},
        {"event": "DEMO_LEDGER"},
        {"event": "REF_BITEQ", "ref": "launch"},
        {"event": "REF_BITEQ", "ref": "science"},
        {"event": "FREEZE_SHA", "sha": "deadbeef"},
        {"event": "LAUNCH_ORDER", "order": "亲发原文占位(合成测试件)"},
    ]
    if upto is None:
        return seq
    out = []
    for e in seq:
        out.append(e)
        if e["event"] == upto:
            break
    return out


def _row(seed, ret=100.0, depth=1, died=False, mode_seq="FFF"):
    return {"seed": seed, "ret": ret, "depth": depth, "died": died,
            "mode_seq": mode_seq}


def _full32_doc(died=2, override=0.0, cap=0.0, descend=0.0, mode_seq="FFF",
                mean=100.0):
    rows = [_row(7000 + i, ret=mean, mode_seq=mode_seq) for i in range(32)]
    return {"rows": rows,
            "agg": {"died": died, "override_rate": override,
                    "cap_rate": cap, "farm_descend_rate": descend,
                    "ret_mean": mean, "n": 32}}


class LegsTableTests(unittest.TestCase):
    """D3 LEGS 表逐字:三腿完整 CLI(共用命令形 + 逐腿附加项)。"""

    def _val(self, cmd, flag):
        return cmd[cmd.index(flag) + 1]

    def test_shared_recipe_verbatim(self):
        for leg in ("v33-base", "v33-cur", "v33-full"):
            cmd = v33.leg_cmd(leg)
            for flag, expected in (("--algo", "mppo"), ("--gamma", "1.0"),
                                   ("--max-steps", "3000"),
                                   ("--n-steps", "512"), ("--num-envs", "4"),
                                   ("--lr", "3e-4"), ("--ent-coef", "0.005"),
                                   ("--seed", "304000"),
                                   ("--total-steps", "499712"),
                                   ("--distill-beta", "0.015625"),
                                   ("--calib-probes", CALIB_LITERAL),
                                   ("--ckpt-every-steps", "98304"),
                                   ("--sentinel-every", "49152"),
                                   ("--dry-anchor-every", "49152"),
                                   # rev4 D3 勘正增列末两枚(十二附二①)
                                   ("--distill-ce-probe-every", "49152"),
                                   ("--drywin-metrics-every", "49152"),
                                   ("--run-name", leg)):
                self.assertEqual(self._val(cmd, flag), expected, (leg, flag))
            for flag in ("--worker", "--allow-legacy-resume",
                         "--calib-record-only"):
                self.assertIn(flag, cmd, (leg, flag))
            # 主权开系默认;一切发无 --board
            self.assertNotIn("--no-drink-sovereignty", cmd)
            self.assertNotIn("--board", cmd)
            self.assertTrue(self._val(cmd, "--resume-from").endswith(
                "v28-worker-leg1/model_final.zip"))
            self.assertTrue(self._val(cmd, "--teacher-sd").endswith(
                "bc-worker/policy_sd.pt"))          # BC_SD(v1)
            self.assertTrue(self._val(cmd, "--teacher-override").endswith(
                "v32/king_anchor_sd.pt"))           # KING_SD 锚随王走不破

    def test_l_base_extras(self):
        cmd = v33.leg_cmd("v33-base")
        self.assertIn("--skip-dry", cmd)            # 共用形不产生 p≡1.0,必显携
        self.assertNotIn("--dry-curriculum-schedule", cmd)
        self.assertNotIn("--bc-aux-lambda", cmd)
        self.assertNotIn("--bc-aux-demos", cmd)

    def test_l_cur_extras(self):
        cmd = v33.leg_cmd("v33-cur")
        self.assertNotIn("--skip-dry", cmd)         # 两旗互斥(E1)
        self.assertEqual(self._val(cmd, "--dry-curriculum-schedule"),
                         MAIN_TABLE_LITERAL)
        self.assertNotIn("--bc-aux-lambda", cmd)

    def test_l_full_extras(self):
        cmd = v33.leg_cmd("v33-full")
        self.assertNotIn("--skip-dry", cmd)
        self.assertEqual(self._val(cmd, "--dry-curriculum-schedule"),
                         MAIN_TABLE_LITERAL)
        self.assertEqual(self._val(cmd, "--bc-aux-lambda"), "0.015625")
        self.assertTrue(self._val(cmd, "--bc-aux-demos").endswith(
            "runs/bc-worker-v2/demos.npz"))         # D3 字面路径

    def test_leg_accounting_constants(self):
        self.assertEqual(v33.NT_TARGET, 3_997_696)
        self.assertEqual(v33.NT_TARGET, v33.KING_STEPS + v33.LEG_STEPS)
        self.assertEqual(v33.SEED, 304_000)
        self.assertEqual(v33.LEG_NAMES, ("v33-base", "v33-cur", "v33-full"))
        self.assertEqual(v33.CURRICULUM_LEGS, ("v33-cur", "v33-full"))


class ExitCodeTableTests(unittest.TestCase):
    """D5 退出码表(集中常量化,逐字)。"""

    def test_exit_code_table_closed_enumeration(self):
        self.assertEqual(set(v33.EXIT_CODES), {0, 2, 3, 4, 5, 6, 7, 8, 9})
        self.assertEqual(v33.EXIT_CODES[0], "案结/幂等")
        self.assertIn("PREFLIGHT_FAIL", v33.EXIT_CODES[3])
        self.assertIn("锁冲突", v33.EXIT_CODES[4])
        self.assertIn("发车前漂移", v33.EXIT_CODES[5])
        self.assertIn("案中漂移", v33.EXIT_CODES[6])
        self.assertEqual(v33.EXIT_CODES[7], "CASE_HALT_G0")
        self.assertEqual(v33.EXIT_CODES[8], "REF_DIVERGENCE")
        self.assertIn("AWAITING_LAUNCH", v33.EXIT_CODES[9])
        self.assertIn("非失败", v33.EXIT_CODES[9])


class LaunchGateTests(unittest.TestCase):
    """W-LAUNCH 发车令闸:无 LAUNCH_ORDER 必 exit 9(非失败态)。"""

    def test_missing_launch_order_exits_9(self):
        with self.assertRaises(SystemExit) as cm:
            v33.launch_gate(_staged_events(upto="FREEZE_SHA"))
        self.assertEqual(cm.exception.code, 9)

    def test_launch_order_present_opens_gate(self):
        self.assertIsNone(v33.launch_gate(_staged_events()))

    def test_gate_sits_between_freeze_and_legs_in_sequence(self):
        seq = v33.STAGE_SEQUENCE
        self.assertLess(seq.index("FREEZE_SHA"), seq.index("LAUNCH_ORDER"))
        self.assertLess(seq.index("LAUNCH_ORDER"), seq.index("LEGS"))


class StageOrderTests(unittest.TestCase):
    """阶段序防后见(D2 严格串行):后阶段在前阶段未 done 时拒跑。"""

    def test_full_prefix_passes(self):
        v33.assert_stage_prereqs(_staged_events(), "LEGS")

    def test_missing_middle_stage_rejected(self):
        events = [e for e in _staged_events()
                  if e["event"] != "G0_GHOST"]
        with self.assertRaisesRegex(v33.PreflightFailure, "G0_GHOST"):
            v33.assert_stage_prereqs(events, "FREEZE_SHA")

    def test_later_stage_refused_when_nothing_done(self):
        with self.assertRaisesRegex(v33.PreflightFailure, "防后见"):
            v33.assert_stage_prereqs([], "N12_GATE")

    def test_n12_fail_event_does_not_satisfy_pass_predicate(self):
        events = _staged_events(upto="BC_REGEN") + [
            {"event": "N12_GATE", "gate": "FAIL"}]
        with self.assertRaisesRegex(v33.PreflightFailure, "N12_GATE"):
            v33.assert_stage_prereqs(events, "G0_ENDPOINT")

    def test_single_ref_biteq_insufficient(self):
        events = [e for e in _staged_events(upto="FREEZE_SHA")
                  if not (e["event"] == "REF_BITEQ"
                          and e.get("ref") == "science")]
        with self.assertRaisesRegex(v33.PreflightFailure, "REF_BITEQ"):
            v33.assert_stage_prereqs(events, "FREEZE_SHA")


class SeedAssertionTests(unittest.TestCase):
    """304000 对账(恰为且仅为 infra-b1 P8 leg_start)/ 308000 处女 / 池守卫。"""

    P8_LINE = json.dumps({"event": "leg_start", "leg": "b1-p8",
                          "seed": 304000})

    def test_304000_exact_provenance_passes(self):
        lines = {"infra-b1": ['{"event": "x"}', self.P8_LINE],
                 "v32": ['{"event": "y"}'], "recal-g1": []}
        v33.assert_seed_304000_provenance(lines)

    def test_304000_extra_occurrence_rejected(self):
        lines = {"infra-b1": [self.P8_LINE],
                 "v32": ['{"event": "z", "note": "seed 304000 misuse"}']}
        with self.assertRaisesRegex(v33.PreflightFailure, "304000"):
            v33.assert_seed_304000_provenance(lines)

    def test_304000_zero_occurrence_rejected(self):
        # "恰为"要求在册:P8 leg_start 缺席同样失败
        with self.assertRaisesRegex(v33.PreflightFailure, "304000"):
            v33.assert_seed_304000_provenance({"infra-b1": ['{"event": "x"}']})

    def test_304000_wrong_event_rejected(self):
        wrong = json.dumps({"event": "exam_ok", "seed": 304000})
        with self.assertRaisesRegex(v33.PreflightFailure, "leg_start"):
            v33.assert_seed_304000_provenance({"infra-b1": [wrong]})

    def test_304000_wrong_ledger_rejected(self):
        with self.assertRaisesRegex(v33.PreflightFailure, "infra-b1"):
            v33.assert_seed_304000_provenance({"v30": [self.P8_LINE]})

    def test_smoke_seed_virgin_positive_and_negative(self):
        v33.assert_seed_virgin({"v32": ['{"event": "x"}']}, 308000)
        with self.assertRaisesRegex(v33.PreflightFailure, "308000"):
            v33.assert_seed_virgin(
                {"infra-b1": ['{"event": "x", "seed": 308000}']}, 308000)

    def test_real_pinned_ledgers_scan(self):
        # 实件扫描(只读):304000 恰一处;308000 零出现(预登记 2026-07-19)
        lines = v33.w_pin_ledger_lines()
        v33.assert_seed_304000_provenance(lines)
        v33.assert_seed_virgin(lines, v33.SMOKE_SEED)

    def test_pool_guard(self):
        v33.assert_pool_guard(308000, "烟测")
        v33.assert_pool_guard(304000, "训练")
        for bad in (6998, 7000, 7031, 7999, 8000, 9000, 8999):
            with self.assertRaisesRegex(v33.PreflightFailure, "撞评测池"):
                v33.assert_pool_guard(bad, "x")


class CanaryCutoffTests(unittest.TestCase):
    """补评截止 = 该腿 s16 FIRING_START 前(驱动器内断言;P-canary)。"""

    def test_s16_fired_predicate(self):
        self.assertFalse(v33.s16_fired([], "v33-base"))
        events = [{"event": "FIRING_START", "tag": "v33-base-s16"}]
        self.assertTrue(v33.s16_fired(events, "v33-base"))
        self.assertFalse(v33.s16_fired(events, "v33-cur"))

    def test_canary_exam_refuses_fresh_after_cutoff(self):
        captured = []
        with mock.patch.object(v33, "log", captured.append):
            d, fresh = v33.canary_exam([], "/nonexistent/policy.npz",
                                       "v33-base-canary1-h", None,
                                       l1_fired=True)
        self.assertIsNone(d)
        self.assertFalse(fresh)
        self.assertEqual(captured[0]["event"], "OPERATIONAL-canary")
        self.assertIn("补评截止", captured[0]["why"])


class QuotaTests(unittest.TestCase):
    """额度计数:P2 评测发 2 / P3 腿点火 2(台账制,耗尽即停机)。"""

    def test_firing_count_and_leg_starts(self):
        events = [{"event": "FIRING_START", "tag": "t"},
                  {"event": "FIRING_START", "tag": "t"},
                  {"event": "FIRING_START", "tag": "other"},
                  {"event": "leg_start", "leg": "v33-base"},
                  {"event": "leg_start", "leg": "v33-cur"}]
        self.assertEqual(v33.firing_count(events, "t"), 2)
        self.assertEqual(v33.leg_starts(events, "v33-base"), 1)
        self.assertEqual(v33.leg_starts(events, "v33-full"), 0)

    def test_exam_case_quota_exhausted_raises(self):
        events = [{"event": "FIRING_START", "tag": "v33-unittest-quota"},
                  {"event": "FIRING_START", "tag": "v33-unittest-quota"}]
        with self.assertRaisesRegex(OperationalFailure, "额度耗尽"):
            v33.exam_case(events, "/nonexistent.npz", "v33-unittest-quota",
                          "7000-7015")

    def test_leg_quota_exhausted_raises_before_any_ignition(self):
        events = _staged_events() + [
            {"event": "leg_start", "leg": "v33-base"},
            {"event": "leg_start", "leg": "v33-base"}]
        with self.assertRaisesRegex(OperationalFailure, "点火额度耗尽"):
            v33.leg_stage(events, "v33-base")


class CurriculumTableTests(unittest.TestCase):
    """退火主表(圈 2 附裁)与腿终复核(实测 p 序列 ≡ 注册表)。"""

    def test_main_table_shape_and_endpoints(self):
        import train_ppo
        self.assertEqual(v33.MAIN_TABLE, MAIN_TABLE_LITERAL)
        self.assertEqual(v33.MAIN_TABLE, train_ppo._DRY_CURRICULUM_MAIN_TABLE)
        table = v33.dry_curriculum_table()
        self.assertEqual(len(table), 244)                  # 147+97 量子
        self.assertEqual(147 * 2048 + 97 * 2048, 499_712)  # 恰等腿长
        self.assertEqual(table[0], 1.0)
        self.assertEqual(table[146], 0.5)                  # 内插语义:末项恰达 0.5
        self.assertTrue(all(p == 0.5 for p in table[147:]))
        self.assertTrue(all(table[i] > table[i + 1] for i in range(146)))

    def test_verify_curriculum_prefix_identity_and_mismatch(self):
        table = (1.0, 0.9, 0.8, 0.7)
        with tempfile.TemporaryDirectory() as d:
            run_dir = pathlib.Path(d)
            p = run_dir / "dry_curriculum.jsonl"
            with open(p, "w") as f:
                for i in range(3):
                    f.write(json.dumps({"rollout_index": i, "p": table[i],
                                        "num_timesteps": 0}) + "\n")
            self.assertEqual(
                v33.verify_curriculum_prefix(run_dir, table, 3), [])
            # 整表移位/恒定值构造必被抓(对抗席构造对应件)
            with open(p, "w") as f:
                for i in range(3):
                    f.write(json.dumps({"rollout_index": i, "p": 0.5,
                                        "num_timesteps": 0}) + "\n")
            self.assertTrue(v33.verify_curriculum_prefix(run_dir, table, 3))
            # 短账(少 rollout)必被抓
            with open(p, "w") as f:
                f.write(json.dumps({"rollout_index": 0, "p": 1.0,
                                    "num_timesteps": 0}) + "\n")
            self.assertTrue(v33.verify_curriculum_prefix(run_dir, table, 3))

    def test_verify_missing_file_fails_loud(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(OperationalFailure, "dry_curriculum"):
                v33.verify_curriculum_prefix(pathlib.Path(d), (1.0,), 1)


class BiteqTests(unittest.TestCase):
    """REF_BITEQ 纯比对:全表逐种子逐字段 + agg 核心 + ret_mean 恒等。"""

    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(v33.W_PIN["v32-ref-launch"][0].read_text())

    def test_identical_doc_passes(self):
        self.assertEqual(v33.biteq_diffs(copy.deepcopy(self.doc), "launch"), {})

    def test_row_field_drift_detected(self):
        mutated = copy.deepcopy(self.doc)
        mutated["rows"][5]["ret"] += 0.1
        diffs = v33.biteq_diffs(mutated, "launch")
        self.assertEqual(diffs["row_diff_seeds"],
                         [sorted(r["seed"] for r in self.doc["rows"])[5]])

    def test_agg_mean_drift_detected(self):
        mutated = copy.deepcopy(self.doc)
        mutated["agg"]["ret_mean"] = 113.1
        diffs = v33.biteq_diffs(mutated, "launch")
        self.assertIn("ret_mean", str(diffs["agg_diff"]))


class SmokeCmdTests(unittest.TestCase):
    """G0-2a/2b 烟测命令形(种子 308000 预登记;bare = v32-sov 逐字含 --skip-dry)。"""

    def _val(self, cmd, flag):
        return cmd[cmd.index(flag) + 1]

    def test_bare_arm_is_v32_sov_verbatim_with_skip_dry(self):
        cmd = v33._smoke_cmd("bare")
        self.assertIn("--skip-dry", cmd)
        self.assertEqual(self._val(cmd, "--seed"), "308000")
        self.assertEqual(self._val(cmd, "--total-steps"), "102400")
        self.assertEqual(self._val(cmd, "--calib-probes"), "3747984,3947984")
        self.assertNotIn("--dry-curriculum-schedule", cmd)
        self.assertNotIn("--ckpt-every-steps", cmd)      # B1 旋钮不携(裸臂)
        # rev4 增列两枚亦不携:裸臂 = v32-sov 逐字不许动(十二附二①)
        self.assertNotIn("--distill-ce-probe-every", cmd)
        self.assertNotIn("--drywin-metrics-every", cmd)
        self.assertNotIn("--no-drink-sovereignty", cmd)

    def test_knobs_arm_pins_p1_and_inactive_bc_aux(self):
        cmd = v33._smoke_cmd("knobs")
        self.assertNotIn("--skip-dry", cmd)              # 两旗互斥
        self.assertEqual(self._val(cmd, "--dry-curriculum-schedule"),
                         "hold:1.0:50")                  # 调度钉 p≡1.0
        self.assertEqual(self._val(cmd, "--bc-aux-lambda"), "0.0")  # λ_bc=0
        self.assertEqual(self._val(cmd, "--distill-ce-probe-every"), "49152")
        self.assertEqual(self._val(cmd, "--drywin-metrics-every"), "49152")
        self.assertEqual(self._val(cmd, "--calib-probes"), "3547136,3596288")
        self.assertEqual(self._val(cmd, "--seed"), "308000")

    def test_func_runs_use_main_table_prefix_semantics(self):
        funcp = v33._smoke_cmd("func-p")
        self.assertEqual(self._val(funcp, "--dry-curriculum-schedule"),
                         MAIN_TABLE_LITERAL)
        self.assertNotIn("--bc-aux-lambda", funcp)
        funcaux = v33._smoke_cmd("func-aux")
        self.assertEqual(self._val(funcaux, "--dry-curriculum-schedule"),
                         MAIN_TABLE_LITERAL)
        self.assertEqual(self._val(funcaux, "--bc-aux-lambda"), "0.015625")
        self.assertTrue(self._val(funcaux, "--bc-aux-demos").endswith(
            "runs/bc-worker-v2/demos.npz"))
        for cmd in (funcp, funcaux):
            self.assertEqual(self._val(cmd, "--seed"), "308000")
            # E5①② 旋钮随共用形在位(rev4;各恰一现,无重复旗)
            for flag in ("--distill-ce-probe-every", "--drywin-metrics-every"):
                self.assertEqual(cmd.count(flag), 1, flag)
                self.assertEqual(self._val(cmd, flag), "49152", flag)

    def test_knobs_and_leg_cmds_carry_e5_knobs_exactly_once(self):
        # 修⑤a 回归:共用形注入后无重复旗(命令形逐字纪律)
        for cmd in (v33._smoke_cmd("knobs"), v33.leg_cmd("v33-base"),
                    v33.leg_cmd("v33-cur"), v33.leg_cmd("v33-full")):
            for flag in ("--distill-ce-probe-every", "--drywin-metrics-every"):
                self.assertEqual(cmd.count(flag), 1, flag)


class WPinFrozenConstantTests(unittest.TestCase):
    """W-PIN 冻结常量对实件抽验(只读;失配即本测先于预检报警)。"""

    def test_all_pinned_files_match_frozen_sha(self):
        for name, (path, expected) in {**v33.W_PIN,
                                       **v33.W_PIN_LEDGERS}.items():
            self.assertTrue(path.is_file(), name)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"W-PIN 失配:{name}")

    def test_event_line_pins_match(self):
        lines = v33.w_pin_ledger_lines()
        for name, (lkey, lineno, event_name, line_id, line_sha) in \
                v33.W_PIN_EVENT_LINES.items():
            raw = lines[lkey][lineno - 1]
            self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
                             line_sha, name)
            e = json.loads(raw)
            self.assertEqual(e.get("event"), event_name, name)
            if line_id is not None:
                self.assertEqual(e.get("line_id"), line_id, name)

    def test_g1_line_constants_match_ledger_values(self):
        lines = v33.w_pin_ledger_lines()["recal-g1"]
        by_id = {json.loads(lines[i - 1])["line_id"]: json.loads(lines[i - 1])
                 for i in range(10, 15)}
        self.assertEqual(round(float(by_id["RG1.2a-mean-worker-leg"]["value"]),
                               2), v33.G1_PAIRED_DIFF)
        self.assertEqual(int(by_id["RG1.1-width"]["value"]),
                         v33.G1_PAIRED_WINS)
        self.assertEqual(int(by_id["RG1.3-death"]["value"]), v33.G1_DIED_MAX)
        self.assertIn("85/92", str(by_id["RG1.4-floor"]["value"]))

    def test_bc_v1_demos_frozen_constant_single_source(self):
        import train_ppo
        self.assertEqual(v33.BC_V1_DEMOS_SHA, train_ppo._BC_V1_DEMOS_SHA256)


class VerdictFunctionTests(unittest.TestCase):
    """判据纯函数:资格(D4-1)/胜者(D4-5)/课⑤(D4-2)/④乙档位(D4-3)。"""

    def test_qual_of_grid(self):
        self.assertTrue(v33.qual_of(_full32_doc())["qual_ok"])
        self.assertFalse(v33.qual_of(_full32_doc(died=9))["qual_ok"])
        q_cap = v33.qual_of(_full32_doc(cap=0.06))
        self.assertFalse(q_cap["qual_ok"])
        self.assertIn("cap_drift_note", q_cap)     # 仅因 cap 失格之漂移候选注记
        q_void = v33.qual_of(_full32_doc(override=0.09))
        self.assertTrue(q_void["void"])
        self.assertFalse(q_void["qual_ok"])
        # died≤8(G1 L2-0 (a) 同值替换):died=8 本身不失格
        self.assertTrue(v33.qual_of(_full32_doc(died=8))["qual_ok"])

    def test_decide_winner_band_and_prescription_preference(self):
        means = {"v33-base": 100.0, "v33-cur": 100.03, "v33-full": 100.02}
        died = {"v33-base": 2, "v33-cur": 2, "v33-full": 2}
        quals = {n: {"qual_ok": True} for n in means}
        win = v33.decide_winner(means, died, quals)
        self.assertEqual(win["winner"], "v33-full")   # 差≤0.05 → died 平 → 处方腿
        died2 = {"v33-base": 1, "v33-cur": 3, "v33-full": 3}
        win2 = v33.decide_winner(means, died2, quals)
        self.assertEqual(win2["winner"], "v33-base")  # died 少者
        self.assertFalse(win2["substituted"])

    def test_decide_winner_substitution_and_empty_pool(self):
        means = {"v33-base": 100.0, "v33-cur": 99.0, "v33-full": 120.0}
        died = {"v33-base": 2, "v33-cur": 2, "v33-full": 9}
        quals = {"v33-base": {"qual_ok": True}, "v33-cur": {"qual_ok": True},
                 "v33-full": {"qual_ok": False}}
        win = v33.decide_winner(means, died, quals)
        self.assertEqual(win["winner"], "v33-base")
        self.assertTrue(win["substituted"])           # 递补入册
        self.assertEqual(win["prelim"], "v33-full")
        none = v33.decide_winner(means, died,
                                 {n: {"qual_ok": False} for n in means})
        self.assertIsNone(none["winner"])

    def test_course5_ruling_branches(self):
        self.assertEqual(v33.course5_ruling(5, 3)["branch"], "success")
        self.assertEqual(v33.course5_ruling(5, 4)["branch"], "noise")
        self.assertEqual(v33.course5_ruling(5, 5)["branch"], "no_improvement")
        low = v33.course5_ruling(4, 2)
        self.assertEqual(low["branch"], "success")    # 锚=4 → 成功线 ≤2
        self.assertIn("note_low_anchor", low)
        floor = v33.course5_ruling(3, 0)
        self.assertEqual(floor["branch"], "floor")    # 锚≤3 → 地板效应不可判
        self.assertIn("不可判", floor["verdict"])

    def test_a12_tier_ruling_grid(self):
        t1 = v33.a12_tier_ruling(0.5, 3, 0.0, 3, 0.0)
        self.assertEqual(t1["tier"], 1)
        self.assertTrue(t1["main_line"])
        self.assertNotIn("circle12_exit", t1)
        t2 = v33.a12_tier_ruling(0.0, 4, 0.0, 3, 0.0)   # died ≤ ctrl+1 一命噪声带
        self.assertEqual(t2["tier"], 2)
        self.assertIn("circle12_exit", t2)              # 圈 12 出口语法强制
        low = v33.a12_tier_ruling(0.05, 3, 0.0, 3, 0.0)
        self.assertEqual(low["tier"], 2)                # 低用量依档序落档2
        self.assertIn("low_use_note", low)
        t3a = v33.a12_tier_ruling(0.5, 6, -10.0, 3, 0.0)
        self.assertEqual(t3a["tier"], 3)
        self.assertIn("(a)", t3a["tier3_subcase"])
        t3b = v33.a12_tier_ruling(0.0, 6, -10.0, 3, 0.0)
        self.assertEqual(t3b["tier"], 3)
        self.assertIn("(b)", t3b["tier3_subcase"])

    def test_a12_control_crossline_branch(self):
        crossed = v33.a12_tier_ruling(0.5, 3, 0.0, 3, 0.2)
        self.assertIn("control_crossline", crossed)     # L-cur ≥0.1 归因降级
        clean = v33.a12_tier_ruling(0.5, 3, 0.0, 3, 0.0)
        self.assertNotIn("control_crossline", clean)


class N12DemosChainTests(unittest.TestCase):
    """修①b/①c(rev4 十二附二④):vacuous 兜底废止改硬失败 + L-full 点火前
    demos 实测字节 ≡ N12_GATE 落定值。"""

    def test_gate_sha_missing_key_hard_fails(self):
        with self.assertRaisesRegex(OperationalFailure, "缺 demos_sha256"):
            v33.n12_gate_demos_sha({"event": "N12_GATE", "gate": "PASS"})
        with self.assertRaisesRegex(OperationalFailure, "缺 demos_sha256"):
            v33.n12_gate_demos_sha({"demos_sha256": "short"})
        with self.assertRaisesRegex(OperationalFailure, "缺 demos_sha256"):
            v33.n12_gate_demos_sha({"demos_sha256": None})

    def test_gate_sha_present_returns_verbatim(self):
        sha = "a" * 64
        self.assertEqual(v33.n12_gate_demos_sha({"demos_sha256": sha}), sha)

    def test_lfull_chain_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            demos = pathlib.Path(d) / "demos.npz"
            demos.write_bytes(b"demo-bytes")
            sha = hashlib.sha256(b"demo-bytes").hexdigest()
            with mock.patch.object(v33, "BC_V2_DEMOS", demos):
                v33.assert_lfull_demos_chain(          # 正例:恒等即静默通过
                    [{"event": "N12_GATE", "gate": "PASS",
                      "demos_sha256": sha}])
                with self.assertRaisesRegex(OperationalFailure, "断裂"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "PASS",
                          "demos_sha256": "b" * 64}])
                with self.assertRaisesRegex(OperationalFailure,
                                            "缺 demos_sha256"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "PASS"}])
                with self.assertRaisesRegex(OperationalFailure, "不在册"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "FAIL",
                          "demos_sha256": sha}])
                # 多发 N12_GATE:取末次 PASS 为落定值
                v33.assert_lfull_demos_chain(
                    [{"event": "N12_GATE", "gate": "PASS",
                      "demos_sha256": "c" * 64},
                     {"event": "N12_GATE", "gate": "PASS",
                      "demos_sha256": sha}])
            with mock.patch.object(v33, "BC_V2_DEMOS",
                                   pathlib.Path(d) / "gone.npz"):
                with self.assertRaisesRegex(OperationalFailure, "demos 缺失"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "PASS",
                          "demos_sha256": sha}])

    def test_wiring_lfull_only_and_before_ignition(self):
        src = inspect.getsource(v33.leg_stage)
        self.assertIn('if leg == "v33-full":', src)
        self.assertIn("assert_lfull_demos_chain(events)", src)
        self.assertLess(src.index("assert_lfull_demos_chain"),
                        src.index('"event": "leg_start"'))
        # 修①b:bc2_stage 幂等分支同用硬取键(vacuous .get 兜底退场)
        src_bc2 = inspect.getsource(v33.bc2_stage)
        self.assertIn("n12_gate_demos_sha(passed[-1])", src_bc2)
        self.assertNotIn('.get("demos_sha256",', src_bc2)


class SentinelEvidenceTests(unittest.TestCase):
    """修②(rev4 十二附二⑤):G0-2a 仪表实燃四件全查之 sentinel/dry-anchor
    双行型分计(sentinel.jsonl 同文件双行型)。"""

    def test_dual_type_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "sentinel.jsonl"
            rows = [json.dumps({"sentinel": "v23", "step": 1, "dry": 0}),
                    json.dumps({"sentinel": "dry-anchor", "step": 1,
                                "mismatch": 0.7515, "n": 2000}),
                    json.dumps({"sentinel": "v23", "step": 2, "final": True})]
            p.write_text("\n".join(rows) + "\n")
            self.assertEqual(v33.sentinel_line_counts(p),
                             {"sentinel": 2, "dry_anchor": 1})

    def test_missing_file_and_single_type_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "sentinel.jsonl"
            self.assertEqual(v33.sentinel_line_counts(p),
                             {"sentinel": 0, "dry_anchor": 0})   # 缺档记 0
            p.write_text(json.dumps({"sentinel": "v23", "step": 1}) + "\n")
            counts = v33.sentinel_line_counts(p)
            self.assertEqual(counts["dry_anchor"], 0)   # 干层锚没跑必 0
            self.assertEqual(counts["sentinel"], 1)

    def test_g0_2a_conjunction_carries_both_limbs(self):
        src = inspect.getsource(v33.g0_nullintrusion_stage)
        self.assertIn('evidence["knobs_sentinel_lines"] >= 1', src)
        self.assertIn('evidence["knobs_dry_anchor_lines"] >= 1', src)


class TableFullPrecisionTests(unittest.TestCase):
    """修③(rev4 十二附二⑥):DRY_CURRICULUM_TABLE 全精度 float 落账,
    与 verify_curriculum_prefix 复核真源同一;round(p,10) 口径废止。"""

    def test_event_table_is_full_precision_and_json_lossless(self):
        captured = []
        with mock.patch.object(v33, "log", captured.append):
            v33.dry_curriculum_table_stage([])
        ev = captured[0]
        table = v33.dry_curriculum_table()
        self.assertEqual(ev["event"], "DRY_CURRICULUM_TABLE")
        self.assertEqual(ev["table"], [float(p) for p in table])
        self.assertEqual(ev["table"][1], table[1])            # 全精度位同一
        self.assertNotEqual(round(table[1], 10), table[1])    # 反例非空:
        # round(,10) 于 linear 段第 1 项即失位,旧口径与复核真源不同一
        self.assertEqual(json.loads(json.dumps(ev["table"])), list(table))

    def test_idempotent_when_already_logged(self):
        captured = []
        with mock.patch.object(v33, "log", captured.append):
            v33.dry_curriculum_table_stage([{"event": "DRY_CURRICULUM_TABLE"}])
        self.assertEqual(captured, [])


class SeedSeqEquivalentTests(unittest.TestCase):
    """修④(rev4 十二附二③):G0-2b episode 种子序列恒等之可见面最强等价物
    ——两跑 progress.jsonl (ep, reward, len) 公共前缀恒等(首 rollout 覆盖
    下界 512;progress.jsonl 无 episode 种子字段,呈报见交接单)。"""

    def test_identical_sequences_full_prefix(self):
        a = [{"ep": i + 1, "reward": 1.0 * i, "len": 300} for i in range(4)]
        out = v33.progress_common_prefix(a, [dict(x) for x in a])
        self.assertEqual(out["prefix_lines"], 4)
        self.assertEqual(out["prefix_len_steps"], 1200)
        self.assertIsNone(out["first_divergence_index"])
        self.assertGreaterEqual(out["prefix_len_steps"],
                                v33.SEED_EQUIV_MIN_PREFIX_STEPS)

    def test_first_line_divergence_caught(self):
        # 初始 episode 种子流被污染之指纹:首行即分歧 → 前缀 0,必不过下界
        a = [{"ep": 1, "reward": 1.0, "len": 600}]
        b = [{"ep": 1, "reward": 2.0, "len": 600}]
        out = v33.progress_common_prefix(a, b)
        self.assertEqual(out["prefix_lines"], 0)
        self.assertEqual(out["first_divergence_index"], 0)
        self.assertLess(out["prefix_len_steps"],
                        v33.SEED_EQUIV_MIN_PREFIX_STEPS)

    def test_mid_divergence_index_and_threshold(self):
        a = [{"ep": 1, "reward": 1.0, "len": 400},
             {"ep": 2, "reward": 2.0, "len": 200},
             {"ep": 3, "reward": 3.0, "len": 100}]
        b = [dict(a[0]), dict(a[1]), {"ep": 3, "reward": 9.0, "len": 100}]
        out = v33.progress_common_prefix(a, b)
        self.assertEqual(out["prefix_lines"], 2)
        self.assertEqual(out["first_divergence_index"], 2)
        self.assertEqual(out["prefix_len_steps"], 600)
        self.assertGreaterEqual(out["prefix_len_steps"],
                                v33.SEED_EQUIV_MIN_PREFIX_STEPS)
        self.assertEqual(out["lines"], [3, 3])

    def test_g0_2b_wires_seed_equiv_limb_into_verdict(self):
        src = inspect.getsource(v33.g0_funcsmoke_stage)
        self.assertIn("progress_common_prefix", src)
        self.assertIn("and seed_seq_ok", src)
        self.assertIn("episode_seed_seq_identity", src)
        self.assertEqual(v33.SEED_EQUIV_MIN_PREFIX_STEPS, 512)  # n-steps 下界


class InstrumentDigestTests(unittest.TestCase):
    """修⑤b(rev4 十二附二①/D8):DRYWIN_METRICS 转录载荷件——文件 sha256 +
    行数 + final 段聚合;缺档/空档 fail-loud;leg_stage 腿终落笔。"""

    def test_digest_prefers_final_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "drywin_metrics.jsonl"
            rows = [{"metrics": "drywin", "step": 1},
                    {"metrics": "drywin", "step": 2, "final": True}]
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            digest = v33.instrument_jsonl_digest(p)
            self.assertEqual(digest["lines"], 2)
            self.assertEqual(digest["final"]["step"], 2)
            self.assertEqual(digest["sha256"],
                             hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertNotIn("fall_back_last_line", digest)

    def test_digest_falls_back_to_last_line_with_note(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "distill_ce_probe.jsonl"
            p.write_text(json.dumps({"probe": "distill-ce", "step": 7}) + "\n")
            digest = v33.instrument_jsonl_digest(p)
            self.assertEqual(digest["final"]["step"], 7)
            self.assertIs(digest["fall_back_last_line"], True)

    def test_digest_fails_loud_on_missing_or_empty(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(OperationalFailure, "仪表档缺失"):
                v33.instrument_jsonl_digest(pathlib.Path(d) / "gone.jsonl")
            empty = pathlib.Path(d) / "drywin_metrics.jsonl"
            empty.write_text("")
            with self.assertRaisesRegex(OperationalFailure, "仪表档为空"):
                v33.instrument_jsonl_digest(empty)

    def test_leg_stage_transcribes_drywin_metrics_event(self):
        src = inspect.getsource(v33.leg_stage)
        self.assertIn('"event": "DRYWIN_METRICS"', src)
        self.assertIn('RUNS / leg / "drywin_metrics.jsonl"', src)
        self.assertIn('RUNS / leg / "distill_ce_probe.jsonl"', src)


class StatsToolTests(unittest.TestCase):
    """MS 计算与配对统计块(R 线口径)。"""

    def test_ms_vectors_hand_computed(self):
        leg_h = {s: _row(s, ret=r) for s, r in
                 {1: 70.0, 2: 100.0, 3: 110.0, 4: 95.0}.items()}
        ref_h = {s: _row(s, ret=100.0) for s in (1, 2, 3, 4)}
        leg_m29 = {s: _row(s, ret=r) for s, r in
                   {1: 105.0, 2: 100.0, 3: 85.0, 4: 105.0}.items()}
        ref_m29 = {s: _row(s, ret=100.0) for s in (1, 2, 3, 4)}
        v = v33.ms_vectors(leg_h, leg_m29, ref_h, ref_m29)
        self.assertEqual(v["signed_dh_minus_dm"],
                         {1: -35.0, 2: 0.0, 3: 25.0, 4: -10.0})
        self.assertEqual(v["ms_max"], 35.0)
        self.assertEqual(v["over_line_seeds"], [1, 3])

    def test_paired_diff_stats_carries_mandatory_parallels(self):
        leg = {7000 + i: _row(7000 + i, ret=100.0 + i) for i in range(4)}
        leg[7017] = _row(7017, ret=0.0)
        ref = {s: _row(s, ret=100.0) for s in leg}
        st = v33.paired_diff_stats(leg, ref)
        for key in ("mean", "median", "sign", "deleveraged", "loo_7017",
                    "wins", "by_seed"):
            self.assertIn(key, st)
        self.assertEqual(st["loo_7017"]["dropped_seed"], 7017)
        self.assertEqual(st["deleveraged"]["dropped_seed"], 7017)


if __name__ == "__main__":
    unittest.main()
