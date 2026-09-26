"""Self-contained fast regression for the v33 content-case driver (counterpart of PREREG-v33-content-case;
no engine, training or evaluation is started; follows the style of tests/test_b1_infra.py and tests/test_content_case.py;
every ledger write path is captured with monkeypatch, and the real ledgers are never touched).

Covers (item by item from the implementation task list):
- verbatim assertions on the LEGS table (the full CLI of the three legs, item by item: the shared command form + the D3 per-leg additions);
- the exit-code table (D5, centralised as constants);
- the W-LAUNCH launch-order gate (without LAUNCH_ORDER it must exit 9; not a failure state);
- stage order against hindsight (a later stage refuses to run while an earlier stage is not done; N12 PASS semantic predicate);
- the 304000/308000 reconciliation assertion functions (positive and negative synthetic ledgers) + the evaluation-pool range guard;
- the canary make-up evaluation cutoff assertion (before that leg's s16 FIRING_START);
- budget counting (P2 evaluations: 2 / P3 leg launches: 2);
- also: the main annealing table and the end-of-leg re-check, REF_BITEQ pure comparison, the smoke command form, spot checks of the
    frozen W-PIN constants against the real files, pure criterion functions (eligibility / winner / course 5 / course 4b tier), the MS computation;
- counterparts of the rev4 item-12 annex-2 additions (fixes 1-5): hard key lookup of the settled N12_GATE value + the pre-launch
    L-full demos byte chain (4); separate counts of the two G0-2a sentinel/dry-anchor row types (5);
    full-precision booking of DRY_CURRICULUM_TABLE (6); the visible-surface equivalent of the G0-2b episode seed-sequence
    identity (3); the two E5 (1)(2) knobs in the shared command form + the DRYWIN_METRICS transcript (1).
"""

from __future__ import annotations

import copy
import fcntl
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
from eval_contract import EvalContractError, OperationalFailure  # noqa: E402

# The W-PIN checks read run archives and ledgers that are not published (the
# v32-ref and b1-ref8k evaluation archives, g1_results.json and the pinned run
# ledgers), so they can only run in a tree that has them.
_PINNED_PRESENT = all(
    path.is_file()
    for path, _ in {**v33.W_PIN, **v33.W_PIN_LEDGERS}.values())

MAIN_TABLE_LITERAL = "linear:1.0:0.5:147,hold:0.5:97"
CALIB_LITERAL = ("3547136,3596288,3645440,3694592,3743744,"
                 "3792896,3842048,3891200,3940352,3989504")


def _staged_events(upto: str | None = None) -> list[dict]:
    """Synthetic ledger: every event of STAGE_SEQUENCE in order (up to and including upto)."""
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
        {"event": "LAUNCH_ORDER", "order": "synthetic launch order (test fixture)"},
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
    """D3 LEGS table verbatim: the full CLI of the three legs (shared command form + per-leg additions)."""

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
                                   # the last two added by the rev4 D3 correction (item 12, annex 2.1)
                                   ("--distill-ce-probe-every", "49152"),
                                   ("--drywin-metrics-every", "49152"),
                                   ("--run-name", leg)):
                self.assertEqual(self._val(cmd, flag), expected, (leg, flag))
            for flag in ("--worker", "--allow-legacy-resume"):
                self.assertIn(flag, cmd, (leg, flag))
            self.assertNotIn("--calib-record-only", cmd,
                             "the G-CAL of a production leg must be able to rule; record-only is not allowed")
            # sovereignty on is the default; nothing is issued with --board
            self.assertNotIn("--no-drink-sovereignty", cmd)
            self.assertNotIn("--board", cmd)
            self.assertTrue(self._val(cmd, "--resume-from").endswith(
                "v28-worker-leg1/model_final.zip"))
            self.assertTrue(self._val(cmd, "--teacher-sd").endswith(
                "bc-worker/policy_sd.pt"))          # BC_SD(v1)
            self.assertTrue(self._val(cmd, "--teacher-override").endswith(
                "v32/king_anchor_sd.pt"))           # the KING_SD anchor follows the king unchanged

    def test_l_base_extras(self):
        cmd = v33.leg_cmd("v33-base")
        self.assertIn("--skip-dry", cmd)            # the shared form does not produce p==1.0, so it must be carried explicitly
        self.assertNotIn("--dry-curriculum-schedule", cmd)
        self.assertNotIn("--bc-aux-lambda", cmd)
        self.assertNotIn("--bc-aux-demos", cmd)

    def test_l_cur_extras(self):
        cmd = v33.leg_cmd("v33-cur")
        self.assertNotIn("--skip-dry", cmd)         # the two flags are mutually exclusive (E1)
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
            "runs/bc-worker-v2/demos.npz"))         # D3 literal path
        self.assertIn("--bc-aux-liveness-preflight", cmd)

    def test_leg_accounting_constants(self):
        self.assertEqual(v33.NT_TARGET, 3_997_696)
        self.assertEqual(v33.NT_TARGET, v33.KING_STEPS + v33.LEG_STEPS)
        self.assertEqual(v33.SEED, 304_000)
        self.assertEqual(v33.LEG_NAMES, ("v33-base", "v33-cur", "v33-full"))
        self.assertEqual(v33.CURRICULUM_LEGS, ("v33-cur", "v33-full"))


class ExitCodeTableTests(unittest.TestCase):
    """D5 exit-code table (centralised as constants, verbatim)."""

    def test_exit_code_table_closed_enumeration(self):
        self.assertEqual(set(v33.EXIT_CODES), {0, 2, 3, 4, 5, 6, 7, 8, 9})
        self.assertEqual(v33.EXIT_CODES[0], "case closed/idempotent")
        self.assertIn("PREFLIGHT_FAIL", v33.EXIT_CODES[3])
        self.assertIn("lock conflict", v33.EXIT_CODES[4])
        self.assertIn("pre-launch drift", v33.EXIT_CODES[5])
        self.assertIn("drift during the case", v33.EXIT_CODES[6])
        self.assertEqual(v33.EXIT_CODES[7], "CASE_HALT_G0")
        self.assertEqual(v33.EXIT_CODES[8], "REF_DIVERGENCE")
        self.assertIn("AWAITING_LAUNCH", v33.EXIT_CODES[9])
        self.assertIn("not a failure", v33.EXIT_CODES[9])


class LaunchGateTests(unittest.TestCase):
    """W-LAUNCH launch-order gate: without LAUNCH_ORDER it must exit 9 (not a failure state)."""

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
    """Stage order against hindsight (D2 strictly serial): a later stage refuses to run while an earlier stage is not done."""

    def test_full_prefix_passes(self):
        v33.assert_stage_prereqs(_staged_events(), "LEGS")

    def test_missing_middle_stage_rejected(self):
        events = [e for e in _staged_events()
                  if e["event"] != "G0_GHOST"]
        with self.assertRaisesRegex(v33.PreflightFailure, "G0_GHOST"):
            v33.assert_stage_prereqs(events, "FREEZE_SHA")

    def test_later_stage_refused_when_nothing_done(self):
        with self.assertRaisesRegex(v33.PreflightFailure, "no-hindsight"):
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
    """304000 reconciliation (exactly and only the infra-b1 P8 leg_start) / 308000 fresh / pool guard."""

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
        # "exactly" requires it on record: a missing P8 leg_start fails too
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

    @unittest.skipUnless(_PINNED_PRESENT, 'pinned run files are not published')
    def test_real_pinned_ledgers_scan(self):
        # Scan of the real files (read-only): 304000 exactly once; 308000 never (pre-registered 2026-07-19)
        lines = v33.w_pin_ledger_lines()
        v33.assert_seed_304000_provenance(lines)
        v33.assert_seed_virgin(lines, v33.SMOKE_SEED)

    def test_pool_guard(self):
        v33.assert_pool_guard(308000, "smoke")
        v33.assert_pool_guard(304000, "training")
        for bad in (6998, 7000, 7031, 7999, 8000, 9000, 8999):
            with self.assertRaisesRegex(v33.PreflightFailure, "collides with an evaluation pool"):
                v33.assert_pool_guard(bad, "x")


class CanaryCutoffTests(unittest.TestCase):
    """Make-up evaluation cutoff = before that leg's s16 FIRING_START (assertion inside the driver; P-canary)."""

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
        self.assertIn("make-up deadline", captured[0]["why"])


class QuotaTests(unittest.TestCase):
    """Budget counting: P2 evaluations 2 / P3 leg launches 2 (ledger-based; stops when exhausted)."""

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
        with self.assertRaisesRegex(OperationalFailure, "budget exhausted"):
            v33.exam_case(events, "/nonexistent.npz", "v33-unittest-quota",
                          "7000-7015")

    def test_leg_quota_exhausted_raises_before_any_ignition(self):
        events = _staged_events() + [
            {"event": "leg_start", "leg": "v33-base"},
            {"event": "leg_start", "leg": "v33-base"}]
        with self.assertRaisesRegex(OperationalFailure, "launch budget exhausted"):
            v33.leg_stage(events, "v33-base")


class CurriculumTableTests(unittest.TestCase):
    """Main annealing table (design decision 2 and its addendum) and the end-of-leg re-check (measured p sequence == the registered table)."""

    def test_main_table_shape_and_endpoints(self):
        import train_ppo
        self.assertEqual(v33.MAIN_TABLE, MAIN_TABLE_LITERAL)
        self.assertEqual(v33.MAIN_TABLE, train_ppo._DRY_CURRICULUM_MAIN_TABLE)
        table = v33.dry_curriculum_table()
        self.assertEqual(len(table), 244)                  # 147+97 quanta
        self.assertEqual(147 * 2048 + 97 * 2048, 499_712)  # exactly the leg length
        self.assertEqual(table[0], 1.0)
        self.assertEqual(table[146], 0.5)                  # interpolation semantics: the last entry reaches exactly 0.5
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
            # a shifted table or a constant-value construction must be caught (counterpart of the adversarial reviewer's construction)
            with open(p, "w") as f:
                for i in range(3):
                    f.write(json.dumps({"rollout_index": i, "p": 0.5,
                                        "num_timesteps": 0}) + "\n")
            self.assertTrue(v33.verify_curriculum_prefix(run_dir, table, 3))
            # a short ledger (missing rollouts) must be caught
            with open(p, "w") as f:
                f.write(json.dumps({"rollout_index": 0, "p": 1.0,
                                    "num_timesteps": 0}) + "\n")
            self.assertTrue(v33.verify_curriculum_prefix(run_dir, table, 3))

    def test_verify_missing_file_fails_loud(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(OperationalFailure, "dry_curriculum"):
                v33.verify_curriculum_prefix(pathlib.Path(d), (1.0,), 1)


@unittest.skipUnless(_PINNED_PRESENT, 'pinned run files are not published')
class BiteqTests(unittest.TestCase):
    """REF_BITEQ pure comparison: the whole table seed by seed and field by field + the agg core + ret_mean identity."""

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
    """G0-2a/2b smoke command form (seed 308000 pre-registered; bare = v32-sov verbatim, including --skip-dry)."""

    def _val(self, cmd, flag):
        return cmd[cmd.index(flag) + 1]

    def test_bare_arm_is_v32_sov_verbatim_with_skip_dry(self):
        cmd = v33._smoke_cmd("bare")
        self.assertIn("--skip-dry", cmd)
        self.assertEqual(self._val(cmd, "--seed"), "308000")
        self.assertEqual(self._val(cmd, "--total-steps"), "102400")
        self.assertEqual(self._val(cmd, "--calib-probes"), "3747984,3947984")
        self.assertNotIn("--dry-curriculum-schedule", cmd)
        self.assertNotIn("--ckpt-every-steps", cmd)      # B1 knobs not carried (bare arm)
        # the two rev4 additions are not carried either: bare arm = v32-sov verbatim, unchanged (item 12, annex 2.1)
        self.assertNotIn("--distill-ce-probe-every", cmd)
        self.assertNotIn("--drywin-metrics-every", cmd)
        self.assertNotIn("--no-drink-sovereignty", cmd)
        self.assertIn("--calib-record-only", cmd)   # the old v32 bare control, verbatim

    def test_knobs_arm_pins_p1_and_inactive_bc_aux(self):
        cmd = v33._smoke_cmd("knobs")
        self.assertNotIn("--skip-dry", cmd)              # the two flags are mutually exclusive
        self.assertEqual(self._val(cmd, "--dry-curriculum-schedule"),
                         "hold:1.0:50")                  # the schedule pins p==1.0
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
            self.assertNotIn("--calib-record-only", cmd)
            self.assertEqual(self._val(cmd, "--seed"), "308000")
            # the E5 (1)(2) knobs are present with the shared form (rev4; each exactly once, no duplicate flags)
            for flag in ("--distill-ce-probe-every", "--drywin-metrics-every"):
                self.assertEqual(cmd.count(flag), 1, flag)
                self.assertEqual(self._val(cmd, flag), "49152", flag)

    def test_knobs_and_leg_cmds_carry_e5_knobs_exactly_once(self):
        # fix 5a regression: no duplicate flags after injection into the shared form (verbatim command-form discipline)
        for cmd in (v33._smoke_cmd("knobs"), v33.leg_cmd("v33-base"),
                    v33.leg_cmd("v33-cur"), v33.leg_cmd("v33-full")):
            for flag in ("--distill-ce-probe-every", "--drywin-metrics-every"):
                self.assertEqual(cmd.count(flag), 1, flag)


@unittest.skipUnless(_PINNED_PRESENT, 'pinned run files are not published')
class WPinFrozenConstantTests(unittest.TestCase):
    """Spot check of the frozen W-PIN constants against the real files (read-only; a mismatch alarms here before the preflight)."""

    def test_all_pinned_files_match_frozen_sha(self):
        for name, (path, expected) in {**v33.W_PIN,
                                       **v33.W_PIN_LEDGERS}.items():
            self.assertTrue(path.is_file(), name)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"W-PIN mismatch: {name}")

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
    """Pure criterion functions: eligibility (D4-1) / winner (D4-5) / course 5 (D4-2) / course 4b tier (D4-3)."""

    def test_qual_of_grid(self):
        self.assertTrue(v33.qual_of(_full32_doc())["qual_ok"])
        self.assertFalse(v33.qual_of(_full32_doc(died=9))["qual_ok"])
        q_cap = v33.qual_of(_full32_doc(cap=0.06))
        self.assertFalse(q_cap["qual_ok"])
        self.assertIn("cap_drift_note", q_cap)     # note on a drift candidate disqualified only by the cap
        q_void = v33.qual_of(_full32_doc(override=0.09))
        self.assertTrue(q_void["void"])
        self.assertFalse(q_void["qual_ok"])
        # died<=8 (G1 L2-0 (a) same-value substitution): died=8 itself does not disqualify
        self.assertTrue(v33.qual_of(_full32_doc(died=8))["qual_ok"])

    def test_decide_winner_band_and_prescription_preference(self):
        means = {"v33-base": 100.0, "v33-cur": 100.03, "v33-full": 100.02}
        died = {"v33-base": 2, "v33-cur": 2, "v33-full": 2}
        quals = {n: {"qual_ok": True} for n in means}
        win = v33.decide_winner(means, died, quals)
        self.assertEqual(win["winner"], "v33-full")   # difference <= 0.05 -> died tied -> the prescribed leg
        died2 = {"v33-base": 1, "v33-cur": 3, "v33-full": 3}
        win2 = v33.decide_winner(means, died2, quals)
        self.assertEqual(win2["winner"], "v33-base")  # fewer deaths wins
        self.assertFalse(win2["substituted"])

    def test_decide_winner_substitution_and_empty_pool(self):
        means = {"v33-base": 100.0, "v33-cur": 99.0, "v33-full": 120.0}
        died = {"v33-base": 2, "v33-cur": 2, "v33-full": 9}
        quals = {"v33-base": {"qual_ok": True}, "v33-cur": {"qual_ok": True},
                 "v33-full": {"qual_ok": False}}
        win = v33.decide_winner(means, died, quals)
        self.assertEqual(win["winner"], "v33-base")
        self.assertTrue(win["substituted"])           # the substitute is entered
        self.assertEqual(win["prelim"], "v33-full")
        none = v33.decide_winner(means, died,
                                 {n: {"qual_ok": False} for n in means})
        self.assertIsNone(none["winner"])

    def test_course5_ruling_branches(self):
        self.assertEqual(v33.course5_ruling(5, 3)["branch"], "success")
        self.assertEqual(v33.course5_ruling(5, 4)["branch"], "noise")
        self.assertEqual(v33.course5_ruling(5, 5)["branch"], "no_improvement")
        low = v33.course5_ruling(4, 2)
        self.assertEqual(low["branch"], "success")    # anchor=4 -> success line <= 2
        self.assertIn("note_low_anchor", low)
        floor = v33.course5_ruling(3, 0)
        self.assertEqual(floor["branch"], "floor")    # anchor<=3 -> floor effect, cannot be decided
        self.assertIn("undecidable", floor["verdict"])

    def test_a12_tier_ruling_grid(self):
        t1 = v33.a12_tier_ruling(0.5, 3, 0.0, 3, 0.0)
        self.assertEqual(t1["tier"], 1)
        self.assertTrue(t1["main_line"])
        self.assertNotIn("circle12_exit", t1)
        t2 = v33.a12_tier_ruling(0.0, 4, 0.0, 3, 0.0)   # died <= ctrl+1: the one-life noise band
        self.assertEqual(t2["tier"], 2)
        self.assertIn("circle12_exit", t2)              # exit syntax of design decision 12 enforced
        low = v33.a12_tier_ruling(0.05, 3, 0.0, 3, 0.0)
        self.assertEqual(low["tier"], 2)                # low usage falls to tier 2 by tier order
        self.assertIn("low_use_note", low)
        t3a = v33.a12_tier_ruling(0.5, 6, -10.0, 3, 0.0)
        self.assertEqual(t3a["tier"], 3)
        self.assertIn("(a)", t3a["tier3_subcase"])
        t3b = v33.a12_tier_ruling(0.0, 6, -10.0, 3, 0.0)
        self.assertEqual(t3b["tier"], 3)
        self.assertIn("(b)", t3b["tier3_subcase"])

    def test_a12_control_crossline_branch(self):
        crossed = v33.a12_tier_ruling(0.5, 3, 0.0, 3, 0.2)
        self.assertIn("control_crossline", crossed)     # L-cur >= 0.1: attribution downgraded
        clean = v33.a12_tier_ruling(0.5, 3, 0.0, 3, 0.0)
        self.assertNotIn("control_crossline", clean)


class N12DemosChainTests(unittest.TestCase):
    """Fixes 1b/1c (rev4 item 12, annex 2.4): the vacuous fallback is abolished in favour of a hard failure + the pre-launch L-full
    measured demos bytes == the settled N12_GATE value."""

    def test_gate_sha_missing_key_hard_fails(self):
        with self.assertRaisesRegex(OperationalFailure, "missing demos_sha256"):
            v33.n12_gate_demos_sha({"event": "N12_GATE", "gate": "PASS"})
        with self.assertRaisesRegex(OperationalFailure, "missing demos_sha256"):
            v33.n12_gate_demos_sha({"demos_sha256": "short"})
        with self.assertRaisesRegex(OperationalFailure, "missing demos_sha256"):
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
                v33.assert_lfull_demos_chain(          # positive case: identity passes silently
                    [{"event": "N12_GATE", "gate": "PASS",
                      "demos_sha256": sha}])
                with self.assertRaisesRegex(OperationalFailure, "identity chain broken"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "PASS",
                          "demos_sha256": "b" * 64}])
                with self.assertRaisesRegex(OperationalFailure,
                                            "missing demos_sha256"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "PASS"}])
                with self.assertRaisesRegex(OperationalFailure, "not recorded"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "FAIL",
                          "demos_sha256": sha}])
                # several N12_GATE events: the last PASS is the settled value
                v33.assert_lfull_demos_chain(
                    [{"event": "N12_GATE", "gate": "PASS",
                      "demos_sha256": "c" * 64},
                     {"event": "N12_GATE", "gate": "PASS",
                      "demos_sha256": sha}])
            with mock.patch.object(v33, "BC_V2_DEMOS",
                                   pathlib.Path(d) / "gone.npz"):
                with self.assertRaisesRegex(OperationalFailure, "demos missing"):
                    v33.assert_lfull_demos_chain(
                        [{"event": "N12_GATE", "gate": "PASS",
                          "demos_sha256": sha}])

    def test_wiring_lfull_only_and_before_ignition(self):
        src = inspect.getsource(v33.leg_stage)
        self.assertIn('if leg == "v33-full":', src)
        self.assertIn("assert_lfull_demos_chain(events)", src)
        self.assertLess(src.index("assert_lfull_demos_chain"),
                        src.index('"event": "leg_start"'))
        # fix 1b: the idempotent bc2_stage branch uses the same hard key lookup (the vacuous .get fallback is gone)
        src_bc2 = inspect.getsource(v33.bc2_stage)
        self.assertIn("n12_gate_demos_sha(passed[-1])", src_bc2)
        self.assertNotIn('.get("demos_sha256",', src_bc2)
        self.assertIn('"--manager-npz", str(M29_NPZ)', src_bc2)
        self.assertIn("expected_manager_sha256=sha256(M29_NPZ)", src_bc2)


class SentinelEvidenceTests(unittest.TestCase):
    """Fix 2 (rev4 item 12, annex 2.5): separate counts of the two sentinel/dry-anchor row types when checking all four live G0-2a
    instruments (two row types in the same sentinel.jsonl file)."""

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
                             {"sentinel": 0, "dry_anchor": 0})   # a missing file counts as 0
            p.write_text(json.dumps({"sentinel": "v23", "step": 1}) + "\n")
            counts = v33.sentinel_line_counts(p)
            self.assertEqual(counts["dry_anchor"], 0)   # the dry-level anchor did not run, so it must be 0
            self.assertEqual(counts["sentinel"], 1)

    def test_g0_2a_conjunction_carries_both_limbs(self):
        src = inspect.getsource(v33.g0_nullintrusion_stage)
        self.assertIn('evidence["knobs_sentinel_lines"] >= 1', src)
        self.assertIn('evidence["knobs_dry_anchor_lines"] >= 1', src)


class TableFullPrecisionTests(unittest.TestCase):
    """Fix 3 (rev4 item 12, annex 2.6): DRY_CURRICULUM_TABLE is booked as full-precision floats,
    identical to the source of truth re-checked by verify_curriculum_prefix; the round(p,10) criterion is abolished."""

    def test_event_table_is_full_precision_and_json_lossless(self):
        captured = []
        with mock.patch.object(v33, "log", captured.append):
            v33.dry_curriculum_table_stage([])
        ev = captured[0]
        table = v33.dry_curriculum_table()
        self.assertEqual(ev["event"], "DRY_CURRICULUM_TABLE")
        self.assertEqual(ev["table"], [float(p) for p in table])
        self.assertEqual(ev["table"][1], table[1])            # full precision, bit-identical
        self.assertNotEqual(round(table[1], 10), table[1])    # the negative case is not vacuous:
        # round(,10) already loses bits at the first entry of the linear segment, so the old criterion differs from the source of truth
        self.assertEqual(json.loads(json.dumps(ev["table"])), list(table))

    def test_idempotent_when_already_logged(self):
        captured = []
        with mock.patch.object(v33, "log", captured.append):
            v33.dry_curriculum_table_stage([{"event": "DRY_CURRICULUM_TABLE"}])
        self.assertEqual(captured, [])


class SeedSeqEquivalentTests(unittest.TestCase):
    """Fix 4 (rev4 item 12, annex 2.3): the strongest visible-surface equivalent of the G0-2b episode seed-sequence identity:
    the common prefix of (ep, reward, len) in progress.jsonl of the two runs is identical (the first rollout covers the
    lower bound of 512; progress.jsonl has no episode-seed field, as noted in the hand-over)."""

    def test_identical_sequences_full_prefix(self):
        a = [{"ep": i + 1, "reward": 1.0 * i, "len": 300} for i in range(4)]
        out = v33.progress_common_prefix(a, [dict(x) for x in a])
        self.assertEqual(out["prefix_lines"], 4)
        self.assertEqual(out["prefix_len_steps"], 1200)
        self.assertIsNone(out["first_divergence_index"])
        self.assertGreaterEqual(out["prefix_len_steps"],
                                v33.SEED_EQUIV_MIN_PREFIX_STEPS)

    def test_first_line_divergence_caught(self):
        # fingerprint of a polluted initial episode-seed stream: diverges at the first line -> prefix 0, must fail the lower bound
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
        self.assertEqual(v33.SEED_EQUIV_MIN_PREFIX_STEPS, 512)  # n-steps lower bound


class InstrumentDigestTests(unittest.TestCase):
    """Fix 5b (rev4 item 12, annex 2.1/D8): the DRYWIN_METRICS transcript payload: file sha256 +
    line count + aggregation of the final segment; a missing or empty file fails loud; written at the end of leg_stage."""

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
            with self.assertRaisesRegex(OperationalFailure, "instrument file missing"):
                v33.instrument_jsonl_digest(pathlib.Path(d) / "gone.jsonl")
            empty = pathlib.Path(d) / "drywin_metrics.jsonl"
            empty.write_text("")
            with self.assertRaisesRegex(OperationalFailure, "instrument file empty"):
                v33.instrument_jsonl_digest(empty)

    def test_leg_stage_transcribes_drywin_metrics_event(self):
        src = inspect.getsource(v33.leg_stage)
        self.assertIn('"event": "DRYWIN_METRICS"', src)
        self.assertIn('RUNS / leg / "drywin_metrics.jsonl"', src)
        self.assertIn('RUNS / leg / "distill_ce_probe.jsonl"', src)


class StatsToolTests(unittest.TestCase):
    """MS computation and the paired statistics block (R-line criterion)."""

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


class LegLockProbeTests(unittest.TestCase):
    """Fix B (launch-time regression audit, major): before launch, a non-blocking flock probe of RUNS/<leg>/.run.lock:
    if the lock is held, an orphan leg stops the run (OperationalFailure) and no leg_start is booked;
    no holder / a leftover lock file passes silently (the kernel releases flock on a crash, so the probe is reliable,
    following the train_ppo._RunLock semantics)."""

    def test_held_lock_halts_as_orphan_leg(self):
        with tempfile.TemporaryDirectory() as d:
            runs = pathlib.Path(d)
            leg_dir = runs / "v33-base"
            leg_dir.mkdir()
            lock = leg_dir / ".run.lock"
            lock.write_text('{"pid": 12345}')
            holder = open(lock, "r+")
            try:
                fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(v33, "RUNS", runs):
                    with self.assertRaisesRegex(OperationalFailure, "orphan leg"):
                        v33.assert_leg_lock_free("v33-base")
            finally:
                fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
                holder.close()

    def test_stale_or_absent_lockfile_passes(self):
        # After a crash/SIGKILL the kernel has released the flock: a leftover lock file does not count as holding the lock
        with tempfile.TemporaryDirectory() as d:
            runs = pathlib.Path(d)
            (runs / "v33-cur").mkdir()
            (runs / "v33-cur" / ".run.lock").write_text('{"pid": 1}')
            with mock.patch.object(v33, "RUNS", runs):
                v33.assert_leg_lock_free("v33-cur")     # leftover lock file: passes silently
                v33.assert_leg_lock_free("v33-full")    # no lock file: passes too

    def test_probe_wired_before_leg_start_in_leg_stage(self):
        src = inspect.getsource(v33.leg_stage)
        self.assertIn("assert_leg_lock_free(leg)", src)
        self.assertLess(src.index("assert_leg_lock_free(leg)"),
                        src.index('"event": "leg_start"'))   # probe first, then book


class RawMeanAdjudicationTests(unittest.TestCase):
    """Fix C (launch-time regression audit, major): the launch mean-difference limb and the MS branch trigger consume mean_raw
    (adjudication on the raw mean, v32 precedent); the 2dp rounded value is only for display in the ledger: rounding inside a 0.005 window
    can silently flip the verdict."""

    def test_mean_raw_carried_and_display_mean_is_its_rounding(self):
        leg = {7000 + i: _row(7000 + i, ret=110.0) for i in range(4)}
        ref = {s: _row(s, ret=100.0) for s in leg}
        st = v33.paired_diff_stats(leg, ref)
        self.assertIn("mean_raw", st)
        self.assertEqual(st["mean_raw"], 10.0)
        self.assertEqual(st["mean"], round(st["mean_raw"], 2))

    def test_17_315_window_borderline_fails_raw_line_but_ledger_shows_2dp(self):
        # Audit counterpart (the 17.315-type 0.005 rounding window at the line): 16 seeds at +17.32 and 16 seeds
        # at +17.315 -> raw mean about 17.3175, exactly inside the [17.315, 17.32) silent-flip window:
        # the rounded reading 17.32 falsely passes the line; under raw semantics < 17.32 does not pass.
        diffs = [17.32] * 16 + [17.315] * 16
        leg = {7000 + i: _row(7000 + i, ret=100.0 + diffs[i])
               for i in range(32)}
        ref = {s: _row(s, ret=100.0) for s in leg}
        st = v33.paired_diff_stats(leg, ref)
        self.assertEqual(st["mean"], 17.32)                # the ledger still shows 2dp
        self.assertGreater(st["mean_raw"], 17.315)         # exactly inside the rounding window
        self.assertLess(st["mean_raw"], v33.G1_PAIRED_DIFF)          # raw stops it
        self.assertGreaterEqual(st["mean"], v33.G1_PAIRED_DIFF)      # rounding falsely passes

    def test_launch_limb_and_ms_branch_consume_raw_not_rounded(self):
        src_main = inspect.getsource(v33._main)
        self.assertIn('st["mean_raw"] >= G1_PAIRED_DIFF', src_main)
        self.assertNotIn('st["mean"] >= G1_PAIRED_DIFF', src_main)
        src_card = inspect.getsource(v33.scorecard_stage)
        self.assertIn('st2["mean_raw"] >= RC9_BRANCH_LINE', src_card)
        self.assertNotIn('st2["mean"] >= RC9_BRANCH_LINE', src_card)


class LegTrainLogNameTests(unittest.TestCase):
    """Fix D (launch-time regression audit, minor): the leg training log file name carries the launch number: run() writes with "w"
    truncation, so with a fixed name a second launch would destroy the first launch's crash-log evidence."""

    def test_log_name_carries_attempt_and_attempts_never_collide(self):
        self.assertEqual(v33.leg_train_log_name("v33-base", 1),
                         "train-v33-base-a1.log")
        self.assertEqual(v33.leg_train_log_name("v33-base", 2),
                         "train-v33-base-a2.log")
        self.assertNotEqual(v33.leg_train_log_name("v33-base", 1),
                            v33.leg_train_log_name("v33-base", 2))

    def test_leg_stage_wires_attempt_before_leg_start_fixed_name_retired(self):
        src = inspect.getsource(v33.leg_stage)
        self.assertIn("leg_train_log_name(leg, attempt)", src)
        self.assertIn("attempt = leg_starts(events, leg) + 1", src)
        # the launch number is computed before this launch's leg_start is booked (N = already on record + 1)
        self.assertLess(src.index("attempt = leg_starts"),
                        src.index('"event": "leg_start"'))
        self.assertNotIn('f"train-{leg}.log"', src)      # the fixed name is gone


class CanaryResidueRecoveryTests(unittest.TestCase):
    """Fix E (launch-time regression audit, minor): EvalContractError raised by validate_adopted where a canary residue is adopted
    no longer passes straight through P1; per P-canary "record failures, do not stop the leg": the residue is sealed as .void,
    the event is recorded, and the function continues in its normal order."""

    @staticmethod
    def _bad_adopt(*_a, **_k):
        raise EvalContractError("synthetic bad adopted artifact")

    def test_bad_residue_voided_logged_no_raise(self):
        captured = []
        with tempfile.TemporaryDirectory() as d:
            eval_dir = pathlib.Path(d)
            tag = "v33-base-canary1-h"
            residue = eval_dir / f"{tag}.json"
            residue.write_text("{broken")
            with mock.patch.object(v33, "EVAL", eval_dir), \
                    mock.patch.object(v33, "log", captured.append), \
                    mock.patch.object(v33, "validate_adopted",
                                      self._bad_adopt):
                doc, fresh = v33.canary_exam([], "/nonexistent/policy.npz",
                                             tag, None, l1_fired=True)
            self.assertIsNone(doc)
            self.assertFalse(fresh)
            self.assertFalse(residue.exists())            # the residue has been moved away
            voids = sorted(eval_dir.glob(f"{tag}.*.void"))
            self.assertEqual(len(voids), 1)               # sealed as .void
            self.assertEqual(captured[0]["event"], "OPERATIONAL-canary")
            self.assertIn("record only, the leg continues", captured[0]["why"])
            self.assertEqual(captured[0]["void"], voids[0].name)

    def test_flow_continues_to_reexam_within_cutoff(self):
        # within the cutoff (that leg's s16 has not fired): after .void the residue goes down the re-exam path, the leg is not stopped
        synthetic = {"rows": [], "agg": {"_sha": "abcd"}}
        captured = []
        with tempfile.TemporaryDirectory() as d:
            eval_dir = pathlib.Path(d)
            tag = "v33-base-canary2-h"
            (eval_dir / f"{tag}.json").write_text("{broken")
            with mock.patch.object(v33, "EVAL", eval_dir), \
                    mock.patch.object(v33, "log", captured.append), \
                    mock.patch.object(v33, "validate_adopted",
                                      self._bad_adopt), \
                    mock.patch.object(v33, "exam",
                                      lambda *_a, **_k: synthetic):
                doc, fresh = v33.canary_exam([], "/nonexistent/policy.npz",
                                             tag, None, l1_fired=False)
            self.assertIs(doc, synthetic)
            self.assertTrue(fresh)


class LaunchCheckIdempotencyTests(unittest.TestCase):
    """Fix F (launch-time regression audit, minor): idempotency guard for booking launch_check in the no-winner branch
    (symmetric with the winner branch): re-entry does not book twice."""

    def test_reentry_does_not_duplicate_ledger_line(self):
        captured = []
        events = []
        limbs = {"verdict": "all three legs ineligible (synthetic)"}
        with mock.patch.object(v33, "log", captured.append):
            v33.log_launch_check_no_winner(events, limbs, {"v33-base": {}})
            v33.log_launch_check_no_winner(events, limbs, {"v33-base": {}})
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["event"], "launch_check")
        self.assertEqual(sum(1 for e in events
                             if e.get("event") == "launch_check"), 1)

    def test_prior_ledger_line_suppresses_relog(self):
        captured = []
        with mock.patch.object(v33, "log", captured.append):
            v33.log_launch_check_no_winner(
                [{"event": "launch_check"}], {"verdict": "x"}, {})
        self.assertEqual(captured, [])

    def test_no_winner_branch_consumes_guard(self):
        self.assertIn("log_launch_check_no_winner(events",
                      inspect.getsource(v33._main))


class DeathSeedSetsTests(unittest.TestCase):
    """Fix G (launch-time regression audit, minor): the control for the death-seed set difference in the course-4b verdict = D4-3 in
    this case's context, ctrl = L-cur x M29 (full32-m29) rows; the v32-ref-science control is retired,
    and the rescued/new_deaths semantics follow (with a context-correction note)."""

    def test_set_difference_taken_from_cur_m29_ctrl_rows(self):
        full = [{"seed": 7000, "died": False}, {"seed": 7001, "died": True},
                {"seed": 7002, "died": False}, {"seed": 7017, "died": True}]
        ctrl = [{"seed": 7000, "died": True}, {"seed": 7001, "died": False},
                {"seed": 7002, "died": False}, {"seed": 7017, "died": True}]
        out = v33.death_seed_sets(full, ctrl)
        self.assertEqual(out["rescued"], [7000])       # ctrl died, full survived
        self.assertEqual(out["new_deaths"], [7001])    # full died, ctrl survived
        self.assertIn("L-cur×M29", out["ctrl"])        # the context-correction note is on record
        # Negative case: the same full against a third-party control death surface (v32-ref-science type) changes the set difference
        # accordingly, proving that the set difference really comes from the ctrl rows passed in, not from any external reference
        science_like = [{"seed": 7000, "died": False},
                        {"seed": 7001, "died": False},
                        {"seed": 7002, "died": True},
                        {"seed": 7017, "died": False}]
        out2 = v33.death_seed_sets(full, science_like)
        self.assertEqual(out2["rescued"], [7002])
        self.assertEqual(out2["new_deaths"], [7001, 7017])

    def test_main_wires_cur_m29_rows_and_science_ref_retired(self):
        src = inspect.getsource(v33._main)
        self.assertIn("death_seed_sets(", src)
        self.assertIn('leg_docs["v33-cur"]["full32-m29"]["rows"]', src)
        self.assertNotIn('refs["science"]["rows"] if r["died"]', src)


class Rc10HalfOpenBandTests(unittest.TestCase):
    """Fix H (launch-time regression audit, minor): the RC.10 damage branch band [0.0, 0.5) is half-open:
    a retention of exactly 0.5 must fall in the healthy branch; band_judge's open-upper criterion applies to this band only,
    and the global closed-interval semantics must not change."""

    def test_exact_half_falls_healthy_not_damaged(self):
        damaged = v33.band_judge(0.5, *v33.RC10_BAND["damaged"],
                                 upper_open=True)
        self.assertFalse(damaged["in_band"])           # 0.5 is not in the damage band
        self.assertIs(damaged["upper_open"], True)
        healthy = v33.band_judge(0.5, *v33.RC10_BAND["healthy"])
        self.assertTrue(healthy["in_band"])            # falls exactly in the healthy branch

    def test_just_below_half_stays_damaged(self):
        self.assertTrue(v33.band_judge(0.4999, *v33.RC10_BAND["damaged"],
                                       upper_open=True)["in_band"])

    def test_global_closed_semantics_untouched(self):
        default = v33.band_judge(0.5, 0.0, 0.5)
        self.assertTrue(default["in_band"])            # the default closed interval is unchanged
        self.assertNotIn("upper_open", default)

    def test_scorecard_applies_open_bound_to_damaged_branch_only(self):
        src = inspect.getsource(v33.scorecard_stage)
        self.assertIn('upper_open=(branch == "damaged")', src)
        self.assertEqual(src.count("upper_open="), 1)  # only the one RC.10 place; no global change


if __name__ == "__main__":
    unittest.main()
