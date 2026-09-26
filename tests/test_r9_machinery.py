"""Unit tests for the machinery of the R9 manager re-education campaign (2026-07-31 implementation self-test; tests only, no runs).

Coverage (per the self-test items of the implementation checklist):
- new train_ppo parameters: --worker-zip/--worker-zip-sha256/--deep-start-curriculum/
    --deep-start-form: --help smoke test and argparse-level mutual exclusion / mode gates (subprocess, no environment started);
- the curriculum prologue core (module-level _deep_start_prologue): unit tests at the stub-environment construction level:
    RNG determinism (derived from the episode seed, global RNG untouched), p=0/1 branches, dive/exhausted forms,
    cap limit, death resampling capped at 8, the four telemetry keys;
- reading the --worker-zip contract (_read_worker_zip_contract): rev26 / view / sovereignty fields and
    sha pinning (pure zipfile, torch is not imported);
- run_r9_reeducation: import smoke test (no disk writes) + constant assertions + depth gauges / staging helpers;
- the verdict core paired_judgment (gate limbs of the PREREG-R9 gate spec, 2026-07-31): the depth superiority limb passes /
    rejects, the wage non-inferiority line (-0.10 x anchor mean) passes / rejects, the record-only limbs stay out of the verdict,
    the semantics of the v31-D3-10 derived death-observation line (absolute lines banned), the family-wise alpha re-split over 3, and
    numerical equivalence assertions against the primitives shared with r8_statistics (t critical value / sign test / Clopper-Pearson)
    (guarding against implementation drift); the depth key names checked live against the rows of the official R8 archive.
No real environment, training or evaluation is started; evaluation-pool seeds are used only as labels of synthetic archives.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

import r8_statistics  # noqa: E402
import run_r9_reeducation as r9  # noqa: E402
import train_ppo  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")


# ---------------------------------------------------------------- stub environment

class _StubRaw:
    """Minimal OptionsEnv skeleton: .env._raw + exhausted + action_masks/step."""

    def __init__(self, *, dive_legal=True, farm_advances=False,
                 die_on_step=None, exhaust_after=None):
        class _Inner:
            pass

        self.env = _Inner()
        self.env._raw = {"dungeon_level": 1}
        self.exhausted = False
        self.dive_legal = dive_legal
        self.farm_advances = farm_advances
        self.die_on_step = die_on_step
        self.exhaust_after = exhaust_after
        self.steps = []
        self.reset_seeds = []

    # --- gym surface ---
    def action_masks(self):
        from diablogym.options_env import DIVE, FARM, RESUPPLY

        masks = [True, True, True]
        masks[DIVE] = bool(self.dive_legal)
        masks[FARM] = not (self.exhausted and masks[DIVE])
        masks[RESUPPLY] = False
        return masks

    def step(self, action):
        from diablogym.options_env import DIVE, FARM

        self.steps.append(int(action))
        index = len(self.steps)
        if self.die_on_step is not None and index >= self.die_on_step:
            return ("obs-dead", 0.0, True, False, {"stub": "dead"})
        if action == DIVE and self.dive_legal:
            self.env._raw["dungeon_level"] += 1
        if action == FARM and self.farm_advances:
            self.env._raw["dungeon_level"] += 1
        if (self.exhaust_after is not None
                and action == FARM
                and index >= self.exhaust_after):
            self.exhausted = True
        return (f"obs-{index}", 0.0, False, False, {"stub": index})

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.env._raw["dungeon_level"] = 1
        self.exhausted = False
        return "obs-reset", {"stub": "reset"}


def _run_prologue(stub, *, seed, p=1.0, target=2, cap=8, form="dive",
                  resample_seeds=None):
    spec = {"p": p, "target": target, "cap": cap}
    resample_pool = list(resample_seeds or range(9_000_000, 9_000_100))

    def resample_seed():
        return resample_pool.pop(0)

    def reset(new_seed):
        return stub.reset(seed=new_seed)

    return train_ppo._deep_start_prologue(
        stub, "obs-init", {"stub": "init"}, seed,
        spec=spec, form=form, resample_seed=resample_seed, reset=reset)


TELEMETRY_KEYS = {"prologue_triggered", "start_dlvl",
                  "prologue_windows", "resamples"}


class DeepStartPrologueTests(unittest.TestCase):
    def test_p_zero_never_triggers_and_never_steps(self):
        stub = _StubRaw()
        obs, info, seed, telemetry = _run_prologue(stub, seed=1234, p=0.0)
        self.assertEqual(set(telemetry), TELEMETRY_KEYS)
        self.assertEqual(telemetry, {
            "prologue_triggered": False, "start_dlvl": 1,
            "prologue_windows": 0, "resamples": 0})
        self.assertEqual(obs, "obs-init")
        self.assertEqual(seed, 1234)
        self.assertEqual(stub.steps, [])
        self.assertEqual(stub.reset_seeds, [])

    def test_p_one_dive_form_reaches_target(self):
        from diablogym.options_env import DIVE

        stub = _StubRaw(dive_legal=True)
        obs, info, seed, telemetry = _run_prologue(
            stub, seed=77, p=1.0, target=2, cap=8)
        self.assertEqual(telemetry, {
            "prologue_triggered": True, "start_dlvl": 2,
            "prologue_windows": 1, "resamples": 0})
        self.assertEqual(stub.steps, [DIVE])
        self.assertEqual(seed, 77)
        self.assertEqual(obs, "obs-1")

    def test_dive_form_falls_back_to_farm_and_caps(self):
        from diablogym.options_env import FARM

        stub = _StubRaw(dive_legal=False, farm_advances=False)
        obs, info, seed, telemetry = _run_prologue(
            stub, seed=5, p=1.0, target=2, cap=3)
        self.assertEqual(telemetry["prologue_windows"], 3)      # cap limit
        self.assertEqual(telemetry["start_dlvl"], 1)            # not reached: reported as is
        self.assertEqual(stub.steps, [FARM, FARM, FARM])        # DIVE masked -> FARM
        self.assertEqual(telemetry["resamples"], 0)

    def test_exhausted_form_stops_on_handover_state(self):
        from diablogym.options_env import FARM

        stub = _StubRaw(dive_legal=True, exhaust_after=2)
        obs, info, seed, telemetry = _run_prologue(
            stub, seed=6, p=1.0, form="exhausted", target=2, cap=8)
        # After the 2nd FARM window exhausted and DIVE legal -> stop at round 3, no further step.
        self.assertEqual(stub.steps, [FARM, FARM])
        self.assertEqual(telemetry["prologue_windows"], 2)
        self.assertTrue(stub.exhausted)

    def test_death_resamples_capped_at_eight_then_plain_start(self):
        stub = _StubRaw(die_on_step=1)      # every attempt dies in the first window
        resample_seeds = list(range(8_000_000, 8_000_050))
        obs, info, seed, telemetry = _run_prologue(
            stub, seed=42, p=1.0, resample_seeds=resample_seeds)
        self.assertEqual(telemetry["prologue_triggered"], True)
        self.assertEqual(telemetry["resamples"],
                         train_ppo._DEEP_START_MAX_RESAMPLES)
        self.assertEqual(telemetry["prologue_windows"], 0)
        # attempts = 1 initial + 8 resamples = 9, each dying in the first window; after giving up, one more reset hands over.
        self.assertEqual(len(stub.steps), 9)
        self.assertEqual(len(stub.reset_seeds), 9)
        self.assertEqual(seed, resample_seeds[8])   # hand-over seed = the new seed after giving up
        self.assertEqual(obs, "obs-reset")

    def test_trigger_rng_is_seed_derived_and_deterministic(self):
        # decision rule = random.Random(seed ^ 0xD1CE).random() < p (independent per episode).
        for seed in (0, 1, 22, 4093, 2_000_003):
            expected = random.Random(seed ^ 0xD1CE).random() < 0.5
            stub = _StubRaw()
            _, _, _, telemetry = _run_prologue(stub, seed=seed, p=0.5)
            self.assertEqual(telemetry["prologue_triggered"], expected,
                             f"seed={seed}")
        # Two runs with the same seed (fresh stubs) give bit-identical telemetry.
        first = _run_prologue(_StubRaw(), seed=908, p=0.5)[3]
        second = _run_prologue(_StubRaw(), seed=908, p=0.5)[3]
        self.assertEqual(first, second)

    def test_global_random_state_untouched(self):
        random.seed(31337)
        before = random.getstate()
        _run_prologue(_StubRaw(), seed=15, p=1.0)
        self.assertEqual(random.getstate(), before)

    def test_parse_deep_start_curriculum(self):
        parse = train_ppo._parse_deep_start_curriculum
        self.assertEqual(parse("p=0.5,target=2,cap=8"),
                         {"p": 0.5, "target": 2, "cap": 8})
        self.assertEqual(parse(" p=1 , target=3 , cap=1 "),
                         {"p": 1.0, "target": 3, "cap": 1})
        for bad in ("", "p=0.5,target=2", "p=0.5,target=2,cap=8,cap=8",
                    "p=0.5,target=2,cap=8,extra=1", "p=1.5,target=2,cap=8",
                    "p=-0.1,target=2,cap=8", "p=0.5,target=1,cap=8",
                    "p=0.5,target=2,cap=0", "p=abc,target=2,cap=8",
                    "p:0.5,target=2,cap=8"):
            with self.assertRaises(ValueError, msg=bad):
                parse(bad)


# ------------------------------------------------- reading the worker zip contract

def _make_worker_zip(path: pathlib.Path, contract) -> str:
    data = {"num_timesteps": 3_764_224}
    if contract is not None:
        data["diablogym_contract"] = contract
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data))
        archive.writestr("_stable_baselines3_version", "2.9.0")
    return hashlib.sha256(path.read_bytes()).hexdigest()


_REV26_CONTRACT = {
    "contract_revision": 26,
    "worker_policy_observation_view": "dual-v4-asymmetric-v3",
    "drink_sovereignty": False,
}


class WorkerZipContractTests(unittest.TestCase):
    def test_reads_rev26_contract_with_sha_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "worker.zip"
            digest = _make_worker_zip(path, dict(_REV26_CONTRACT))
            contract = train_ppo._read_worker_zip_contract(
                path, expected_sha256=digest)
            self.assertEqual(contract["contract_revision"], 26)
            self.assertEqual(contract["worker_policy_observation_view"],
                             "dual-v4-asymmetric-v3")
            self.assertIs(contract["drink_sovereignty"], False)

    def test_rejects_sha_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "worker.zip"
            _make_worker_zip(path, dict(_REV26_CONTRACT))
            with self.assertRaisesRegex(ValueError, "SHA256 drift"):
                train_ppo._read_worker_zip_contract(
                    path, expected_sha256="0" * 64)

    def test_rejects_wrong_revision_missing_contract_and_non_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = pathlib.Path(tmp) / "wrong.zip"
            _make_worker_zip(wrong, {**_REV26_CONTRACT, "contract_revision": 25})
            with self.assertRaisesRegex(ValueError, "contract_revision"):
                train_ppo._read_worker_zip_contract(wrong)
            bare = pathlib.Path(tmp) / "bare.zip"
            _make_worker_zip(bare, None)
            with self.assertRaisesRegex(ValueError, "diablogym_contract"):
                train_ppo._read_worker_zip_contract(bare)
            not_zip = pathlib.Path(tmp) / "not.zip"
            not_zip.write_bytes(b"not a zip")
            with self.assertRaises(ValueError):
                train_ppo._read_worker_zip_contract(not_zip)
            with self.assertRaises(ValueError):
                train_ppo._read_worker_zip_contract(
                    pathlib.Path(tmp) / "absent.zip")


# ------------------------------------------------- train_ppo CLI gates (argparse level)

def _cli(args):
    return subprocess.run(
        [PY, str(ROOT / "train" / "train_ppo.py"), *args],
        capture_output=True, text=True, cwd=ROOT, timeout=300)


class TrainPpoCliGateTests(unittest.TestCase):
    def test_help_lists_new_knobs(self):
        proc = _cli(["--help"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ("--worker-zip", "--worker-zip-sha256",
                     "--deep-start-curriculum", "--deep-start-form"):
            self.assertIn(flag, proc.stdout)

    def test_worker_zip_conflicts_with_worker_npz(self):
        proc = _cli(["--options", "--algo", "mppo", "--gamma", "1.0",
                     "--max-steps", "3000", "--n-steps", "64", "--seed", "22",
                     "--worker-zip", "x.zip", "--worker-npz", "y.npz"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--worker-zip and --worker-npz are mutually exclusive", proc.stderr)

    def test_worker_zip_requires_options_mode(self):
        proc = _cli(["--worker-zip", "x.zip"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--worker-zip can only be used with --options", proc.stderr)

    def test_curriculum_gates(self):
        proc = _cli(["--deep-start-curriculum", "p=0.5,target=2,cap=8"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--deep-start-curriculum can only be used with --options",
                      proc.stderr)
        proc = _cli(["--options", "--algo", "mppo", "--gamma", "1.0",
                     "--max-steps", "3000", "--deep-start-form", "dive"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--deep-start-form must be used with --deep-start-curriculum",
                      proc.stderr)


# ------------------------------------------------- synthetic paired archives (R9 role geometry)

N_PAIRS = 128
SEEDS = list(range(2_123_000, 2_123_000 + N_PAIRS))
SHA_A = "a" * 64
SHA_B = "b" * 64
NPZ_SHA_CAND = "c" * 64
WORKER_SHA_CAND = "f" * 64
BUNDLE_SHA = "1" * 64


def _synthetic_rows(*, depth_boost=100, wage_factor=1.0, ret_delta=8.0,
                    kills_delta=2, candidate_extra_deaths=0):
    """Synthetic paired rows (gate limbs as knobs): depth_boost = depth+1 on the candidate's first N seeds (main metric
    offset knob); wage_factor = candidate wage multiplier (0.95 = slight drop / 0.80 = below the 10% line);
    ret_delta/kills_delta = record-only limb knobs; candidate_extra_deaths = extra deaths on the candidate side
    (creates McNemar discordant pairs and offsets the observation line)."""
    rng = random.Random(90210)
    baseline, candidate = [], []
    flips = 0
    for index, seed in enumerate(SEEDS):
        wage = round(rng.uniform(20.0, 60.0), 3)
        kills = rng.randrange(0, 40)
        ret = round(wage + rng.uniform(-5.0, 25.0), 3)
        depth = rng.choice((1, 1, 2, 2, 3))
        b_died = rng.random() < 0.55
        baseline.append({
            "seed": seed, "ret": ret, "kills": kills,
            "farm_worker_wage": wage, "farm_worker_kills": kills // 2,
            "depth": depth, "died": b_died,
        })
        c_died = b_died
        if not b_died and flips < candidate_extra_deaths:
            c_died = True
            flips += 1
        candidate.append({
            "seed": seed,
            "ret": round(ret + ret_delta + rng.uniform(-2.0, 2.0), 3),
            "kills": max(0, kills + kills_delta),
            "farm_worker_wage": round(wage * wage_factor, 3),
            "farm_worker_kills": kills // 2,
            "depth": depth + (1 if index < depth_boost else 0),
            "died": c_died,
        })
    return baseline, candidate


def _doc(rows, *, worker, manager):
    protocol = {"name": "diablogym.eval_assembled", "version": 4,
                "environment": "OptionsEnv", "max_steps": 3000,
                "action_selection": "argmax_with_action_masks",
                "manager_forward": "numpy_tanh_mlp",
                "reward": "undiscounted_manager_ledger",
                "deterministic": True, "seeds": list(SEEDS)}
    runtime = {"python_protocol": {"sha256": BUNDLE_SHA}}
    return {"schema_version": 5,
            "meta": {"protocol": protocol, "worker": worker,
                     "manager": manager, "runtime": runtime},
            "agg": {}, "rows": copy.deepcopy(rows)}


def _r9_geometry(b_rows, c_rows):
    """R9 dual geometry: a shared certified worker zip (identical); the variable is the manager npz."""
    worker = {"kind": "sb3_checkpoint", "sha256": r9.W_ZIP_SHA,
              "num_timesteps": 3_764_224, "gate_report_sha256": None}
    baseline = _doc(b_rows, worker=dict(worker),
                    manager={"kind": "numpy_policy",
                             "path": "/models/m29/policy.npz",
                             "sha256": r9.M29_SHA, "num_timesteps": None,
                             "gate_report_sha256": None})
    candidate = _doc(c_rows, worker=dict(worker),
                     manager={"kind": "numpy_policy",
                              "path": "/runs/r9-mfresh/policy.npz",
                              "sha256": NPZ_SHA_CAND, "num_timesteps": None,
                              "gate_report_sha256": None})
    return baseline, candidate


class PairedJudgmentTests(unittest.TestCase):
    GATE_CHECKS = {
        "depth.mean_lcb", "depth.exact_sign",
        "farm_worker_wage.noninferiority_lcb",
        "deaths.noninferiority_upper_bound",
        "deaths.observed_within_derived_line",
    }

    def _judge(self, b_rows, c_rows, *, phase="development"):
        base, cand = _r9_geometry(b_rows, c_rows)
        return r9.paired_judgment(base, cand, baseline_sha256=SHA_A,
                                  candidate_sha256=SHA_B, phase=phase)

    def test_depth_gate_passes_and_gate_set_is_exact(self):
        # depth shifted up systematically + wage flat + no new deaths -> every gate passes;
        # also pins the set of gate checks and the alpha re-split (0.05/3).
        b_rows, c_rows = _synthetic_rows(depth_boost=100, wage_factor=1.0,
                                         ret_delta=-30.0, kills_delta=-10)
        subject = self._judge(b_rows, c_rows)
        self.assertEqual(set(subject["verdict"]["checks"]), self.GATE_CHECKS)
        self.assertEqual(subject["constraint_count"], 3)
        self.assertEqual(subject["per_constraint_alpha"],
                         r9.FAMILYWISE_ALPHA / 3)
        self.assertTrue(subject["verdict"]["checks"]["depth.mean_lcb"])
        self.assertTrue(subject["verdict"]["checks"]["depth.exact_sign"])
        self.assertTrue(subject["metrics"]["depth"]["passed"])
        self.assertEqual(subject["metrics"]["depth"]["kind"], "superiority")
        self.assertEqual(subject["verdict"]["status"], "PASS")
        self.assertEqual(subject["schema_version"],
                         "diablogym-r9-paired-statistics/2")
        self.assertTrue(subject["method_revision"].startswith(
            r8_statistics.R8_METHOD_REVISION))

    def test_depth_gate_rejects_flat_depth(self):
        # candidate depth identical to the anchor seed by seed (all ties) -> both checks of the main metric fail, the verdict is FAIL.
        b_rows, c_rows = _synthetic_rows(depth_boost=0)
        subject = self._judge(b_rows, c_rows)
        self.assertEqual(subject["verdict"]["status"], "FAIL")
        self.assertFalse(subject["verdict"]["checks"]["depth.mean_lcb"])
        self.assertFalse(subject["verdict"]["checks"]["depth.exact_sign"])
        self.assertIn("depth.mean_lcb", subject["verdict"]["failed_checks"])
        self.assertIn("depth.exact_sign", subject["verdict"]["failed_checks"])
        # all ties -> the r8 primitive sign test degenerates to p=1 (non_ties=0).
        self.assertEqual(
            subject["metrics"]["depth"]["sign_test"]
            ["one_sided_exact_p_value"], 1.0)

    def test_wage_noninferiority_line_pass_and_reject(self):
        # a slight drop (-5%) is inside the -10% x anchor-mean line -> passes; exact_sign is not a gate.
        b_rows, c_rows = _synthetic_rows(depth_boost=100, wage_factor=0.95)
        subject = self._judge(b_rows, c_rows)
        wage = subject["metrics"]["farm_worker_wage"]
        expected_line = -r9.WAGE_NI_FRACTION * (
            math.fsum(float(r["farm_worker_wage"]) for r in b_rows) / N_PAIRS)
        self.assertEqual(wage["noninferiority"]["minimum_effect"],
                         expected_line)
        self.assertEqual(wage["noninferiority"]["margin_fraction"],
                         r9.WAGE_NI_FRACTION)
        self.assertIn("anchor_wage_mean", wage["noninferiority"]["formula"])
        self.assertEqual(wage["kind"], "noninferiority")
        self.assertLess(wage["improvement_mean"], 0.0)   # really a drop, not superiority
        self.assertTrue(
            subject["verdict"]["checks"]["farm_worker_wage.noninferiority_lcb"])
        self.assertNotIn("farm_worker_wage.exact_sign",
                         subject["verdict"]["checks"])
        self.assertIs(wage["sign_test"]["required"], False)
        self.assertEqual(subject["verdict"]["status"], "PASS")
        # below the 10% line (-20%) -> stopped.
        b_rows, c_rows = _synthetic_rows(depth_boost=100, wage_factor=0.80)
        subject = self._judge(b_rows, c_rows)
        self.assertFalse(
            subject["verdict"]["checks"]["farm_worker_wage.noninferiority_lcb"])
        self.assertIn("farm_worker_wage.noninferiority_lcb",
                      subject["verdict"]["failed_checks"])
        self.assertEqual(subject["verdict"]["status"], "FAIL")

    def test_record_only_limbs_do_not_gate(self):
        # A large degradation of ret/kills does not affect the verdict; still computed and archived, but not in checks.
        b_rows, c_rows = _synthetic_rows(depth_boost=100, wage_factor=1.0,
                                         ret_delta=-40.0, kills_delta=-20)
        subject = self._judge(b_rows, c_rows)
        self.assertEqual(subject["verdict"]["status"], "PASS")
        self.assertEqual(set(subject["record_only_metrics"]),
                         {"ret", "kills", "farm_worker_kills"})
        record_ret = subject["record_only_metrics"]["ret"]
        self.assertIs(record_ret["record_only"], True)
        self.assertLess(record_ret["improvement_mean"], 0.0)
        self.assertFalse(record_ret["mean_lcb_indicative"])
        for name in subject["verdict"]["checks"]:
            self.assertFalse(
                name.startswith(("ret.", "kills.", "farm_worker_kills.")),
                name)
        self.assertEqual(subject["rules"]["record_only"],
                         ["ret", "kills", "farm_worker_kills"])
        self.assertEqual(
            [rule["key"] for rule in subject["rules"]["gating"]],
            ["depth", "farm_worker_wage", "died"])

    def test_derived_death_line_relaxes_and_still_bounds(self):
        # Candidate has 3 more deaths than the anchor: the old absolute check "observed <= anchor" must reject; the derived line (anchor + 0.10 x 128) passes.
        b_rows, c_rows = _synthetic_rows(depth_boost=100,
                                         candidate_extra_deaths=3)
        subject = self._judge(b_rows, c_rows)
        death = subject["death_noninferiority"]
        self.assertGreater(death["candidate_deaths"], death["baseline_deaths"])
        self.assertTrue(
            subject["verdict"]["checks"]["deaths.observed_within_derived_line"])
        self.assertEqual(subject["verdict"]["status"], "PASS")
        line = death["derived_observed_line"]
        self.assertEqual(line["line_deaths"],
                         death["baseline_deaths"] + 0.10 * N_PAIRS)
        # The line value must be derived from the anchor (the formula references anchor_deaths), not any absolute constant.
        self.assertIn("anchor_deaths", line["formula"])
        self.assertEqual(line["anchor_deaths"], death["baseline_deaths"])
        # beyond the margin (+14 > 12.8) -> stopped.
        b_rows, c_rows = _synthetic_rows(depth_boost=100,
                                         candidate_extra_deaths=14)
        subject = self._judge(b_rows, c_rows)
        self.assertFalse(
            subject["verdict"]["checks"]["deaths.observed_within_derived_line"])
        self.assertIn("deaths.observed_within_derived_line",
                      subject["verdict"]["failed_checks"])

    def test_shared_primitive_equivalence_with_r8(self):
        # Gate spec item 2: the equivalence regression now pins the shared primitives (t critical value / sign test / Clopper-Pearson),
        # and the LCB assembly of the verdict core is re-checked independently by hand.
        b_rows, c_rows = _synthetic_rows(depth_boost=77, wage_factor=0.97,
                                         candidate_extra_deaths=5)
        subject = self._judge(b_rows, c_rows)
        per_alpha = r9.FAMILYWISE_ALPHA / r9.GATE_CONSTRAINT_COUNT
        self.assertEqual(subject["per_constraint_alpha"], per_alpha)
        t_critical = r8_statistics._student_t_upper_critical(
            per_alpha, N_PAIRS - 1)
        depth = subject["metrics"]["depth"]
        self.assertEqual(depth["one_sided_t_critical"], t_critical)
        diffs = [float(c["depth"]) - float(b["depth"])
                 for b, c in zip(b_rows, c_rows)]
        wins = sum(d > 0.0 for d in diffs)
        losses = sum(d < 0.0 for d in diffs)
        self.assertEqual(
            depth["sign_test"]["one_sided_exact_p_value"],
            r8_statistics._exact_sign_p_value(wins, wins + losses))
        mean = math.fsum(diffs) / N_PAIRS
        stddev = math.sqrt(
            math.fsum((d - mean) ** 2 for d in diffs) / (N_PAIRS - 1))
        self.assertEqual(depth["lower_confidence_bound"],
                         mean - t_critical * (stddev / math.sqrt(N_PAIRS)))
        death = subject["death_noninferiority"]
        candidate_only = sum(
            (not b["died"]) and c["died"] for b, c in zip(b_rows, c_rows))
        baseline_only = sum(
            b["died"] and (not c["died"]) for b, c in zip(b_rows, c_rows))
        discordant = candidate_only + baseline_only
        self.assertGreater(discordant, 0)
        self.assertEqual(death["component_alpha"], per_alpha / 2.0)
        theta = r8_statistics._clopper_pearson_upper(
            candidate_only, discordant, per_alpha / 2.0)
        self.assertEqual(
            death["candidate_only_conditional_theta_upper_bound"], theta)
        self.assertEqual(
            death["candidate_minus_baseline_risk_upper_bound"],
            (2.0 * theta - 1.0) * discordant / N_PAIRS)

    def test_role_geometry_guards(self):
        b_rows, c_rows = _synthetic_rows()
        base, cand = _r9_geometry(b_rows, c_rows)
        wrong_mgr = copy.deepcopy(base)
        wrong_mgr["meta"]["manager"]["sha256"] = NPZ_SHA_CAND
        with self.assertRaisesRegex(ValueError, "M29"):
            r9.paired_judgment(wrong_mgr, cand, baseline_sha256=SHA_A,
                               candidate_sha256=SHA_B, phase="development")
        wrong_worker = copy.deepcopy(cand)
        wrong_worker["meta"]["worker"]["sha256"] = WORKER_SHA_CAND
        with self.assertRaisesRegex(ValueError, "certified release"):
            r9.paired_judgment(base, wrong_worker, baseline_sha256=SHA_A,
                               candidate_sha256=SHA_B, phase="development")
        same_mgr = copy.deepcopy(cand)
        same_mgr["meta"]["manager"]["sha256"] = r9.M29_SHA
        with self.assertRaisesRegex(ValueError, "must differ"):
            r9.paired_judgment(base, same_mgr, baseline_sha256=SHA_A,
                               candidate_sha256=SHA_B, phase="development")

    def test_phase_floor_enforced(self):
        b_rows, c_rows = _synthetic_rows()
        short_seeds = SEEDS[:127]
        base, cand = _r9_geometry(b_rows[:127], c_rows[:127])
        for doc in (base, cand):
            doc["meta"]["protocol"]["seeds"] = list(short_seeds)
        with self.assertRaisesRegex(ValueError, "needs at least 128"):
            r9.paired_judgment(base, cand, baseline_sha256=SHA_A,
                               candidate_sha256=SHA_B, phase="development")
        with self.assertRaisesRegex(ValueError, "needs at least 256"):
            full_base, full_cand = _r9_geometry(b_rows, c_rows)
            r9.paired_judgment(full_base, full_cand, baseline_sha256=SHA_A,
                               candidate_sha256=SHA_B, phase="final")


# ------------------------------------------------- run_r9 driver constants and helpers

class RunR9DriverTests(unittest.TestCase):
    def test_import_did_not_create_campaign_dir(self):
        # The import smoke test already happened at module top; during implementation the campaign directory must not exist
        # (when this test is rerun after launch, the directory is a campaign output and outside the scope of this assertion).
        ledger_exists = (r9.R9 / "gate_ledger.jsonl").exists()
        if not ledger_exists:
            self.assertFalse(r9.R9.exists(),
                             "import run_r9_reeducation must not create the campaign directory")

    def test_frozen_campaign_constants(self):
        self.assertEqual(r9.CALIBRATED_PROTOCOL_VERSION, 4)
        self.assertEqual(r9.STEPS, 160_000)
        self.assertEqual(r9.POOL_A, (2_123_000, 2_123_127))
        self.assertEqual(r9.POOL_B, (2_124_000, 2_124_127))
        self.assertEqual(r9.POOL_FINAL, (2_125_000, 2_125_255))
        self.assertEqual(r9.R8_FINAL_SEEDS, (2_122_000, 2_122_255))
        self.assertEqual(r9.DEATH_MARGIN, 0.10)
        self.assertEqual(r9.FAMILYWISE_ALPHA, 0.05)
        self.assertEqual(r9.ABANDON_FACTOR, 0.66)
        self.assertEqual((r9.FLOOR_NUM, r9.FLOOR_DEN), (85.0, 92.0))
        self.assertEqual(
            r9.W_ZIP_SHA,
            "2837288dad19a685925558a0d86e1cecd951d5f065d1d2f367f667c13b9cf006")
        self.assertEqual(
            r9.M29_SHA,
            "894413884d04adfdb2a574866a15dfed0c1c01d6781403d9ab4ff07b1f7b66d6")
        self.assertEqual(set(r9.ARMS), {"r9-mfresh", "r9-mcurr"})
        self.assertNotIn("--deep-start-curriculum", r9.ARMS["r9-mfresh"])
        curr = r9.ARMS["r9-mcurr"]
        self.assertIn("--deep-start-curriculum", curr)
        self.assertEqual(curr[curr.index("--deep-start-curriculum") + 1],
                         "p=0.5,target=2,cap=8")
        # The two arms share one recipe (the only variable is the curriculum): seed/ent/lr identical.
        self.assertEqual(r9.ARMS["r9-mfresh"], curr[:len(r9.ARMS["r9-mfresh"])])
        # Gate limbs (gate spec): depth superiority as the main metric + wage non-inferiority + deaths; alpha re-split over 3.
        self.assertEqual(r9.DEPTH_RULE.key, "depth")
        self.assertEqual(float(r9.DEPTH_RULE.minimum_effect), 0.0)
        self.assertTrue(r9.DEPTH_RULE.require_sign_test)
        self.assertEqual(r9.WAGE_KEY, "farm_worker_wage")
        self.assertEqual(r9.WAGE_NI_FRACTION, 0.10)
        self.assertEqual(r9.GATE_CONSTRAINT_COUNT, 3)
        self.assertEqual([rule.key for rule in r9.RECORD_ONLY_RULES],
                         ["ret", "kills", "farm_worker_kills"])
        self.assertEqual(r9.GAUGE_BASELINES, {
            "first_forced_handover_median_micro_steps": 1495,
            "dive_window_success_rate": 0.25,
            "dlvl_dwell_ratio_l1_l2": [16058, 3512]})
        tags = list(r9.TAG_ANCHOR.values())
        tags += [f"{arm}-{pool}-{r9.POOLS[pool][0]}"
                 for arm in r9.ARMS for pool in ("a", "b")]
        tags += [f"r9-final-{arm}-{r9.POOL_FINAL[0]}" for arm in r9.ARMS]
        self.assertEqual(len(tags), len(set(tags)), "target tags must be globally unique")
        self.assertEqual(len(r9.R8_FINAL), 2)
        for spec in r9.R8_FINAL.values():
            self.assertEqual(len(spec["archive_sha256"]), 64)

    def test_seed_helpers_and_gauges(self):
        self.assertEqual(r9.seeds_arg((2_123_000, 2_123_127)),
                         "2123000-2123127")
        rows = [
            {"seed": 2_123_000, "ret": 10.0, "depth": 1, "died": True,
             "mode_seq": "FF"},
            {"seed": 2_123_001, "ret": 30.0, "depth": 3, "died": False,
             "mode_seq": "FDFD"},
            {"seed": 2_123_002, "ret": 20.0, "depth": 2, "died": False,
             "mode_seq": "FDD†"},
        ]
        mapped = r9.by_seed(rows, (2_123_000, 2_123_002))
        self.assertEqual(set(mapped), {2_123_000, 2_123_001, 2_123_002})
        with self.assertRaisesRegex(ValueError, "abnormal seed set"):
            r9.by_seed(rows, (2_123_000, 2_123_003))
        self.assertEqual(r9.depth_hist(rows), {"1": 1, "2": 1, "3": 1})
        self.assertEqual(r9.l3_plus(rows), 1)
        self.assertEqual(r9.died_seeds(rows), [2_123_000])
        gauges = r9.dive_gauges(rows)
        self.assertEqual(gauges["dive_windows"], 4)     # D count includes D-dagger windows
        self.assertEqual(gauges["descents"], 3)         # (1-1)+(3-1)+(2-1)
        self.assertAlmostEqual(gauges["dive_window_success_rate"], 0.75)
        dashboard = r9.depth_dashboard(rows)
        for key in ("depth_hist", "l3_plus", "died_seeds", "dive_per_ep",
                    "bonus_per_ep", "dive_window_success_rate"):
            self.assertIn(key, dashboard)
        report = r9.gauge_report(rows, rows, r9.TAG_ANCHOR["final"])
        self.assertEqual(report["baselines_foundation_dossier"],
                         r9.GAUGE_BASELINES)
        self.assertIn("probe_r9_dive", report["probe_level_gauges"]
                      ["anchor_recheck_cmd"])
        self.assertEqual(
            report["probe_level_gauges"]["gauges"],
            ["first_forced_handover_median_micro_steps",
             "dlvl_dwell_ratio_l1_l2"])

    def test_winner_selection_depth_band_died_break(self):
        # Final spec item: winner selection = paired depth mean difference, band 0.10; ties inside the band go to (1) fewer deaths (2) mfresh;
        # ret is removed from winner selection (still recorded in quals, not decisive).
        self.assertEqual(r9.WINNER_DEPTH_TIE_BAND, 0.10)

        def quals(mf_depth, mc_depth, mf_died, mc_died):
            return {
                "r9-mfresh": {"depth_paired_mean_ab": mf_depth,
                              "pooled_died": mf_died,
                              "paired_mean_ab": 99.0},    # ret as a counterweight in the opposite direction: must have no effect
                "r9-mcurr": {"depth_paired_mean_ab": mc_depth,
                             "pooled_died": mc_died,
                             "paired_mean_ab": -99.0},
            }

        both = ["r9-mfresh", "r9-mcurr"]
        # outside the band: a clear depth difference (0.30 > 0.10) -> the higher wins; ret/died in the opposite direction have no effect.
        self.assertEqual(
            r9.select_winner(quals(0.20, 0.50, 10, 90), both), "r9-mcurr")
        # tie inside the band (0.05 <= 0.10) -> died breaks the tie: fewer deaths wins (tiebreak case added to the spec).
        self.assertEqual(
            r9.select_winner(quals(0.45, 0.50, 70, 60), both), "r9-mcurr")
        self.assertEqual(
            r9.select_winner(quals(0.50, 0.45, 60, 70), both), "r9-mfresh")
        # depth and died both tied -> take mfresh (Occam: the curriculum arm has to prove itself by a depth advantage).
        self.assertEqual(
            r9.select_winner(quals(0.50, 0.50, 60, 60), both), "r9-mfresh")
        # only one arm passes the gate: take it directly; an empty pass list is rejected.
        self.assertEqual(
            r9.select_winner(quals(0.20, 0.50, 10, 90), ["r9-mfresh"]),
            "r9-mfresh")
        with self.assertRaises(ValueError):
            r9.select_winner(quals(0.2, 0.5, 1, 1), [])

    def test_depth_key_matches_official_archive_rows(self):
        # Gate spec item 1: the depth key names follow the live archives, checked against the rows of the official R8 final-exam archive.
        archive = (ROOT / "train" / "runs" / "eval-assembled"
                   / "official-r8-final-baseline-2122000.json")
        rows = json.loads(archive.read_text())["rows"]
        self.assertIn(r9.DEPTH_RULE.key, rows[0])
        self.assertNotIn("dungeon_level", rows[0])
        self.assertTrue(all(isinstance(row[r9.DEPTH_RULE.key], int)
                            and not isinstance(row[r9.DEPTH_RULE.key], bool)
                            for row in rows))

    def test_probe_mode_seq_reconstruction(self):
        windows = [{"opt": 0, "reason": "tau"}, {"opt": 1, "reason": "descend"},
                   {"opt": 2, "reason": "cap"}, {"opt": 1, "reason": "death"}]
        self.assertEqual(r9.probe_mode_seq(windows), "FDRD†")

    def test_stage_eval_file_immutable_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "src.bin"
            source.write_bytes(b"payload-1")
            digest = hashlib.sha256(b"payload-1").hexdigest()
            destination = pathlib.Path(tmp) / "staging" / "worker.zip"
            staged = r9._stage_eval_file(source, destination,
                                         expected_sha256=digest)
            self.assertEqual(staged, digest)
            self.assertEqual(destination.read_bytes(), b"payload-1")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
            # Re-staging identical bytes is idempotent; different bytes refuse to overwrite.
            self.assertEqual(
                r9._stage_eval_file(source, destination,
                                    expected_sha256=digest), digest)
            drifted = pathlib.Path(tmp) / "drift.bin"
            drifted.write_bytes(b"payload-2")
            with self.assertRaisesRegex(ValueError, "content drift"):
                r9._stage_eval_file(drifted, destination)
            with self.assertRaisesRegex(ValueError, "SHA drift"):
                r9._stage_eval_file(source, pathlib.Path(tmp) / "other.bin",
                                    expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
