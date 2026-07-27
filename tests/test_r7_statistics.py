"""Pure regression tests for the deterministic R7 paired-statistics gate."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import r7_statistics as stats  # noqa: E402


BASELINE_ARCHIVE_SHA = "a" * 64
CANDIDATE_ARCHIVE_SHA = "b" * 64


def _archives(
    n: int,
    *,
    wage_improvements=None,
    kill_improvements=None,
    baseline_deaths=None,
    candidate_deaths=None,
):
    wage_improvements = (
        list(wage_improvements)
        if wage_improvements is not None
        else [1.0] * n
    )
    kill_improvements = (
        list(kill_improvements)
        if kill_improvements is not None
        else [1] * n
    )
    baseline_deaths = set(baseline_deaths or ())
    candidate_deaths = set(candidate_deaths or ())
    seeds = list(range(50_000, 50_000 + n))
    baseline_rows = []
    candidate_rows = []
    for index, seed in enumerate(seeds):
        baseline_rows.append({
            "seed": seed,
            "wage": 100.0,
            "kills": 40,
            "died": index in baseline_deaths,
        })
        candidate_rows.append({
            "seed": seed,
            "wage": 100.0 + wage_improvements[index],
            "kills": 40 + kill_improvements[index],
            "died": index in candidate_deaths,
        })
    protocol = {"seeds": seeds, "deterministic": True}
    manager = {
        "kind": "numpy_policy",
        "path": "/frozen-staging/baseline/manager.npz",
        "sha256": "c" * 64,
        "num_timesteps": None,
        "gate_report_sha256": None,
    }
    runtime = {"python_protocol": {"sha256": "d" * 64}}
    baseline = {
        "schema_version": 5,
        "meta": {
            "protocol": copy.deepcopy(protocol),
            "worker": {"sha256": "e" * 64},
            "manager": copy.deepcopy(manager),
            "runtime": copy.deepcopy(runtime),
        },
        "agg": {},
        "rows": baseline_rows,
    }
    candidate = {
        "schema_version": 5,
        "meta": {
            "protocol": copy.deepcopy(protocol),
            "worker": {"sha256": "f" * 64},
            "manager": {
                **copy.deepcopy(manager),
                "path": "/frozen-staging/candidate/manager.npz",
            },
            "runtime": copy.deepcopy(runtime),
        },
        "agg": {},
        "rows": candidate_rows,
    }
    return baseline, candidate


def _analyze(
    baseline,
    candidate,
    *,
    phase="development",
    margin=0.05,
    rules=None,
):
    return stats.analyze_paired_archives(
        baseline,
        candidate,
        baseline_archive_sha256=BASELINE_ARCHIVE_SHA,
        candidate_archive_sha256=CANDIDATE_ARCHIVE_SHA,
        phase=phase,
        metric_rules=rules or (
            stats.MetricRule("wage"),
            stats.MetricRule("kills"),
        ),
        death_noninferiority_margin=margin,
    )


class DistributionMathTests(unittest.TestCase):
    def test_student_t_critical_matches_high_precision_reference(self):
        # Values independently generated from the regularized-beta definition.
        self.assertAlmostEqual(
            stats._student_t_upper_critical(0.05, 127),
            1.6569403435420647,
            places=11,
        )
        self.assertAlmostEqual(
            stats._student_t_upper_critical(0.0125, 255),
            2.2547155186170773,
            places=11,
        )

    def test_exact_sign_tail_has_no_rng(self):
        self.assertEqual(stats._exact_sign_p_value(0, 0), 1.0)
        self.assertEqual(stats._exact_sign_p_value(8, 8), 1 / 256)
        self.assertEqual(stats._exact_sign_p_value(4, 8), 163 / 256)


class PairedGateTests(unittest.TestCase):
    def test_manager_staging_path_is_ignored_but_all_content_identity_is_strict(self):
        baseline, candidate = _archives(128)
        self.assertNotEqual(
            baseline["meta"]["manager"]["path"],
            candidate["meta"]["manager"]["path"],
        )
        self.assertEqual(
            _analyze(baseline, candidate)["verdict"]["status"],
            "PASS",
        )

        for field, drifted_value in (
            ("sha256", "9" * 64),
            ("kind", "different_manager_type"),
            ("num_timesteps", 1),
            ("gate_report_sha256", "8" * 64),
        ):
            with self.subTest(field=field):
                baseline, candidate = _archives(128)
                candidate["meta"]["manager"][field] = drifted_value
                with self.assertRaisesRegex(
                    stats.StatisticsContractError,
                    "manager",
                ):
                    _analyze(baseline, candidate)

    def test_clear_effects_pass_and_result_is_byte_deterministic(self):
        baseline, candidate = _archives(128)
        first = _analyze(baseline, candidate)
        second = _analyze(
            copy.deepcopy(baseline),
            copy.deepcopy(candidate),
            rules=(
                stats.MetricRule("kills"),
                stats.MetricRule("wage"),
            ),
        )
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )
        self.assertEqual(first["verdict"]["status"], "PASS")
        self.assertEqual(first["required_pairs"], 128)
        self.assertEqual(first["rules"][0]["key"], "kills")
        self.assertEqual(first["rules"][1]["key"], "wage")

    def test_phase_sample_floors_cannot_be_lowered(self):
        baseline, candidate = _archives(127)
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "at least 128"
        ):
            _analyze(baseline, candidate)

        baseline, candidate = _archives(255)
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "at least 256"
        ):
            _analyze(baseline, candidate, phase="final")
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "cannot be below"
        ):
            stats.analyze_paired_archives(
                baseline,
                candidate,
                baseline_archive_sha256=BASELINE_ARCHIVE_SHA,
                candidate_archive_sha256=CANDIDATE_ARCHIVE_SHA,
                phase="final",
                metric_rules=(stats.MetricRule("wage"),),
                death_noninferiority_margin=0.03,
                minimum_pairs=128,
            )

    def test_positive_outlier_mean_cannot_defeat_exact_sign_gate(self):
        improvements = [100.0] * 20 + [-1.0] * 108
        baseline, candidate = _archives(
            128,
            wage_improvements=improvements,
        )
        result = _analyze(
            baseline,
            candidate,
            rules=(stats.MetricRule("wage"),),
        )
        wage = result["metrics"]["wage"]
        self.assertGreater(wage["improvement_mean"], 0.0)
        self.assertGreater(wage["lower_confidence_bound"], 0.0)
        self.assertFalse(wage["sign_test"]["passed"])
        self.assertEqual(result["verdict"]["status"], "FAIL")

    def test_noisy_small_mean_fails_one_sided_lower_bound(self):
        improvements = [2.0, -2.0] * 64
        improvements[0] = 2.1
        baseline, candidate = _archives(
            128,
            wage_improvements=improvements,
        )
        result = _analyze(
            baseline,
            candidate,
            rules=(
                stats.MetricRule("wage", require_sign_test=False),
            ),
        )
        self.assertGreater(result["metrics"]["wage"]["improvement_mean"], 0.0)
        self.assertLess(
            result["metrics"]["wage"]["lower_confidence_bound"], 0.0
        )
        self.assertIn("wage.mean_lcb", result["verdict"]["failed_checks"])

    def test_death_gate_uses_confidence_bound_not_only_observed_count(self):
        baseline, candidate = _archives(128)
        strict = _analyze(
            baseline,
            candidate,
            margin=0.0,
            rules=(stats.MetricRule("wage"),),
        )
        self.assertEqual(strict["death_noninferiority"]["baseline_deaths"], 0)
        self.assertEqual(strict["death_noninferiority"]["candidate_deaths"], 0)
        self.assertFalse(
            strict["death_noninferiority"]["noninferiority_passed"]
        )
        self.assertEqual(strict["verdict"]["status"], "FAIL")

        practical = _analyze(
            baseline,
            candidate,
            margin=0.05,
            rules=(stats.MetricRule("wage"),),
        )
        self.assertTrue(
            practical["death_noninferiority"]["noninferiority_passed"]
        )
        self.assertEqual(practical["verdict"]["status"], "PASS")

    def test_observed_death_increase_is_a_separate_hard_failure(self):
        baseline, candidate = _archives(
            256,
            baseline_deaths=range(20),
            candidate_deaths=range(21),
        )
        result = _analyze(
            baseline,
            candidate,
            phase="final",
            margin=0.05,
        )
        death = result["death_noninferiority"]
        self.assertGreater(death["candidate_deaths"], death["baseline_deaths"])
        self.assertFalse(death["observed_not_higher_passed"])
        self.assertIn(
            "deaths.observed_not_higher", result["verdict"]["failed_checks"]
        )

    def test_archive_pairing_and_sha_contract_fail_closed(self):
        baseline, candidate = _archives(128)
        candidate["meta"]["protocol"]["seeds"] = list(
            reversed(candidate["meta"]["protocol"]["seeds"])
        )
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "protocols or seed lists"
        ):
            _analyze(baseline, candidate)

        baseline, candidate = _archives(128)
        candidate["rows"][0]["seed"] += 1
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "row order"
        ):
            _analyze(baseline, candidate)

        baseline, candidate = _archives(128)
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "complete lowercase SHA"
        ):
            stats.analyze_paired_archives(
                baseline,
                candidate,
                baseline_archive_sha256="short",
                candidate_archive_sha256=CANDIDATE_ARCHIVE_SHA,
                phase="development",
                metric_rules=(stats.MetricRule("wage"),),
                death_noninferiority_margin=0.03,
            )

    def test_nonfinite_metric_and_nonboolean_death_are_rejected(self):
        baseline, candidate = _archives(128)
        candidate["rows"][3]["wage"] = float("nan")
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "must be finite"
        ):
            _analyze(baseline, candidate)

        baseline, candidate = _archives(128)
        candidate["rows"][3]["died"] = 0
        with self.assertRaisesRegex(
            stats.StatisticsContractError, "must be bool"
        ):
            _analyze(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
