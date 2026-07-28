"""R8 认证战役冻结协议测试(PREREG-R8,批文「接受并且开始R8」+「4 核认证纯度」)。"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[1] / "train"))

import r8_statistics  # noqa: E402
import run_r8_certification as r8  # noqa: E402


class R8FrozenProtocolTests(unittest.TestCase):
    def test_campaign_identity_frozen(self):
        self.assertEqual(r8.CAMPAIGN_REVISION, 1)
        self.assertEqual(r8.NUM_ENVS, 4)  # A 案:与 R7 配方逐字节同款
        self.assertEqual(r8.N_STEPS, 512)
        self.assertEqual(r8.LEG_STEPS, 266_240)
        self.assertEqual(r8.TRAIN_CALLS, 130)
        self.assertEqual(
            r8.RECIPES, {"risk64": {"additional_terminal_death_cost": 64.0}})
        self.assertEqual(r8.RECIPE_PREFERENCE, ("risk64",))
        self.assertEqual(r8.MIN_REPLICATIONS_PASSING_BOTH_POOLS, 2)

    def test_pool_ledger_frozen(self):
        self.assertEqual(
            r8.DEV_POOLS["dev-a"], tuple(range(2_112_000, 2_112_128)))
        self.assertEqual(
            r8.DEV_POOLS["dev-b"], tuple(range(2_113_000, 2_113_128)))
        self.assertEqual(r8.FINAL_POOL, tuple(range(2_122_000, 2_122_256)))
        self.assertEqual(
            r8.DEVELOPMENT_TRAIN_SEEDS, (2_131_000, 2_131_100, 2_131_200))
        self.assertEqual(r8.PRODUCTION_TRAIN_SEED, 2_131_900)
        r8._require_seed_discipline()

    def test_death_scale_frozen(self):
        self.assertEqual(r8.FINAL_DEATH_MARGIN, 0.10)
        self.assertEqual(r8.DEVELOPMENT_DEATH_MARGIN, 0.10)
        self.assertEqual(
            r8.CAMPAIGN_RECIPE["statistics"]["method"],
            "paired-t-sign-death-mcnemar-exact/1")
        self.assertEqual(
            r8.CAMPAIGN_RECIPE["development_gate"],
            "prereg-checks-minus-deaths.noninferiority_upper_bound")

    def test_tags_derived_from_pools(self):
        self.assertEqual(
            r8._eval_tag("final", "baseline"),
            "official-r8-final-baseline-2122000")
        self.assertEqual(
            r8._eval_tag("dev-a", "candidate", recipe="risk64",
                         seed=2_131_000),
            "r8-dev-a-risk64-s2131000")

    def test_no_r7_amendment_residue(self):
        source = pathlib.Path(r8.__file__).read_text()
        for token in ("AMENDMENT", "_amendment", "adopt-", "r7_statistics"):
            self.assertNotIn(token, source)

    def test_registry_shared_ledger_accepts_both_campaign_names(self):
        # 全局 final registry 与 R7 共册:R7 历史记录(2_120/2_121)双名放行
        source = pathlib.Path(r8.__file__).read_text()
        self.assertIn("_r7_final_pool_registry", source)
        self.assertIn(
            '{"r7-official-final", "r8-official-final"}', source)
        self.assertEqual(
            r8._final_pool_spec()["pool_name"], "r8-official-final")


class R8DeathScaleKATTests(unittest.TestCase):
    """精确条件 McNemar 尺的已知答案测试(校准向量 = R7 终考实况)。"""

    def _ucb(self, c_only, b_only, n=256, alpha=0.005):
        discordant = c_only + b_only
        if not discordant:
            return 0.0
        theta_up = r8_statistics._clopper_pearson_upper(
            c_only, discordant, alpha)
        return (2.0 * theta_up - 1.0) * discordant / n

    def test_r7_final_calibration_vector(self):
        # R7 终考实况:c=31, b=31, n=256 → 0.0805(旧 CP 尺给 0.1086)
        self.assertAlmostEqual(self._ucb(31, 31), 0.0805, places=4)

    def test_zero_discordance_gives_zero_bound(self):
        self.assertEqual(self._ucb(0, 0), 0.0)

    def test_monotone_in_candidate_only(self):
        self.assertLess(self._ucb(20, 42), self._ucb(31, 31))
        self.assertLess(self._ucb(31, 31), self._ucb(42, 20))

    def test_method_and_schema_strings(self):
        self.assertEqual(
            r8_statistics.R8_METHOD_REVISION,
            "paired-t-sign-death-mcnemar-exact/1")
        self.assertEqual(
            r8_statistics.R8_STATISTICS_SCHEMA,
            "diablogym-r8-paired-statistics/1")


class R8ReplicationGateTests(unittest.TestCase):
    _CHECKS = (
        "deaths.noninferiority_upper_bound",
        "deaths.observed_not_higher",
        "farm_worker_wage.exact_sign", "farm_worker_wage.mean_lcb",
        "farm_worker_kills.exact_sign", "farm_worker_kills.mean_lcb",
        "kills.exact_sign", "kills.mean_lcb",
        "ret.exact_sign", "ret.mean_lcb",
        "gear.action14_progression",
    )

    def _analysis(self, failed):
        checks = {name: True for name in self._CHECKS}
        for name in failed:
            checks[name] = False
        return {"verdict": {
            "status": "PASS" if not failed else "FAIL",
            "checks": checks,
            "failed_checks": sorted(failed),
        }}

    def test_ni_only_failure_passes_gate(self):
        self.assertTrue(r8._replication_leg_passes(
            self._analysis(["deaths.noninferiority_upper_bound"])))
        self.assertTrue(r8._replication_leg_passes(self._analysis([])))

    def test_any_other_failure_blocks(self):
        for extra in ("deaths.observed_not_higher",
                      "farm_worker_wage.mean_lcb", "ret.exact_sign"):
            self.assertFalse(r8._replication_leg_passes(self._analysis(
                ["deaths.noninferiority_upper_bound", extra])), extra)

    def test_inconsistent_verdict_rejected(self):
        broken = self._analysis(["deaths.noninferiority_upper_bound"])
        broken["verdict"]["status"] = "PASS"
        with self.assertRaises(r8.CampaignError):
            r8._replication_leg_passes(broken)


if __name__ == "__main__":
    unittest.main()
