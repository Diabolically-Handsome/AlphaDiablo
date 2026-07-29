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
        self.assertEqual(r8.CAMPAIGN_REVISION, 2)
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
        for token in ("AMENDMENT", "_amendment5", "_amendment6",
                      "adopt-development", "adopt-final-incident",
                      "r7_statistics", "official-r7-final"):
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


class R8Amendment1GateTests(unittest.TestCase):
    def test_constants_frozen(self):
        self.assertEqual(r8.CAMPAIGN_REVISION, 2)
        self.assertEqual(r8.R8A1_POOLED_DEATH_EXCESS_MARGIN, 0.05)
        self.assertEqual(
            r8.R8A1_DROPPED_CHECKS,
            frozenset({"deaths.noninferiority_upper_bound",
                       "deaths.observed_not_higher"}))
        self.assertEqual(len(r8.R8A1_PRE_LAUNCHER_SHA256), 64)
        self.assertEqual(len(r8.R8A1_PRE_RECIPE_SHA256), 64)
        self.assertNotEqual(
            r8.CAMPAIGN_RECIPE_SHA256, r8.R8A1_PRE_RECIPE_SHA256)

    def _analysis(self, failed, cand=98, base=98):
        checks = {name: True for name in r8.R8A1_PREREG_CHECK_KEYS}
        for name in failed:
            checks[name] = False
        return {
            "verdict": {
                "status": "PASS" if not failed else "FAIL",
                "checks": checks,
                "failed_checks": sorted(failed),
            },
            "death_noninferiority": {
                "candidate_deaths": cand, "baseline_deaths": base,
            },
        }

    def test_nondeath_gate(self):
        ni = "deaths.noninferiority_upper_bound"
        onh = "deaths.observed_not_higher"
        self.assertTrue(
            r8._r8a1_leg_nondeath_passes(self._analysis([ni]))["passed"])
        self.assertTrue(
            r8._r8a1_leg_nondeath_passes(
                self._analysis([ni, onh]))["passed"])
        self.assertFalse(
            r8._r8a1_leg_nondeath_passes(
                self._analysis([ni, "ret.exact_sign"]))["passed"])
        broken = self._analysis([ni])
        del broken["verdict"]["checks"]["kills.mean_lcb"]
        with self.assertRaises(r8.CampaignError):
            r8._r8a1_leg_nondeath_passes(broken)

    def test_selection_mirrors_frozen_scene(self):
        ni = "deaths.noninferiority_upper_bound"
        onh = "deaths.observed_not_higher"
        seeds = r8.DEVELOPMENT_TRAIN_SEEDS
        fixture = {
            f"dev-a:risk64:{seeds[0]}": self._analysis([ni, onh], 101, 98),
            f"dev-b:risk64:{seeds[0]}": self._analysis([ni], 88, 96),
            f"dev-a:risk64:{seeds[1]}": self._analysis(
                [ni, onh, "farm_worker_wage.mean_lcb"], 102, 98),
            f"dev-b:risk64:{seeds[1]}": self._analysis(
                [ni, onh, "ret.exact_sign"], 104, 96),
            f"dev-a:risk64:{seeds[2]}": self._analysis([ni], 95, 98),
            f"dev-b:risk64:{seeds[2]}": self._analysis([ni], 94, 96),
        }
        sel = r8._r8a1_selection(fixture)
        self.assertEqual(sel["selected_recipe"], "risk64")
        self.assertEqual(
            sel["per_recipe"]["risk64"]["seeds_qualifying"],
            [seeds[0], seeds[2]])
        d0 = sel["derivation"][f"risk64:{seeds[0]}"]
        self.assertAlmostEqual(
            d0["pooled_death"]["excess"], -5/256, places=6)

    def test_pooled_disaster_threshold_blocks(self):
        ni = "deaths.noninferiority_upper_bound"
        seeds = r8.DEVELOPMENT_TRAIN_SEEDS
        fixture = {}
        for i, seed in enumerate(seeds):
            # 全部腿 nondeath 干净,但第一腿双池合计超额 +14/256 > 5pp
            cand = 105 if (i == 0) else 98
            fixture[f"dev-a:risk64:{seed}"] = self._analysis([ni], cand, 98)
            fixture[f"dev-b:risk64:{seed}"] = self._analysis([ni], cand, 98)
        sel = r8._r8a1_selection(fixture)
        self.assertEqual(
            sel["per_recipe"]["risk64"]["seeds_qualifying"],
            [seeds[1], seeds[2]])
        self.assertFalse(
            sel["derivation"][f"risk64:{seeds[0]}"]["pooled_death"]["passed"])

    def test_machinery_registered(self):
        source = pathlib.Path(r8.__file__).read_text()
        self.assertIn("adopt-replication", source)
        self.assertIn("复现评测阶段已封存", source)
        self.assertIn("复现训练阶段已封存", source)
        self.assertTrue(callable(r8.command_adopt_replication))
        self.assertTrue(callable(r8._validate_r8a1_adoption))


if __name__ == "__main__":
    unittest.main()
