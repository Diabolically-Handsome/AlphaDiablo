"""分腿 driver 的纯状态机契约测试（不启动训练或评测）。"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import run_v24_legs as v24  # noqa: E402
import run_v25_election as v25  # noqa: E402
import run_v26_legs as v26  # noqa: E402
import run_v27_legs as v27  # noqa: E402
import run_v28_legs as v28  # noqa: E402
import run_v29_relection as v29  # noqa: E402
import run_v30_relay as v30  # noqa: E402


LEG_DRIVERS = (v24, v26, v27, v28)
GCAL_DRIVERS = (v24, v26, v27)
ALL_DRIVERS = (*LEG_DRIVERS, v30)


class DriverContractTests(unittest.TestCase):
    def test_every_training_launch_uses_an_exact_rollout_quantum(self):
        for module in ALL_DRIVERS:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.LEG % module.QUANTUM, 0)
        self.assertEqual(v25.EXPECTED_STEPS % (64 * 4), 0)
        self.assertEqual(v29.STEPS % (64 * 4), 0)

    def test_every_launch_is_bounded_by_remaining_budget(self):
        for module in ALL_DRIVERS:
            with self.subTest(module=module.__name__):
                # 三次大幅失败足以让旧实现尚在前腿时就继续申请整腿并超额。
                spent = 0
                for fraction in (3, 3, 2):
                    allocated = module.budgeted_leg_steps(spent)
                    self.assertLessEqual(spent + allocated, module.BUDGET_STEPS)
                    observed = min(allocated, module.LEG * fraction // 4)
                    spent += observed
                allocated = module.budgeted_leg_steps(spent)
                self.assertLessEqual(spent + allocated, module.BUDGET_STEPS)
                self.assertEqual(allocated % module.QUANTUM, 0)

                self.assertEqual(
                    module.budgeted_leg_steps(module.BUDGET_STEPS), 0)
                self.assertEqual(
                    module.budgeted_leg_steps(module.BUDGET_STEPS - 1), 0)
                with self.assertRaises(module.OperationalFailure):
                    module.ensure_retry_budget(module.BUDGET_STEPS)
                # 仍有一个量子时允许最后一次有界重试。
                module.ensure_retry_budget(module.BUDGET_STEPS - module.QUANTUM)

    def test_observed_attempt_delta_uses_status_and_rejects_overshoot(self):
        base, allocated = 100_000, 8_192
        for module in LEG_DRIVERS:
            with self.subTest(module=module.__name__):
                result = {"global_steps": 0, "status_steps": base + allocated}
                self.assertEqual(
                    module.observed_attempt_steps(base, result, allocated), allocated)
                result["status_steps"] += 1
                with self.assertRaises(module.OperationalFailure):
                    module.observed_attempt_steps(base, result, allocated)

        self.assertEqual(
            v30.observed_attempt_steps(base, 0, base + allocated, allocated), allocated)
        with self.assertRaises(v30.OperationalFailure):
            v30.observed_attempt_steps(base, 0, base + allocated + 1, allocated)

        # 续航/接力的历史 num_timesteps 起点不是新步预算的一部分。
        self.assertEqual(v28.observed_attempt_steps(
            v28.START,
            {"global_steps": 0, "status_steps": v28.START + allocated},
            allocated), allocated)
        self.assertEqual(v30.observed_attempt_steps(
            v30.START_NT, 0, v30.START_NT + allocated, allocated), allocated)

    def test_failed_attempts_charge_the_full_allocation(self):
        for module in ALL_DRIVERS:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.failed_attempt_charge(0, 8_192), 8_192)
                self.assertEqual(module.failed_attempt_charge(17, 8_192), 8_192)
                with self.assertRaises(module.OperationalFailure):
                    module.failed_attempt_charge(8_193, 8_192)

    def test_gcal_early_stop_uses_current_status_not_final_zip(self):
        for module in GCAL_DRIVERS:
            with self.subTest(module=module.__name__):
                result = {
                    "rc": 0,
                    "global_steps": 0,
                    "status_steps": module.QUANTUM * 10,
                    "model_fresh": False,
                }
                records = [{"step": module.QUANTUM * 9, "tripped": True}]
                self.assertTrue(module.is_gcal_stop(
                    1, result, 0, module.LEG, records))
                self.assertFalse(module.is_gcal_stop(
                    1, result, 0, module.LEG, []))
                self.assertFalse(module.is_gcal_stop(
                    2, result, 0, module.LEG, records))
                result["status_steps"] = module.LEG
                self.assertTrue(module.is_gcal_stop(
                    1, result, 0, module.LEG, records))
                result["status_steps"] += module.QUANTUM
                self.assertFalse(module.is_gcal_stop(
                    1, result, 0, module.LEG, records))

    def test_recalibration_gets_a_fresh_retry_counter(self):
        for module in GCAL_DRIVERS:
            with self.subTest(module=module.__name__):
                attempts = {1: 4, 2: 1}
                module.reset_recalibration_attempts(attempts)
                self.assertEqual(attempts, {1: 0, 2: 1})

    def test_skipped_probes_are_never_authorization_inputs(self):
        for module in (v28, v30):
            with self.subTest(module=module.__name__):
                self.assertTrue(module.candidate_probe_eligible(True))
                self.assertFalse(module.candidate_probe_eligible(False))
                self.assertFalse(module.candidate_probe_eligible(None))
        self.assertEqual(v30.missing_authorization_arms({"king": "model.zip"}), ["bc"])
        self.assertEqual(v30.missing_authorization_arms({"bc": "model.zip"}), ["king"])
        self.assertEqual(
            v30.missing_authorization_arms({"king": "k.zip", "bc": "b.zip"}), [])

    def test_operational_failure_maps_to_nonzero_exit(self):
        for module in ALL_DRIVERS:
            with self.subTest(module=module.__name__), contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(
                    module, "exclusive_lock", return_value=contextlib.nullcontext()))
                stack.enter_context(mock.patch.object(
                    module, "_main", side_effect=module.OperationalFailure("injected")))
                stack.enter_context(mock.patch.object(module, "log"))
                if hasattr(module, "attention"):
                    stack.enter_context(mock.patch.object(module, "attention"))
                with self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 2)

    def test_each_driver_holds_a_process_lock_for_the_whole_run(self):
        for module in ALL_DRIVERS:
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module, "exclusive_lock",
                                      return_value=contextlib.nullcontext()) as lock, \
                    mock.patch.object(module, "_main") as inner:
                module.main()
                lock.assert_called_once_with(
                    module.DRIVER_LOCK, module.DRIVER_LOCK_PURPOSE)
                inner.assert_called_once_with()

    def test_preregistered_file_identities_use_full_sha256(self):
        constants = (
            v26.ANCHOR_SHA, v27.ANCHOR_SHA, v28.ANCHOR_SHA, v28.BASELINE_SHA,
            v30.M29_SHA, v30.W_SHA, v30.W_NPZ_SHA, v30.SCI_SHA, v30.LAUNCH_SHA,
        )
        for digest in constants:
            self.assertEqual(len(digest), 64)
            int(digest, 16)

        # 旧的“只验文件 SHA 就放行 legacy JSON”入口必须物理删除；活动锚
        # 统一由 schema/runtime/content 绑定的 read_comparable_* 读取。
        for module in (v26, v27, v28, v30):
            self.assertFalse(hasattr(module, "read_sha_bound_json"))


if __name__ == "__main__":
    unittest.main()
