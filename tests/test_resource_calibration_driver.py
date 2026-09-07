"""Measurement contracts for the diagnostic route runner, without training."""
from copy import deepcopy
from concurrent.futures import Future
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import calibrate_resource_protocol as calibration


def raw(depth=1, *, ready=False, source=1, sequence=1):
    return {"dungeon_level": depth, "is_set_level": False, "player_x": 10,
        "player_y": 10, "hp": 70, "max_hp": 70, "gold": 100,
        "belt_heals": 4, "belt_heal_kinds": [1, 1, 1, 1], "equipped_items": [],
        "resource_state": {"enabled": True, "readiness": {"ready": ready,
            "failures": [] if ready else ["durability"]},
            "town": {"active_vendor": "none", "stock": []},
            "transition": {"sequence": sequence, "accepted": True,
                "source_depth": source, "target_depth": depth,
                "source_is_set": False, "target_is_set": False,
                "pretransition_ready": ready}}}


class CalibrationOutcomeTests(unittest.TestCase):
    def test_unready_real_return_is_observed_and_never_formal_success(self):
        result = calibration.calibration_outcome(stop_reason="first_town_return",
            service_start=3607, return_beat=4420, final_beat=4420, died=False,
            readiness={"ready": False}, service_cap=2400)
        self.assertTrue(result["actual_return_observed"])
        self.assertFalse(result["ready_on_return"])
        self.assertEqual(result["complete_service_microsteps"], 813)
        self.assertFalse(result["route_right_censored"])
        self.assertFalse(result["formal_metric_eligible"])
        self.assertNotIn("joint_success", result)

    def test_budget_censor_and_competing_death_are_distinct(self):
        common = dict(service_start=3600, return_beat=None, final_beat=6000,
                      readiness={"ready": False}, service_cap=2400)
        censored = calibration.calibration_outcome(stop_reason="resource_service_cap", died=False, **common)
        death = calibration.calibration_outcome(stop_reason="death", died=True, **common)
        self.assertTrue(censored["route_right_censored"])
        self.assertFalse(censored["route_competing_death"])
        self.assertFalse(death["route_right_censored"])
        self.assertTrue(death["route_competing_death"])
        self.assertIsNone(death["complete_service_microsteps"])

    def test_death_before_service_is_neither_route_censor_nor_competing_route_death(self):
        result = calibration.calibration_outcome(stop_reason="pre_service_death",
            service_start=None, return_beat=None, final_beat=1200, died=True,
            readiness={"ready": False}, service_cap=2400)
        self.assertTrue(result["pre_service_death"])
        self.assertFalse(result["route_competing_death"])
        self.assertFalse(result["route_right_censored"])
        self.assertIsNone(result["service_microsteps_observed"])

    def test_no_trigger_at_episode_cap_is_not_a_censored_route(self):
        ledger = calibration.CalibrationLedger(10000)
        service = SimpleNamespace(attempted=False, start_steps=0, reason=None)
        state = raw()
        ledger(SimpleNamespace(max_steps=6000), state, 1)
        result = ledger.finish_calibration(state, 1, service, 2400)
        self.assertEqual(result["stop_reason"], "no_service_trigger_before_episode_cap")
        self.assertIsNone(result["service_start_beat"])
        self.assertFalse(result["route_right_censored"])

    def test_first_actual_return_stops_on_same_native_microstep(self):
        ledger = calibration.CalibrationLedger(10000)
        ledger.service_object = SimpleNamespace(attempted=True, start_steps=0, reason=None)
        env = SimpleNamespace(max_steps=10000)
        ledger(env, raw(0, source=1), 1)
        state = raw(1, source=0, sequence=2)
        ledger(env, state, 2)
        self.assertEqual(env.max_steps, 2)
        self.assertEqual(ledger.stop_reason, "first_town_return")
        result = ledger.finish_calibration(state, 2, ledger.service_object, 2400)
        self.assertEqual(result["complete_service_microsteps"], 2)
        self.assertFalse(result["ready_on_return"])
        self.assertNotIn("alive_at_followup", result)
        with self.assertRaisesRegex(RuntimeError, "accounting"):
            ledger.finish_calibration(state, 3, ledger.service_object, 2400)

    def test_unexpected_l2_is_a_separate_stopping_endpoint(self):
        ledger = calibration.CalibrationLedger(10000)
        env = SimpleNamespace(max_steps=6000)
        ledger(env, raw(2, ready=True), 1)
        self.assertEqual(ledger.stop_reason, "unexpected_early_l2")
        self.assertEqual(env.max_steps, 1)

    def test_prefix_checkpoint_copies_the_exact_native_audit(self):
        ledger = calibration.CalibrationLedger(10000)
        ledger.prefix_target = 1
        state = raw()
        ledger(SimpleNamespace(max_steps=6000), state, 1)
        expected = deepcopy(ledger.production_prefix)
        ledger(SimpleNamespace(max_steps=6000), raw(sequence=2), 2)
        self.assertEqual(expected["micro_steps"], 1)
        self.assertEqual(calibration.compare_prefix(ledger.production_prefix, expected), [])
        expected["gold"] -= 1
        self.assertEqual(calibration.compare_prefix(ledger.production_prefix, expected), ["gold"])


class EconomicCaptureTests(unittest.TestCase):
    def test_walk_positions_do_not_duplicate_unchanged_market_snapshots(self):
        ledger = calibration.CalibrationLedger(10000)
        state = raw()
        first = ledger.capture_economic(state, 0, "service_start", marker=True)
        state["player_x"] += 1
        state["resource_state"]["gold_items"] = [{"x": 99, "y": 22}]
        second = ledger.capture_economic(state, 3, "walk")
        self.assertEqual(first, second)
        self.assertEqual(len(ledger.economic_snapshots), 1)
        state["gold"] -= 50
        third = ledger.capture_economic(state, 4, "buy")
        self.assertNotEqual(third, first)
        self.assertEqual(len(ledger.economic_snapshots), 2)

    def test_base_cap_expands_only_inside_started_script_and_receipts_reference_pool(self):
        for extend in (False, True):
            with self.subTest(extend=extend):
                ledger = calibration.CalibrationLedger(10000)
                ledger.extend_service_horizon = extend
                ledger.service_object = SimpleNamespace(attempted=True, start_steps=3607)
                inner = SimpleNamespace(_raw=raw(), _resource_actual_microsteps=3607, max_steps=6000)
                def command(action):
                    inner._resource_actual_microsteps += 1
                    inner._raw["gold"] -= 50
                    return inner._raw, 1, {"accepted": True, "price": 50}
                inner._execute_resource_command = command
                calibration.capture_resource_commands(inner, ledger)
                inner._execute_resource_command(("buy", "healer", 1, 2, 3, 4, 5))
                self.assertEqual(inner.max_steps, 10000 if extend else 6000)
                self.assertEqual(ledger.horizon_switch_beat, 3607 if extend else None)
                event = ledger.resource_commands[0]
                before = ledger.economic_snapshots[event["before_snapshot_id"]]
                after = ledger.economic_snapshots[event["after_snapshot_id"]]
                self.assertEqual(before["gold"] - after["gold"], 50)
                self.assertEqual(event["receipt"]["price"], 50)
                self.assertEqual(event["micro_steps"], 1)

    def test_unstarted_script_cannot_expand_the_horizon(self):
        ledger = calibration.CalibrationLedger(10000)
        ledger.extend_service_horizon = True
        ledger.service_object = SimpleNamespace(attempted=False)
        inner = SimpleNamespace(_raw=raw(), _resource_actual_microsteps=10, max_steps=6000,
                                _execute_resource_command=lambda command: None)
        calibration.capture_resource_commands(inner, ledger)
        with self.assertRaisesRegex(RuntimeError, "before the pure resource"):
            inner._execute_resource_command(("walk", 12, 13))
        self.assertEqual(inner.max_steps, 6000)


class BoundedSchedulingTests(unittest.TestCase):
    def test_results_are_written_in_job_order_with_bounded_submission(self):
        class Executor:
            def __init__(self):
                self.submitted = []
                self.first = None
            def submit(self, function, payload):
                self.submitted.append(payload["index"])
                future = Future()
                if payload["index"] == 0:
                    self.first = future
                else:
                    # Job 1 completes before job 0, then later jobs complete
                    # immediately. Output must still be 0, 1, 2, 3.
                    future.set_result({"ok": True, "row": payload["index"]})
                    if payload["index"] == 1:
                        self.first.set_result({"ok": True, "row": 0})
                return future
        executor = Executor()
        iterator = calibration.ordered_bounded_results(executor, None,
            [{"index": i} for i in range(4)], 2)
        self.assertEqual(next(iterator), 0)
        self.assertEqual(executor.submitted, [0, 1])
        self.assertEqual(list(iterator), [1, 2, 3])

    def test_later_job_failure_cancels_pending_and_never_submits_tail(self):
        class Executor:
            def __init__(self):
                self.submitted = []
                self.first = None
            def submit(self, function, payload):
                self.submitted.append(payload["index"])
                future = Future()
                if payload["index"] == 0:
                    self.first = future
                else:
                    future.set_result({"ok": False, "error": "intentional failure", "evidence": {"index": 1}})
                return future
        executor = Executor()
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            list(calibration.ordered_bounded_results(executor, None,
                 [{"index": i} for i in range(10)], 3))
        self.assertEqual(executor.submitted, [0, 1])
        self.assertTrue(executor.first.cancelled())

    def test_reused_rows_are_not_sent_to_a_policy_process(self):
        class Executor:
            def submit(self, function, payload):
                raise AssertionError("reused rows must not run an episode")
        row = {"seed": 2114000}
        actual = list(calibration.ordered_bounded_results(Executor(), None,
            [{"reused_row": row}], 1))
        self.assertEqual(actual, [row])
        self.assertIsNot(actual[0], row)


if __name__ == "__main__":
    unittest.main()
