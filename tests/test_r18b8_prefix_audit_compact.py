"""R18-B8 (2026-09-08): the earned-prefix audit must not grow quadratically.

The R18-B arm (r18-arm-a-loot-2) died of ENOSPC at 02:19 after the rollout-boundary audit
reached 6.2 GB (full attempt history with state dumps every 2048 learner steps).
PrefixAuditCallback.compact_ledger keeps counters and the latest attempts without state dumps;
the complete ledger is still written once at training_end.
"""
import json
import unittest

import train_ppo


def _ledger(n_attempts=10, n_openings=5):
    attempts = []
    for i in range(n_attempts):
        attempts.append({"attempt": i + 1, "status": "handoff", "microsteps": 100 * i,
                         "dive_openings": [{"window_id": j, "eligible": j % 2 == 0,
                                            "state": {"blob": "x" * 5000}} for j in range(n_openings)],
                         "windows": [{"w": j} for j in range(20)]})
    return [{"prefix_attempts": n_attempts, "prefix_microsteps": 12345, "attempts": attempts}]


class CompactLedgerTests(unittest.TestCase):
    def test_compact_drops_state_dumps_and_bounds_history(self):
        full = _ledger()
        compact = train_ppo.PrefixAuditCallback.compact_ledger(full)
        self.assertEqual(len(compact), 1)
        row = compact[0]
        self.assertEqual(row["attempts_count"], 10)
        self.assertEqual(row["prefix_microsteps"], 12345)
        self.assertLessEqual(len(row["attempts"]), train_ppo.PrefixAuditCallback._ROLLOUT_ATTEMPT_KEEP)
        self.assertEqual(row["attempts"][-1]["attempt"], 10)
        for attempt in row["attempts"]:
            self.assertNotIn("windows", attempt)
            self.assertEqual(attempt["dive_openings_count"], 5)
            for opening in attempt["dive_openings"]:
                self.assertNotIn("state", opening)
        self.assertLess(len(json.dumps(compact)), len(json.dumps(full)) // 20)

    def test_compact_leaves_the_source_ledger_untouched(self):
        full = _ledger(3, 2)
        before = json.dumps(full, sort_keys=True)
        train_ppo.PrefixAuditCallback.compact_ledger(full)
        self.assertEqual(before, json.dumps(full, sort_keys=True))

    def test_non_dict_rows_pass_through(self):
        self.assertEqual(train_ppo.PrefixAuditCallback.compact_ledger([None, 3]), [None, 3])


if __name__ == "__main__":
    unittest.main()
