"""活动选模锚必须是当前 runtime/content 下的 schema-v2+ 档案。"""

from __future__ import annotations

import inspect
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import eval_contract  # noqa: E402
import run_v24_legs as v24  # noqa: E402
import run_v25_election as v25  # noqa: E402
import run_v26_legs as v26  # noqa: E402
import run_v27_legs as v27  # noqa: E402
import run_v28_legs as v28  # noqa: E402
import run_v29_relection as v29  # noqa: E402
import run_v30_relay as relay  # noqa: E402
import run_v30_verdict as verdict  # noqa: E402


REFERENCE_CASES = (
    (v25, "read_comparable_anchor", v25.ARCHIVE, v25.LEG7_ZIP, None,
     "v24-G3-leg7", range(7000, 7032)),
    (v26, "read_comparable_anchor", v26.ANCHOR, v26.ANCHOR_WORKER, None,
     "v24-G3-leg7", range(7000, 7032)),
    (v27, "read_comparable_anchor", v27.ANCHOR, v27.ANCHOR_WORKER, None,
     "v24-G3-leg7", range(7000, 7032)),
    (v28, "read_comparable_anchor", v28.ANCHOR, v28.ANCHOR_WORKER, None,
     "v24-G3-leg7", range(7000, 7032)),
    (v28, "read_comparable_baseline", v28.BASELINE, v28.BASE_CKPT, None,
     "v26-G3-leg6", range(7000, 7032)),
    (v29, "read_comparable_anchor", v29.ARCHIVE, v29.W_ZIP, None,
     "v28-G3-leg1", range(7000, 7032)),
    (relay, "read_comparable_science", relay.SCI_ANCHOR, relay.W_ZIP,
     relay.M29_NPZ, "v29-mfresh-full32", range(7000, 7032)),
    (relay, "read_comparable_launch", relay.LAUNCH_ANCHOR, relay.W_ZIP,
     None, "v28-G3-leg1", range(7000, 7032)),
    (relay, "read_comparable_screen", relay.SCREEN_BASE_JSON, relay.W_ZIP,
     relay.M29_NPZ, "v29-mfresh-s16", range(7000, 7016)),
    (verdict, "read_comparable_king_archive",
     verdict.EVAL / "v30-king-full32.json", verdict.ARM_MODELS["king"],
     verdict.M29_NPZ, "v30-king-full32", range(7000, 7032)),
)


class ActiveReferenceContractTests(unittest.TestCase):
    def test_every_active_reference_binds_fixed_assembly_and_current_identity(self):
        snapshot = {"worker": {}, "manager": {}, "runtime": {}}
        document = {"schema_version": 999, "agg": {}, "rows": []}
        for module, helper_name, path, worker, manager, tag, seeds in REFERENCE_CASES:
            with self.subTest(module=module.__name__, helper=helper_name), \
                    mock.patch.object(
                        module, "freeze_eval_identity", return_value=snapshot
                    ) as freeze, \
                    mock.patch.object(
                        module, "expected_eval_identity", return_value={"bound": tag}
                    ) as expected, \
                    mock.patch.object(
                        module, "read_eval_archive", return_value=document
                    ) as read_archive, \
                    mock.patch.object(module, "verify_eval_identity") as verify:
                result = getattr(module, helper_name)()

            self.assertIs(result, document)
            freeze.assert_called_once_with(module.ROOT, worker, manager)
            expected.assert_called_once_with(snapshot, tag=tag, seeds=seeds)
            read_archive.assert_called_once_with(path, bound=tag)
            verify.assert_called_once_with(snapshot, module.ROOT)

    def test_legacy_or_wrong_identity_reference_fails_with_rebaseline_message(self):
        for module, helper_name, *_ in REFERENCE_CASES:
            with self.subTest(module=module.__name__, helper=helper_name), \
                    mock.patch.object(module, "freeze_eval_identity",
                                      return_value={"worker": {}, "manager": {},
                                                    "runtime": {}}), \
                    mock.patch.object(module, "expected_eval_identity",
                                      return_value={}), \
                    mock.patch.object(
                        module, "read_eval_archive",
                        side_effect=eval_contract.EvalContractError("legacy")
                    ):
                with self.assertRaises(module.OperationalFailure) as raised:
                    getattr(module, helper_name)()
            self.assertIn("环境语义变更后须", str(raised.exception))

    def test_no_driver_grants_legacy_trust_to_active_archives(self):
        for module in (v24, v25, v26, v27, v28, v29, relay, verdict):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("trusted_legacy_sha256", source)
                self.assertNotIn("read_sha_bound_json", source)

    def test_all_static_threshold_drivers_fail_closed_after_protocol_change(self):
        for module in (v24, v25, v26, v27, v28, v29, relay, verdict):
            with self.subTest(module=module.__name__), mock.patch.object(
                    module, "PROTOCOL_VERSION",
                    module.CALIBRATED_PROTOCOL_VERSION + 1):
                with self.assertRaises(module.OperationalFailure) as raised:
                    module.require_calibrated_protocol()
            self.assertIn("重跑 protocol-v3 基线", str(raised.exception))

    def test_protocol_gate_is_wired_before_every_driver_preflight(self):
        for module in (v24, v25, v26, v27, v28, v29, relay):
            failure = module.OperationalFailure("protocol gate sentinel")
            with self.subTest(module=module.__name__), mock.patch.object(
                    module, "require_calibrated_protocol",
                    side_effect=failure) as gate:
                with self.assertRaises(module.OperationalFailure) as raised:
                    module.preflight()
            self.assertIs(raised.exception, failure)
            gate.assert_called_once_with()

        failure = verdict.OperationalFailure("protocol gate sentinel")
        with mock.patch.object(
                verdict, "require_calibrated_protocol",
                side_effect=failure) as gate:
            with self.assertRaises(verdict.OperationalFailure) as raised:
                verdict._main()
        self.assertIs(raised.exception, failure)
        gate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
