"""v24/v26 分腿驱动的新评测必须绑定 schema-v2 身份。"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import eval_contract  # noqa: E402
import run_v24_legs as v24  # noqa: E402
import run_v26_legs as v26  # noqa: E402


def _snapshot() -> dict:
    files = {name: "d" * 64 for name in eval_contract.PROTOCOL_SOURCE_FILES}
    return {
        "worker": {
            "kind": "sb3_checkpoint", "path": "/frozen/worker.zip",
            "sha256": "a" * 64, "num_timesteps": 12_345,
            "gate_report_sha256": None,
        },
        "manager": {
            "kind": "numpy_policy", "path": "/frozen/manager.npz",
            "sha256": "b" * 64, "num_timesteps": None,
            "gate_report_sha256": None,
        },
        "runtime": {
            "bridge": {"path": "/frozen/bridge.so", "sha256": "c" * 64},
            "engine": {"path": "/frozen/engine.so", "sha256": "e" * 64},
            "content": {
                "game_data": {"path": "/frozen/data/DIABDAT.MPQ",
                              "sha256": "f" * 64},
                "assets": {
                    "path": "/frozen/build/engine/devilutionx.app/Contents/Resources",
                    "sha256": "1" * 64, "file_count": 2,
                },
            },
            "versions": eval_contract.runtime_versions_identity(),
            "python_protocol": {
                "files": files,
                "sha256": eval_contract.source_bundle_sha256(files),
            },
        },
    }


class LegacyLegEvalContractTests(unittest.TestCase):
    def test_exam_uses_frozen_paths_and_strict_expected_identity(self):
        snapshot = _snapshot()
        for module in (v24, v26):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as td:
                runs = pathlib.Path(td)
                output = runs / "eval-assembled" / "identity.json"
                output.parent.mkdir(parents=True)

                def fake_run(command, *_args, **_kwargs):
                    self.assertEqual(command[command.index("--worker") + 1],
                                     "/frozen/worker.zip")
                    self.assertEqual(command[command.index("--manager-npz") + 1],
                                     "/frozen/manager.npz")
                    output.write_text("{}")
                    return 0

                document = {"agg": {"ret_mean": 1.0}, "rows": []}
                with mock.patch.object(module, "RUNS", runs), \
                        mock.patch.object(module, "freeze_eval_identity",
                                          return_value=snapshot) as freeze, \
                        mock.patch.object(module, "run_process", side_effect=fake_run), \
                        mock.patch.object(module, "read_eval_archive",
                                          return_value=document) as read_archive, \
                        mock.patch.object(module, "verify_eval_identity") as verify, \
                        mock.patch.object(module, "sha16", return_value="archive-sha"):
                    result = module.exam(
                        pathlib.Path("candidate.zip"), "identity", "7000-7031",
                        **({"include_rows": True} if module is v26 else {}))

                expected_agg = {"ret_mean": 1.0, "_sha": "archive-sha"}
                if module is v26:
                    self.assertEqual(result, (expected_agg, document["rows"]))
                else:
                    self.assertEqual(result, expected_agg)
                freeze.assert_called_once_with(
                    module.ROOT, pathlib.Path("candidate"), module.DEFAULT_MANAGER_NPZ)
                expected = eval_contract.expected_eval_identity(
                    snapshot, tag="identity", seeds=range(7000, 7032))
                read_archive.assert_called_once_with(output, **expected)
                verify.assert_called_once_with(snapshot, module.ROOT)

    def test_identity_drift_voids_new_archive(self):
        snapshot = _snapshot()
        for module in (v24, v26):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as td:
                runs = pathlib.Path(td)
                output = runs / "eval-assembled" / "drift.json"
                output.parent.mkdir(parents=True)

                def fake_run(*_args, **_kwargs):
                    output.write_text("{}")
                    return 0

                with mock.patch.object(module, "RUNS", runs), \
                        mock.patch.object(module, "freeze_eval_identity",
                                          return_value=snapshot), \
                        mock.patch.object(module, "run_process", side_effect=fake_run), \
                        mock.patch.object(module, "read_eval_archive",
                                          return_value={"agg": {}, "rows": []}), \
                        mock.patch.object(
                            module, "verify_eval_identity",
                            side_effect=eval_contract.EvalContractError("drift")):
                    self.assertIsNone(module.exam(pathlib.Path("candidate.zip"),
                                                  "drift", "7-8"))

                self.assertFalse(output.exists())
                self.assertEqual(len(list(output.parent.glob("drift.*.void"))), 1)

    def test_nonzero_eval_voids_partial_archive(self):
        snapshot = _snapshot()
        for module in (v24, v26):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as td:
                runs = pathlib.Path(td)
                output = runs / "eval-assembled" / "failed.json"
                output.parent.mkdir(parents=True)

                def fake_run(*_args, **_kwargs):
                    output.write_text("partial")
                    return 9

                with mock.patch.object(module, "RUNS", runs), \
                        mock.patch.object(module, "freeze_eval_identity",
                                          return_value=snapshot), \
                        mock.patch.object(module, "run_process", side_effect=fake_run):
                    self.assertIsNone(module.exam(pathlib.Path("candidate.zip"),
                                                  "failed", "7-8"))

                self.assertFalse(output.exists())
                self.assertEqual(len(list(output.parent.glob("failed.*.void"))), 1)


if __name__ == "__main__":
    unittest.main()
