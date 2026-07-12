"""选举/续判驱动的评测身份与退出码故障注入测试。"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import eval_contract  # noqa: E402
import run_v25_election as v25  # noqa: E402
import run_v29_relection as v29  # noqa: E402
import run_v30_verdict as verdict  # noqa: E402


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


class EvalDriverContractTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_object_keys(self):
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.strict_json_loads('{"agg":1,"agg":2}')

    def test_every_exam_binds_frozen_identity_and_rehashes_after_read(self):
        snapshot = _snapshot()
        cases = ((v25, "RUNS"), (v29, "EVAL"), (verdict, "EVAL"))
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for index, (module, output_attr) in enumerate(cases):
                with self.subTest(module=module.__name__), mock.patch.object(
                        module, output_attr, root):
                    tag = f"contract-{index}"
                    output = (root / "eval-assembled" / f"{tag}.json"
                              if module is v25 else root / f"{tag}.json")
                    output.parent.mkdir(parents=True, exist_ok=True)

                    def fake_run(command, *_args, **_kwargs):
                        self.assertIn("/frozen/worker.zip", command)
                        manager_index = command.index("--manager-npz") + 1
                        self.assertEqual(command[manager_index], "/frozen/manager.npz")
                        output.write_text("{}")
                        return 0

                    document = {"agg": {}, "rows": []}
                    with mock.patch.object(
                            module, "freeze_eval_identity", return_value=snapshot), \
                            mock.patch.object(module, "run", side_effect=fake_run), \
                            mock.patch.object(
                                module, "read_eval_archive", return_value=document
                            ) as read_archive, \
                            mock.patch.object(module, "verify_eval_identity") as verify, \
                            mock.patch.object(module, "sha16", return_value="archive-sha"):
                        result = module.exam(
                            "unfrozen-input", tag, "7000-7031",
                            **({} if module is verdict else {"manager_npz": "manager"}))

                    self.assertIs(result, document)
                    expected = eval_contract.expected_eval_identity(
                        snapshot, tag=tag, seeds=range(7000, 7032))
                    read_archive.assert_called_once_with(output, **expected)
                    verify.assert_called_once_with(snapshot, module.ROOT)

    def test_post_eval_identity_drift_voids_archive(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(v29, "EVAL", pathlib.Path(directory)), \
                mock.patch.object(v29, "freeze_eval_identity", return_value=snapshot), \
                mock.patch.object(v29, "read_eval_archive",
                                  return_value={"agg": {}, "rows": []}), \
                mock.patch.object(v29, "verify_eval_identity",
                                  side_effect=eval_contract.EvalContractError("drift")):
            output = pathlib.Path(directory) / "drift.json"

            def fake_run(*_args, **_kwargs):
                output.write_text("{}")
                return 0

            with mock.patch.object(v29, "run", side_effect=fake_run):
                self.assertIsNone(v29.exam("worker", "drift", "7000-7031"))
            self.assertFalse(output.exists())
            self.assertEqual(len(list(pathlib.Path(directory).glob("drift.*.void"))), 1)

    def test_operational_failure_is_nonzero_but_scientific_return_is_zero(self):
        for module in (v25, v29, verdict):
            with self.subTest(module=module.__name__), mock.patch.object(module, "log"), \
                    mock.patch.object(module, "exclusive_lock",
                                      return_value=contextlib.nullcontext()), \
                    mock.patch.object(module, "_main",
                                      side_effect=module.OperationalFailure("injected")):
                attention = (mock.patch.object(module, "attention")
                             if hasattr(module, "attention") else contextlib.nullcontext())
                with attention, self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 2)

            with self.subTest(module=f"{module.__name__}-scientific"), \
                    mock.patch.object(module, "exclusive_lock",
                                      return_value=contextlib.nullcontext()), \
                    mock.patch.object(module, "_main", return_value=None):
                self.assertIsNone(module.main())

    def test_verdict_pins_all_reused_assets_with_full_sha256(self):
        digests = (verdict.KING_ARCHIVE_SHA256, *verdict.ARM_MODEL_SHA256.values(),
                   verdict.SCI_SHA, verdict.LAUNCH_SHA, verdict.M29_SHA,
                   v25.ARCHIVE_SHA, v25.LEG7_ZIP_SHA, v25.LEG7_NPZ_SHA,
                   v25.V22H_ZIP_SHA, v29.ARCHIVE_SHA, v29.W_ZIP_SHA,
                   v29.W_NPZ_SHA)
        for digest in digests:
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_warm_export_manifest_must_bind_the_exact_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source, artifact = root / "source.zip", root / "policy.pt"
            manifest = root / "policy.pt.manifest.json"
            source.write_bytes(b"source")
            artifact.write_bytes(b"artifact")
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            payload = {
                "schema_version": 1,
                "artifact_type": "checkpoint_policy_state",
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "source_checkpoint": str(source.resolve()),
                "source_checkpoint_sha256": source_sha,
                "tensor_count": 12,
            }
            manifest.write_text(json.dumps(payload))
            patches = (
                mock.patch.object(v25, "V22H_ZIP", source),
                mock.patch.object(v25, "V22H_ZIP_SHA", source_sha),
                mock.patch.object(v25, "WARM_SD", artifact),
                mock.patch.object(v25, "WARM_MANIFEST", manifest),
            )
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                v25.validate_warm_export()
                payload["source_checkpoint_sha256"] = "0" * 64
                manifest.write_text(json.dumps(payload))
                with self.assertRaises(v25.OperationalFailure):
                    v25.validate_warm_export()


if __name__ == "__main__":
    unittest.main()
