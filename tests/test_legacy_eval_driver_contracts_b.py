"""v27/v28/v30-relay 新评测入口的 schema-v2 身份契约测试。"""

from __future__ import annotations

import contextlib
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import eval_contract  # noqa: E402
import run_v27_legs as v27  # noqa: E402
import run_v28_legs as v28  # noqa: E402
import run_v30_relay as relay  # noqa: E402


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
            "sha256": v27.DEFAULT_MANAGER_SHA, "num_timesteps": None,
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


class LegacyEvalDriverContractsBTests(unittest.TestCase):
    CASES = (v27, v28, relay)

    def _patch_output(self, stack: contextlib.ExitStack, module, base: pathlib.Path,
                      tag: str) -> pathlib.Path:
        if module is v27:
            runs = base / "runs"
            output = runs / "eval-assembled" / f"{tag}.json"
            stack.enter_context(mock.patch.object(module, "RUNS", runs))
            stack.enter_context(mock.patch.object(module, "V24", base / "driver"))
        else:
            eval_dir = base / "eval-assembled"
            output = eval_dir / f"{tag}.json"
            stack.enter_context(mock.patch.object(module, "EVAL", eval_dir))
            driver_attr = "V28" if module is v28 else "V30"
            stack.enter_context(mock.patch.object(module, driver_attr, base / "driver"))
        output.parent.mkdir(parents=True, exist_ok=True)
        if module is relay:
            stack.enter_context(mock.patch.object(module, "require_sha256"))
        return output

    def test_all_exams_use_frozen_paths_and_validate_v2_archive(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            for index, module in enumerate(self.CASES):
                with self.subTest(module=module.__name__), contextlib.ExitStack() as stack:
                    tag = f"identity-{index}"
                    output = self._patch_output(stack, module, base / str(index), tag)
                    model = pathlib.Path("unfrozen/model.zip")

                    def fake_run(command, *_args, **_kwargs):
                        self.assertEqual(command[command.index("--worker") + 1],
                                         "/frozen/worker.zip")
                        self.assertEqual(command[command.index("--manager-npz") + 1],
                                         "/frozen/manager.npz")
                        output.write_text("{}")
                        return 0

                    document = {"agg": {"ret_mean": 1.0},
                                "rows": [{"seed": 7}, {"seed": 8}]}
                    freeze = stack.enter_context(mock.patch.object(
                        module, "freeze_eval_identity", return_value=snapshot))
                    runner_name = "run_process" if module in (v27, v28) else "run"
                    stack.enter_context(mock.patch.object(
                        module, runner_name, side_effect=fake_run))
                    read_archive = stack.enter_context(mock.patch.object(
                        module, "read_eval_archive", return_value=document))
                    verify = stack.enter_context(mock.patch.object(
                        module, "verify_eval_identity"))
                    stack.enter_context(mock.patch.object(
                        module, "sha16", return_value="archive-sha"))

                    result = module.exam(
                        model, tag, "7-8",
                        **({"include_rows": True} if module is v27 else {}))

                    manager = module.M29_NPZ if module is relay else None
                    freeze.assert_called_once_with(module.ROOT, model, manager)
                    expected = eval_contract.expected_eval_identity(
                        snapshot, tag=tag, seeds=[7, 8])
                    read_archive.assert_called_once_with(output, **expected)
                    verify.assert_called_once_with(snapshot, module.ROOT)
                    self.assertEqual(document["agg"]["_sha"], "archive-sha")
                    if module is v27:
                        self.assertIs(result[0], document["agg"])
                        self.assertIs(result[1], document["rows"])
                    elif module is v28:
                        self.assertIs(result[0], document["agg"])
                        self.assertIs(result[1], document["rows"])
                    else:
                        self.assertIs(result, document)

    def test_identity_drift_voids_each_new_archive(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            for index, module in enumerate(self.CASES):
                with self.subTest(module=module.__name__), contextlib.ExitStack() as stack:
                    tag = f"drift-{index}"
                    output = self._patch_output(stack, module, base / str(index), tag)

                    def fake_run(*_args, **_kwargs):
                        output.write_text("{}")
                        return 0

                    stack.enter_context(mock.patch.object(
                        module, "freeze_eval_identity", return_value=snapshot))
                    runner_name = "run_process" if module in (v27, v28) else "run"
                    stack.enter_context(mock.patch.object(
                        module, runner_name, side_effect=fake_run))
                    stack.enter_context(mock.patch.object(
                        module, "read_eval_archive",
                        return_value={"agg": {}, "rows": []}))
                    stack.enter_context(mock.patch.object(
                        module, "verify_eval_identity",
                        side_effect=eval_contract.EvalContractError("drift")))

                    self.assertIsNone(module.exam(pathlib.Path("model.zip"), tag, "7-8"))
                    self.assertFalse(output.exists())
                    self.assertEqual(len(list(output.parent.glob(f"{tag}.*.void"))), 1)

    def test_nonzero_eval_exit_voids_partial_archive(self):
        snapshot = _snapshot()
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            for index, module in enumerate(self.CASES):
                with self.subTest(module=module.__name__), contextlib.ExitStack() as stack:
                    tag = f"failed-{index}"
                    output = self._patch_output(stack, module, base / str(index), tag)

                    def fake_run(*_args, **_kwargs):
                        output.write_text("partial")
                        return 17

                    stack.enter_context(mock.patch.object(
                        module, "freeze_eval_identity", return_value=snapshot))
                    runner_name = "run_process" if module in (v27, v28) else "run"
                    stack.enter_context(mock.patch.object(
                        module, runner_name, side_effect=fake_run))
                    read_archive = stack.enter_context(mock.patch.object(
                        module, "read_eval_archive"))

                    self.assertIsNone(module.exam(pathlib.Path("model.zip"), tag, "7-8"))
                    read_archive.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertEqual(len(list(output.parent.glob(f"{tag}.*.void"))), 1)

    def test_seed_ranges_are_strictly_parsed_before_freezing(self):
        bad_values = (None, "", "-1-2", "2-1", "1-2-3", " 1-2", "1-2 ")
        for module in self.CASES:
            for value in bad_values:
                with self.subTest(module=module.__name__, seeds=value):
                    with self.assertRaises(ValueError):
                        module.parse_seed_range(value)
            self.assertEqual(module.parse_seed_range("0-2"), [0, 1, 2])

        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            for index, module in enumerate(self.CASES):
                with self.subTest(module=f"{module.__name__}-exam"), \
                        contextlib.ExitStack() as stack:
                    self._patch_output(stack, module, base / str(index), "bad-seeds")
                    freeze = stack.enter_context(mock.patch.object(
                        module, "freeze_eval_identity"))
                    runner_name = "run_process" if module in (v27, v28) else "run"
                    runner = stack.enter_context(mock.patch.object(module, runner_name))
                    self.assertIsNone(module.exam(
                        pathlib.Path("model.zip"), "bad-seeds", "2-1"))
                    freeze.assert_not_called()
                    runner.assert_not_called()

    def test_default_manager_hash_remains_pinned_when_path_is_explicit(self):
        snapshot = _snapshot()
        snapshot["manager"]["sha256"] = "e" * 64
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            for index, module in enumerate((v27, v28)):
                with self.subTest(module=module.__name__), contextlib.ExitStack() as stack:
                    self._patch_output(stack, module, base / str(index), "manager-drift")
                    stack.enter_context(mock.patch.object(
                        module, "freeze_eval_identity", return_value=snapshot))
                    runner = stack.enter_context(mock.patch.object(module, "run_process"))
                    self.assertIsNone(module.exam(
                        pathlib.Path("model.zip"), "manager-drift", "7-8"))
                    runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
