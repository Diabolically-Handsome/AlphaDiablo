"""旧评估入口的快照、资源回收与原子排行榜契约。"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

import evaluate as standard  # noqa: E402
import evaluate_deep as deep  # noqa: E402
import evaluate_options as options  # noqa: E402
import probe_options as probe  # noqa: E402
import eval_contract  # noqa: E402
from eval_contract import OutputReservationError  # noqa: E402


def _checkpoint(path: pathlib.Path, module: str) -> bytes:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(
            {"policy_class": {"__module__": module}}))
    return path.read_bytes()


def _oracle() -> dict:
    return {"arms": {"rush": [
        {"seed": seed, "snaps": {"3000": {"ret": float(seed)}}}
        for seed in probe.PROBE_SEEDS
    ]}}


def _contract(label: str = "test") -> dict:
    return {
        "schema_version": 1,
        "protocol_version": eval_contract.PROTOCOL_VERSION,
        "evaluator": label,
        "protocol": {"name": label, "disable_level_backtracking": True},
        "runtime": {
            "bridge": {"path": "/frozen/bridge.so", "sha256": "a" * 64},
            "engine": {"path": "/frozen/engine.so", "sha256": "b" * 64},
            "content": {
                "game_data": {"path": "/frozen/data/DIABDAT.MPQ",
                              "sha256": "c" * 64},
                "assets": {"path": "/frozen/Contents/Resources",
                           "sha256": "d" * 64, "file_count": 2},
            },
            "versions": eval_contract.runtime_versions_identity(),
            "python_protocol": {"sha256": "e" * 64, "files": {"x": "e" * 64}},
        },
        "sources": {"sha256": "f" * 64, "files": {"train/evaluate.py": "f" * 64}},
    }


class _FakeEnv:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.closed = False
        self.env = SimpleNamespace(_raw={"belt_heals": 2})
        type(self).instances.append(self)

    def reset(self, *, seed=None):
        self.seed = seed
        return [float(seed or 0)], {}

    def action_masks(self):
        return [True]

    def step(self, _action):
        return [0.0], 0.0, True, False, {
            "episode_seed": self.seed,
            "episode_extra": {"kills": 3, "depth": 2, "died": False},
            "option_extra": {"reason": "end", "mode_seq": "F", "tau": 1},
        }

    def close(self):
        self.closed = True


class _GoodModel:
    def predict(self, *_args, **_kwargs):
        return 0, None


class _FailModel:
    def predict(self, *_args, **_kwargs):
        raise RuntimeError("injected predict failure")


def _loader(model, seen: list[bytes]):
    class Loader:
        @classmethod
        def load(cls, source, *, device):
            del cls
            assert device == "cpu"
            assert isinstance(source, io.BytesIO)
            seen.append(source.getvalue())
            return model

    return Loader


class LegacyEvaluatorTests(unittest.TestCase):
    def setUp(self):
        _FakeEnv.instances.clear()
        # These unit tests replace the native environment with controlled
        # modules and exercise snapshot/order/cleanup/publication semantics.
        # A full-suite process may already have imported the real bridge in an
        # earlier module, so isolate that unrelated process-global fact here.
        # The dedicated guard test below deliberately keeps the real guard.
        if self._testMethodName != (
                "test_standalone_evaluators_reject_preloaded_or_wrong_bridge"):
            for module in (standard, deep, options, probe):
                patcher = mock.patch.object(
                    module, "require_fresh_native_runtime",
                    return_value=None)
                patcher.start()
                self.addCleanup(patcher.stop)

    def test_default_contract_freezes_before_runtime_dependency_imports(self):
        class StopModule(types.ModuleType):
            def __init__(self, name, symbol, events):
                super().__init__(name)
                self._symbol = symbol
                self._events = events

            def __getattr__(self, name):
                if name == self._symbol:
                    self._events.append("dependency-import")
                    raise RuntimeError("stop after dependency import")
                raise AttributeError(name)

        cases = (
            (standard, "main_contract", "diablogym", "DiabloGymEnv",
             lambda: standard.evaluate("model")),
            (deep, "deep_contract", "sb3_contrib", "MaskablePPO",
             lambda: deep.evaluate("model")),
            (options, "hierarchy_contract", "sb3_contrib", "MaskablePPO",
             lambda: options.evaluate("model", hier=True)),
        )
        for module, contract_name, dependency, symbol, invoke in cases:
            events = []

            def freeze():
                events.append("freeze")
                return _contract(module.__name__)

            stopper = StopModule(dependency, symbol, events)
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module, contract_name, side_effect=freeze), \
                    mock.patch.dict(sys.modules, {dependency: stopper}):
                with self.assertRaisesRegex(RuntimeError,
                                            "dependency import"):
                    invoke()
            self.assertEqual(events, ["freeze", "dependency-import"])

    def test_standalone_contract_binds_protocol_runtime_content_and_sources(self):
        contract = standard.main_contract()
        self.assertEqual(contract["protocol_version"], 4)
        self.assertTrue(contract["protocol"]["disable_level_backtracking"])
        self.assertEqual(set(contract["runtime"]),
                         {"bridge", "engine", "content", "versions",
                          "python_protocol"})
        self.assertEqual(set(contract["runtime"]["content"]),
                         {"game_data", "assets"})
        self.assertEqual(set(contract["sources"]["files"]),
                         set(standard.MAIN_SOURCE_FILES))
        standard.verify_standalone_contract(contract)

        drifted = copy.deepcopy(contract)
        drifted["runtime"]["content"]["game_data"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "MPQ/Resources"):
            standard.verify_standalone_contract(drifted)

    def test_standalone_evaluators_reject_preloaded_or_wrong_bridge(self):
        with mock.patch.dict(sys.modules, {"_diablogym": object()}):
            with self.assertRaisesRegex(RuntimeError, "未预载.*新进程"):
                standard.require_fresh_native_runtime("audit")

        wrong = types.ModuleType("_diablogym")
        wrong.__file__ = "/wrong/_diablogym.so"
        contract = _contract("native-path")
        with mock.patch.dict(sys.modules, {"_diablogym": wrong}):
            with self.assertRaisesRegex(RuntimeError, "实际加载 bridge 路径"):
                standard.verify_loaded_native_runtime(contract)

    def test_checkpoint_kind_hash_and_load_share_one_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pathlib.Path(directory) / "model.zip"
            original = _checkpoint(
                checkpoint, "sb3_contrib.common.maskable.policies")
            path, payload, digest = standard.checkpoint_snapshot(
                checkpoint.with_suffix(""))
            _checkpoint(checkpoint, "sb3_contrib.common.recurrent.policies")

            self.assertEqual(path, checkpoint.resolve())
            self.assertEqual(payload, original)
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())
            self.assertEqual(
                standard._model_kind_from_payload(payload, checkpoint), "masked")
            checkpoint.write_bytes(original)
            standard.verify_checkpoint_identity(checkpoint, digest)
            checkpoint.write_bytes(b"replaced after evaluation")
            with self.assertRaisesRegex(RuntimeError, "checkpoint 发生变化"):
                standard.verify_checkpoint_identity(checkpoint, digest)

        seen = []
        fake_diablo = types.ModuleType("diablogym")
        fake_diablo.DiabloGymEnv = _FakeEnv
        fake_sb3 = types.ModuleType("stable_baselines3")
        fake_sb3.PPO = _loader(_GoodModel(), seen)
        fake_models = types.ModuleType("models")
        frozen = b"one immutable checkpoint payload"
        contract = _contract("main")
        with mock.patch.object(
                standard, "checkpoint_snapshot",
                return_value=(pathlib.Path("/frozen/model.zip"), frozen, "a" * 64)), \
                mock.patch.object(standard, "main_contract", return_value=contract), \
                mock.patch.object(standard, "_model_kind_from_payload", return_value="ppo"), \
                mock.patch.object(standard, "verify_loaded_native_runtime"), \
                mock.patch.object(standard, "verify_standalone_contract") as verify_runtime, \
                mock.patch.object(standard, "verify_checkpoint_identity") as verify_model, \
                mock.patch.object(standard, "SEEDS", [7]), \
                mock.patch.dict(sys.modules, {
                    "diablogym": fake_diablo,
                    "stable_baselines3": fake_sb3,
                    "models": fake_models,
                }):
            result = standard.evaluate("mutable/path")
        self.assertEqual(seen, [frozen])
        self.assertEqual(result["model_sha256"], "a" * 64)
        self.assertEqual(result["contract"], contract)
        verify_runtime.assert_called_once_with(contract)
        verify_model.assert_called_once_with(
            pathlib.Path("/frozen/model.zip"), "a" * 64)
        self.assertTrue(_FakeEnv.instances[-1].closed)

    def test_deep_and_options_close_environment_on_prediction_failure(self):
        frozen = (pathlib.Path("/frozen/model.zip"), b"payload", "b" * 64)
        seen = []
        fake_sb3 = types.ModuleType("sb3_contrib")
        fake_sb3.MaskablePPO = _loader(_FailModel(), seen)
        fake_models = types.ModuleType("models")
        fake_diablo = types.ModuleType("diablogym")
        fake_diablo.DiabloGymEnv = _FakeEnv
        fake_diablo.OptionsEnv = _FakeEnv
        fake_diablo.StagnationClockWrapper = lambda env: env

        with mock.patch.dict(sys.modules, {
                "diablogym": fake_diablo, "sb3_contrib": fake_sb3,
                "models": fake_models,
        }), mock.patch.object(deep, "deep_contract", return_value=_contract("deep")), \
                mock.patch.object(deep, "checkpoint_snapshot", return_value=frozen), \
                mock.patch.object(deep, "verify_loaded_native_runtime"), \
                mock.patch.object(deep, "SEEDS", [1]):
            with self.assertRaisesRegex(RuntimeError, "predict failure"):
                deep.evaluate("model")
            self.assertTrue(_FakeEnv.instances[-1].closed)

        with mock.patch.dict(sys.modules, {
                "diablogym": fake_diablo, "sb3_contrib": fake_sb3,
        }), mock.patch.object(options, "hierarchy_contract",
                              return_value=_contract("hier")), \
                mock.patch.object(options, "checkpoint_snapshot", return_value=frozen), \
                mock.patch.object(options, "verify_loaded_native_runtime"), \
                mock.patch.object(options, "SEEDS", [1]):
            with self.assertRaisesRegex(RuntimeError, "predict failure"):
                options.evaluate("model", hier=True)
            self.assertTrue(_FakeEnv.instances[-1].closed)

    def test_deep_and_options_reverify_runtime_and_model_after_success(self):
        frozen = (pathlib.Path("/frozen/model.zip"), b"payload", "b" * 64)
        fake_sb3 = types.ModuleType("sb3_contrib")
        fake_sb3.MaskablePPO = _loader(_GoodModel(), [])
        fake_models = types.ModuleType("models")
        fake_diablo = types.ModuleType("diablogym")
        fake_diablo.DiabloGymEnv = _FakeEnv
        fake_diablo.OptionsEnv = _FakeEnv
        fake_diablo.StagnationClockWrapper = lambda env: env

        deep_contract = _contract("deep")
        with mock.patch.dict(sys.modules, {
                "diablogym": fake_diablo, "sb3_contrib": fake_sb3,
                "models": fake_models,
        }), mock.patch.object(deep, "checkpoint_snapshot", return_value=frozen), \
                mock.patch.object(deep, "SEEDS", [1]), \
                mock.patch.object(deep, "verify_loaded_native_runtime"), \
                mock.patch.object(deep, "verify_standalone_contract") as runtime_verify, \
                mock.patch.object(deep, "verify_checkpoint_identity") as model_verify:
            result = deep.evaluate("model", contract=deep_contract)
        self.assertEqual(result["contract"], deep_contract)
        runtime_verify.assert_called_once_with(deep_contract)
        model_verify.assert_called_once_with(frozen[0], frozen[2])

        hierarchy_contract = _contract("hierarchy")
        with mock.patch.dict(sys.modules, {
                "diablogym": fake_diablo, "sb3_contrib": fake_sb3,
        }), mock.patch.object(options, "checkpoint_snapshot", return_value=frozen), \
                mock.patch.object(options, "SEEDS", [1]), \
                mock.patch.object(options, "verify_loaded_native_runtime"), \
                mock.patch.object(options, "verify_standalone_contract") as runtime_verify, \
                mock.patch.object(options, "verify_checkpoint_identity") as model_verify:
            agg, _rows = options.evaluate(
                "model", hier=True, contract=hierarchy_contract)
        self.assertEqual(agg["contract"], hierarchy_contract)
        runtime_verify.assert_called_once_with(hierarchy_contract)
        model_verify.assert_called_once_with(frozen[0], frozen[2])

    def test_current_board_rejects_legacy_or_drift_and_upserts_atomically(self):
        header = "# board v4\n\n| run | score |\n|---|---|\n"
        contract = _contract("board")
        with tempfile.TemporaryDirectory() as directory:
            board = pathlib.Path(directory) / "board.md"
            legacy = "# legacy\n\n| run | score |\n|---|---|\n| alpha | 1 |\n"
            board.write_text(legacy)
            before = board.read_bytes()
            with self.assertRaisesRegex(ValueError, "旧榜只读"):
                standard.ensure_leaderboard_compatible(
                    board, contract, initial_text=header)
            self.assertEqual(board.read_bytes(), before)

            board.unlink()
            row2 = standard.model_leaderboard_row(
                "| alpha | 2 |", row_key="alpha", contract=contract,
                model_path="/frozen/a.zip", model_sha256="1" * 64, mode="ppo")
            with mock.patch.object(standard, "verify_standalone_contract"), \
                    mock.patch.object(standard, "verify_checkpoint_identity"):
                standard.upsert_leaderboard_rows(
                    board, {"alpha": row2}, contract=contract, initial_text=header)
                # 同一行的幂等重放不会生成重复项或重写文件。
                stable = board.read_bytes()
                standard.upsert_leaderboard_rows(
                    board, {"alpha": row2}, contract=contract, initial_text=header)
                self.assertEqual(board.read_bytes(), stable)
            conflicting = standard.model_leaderboard_row(
                "| alpha | 3 |", row_key="alpha", contract=contract,
                model_path="/frozen/a.zip", model_sha256="1" * 64, mode="ppo")
            with mock.patch.object(standard, "verify_standalone_contract"), \
                    mock.patch.object(standard, "verify_checkpoint_identity"):
                with self.assertRaisesRegex(ValueError, "拒绝静默覆盖"):
                    standard.upsert_leaderboard_rows(
                        board, {"alpha": conflicting}, contract=contract,
                        initial_text=header)
            text = board.read_text()
            self.assertEqual(sum(line.startswith("| alpha |")
                                 for line in text.splitlines()), 1)
            self.assertIn("| alpha | 2 |", text)
            standard.ensure_leaderboard_compatible(
                board, contract, initial_text=header)

            drifted = _contract("other-runtime")
            stable = board.read_bytes()
            with self.assertRaisesRegex(ValueError, "旧榜只读"):
                standard.ensure_leaderboard_compatible(
                    board, drifted, initial_text=header)
            self.assertEqual(board.read_bytes(), stable)

            tampered = board.read_text().replace("| alpha | 2 |", "| alpha | 999 |")
            board.write_text(tampered)
            with self.assertRaisesRegex(ValueError, "可见行"):
                standard.ensure_leaderboard_compatible(
                    board, contract, initial_text=header)
            board.write_bytes(stable)

            header_tampered = board.read_text().replace("| run | score |",
                                                        "| run | other metric |")
            board.write_text(header_tampered)
            with self.assertRaisesRegex(ValueError, "表头/列协议"):
                standard.ensure_leaderboard_compatible(
                    board, contract, initial_text=header)
            board.write_bytes(stable)

            locked_before = board.read_bytes()
            lock = board.parent / "shared-board.lock"
            with standard.exclusive_lock(lock, "test board"):
                with self.assertRaises(OutputReservationError):
                    standard.upsert_leaderboard_rows(
                        board, {"alpha": row2}, contract=contract,
                        initial_text=header, lock_path=lock)
            self.assertEqual(board.read_bytes(), locked_before)

            before = board.read_bytes()
            with mock.patch.object(standard.os, "replace",
                                   side_effect=OSError("injected replace failure")):
                with self.assertRaisesRegex(OSError, "replace failure"):
                    standard.atomic_write_text(board, "corrupt replacement")
            self.assertEqual(board.read_bytes(), before)
            self.assertEqual(list(board.parent.glob(f".{board.name}.*.tmp")), [])

    def test_standard_main_upserts_one_sha_bound_row(self):
        contract = _contract("main-board")
        result = {
            "mean": 1.0, "median": 1.0, "max": 2, "zero": "0/32",
            "depth2": 1, "secs": 0.1, "model_sha256": "c" * 64,
            "model": "/frozen/run-a/model.zip", "mode": "ppo",
            "contract": contract,
            "contract_sha256": standard.contract_sha256(contract),
        }
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(standard, "LEADERBOARD",
                                  pathlib.Path(directory) / "board.md"), \
                mock.patch.object(standard, "LEADERBOARD_LOCK",
                                  pathlib.Path(directory) / "board.lock"), \
                mock.patch.object(standard, "main_contract", return_value=contract), \
                mock.patch.object(standard, "verify_standalone_contract"), \
                mock.patch.object(standard, "verify_checkpoint_identity"), \
                mock.patch.object(standard, "evaluate", return_value=result), \
                mock.patch.object(sys, "argv", ["evaluate.py", "/tmp/run-a/model"]):
            standard.main()
            standard.main()
            text = standard.LEADERBOARD.read_text()
            self.assertEqual(sum(line.startswith("| run-a@cccccccccccccccc |")
                                 for line in text.splitlines()), 1)
            standard.ensure_leaderboard_compatible(
                standard.LEADERBOARD, contract,
                initial_text=standard.LEADERBOARD_HEADER)
            row = next(line for line in text.splitlines()
                       if line.startswith("| run-a@cccccccccccccccc |"))
            provenance = standard._validate_row_marker(row, contract)
            self.assertEqual(provenance["model_sha256"], "c" * 64)

    def test_publish_lock_rehashes_model_immediately_before_commit(self):
        contract = _contract("commit-window")
        header = "# board\n\n| run | score |\n|---|---|\n"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / "model.zip"
            model.write_bytes(b"evaluated-bytes")
            digest = hashlib.sha256(b"evaluated-bytes").hexdigest()
            row = standard.model_leaderboard_row(
                "| alpha | 1 |", row_key="alpha", contract=contract,
                model_path=str(model), model_sha256=digest, mode="ppo")
            # 更换发生在 evaluate() 的第一次复验之后、榜单 commit 之前。
            model.write_bytes(b"replacement")
            board = root / "board.md"
            with mock.patch.object(standard, "verify_standalone_contract"):
                with self.assertRaisesRegex(RuntimeError, "checkpoint 发生变化"):
                    standard.upsert_leaderboard_rows(
                        board, {"alpha": row}, contract=contract,
                        initial_text=header)
            self.assertFalse(board.exists())

    def test_main_rejects_legacy_board_before_evaluating_model(self):
        contract = _contract("main-board")
        with tempfile.TemporaryDirectory() as directory:
            board = pathlib.Path(directory) / "board.md"
            board.write_text(
                "# legacy\n\n| run | score |\n|---|---|\n| old | 1 |\n")
            with mock.patch.object(standard, "LEADERBOARD", board), \
                    mock.patch.object(standard, "main_contract", return_value=contract), \
                    mock.patch.object(standard, "evaluate") as evaluate_model, \
                    mock.patch.object(sys, "argv", ["evaluate.py", "/tmp/run/model"]):
                with self.assertRaisesRegex(ValueError, "旧榜只读"):
                    standard.main()
            evaluate_model.assert_not_called()

    def test_v4_boards_are_separate_and_hierarchy_writers_share_contract(self):
        expected = ROOT / "train" / "runs" / "eval-locks" / "leaderboard-hierarchy.lock"
        self.assertEqual(options.LB_LOCK, expected)
        self.assertEqual(probe.LB_LOCK, expected)
        self.assertEqual(options.LB, probe.LB)
        self.assertEqual(standard.LEADERBOARD.name, "leaderboard-v4.md")
        self.assertEqual(deep.LEADERBOARD.name, "leaderboard-deep-v4.md")
        self.assertEqual(options.LB.name, "leaderboard-hierarchy-v4.md")
        self.assertEqual(options.hierarchy_contract(), probe.hierarchy_contract())

        contract = _contract("hierarchy")
        agg = {
            "ret_mean": 1.0, "ret_median": 1.0, "died": 0,
            "depth_median": 1.0, "l3": 0, "kills_mean": 1.0,
            "opt_share": {}, "reasons": {}, "spiral_seqs": 0,
            "model_sha256": "d" * 64,
            "model": "/frozen/run-b/model.zip",
            "mode": "options-manager", "contract": contract,
            "contract_sha256": standard.contract_sha256(contract),
        }
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(options, "LB", pathlib.Path(directory) / "board.md"), \
                mock.patch.object(options, "hierarchy_contract", return_value=contract), \
                mock.patch.object(options, "evaluate", return_value=(agg, [])), \
                mock.patch.object(options, "upsert_leaderboard_rows") as upsert, \
                mock.patch.object(
                    sys, "argv", ["evaluate_options.py", "/tmp/run-b/model", "--options"]):
            options.main()
        self.assertEqual(upsert.call_args.kwargs["lock_path"], expected)
        self.assertEqual(upsert.call_args.kwargs["contract"], contract)
        row = next(iter(upsert.call_args.args[1].values()))
        provenance = standard._validate_row_marker(row, contract)
        self.assertEqual(provenance["model_sha256"], "d" * 64)

    def test_probe_policy_falls_back_to_first_legal_option(self):
        class HandoffEnv:
            def __init__(self, mask):
                self.mask = mask
                self.seen = []

            def reset(self, *, seed):
                self.seed = seed
                return [0.0], {}

            def action_masks(self):
                return self.mask

            def step(self, action):
                self.seen.append(int(action))
                return [0.0], 1.0, True, False, {
                    "episode_seed": self.seed,
                    "episode_extra": {
                        "kills": 0, "depth": 1, "died": False,
                    },
                    "option_extra": {
                        "reason": "end", "mode_seq": "D", "tau": 1,
                    },
                }

        env = HandoffEnv([False, True, False])
        row = probe.run_policy(
            env, lambda _env, _mask: probe.FARM, seed=7)
        self.assertEqual(env.seen, [probe.DIVE])
        self.assertEqual(row["ret"], 1.0)

        with self.assertRaisesRegex(ValueError, "动作掩码全假"):
            probe.run_policy(
                HandoffEnv([False, False, False]),
                lambda _env, _mask: probe.FARM,
                seed=8,
            )

    def test_probe_close_failure_prevents_any_publication(self):
        class CloseFailure:
            def __init__(self, **_kwargs):
                pass

            def close(self):
                raise RuntimeError("injected close failure")

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            oracle = root / "oracle.json"
            output = root / "probe.json"
            oracle.write_text(json.dumps(_oracle()))
            with mock.patch.object(probe, "OptionsEnv", CloseFailure), \
                    mock.patch.object(probe, "verify_loaded_native_runtime"), \
                    mock.patch.object(probe, "hierarchy_contract",
                                      return_value=_contract("hierarchy")), \
                    mock.patch.object(probe, "_collect_probe",
                                      return_value=({}, True, True)), \
                    mock.patch.object(probe, "_publish_probe") as publish, \
                    mock.patch.object(sys, "argv", [
                        "probe_options.py", "--oracle", str(oracle),
                        "--output", str(output),
                    ]):
                with self.assertRaisesRegex(RuntimeError, "close failure"):
                    probe.main()
            publish.assert_not_called()
            self.assertFalse(output.exists())

    def test_probe_rejects_existing_or_board_output_before_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            oracle = root / "oracle.json"
            oracle.write_text(json.dumps(_oracle()))
            existing = root / "existing.json"
            existing.write_text("historical result")
            for output, message in (
                    (existing, "拒绝覆写"),
                    (root / "board.md", "不能覆盖当前协议排行榜")):
                stderr = io.StringIO()
                with self.subTest(output=output), \
                        mock.patch.object(probe, "LB", root / "board.md"), \
                        mock.patch.object(probe, "LB_LOCK", root / "board.lock"), \
                        mock.patch.object(probe, "OptionsEnv") as env, \
                        mock.patch.object(sys, "argv", [
                            "probe_options.py", "--oracle", str(oracle),
                            "--output", str(output),
                        ]), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        probe.main()
                    self.assertEqual(raised.exception.code, 2)
                    env.assert_not_called()
                self.assertIn(message, stderr.getvalue())
                self.assertEqual(existing.read_text(), "historical result")

    def test_probe_reverifies_contract_before_publication(self):
        class CloseSuccess:
            def __init__(self, **_kwargs):
                pass

            def close(self):
                events.append("close")

        events = []
        contract = _contract("hierarchy")
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            oracle = root / "oracle.json"
            oracle.write_text(json.dumps(_oracle()))

            def freeze_contract():
                events.append("freeze")
                return contract

            def import_native_env():
                events.append("native-import")
                return CloseSuccess

            def verify(value):
                self.assertEqual(value, contract)
                events.append("verify")

            def publish(*_args):
                events.append("publish")
                return 0

            with mock.patch.object(probe, "_options_env_class",
                                   side_effect=import_native_env), \
                    mock.patch.object(probe, "verify_loaded_native_runtime"), \
                    mock.patch.object(probe, "hierarchy_contract",
                                      side_effect=freeze_contract), \
                    mock.patch.object(probe, "_collect_probe",
                                      return_value=({"meta": {
                                          "contract": contract,
                                          "contract_sha256":
                                              standard.contract_sha256(contract),
                                      }}, True, True)), \
                    mock.patch.object(probe, "verify_standalone_contract",
                                      side_effect=verify), \
                    mock.patch.object(probe, "_publish_probe",
                                      side_effect=publish), \
                    mock.patch.object(sys, "argv", [
                        "probe_options.py", "--oracle", str(oracle),
                        "--output", str(root / "probe.json"),
                    ]):
                self.assertEqual(probe.main(), 0)
        self.assertEqual(events, [
            "freeze", "native-import", "close", "verify", "publish"])

    def test_probe_output_and_script_rows_bind_v4_contract(self):
        contract = _contract("hierarchy")
        episode_rows = [
            {"seed": seed, "ret": 1.0, "died": False, "depth": 1}
            for seed in probe.EVAL_SEEDS
        ]
        with tempfile.TemporaryDirectory() as directory:
            oracle_path = pathlib.Path(directory) / "oracle.json"
            oracle_path.write_bytes(b"oracle-bytes")
            oracle_sha = hashlib.sha256(b"oracle-bytes").hexdigest()
            out = {
                "meta": {"oracle_path": str(oracle_path),
                         "oracle_sha256": oracle_sha,
                         "contract": contract,
                         "contract_sha256": standard.contract_sha256(contract)},
                "probe": {},
                "eval_refs": {name: copy.deepcopy(episode_rows)
                              for name in probe.POLICIES},
            }
            with mock.patch.object(
                    probe, "LB", pathlib.Path(directory) / "board.md"), \
                    mock.patch.object(probe, "LB_LOCK",
                                      pathlib.Path(directory) / "board.lock"), \
                    mock.patch.object(probe, "verify_standalone_contract"), \
                    mock.patch.object(standard, "verify_standalone_contract"):
                args = SimpleNamespace(output=pathlib.Path(directory) / "probe.json",
                                       write_board=True)
                self.assertEqual(probe._publish_probe(args, out, True, True), 0)
                persisted = json.loads(args.output.read_text())
                self.assertEqual(persisted["meta"]["contract_sha256"],
                                 standard.contract_sha256(contract))
                standard.ensure_leaderboard_compatible(
                    probe.LB, contract, initial_text=probe.LEADERBOARD_HEADER)
                for line in probe.LB.read_text().splitlines():
                    if "(scripted ref)" in line:
                        provenance = standard._validate_row_marker(line, contract)
                        self.assertEqual(provenance["kind"], "scripted_ref")
                        self.assertEqual(provenance["oracle_sha256"], oracle_sha)

    def test_probe_oracle_drift_prevents_diagnostic_json(self):
        contract = _contract("hierarchy")
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            oracle = root / "oracle.json"
            oracle.write_bytes(b"original")
            out = {
                "meta": {"oracle_path": str(oracle),
                         "oracle_sha256": hashlib.sha256(b"original").hexdigest(),
                         "contract": contract,
                         "contract_sha256": standard.contract_sha256(contract)},
                "probe": {}, "eval_refs": {},
            }
            oracle.write_bytes(b"replacement")
            args = SimpleNamespace(output=root / "probe.json", write_board=False)
            with mock.patch.object(probe, "verify_standalone_contract"):
                with self.assertRaisesRegex(RuntimeError, "oracle"):
                    probe._publish_probe(args, out, False, False)
            self.assertFalse(args.output.exists())

    def test_oracle_rejects_duplicate_semantic_seed(self):
        document = _oracle()
        document["arms"]["rush"].append(document["arms"]["rush"][0])
        with self.assertRaisesRegex(ValueError, "重复 seed"):
            probe.validate_oracle_rush(document)

    def test_probe_rejects_duplicate_key_oracle_before_environment_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            oracle = root / "oracle.json"
            oracle.write_text('{"arms":{},"arms":{}}')
            with mock.patch.object(probe, "OptionsEnv") as env, \
                    mock.patch.object(sys, "argv", [
                        "probe_options.py", "--oracle", str(oracle),
                        "--output", str(root / "probe.json"),
                    ]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    probe.main()
            self.assertEqual(raised.exception.code, 2)
            env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
