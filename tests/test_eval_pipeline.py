"""评估/选模链路的纯函数回归测试（不启动引擎或训练）。"""

from __future__ import annotations

import copy
import hashlib
import json
import multiprocessing
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

import dashboard  # noqa: E402
import check_teacher_parity  # noqa: E402
import eval_assembled  # noqa: E402
import eval_contract  # noqa: E402
import evaluate  # noqa: E402
import run_v25_election  # noqa: E402
import run_v28_legs  # noqa: E402
import run_v30_relay  # noqa: E402


def _hold_output_reservation(path, ready, release):
    with eval_contract.reserve_output(path):
        ready.set()
        release.wait(10)


def _hold_tag_and_board_reservation(tag_path, board_lock, ready, release):
    with eval_contract.reserve_output(tag_path), \
            eval_contract.exclusive_lock(board_lock, "排行榜"):
        ready.set()
        release.wait(10)


def _runtime_content_fixture(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    data = root / "game-data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "spawn.mpq").write_bytes(b"spawn-v1")
    assets = eval_contract.default_assets_dir(root)
    (assets / "txtdata").mkdir(parents=True, exist_ok=True)
    (assets / "ASSETS_VERSION").write_bytes(b"1")
    (assets / "txtdata" / "monsters.tsv").write_bytes(b"monster\n")
    return data, assets


def _valid_v2_archive():
    rows = [
        {"seed": 7, "ret": 1.0, "depth": 1, "died": False, "kills": 2,
         "farm_n": 1, "farm_tau_mean": 2.0, "farm_tau_sum": 2,
         "farm_descend": 0, "windows": 1, "beats": 2, "overrides": 0,
         "cap": 0, "mode_seq": "F"},
        {"seed": 8, "ret": 3.0, "depth": 3, "died": True, "kills": 4,
         "farm_n": 1, "farm_tau_mean": 4.0, "farm_tau_sum": 4,
         "farm_descend": 1, "windows": 2, "beats": 4, "overrides": 1,
         "cap": 1, "mode_seq": "FD†"},
    ]
    source_files = {name: "d" * 64 for name in eval_contract.PROTOCOL_SOURCE_FILES}
    bundle_sha = eval_contract.source_bundle_sha256(source_files)
    worker = {
        "kind": "sb3_checkpoint", "path": "/frozen/worker.zip",
        "sha256": "a" * 64, "num_timesteps": 1234,
        "gate_report_sha256": None,
    }
    manager = {
        "kind": "numpy_policy", "path": "/frozen/manager.npz",
        "sha256": "b" * 64, "num_timesteps": None,
        "gate_report_sha256": None,
    }
    runtime = {
        "bridge": {"path": "/frozen/_diablogym.so", "sha256": "c" * 64},
        "engine": {"path": "/frozen/libdevilutionx.so", "sha256": "e" * 64},
        "content": {
            "game_data": {"path": "/frozen/data/DIABDAT.MPQ",
                          "sha256": "f" * 64},
            "assets": {
                "path": "/frozen/build/engine/devilutionx.app/Contents/Resources",
                "sha256": "1" * 64, "file_count": 2,
            },
        },
        "versions": eval_contract.runtime_versions_identity(),
        "python_protocol": {"sha256": bundle_sha, "files": source_files},
    }
    agg = eval_contract.recompute_agg(rows)
    agg.update({"worker_calls": 2, "worker_action_hist": {"9": 2},
                "worker_divergences": 1,
                "script_divergence_rate": 0.5})
    return {
        "schema_version": 2,
        "meta": eval_contract.make_meta(tag="audit-v2", seeds=[7, 8],
                                        worker=worker, manager=manager, runtime=runtime),
        "agg": agg,
        "rows": rows,
    }


class EvalPipelineTests(unittest.TestCase):
    def test_assembled_v3_board_is_separate_and_archive_bound(self):
        self.assertEqual(eval_assembled.LB.name, "leaderboard-assembled-v3.md")
        self.assertNotEqual(eval_assembled.LB.name, "leaderboard-hier.md")
        contract = eval_assembled.assembled_board_contract()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            archive = root / "gold.json"
            archive.write_bytes(b"frozen-eval-archive")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            key = evaluate.versioned_row_key("gold", digest)
            visible = f"| {key} | 1 | 1 | 0/32 | 2 | audit |"
            row = evaluate.assembled_leaderboard_row(
                visible, row_key=key, contract=contract,
                archive_path=str(archive), archive_sha256=digest,
                worker_sha256="a" * 64, manager_sha256="b" * 64)
            board = root / "board.md"
            with mock.patch.object(evaluate, "verify_standalone_contract"):
                evaluate.upsert_leaderboard_rows(
                    board, {key: row}, contract=contract,
                    initial_text=eval_assembled.ASSEMBLED_LEADERBOARD_HEADER)
                provenance = evaluate._validate_row_marker(
                    next(line for line in board.read_text().splitlines()
                         if line.startswith(f"| {key} |")), contract)
                self.assertEqual(provenance["kind"], "assembled")
                self.assertEqual(provenance["archive_sha256"], digest)

                archive.write_bytes(b"replacement")
                with self.assertRaisesRegex(RuntimeError, "checkpoint 发生变化"):
                    evaluate.upsert_leaderboard_rows(
                        board, {key: row}, contract=contract,
                        initial_text=eval_assembled.ASSEMBLED_LEADERBOARD_HEADER)

    def test_eval_assembled_import_does_not_preload_native_runtime(self):
        code = f"""
import pathlib, sys
root = pathlib.Path({str(ROOT)!r})
sys.path.insert(0, str(root / 'train'))
sys.path.insert(0, str(root / 'python'))
import eval_assembled
if '_diablogym' in sys.modules or 'diablogym' in sys.modules:
    raise SystemExit('eval_assembled import preloaded native runtime')
"""
        command = [sys.executable]
        if not __debug__:
            command.append("-O")
        completed = subprocess.run(
            [*command, "-c", code], cwd=ROOT, text=True,
            capture_output=True, check=False)
        self.assertEqual(
            completed.returncode, 0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}")

    def test_eval_assembled_main_rejects_a_preloaded_bridge(self):
        with mock.patch.dict(sys.modules, {"_diablogym": object()}), \
                mock.patch.object(
                    sys, "argv", ["eval_assembled.py", "--worker", "script"]):
            with self.assertRaisesRegex(
                    eval_contract.EvalContractError, "未预载.*新进程"):
                eval_assembled.main()

    def test_eval_assembled_closes_environment_on_manager_failure(self):
        class FailingManager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                raise RuntimeError("injected manager failure")

        class FakeEnv:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.closed = False
                type(self).instances.append(self)

            def reset(self, *, seed):
                return [seed], {}

            def action_masks(self):
                return [True]

            def close(self):
                self.closed = True

        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(FailingManager, FakeEnv, object())):
            with self.assertRaisesRegex(RuntimeError, "manager failure"):
                eval_assembled.evaluate(None, [7], manager_npz="manager.npz",
                                        manager_sha256="a" * 64)
        self.assertTrue(FakeEnv.instances[-1].closed)

    def test_eval_assembled_closes_on_worker_failure_and_breaks_env_cycle(self):
        class Manager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                return eval_assembled.FARM

        class WorkerEnv:
            instances = []

            def __init__(self, *_args, workers, **_kwargs):
                self.closed = False
                self._workers = workers
                self.env = types.SimpleNamespace(
                    _raw={}, action_masks=lambda: [True] * 15)
                type(self).instances.append(self)

            def reset(self, *, seed):
                return [seed], {}

            def action_masks(self):
                return [True] * 3

            def step(self, _option):
                self._workers[eval_assembled.FARM]([0.0], [True] * 15)
                raise AssertionError("worker failure should have propagated")

            def close(self):
                self.closed = True

        def failing_worker(_obs, _mask):
            raise RuntimeError("injected worker predict failure")

        caller_workers = {eval_assembled.FARM: failing_worker}
        fake_package = types.ModuleType("diablogym")
        fake_package.__path__ = []
        fake_options = types.ModuleType("diablogym.options_env")
        fake_options.dispatch = lambda *_args: 0
        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(Manager, WorkerEnv, object())), \
                mock.patch.dict(sys.modules, {
                    "diablogym": fake_package,
                    "diablogym.options_env": fake_options,
                }):
            with self.assertRaisesRegex(RuntimeError, "worker predict failure"):
                eval_assembled.evaluate(
                    caller_workers, [7], manager_npz="manager.npz",
                    manager_sha256="a" * 64)
        instance = WorkerEnv.instances[-1]
        self.assertTrue(instance.closed)
        self.assertEqual(instance._workers, {})
        self.assertIs(caller_workers[eval_assembled.FARM], failing_worker)

    def test_eval_assembled_closes_environment_after_success(self):
        class Manager:
            def __init__(self, *_args, **_kwargs):
                pass

            def choose(self, *_args):
                return eval_assembled.FARM

        class SuccessfulEnv:
            instances = []

            def __init__(self, *_args, **_kwargs):
                self.closed = False
                self.env = types.SimpleNamespace(
                    _raw={"dungeon_level": 2, "dead": False}, _ep_kills=3)
                type(self).instances.append(self)

            def reset(self, *, seed):
                return [seed], {}

            def action_masks(self):
                return [True] * 3

            def step(self, _option):
                return [0.0], 1.5, True, False, {"option_extra": {
                    "beats": 1, "overrides": 0, "reason": "done",
                    "opt": eval_assembled.FARM, "tau": 1, "mode_seq": "F",
                }}

            def close(self):
                self.closed = True

        with mock.patch.object(
                eval_assembled, "_native_runtime",
                return_value=(Manager, SuccessfulEnv, object())):
            rows, engage = eval_assembled.evaluate(
                None, [7], manager_npz="manager.npz",
                manager_sha256="a" * 64)
        self.assertIsNone(engage)
        self.assertEqual(rows[0]["ret"], 1.5)
        self.assertTrue(SuccessfulEnv.instances[-1].closed)

    def test_teacher_parity_rejects_temperature_scaled_copy(self):
        import numpy as np
        import torch

        class Teacher(torch.nn.Module):
            def forward(self, obs):
                # 15 维动作头，正比例缩放不改变 argmax，却改变温度/概率。
                return obs[:, :15]

        class Net:
            def __init__(self, scale):
                self.scale = scale

            def logits(self, obs):
                return obs[:15] * self.scale

        obs = np.random.default_rng(0).standard_normal((1000, 20)).astype(np.float32)
        exact = check_teacher_parity.parity_metrics(Teacher(), Net(1.0), obs)
        scaled = check_teacher_parity.parity_metrics(Teacher(), Net(0.5), obs)
        self.assertTrue(check_teacher_parity.parity_passes(exact))
        self.assertEqual(scaled["raw_argmax_mismatch"], 0)
        self.assertFalse(check_teacher_parity.parity_passes(scaled))

    def test_model_kind_comes_from_checkpoint_metadata(self):
        cases = {
            "masked": "sb3_contrib.common.maskable.policies",
            "recurrent": "sb3_contrib.common.recurrent.policies",
            "ppo": "stable_baselines3.common.policies",
        }
        with tempfile.TemporaryDirectory() as directory:
            for expected, module in cases.items():
                path = pathlib.Path(directory) / f"opaque-{expected}.zip"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("data", json.dumps(
                        {"policy_class": {"__module__": module}}))
                # 故意省略 .zip，覆盖 SB3 常见命令行写法。
                self.assertEqual(evaluate.model_kind(str(path.with_suffix(""))), expected)

    def test_seed_parser_and_tag_reject_unsafe_inputs(self):
        self.assertEqual(eval_assembled.parse_seeds("7-9"), [7, 8, 9])
        with self.assertRaises(Exception):
            eval_assembled.parse_seeds("9-7")
        with self.assertRaises(Exception):
            eval_assembled.parse_seeds("0-4294967296")
        with self.assertRaises(Exception):
            eval_assembled.safe_tag("../overwrite")

    def test_default_manager_identity_is_content_pinned(self):
        snapshot = eval_contract.freeze_eval_identity(ROOT, "script")
        self.assertEqual(snapshot["manager"]["sha256"],
                         eval_contract.DEFAULT_MANAGER_SHA256)

    def test_runtime_identity_binds_loaded_engine_binary(self):
        engine_path = eval_contract.engine_binary_path(ROOT)
        runtime = eval_contract.runtime_identity(
            ROOT, eval_contract.bridge_binary_path(ROOT))
        self.assertEqual(runtime["engine"]["path"], str(engine_path))
        self.assertEqual(runtime["engine"]["sha256"],
                         eval_contract.sha256_file(engine_path))
        game_data = runtime["content"]["game_data"]
        self.assertTrue(pathlib.Path(game_data["path"]).is_absolute())
        self.assertEqual(game_data["sha256"],
                         eval_contract.sha256_file(game_data["path"]))
        assets = runtime["content"]["assets"]
        self.assertEqual(assets["path"],
                         str(eval_contract.default_assets_dir(ROOT)))
        self.assertGreater(assets["file_count"], 0)
        self.assertEqual(runtime["versions"]["packages"],
                         eval_contract.RUNTIME_PACKAGE_VERSIONS)

        real_version = eval_contract.importlib.metadata.version

        def drifted_version(name):
            return "0.0.0" if name == "numpy" else real_version(name)

        with mock.patch.object(eval_contract.importlib.metadata, "version",
                               side_effect=drifted_version):
            with self.assertRaisesRegex(eval_contract.EvalContractError,
                                        "运行时版本漂移"):
                eval_contract.runtime_versions_identity()

    def test_game_data_priority_and_resources_tree_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            data = root / "data"
            data.mkdir()
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.game_data_identity(data)

            spawn = data / "spawn.mpq"
            spawn.write_bytes(b"spawn")
            identity = eval_contract.game_data_identity(data)
            self.assertEqual(pathlib.Path(identity["path"]).name, "spawn.mpq")
            self.assertEqual(identity["sha256"], hashlib.sha256(b"spawn").hexdigest())

            lower = data / "diabdat.mpq"
            lower.write_bytes(b"full-lower")
            identity = eval_contract.game_data_identity(data)
            self.assertIn(pathlib.Path(identity["path"]).name,
                          {"DIABDAT.MPQ", "diabdat.mpq"})
            self.assertEqual(identity["sha256"],
                             hashlib.sha256(b"full-lower").hexdigest())
            uppercase = data / "DIABDAT.MPQ"
            same_entry = uppercase.exists() and uppercase.samefile(lower)
            if not same_entry:  # case-sensitive filesystem
                uppercase.write_bytes(b"full-upper")
                identity = eval_contract.game_data_identity(data)
                self.assertEqual(pathlib.Path(identity["path"]).name, "DIABDAT.MPQ")
                self.assertEqual(identity["sha256"],
                                 hashlib.sha256(b"full-upper").hexdigest())

            assets = root / "Resources"
            assets.mkdir()
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.assets_tree_identity(assets)
            (assets / "z").mkdir()
            (assets / "z" / "last.bin").write_bytes(b"last")
            (assets / "first.bin").write_bytes(b"first")
            first = eval_contract.assets_tree_identity(assets)
            second = eval_contract.assets_tree_identity(assets)
            self.assertEqual(first, second)
            self.assertEqual(first["file_count"], 2)
            (assets / "z" / "last.bin").write_bytes(b"changed")
            changed = eval_contract.assets_tree_identity(assets)
            self.assertNotEqual(changed["sha256"], first["sha256"])
            (assets / "link.bin").symlink_to(assets / "first.bin")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.assets_tree_identity(assets)

    def test_post_eval_rehash_rejects_engine_binary_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for relative in eval_contract.PROTOCOL_SOURCE_FILES:
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(relative)
            bridge_path = root / "build" / eval_contract.bridge_binary_path(ROOT).name
            bridge_path.parent.mkdir(parents=True, exist_ok=True)
            bridge_path.write_bytes(b"bridge")
            engine_path = (root / "build" / "engine"
                           / eval_contract.engine_binary_path(ROOT).name)
            engine_path.parent.mkdir(parents=True, exist_ok=True)
            engine_path.write_bytes(b"engine-v1")
            data_dir, assets_dir = _runtime_content_fixture(root)
            manager_path = root / "manager.npz"
            manager_path.write_bytes(b"manager")
            runtime = eval_contract.runtime_identity(
                root, bridge_path, engine_path,
                data_dir=data_dir, assets_dir=assets_dir)
            snapshot = {
                "worker": eval_contract.script_worker_identity(
                    runtime["python_protocol"]["sha256"]),
                "manager": eval_contract.file_identity(
                    "numpy_policy", manager_path),
                "runtime": runtime,
            }
            eval_contract.verify_eval_identity(snapshot, root)
            game_path = data_dir / "spawn.mpq"
            game_path.write_bytes(b"spawn-v2")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_eval_identity(snapshot, root)
            game_path.write_bytes(b"spawn-v1")
            eval_contract.verify_eval_identity(snapshot, root)
            asset_path = assets_dir / "ASSETS_VERSION"
            asset_path.write_bytes(b"2")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_eval_identity(snapshot, root)
            asset_path.write_bytes(b"1")
            eval_contract.verify_eval_identity(snapshot, root)
            engine_path.write_bytes(b"engine-v2")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_eval_identity(snapshot, root)
            duplicate_engine = (root / "build" / "engine" / "Debug"
                                / engine_path.name)
            duplicate_engine.parent.mkdir(parents=True, exist_ok=True)
            duplicate_engine.write_bytes(b"ambiguous-engine")
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.engine_binary_path(root)

    def test_digest_uses_unrounded_tau_sum(self):
        base = {"ret": 0, "died": False, "depth": 1, "kills": 0,
                "farm_n": 1, "farm_descend": 0, "overrides": 0,
                "beats": 1, "cap": 0, "windows": 1}
        rows = [dict(base, farm_tau_mean=1.0, farm_tau_sum=1.04),
                dict(base, farm_tau_mean=1.1, farm_tau_sum=1.05)]
        self.assertEqual(eval_assembled.digest(rows)["farm_tau_mean"], 1.0)

    def test_pairing_helpers_reject_duplicate_seed_rows(self):
        rows = [{"seed": seed} for seed in range(7000, 7032)]
        self.assertEqual(len(run_v25_election.by_seed(rows)), 32)
        duplicated = rows[:-1] + [{"seed": 7030}]
        with self.assertRaises(ValueError):
            run_v25_election.by_seed(duplicated)
        with self.assertRaises(ValueError):
            run_v30_relay.by_seed(duplicated)

    def test_probe_reachability_uses_rollout_alignment(self):
        # 450,000 的首个可见 rollout 是 450,560；端点恰落这里仍然可达。
        for helper in (run_v28_legs.reachable_probes, run_v30_relay.reachable_probes):
            self.assertEqual(helper(10_000, 450_560), [260_000, 460_000])
            self.assertEqual(helper(10_000, 450_559), [260_000])
            self.assertEqual(helper(10_000, 249_999), [])

    def test_dashboard_reads_only_requested_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "progress.jsonl"
            path.write_text("".join(f'{{"i":{i}}}\n' for i in range(10_000)))
            self.assertEqual(dashboard.tail_lines(path, 3), [
                '{"i":9997}', '{"i":9998}', '{"i":9999}',
            ])

    def test_eval_v2_binds_expected_model_identity(self):
        document = _valid_v2_archive()
        expected = {
            "expected_tag": "audit-v2", "expected_seeds": [7, 8],
            "expected_worker_sha256": "a" * 64,
            "expected_manager_sha256": "b" * 64,
            "expected_worker_num_timesteps": 1234,
            "expected_engine_sha256": "e" * 64,
            "expected_game_data_path": "/frozen/data/DIABDAT.MPQ",
            "expected_game_data_sha256": "f" * 64,
            "expected_assets_path":
                "/frozen/build/engine/devilutionx.app/Contents/Resources",
            "expected_assets_sha256": "1" * 64,
            "expected_assets_file_count": 2,
        }
        eval_contract.validate_eval_archive(document, **expected)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "archive.json"
            path.write_text(json.dumps(document, allow_nan=False))
            # 落盘/重读后 histogram 的 JSON 字典键会变成字符串，也必须通过。
            eval_contract.read_eval_archive(path, **expected)
            wrong = dict(expected, expected_worker_sha256="f" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                # JSON 本身完整、seed 正确，但来自另一份 worker。
                eval_contract.read_eval_archive(path, **wrong)
            wrong_engine = dict(expected, expected_engine_sha256="f" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path, **wrong_engine)
            wrong_data = dict(expected, expected_game_data_sha256="0" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path, **wrong_data)
            wrong_assets = dict(expected, expected_assets_sha256="0" * 64)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path, **wrong_assets)
            for key, value in (
                    ("expected_game_data_path", "/other/data/DIABDAT.MPQ"),
                    ("expected_assets_path", "/other/Contents/Resources"),
                    ("expected_assets_file_count", 3)):
                with self.subTest(expected_field=key):
                    wrong_content = dict(expected, **{key: value})
                    with self.assertRaises(eval_contract.EvalContractError):
                        eval_contract.read_eval_archive(path, **wrong_content)

    def test_expected_identity_carries_full_content_contract(self):
        document = _valid_v2_archive()
        meta = document["meta"]
        expected = eval_contract.expected_eval_identity(
            {"worker": meta["worker"], "manager": meta["manager"],
             "runtime": meta["runtime"]},
            tag=meta["tag"], seeds=meta["protocol"]["seeds"])
        self.assertEqual(expected["expected_game_data_path"],
                         "/frozen/data/DIABDAT.MPQ")
        self.assertEqual(expected["expected_game_data_sha256"], "f" * 64)
        self.assertEqual(expected["expected_assets_sha256"], "1" * 64)
        self.assertEqual(expected["expected_assets_file_count"], 2)

    def test_eval_v2_recomputes_agg_and_rejects_nonfinite_rows(self):
        document = _valid_v2_archive()
        tampered = copy.deepcopy(document)
        tampered["agg"]["ret_mean"] = 1_000_000_000.0
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(tampered)

        nonfinite = copy.deepcopy(document)
        nonfinite["rows"][0]["ret"] = float("nan")
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(nonfinite)
        with self.assertRaises(ValueError):
            json.dumps(nonfinite, allow_nan=False)

        impossible_sequence = copy.deepcopy(document)
        impossible_sequence["rows"][0]["mode_seq"] = "R"
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(impossible_sequence)

        naive_timestamp = copy.deepcopy(document)
        naive_timestamp["meta"]["created_at_utc"] = "2026-07-12T12:00:00"
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(naive_timestamp)

        floating_tau = copy.deepcopy(document)
        floating_tau["rows"][0]["farm_tau_sum"] = 2.0
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(floating_tau)

        impossible_worker_calls = copy.deepcopy(document)
        impossible_worker_calls["agg"]["worker_calls"] = 7
        impossible_worker_calls["agg"]["worker_action_hist"] = {"9": 7}
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(impossible_worker_calls)

        missing_content = copy.deepcopy(document)
        del missing_content["meta"]["runtime"]["content"]
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(missing_content)

        bad_assets_count = copy.deepcopy(document)
        bad_assets_count["meta"]["runtime"]["content"]["assets"]["file_count"] = 0
        with self.assertRaises(eval_contract.EvalContractError):
            eval_contract.validate_eval_archive(bad_assets_count)

    def test_protocol_semantic_break_is_version_three(self):
        self.assertEqual(eval_contract.SCHEMA_VERSION, 2)
        self.assertEqual(eval_contract.PROTOCOL_VERSION, 3)

    def test_same_tag_reservation_is_cross_process_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "same-tag.json"
            context = multiprocessing.get_context("spawn")
            ready, release = context.Event(), context.Event()
            process = context.Process(
                target=_hold_output_reservation, args=(output, ready, release))
            process.start()
            try:
                self.assertTrue(ready.wait(10), "子进程未取得评测 reservation")
                with self.assertRaises(eval_contract.OutputReservationError):
                    with eval_contract.reserve_output(output):
                        pass
            finally:
                release.set()
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            self.assertEqual(process.exitcode, 0)
            with eval_contract.reserve_output(output):
                pass

    def test_different_tags_share_one_leaderboard_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            tag_a, tag_b = root / "tag-a.json", root / "tag-b.json"
            board_lock = root / ".leaderboard.lock"
            context = multiprocessing.get_context("spawn")
            ready, release = context.Event(), context.Event()
            process = context.Process(target=_hold_tag_and_board_reservation,
                                      args=(tag_a, board_lock, ready, release))
            process.start()
            try:
                self.assertTrue(ready.wait(10), "子进程未取得排行榜锁")
                # 不同 tag 自身并不冲突，但排行榜的 read/check/replace 必须冲突。
                with eval_contract.reserve_output(tag_b):
                    with self.assertRaises(eval_contract.OutputReservationError):
                        with eval_contract.exclusive_lock(board_lock, "排行榜"):
                            pass
            finally:
                release.set()
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            self.assertEqual(process.exitcode, 0)

    def test_bc_identity_rechecks_gate_report_after_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            weights, report = root / "policy_sd.pt", root / "bc_report.json"
            weights.write_bytes(b"weights-v1")
            report.write_text('{"data_gate":"PASS"}')
            identity = eval_contract.file_identity(
                "bc_state_dict", weights, gate_report_path=report)
            eval_contract.verify_file_identity(identity)
            report.write_text('{"data_gate":"PASS","changed":true}')
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.verify_file_identity(identity)

    def test_eval_bc_capture_uses_full_training_gate_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            weights, report = root / "policy_sd.pt", root / "bc_report.json"
            weights.write_bytes(b"weights")
            # 这组字段曾足以越过 eval_assembled 的手写子集校验；它没有
            # held-out 指标/逐类召回/demos 绑定，绝不能被命名为 PASS BC。
            report.write_text(json.dumps({
                "data_gate": "PASS",
                "policy_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "protocol_version": eval_contract.PROTOCOL_VERSION,
                "implementation_sha256": "a" * 64,
                "generator_sha256": "b" * 64,
                "manager_npz_sha256": "c" * 64,
            }))
            with self.assertRaisesRegex(ValueError, "字段/schema"):
                eval_assembled.capture_passed_bc(weights)

    def test_probe_reference_must_use_exact_seed_set(self):
        rows = [{"seed": 7, "ret": 1.0, "depth": 1, "died": False}]
        reference = [
            {"seed": 7, "ep_R": 1.0, "depth": 1, "died": False},
            {"seed": 8, "ep_R": 2.0, "depth": 1, "died": False},
        ]
        with self.assertRaises(ValueError):
            eval_assembled.compare_probe_rows(rows, reference)

    def test_legacy_archive_requires_exact_explicit_sha(self):
        legacy = {"agg": {"n": 1}, "rows": [{"seed": 7}]}
        payload = json.dumps(legacy).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy.json"
            path.write_bytes(payload)
            with self.assertRaises(eval_contract.EvalContractError):
                eval_contract.read_eval_archive(path)
            trusted = hashlib.sha256(payload).hexdigest()
            self.assertEqual(eval_contract.read_eval_archive(
                path, trusted_legacy_sha256=trusted), legacy)


if __name__ == "__main__":
    unittest.main()
