"""不启动游戏的资源所有权/崩溃回收回归。"""

from __future__ import annotations

import fcntl
import multiprocessing
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from diablogym.env import (  # noqa: E402
    DiabloGymEnv,
    _TEMP_SAVE_LEGACY_PREFIX,
    _TEMP_SAVE_LOCK,
    _TEMP_SAVE_PREFIX,
    _cleanup_stale_temp_save_dirs,
    _temp_save_registry_lock,
)
import diablogym.env as env_module  # noqa: E402


def _cleanup_with_registry_probe(root: str, connection) -> None:
    """Spawn target: expose the instant cleanup waits on the registry lock."""
    original = env_module._temp_save_registry_lock

    @contextmanager
    def observed(base):
        connection.send("attempt")
        if connection.recv() != "go":
            raise RuntimeError("registry probe handshake failed")
        with original(base):
            connection.send("acquired")
            yield

    env_module._temp_save_registry_lock = observed
    try:
        connection.send(("done", _cleanup_stale_temp_save_dirs(
            pathlib.Path(root))))
    except BaseException as exc:
        connection.send(("error", repr(exc)))
    finally:
        connection.close()


class TempSaveCleanupTests(unittest.TestCase):
    def test_only_unowned_temp_save_directories_are_reclaimed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            stale = root / f"{_TEMP_SAVE_PREFIX}stale"
            live = root / f"{_TEMP_SAVE_PREFIX}live"
            crashed_publish = root / f"{_TEMP_SAVE_PREFIX}half-published"
            legacy = root / f"{_TEMP_SAVE_LEGACY_PREFIX}legacy"
            for path in (stale, live, crashed_publish, legacy):
                path.mkdir()
            (stale / _TEMP_SAVE_LOCK).write_text("dead")
            live_marker = live / _TEMP_SAVE_LOCK
            live_marker.write_text("live")
            owner = open(live_marker, "a+", encoding="utf-8")
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.assertEqual(_cleanup_stale_temp_save_dirs(root), 2)
                self.assertFalse(stale.exists())
                self.assertFalse(crashed_publish.exists())
                self.assertTrue(live.exists())
                # 无 owner marker 的旧版目录没有可靠所有权证据，必须保留。
                self.assertTrue(legacy.exists())
            finally:
                fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
                owner.close()
            self.assertEqual(_cleanup_stale_temp_save_dirs(root), 1)
            self.assertFalse(live.exists())

    def test_cleanup_cannot_delete_a_half_published_live_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            publishing = root / f"{_TEMP_SAVE_PREFIX}publishing"
            publishing.mkdir()
            marker = publishing / _TEMP_SAVE_LOCK
            marker.touch()
            owner = open(marker, "a+", encoding="utf-8")
            parent, child = multiprocessing.get_context("spawn").Pipe()
            process = multiprocessing.get_context("spawn").Process(
                target=_cleanup_with_registry_probe,
                args=(str(root), child))
            try:
                with _temp_save_registry_lock(root):
                    process.start()
                    child.close()
                    self.assertTrue(parent.poll(10), "cleanup child did not start")
                    self.assertEqual(parent.recv(), "attempt")
                    parent.send("go")
                    # The cleaner has reached the exact registry acquisition
                    # point and must remain blocked until publication commits.
                    self.assertFalse(parent.poll(0.25))
                    fcntl.flock(
                        owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                self.assertTrue(parent.poll(10), "cleanup stayed blocked")
                self.assertEqual(parent.recv(), "acquired")
                self.assertTrue(parent.poll(10), "cleanup did not finish")
                self.assertEqual(parent.recv(), ("done", 0))
                process.join(10)
                self.assertEqual(process.exitcode, 0)
                self.assertTrue(publishing.exists())
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                parent.close()
                try:
                    fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
                finally:
                    owner.close()

            self.assertEqual(_cleanup_stale_temp_save_dirs(root), 1)
            self.assertFalse(publishing.exists())

    def test_async_exception_after_native_init_preserves_committed_scratch(self):
        class FakeDirectory:
            name = "/tmp/diablogym-saves-async"

            def __init__(self):
                self.cleaned = False

            def cleanup(self):
                self.cleaned = True

        class FakeLock:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        directory, lock = FakeDirectory(), FakeLock()
        assets = str((ROOT / "build" / "engine" / "devilutionx.app"
                      / "Contents" / "Resources").resolve())
        data = str((ROOT / "fake-data").resolve())
        committed = (assets, directory.name, data, 0)
        saved = (
            DiabloGymEnv._engine_initialized,
            DiabloGymEnv._engine_pid,
            DiabloGymEnv._engine_config,
            DiabloGymEnv._temp_save_dir,
            DiabloGymEnv._temp_save_lock,
        )
        try:
            DiabloGymEnv._engine_initialized = False
            DiabloGymEnv._engine_pid = None
            DiabloGymEnv._engine_config = None
            DiabloGymEnv._temp_save_dir = None
            DiabloGymEnv._temp_save_lock = None
            with mock.patch.object(
                    env_module, "_create_locked_temp_save_dir",
                    return_value=(directory, lock)), \
                    mock.patch.object(
                        env_module.bridge, "engine_config",
                        side_effect=[None, committed]), \
                    mock.patch.object(
                        env_module.bridge, "init",
                        side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    DiabloGymEnv(assets_dir=assets, data_dir=data)
            self.assertTrue(DiabloGymEnv._engine_initialized)
            self.assertEqual(DiabloGymEnv._engine_config, committed)
            self.assertFalse(directory.cleaned)
            self.assertFalse(lock.closed)
        finally:
            (DiabloGymEnv._engine_initialized,
             DiabloGymEnv._engine_pid,
             DiabloGymEnv._engine_config,
             DiabloGymEnv._temp_save_dir,
             DiabloGymEnv._temp_save_lock) = saved

    def test_normal_init_commits_initialized_flag_last(self):
        class InterruptingMeta(type):
            armed = True

            def __setattr__(cls, name, value):
                super().__setattr__(name, value)
                if (type(cls).armed and name == "_engine_pid"
                        and value == os.getpid()):
                    type(cls).armed = False
                    raise KeyboardInterrupt

        class InterruptingEnv(DiabloGymEnv, metaclass=InterruptingMeta):
            _engine_initialized = False
            _engine_pid = None
            _engine_config = None
            _active_token = None
            _temp_save_dir = None
            _temp_save_lock = None
            _atfork_registered = True

        assets = str((ROOT / "build" / "engine" / "devilutionx.app"
                      / "Contents" / "Resources").resolve())
        saves = str((ROOT / "fake-saves").resolve())
        data = str((ROOT / "fake-data").resolve())
        committed = (assets, saves, data, 0)
        with mock.patch.object(env_module, "DiabloGymEnv", InterruptingEnv), \
                mock.patch.object(env_module.bridge, "engine_config",
                                  side_effect=[None, committed]), \
                mock.patch.object(env_module.bridge, "init") as initialize:
            with self.assertRaises(KeyboardInterrupt):
                InterruptingEnv(
                    assets_dir=assets, save_dir=saves, data_dir=data)
            self.assertFalse(InterruptingEnv._engine_initialized)
            self.assertEqual(InterruptingEnv._engine_config, committed)
            self.assertEqual(InterruptingEnv._engine_pid, os.getpid())

            recovered = InterruptingEnv(
                assets_dir=assets, save_dir=saves, data_dir=data)
            self.assertTrue(InterruptingEnv._engine_initialized)
            self.assertEqual(InterruptingEnv._engine_config, committed)
            initialize.assert_called_once()

            with mock.patch.object(env_module.bridge, "reset",
                                   side_effect=KeyboardInterrupt), \
                    mock.patch.object(env_module.bridge, "end_game"):
                with self.assertRaises(KeyboardInterrupt):
                    recovered.reset(seed=7)
            self.assertIsNone(InterruptingEnv._active_token)


if __name__ == "__main__":
    unittest.main()
