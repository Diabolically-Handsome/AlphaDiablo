"""The main MPQ must come from the explicit data_dir and must not silently fall back to cwd/system directories."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
GAME_DATA = (pathlib.Path.home() / "Library" / "Application Support"
             / "diasurgical" / "devilution")

if not any((GAME_DATA / name).is_file()
           for name in ("DIABDAT.MPQ", "diabdat.mpq", "spawn.mpq")):
    raise RuntimeError(f"no main MPQ in the test cwd: {GAME_DATA}")

with tempfile.TemporaryDirectory() as directory:
    scratch = pathlib.Path(directory)
    empty_data = scratch / "empty-data"
    saves = scratch / "saves"
    empty_data.mkdir()
    saves.mkdir()
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join(
            (str(ROOT / "python"), str(ROOT / "build"))),
        "DIABLOGY_EMPTY_DATA": str(empty_data),
        "DIABLOGY_SAVES": str(saves),
        "DIABLOGY_ASSETS": str(
            ROOT / "build" / "engine" / "devilutionx.app"
            / "Contents" / "Resources"),
    })
    code = '''\
import os
from diablogym import bridge
try:
    bridge.init(
        assets_dir=os.environ["DIABLOGY_ASSETS"],
        save_dir=os.environ["DIABLOGY_SAVES"],
        data_dir=os.environ["DIABLOGY_EMPTY_DATA"],
    )
except RuntimeError as exc:
    if "data_dir is missing" not in str(exc):
        raise
    print("ARCHIVE_ORIGIN_REJECTED")
else:
    raise AssertionError("an empty data_dir loaded the main MPQ from cwd")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=GAME_DATA, env=env,
        text=True, capture_output=True, check=False)
    if result.returncode != 0 or "ARCHIVE_ORIGIN_REJECTED" not in result.stdout:
        raise AssertionError(
            f"main MPQ origin gate failed: rc={result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}")

print("PASS: the main MPQ origin is pinned to the explicit data_dir")
