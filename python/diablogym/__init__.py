"""DiabloGym —— 基于 DevilutionX 的 Diablo I 强化学习环境(v0)。"""

import importlib.util
import pathlib
import sys

# C++ 扩展 _diablogym 由 build.sh 产出于 ../../build/,按文件路径加载,免安装
_build_dir = pathlib.Path(__file__).resolve().parents[2] / "build"


def _load_bridge():
    candidates = sorted(_build_dir.glob("_diablogym*.so"))
    if not candidates:
        raise ImportError(
            f"找不到 _diablogym 扩展(查找于 {_build_dir})。先运行 diablogym/build.sh"
        )
    spec = importlib.util.spec_from_file_location("_diablogym", candidates[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules["_diablogym"] = module
    spec.loader.exec_module(module)
    return module


bridge = _load_bridge()

from .env import DiabloGymEnv  # noqa: E402
from .options_env import OptionsEnv, StagnationClockWrapper  # noqa: E402

__all__ = ["bridge", "DiabloGymEnv", "OptionsEnv", "StagnationClockWrapper"]
