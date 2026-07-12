"""DiabloGym —— 基于 DevilutionX 的 Diablo I 强化学习环境(v0)。"""

import importlib.util
import pathlib
import sysconfig
import sys

# C++ 扩展 _diablogym 由 build.sh 产出于 ../../build/,按文件路径加载,免安装
_build_dir = pathlib.Path(__file__).resolve().parents[2] / "build"


def _load_bridge():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    exact = _build_dir / f"_diablogym{suffix}" if suffix else None
    candidates = sorted(_build_dir.glob("_diablogym*.so"))
    if exact is None or not exact.is_file():
        found = ", ".join(p.name for p in candidates) or "无"
        raise ImportError(
            f"找不到当前 Python ABI({suffix})对应的 _diablogym 扩展"
            f"(查找于 {_build_dir}；现有: {found})。请用当前解释器重新运行 build.sh"
            "；本项目当前只支持源码检出目录中的 editable install，不提供独立 wheel 运行时"
        )
    spec = importlib.util.spec_from_file_location("_diablogym", exact)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为原生扩展创建加载器: {exact}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_diablogym"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get("_diablogym") is module:
            del sys.modules["_diablogym"]
        raise
    return module


bridge = _load_bridge()

from .env import DiabloGymEnv  # noqa: E402
from .options_env import OptionsEnv, StagnationClockWrapper  # noqa: E402
from .worker_env import NumpyManager, WorkerWindowEnv  # noqa: E402

__all__ = ["bridge", "DiabloGymEnv", "OptionsEnv", "StagnationClockWrapper",
           "NumpyManager", "WorkerWindowEnv"]
