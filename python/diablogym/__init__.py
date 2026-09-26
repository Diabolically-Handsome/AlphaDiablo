"""DiabloGym: a Diablo I reinforcement-learning environment built on DevilutionX (v0)."""

import importlib.util
import pathlib
import sysconfig
import sys

# The C++ extension _diablogym is built by build.sh into ../../build/ and loaded by file path, no install needed
_build_dir = pathlib.Path(__file__).resolve().parents[2] / "build"


def _load_bridge():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    exact = _build_dir / f"_diablogym{suffix}" if suffix else None
    candidates = sorted(_build_dir.glob("_diablogym*.so"))
    if exact is None or not exact.is_file():
        found = ", ".join(p.name for p in candidates) or "none"
        raise ImportError(
            f"No _diablogym extension found for the current Python ABI ({suffix})"
            f" (searched {_build_dir}; present: {found}). Rerun build.sh with the current interpreter"
            "; this project currently supports only an editable install in a source checkout, not a standalone wheel runtime"
        )
    spec = importlib.util.spec_from_file_location("_diablogym", exact)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create a loader for the native extension: {exact}")
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
