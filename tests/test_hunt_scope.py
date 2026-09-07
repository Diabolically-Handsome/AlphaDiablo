"""R18-G hunt_scope: the a10 global-hunt gate per floor (pure Python, no engine)."""
import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _env_class():
    """Load env.py through a stub package so no native bridge is imported."""
    if "_r18g_pkg.env" in sys.modules:
        return sys.modules["_r18g_pkg.env"].DiabloGymEnv
    pkg = types.ModuleType("_r18g_pkg")
    pkg.__path__ = ["/home/laure/AlphaDiablo/diablogym/python/diablogym"]
    sys.modules["_r18g_pkg"] = pkg
    bridge = types.ModuleType("_r18g_pkg.bridge")
    for name in ("WM_DIABPREVLVL", "WM_DIABNEXTLVL", "WM_DIABRTNLVL"):
        setattr(bridge, name, 0)
    sys.modules["_r18g_pkg.bridge"] = bridge
    try:
        module = importlib.import_module("_r18g_pkg.env")
    except Exception as exc:  # pragma: no cover - environment without the package deps
        pytest.skip(f"env.py not importable without the package: {exc!r}")
    return module.DiabloGymEnv


def _fake(scope):
    return SimpleNamespace(hunt_scope=scope)


def test_default_scope_allows_hunt_everywhere():
    cls = _env_class()
    fake = _fake("all")
    for raw in ({"dungeon_level": 1}, {"dungeon_level": 2}, {"dungeon_level": 5, "is_set_level": False},
                {"dungeon_level": 2, "is_set_level": True}, {}):
        assert cls._hunt_allowed_here(fake, raw) is True


def test_l1_only_blocks_main_l2_plus_but_keeps_l1_and_set_levels():
    cls = _env_class()
    fake = _fake("l1-only")
    assert cls._hunt_allowed_here(fake, {"dungeon_level": 1}) is True
    assert cls._hunt_allowed_here(fake, {"dungeon_level": 0}) is True
    assert cls._hunt_allowed_here(fake, {}) is True
    assert cls._hunt_allowed_here(fake, {"dungeon_level": None}) is True
    assert cls._hunt_allowed_here(fake, {"dungeon_level": 2}) is False
    assert cls._hunt_allowed_here(fake, {"dungeon_level": 16}) is False
    assert cls._hunt_allowed_here(fake, {"dungeon_level": 2, "is_set_level": True}) is True


def test_missing_attribute_behaves_as_all():
    cls = _env_class()
    assert cls._hunt_allowed_here(SimpleNamespace(), {"dungeon_level": 3}) is True
