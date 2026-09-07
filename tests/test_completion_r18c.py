"""R18-C: the second immutable completion recipe (completion-l2-r18c) and its clock semantics."""
import importlib
import sys
import types

import pytest


def _clock_module():
    """Load completion_clock without importing the diablogym package (no native bridge)."""
    name = "_r18c_completion_clock"
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType("_r18c_pkg")
    pkg.__path__ = ["/home/laure/AlphaDiablo/diablogym/python/diablogym"]
    sys.modules["_r18c_pkg"] = pkg
    module = importlib.import_module("_r18c_pkg.completion_clock")
    sys.modules[name] = module
    return module


def test_r18c_recipe_identity_and_immutability():
    m = _clock_module()
    v1, r18c = m.COMPLETION_L2_V1, m.COMPLETION_L2_R18C
    assert r18c.protocol == "completion-l2-r18c"
    assert r18c.followup_microsteps == 9000
    for key in ("actor_denominator", "arrival_microsteps", "collect_command_window_microsteps",
                "service_microsteps", "farm_microsteps"):
        assert getattr(r18c, key) == getattr(v1, key)
    assert v1.as_dict()["followup_microsteps"] == 1800 and v1.protocol == "completion-l2-v1"
    with pytest.raises(TypeError):
        m.CompletionRecipeR18C(followup_microsteps=1)
    with pytest.raises(TypeError):
        m.CompletionRecipe(followup_microsteps=9000)
    assert m.COMPLETION_PROTOCOLS == ("completion-l2-v1", "completion-l2-r18c")
    assert m.COMPLETION_RECIPES["completion-l2-r18c"] is r18c
    assert r18c != v1


def test_clock_accepts_only_registered_recipes():
    m = _clock_module()
    assert m.CompletionClock().recipe is m.COMPLETION_L2_V1
    assert m.CompletionClock(m.COMPLETION_L2_R18C).recipe is m.COMPLETION_L2_R18C
    with pytest.raises(ValueError):
        m.CompletionClock(object())
    # a bare CompletionRecipe() equals COMPLETION_L2_V1 and is therefore accepted (same as before)
    assert m.CompletionClock(m.CompletionRecipe()).recipe == m.COMPLETION_L2_V1


def _scene(level, is_set=False):
    return {"engine_level": level, "is_set_level": is_set, "dungeon_level": level}


def test_r18c_followup_is_9000_after_first_l2_arrival():
    m = _clock_module()
    clock = m.CompletionClock(m.COMPLETION_L2_R18C)
    assert clock.state.physical_deadline == 12000
    state = None
    for step in range(1, 101):
        state = clock.observe_native_tick(step, _scene(1), _scene(1))
    assert state.first_arrival_micro_step is None and state.physical_deadline == 12000
    state = clock.observe_native_tick(101, _scene(1), _scene(2))
    assert state.first_arrival_micro_step == 101
    assert state.physical_deadline == 101 + 9000
    # leaving L2 (a retreat) and re-entering never restarts the follow-up
    state = clock.observe_native_tick(102, _scene(2), _scene(1))
    state = clock.observe_native_tick(103, _scene(1), _scene(2))
    assert state.first_arrival_micro_step == 101 and state.physical_deadline == 9101
    v1 = m.CompletionClock()
    for step in range(1, 101):
        v1.observe_native_tick(step, _scene(1), _scene(1))
    assert v1.observe_native_tick(101, _scene(1), _scene(2)).physical_deadline == 101 + 1800
