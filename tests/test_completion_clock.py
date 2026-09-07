"""Pure clock fixtures: no package initialization, native bridge or game run."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import importlib.util
from pathlib import Path
import random
import sys

import pytest


# Import this standalone stdlib module without diablogym.__init__ loading native.
_NAME = "_completion_clock_pure_test_module"
_SPEC = importlib.util.spec_from_file_location(
    _NAME, Path(__file__).resolve().parents[1] / "python/diablogym/completion_clock.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CompletionClock = _MODULE.CompletionClock
CompletionRecipe = _MODULE.CompletionRecipe
COMPLETION_L2_V1 = _MODULE.COMPLETION_L2_V1


def raw(level=1, is_set=False, **extra):
    return dict(engine_level=level, is_set_level=is_set, **extra)


def advance(clock, target, scene=None):
    scene = raw() if scene is None else scene
    for step in range(clock.state.micro_step + 1, target + 1):
        clock.observe_native_tick(step, scene, scene)
    return clock.state


def test_recipe_is_exact_frozen_and_cannot_accept_budget_overrides():
    assert COMPLETION_L2_V1.as_dict() == dict(
        protocol="completion-l2-v1", actor_denominator=6000,
        arrival_microsteps=12000, followup_microsteps=1800,
        collect_command_window_microsteps=900, service_microsteps=3000,
        farm_microsteps=3600,
    )
    with pytest.raises(FrozenInstanceError):
        COMPLETION_L2_V1.arrival_microsteps = 6000
    with pytest.raises(TypeError):
        CompletionRecipe(arrival_microsteps=6000)
    with pytest.raises(ValueError):
        CompletionClock(None)  # Legacy default None belongs to the integrator.
    detached = COMPLETION_L2_V1.as_dict()
    detached["arrival_microsteps"] = 1
    assert COMPLETION_L2_V1.arrival_microsteps == 12000


def test_late_arrival_keeps_full_followup_to_13799():
    clock = CompletionClock()
    advance(clock, 11998)
    state = clock.observe_native_tick(11999, raw(), raw(2))
    assert state.first_arrival_micro_step == 11999
    assert state.physical_deadline == 13799
    assert state.remaining_microsteps == 1800
    assert not state.budget_exhausted
    advance(clock, 13799, raw(2))
    assert clock.state.budget_exhausted
    assert clock.state.remaining_microsteps == 0


def test_early_arrival_shortens_physical_limit_without_changing_actor_denominator():
    clock = CompletionClock()
    advance(clock, 9)
    clock.observe_native_tick(10, raw(), raw(2))
    assert clock.state.physical_deadline == 1810 < 12000
    advance(clock, 1810, raw(2))
    assert clock.recipe.actor_denominator == 6000
    before = clock.state
    with pytest.raises(ValueError, match="deadline"):
        clock.observe_native_tick(1811, raw(2), raw(2))
    assert clock.state is before


def test_arrival_on_last_allowed_tick_is_valid_but_after_deadline_is_not():
    at_limit = CompletionClock()
    advance(at_limit, 11999)
    assert at_limit.observe_native_tick(12000, raw(), raw(2)).physical_deadline == 13800
    after_limit = CompletionClock()
    advance(after_limit, 12000)
    before = after_limit.state
    with pytest.raises(ValueError, match="deadline"):
        after_limit.observe_native_tick(12001, raw(), raw(2))
    assert after_limit.state is before
    assert after_limit.state.first_arrival_micro_step is None


@pytest.mark.parametrize("step", [0, -1, 2, 1.0, True, "1", None])
def test_missing_duplicate_backward_or_invalid_ticks_fail_atomically(step):
    clock = CompletionClock()
    before = clock.state
    with pytest.raises(ValueError):
        clock.observe_native_tick(step, raw(), raw(2))
    assert clock.state is before


def test_duplicate_or_backward_after_real_tick_does_not_consume_arrival():
    clock = CompletionClock()
    clock.observe_native_tick(1, raw(), raw())
    for step in (0, 1):
        before = clock.state
        with pytest.raises(ValueError):
            clock.observe_native_tick(step, raw(), raw(2))
        assert clock.state is before
    assert clock.observe_native_tick(2, raw(), raw(2)).first_arrival_micro_step == 2


def test_task_l2_does_not_arrive_but_actual_return_into_ordinary_l2_does():
    clock = CompletionClock()
    task = raw(2, True, dungeon_level=2)
    clock.observe_native_tick(1, raw(), task)
    assert clock.state.first_arrival_micro_step is None
    clock.observe_native_tick(2, task, raw(5, True, dungeon_level=2))
    assert clock.state.first_arrival_micro_step is None
    clock.observe_native_tick(3, raw(5, True, dungeon_level=2), raw(2))
    assert clock.state.first_arrival_micro_step == 3
    assert clock.state.physical_deadline == 1803


def test_conceptual_depth_and_arrival_claims_cannot_replace_real_scene_facts():
    clock = CompletionClock()
    before = clock.state
    with pytest.raises(ValueError, match="mismatch"):
        clock.observe_native_tick(1, raw(), raw(1, dungeon_level=2, arrived=True))
    assert clock.state is before
    clock.observe_native_tick(1, raw(), raw(1, dungeon_level=1, arrived=True))
    assert clock.state.first_arrival_micro_step is None
    # Being in the same main L2 already does not fabricate an entry event.
    clock.observe_native_tick(2, raw(2), raw(2))
    assert clock.state.first_arrival_micro_step is None


def test_repeated_returns_and_task_visits_never_extend_or_pause_followup():
    clock = CompletionClock()
    clock.observe_native_tick(1, raw(), raw(2))
    scenes = [raw(2), raw(1), raw(2), raw(2, True), raw(2), raw(0), raw(2)]
    for step, (source, target) in enumerate(zip(scenes, scenes[1:]), 2):
        clock.observe_native_tick(step, source, target)
        assert clock.state.first_arrival_micro_step == 1
        assert clock.state.physical_deadline == 1801
        assert clock.state.remaining_microsteps == 1801 - step


@pytest.mark.parametrize("bad", [
    {}, {"dungeon_level": 2, "is_set_level": False},
    {"engine_level": 2}, raw(2, 0), raw(True), raw(2.0), raw(-1), [], None,
])
@pytest.mark.parametrize("side", ["source", "target"])
def test_scene_validation_is_strict_and_atomic(bad, side):
    clock = CompletionClock()
    before = clock.state
    source, target = (bad, raw(2)) if side == "source" else (raw(), bad)
    with pytest.raises(ValueError, match="native scene"):
        clock.observe_native_tick(1, source, target)
    assert clock.state is before


def test_reset_clears_episode_history_and_preserves_old_detached_frozen_receipt():
    clock = CompletionClock()
    old = clock.observe_native_tick(1, raw(), raw(2))
    with pytest.raises(FrozenInstanceError):
        old.physical_deadline = 99999
    state = clock.reset()
    assert asdict(state) == dict(micro_step=0, first_arrival_micro_step=None,
                                physical_deadline=12000)
    assert old.first_arrival_micro_step == 1 and old.physical_deadline == 1801
    assert CompletionClock().state == state
    advance(clock, 4)
    assert clock.observe_native_tick(5, raw(), raw(2)).physical_deadline == 1805


def test_observation_and_native_terminal_facts_and_rng_are_not_modified():
    clock = CompletionClock()
    source = raw(hp=86, dead=False, terminated=False, truncated=False)
    target = raw(2, hp=0, dead=True, terminated=True, truncated=False,
                 resource_state={"readiness": {"ready": False}})
    originals = deepcopy((source, target))
    rng = random.getstate()
    state = clock.observe_native_tick(1, source, target)
    assert random.getstate() == rng
    assert (source, target) == originals
    assert state.first_arrival_micro_step == 1  # Real scene fact, even if dead.
    assert not any(hasattr(state, key) for key in (
        "dead", "terminated", "truncated", "success", "reward", "ready",
    ))
    # Exhaustion is a timing fact, never a fabricated terminal/native failure.
    other = CompletionClock()
    alive = raw(dead=False, hp=1, terminated=False, truncated=False)
    advance(other, 12000, alive)
    assert other.state.budget_exhausted
    assert alive == raw(dead=False, hp=1, terminated=False, truncated=False)
