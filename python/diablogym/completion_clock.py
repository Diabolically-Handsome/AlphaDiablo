"""Passive physical deadlines for the explicit ``completion-l2-v1`` recipe.

The caller supplies each already completed native tick, starting at one after
the normal episode reset. ``engine_level`` is the raw export of ``currlevel``;
``dungeon_level`` is a conceptual depth and is deliberately not used here.
This module neither advances the engine nor decides death, termination,
truncation, rewards or success. Those remain facts owned by the environment.
In particular, an exhausted deadline does not imply death or failure to ever
complete the game, and an arrival does not imply survival.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from numbers import Integral


@dataclass(frozen=True)
class CompletionRecipe:
    """Immutable versioned constants; a different recipe needs a new version."""

    protocol: str = field(default="completion-l2-v1", init=False)
    actor_denominator: int = field(default=6000, init=False)
    arrival_microsteps: int = field(default=12000, init=False)
    followup_microsteps: int = field(default=1800, init=False)
    collect_command_window_microsteps: int = field(default=900, init=False)
    service_microsteps: int = field(default=3000, init=False)
    farm_microsteps: int = field(default=3600, init=False)

    def as_dict(self) -> dict:
        """Return a detached, complete identity for the training contract."""
        return asdict(self)


COMPLETION_L2_V1 = CompletionRecipe()


@dataclass(frozen=True)
class CompletionRecipeR18C(CompletionRecipe):
    """R18-C (2026-09-07): same arrival deadline and observation denominator as v1,
    follow-up widened 1800 -> 9000 so a zero-training probe can watch
    retreat -> recover -> re-descend loops. Immutable like v1.
    R18-B2/R18-B3 (2026-09-07): this is an accepted training clock -- see
    train_ppo --worker-time-protocol and eval_contract.LOOT_SERVICE_RECIPE_VERSIONS,
    which registers the sustain-loot-v1 recipe version that belongs to it."""

    protocol: str = field(default="completion-l2-r18c", init=False)
    followup_microsteps: int = field(default=9000, init=False)


COMPLETION_L2_R18C = CompletionRecipeR18C()
COMPLETION_RECIPES = {COMPLETION_L2_V1.protocol: COMPLETION_L2_V1,
                      COMPLETION_L2_R18C.protocol: COMPLETION_L2_R18C}
COMPLETION_PROTOCOLS = tuple(COMPLETION_RECIPES)


@dataclass(frozen=True)
class CompletionClockState:
    """Timing evidence only, with no synthetic native terminal flags."""

    micro_step: int
    first_arrival_micro_step: int | None
    physical_deadline: int

    @property
    def remaining_microsteps(self) -> int:
        return max(0, self.physical_deadline - self.micro_step)

    @property
    def budget_exhausted(self) -> bool:
        return self.micro_step >= self.physical_deadline


def _ordinary_l2(raw: Mapping) -> bool:
    if not isinstance(raw, Mapping):
        raise ValueError("native scene must be a mapping")
    level = raw.get("engine_level")
    is_set = raw.get("is_set_level")
    if (isinstance(level, bool) or not isinstance(level, Integral)
            or level < 0 or type(is_set) is not bool):
        raise ValueError("native scene requires integer engine_level and bool is_set_level")
    if not is_set and "dungeon_level" in raw:
        depth = raw["dungeon_level"]
        if (isinstance(depth, bool) or not isinstance(depth, Integral)
                or depth != level):
            raise ValueError("native scene ordinary engine_level/dungeon_level mismatch")
    return int(level) == 2 and not is_set


class CompletionClock:
    """One episode's passive clock, reset explicitly for every new episode.

    Observe exactly once per physical native tick. Input validation and deadline
    checks complete before changing state, so malformed or late input is atomic.
    Integrators keep the actor clock at ``recipe.actor_denominator`` and apply
    ``state.physical_deadline`` only to the environment's physical limit.
    """

    def __init__(self, recipe: CompletionRecipe = COMPLETION_L2_V1):
        registered = COMPLETION_RECIPES.get(getattr(recipe, "protocol", None))
        if registered is None or type(recipe) is not type(registered) or recipe != registered:
            raise ValueError("only the immutable completion-l2-v1 / completion-l2-r18c recipes are supported")
        self._recipe = recipe
        self.reset()

    @property
    def recipe(self) -> CompletionRecipe:
        return self._recipe

    @property
    def state(self) -> CompletionClockState:
        return self._state

    def reset(self) -> CompletionClockState:
        """Reset after normal environment reset; no engine calls or elapsed tick."""
        self._state = CompletionClockState(0, None, self.recipe.arrival_microsteps)
        return self.state

    def observe_native_tick(
        self, micro_step: int, source_raw: Mapping, target_raw: Mapping,
    ) -> CompletionClockState:
        """Record a completed tick and, at most once, an actual ordinary L2 entry.

        Arrival on the final allowed arrival tick is valid; arrival after it is
        rejected. Early arrival shortens the original physical limit to the
        same full 1800-tick follow-up. Leaving L2, quest visits and re-entry never
        restart that follow-up, which measures elapsed ticks across all scenes.
        """
        if isinstance(micro_step, bool) or not isinstance(micro_step, Integral):
            raise ValueError("micro_step must be an integer native tick")
        step = int(micro_step)
        current = self.state
        if step != current.micro_step + 1:
            raise ValueError("native ticks must be contiguous and strictly increasing")
        if step > current.physical_deadline:
            raise ValueError("native tick exceeds the active physical deadline")
        source_l2 = _ordinary_l2(source_raw)
        target_l2 = _ordinary_l2(target_raw)
        arrival = current.first_arrival_micro_step
        deadline = current.physical_deadline
        if arrival is None and target_l2 and not source_l2:
            arrival = step
            deadline = arrival + self.recipe.followup_microsteps
        self._state = CompletionClockState(step, arrival, deadline)
        return self.state
