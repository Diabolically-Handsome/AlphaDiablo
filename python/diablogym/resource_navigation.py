"""Bounded recovery for the opt-in resource route's moving monster obstacles.

This module only selects ordinary commands from the latest observation. It
never steps the engine, edits occupancy, calls a policy, or declares a monster
dead. The caller must replan with real monster occupancy before EVERY decision
and execute the returned command through ``step_resource``. All execution,
including animation completion/settling, must fit ``max_microsteps`` and the
service's absolute native deadline. Legacy callers need not import this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class NavigationDecision:
    command: tuple | None
    max_microsteps: int
    reason: str


@dataclass
class ResourceNavigationRecovery:
    """Wait for moving blockers or fight an observed adjacent hostile.

    ``now`` and ``service_deadline`` are actual absolute native microstep
    clocks, not command counts. This planner cannot extend either deadline.
    A blocked episode additionally has a finite local budget and a no-progress
    fence. The returned execution budget also covers the engine settle fence.
    Only an actually replanned route clears a blocked episode; disappearance
    from the observed monster list is never treated as a kill or a clear path.
    """

    max_blocked_microsteps: int = 180
    max_no_progress_microsteps: int = 48
    attack_microsteps: int = 12
    heal_at_percent: int = 50
    active: bool = field(default=False, init=False)
    episodes: int = field(default=0, init=False)
    route_resumptions: int = field(default=0, init=False)
    _start: int | None = field(default=None, init=False)
    _last_progress: int | None = field(default=None, init=False)
    _last_now: int | None = field(default=None, init=False)
    _scene: tuple | None = field(default=None, init=False)
    _target: tuple | None = field(default=None, init=False)
    _position: tuple | None = field(default=None, init=False)
    _block_origin: tuple | None = field(default=None, init=False)
    _monster_hp: dict = field(default_factory=dict, init=False)
    _focus: tuple | None = field(default=None, init=False)
    _last_decision: NavigationDecision | None = field(default=None, init=False)
    _requested: dict = field(default_factory=dict, init=False)
    _last_request_clock: int | None = field(default=None, init=False)
    _last_reason: str = field(default="idle", init=False)

    def __post_init__(self):
        for name in ("max_blocked_microsteps", "max_no_progress_microsteps", "attack_microsteps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.heal_at_percent) is not int or not 0 < self.heal_at_percent < 100:
            raise ValueError("heal_at_percent must be an integer between 1 and 99")

    @staticmethod
    def _scene_key(raw):
        return (int(raw["dungeon_level"]), bool(raw.get("is_set_level", False)),
                int(raw.get("set_level_id", 0)))

    @staticmethod
    def _position_of(raw):
        return (int(raw.get("future_x", raw["player_x"])),
                int(raw.get("future_y", raw["player_y"])))

    @staticmethod
    def _monster_key(monster):
        # Identity remains stable while the monster walks or takes damage.
        return (int(monster["id"]), int(monster.get("type", -1)),
                int(monster.get("rnd_item_seed_hi", 0)),
                int(monster.get("rnd_item_seed_lo", 0)))

    @classmethod
    def _visible_hostiles(cls, raw):
        return [m for m in raw.get("monsters", ())
                if bool(m.get("visible", False))
                and not bool(m.get("is_invalid", False))
                and not bool(m.get("is_player_minion", False))
                and int(m.get("type", -1)) != 109
                and int(m.get("hp", 0)) > 0]

    @staticmethod
    def _instant_heals(raw):
        # Legacy raw.belt_heals can include scrolls; use real instant kinds.
        kinds = raw.get("belt_heal_kinds")
        if kinds is not None:
            return sum(int(kind) in (1, 2, 3, 4) for kind in kinds)
        state = raw.get("resource_state", {})
        return int(state.get("readiness", {}).get("belt_heals", 0))

    def _decision(self, command, budget, reason, now):
        result = NavigationDecision(command, budget, reason)
        self._last_decision = result
        self._last_reason = reason
        # Auditing selection requests is distinct from accepted/executed actions.
        # Re-reading a decision at the same real clock cannot add elapsed time.
        if command is not None and now != self._last_request_clock:
            name = command[0]
            self._requested[name] = self._requested.get(name, 0) + 1
            self._last_request_clock = now
        return result

    def next_command(self, raw: Mapping, target: tuple[int, int], *, now: int,
                     service_deadline: int, path_available: bool = False):
        """Select one bounded command after a fresh monster-aware route probe.

        Command contracts: ``('attack_monster', id, max_microsteps)`` is a native *adjacent*
        controller attack, not an unrestricted chase; ``('drink',)`` consumes a
        real instantaneous belt potion; ``('wait',)`` advances one real tick.
        ``None`` means resume the caller's path or stop for the stated reason.
        The caller must not execute a non-None command twice without stepping.
        """
        if type(now) is not int or type(service_deadline) is not int or now < 0 or service_deadline < 0:
            raise ValueError("native clocks must be nonnegative integer microsteps")
        if self._last_now is not None and now < self._last_now:
            raise ValueError("native microstep clock cannot move backwards")
        self._last_now = now
        scene = self._scene_key(raw)
        target = (int(target[0]), int(target[1]))
        if raw.get("dead") or raw.get("game_over") or raw.get("victory") or int(raw.get("hp", 0)) <= 0:
            return self._decision(None, 0, "resource_navigation_terminal", now)
        if now >= service_deadline:
            return self._decision(None, 0, "resource_service_cap", now)
        if self.active and self._scene != scene:
            self.active = False
            self._focus = None
            return self._decision(None, 0, "resource_navigation_scene_changed", now)
        if scene[0] <= 0 or scene[1]:
            return self._decision(None, 0, "resource_navigation_outside_main_dungeon", now)
        position = self._position_of(raw)
        if path_available:
            # A route probe is not an executed walk. Preserve the blocked
            # budget until the real player position has actually moved away;
            # transient path availability/rejection cannot reset a stuck actor.
            if self.active and position != self._block_origin:
                self.route_resumptions += 1
                self.active = False
                self._focus = None
            return self._decision(None, 0, "route_available", now)

        monsters = self._visible_hostiles(raw)
        health = {self._monster_key(m): (
            (int(m["hp_fixed_hi"]) << 16) + int(m["hp_fixed_lo"])
            if "hp_fixed_hi" in m and "hp_fixed_lo" in m else int(m["hp"]) * 64)
            for m in monsters}
        if not self.active:
            self.active = True
            self.episodes += 1
            self._start = self._last_progress = now
            self._scene = scene
            self._target = target
            self._position = self._block_origin = position
            self._monster_hp = health
            self._focus = None
        elif self._target != target:
            # Changing a waypoint must not reset a stuck actor's budget.
            self._target = target
        if position != self._position or any(
                key in self._monster_hp and hp < self._monster_hp[key]
                for key, hp in health.items()):
            self._last_progress = now
        self._position = position
        self._monster_hp = health
        local_left = self.max_blocked_microsteps - (now - self._start)
        progress_left = self.max_no_progress_microsteps - (now - self._last_progress)
        if local_left <= 0:
            return self._decision(None, 0, "resource_navigation_recovery_cap", now)
        if progress_left <= 0:
            return self._decision(None, 0, "resource_navigation_no_progress", now)
        remaining = min(service_deadline - now, local_left, progress_left)
        if (int(raw.get("max_hp", 0)) > 0
                and int(raw["hp"]) * 100 <= int(raw["max_hp"]) * self.heal_at_percent
                and self._instant_heals(raw) > 0):
            return self._decision(("drink",), min(1, remaining), "heal_before_blocker_combat", now)
        adjacent = [m for m in monsters if max(
            abs(int(m.get("future_x", m["x"])) - position[0]),
            abs(int(m.get("future_y", m["y"])) - position[1])) <= 1]
        focus = next((m for m in adjacent if self._monster_key(m) == self._focus), None)
        if focus is None and adjacent:
            focus = min(adjacent, key=lambda m: (
                int(m["hp"]),
                max(abs(int(m.get("future_x", m["x"])) - target[0]),
                    abs(int(m.get("future_y", m["y"])) - target[1])),
                int(m["id"])))
        if focus is not None:
            self._focus = self._monster_key(focus)
            attack_budget = min(self.attack_microsteps, remaining)
            return self._decision(("attack_monster", int(focus["id"]), attack_budget),
                                  attack_budget,
                                  "fight_adjacent_blocker", now)
        self._focus = None
        return self._decision(("wait",), min(1, remaining), "wait_for_dynamic_occupancy", now)

    def telemetry(self):
        return {"active": self.active, "episodes": self.episodes,
                "route_resumptions": self.route_resumptions,
                "reason": self._last_reason,
                "start_microstep": self._start,
                "last_progress_microstep": self._last_progress,
                "last_observed_microstep": self._last_now,
                "requested_commands": dict(self._requested),
                "max_blocked_microsteps": self.max_blocked_microsteps,
                "max_no_progress_microsteps": self.max_no_progress_microsteps}
