"""R18-A retreat-v1: the scripted return-to-town interface for main L2+.

Chairman's diagnosis (2026-09-06): the manager has no word for "retreat" and
the worker has no hands for it, so on L2 the pair fights until death. This
module gives the pair the missing interface with the smallest possible change:

* the manager keeps its three words; on main L2+ the RESUPPLY word is served
  by this script instead of the town service (the mask collapses to RESUPPLY
  while a retreat is active, exactly like the town trip);
* the script walks to the up-stairs, drinks on the way when the belt allows,
  ascends one floor (engine receipt ``retreat_ascent``) and hands control back
  on the floor above, where the ordinary coach and town service continue
  (each completed retreat earns one extra town trip under the loot economy);
* a failed retreat never ends the episode: the script hands control back and
  the pair keeps fighting (``("complete",)``; ``("finish", ...)`` is reserved
  for the town service's fail-closed itinerary).

``resource_retreat="off"`` (the default) constructs no service and leaves every
frozen path byte-identical.
"""
from dataclasses import dataclass, field

from .resource_protocol import (
    RESOURCE_RETREAT_PROTOCOLS, validate_retreat_protocol, validate_native_retreat)

RETREAT_SERVICE_CAP = 900          # microsteps per retreat attempt
RETREAT_MAX_PER_EPISODE = 3        # mirrors MaxResourceRetreats in the engine
RETREAT_FAILURE_COOLDOWN = 300     # microsteps before a failed attempt may be retried
RETREAT_DRINK_SPACING = 20         # microsteps between script-owned drinks
RETREAT_WALK_REJECTIONS = 8        # consecutive rejected walks before giving up

__all__ = ["RESOURCE_RETREAT_PROTOCOLS", "RetreatPolicy", "RetreatService",
           "validate_retreat_protocol", "validate_native_retreat",
           "RETREAT_SERVICE_CAP", "RETREAT_MAX_PER_EPISODE"]


@dataclass(frozen=True)
class RetreatPolicy:
    """The manager-side retreat law (ruling 3: HP feeds only drinking and the
    return-to-town decision — this is that decision)."""
    hp_fraction: float = 0.5            # retreat when hp <= hp_fraction * max_hp
    empty_belt_hp_fraction: float = 0.75  # ... or the belt is empty and hp <= this
    pressure_radius: int = 6
    pressure_count: int = 0             # 0 = pressure trigger off
    drink_hp_fraction: float = 0.4      # drink while retreating below this

    def __post_init__(self):
        for name in ("hp_fraction", "empty_belt_hp_fraction", "drink_hp_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1]")
        if type(self.pressure_radius) is not int or self.pressure_radius < 0:
            raise ValueError("pressure_radius must be a non-negative int")
        if type(self.pressure_count) is not int or self.pressure_count < 0:
            raise ValueError("pressure_count must be a non-negative int")

    def as_dict(self):
        return {"hp_fraction": self.hp_fraction,
                "empty_belt_hp_fraction": self.empty_belt_hp_fraction,
                "pressure_radius": self.pressure_radius,
                "pressure_count": self.pressure_count,
                "drink_hp_fraction": self.drink_hp_fraction}


def belt_heals(raw):
    kinds = raw.get("belt_heal_kinds")
    if kinds is not None:
        return sum(int(value) in (1, 2, 3, 4) for value in kinds)
    return int(raw.get("resource_state", {}).get("readiness", {}).get("belt_heals", 0))


def alive_monsters_within(raw, radius):
    px, py = int(raw["player_x"]), int(raw["player_y"])
    count = 0
    for monster in raw.get("monsters", ()):
        if int(monster.get("hp", 0)) <= 0 or int(monster.get("type", -1)) == 109:
            continue
        if bool(monster.get("is_invalid", False)):
            continue
        if "x" not in monster or "y" not in monster:
            continue
        if max(abs(int(monster["x"]) - px), abs(int(monster["y"]) - py)) <= radius:
            count += 1
    return count


def retreats_started(raw):
    """Engine counter of COMPLETED retreats (incremented on arrival one floor up,
    not at authorization); the key name is historical."""
    return int(raw.get("resource_state", {}).get("retreats_started", 0) or 0)


@dataclass
class RetreatService:
    policy: RetreatPolicy = field(default_factory=RetreatPolicy)
    attempted: bool = False
    active: bool = False
    phase: str = "idle"
    reason: str | None = None
    trigger: str | None = None
    start_steps: int = 0
    steps: int = 0
    start_depth: int = 0
    drinks: int = 0
    attempts: list = field(default_factory=list)
    phase_steps: dict = field(default_factory=dict)
    _last_phase: str = "idle"
    _last_phase_step: int = 0
    _cooldown_until: int = 0
    _walk_rejections: int = 0
    _last_drink_step: int | None = None
    _give_up: str | None = None

    # ---- clocks (same shape the RESUPPLY window loop expects of a service) ----
    @property
    def service_microstep_cap(self):
        return RETREAT_SERVICE_CAP

    @property
    def command_microstep_deadline(self):
        return self.start_steps + RETREAT_SERVICE_CAP

    def record_steps(self, absolute_steps, phase=None):
        elapsed = int(absolute_steps) - self._last_phase_step
        charged = self._last_phase if phase is None else phase
        if elapsed > 0:
            self.phase_steps[charged] = self.phase_steps.get(charged, 0) + elapsed
        self._last_phase_step = int(absolute_steps)
        self.steps = int(absolute_steps) - self.start_steps

    # ---- the manager-side law ----
    def trigger_reason(self, raw, steps):
        """Return the trigger name if the retreat law fires now, else None."""
        if self.active or raw is None:
            return None
        if raw.get("dead") or raw.get("game_over") or raw.get("is_set_level"):
            return None
        if int(raw.get("dungeon_level", 0)) < 2:
            return None
        if int(steps) < self._cooldown_until:
            return None
        if retreats_started(raw) >= RETREAT_MAX_PER_EPISODE:
            return None
        hp, max_hp = int(raw.get("hp", 0)), max(1, int(raw.get("max_hp", 1)))
        if hp <= 0:
            return None
        belt = belt_heals(raw)
        if hp <= self.policy.hp_fraction * max_hp:
            return "low_hp"
        if belt == 0 and hp <= self.policy.empty_belt_hp_fraction * max_hp:
            return "empty_belt"
        if (self.policy.pressure_count
                and alive_monsters_within(raw, self.policy.pressure_radius) >= self.policy.pressure_count):
            return "pressure"
        return None

    def start(self, raw, steps, bridge, trigger):
        if self.active:
            raise RuntimeError("retreat already active")
        if int(raw.get("dungeon_level", 0)) < 2 or raw.get("is_set_level"):
            raise RuntimeError("retreat can start only on main L2 or deeper")
        bridge.configure_retreat(True)
        self.attempted = self.active = True
        self.phase = self._last_phase = "ascend"
        self.reason = None
        self.trigger = trigger
        self.start_steps = self._last_phase_step = int(steps)
        self.steps = 0
        self.start_depth = int(raw["dungeon_level"])
        self._walk_rejections = 0
        self._give_up = None
        self._last_drink_step = None
        self.attempts.append({
            "trigger": trigger, "beat0": int(steps), "depth0": self.start_depth,
            "hp0": int(raw.get("hp", 0)), "max_hp": int(raw.get("max_hp", 0)),
            "belt0": belt_heals(raw),
            "monsters_near0": alive_monsters_within(raw, self.policy.pressure_radius),
            "outcome": None, "reason": None, "beat1": None, "hp1": None, "drinks": 0})

    # ---- the worker-side hands ----
    def command(self, env, bridge):
        self.record_steps(env._steps)
        command = self._command(env, bridge)
        self._last_phase = self.phase
        return command

    def _command(self, env, bridge):
        raw = env._raw
        depth = int(raw["dungeon_level"])
        if raw.get("is_set_level"):
            return self._end(env, bridge, "retreat_unexpected_scene", failed=True)
        if depth == self.start_depth - 1:
            return self._end(env, bridge, "retreat_ascended", failed=False)
        if depth != self.start_depth:
            return self._end(env, bridge, "retreat_unexpected_scene", failed=True)
        if raw.get("dead") or raw.get("game_over") or int(raw.get("hp", 0)) <= 0:
            return self._end(env, bridge, "retreat_died", failed=True)
        if self._give_up is not None:
            return self._end(env, bridge, self._give_up, failed=True)
        if self.steps >= self.service_microstep_cap:
            return self._end(env, bridge, "retreat_cap", failed=True)
        hp, max_hp = int(raw.get("hp", 0)), max(1, int(raw.get("max_hp", 1)))
        if (hp <= self.policy.drink_hp_fraction * max_hp and belt_heals(raw) > 0
                and (self._last_drink_step is None
                     or env._steps - self._last_drink_step >= RETREAT_DRINK_SPACING)):
            self._last_drink_step = int(env._steps)
            return ("drink",)
        stairs = [t for t in raw.get("triggers", []) if t.get("msg") == bridge.WM_DIABPREVLVL]
        if not stairs:
            return self._end(env, bridge, "retreat_stairs_missing", failed=True)
        px, py = int(raw["player_x"]), int(raw["player_y"])
        stairs.sort(key=lambda t: max(abs(int(t["x"]) - px), abs(int(t["y"]) - py)))
        target = stairs[0]
        if (px, py) == (int(target["x"]), int(target["y"])):
            return ("wait",)
        command = self._walk(env, int(target["x"]), int(target["y"]))
        if command is None:
            return self._end(env, bridge, "retreat_stairs_unreachable", failed=True)
        return command

    @staticmethod
    def _walk(env, x, y):
        raw = env._raw
        path = env._plan_descend_path(raw, x, y, avoid_monsters=True)
        if not path:
            path = env._plan_descend_path(raw, x, y, avoid_monsters=False)
        if not path:
            return None
        nx, ny, door = path[0]
        if door:
            from . import bridge
            if not bool(bridge.probe_tile(int(nx), int(ny))["walkable"]):
                return ("open", int(nx), int(ny))
        return ("walk", int(nx), int(ny))

    def receipt(self, command, receipt):
        kind = command[0]
        if kind in ("walk", "open"):
            if receipt.get("accepted"):
                self._walk_rejections = 0
            else:
                self._walk_rejections += 1
                if self._walk_rejections >= RETREAT_WALK_REJECTIONS:
                    self._give_up = "retreat_walk_rejected"
        elif kind == "drink" and receipt.get("accepted"):
            self.drinks += 1
            if self.attempts:
                self.attempts[-1]["drinks"] += 1

    def _end(self, env, bridge, reason, *, failed):
        self.active = False
        self.phase = "done"
        self.reason = reason
        if failed:
            self._cooldown_until = int(env._steps) + RETREAT_FAILURE_COOLDOWN
            try:
                bridge.configure_retreat(False)
            except RuntimeError as exc:  # engine out of game after death: nothing to revoke
                self.attempts[-1]["revoke_error"] = str(exc)
        if self.attempts:
            record = self.attempts[-1]
            record["outcome"] = "failed" if failed else "ascended"
            record["reason"] = reason
            record["beat1"] = int(env._steps)
            record["hp1"] = int(env._raw.get("hp", 0))
        return ("complete",)

    def telemetry(self):
        return {"protocol": "retreat-v1", "policy": self.policy.as_dict(),
                "attempted": self.attempted, "active": self.active, "phase": self.phase,
                "reason": self.reason, "trigger": self.trigger, "steps": self.steps,
                "drinks": self.drinks, "attempts": [dict(a) for a in self.attempts],
                "ascended": sum(1 for a in self.attempts if a.get("outcome") == "ascended"),
                "failed": sum(1 for a in self.attempts if a.get("outcome") == "failed"),
                "phase_steps": dict(self.phase_steps)}
