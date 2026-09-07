"""Observed-gold services with distinct v3 and v4 collect command windows."""
from copy import deepcopy
from dataclasses import dataclass, field

from .resource_gold_memory import ObservedGoldMemory
from .resource_protocol import native_readiness
from .resource_sustain import SustainResourceService, item_identity

OBSERVED_GOLD_COLLECT_CAP = 300


@dataclass
class SustainGoldMemoryService(SustainResourceService):
    gold_memory: ObservedGoldMemory = field(default_factory=ObservedGoldMemory)
    collect_stop_reason: str | None = None
    _gold_request: dict | None = None

    @property
    def collect_microstep_cap(self):
        # Keep v3's original constant lookup for explicitly identified external
        # diagnostics. Production v3 still defaults to its historical 300.
        return OBSERVED_GOLD_COLLECT_CAP

    @property
    def command_microstep_deadline(self):
        deadline = self.start_steps + self.service_microstep_cap
        if self.phase == "collect":
            deadline = min(deadline, self.start_steps + self.collect_microstep_cap)
        return deadline

    def telemetry(self):
        info = super().telemetry()
        info.update(policy="sustain-v3", gold_memory=self.gold_memory.telemetry(),
                    collect_microstep_cap=self.collect_microstep_cap,
                    collect_stop_reason=self.collect_stop_reason)
        return info

    def _leave_collect(self, env, bridge, reason):
        self.collect_stop_reason = reason
        self._gold_target = None
        if native_readiness(env._raw)["ready"]:
            self.active = False
            self.phase = "done"
            self.reason = "ready_without_town"
            return ("complete",)
        self.phase = "outbound"
        bridge.configure_town_service(True)
        return super()._command(env, bridge)

    def _command(self, env, bridge):
        if self.phase != "collect":
            return super()._command(env, bridge)
        raw = env._raw
        if raw.get("is_set_level") or int(raw["dungeon_level"]) != 1:
            return self._finish("resource_unexpected_scene")
        if self.steps >= self.service_microstep_cap:
            return self._finish("resource_service_cap")
        gold = int(raw.get("gold", 0))
        if gold > self._last_gold:
            self.gold_collected += gold - self._last_gold
        self._last_gold = gold
        self.gold_memory.observe(raw, int(env._steps))
        if self.steps >= self.collect_microstep_cap:
            return self._leave_collect(env, bridge, "collect_budget")
        while True:
            if self._gold_target is None:
                candidates = self.gold_memory.candidates()
                candidates.sort(key=lambda item: (
                    max(abs(int(item["x"]) - int(raw["player_x"])),
                        abs(int(item["y"]) - int(raw["player_y"]))),
                    item_identity(item)))
                self._gold_target = candidates[0] if candidates else None
                self._gold_navigation_seen.clear()
            item = self._gold_target
            if item is None:
                return self._leave_collect(env, bridge, "observed_targets_exhausted")
            position = (int(raw["player_x"]), int(raw["player_y"]))
            target = (int(item["x"]), int(item["y"]))
            if position == target:
                actual = next((current for current in raw["resource_state"].get("gold_items", [])
                    if item_identity(current) == item_identity(item)
                    and (int(current["x"]), int(current["y"])) == position), None)
                self._gold_target = None
                if actual is None:
                    if not self.gold_memory.mark_gone(item, raw, int(env._steps)):
                        raise RuntimeError("Known gold disappearance lacks an on-tile real observation")
                    continue
                self._gold_request = dict(actual)
                self.gold_memory.mark_attempted(actual, "pickup_requested", int(env._steps))
                return ("gold", *self._gold_key(actual))
            command = self._walk(env, *target)
            signature = (position, command)
            if command is not None and signature not in self._gold_navigation_seen:
                self._gold_navigation_seen.add(signature)
                return command
            self.gold_memory.mark_attempted(item, "route_unavailable", int(env._steps))
            self._gold_target = None
            self.gold_navigation_failures += 1

    def receipt(self, command, receipt):
        super().receipt(command, receipt)
        if command[0] == "gold" and self._gold_request is not None:
            # receipt precedes ResourceService.record_steps; take the actual
            # passive observation clock, never manufacture elapsed microsteps.
            observed_step = self.gold_memory.telemetry()["last_observed_step"]
            self.gold_memory.mark_attempted(self._gold_request,
                str(receipt.get("reason", "request_accepted" if receipt.get("accepted") else "request_rejected")),
                self.start_steps + self.steps if observed_step is None else observed_step)
            self._gold_request = None


@dataclass
class SustainGoldExtendedService(SustainGoldMemoryService):
    """V4 stops issuing collect commands after a 450-microstep window.

    A previously committed native animation may finish in outbound. Every tail
    beat still consumes the same total service budget; the cutoff is not a
    promise that the player is idle or that a physical action already finished.
    This class adds only passive handoff evidence to the shared execution path.
    """
    _collect_handoff: dict | None = field(default=None, init=False, repr=False)

    @property
    def collect_microstep_cap(self):
        return 450

    def telemetry(self):
        info = super().telemetry()
        info.pop("collect_microstep_cap")
        info.update(policy="sustain-v4", collect_command_window_microsteps=450,
                    collect_handoff=deepcopy(self._collect_handoff))
        return info

    def _leave_collect(self, env, bridge, reason):
        if self._collect_handoff is None:
            raw = env._raw
            clock = int(getattr(env, "_resource_actual_microsteps", env._steps))
            self._collect_handoff = {
                "absolute_microstep": clock,
                "elapsed_service_microsteps": clock - self.start_steps,
                "collect_command_deadline": self.command_microstep_deadline,
                "reason": reason,
                "player_mode": raw.get("player_mode"),
                "tile": [raw.get("player_x"), raw.get("player_y")],
                "future": [raw.get("future_x"), raw.get("future_y")],
                "walkpath0": raw.get("walkpath0"),
                "dest_action": raw.get("dest_action"),
            }
        return super()._leave_collect(env, bridge, reason)
