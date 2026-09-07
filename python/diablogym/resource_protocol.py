"""Opt-in L1 resource service. All admission facts come from the native engine."""
from __future__ import annotations
from dataclasses import dataclass, field

RESOURCE_PROTOCOLS = ("off", "l2-town-v1")
RESOURCE_PURCHASE_MODES = ("none", "heal", "potions", "armor", "full")
# R17.1 chairman ruling 3 (2026-09-06). "veto-v1": the native seven-condition hard
# gate vetoes L1->L2 and masks DIVE/a11 (the R18-R23 behaviour, kept bit-for-bit).
# "coach-v03": the law is the six conditions clvl/AC/dmg/belt/durability/weapon
# (HP excluded); the engine guard only records; the frozen forced-descent mask law
# stands; forced-unready descents are accounted and paid no escrow.
RESOURCE_READINESS_LAWS = ("veto-v1", "coach-v03")
RESOURCE_FARM_CAP = 3600
RESOURCE_SERVICE_CAP = 600


@dataclass(frozen=True, kw_only=True)
class ResourceCalibration:
    """Explicit diagnostic identity; never accepted by Worker training wrappers."""
    calibration_id: str
    service_microstep_cap: int = RESOURCE_SERVICE_CAP
    farm_scene_microstep_cap: int = RESOURCE_FARM_CAP

    def __post_init__(self):
        if not isinstance(self.calibration_id, str) or not self.calibration_id.strip():
            raise ValueError("calibration_id must be a nonempty diagnostic identity")
        for name in ("service_microstep_cap", "farm_scene_microstep_cap"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a finite positive integer")

    @property
    def worker_farm_scene_denominator(self):
        # Trigger timing is the intervention. Preserve the production actor's
        # normalization before that intervention starts the pure script.
        return RESOURCE_FARM_CAP

    def metadata(self):
        return {"version": "resource-service-calibration-v1",
                "calibration_id": self.calibration_id,
                "diagnostic_only": True, "formal_metric_eligible": False,
                "service_microstep_cap": self.service_microstep_cap,
                "farm_scene_microstep_cap": self.farm_scene_microstep_cap,
                "worker_farm_scene_denominator": self.worker_farm_scene_denominator,
                "production_service_microstep_cap": RESOURCE_SERVICE_CAP,
                "production_farm_scene_microstep_cap": RESOURCE_FARM_CAP}


def validate_resource_calibration(protocol, calibration):
    if calibration is not None:
        if not isinstance(calibration, ResourceCalibration):
            raise TypeError("resource_calibration must be ResourceCalibration or None")
        if protocol != "l2-town-v1":
            raise ValueError("resource_calibration requires l2-town-v1")
    return calibration


def validate_resource_config(protocol, mode):
    if protocol not in RESOURCE_PROTOCOLS:
        raise ValueError(f"Unknown resource_protocol: {protocol!r}")
    if mode not in RESOURCE_PURCHASE_MODES:
        raise ValueError(f"Unknown resource_purchase_mode: {mode!r}")
    return protocol, mode


def validate_ordinary_armor_scope(protocol, mode, enabled):
    if not isinstance(enabled, bool):
        raise ValueError("resource_ordinary_armor_scope must be a bool")
    if enabled and (protocol != "l2-town-v1" or mode != "full"):
        raise ValueError("ordinary armor scope requires l2-town-v1 and full purchase mode")
    return enabled


def validate_native_armor_scope(raw, expected=False):
    """Check the already returned raw marker without changing observations."""
    state = raw.get("resource_state", {})
    actual = state.get("ordinary_armor_scope", False)
    if actual is not expected or (expected and state.get("enabled") is not True):
        raise RuntimeError("Native ordinary_armor_scope does not match the configured service version")


def validate_equipment_readiness_preservation(protocol, enabled):
    if not isinstance(enabled, bool):
        raise ValueError("resource_preserve_equipment_readiness must be a bool")
    if enabled and protocol != "l2-town-v1":
        raise ValueError("equipment readiness preservation requires l2-town-v1")
    return enabled


def validate_native_equipment_readiness_preservation(raw, expected=False):
    """Validate native feature identity, never reproduce equipment gate formulas."""
    state = raw.get("resource_state", {})
    actual = state.get("preserve_equipment_readiness", False)
    if actual is not expected or (expected and state.get("enabled") is not True):
        raise RuntimeError("Native preserve_equipment_readiness does not match the configured service version")


def native_readiness(raw):
    state = raw.get("resource_state")
    if not isinstance(state, dict) or not state.get("enabled"):
        raise RuntimeError("Enabled resource protocol lacks native resource_state")
    ready = state.get("readiness")
    if not isinstance(ready, dict) or not isinstance(ready.get("ready"), bool):
        raise RuntimeError("Native readiness verdict is missing or invalid")
    return ready


def validate_readiness_law(protocol, law):
    if law not in RESOURCE_READINESS_LAWS:
        raise ValueError(f"Unknown resource_readiness_law: {law!r}")
    if law != "veto-v1" and protocol != "l2-town-v1":
        raise ValueError("resource_readiness_law coach-v03 requires l2-town-v1")
    return law


def validate_native_readiness_law(raw, expected_law="veto-v1"):
    """Fail closed on the configured flag, not on the native payload's presence."""
    state = raw.get("resource_state")
    if expected_law == "veto-v1":
        if isinstance(state, dict) and state.get("readiness_advisory", False):
            raise RuntimeError("Native readiness_advisory is set but the law is veto-v1")
        return
    if not isinstance(state, dict) or state.get("enabled") is not True:
        raise RuntimeError("coach-v03 requires an enabled native resource_state")
    if state.get("readiness_advisory") is not True:
        raise RuntimeError("Native bridge did not acknowledge readiness_advisory (rebuild required)")
    ready = state.get("readiness")
    if not isinstance(ready, dict) or not isinstance(ready.get("ready_excluding_health"), bool):
        raise RuntimeError("Native readiness lacks ready_excluding_health (rebuild required)")


def law_ready(raw, law="veto-v1"):
    """Readiness verdict under the configured law (see RESOURCE_READINESS_LAWS)."""
    ready = native_readiness(raw)
    if law == "coach-v03":
        verdict = ready.get("ready_excluding_health")
        if not isinstance(verdict, bool):
            raise RuntimeError("Native readiness lacks ready_excluding_health (rebuild required)")
        return verdict
    return ready["ready"]


def progression_allowed(raw, law="veto-v1"):
    # Quest mechanisms and return portals retain DIVE authority. A new main
    # floor is a separate operation and is ultimately checked by the engine.
    if raw.get("is_set_level") or raw.get("progression_targets"):
        return True
    if law == "coach-v03":
        # The frozen mask law governs DIVE/a11; the engine guard only records.
        return True
    depth = int(raw["dungeon_level"])
    return depth == 0 or (depth == 1 and native_readiness(raw)["ready"])


@dataclass
class ResourceService:
    mode: str = "full"
    calibration: ResourceCalibration | None = field(default=None, kw_only=True)
    start_farm_steps: int | None = field(default=None, kw_only=True)
    attempted: bool = False
    active: bool = False
    phase: str = "idle"
    start_steps: int = 0
    steps: int = 0
    reason: str | None = None
    trigger: str | None = None
    purchases: int = 0
    gold_spent: int = 0
    repairs: int = 0
    repair_gold_spent: int = 0
    repair_failures: list = field(default_factory=list)
    _repair_attempted: set = field(default_factory=set)
    gold_collected: int = 0
    _last_gold: int = 0
    _gold_seen: set = field(default_factory=set)
    _gold_target: dict | None = None
    _gold_navigation_seen: set = field(default_factory=set)
    gold_navigation_failures: int = 0
    phase_steps: dict = field(default_factory=dict)
    _last_phase: str = "idle"
    _last_phase_step: int = 0
    _shop_failures: set = field(default_factory=set)
    _town_healed: bool = False
    _armor_done: bool = False
    _armor_pending: tuple | None = None
    _potions_done: bool = False
    _talk_attempts: int = 0

    def __post_init__(self):
        validate_resource_calibration("l2-town-v1", self.calibration)

    @property
    def service_microstep_cap(self):
        return (RESOURCE_SERVICE_CAP if self.calibration is None
                else self.calibration.service_microstep_cap)

    def start(self, raw, steps, trigger, *, farm_scene_steps=None):
        if self.attempted:
            return
        self.attempted = self.active = True
        self.phase = "collect"
        self.start_steps = int(steps)
        self.start_farm_steps = (None if farm_scene_steps is None else int(farm_scene_steps))
        self.trigger = trigger
        self._last_gold = int(raw.get("gold", 0))
        self._last_phase = "collect"
        self._last_phase_step = int(steps)

    def telemetry(self):
        info = {"mode": self.mode, "attempted": self.attempted,
                "active": self.active, "phase": self.phase,
                "steps": self.steps, "reason": self.reason,
                "trigger": self.trigger, "purchases": self.purchases,
                "gold_spent": self.gold_spent,
                "repairs": self.repairs, "repair_gold_spent": self.repair_gold_spent,
                "repair_failures": list(self.repair_failures),
                "gold_collected": self.gold_collected,
                "gold_navigation_failures": self.gold_navigation_failures,
                "phase_steps": dict(self.phase_steps)}
        if self.calibration is not None:
            info["calibration"] = self.calibration.metadata()
            info["start_microstep"] = self.start_steps if self.attempted else None
            info["start_farm_microsteps"] = self.start_farm_steps
            info["trigger_threshold_excess"] = (
                self.start_farm_steps - self.calibration.farm_scene_microstep_cap
                if self.trigger == "farm_cap" and self.start_farm_steps is not None
                else None)
        return info

    def _finish(self, reason):
        self.reason = reason
        self.active = False
        self.phase = "done"
        return ("finish", reason)

    def record_steps(self, absolute_steps, phase=None):
        elapsed = int(absolute_steps) - self._last_phase_step
        charged_phase = self._last_phase if phase is None else phase
        if elapsed > 0:
            self.phase_steps[charged_phase] = self.phase_steps.get(charged_phase, 0) + elapsed
        self._last_phase_step = int(absolute_steps)
        self.steps = int(absolute_steps) - self.start_steps

    def command(self, env, bridge):
        self.record_steps(env._steps)
        command = self._command(env, bridge)
        self._last_phase = self.phase
        return command

    def _command(self, env, bridge):
        raw = env._raw
        gold = int(raw.get("gold", 0))
        if gold > self._last_gold:
            self.gold_collected += gold - self._last_gold
        self._last_gold = gold
        state = raw["resource_state"]
        depth = int(raw["dungeon_level"])
        if raw.get("is_set_level") or depth not in (0, 1):
            return self._finish("resource_unexpected_scene")
        # Returning on the final allowed native microstep is complete. This
        # settlement consumes no microsteps and never authorizes an over-budget step.
        if (self.phase == "return" and depth == 1
                and self.steps <= self.service_microstep_cap):
            bridge.configure_town_service(False)
            if native_readiness(raw)["ready"]:
                self.active = False
                self.phase = "done"
                self.reason = "complete"
                return ("complete",)
            return self._finish_or_farm(env, "resource_unreachable")
        if self.steps >= self.service_microstep_cap:
            return self._finish("resource_service_cap")
        if self.phase == "collect":
            items = state.get("gold_items", [])
            items = sorted(items, key=lambda item: (
                max(abs(int(item["x"]) - int(raw["player_x"])),
                    abs(int(item["y"]) - int(raw["player_y"]))),
                int(item.get("id", item.get("active_id", 0)))))
            # Visibility changes while walking. Keep the identity and last
            # observed coordinate of one target instead of retargeting to each
            # newly illuminated Chebyshev-nearest pile (seed 2114004 oscillated
            # between two tiles for all 600 service microsteps).
            while True:
                if self._gold_target is None:
                    self._gold_target = next((dict(item) for item in items
                        if self._gold_key(item) not in self._gold_seen), None)
                    self._gold_navigation_seen.clear()
                item = self._gold_target
                if item is None:
                    break
                key = self._gold_key(item)
                position = (int(raw["player_x"]), int(raw["player_y"]))
                target = (int(item["x"]), int(item["y"]))
                if position == target:
                    self._gold_seen.add(key)
                    self._gold_target = None
                    # A native automatic pickup or another effect may already
                    # have removed this known pile while the walk settled.
                    if any(self._gold_key(current) == key for current in items):
                        return ("gold", *key)
                    continue
                command = self._walk(env, *target)
                navigation = (position, command)
                if command is not None and navigation not in self._gold_navigation_seen:
                    self._gold_navigation_seen.add(navigation)
                    return command
                self._gold_seen.add(key)
                self._gold_target = None
                self.gold_navigation_failures += 1
            if native_readiness(raw)["ready"]:
                self.active = False
                self.phase = "done"
                self.reason = "ready_without_town"
                return ("complete",)
            if self.mode == "none":
                return self._finish_or_farm(env, "resource_unreachable_no_town")
            self.phase = "outbound"
            bridge.configure_town_service(True)
        if self.phase == "outbound":
            if depth == 0:
                self.phase = "smith" if self.mode in ("armor", "full") else "healer"
            else:
                return self._stairs(env, bridge.WM_DIABPREVLVL)
        town = state.get("town", {})
        if self.phase == "healer":
            if depth != 0:
                return self._finish("resource_unexpected_scene")
            if int(raw["hp"]) >= int(raw["max_hp"]):
                self._town_healed = True
            if not self._town_healed:
                command = self._visit(env, "healer", town)
                if command:
                    return command
                # Pepin can present a quest speech before the healing branch.
                if town.get("dialog_active"):
                    return ("dismiss",)
                self._talk_attempts += 1
                if self._talk_attempts > 8:
                    return self._finish("resource_healer_unavailable")
                return self._talk(env, "healer", town)
            self.phase = "potions"
        if self.phase == "smith":
            if not native_readiness(raw).get("armor_service_needed", True):
                self._armor_done = True
            if self._armor_pending is not None:
                pending = self._armor_pending
                self._armor_pending = None
                for item in state.get("inventory_items", []):
                    identity = tuple(int(item[k]) for k in
                        ("seed_hi", "seed_lo", "create_info", "base_id"))
                    if identity == pending:
                        return ("equip", int(item["index"]), *identity)
                # StoreAutoPlace can directly equip an empty chest slot.
                # Otherwise the final native readiness verdict exposes failure.
            if not self._armor_done:
                command = self._visit(env, "smith", town)
                if command:
                    return command
                candidates = [item for item in town.get("stock", [])
                    if item.get("vendor") == "smith"
                    and item.get("can_use") and item.get("can_fit")
                    and item.get("meets_armor_gate", False)
                    and int(item.get("price", 0)) <= int(raw.get("gold", 0))
                    and self._stock_key(item) not in self._shop_failures]
                candidates.sort(key=lambda item: (int(item["price"]), int(item["index"])))
                if candidates:
                    self._armor_done = True
                    return self._buy(candidates[0])
                self._armor_done = True
            self.phase = "repair" if self.mode == "full" else "healer"
        if self.phase == "repair":
            if self.mode != "full":
                return self._finish("resource_invalid_repair_mode")
            if not native_readiness(raw).get("repair_service_needed", True):
                self.phase = "healer"
                return self._command(env, bridge)
            command = self._visit(env, "smith", town)
            if command:
                return command
            candidates = [quote for quote in town.get("repair_quotes", [])
                if quote.get("repair_needed_for_gate")
                and self._repair_key(quote) not in self._repair_attempted]
            candidates.sort(key=lambda quote: (int(quote["price"]), int(quote["slot"])))
            for quote in candidates:
                identity = self._repair_key(quote)
                self._repair_attempted.add(identity)
                if int(quote["price"]) > int(raw.get("gold", 0)):
                    self.repair_failures.append("unaffordable_repair")
                    continue
                return ("repair", *identity, int(quote["durability"]), int(quote["price"]))
            self.phase = "healer"
        if self.phase == "potions":
            ready = native_readiness(raw)
            if (self.mode in ("potions", "full")
                    and int(ready.get("belt_heals", raw.get("belt_heals", 0))) < 8
                    and int(ready.get("belt_free_slots", raw.get("belt_free_slots", 0))) > 0):
                command = self._visit(env, "healer", town)
                if command:
                    return command
                candidates = [item for item in town.get("stock", [])
                    if item.get("vendor") == "healer" and int(item.get("heal_kind", 0)) > 0
                    and item.get("can_use") and item.get("can_fit")
                    and int(item.get("price", 0)) <= int(raw.get("gold", 0))
                    and self._stock_key(item) not in self._shop_failures]
                candidates.sort(key=lambda item: (int(item["price"]), int(item["index"])))
                if candidates:
                    return self._buy(candidates[0])
            self.phase = "return"
        if self.phase == "return":
            if town.get("dialog_active") or town.get("active_vendor") not in (None, "none"):
                return ("dismiss",)
            return self._stairs(env, bridge.WM_DIABNEXTLVL)
        if self.phase == "healer":
            return self._command(env, bridge)
        return self._finish("resource_invalid_phase")

    def _finish_or_farm(self, env, unavailable_reason):
        raw = env._raw
        alive = any(int(m.get("hp", 0)) > 0 and int(m.get("type", -1)) != 109
                    for m in raw.get("monsters", []))
        controller, _ = env.controller_action_context()
        growth = alive or bool(controller[13]) or bool(controller[14])
        if growth:
            self.active = False
            self.phase = "done"
            self.reason = "growth_remaining"
            return ("complete",)
        return self._finish(unavailable_reason)

    @staticmethod
    def _gold_key(item):
        return (int(item.get("id", item.get("active_id", 0))),
                *(int(item[k]) for k in ("seed_hi", "seed_lo", "create_info", "base_id")))

    @staticmethod
    def _repair_key(quote):
        return (int(quote["slot"]), *(int(quote[k]) for k in
                ("seed_hi", "seed_lo", "create_info", "base_id")))

    @staticmethod
    def _stock_key(item):
        return (item["vendor"], int(item["index"]), int(item["seed_hi"]),
                int(item["seed_lo"]), int(item["create_info"]), int(item["base_id"]))

    def _buy(self, item):
        return ("buy", *self._stock_key(item))

    def receipt(self, command, receipt):
        if command[0] == "repair":
            if receipt.get("accepted"):
                self.repairs += 1
                paid = int(receipt.get("price", 0))
                self.repair_gold_spent += paid
                self.gold_spent += paid
            else:
                self.repair_failures.append(str(receipt.get("reason", "repair_rejected")))
            return
        if command[0] != "buy":
            return
        if receipt.get("accepted"):
            if command[1] == "smith":
                self._armor_pending = tuple(command[3:])
            self.purchases += 1
            self.gold_spent += int(receipt.get("price", 0))
        else:
            self._shop_failures.add(tuple(command[1:]))

    def _stairs(self, env, message):
        raw = env._raw
        stairs = [t for t in raw.get("triggers", []) if t.get("msg") == message]
        if not stairs:
            return self._finish("resource_stairs_missing")
        stairs.sort(key=lambda t: max(abs(int(t["x"])-int(raw["player_x"])),
                                     abs(int(t["y"])-int(raw["player_y"]))))
        target = stairs[0]
        if (raw["player_x"], raw["player_y"]) == (target["x"], target["y"]):
            return ("wait",)
        command = self._walk(env, int(target["x"]), int(target["y"]))
        return command if command else self._finish("resource_stairs_unreachable")

    def _visit(self, env, vendor, town):
        if town.get("active_vendor") == vendor and not town.get("dialog_active"):
            return None
        if town.get("dialog_active") or town.get("active_vendor") not in (None, "none", vendor):
            return ("dismiss",)
        return self._talk(env, vendor, town)

    def _talk(self, env, vendor, town):
        npc = next((n for n in town.get("npcs", []) if n.get("type") == vendor), None)
        if npc is None:
            return self._finish("resource_npc_missing")
        raw = env._raw
        if max(abs(int(npc["x"])-int(raw["player_x"])),
               abs(int(npc["y"])-int(raw["player_y"]))) <= 1:
            return ("talk", int(npc["id"]))
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            command = self._walk(env, int(npc["x"])+dx, int(npc["y"])+dy)
            if command:
                return command
        return self._finish("resource_npc_unreachable")

    @staticmethod
    def _walk(env, x, y):
        raw = env._raw
        path = env._plan_descend_path(raw, x, y, avoid_monsters=True)
        if not path:
            return None
        nx, ny, door = path[0]
        if door:
            # The planner's door channel marks the object, including an open
            # door. Ordinary descent clears its cached edge after opening;
            # this service replans each decision, so consult the same native
            # walkability probe before operating (which otherwise closes it).
            from . import bridge
            if not bool(bridge.probe_tile(int(nx), int(ny))["walkable"]):
                return ("open", int(nx), int(ny))
        return ("walk", int(nx), int(ny))
