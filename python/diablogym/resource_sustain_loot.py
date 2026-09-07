"""Two bounded, observed-resource town trips; no new policy or engine query.

The completion clock remains the sole episode deadline owner.  This service
adds only ordinary loot pickup/sale commands and a guarded second trip to the
v7 purchase planner.  Observed item values are never credited as income.
"""
from copy import deepcopy
from dataclasses import dataclass, field, fields

from .completion_clock import COMPLETION_L2_V1
from .resource_protocol import native_readiness
from .resource_sustain_completion import SustainCompletionService


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"Invalid {name}")
    return value


def _identity(item):
    return tuple(_integer(item[k], k) for k in
                 ("seed_hi", "seed_lo", "create_info", "base_id"))


def _safe_sale_item(item):
    return (item.get("empty") is not True and item.get("sellable") is True
            and item.get("is_quest") is False and item.get("upgrade") is False
            and item.get("reserved_upgrade") is False)


@dataclass
class SustainLootService(SustainCompletionService):
    """The caller creates a new instance on every normal episode reset."""
    trip_count: int = field(default=0, init=False)
    gold_sold: int = field(default=0, init=False)
    sales: int = field(default=0, init=False)
    loot_collected: int = field(default=0, init=False)
    loot_failures: list = field(default_factory=list, init=False)
    sale_failures: list = field(default_factory=list, init=False)
    terminal_cash_events: list = field(default_factory=list, init=False)
    death_cash_change: int = field(default=0, init=False)
    _trips: list = field(default_factory=list, init=False, repr=False)
    _loot_memory: dict = field(default_factory=dict, init=False, repr=False)
    _loot_attempted: set = field(default_factory=set, init=False, repr=False)
    _loot_target: dict | None = field(default=None, init=False, repr=False)
    _sale_attempted: set = field(default_factory=set, init=False, repr=False)
    _protected: set = field(default_factory=set, init=False, repr=False)
    _purchase_pending: set = field(default_factory=set, init=False, repr=False)
    _latest: dict | None = field(default=None, init=False, repr=False)
    _observed_step: int | None = field(default=None, init=False, repr=False)
    _request: dict | None = field(default=None, init=False, repr=False)
    _start_gold: int = field(default=0, init=False, repr=False)
    _visited_town: bool = field(default=False, init=False, repr=False)
    _sale_town_sequence: tuple | None = field(default=None, init=False, repr=False)
    _finalize_pending: bool = field(default=False, init=False, repr=False)
    _return_baseline: dict | None = field(default=None, init=False, repr=False)
    _return_step: int | None = field(default=None, init=False, repr=False)
    _trigger_changes: list = field(default_factory=list, init=False, repr=False)
    _last_start_denial: str | None = field(default=None, init=False, repr=False)
    _remaining_resource_evidence: dict = field(default_factory=dict, init=False, repr=False)
    _reopened_gold: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.mode != "full" or self.calibration is not None:
            raise ValueError("sustain-loot-v1 requires full, uncalibrated completion-l2-v1")
        super().__post_init__()

    @staticmethod
    def _snapshot(raw):
        state = raw.get("resource_state", {})
        if state.get("loot_economy") is not True:
            raise RuntimeError("sustain-loot-v1 requires native loot_economy=True")
        inventory = state.get("inventory_state")
        if not isinstance(inventory, dict) or not isinstance(inventory.get("items"), list):
            raise RuntimeError("Native complete inventory_state is required")
        items = deepcopy(inventory["items"])
        ids = [_identity(i) for i in items
               if i.get("empty") is not True and i.get("is_equipment") is True]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Duplicate native inventory identity")
        return {"gold": _integer(raw["gold"], "gold"),
                "dungeon_level": int(raw["dungeon_level"]),
                "is_set_level": bool(raw.get("is_set_level")),
                "dead": bool(raw.get("dead") or raw.get("game_over")),
                "player_mode": raw.get("player_mode"),
                "char_level": raw.get("char_level"),
                "experience": raw.get("xp"),
                "readiness": deepcopy(native_readiness(raw)),
                "equipped_items": deepcopy(raw.get("equipped_items", [])),
                "inventory": items}

    def observe(self, raw, steps):
        """Consume the existing observation, without observe/predict/mask calls."""
        steps = _integer(steps, "observation clock")
        if self._observed_step is not None and steps < self._observed_step:
            raise RuntimeError("Loot service observation clock moved backwards")
        self._latest = self._snapshot(raw)
        self._observed_step = steps
        self.gold_memory.observe(raw, steps)
        if self.active and int(raw["dungeon_level"]) == 0:
            self._visited_town = True
        # A former upgrade can later become genuinely unused equipment. Only
        # the current native verdict protects it; pending purchases are separate.
        self._protected = {_identity(item) for item in self._latest["inventory"]
                           if item.get("reserved_upgrade") is True or item.get("upgrade") is True}
        if int(raw["dungeon_level"]) == 1 and not raw.get("is_set_level"):
            for item in raw["resource_state"].get("loot_items", []):
                identity = _identity(item)
                if not _safe_sale_item(item):
                    self._loot_memory.pop(identity, None)
                    continue
                if _integer(item["sell_price"], "loot sell price") == 0:
                    continue
                self._loot_memory[identity] = dict(deepcopy(item), last_seen=steps)

    def _fingerprint(self, snapshot):
        ready = snapshot["readiness"]
        equipped = tuple((_identity(i), i.get("durability"), i.get("max_durability"))
                         for i in snapshot["equipped_items"] if i.get("present"))
        saleable = {_identity(i) for i in snapshot["inventory"]
                    if _safe_sale_item(i) and _identity(i) not in self._protected
                    and _identity(i) not in self._purchase_pending}
        return {"gold": snapshot["gold"], "belt_heals": ready.get("belt_heals"),
                "clvl": ready.get("clvl", snapshot["char_level"]),
                "experience": snapshot["experience"],
                "max_hp_fixed": ready.get("max_hp_fixed"),
                "equipped": equipped,
                "saleable": saleable | set(self._loot_memory),
                "hp_fixed": ready.get("hp_fixed")}

    def observed_remaining_opportunity(self, raw):
        """Separate last-seen search leads from currently verified capacity.

        A remembered price does not certify location, income or path. A second
        trip without other resource change needs a currently visible native
        item and actual capacity, rather than merely another idle clock tick.
        """
        state = raw["resource_state"]
        protected = self._protected | self._purchase_pending
        remembered_gold = self.gold_memory.candidates()
        remembered_loot = [i for k, i in self._loot_memory.items()
                           if k not in protected and _safe_sale_item(i)]
        carried = state["inventory_state"]
        held_empty = carried.get("held_empty") is True
        # An actual empty cell conservatively establishes room for gold. Do
        # not invent a stack limit or infer spare capacity from wallet size.
        gold_room = held_empty and isinstance(carried.get("free_cells"), int) and carried["free_cells"] > 0
        confirmed_gold = [deepcopy(i) for i in state.get("gold_items", [])
                          if gold_room and _integer(i["value"], "observed gold value") > 0]
        confirmed_loot = [deepcopy(i) for i in state.get("loot_items", [])
                          if held_empty and i.get("can_fit") is True and _safe_sale_item(i)
                          and _identity(i) not in protected and int(i["sell_price"]) > 0]
        return {"remembered_gold_identities": [list(_identity(i)) for i in remembered_gold],
                "remembered_loot_identities": [list(_identity(i)) for i in remembered_loot],
                "confirmed_gold": confirmed_gold, "confirmed_loot": confirmed_loot,
                "memory_is_income_or_route_proof": False,
                "remaining_trip_slots": max(0, self.trip_limit(raw)-self.trip_count)}

    def _finish_or_farm(self, env, unavailable_reason):
        raw = env._raw
        if (int(raw["dungeon_level"]) != 1 or raw.get("is_set_level")
                or raw.get("dead") or raw.get("game_over")):
            return super()._finish_or_farm(env, unavailable_reason)
        evidence = self.observed_remaining_opportunity(raw)
        self._remaining_resource_evidence = deepcopy(evidence)
        remaining = bool(evidence["remembered_gold_identities"]
                         or evidence["remembered_loot_identities"]
                         or evidence["confirmed_gold"] or evidence["confirmed_loot"])
        if remaining and self.trip_count < self.trip_limit(raw):
            self.active = False
            self.phase, self.reason = "done", "growth_remaining"
            return ("complete",)
        # Preserve ordinary combat/growth after the second trip. If only
        # shopping could help, explicitly record the exhausted trip limit.
        command = super()._finish_or_farm(env, unavailable_reason)
        if remaining and self.trip_count >= self.trip_limit(raw) and command[0] == "finish":
            return self._finish("resource_trip_limit_remaining_resources")
        return command

    @staticmethod
    def trip_limit(raw):
        """Two loot trips, plus one per completed R18-A retreat (engine-counted;
        the key is absent, hence zero, whenever retreat-v1 is off)."""
        state = raw.get("resource_state", {}) if isinstance(raw, dict) else {}
        return 2 + int((state or {}).get("retreats_started", 0) or 0)

    def maybe_start(self, raw, steps, *, farm_scene_steps, cleared):
        steps = _integer(steps, "start clock")
        farm_scene_steps = _integer(farm_scene_steps, "FARM clock")
        if not isinstance(cleared, bool):
            raise ValueError("cleared must be boolean")
        self.observe(raw, steps)
        current = self._latest
        denial = None
        if self.active:
            denial = "service_active"
        elif self.trip_count >= self.trip_limit(raw):
            denial = "trip_limit"
        elif (current["dead"] or current["is_set_level"]
              or current["dungeon_level"] != 1 or current["player_mode"] != 0):
            denial = "not_idle_alive_main_l1"
        elif current["readiness"]["ready"] or not current["readiness"]["failures"]:
            denial = "no_native_resource_deficit"
        changes = []
        if denial is None and self.trip_count == 0:
            if not cleared and farm_scene_steps < COMPLETION_L2_V1.farm_microsteps:
                denial = "first_farm_trigger_pending"
        elif denial is None:
            if self._return_baseline is None or steps <= self._return_step:
                denial = "no_settled_return_or_elapsed_play"
            else:
                before, now = self._return_baseline, self._fingerprint(current)
                for key in ("gold", "belt_heals", "clvl", "experience", "max_hp_fixed", "equipped"):
                    if before[key] != now[key]:
                        changes.append(key)
                if now["saleable"] - before["saleable"]:
                    changes.append("new_observed_saleable_loot")
                if ("health" in current["readiness"]["failures"]
                        and isinstance(now["hp_fixed"], int)
                        and isinstance(before["hp_fixed"], int)
                        and now["hp_fixed"] < before["hp_fixed"]):
                    changes.append("actual_health_consumption")
                evidence = self.observed_remaining_opportunity(raw)
                self._remaining_resource_evidence = deepcopy(evidence)
                if evidence["confirmed_gold"] or evidence["confirmed_loot"]:
                    changes.append("known_remaining_resource_currently_collectible")
                if not changes:
                    denial = "no_resource_change_since_return"
        self._last_start_denial = denial
        if denial is not None:
            return False
        self._begin_trip(raw, steps, "resource_change" if self.trip_count else
                         ("farm_cap" if not cleared else "cleared"), farm_scene_steps, changes)
        return True

    def start(self, raw, steps, trigger, *, farm_scene_steps=None):
        raise RuntimeError("sustain-loot-v1 must use maybe_start; direct start bypasses trip guards")

    def _begin_trip(self, raw, steps, trigger, farm_steps, changes):
        # Reset only inherited per-trip state; the same service and observed
        # memories remain attached to the existing observation callback.
        fresh = SustainCompletionService(self.mode, calibration=self.calibration)
        for descriptor in fields(SustainCompletionService):
            if descriptor.name != "gold_memory":
                setattr(self, descriptor.name, deepcopy(getattr(fresh, descriptor.name)))
        self.gold_sold = self.sales = self.loot_collected = 0
        self.loot_failures, self.sale_failures = [], []
        self.terminal_cash_events, self.death_cash_change = [], 0
        self._loot_attempted, self._sale_attempted = set(), set()
        self._loot_target = self._request = self._sale_town_sequence = None
        self._visited_town = self._finalize_pending = False
        self._start_gold = int(raw["gold"])
        self._trigger_changes = list(changes)
        self.trip_count += 1
        super().start(raw, steps, trigger, farm_scene_steps=farm_steps)
        self._reopened_gold = []
        if self.trip_count == 2:
            # The original memory has no per-trip retry API. Reopen only exact
            # current, non-gone identities certified above; never off-screen
            # or previously consumed candidates merely because the trip reset.
            for item in self.observed_remaining_opportunity(raw)["confirmed_gold"]:
                key = _identity(item)
                entry = self.gold_memory._items.get(key)
                if entry is not None and not entry["gone"] and entry["attempted"]:
                    self._reopened_gold.append({"identity": list(key),
                        "previous_attempt_reason": entry["attempt_reason"],
                        "confirmed_at": steps})
                    entry.update(attempted=False, attempt_reason=None, attempted_at=None)

    def command(self, env, bridge):
        if not self.active:
            raise RuntimeError("Cannot issue a command outside an active loot trip")
        if self._request is not None:
            raise RuntimeError("Previous resource command has no receipt")
        self.observe(env._raw, int(env._steps))
        before = deepcopy(self._latest)
        command = super().command(env, bridge)
        self._request = {"command": tuple(command), "before": before,
                         "clock": int(env._steps)}
        return command

    def _command(self, env, bridge):
        raw = env._raw
        # All cash credits are checked against actual command receipts below;
        # inherited wallet-difference collection must never count a sale twice.
        self._last_gold = int(raw["gold"])
        if self.phase == "outbound" and int(raw["dungeon_level"]) == 0:
            self.phase = "sell_idle"
        if self.phase == "sell_idle":
            if raw.get("is_set_level") or int(raw["dungeon_level"]) != 0:
                return self._finish("resource_unexpected_scene")
            if self.steps >= self.service_microstep_cap:
                return self._finish("resource_service_cap")
            state, town = raw["resource_state"], raw["resource_state"]["town"]
            sequence = (int(state["town_seed"]), int(state["town_restock_sequence"]))
            if self._sale_town_sequence not in (None, sequence):
                raise RuntimeError("Town generation changed inside loot sale phase")
            self._sale_town_sequence = sequence
            visit = self._visit(env, "smith", town)
            if visit:
                return visit
            owned = {_identity(i): i for i in self._latest["inventory"]
                     if i.get("empty") is not True}
            equipped = {_identity(i) for i in raw.get("equipped_items", []) if i.get("present")}
            candidates = []
            for quote in town.get("sell_quotes", []):
                identity = _identity(quote)
                item = owned.get(identity)
                if (quote.get("vendor") != "smith" or item is None or not _safe_sale_item(item)
                        or identity in equipped | self._protected | self._purchase_pending
                        or identity in self._sale_attempted):
                    continue
                index = _integer(quote["index"], "sale inventory index")
                if index != item["index"]:
                    raise RuntimeError("Sale quote inventory index differs from live inventory")
                price = _integer(quote["price"], "native sale price", 1)
                candidates.append((identity, index, price))
            if candidates:
                identity, index, price = min(candidates)
                return ("sell", "smith", index, *identity, price)
            self.phase = "survey_smith"
        if (self.phase == "collect" and self.steps < self.collect_microstep_cap
                and self.steps < self.service_microstep_cap and not raw.get("is_set_level")
                and int(raw["dungeon_level"]) == 1):
            command = self._collect_loot(env)
            if command is not None:
                return command
        return super()._command(env, bridge)

    def _collect_loot(self, env):
        raw = env._raw
        position = (int(raw["player_x"]), int(raw["player_y"]))
        distance = lambda item: max(abs(int(item["x"]) - position[0]),
                                    abs(int(item["y"]) - position[1]))
        while True:
            if self._loot_target is None:
                if self._gold_target is not None:
                    return None
                candidates = [i for identity, i in self._loot_memory.items()
                              if identity not in self._loot_attempted | self._protected
                              and i.get("can_fit") is True]
                if not candidates:
                    return None
                item = min(candidates, key=lambda i: (distance(i), _identity(i)))
                gold = self.gold_memory.candidates()
                if gold and min(distance(i) for i in gold) <= distance(item):
                    return None
                self._loot_target = deepcopy(item)
                self._gold_navigation_seen.clear()
            item = self._loot_target
            identity, target = _identity(item), (int(item["x"]), int(item["y"]))
            if position == target:
                current = next((i for i in raw["resource_state"].get("loot_items", [])
                                if _identity(i) == identity
                                and (int(i["x"]), int(i["y"])) == position), None)
                self._loot_attempted.add(identity)
                self._loot_target = None
                if current is None or not _safe_sale_item(current) or current.get("can_fit") is not True:
                    self.loot_failures.append({"identity": identity, "reason": "no_current_safe_loot_at_target"})
                    self._loot_memory.pop(identity, None)
                    continue
                return ("loot", _integer(current["active_id"], "loot handle"), *target, *identity)
            command = self._walk(env, *target)
            signature = (position, command)
            # The inherited recovery clears this guard when it deliberately
            # retries bounded combat/wait. Keep the same contract for loot.
            if command is not None and signature not in self._gold_navigation_seen:
                self._gold_navigation_seen.add(signature)
                return command
            self._loot_attempted.add(identity)
            self.loot_failures.append({"identity": identity, "reason": "route_unavailable"})
            self._loot_target = None

    def receipt(self, command, receipt):
        request = self._request
        if request is None or tuple(command) != request["command"]:
            raise RuntimeError("Resource receipt does not match the pending command")
        if not isinstance(receipt.get("accepted"), bool):
            raise RuntimeError("Resource receipt needs a native boolean accepted")
        before, after = request["before"], self._latest
        if after is None or self._observed_step < request["clock"]:
            raise RuntimeError("Resource receipt lacks an existing post-action observation")
        accepted = receipt["accepted"]
        delta = after["gold"] - before["gold"]
        kind = command[0]
        l1_death = after["dead"] and before["dungeon_level"] == 1
        if kind in ("sell", "loot"):
            if receipt.get("price") != 0:
                raise RuntimeError("Loot/sale must not report paid expenditure")
            native_gold_after = _integer(receipt.get("gold_after"), "native post-transaction gold")
            if (receipt.get("gold_before") != before["gold"]
                    or (native_gold_after != after["gold"] and not (kind == "loot" and l1_death))):
                raise RuntimeError("Loot/sale native wallet receipt mismatch")
            expected = tuple(command[3:7] if kind == "sell" else command[4:8])
            if _identity(receipt) != expected:
                raise RuntimeError("Loot/sale receipt source identity mismatch")
            received = _integer(receipt.get("received"), "native sale received")
            if kind == "sell":
                if receipt.get("index") != command[2] or receipt.get("quoted_price") != command[-1]:
                    raise RuntimeError("Sale receipt index mismatch")
                self._sale_attempted.add(expected)
                ids = {_identity(i) for i in after["inventory"] if i.get("empty") is not True}
                if accepted:
                    if received != command[-1] or delta != received or expected in ids:
                        raise RuntimeError("Accepted sale lacks exact cash/identity effect")
                    self.gold_sold += received
                    self.sales += 1
                elif received != 0 or delta != 0 or before["inventory"] != after["inventory"]:
                    raise RuntimeError("Rejected sale mutated cash or inventory")
                else:
                    self.sale_failures.append({"identity": expected, "reason": receipt.get("reason")})
            else:
                if received != 0 or native_gold_after != before["gold"]:
                    raise RuntimeError("Loot pickup incorrectly produced cash")
                if not l1_death and delta != 0:
                    raise RuntimeError("Loot pickup settlement changed cash")
                if accepted:
                    # Normal pickup clears CF_PREGEN: use the actual returned
                    # inventory identity, never pretend the source id is stable.
                    actual = _identity(receipt["inventory_item"])
                    old = {_identity(i) for i in before["inventory"] if i.get("empty") is not True}
                    new = {_identity(i) for i in after["inventory"] if i.get("empty") is not True}
                    if actual in old or (actual not in new and not l1_death):
                        raise RuntimeError("Accepted loot absent from actual inventory")
                    self.loot_collected += 1
                    self._loot_memory.pop(expected, None)
                elif before["inventory"] != after["inventory"] and not l1_death:
                    raise RuntimeError("Rejected loot changed inventory")
                else:
                    self.loot_failures.append({"identity": expected, "reason": receipt.get("reason")})
                if l1_death:
                    self.death_cash_change += after["gold"] - native_gold_after
                    self.terminal_cash_events.append({"command": list(command),
                        "reason": "native_loot_receipt_then_death_settlement",
                        "pickup_amount_known": True, "native_gold_after": native_gold_after,
                        "posttick_gold": after["gold"], "death_cash_change": after["gold"] - native_gold_after})
        elif kind == "gold":
            if l1_death:
                # This legacy API only accepts a queued pickup request; it
                # gives no synchronous amount. Death can drop gold in the
                # same tick. Neither pickup income nor gross loss is known.
                self.terminal_cash_events.append({"command": list(command),
                    "reason": "queued_gold_pickup_and_death_amount_unresolved",
                    "pickup_amount_known": False, "accepted_request": accepted,
                    "gold_before": before["gold"], "posttick_gold": after["gold"],
                    "net_observed_change": delta})
            elif delta < 0 or (not accepted and delta != 0):
                raise RuntimeError("Gold pickup cash effect disagrees with receipt")
            else:
                # accepted means the native command was accepted, not that
                # an item necessarily fit. Credit only actual positive cash.
                self.gold_collected += delta
        elif kind in ("buy", "repair"):
            price = _integer(receipt.get("price", 0), "purchase/repair price")
            if delta != (-price if accepted else 0):
                raise RuntimeError("Purchase/repair cash effect disagrees with receipt")
            if accepted and kind == "buy" and command[1] == "smith":
                self._purchase_pending.add(tuple(command[3:]))
        elif kind in ("equip", "unequip") and delta != 0:
            raise RuntimeError("Equipment mutation changed cash")
        elif l1_death and delta:
            self.death_cash_change += delta
            self.terminal_cash_events.append({"command": list(command),
                "reason": "death_settlement_cash_change", "pickup_amount_known": True,
                "gold_before": before["gold"], "posttick_gold": after["gold"],
                "death_cash_change": delta})
        super().receipt(command, receipt)
        if kind == "equip" and accepted:
            actual = tuple(command[2:])
            if not any(i.get("present") and _identity(i) == actual for i in after["equipped_items"]):
                raise RuntimeError("Accepted equipment not present in actual equipped slots")
            self._purchase_pending.discard(actual)
        self._last_gold = after["gold"]
        self._request = None
        if not self.active:
            self._finalize_pending = True

    def record_steps(self, absolute_steps, phase=None):
        super().record_steps(absolute_steps, phase)
        if self._finalize_pending:
            self._finalize_pending = False
            self._seal_trip(int(absolute_steps))

    def _seal_trip(self, steps):
        current = self._latest
        residual = current["gold"] - (self._start_gold + self.gold_collected
                                      + self.gold_sold - self.gold_spent + self.death_cash_change)
        unknown_pickup = any(not event["pickup_amount_known"] for event in self.terminal_cash_events)
        record = {"trip": self.trip_count, "start_microstep": self.start_steps,
                  "end_microstep": steps, "steps": self.steps, "reason": self.reason,
                  "end_depth": current["dungeon_level"], "dead": current["dead"],
                  "visited_town": self._visited_town, "start_gold": self._start_gold,
                  "end_gold": current["gold"], "gold_collected": self.gold_collected,
                  "gold_sold": self.gold_sold, "gold_spent": self.gold_spent,
                  "repair_gold_spent": self.repair_gold_spent, "purchases": self.purchases,
                  "repairs": self.repairs, "sales": self.sales, "loot_collected": self.loot_collected,
                  "death_cash_change": self.death_cash_change,
                  "terminal_cash_events": deepcopy(self.terminal_cash_events),
                  "gold_collected_complete": not unknown_pickup,
                  "cash_reconciliation": ("death_pickup_amount_unknown" if unknown_pickup
                                          else "complete" if residual == 0 else "unexplained_terminal_change"),
                  "cash_residual": residual, "trigger_changes": list(self._trigger_changes),
                  "reopened_current_gold": deepcopy(self._reopened_gold),
                  "phase_steps": deepcopy(self.phase_steps)}
        self._trips.append(record)
        if residual and not current["dead"]:
            raise RuntimeError("Completed loot trip has unexplained cash movement")
        if (self._visited_town and current["dungeon_level"] == 1
                and not current["is_set_level"] and not current["dead"]
                and current["player_mode"] == 0 and self.reason in ("complete", "growth_remaining")):
            self._return_baseline = self._fingerprint(current)
            self._return_step = steps

    def finish_episode(self, raw, steps, reason):
        """Archive an already settled real terminal; never alter the game."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A real episode terminal reason is required")
        self.observe(raw, _integer(steps, "terminal clock"))
        if self.trip_count == 0 or (self._trips and self._trips[-1]["trip"] == self.trip_count):
            return
        # Options calls this only after its normal receipt/step accounting. An
        # unexecuted pending request cannot be quietly fabricated as a receipt.
        if self._request is not None:
            raise RuntimeError("Episode ended with an unaccounted resource command")
        self.active = False
        self.phase, self.reason = "done", reason
        self.record_steps(steps)
        if not self._trips or self._trips[-1]["trip"] != self.trip_count:
            self._seal_trip(steps)

    def telemetry(self):
        result = super().telemetry()
        counters = ("gold_collected", "gold_sold", "gold_spent", "repair_gold_spent",
                    "purchases", "repairs", "sales", "loot_collected", "steps", "death_cash_change")
        archived_current = bool(self._trips and self._trips[-1]["trip"] == self.trip_count)
        cumulative = {k: sum(t[k] for t in self._trips)
                      + (0 if archived_current else getattr(self, k)) for k in counters}
        result.update(policy="sustain-loot-v1", trip_limit=2, trip_count=self.trip_count,
                      gold_sold=self.gold_sold, sales=self.sales, loot_collected=self.loot_collected,
                      death_cash_change=self.death_cash_change,
                      terminal_cash_events=deepcopy(self.terminal_cash_events),
                      gold_collected_complete=not any(not e["pickup_amount_known"] for e in self.terminal_cash_events),
                      loot_failures=deepcopy(self.loot_failures), sale_failures=deepcopy(self.sale_failures),
                      trips=deepcopy(self._trips), cumulative=cumulative,
                      last_start_denial=self._last_start_denial,
                      trigger_changes=list(self._trigger_changes),
                      remaining_resource_evidence=deepcopy(self._remaining_resource_evidence),
                      reopened_current_gold=deepcopy(self._reopened_gold),
                      loot_memory={"remembered_identities": len(self._loot_memory),
                                   "attempted_this_trip": len(self._loot_attempted),
                                   "last_observed_sale_value": sum(i["sell_price"] for i in self._loot_memory.values())})
        return result
