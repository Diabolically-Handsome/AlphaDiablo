"""Episode-local memory of gold actually supplied by native L1 observation.

This module neither queries the engine nor changes observations or clocks.
Remembered amounts are candidate values, never inventory or earned income.
The caller must reset this object for every new episode, including same-seed
resets, and execute all navigation and pickups through ordinary native actions.
"""
from collections import Counter
from copy import deepcopy


IDENTITY_FIELDS = ("seed_hi", "seed_lo", "create_info", "base_id")


class ObservedGoldMemory:
    def __init__(self):
        self.reset()

    @staticmethod
    def identity(item):
        """An active slot is only a current handle, never persistent identity."""
        return tuple(int(item[name]) for name in IDENTITY_FIELDS)

    @staticmethod
    def _main_l1(raw):
        state = raw.get("resource_state") or {}
        return (int(raw.get("dungeon_level", -1)) == 1
                and not raw.get("is_set_level", False)
                and state.get("enabled") is True
                and "gold_items" in state)

    def reset(self):
        self._items = {}
        self._observation_calls = 0
        self._last_observed_step = None
        self._out_of_order_observations = 0
        self._attempt_events = 0
        self._gone_confirmations = 0

    def observe(self, raw, absolute_steps):
        """Consume an existing native result; no absence inference off-screen."""
        step = int(absolute_steps)
        self._observation_calls += 1
        if self._last_observed_step is not None and step < self._last_observed_step:
            self._out_of_order_observations += 1
            return
        self._last_observed_step = step
        if not self._main_l1(raw):
            return
        for item in raw["resource_state"]["gold_items"]:
            identity = self.identity(item)
            if int(item["value"]) <= 0:
                continue
            if identity not in self._items:
                self._items[identity] = {
                    **dict(zip(IDENTITY_FIELDS, identity)),
                    "first_seen": step, "attempted": False,
                    "attempt_reason": None, "attempted_at": None,
                    "gone": False, "gone_reason": None, "gone_at": None,
                }
            entry = self._items[identity]
            entry.update(active_id=int(item["active_id"]), x=int(item["x"]),
                         y=int(item["y"]), value=int(item["value"]),
                         last_seen=step, gone=False)

    def candidates(self, *, include_attempted=False):
        """Return detached last-observed candidates, not guaranteed live piles."""
        return [deepcopy(item) for identity, item in sorted(self._items.items())
                if not item["gone"] and (include_attempted or not item["attempted"])]

    def mark_attempted(self, item, reason, absolute_steps):
        entry = self._items.get(self.identity(item))
        if entry is None or int(absolute_steps) < entry["last_seen"]:
            return False
        entry.update(attempted=True, attempt_reason=str(reason),
                     attempted_at=int(absolute_steps))
        self._attempt_events += 1
        return True

    def mark_gone(self, item, raw, absolute_steps, reason="native_absent_at_target"):
        """Confirm absence only at the remembered tile in a fresh native raw.

        A stale handle or a pile outside current illumination is not evidence
        that gold was picked up. This method never records any earned amount.
        """
        identity = self.identity(item)
        entry = self._items.get(identity)
        step = int(absolute_steps)
        if (entry is None or not self._main_l1(raw)
                or step < entry["last_seen"]
                or (self._last_observed_step is not None and step < self._last_observed_step)
                or (int(raw.get("player_x", -1)), int(raw.get("player_y", -1)))
                   != (entry["x"], entry["y"])):
            return False
        if any(self.identity(current) == identity
               for current in raw["resource_state"]["gold_items"]):
            return False
        if not entry["gone"]:
            self._gone_confirmations += 1
        entry.update(gone=True, gone_reason=str(reason), gone_at=step,
                     attempted=True, attempt_reason=str(reason), attempted_at=step)
        return True

    def telemetry(self):
        candidates = self.candidates()
        return {
            "version": "observed-l1-gold-memory-v1",
            "observation_calls": self._observation_calls,
            "last_observed_step": self._last_observed_step,
            "out_of_order_observations": self._out_of_order_observations,
            "remembered_identities": len(self._items),
            "remembered_last_observed_value": sum(i["value"] for i in self._items.values()),
            "candidate_identities": len(candidates),
            "candidate_last_observed_value": sum(i["value"] for i in candidates),
            "attempt_events": self._attempt_events,
            "attempted_identities": sum(i["attempted"] for i in self._items.values()),
            "attempt_reasons": dict(Counter(i["attempt_reason"] for i in self._items.values() if i["attempted"])),
            "gone_identities": sum(i["gone"] for i in self._items.values()),
            "native_gone_confirmations": self._gone_confirmations,
        }
