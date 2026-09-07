"""Zero-training, paired L2 resource pilot with fixed post-arrival exposure.

Every arm uses the same native hard gate. Non-arrivals never count as success.
Smoke: 16 paired native repeat runs plus a 16-seed certified-worker smoke.
Pilot: 48 already-consumed seeds x five arms; requires the PASS smoke receipt.
No observations, policy parameters, rewards or native inventories are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))
VERSION = "l2-town-resource-pilot-v1"
ARMS = ("none", "heal", "potions", "armor", "full")
SEEDS = tuple(range(2114000, 2114048))
ARRIVAL_BUDGET = 6000
FOLLOWUP_BEATS = 1800
R16 = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
CERTIFIED = ROOT / "train/runs/r9-reeducation/staging/worker.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def followup_outcome(first_arrival: int | None, final_beat: int, *, died: bool,
                     death_beat: int | None = None,
                     arrival_budget: int = ARRIVAL_BUDGET,
                     followup: int = FOLLOWUP_BEATS) -> dict:
    """Censoring is unknown, never alive; no arrival is joint failure."""
    arrived = first_arrival is not None and first_arrival <= arrival_budget
    if not arrived:
        return {"arrived_in_budget": False, "followup_complete": False,
                "followup_censored": False, "alive_at_followup": None,
                "joint_success": False}
    horizon = int(first_arrival) + followup
    if died:
        if death_beat is None:
            raise ValueError("death_beat is required for a death outcome")
        alive = death_beat > horizon
        censored = False
    else:
        censored = final_beat < horizon
        alive = None if censored else True
    return {"arrived_in_budget": True, "followup_complete": not censored,
            "followup_censored": censored, "alive_at_followup": alive,
            "joint_success": alive is True}


class ExposureLedger:
    """Read-only callback for each real native microtick under the new protocol."""
    def __init__(self, arrival_budget=ARRIVAL_BUDGET, followup=FOLLOWUP_BEATS):
        self.arrival_budget = arrival_budget
        self.followup = followup
        self.first_arrival = None
        self.first_readiness = None
        self.death_beat = None
        self.last_beat = 0
        self.last_depth = 1
        self.last_is_set_level = False
        self.max_depth = 1
        self.by_depth = Counter()
        self.transitions = []
        self.scene_changes = []
        self.town_entry_snapshots = []
        self.shop_stock_snapshots = []
        self.phase_changes = []
        self.service_state_provider = None
        self._last_service_phase = None
        self._last_shop_vendor = None
        self._last_shop_stock = None
        self.completed_town_trips = 0
        self.town_departed = False
        self.last_sequence = None
        self.readiness_counts = Counter()
        self.failure_counts = Counter()
        self.ready_seen = False
        self.last_readiness = None
        self.trace = hashlib.sha256()

    def __call__(self, env, raw, beat):
        beat = int(beat)
        if beat != self.last_beat + 1:
            raise RuntimeError(f"native microtick gap/repeat: {self.last_beat}->{beat}")
        scene_key = (f"set:{self.last_depth}" if self.last_is_set_level else str(self.last_depth))
        self.by_depth[scene_key] += 1
        state = raw.get("resource_state")
        if not isinstance(state, dict) or not state.get("enabled"):
            raise RuntimeError("enabled native resource_state missing")
        if self.service_state_provider is not None:
            service = self.service_state_provider()
            if service.get("phase") != self._last_service_phase:
                self._last_service_phase = service.get("phase")
                self.phase_changes.append({"first_observed_beat": beat,
                    "phase": self._last_service_phase,
                    "position": [raw.get("player_x"), raw.get("player_y")],
                    "dungeon_level": raw.get("dungeon_level"),
                    "service": deepcopy(service)})
        town = state.get("town") or {}
        stock = town.get("stock") or []
        vendor = town.get("active_vendor")
        if stock and (vendor != self._last_shop_vendor or stock != self._last_shop_stock):
            self.shop_stock_snapshots.append({"beat": beat, "vendor": vendor,
                "position": [raw.get("player_x"), raw.get("player_y")],
                "gold": int(raw.get("gold", 0)),
                "rng": raw.get("rng", state.get("rng")),
                "stock_sha256": canonical_hash(stock), "stock": deepcopy(stock)})
        if vendor != self._last_shop_vendor or stock != self._last_shop_stock:
            self._last_shop_vendor = vendor
            self._last_shop_stock = deepcopy(stock)
        receipt = state.get("readiness")
        if not isinstance(receipt, dict) or type(receipt.get("ready")) is not bool:
            raise RuntimeError("canonical native readiness receipt missing")
        self.last_readiness = deepcopy(receipt)
        self.ready_seen |= receipt["ready"]
        self.readiness_counts[str(receipt["ready"])] += 1
        for failure in receipt.get("failures", []):
            self.failure_counts[str(failure)] += 1
        depth = int(raw.get("dungeon_level", self.last_depth))
        dead = bool(raw.get("dead"))
        current_set = bool(raw.get("is_set_level"))
        if not dead and (depth, current_set) != (self.last_depth, self.last_is_set_level):
            self.scene_changes.append({"beat": beat, "source_depth": self.last_depth,
                                       "source_is_set": self.last_is_set_level,
                                       "target_depth": depth, "target_is_set": current_set})
            if not current_set and not self.last_is_set_level:
                if self.last_depth == 1 and depth == 0:
                    self.town_departed = True
                    self.town_entry_snapshots.append({"beat": beat,
                        "position": [raw.get("player_x"), raw.get("player_y")],
                        "gold": int(raw.get("gold", 0)),
                        "rng": raw.get("rng", state.get("rng")),
                        "resource_state": deepcopy(state)})
                elif self.last_depth == 0 and depth == 1 and self.town_departed:
                    self.completed_town_trips += 1
                    self.town_departed = False
        if dead and self.death_beat is None:
            self.death_beat = beat
        transition = state.get("transition") or {}
        sequence = transition.get("sequence")
        if sequence is not None and sequence != self.last_sequence:
            self.last_sequence = sequence
            if int(sequence) > 0:
                self.transitions.append({"beat": beat, **deepcopy(transition)})
        if (self.first_arrival is None and self.last_depth == 1 and depth == 2
                and not self.last_is_set_level and not raw.get("is_set_level")):
            if beat > self.arrival_budget:
                raise RuntimeError("L2 arrived after the frozen arrival budget")
            if transition.get("pretransition_ready") is not True:
                raise RuntimeError("L1->L2 lacks an affirmative pre-transition readiness receipt")
            self.first_arrival = beat
            self.first_readiness = deepcopy(transition)
            # Evaluation stopping rule only; the policy input remains unchanged.
            env.max_steps = beat + self.followup
        if not raw.get("is_set_level") and not dead:
            self.max_depth = max(self.max_depth, depth)
        trace_row = (beat, depth, bool(raw.get("is_set_level")), int(raw.get("player_x", -1)),
                     int(raw.get("player_y", -1)), int(raw.get("hp", 0)),
                     int(raw.get("gold", 0)), int(raw.get("belt_heals", 0)),
                     int(raw.get("monster_kill_total", 0)), receipt, transition)
        self.trace.update(json.dumps(trace_row, sort_keys=True,
                                     separators=(",", ":")).encode())
        self.last_beat = beat
        if not dead:
            self.last_depth = depth
            self.last_is_set_level = bool(raw.get("is_set_level"))

    def finish(self, raw, settled_steps):
        if int(settled_steps) != self.last_beat:
            raise RuntimeError("Python microstep accounting differs from native tick ledger")
        if self.last_beat > self.arrival_budget + self.followup:
            raise RuntimeError("pilot exceeded absolute exposure budget")
        died = bool(raw.get("dead"))
        outcome = followup_outcome(self.first_arrival, self.last_beat,
                                  died=died, death_beat=self.death_beat,
                                  arrival_budget=self.arrival_budget,
                                  followup=self.followup)
        return {**outcome, "first_l2_beat": self.first_arrival,
                "arrival_readiness_receipt": self.first_readiness,
                "micro_steps": self.last_beat, "died": died,
                "death_beat": self.death_beat,
                "died_on_l2": died and self.last_depth == 2 and not self.last_is_set_level,
                "max_depth": self.max_depth,
                "beats_by_depth": dict(self.by_depth),
                "l2_microsteps": self.by_depth.get("2", 0),
                "ready_seen": self.ready_seen,
                "last_readiness": self.last_readiness,
                "readiness_tick_counts": dict(self.readiness_counts),
                "readiness_failure_tick_counts": dict(self.failure_counts),
                "transitions": self.transitions,
                "scene_changes": self.scene_changes,
                "town_entry_snapshots": self.town_entry_snapshots,
                "shop_stock_snapshots": self.shop_stock_snapshots,
                "phase_changes": self.phase_changes,
                "completed_town_trips": self.completed_town_trips,
                "trace_sha256": self.trace.hexdigest()}


def identities(worker, certified):
    from diablogym import bridge
    from eval_contract import runtime_identity
    runtime = runtime_identity(ROOT, Path(bridge.__file__))
    return {"runtime": runtime,
            "native_sources": {name: sha256(ROOT / name)
                               for name in ("src/diablogym.cpp", "src/resource_protocol.hpp")},
            "worker_sha256": sha256(worker),
            "certified_worker_sha256": sha256(certified),
            "driver_sha256": sha256(Path(__file__))}


def load_policy(path):
    # Load from this checkout, never the legacy probe's hardcoded home path.
    import torch
    from leashed_ppo import LeashedMaskablePPO
    torch.set_num_threads(1)
    model = LeashedMaskablePPO.load(path, env=None, device="cpu",
                                   teacher_path=None, teacher_sha256=None)
    def cb(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)
    cb.on_beat = None
    cb.episode_reseed = lambda seed: model.set_random_seed(int(seed))
    cb.diablogym_worker_observation_view = "dual-v4-asymmetric-v3"
    cb.diablogym_worker_action12_mode = "environment-mask"
    return cb


class PilotEpisodeFailure(RuntimeError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def run_episode(cb, seed, mode):
    from diablogym.options_env import OptionsEnv, FARM, DIVE
    ledger = ExposureLedger()
    env = OptionsEnv(max_steps=ARRIVAL_BUDGET, workers={FARM: cb, DIVE: cb},
                     drink_sovereignty=True,
                     worker_observation_view="dual-v4-asymmetric-v3",
                     manager_observation_view="legacy-v3",
                     dive_live_sovereignty=True, reward_economy="v4",
                     explore_global_hunt=True, farm_scene_cap=3600,
                     reset_layer_clock_on_window=True,
                     resource_protocol="l2-town-v1", resource_purchase_mode=mode)
    windows = []
    try:
        env.reset(seed=seed)
        cb.episode_reseed(seed)
        ledger.service_state_provider = env.resource_service.telemetry
        env.env.resource_tick_callback = ledger
        done = trunc = False
        empty_windows = 0
        while not (done or trunc):
            before = int(env.env._steps)
            choice = env.resource_option_choice()
            if choice is None:
                raise RuntimeError("resource manager did not choose an executable action")
            masks = env.action_masks()
            if not masks[int(choice)]:
                raise RuntimeError("resource manager selected a masked action")
            _, _, done, trunc, info = env.step(int(choice))
            after = int(env.env._steps)
            extra = info.get("option_extra") or {}
            resource = extra.get("resource") or info.get("resource_protocol")
            windows.append({"option": int(choice), "before": before,
                            "after": after, "reason": extra.get("reason"),
                            "resource": deepcopy(resource)})
            empty_windows = empty_windows + 1 if after == before else 0
            if empty_windows > 8:
                raise RuntimeError("more than eight zero-tick option windows")
        raw = env.env._raw
        row = {"seed": seed, "mode": mode, **ledger.finish(raw, env.env._steps),
               "final_resource": deepcopy(info.get("resource_protocol")),
               "service": deepcopy(env.resource_service.telemetry()),
               "windows": windows, "victory": bool(raw.get("victory")),
               "char_level": int(raw.get("char_level", 0)),
               "armor_class": int(raw.get("armor_class", 0)),
               "gold": int(raw.get("gold", 0)),
               "belt_heals": int(raw.get("belt_heals", 0))}
        row["resource_failure"] = (None if row["arrived_in_budget"] else
                                   ("death_before_arrival" if row["died"] else
                                    ((row["final_resource"] or {}).get("terminal_reason")
                                     or row["service"].get("reason")
                                     or "arrival_budget_exhausted")))
        if row["followup_censored"]:
            raise RuntimeError("fixed-exposure pilot ended alive before its followup horizon")
        return row
    except Exception as exc:
        current = env.env._raw or {}
        evidence = {"seed": seed, "mode": mode, "error_type": type(exc).__name__,
                    "error": str(exc), "native_microsteps": ledger.last_beat,
                    "python_microsteps": int(env.env._steps),
                    "dungeon_level": current.get("dungeon_level"),
                    "resource_state": deepcopy(current.get("resource_state")),
                    "service": deepcopy(env.resource_service.telemetry()),
                    "last_windows": windows[-5:]}
        raise PilotEpisodeFailure(str(exc), evidence) from exc
    finally:
        cb.on_beat = None
        env.close()


def smoke_coverage(rows, certified_rows, repeat_rows):
    """A runtime without exercised town service is not an engineering PASS."""
    def completed(row):
        return (row.get("completed_town_trips", 0) >= 1
                and row.get("service", {}).get("reason") == "complete")
    result = {"r16_n": len(rows), "repeat_n": len(repeat_rows),
              "certified_n": len(certified_rows),
              "r16_effective_roundtrips": sum(completed(r) for r in rows),
              "certified_effective_roundtrips": sum(completed(r) for r in certified_rows)}
    result["pass"] = (result["r16_n"] == result["repeat_n"] == result["certified_n"] == 16
                      and result["r16_effective_roundtrips"] > 0
                      and result["certified_effective_roundtrips"] > 0)
    return result


def summarize(rows):
    result = {}
    for mode in ARMS:
        arm = [r for r in rows if r["mode"] == mode]
        if not arm:
            continue
        arrived = [r for r in arm if r["arrived_in_budget"]]
        result[mode] = {"n": len(arm),
                       "joint_successes": sum(r["joint_success"] for r in arm),
                       "arrived": len(arrived),
                       "ready_seen": sum(r["ready_seen"] for r in arm),
                       "died": sum(r["died"] for r in arm),
                       "died_on_l2": sum(r["died_on_l2"] for r in arm),
                       "censored": sum(r["followup_censored"] for r in arm),
                       "l2_microsteps": sum(r["l2_microsteps"] for r in arm),
                       "arrival_beats": [r["first_l2_beat"] for r in arrived],
                       "failure_reasons": dict(Counter(r["resource_failure"] for r in arm
                                                       if r["resource_failure"]))}
    base = {r["seed"]: r for r in rows if r["mode"] == "none"}
    for mode in ARMS[1:]:
        arm = {r["seed"]: r for r in rows if r["mode"] == mode}
        if arm and base:
            if set(arm) != set(base):
                raise ValueError("paired arm seed sets differ")
            saved = sum(arm[s]["joint_success"] and not base[s]["joint_success"] for s in arm)
            lost = sum(base[s]["joint_success"] and not arm[s]["joint_success"] for s in arm)
            result[mode]["paired_vs_none"] = {"saved": saved, "lost": lost,
                                              "net": saved-lost, "n": len(arm)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--worker", type=Path, default=R16)
    parser.add_argument("--certified-worker", type=Path, default=CERTIFIED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--engineering-receipt", type=Path)
    args = parser.parse_args()
    before = identities(args.worker, args.certified_worker)
    if args.phase == "pilot":
        if args.engineering_receipt is None:
            parser.error("pilot requires --engineering-receipt from a PASS smoke run")
        receipt = json.loads(args.engineering_receipt.read_text())
        if (receipt.get("phase") != "smoke" or receipt.get("status") != "PASS"
                or receipt.get("identity") != before):
            parser.error("engineering receipt is not PASS for this exact implementation/model")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from eval_contract import resource_service_recipe
    config = {"version": VERSION, "phase": args.phase, "protocol": "l2-town-v1",
              "arms": (["full"] if args.phase == "smoke" else list(ARMS)),
              "seeds": list(SEEDS[:16] if args.phase == "smoke" else SEEDS),
              "arrival_budget": ARRIVAL_BUDGET, "followup": FOLLOWUP_BEATS,
              "decoding": "sample", "torch_threads": 1,
              "observation_view": "dual-v4-asymmetric-v3",
              "readiness_source": "native pretransition receipt",
              "resource_service_recipes": {mode: resource_service_recipe("l2-town-v1", mode)
                                           for mode in (["full"] if args.phase == "smoke" else ARMS)},
              "training_steps": 0, "identity": before}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    rows = []
    repeat_rows = []
    certified_rows = []
    cb = load_policy(args.worker)
    status = "PASS"
    error = None
    try:
        jobs = ([(s, "full") for s in SEEDS[:16]] if args.phase == "smoke"
                else [(s, mode) for s in SEEDS for mode in ARMS])
        with (args.output_dir / "rows.jsonl").open("x") as stream:
            for seed, mode in jobs:
                row = run_episode(cb, seed, mode)
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True)+"\n")
                stream.flush()
                print(f"{args.phase} {mode} seed={seed} joint={row['joint_success']} "
                      f"L2={row['first_l2_beat']} beats={row['micro_steps']}", flush=True)
        if args.phase == "smoke":
            with (args.output_dir / "repeat_rows.jsonl").open("x") as stream:
                for original in rows:
                    repeated = run_episode(cb, original["seed"], "full")
                    repeat_rows.append(repeated)
                    stream.write(json.dumps(repeated, sort_keys=True)+"\n")
                    stream.flush()
                    if repeated != original:
                        raise RuntimeError(f"native/policy deterministic repeat mismatch seed={original['seed']}")
            certified_cb = load_policy(args.certified_worker)
            with (args.output_dir / "certified_rows.jsonl").open("x") as stream:
                for seed in SEEDS[:16]:
                    row = run_episode(certified_cb, seed, "full")
                    certified_rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True)+"\n")
                    stream.flush()
                    print(f"certified smoke seed={seed} roundtrips={row['completed_town_trips']} "
                          f"L2={row['first_l2_beat']}", flush=True)
            if not smoke_coverage(rows, certified_rows, repeat_rows)["pass"]:
                raise RuntimeError("insufficient exercised town roundtrip coverage in R16/certified smoke")
    except Exception as exc:
        status, error = "FAIL", f"{type(exc).__name__}: {exc}"
        if isinstance(exc, PilotEpisodeFailure):
            (args.output_dir / "failure.json").write_text(json.dumps(exc.evidence, indent=2))
    after = identities(args.worker, args.certified_worker)
    if before != after:
        status, error = "FAIL", "source/native/weights identity drifted during run"
    report = {**config, "status": status, "error": error,
              "summary": summarize(rows), "rows_sha256": canonical_hash(rows),
              "repeat_count": len(repeat_rows),
              "certified_smoke_summary": summarize(certified_rows),
              "certified_rows_sha256": canonical_hash(certified_rows),
              "smoke_coverage": (smoke_coverage(rows, certified_rows, repeat_rows)
                                 if args.phase == "smoke" else None),
              "training_authorized": False,
              "engineering_receipt_sha256": (sha256(args.engineering_receipt)
                                               if args.engineering_receipt else None),
              "limitations": ["No-arrival is joint failure, never a survival success.",
                              "Conditional survival is selected by the hard gate.",
                              "Five arms estimate resource policy effects, not isolated item physiology.",
                              "This is a zero-training engineering/development probe, not certification."]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": status, "summary": report["summary"], "error": error}), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
