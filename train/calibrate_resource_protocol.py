"""Diagnostic-only route, trigger-time and economic calibration; never training.

Production readiness and its 600-microstep service cap remain unchanged. The
2400/10000 diagnostic stops on the first real town return, regardless of ready.
The legacy ledger is reused only for native microtick/receipt audits, never its
L2-followup outcome or success calculation.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import multiprocessing
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))
import probe_resource_protocol as audit
from eval_contract import resource_service_recipe

VERSION = "resource-route-calibration-v1"
SERVICE_CAP = 2400
EPISODE_CAP = 10000
SEEDS = audit.SEEDS
ECONOMIC_COMMANDS = {"gold", "talk", "dismiss", "buy", "equip", "repair"}


def economic_snapshot(raw, beat, stage, *, command=None, receipt=None, command_id=None):
    return {"beat": int(beat), "stage": stage, "command_id": command_id,
            "command": list(command) if command is not None else None,
            "receipt": deepcopy(receipt), "gold": int(raw.get("gold", 0)),
            "hp": raw.get("hp"), "max_hp": raw.get("max_hp"),
            "position": [raw.get("player_x"), raw.get("player_y")],
            "dungeon_level": raw.get("dungeon_level"),
            "equipped_items": deepcopy(raw.get("equipped_items", [])),
            "belt_heal_kinds": deepcopy(raw.get("belt_heal_kinds", [])),
            "resource_state": deepcopy(raw.get("resource_state", {}))}


def calibration_outcome(*, stop_reason, service_start, return_beat, final_beat,
                        died, readiness, service_cap):
    returned = return_beat is not None
    administrative = stop_reason in ("resource_service_cap", "episode_cap")
    return {"diagnostic_only": True, "formal_metric_eligible": False,
            "stop_reason": stop_reason, "service_start_beat": service_start,
            "first_town_return_beat": return_beat,
            "actual_return_observed": returned,
            "ready_on_return": readiness.get("ready") if returned else None,
            "service_microsteps_observed": (int(final_beat) - service_start
                                             if service_start is not None else None),
            "complete_service_microsteps": (return_beat - service_start
                                             if returned and service_start is not None else None),
            "route_right_censored": bool(not returned and not died and administrative),
            "pre_service_death": bool(died and service_start is None),
            "route_competing_death": bool(died and not returned and service_start is not None),
            "diagnostic_return_within_production_cap": bool(
                returned and service_start is not None and return_beat - service_start <= 600),
            "service_microstep_cap": service_cap}


class CalibrationLedger(audit.ExposureLedger):
    """Reuse native audit observations; supply independent stopping/outcomes."""
    def __init__(self, episode_cap):
        super().__init__(arrival_budget=episode_cap, followup=0)
        self.episode_cap = episode_cap
        self.return_beat = None
        self.town_entry_beat = None
        self.service_start_beat = None
        self.stop_reason = None
        self.economic_snapshots = []
        self.economic_events = []
        self.resource_commands = []
        self._economic_signature = None
        self._economic_snapshot_id = None
        self._script_snapshot_taken = False
        self.service_object = None
        self.extend_service_horizon = False
        self.horizon_switch_beat = None
        self.worker_calls = 0
        self.prefix_target = None
        self.production_prefix = None
        self.policy_prefix_at_1800 = None

    def __call__(self, env, raw, beat):
        previous_trips = self.completed_town_trips
        super().__call__(env, raw, beat)
        if int(beat) == 1800:
            self.policy_prefix_at_1800 = self.trace.hexdigest()
        if self.prefix_target is not None and int(beat) == self.prefix_target:
            self.production_prefix = self.prefix_projection(raw)
        service = self.service_object
        if service is not None and service.attempted and self.service_start_beat is None:
            self.service_start_beat = int(service.start_steps)
        state = raw["resource_state"]
        self.capture_economic(raw, beat, "observed_economic_change")
        if self.town_entry_snapshots and self.town_entry_beat is None:
            self.town_entry_beat = self.town_entry_snapshots[0]["beat"]
            self.capture_economic(raw, beat, "town_entry", marker=True)
        if self.completed_town_trips > previous_trips:
            receipt = state.get("transition", {})
            if (not service or not service.attempted or receipt.get("accepted") is not True
                    or receipt.get("source_depth") != 0 or receipt.get("target_depth") != 1
                    or receipt.get("source_is_set") or receipt.get("target_is_set")):
                raise RuntimeError("return lacks an accepted canonical main-town-to-L1 receipt")
            self.return_beat = int(beat)
            self.stop_reason = "first_town_return"
            env.max_steps = int(beat)
        elif self.first_arrival is not None:
            # An unexpected early L2 is a separate endpoint, never a success.
            self.stop_reason = "unexpected_early_l2"
            env.max_steps = int(beat)

    def capture_economic(self, raw, beat, stage, *, marker=False):
        state = raw.get("resource_state") or {}
        # Position and the currently illuminated ground-item list are not
        # economic changes. Command events retain exact positions separately.
        signature = audit.canonical_hash((raw.get("gold"), raw.get("hp"), raw.get("max_hp"),
            raw.get("equipped_items"), raw.get("belt_heal_kinds"), raw.get("dungeon_level"),
            state.get("readiness"), state.get("inventory_items"), state.get("town"),
            state.get("town_seed"), state.get("town_restock_sequence")))
        changed = signature != self._economic_signature
        if changed:
            self._economic_signature = signature
            self._economic_snapshot_id = len(self.economic_snapshots)
            snapshot = economic_snapshot(raw, beat, stage)
            snapshot["snapshot_id"] = self._economic_snapshot_id
            self.economic_snapshots.append(snapshot)
        if changed or marker:
            self.economic_events.append({"beat": int(beat), "stage": stage,
                "snapshot_id": self._economic_snapshot_id,
                "position": [raw.get("player_x"), raw.get("player_y")]})
        return self._economic_snapshot_id

    def prefix_projection(self, raw):
        return {"trace_sha256": self.trace.hexdigest(), "micro_steps": self.last_beat,
                "gold": int(raw.get("gold", 0)), "belt_heals": int(raw.get("belt_heals", 0)),
                "armor_class": int(raw.get("armor_class", 0)),
                "char_level": int(raw.get("char_level", 0)),
                "last_readiness": deepcopy(raw["resource_state"]["readiness"]),
                "transitions": deepcopy(self.transitions), "scene_changes": deepcopy(self.scene_changes),
                "shop_stock_snapshots": deepcopy(self.shop_stock_snapshots)}

    def finish_calibration(self, raw, settled_steps, service, service_cap):
        if int(settled_steps) != self.last_beat:
            raise RuntimeError("Python/native calibration microstep accounting differs")
        if self.last_beat > self.episode_cap:
            raise RuntimeError("calibration exceeded its absolute episode cap")
        if self.return_beat is not None and self.last_beat != self.return_beat:
            raise RuntimeError("calibration advanced after the first real town return")
        if service.attempted:
            self.service_start_beat = int(service.start_steps)
        died = bool(raw.get("dead"))
        reason = self.stop_reason or (("pre_service_death" if not service.attempted
                                       else "death_during_service") if died else service.reason or
            ("no_service_trigger_before_episode_cap" if not service.attempted else "episode_cap"))
        ready = deepcopy(raw["resource_state"]["readiness"])
        return {**calibration_outcome(stop_reason=reason,
                    service_start=self.service_start_beat, return_beat=self.return_beat,
                    final_beat=self.last_beat, died=died, readiness=ready,
                    service_cap=service_cap),
                "micro_steps": self.last_beat, "died": died, "death_beat": self.death_beat,
                "base_horizon_switch_beat": self.horizon_switch_beat,
                "policy_calls": self.worker_calls, "policy_calls_after_service_start": 0,
                "production_prefix": self.production_prefix,
                "policy_prefix_at_1800": self.policy_prefix_at_1800,
                "ready_seen": self.ready_seen, "last_readiness": ready,
                "town_entry_beat": self.town_entry_beat,
                "town_microsteps_to_return": (self.return_beat - self.town_entry_beat
                    if self.return_beat is not None and self.town_entry_beat is not None else None),
                "first_l2_beat": self.first_arrival, "max_depth": self.max_depth,
                "beats_by_depth": dict(self.by_depth), "l2_microsteps": self.by_depth.get("2", 0),
                "readiness_failure_tick_counts": dict(self.failure_counts),
                "transitions": self.transitions, "scene_changes": self.scene_changes,
                "town_entry_snapshots": self.town_entry_snapshots,
                "shop_stock_snapshots": self.shop_stock_snapshots,
                "phase_changes": self.phase_changes,
                "completed_town_trips": self.completed_town_trips,
                "trace_sha256": self.trace.hexdigest(),
                "economic_snapshots": self.economic_snapshots,
                "economic_events": self.economic_events,
                "resource_commands": self.resource_commands}


def capture_resource_commands(inner, ledger):
    original = inner._execute_resource_command
    sequence = 0
    def execute(command):
        nonlocal sequence
        sequence += 1
        before = int(inner._resource_actual_microsteps)
        if ledger.extend_service_horizon and ledger.horizon_switch_beat is None:
            if ledger.service_object is None or not ledger.service_object.attempted:
                raise RuntimeError("cannot extend horizon before the pure resource script starts")
            ledger.horizon_switch_beat = before
            inner.max_steps = ledger.episode_cap
        if not ledger._script_snapshot_taken:
            ledger._script_snapshot_taken = True
            ledger.capture_economic(inner._raw, before, "service_start", marker=True)
        before_position = [inner._raw.get("player_x"), inner._raw.get("player_y")]
        before_snapshot = ledger.capture_economic(inner._raw, before, "before_command")
        raw, beats, receipt = original(command)
        after = int(inner._resource_actual_microsteps)
        if after - before != int(beats):
            raise RuntimeError("resource command microsteps differ from actual native steps")
        after_snapshot = ledger.capture_economic(raw, after, "after_command")
        ledger.resource_commands.append({"command_id": sequence, "command": list(command),
            "before_beat": before, "after_beat": after, "micro_steps": int(beats),
            "before_position": before_position,
            "after_position": [raw.get("player_x"), raw.get("player_y")],
            "before_snapshot_id": before_snapshot, "after_snapshot_id": after_snapshot,
            "receipt": deepcopy(receipt)})
        return raw, beats, receipt
    inner._execute_resource_command = execute


def run_episode(cb, seed, model_name, farm_trigger, calibration_id, *, prefix=False, baseline=None):
    from diablogym.options_env import OptionsEnv, FARM, DIVE
    from diablogym.resource_protocol import ResourceCalibration
    service_cap, episode_cap = (600, 6000) if prefix else (SERVICE_CAP, EPISODE_CAP)
    ledger = CalibrationLedger(episode_cap)
    config = ResourceCalibration(calibration_id=calibration_id,
        service_microstep_cap=service_cap, farm_scene_microstep_cap=farm_trigger)
    ledger.extend_service_horizon = not prefix
    ledger.prefix_target = int(baseline["micro_steps"]) if baseline is not None else None
    def guarded_policy(obs, mask):
        if ledger.service_object is not None and ledger.service_object.attempted:
            raise RuntimeError("policy called after the resource script began")
        ledger.worker_calls += 1
        return cb(obs, mask)
    guarded_policy.on_beat = None
    guarded_policy.diablogym_worker_observation_view = cb.diablogym_worker_observation_view
    guarded_policy.diablogym_worker_action12_mode = cb.diablogym_worker_action12_mode
    env = OptionsEnv(max_steps=6000, workers={FARM: guarded_policy, DIVE: guarded_policy},
        drink_sovereignty=True, worker_observation_view="dual-v4-asymmetric-v3",
        manager_observation_view="legacy-v3", dive_live_sovereignty=True,
        reward_economy="v4", explore_global_hunt=True, farm_scene_cap=3600,
        reset_layer_clock_on_window=True, resource_protocol="l2-town-v1",
        resource_purchase_mode="full", resource_calibration=config)
    windows = []
    info = {}
    try:
        env.reset(seed=seed)
        cb.episode_reseed(seed)
        ledger.service_object = env.resource_service
        ledger.service_state_provider = env.resource_service.telemetry
        env.env.resource_tick_callback = ledger
        capture_resource_commands(env.env, ledger)
        ledger.capture_economic(env.env._raw, 0, "episode_start", marker=True)
        done = trunc = False
        empty_windows = 0
        while not (done or trunc or ledger.stop_reason):
            state = env.env._raw
            choice = env.resource_option_choice()
            if choice is None or not env.action_masks()[int(choice)]:
                raise RuntimeError("calibration manager selected no executable option")
            if (int(choice) == DIVE and not state.get("is_set_level")
                    and int(state["dungeon_level"]) == 1
                    and state["resource_state"]["readiness"]["ready"]
                    and not env.resource_service.active):
                ledger.stop_reason = "ready_without_trip"
                break
            before = int(env.env._steps)
            _, _, done, trunc, info = env.step(int(choice))
            after = int(env.env._steps)
            extra = info.get("option_extra") or {}
            windows.append({"option": int(choice), "before": before, "after": after,
                "reason": extra.get("reason"), "resource": deepcopy(extra.get("resource"))})
            empty_windows = empty_windows + 1 if before == after else 0
            if empty_windows > 8:
                raise RuntimeError("calibration has more than eight empty option windows")
            # Measure one resource attempt only, including a zero-tick finish.
            if env.resource_service.attempted and not env.resource_service.active:
                break
        state = env.env._raw
        if env.max_steps != 6000:
            raise RuntimeError("calibration changed the Options/policy observation horizon")
        ledger.capture_economic(state, ledger.last_beat, "episode_endpoint", marker=True)
        if baseline is not None:
            differences = compare_prefix(ledger.production_prefix or {}, baseline)
            if differences:
                raise RuntimeError(f"production 600-microstep prefix changed: {differences}")
        return {"seed": seed, "model": model_name, "mode": "full",
                "farm_trigger_microsteps": farm_trigger, "episode_microstep_cap": episode_cap,
                **ledger.finish_calibration(state, env.env._steps,
                                            env.resource_service, service_cap),
                "service": deepcopy(env.resource_service.telemetry()),
                "final_resource": deepcopy(info.get("resource_protocol")), "windows": windows,
                "gold": int(state.get("gold", 0)), "belt_heals": int(state.get("belt_heals", 0)),
                "armor_class": int(state.get("armor_class", 0)),
                "char_level": int(state.get("char_level", 0))}
    except Exception as exc:
        current = env.env._raw or {}
        raise audit.PilotEpisodeFailure(str(exc), {"seed": seed, "model": model_name,
            "farm_trigger_microsteps": farm_trigger, "native_microsteps": ledger.last_beat,
            "python_microsteps": int(env.env._steps), "error": str(exc),
            "resource_state": deepcopy(current.get("resource_state")),
            "service": deepcopy(env.resource_service.telemetry()),
            "economic_snapshots": ledger.economic_snapshots,
            "economic_events": ledger.economic_events,
            "resource_commands": ledger.resource_commands,
            "last_windows": windows[-5:]}) from exc
    finally:
        cb.on_beat = None
        env.close()


def identities(worker, certified):
    result = audit.identities(worker, certified)
    result["exposure_audit_driver_sha256"] = result.pop("driver_sha256")
    result["calibration_driver_sha256"] = audit.sha256(Path(__file__))
    return result


def summarize(rows):
    result = {}
    for model, trigger in sorted({(r["model"], r["farm_trigger_microsteps"]) for r in rows}):
        group = [r for r in rows if (r["model"], r["farm_trigger_microsteps"]) == (model, trigger)]
        observed = [r["complete_service_microsteps"] for r in group if r["actual_return_observed"]]
        result[f"{model}:trigger={trigger}"] = {
            "n": len(group), "observed_returns": len(observed),
            "ready_on_return": sum(r["ready_on_return"] is True for r in group),
            "right_censored_routes": sum(r["route_right_censored"] for r in group),
            "pre_service_deaths": sum(r["pre_service_death"] for r in group),
            "competing_deaths": sum(r["route_competing_death"] for r in group),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in group)),
            "completed_service_microsteps_only": sorted(observed),
            "return_readiness_failures": dict(Counter(f for r in group
                if r["actual_return_observed"] for f in r["last_readiness"].get("failures", []))),
            "gold_spent": sum(r["service"]["gold_spent"] for r in group)}
    return result


def paired_trigger_comparison(rows):
    result = {}
    for model in sorted({r["model"] for r in rows}):
        late = {r["seed"]: r for r in rows if r["model"] == model and r["farm_trigger_microsteps"] == 3600}
        early = {r["seed"]: r for r in rows if r["model"] == model and r["farm_trigger_microsteps"] == 1800}
        shared = sorted(set(late) & set(early))
        if not shared:
            continue
        mismatches = []
        for seed in shared:
            a, b = late[seed], early[seed]
            if a["policy_prefix_at_1800"] is not None or b["policy_prefix_at_1800"] is not None:
                equal = a["policy_prefix_at_1800"] == b["policy_prefix_at_1800"]
            else:
                equal = a["micro_steps"] == b["micro_steps"] and a["trace_sha256"] == b["trace_sha256"]
            if not equal:
                mismatches.append(seed)
        result[model] = {"paired_n": len(shared), "common_policy_prefix_mismatches": mismatches,
            "early_return_only": sum(early[s]["actual_return_observed"] and not late[s]["actual_return_observed"] for s in shared),
            "late_return_only": sum(late[s]["actual_return_observed"] and not early[s]["actual_return_observed"] for s in shared),
            "early_ready_return_only": sum(early[s]["ready_on_return"] is True and late[s]["ready_on_return"] is not True for s in shared),
            "late_ready_return_only": sum(late[s]["ready_on_return"] is True and early[s]["ready_on_return"] is not True for s in shared)}
    return result


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def compare_prefix(row, baseline):
    # Python administrative calibration labels intentionally differ. Native
    # trace, actual exposure, readiness and market snapshots must match.
    keys = ("trace_sha256", "micro_steps", "gold", "belt_heals", "armor_class", "char_level",
            "last_readiness", "transitions", "scene_changes", "shop_stock_snapshots")
    return [key for key in keys if row.get(key) != baseline.get(key)]


def load_baseline(directory, identity):
    path = directory / "summary.json"
    old = json.loads(path.read_text())
    previous = old["identity"]
    for key in ("worker_sha256", "certified_worker_sha256", "native_sources"):
        if previous.get(key) != identity.get(key):
            raise ValueError(f"baseline identity differs: {key}")
    for key in ("bridge", "engine"):
        if previous["runtime"][key]["sha256"] != identity["runtime"][key]["sha256"]:
            raise ValueError(f"baseline native binary differs: {key}")
    for key in ("game_data", "assets"):
        if previous["runtime"]["content"][key]["sha256"] != identity["runtime"]["content"][key]["sha256"]:
            raise ValueError(f"baseline game content differs: {key}")
    if old.get("protocol") != "l2-town-v1" or old.get("arrival_budget") != 6000:
        raise ValueError("baseline is not the original resource 6000-horizon protocol")
    baseline, files = {}, {"summary.json": audit.sha256(path)}
    for model, name, expected in (("r16", "rows.jsonl", "rows_sha256"),
                                  ("certified", "certified_rows.jsonl", "certified_rows_sha256")):
        rows = read_rows(directory / name)
        if audit.canonical_hash(rows) != old.get(expected):
            raise ValueError(f"baseline rows differ from their receipt: {name}")
        files[name] = audit.sha256(directory / name)
        for row in rows:
            if row.get("mode") != "full":
                raise ValueError("baseline contains a different service arm")
            baseline[(model, row["seed"])] = row
    if any((model, seed) not in baseline for model in ("r16", "certified") for seed in SEEDS[:16]):
        raise ValueError("baseline needs the original 16 seeds for both models")
    return baseline, files


def load_receipt(path, identity, phase, calibration_id):
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "PASS" or receipt.get("phase") != phase
            or receipt.get("identity") != identity or receipt.get("calibration_id") != calibration_id):
        raise ValueError(f"matching PASS {phase} calibration receipt is required")
    return receipt


_PROCESS_POLICIES = None


def initialize_policy_process(worker, certified):
    global _PROCESS_POLICIES
    _PROCESS_POLICIES = {"r16": audit.load_policy(Path(worker)),
                         "certified": audit.load_policy(Path(certified))}


def execute_calibration_job(payload):
    try:
        model, seed, trigger = payload["job"]
        row = run_episode(_PROCESS_POLICIES[model], seed, model, trigger,
                          payload["calibration_id"], prefix=payload["prefix"],
                          baseline=payload["baseline"])
        return {"ok": True, "row": row}
    except Exception as exc:
        # Native episode failure exceptions are not assumed pickle-safe.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "evidence": getattr(exc, "evidence", {"job": payload["job"], "error": str(exc)})}


def ordered_bounded_results(executor, function, payloads, max_in_flight):
    """Preserve job order, bound submissions, and observe any failure before refill."""
    pending = {}
    submitted = emitted = 0
    def checked(future):
        result = future.result()
        if result.get("ok") is not True:
            raise audit.PilotEpisodeFailure(result.get("error", "process job failed"),
                                             result.get("evidence", {}))
        return result["row"]
    try:
        while emitted < len(payloads):
            # A later-index failure must stop refilling even while an earlier
            # job is slow. This keeps failure from launching the remainder.
            for future in pending.values():
                if future.done():
                    checked(future)
            while submitted < len(payloads) and len(pending) < max_in_flight:
                payload = payloads[submitted]
                if payload.get("reused_row") is not None:
                    future = concurrent.futures.Future()
                    future.set_result({"ok": True, "row": deepcopy(payload["reused_row"])})
                else:
                    future = executor.submit(function, payload)
                pending[submitted] = future
                submitted += 1
                for existing in pending.values():
                    if existing.done():
                        checked(existing)
            target = pending[emitted]
            if not target.done():
                concurrent.futures.wait([f for f in pending.values() if not f.done()],
                                        return_when=concurrent.futures.FIRST_COMPLETED)
                continue
            row = checked(target)
            del pending[emitted]
            emitted += 1
            yield row
    finally:
        for future in pending.values():
            future.cancel()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("check", "prefix", "route", "timing"))
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1,
                        help="spawn processes, bounded in flight; one Torch thread each")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=audit.R16)
    parser.add_argument("--certified-worker", type=Path, default=audit.CERTIFIED)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--prefix-receipt", type=Path)
    parser.add_argument("--route-receipt", type=Path)
    args = parser.parse_args(argv)
    before = identities(args.worker, args.certified_worker)
    baseline = {}
    reused = []
    baseline_dir = args.baseline_dir
    baseline_files = None
    if args.phase in ("check", "prefix"):
        if args.baseline_dir is None:
            parser.error("check/prefix requires the archived v4 --baseline-dir")
        baseline, baseline_files = load_baseline(args.baseline_dir, before)
    elif args.phase == "route":
        if args.prefix_receipt is None:
            parser.error("route requires --prefix-receipt")
        prefix_receipt = load_receipt(args.prefix_receipt, before, "prefix", args.calibration_id)
        baseline_dir = Path(prefix_receipt["baseline_dir"])
        baseline, baseline_files = load_baseline(baseline_dir, before)
        if baseline_files != prefix_receipt.get("baseline_files"):
            parser.error("archived baseline changed after prefix verification")
    else:
        if args.route_receipt is None:
            parser.error("timing requires --route-receipt")
        receipt = load_receipt(args.route_receipt, before, "route", args.calibration_id)
        reused = read_rows(args.route_receipt.parent / "rows.jsonl")
        if receipt.get("rows_sha256") != audit.canonical_hash(reused):
            parser.error("route rows do not match the frozen receipt")
    if args.output_dir.exists():
        parser.error("output-dir must not exist; preserve prior calibration evidence")
    args.output_dir.mkdir(parents=True)
    config = {"version": VERSION, "phase": args.phase, "calibration_id": args.calibration_id,
        "identity": before, "diagnostic_only": True, "formal_metric_eligible": False,
        "training_steps": 0, "production_service_cap": 600,
        "service_microstep_cap": 600 if args.phase == "prefix" else SERVICE_CAP,
        "episode_microstep_cap": 6000 if args.phase == "prefix" else EPISODE_CAP,
        "initial_episode_horizon": 6000, "worker_observation_horizon": 6000,
        "horizon_extension": "base-only at first pure resource command; no later policy call",
        "observation_view": "dual-v4-asymmetric-v3", "decoding": "sampled-reseed-per-episode",
        "torch_threads": 1, "workers": args.workers, "process_start_method": "spawn",
        "max_in_flight": args.workers, "repeat_execution": "serial in independent main-process policy",
        "production_recipe": resource_service_recipe("l2-town-v1", "full"),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "baseline_files": baseline_files,
        "prefix_receipt_sha256": audit.sha256(args.prefix_receipt) if args.prefix_receipt else None,
        "route_receipt_sha256": audit.sha256(args.route_receipt) if args.route_receipt else None,
        "reused_rows": len(reused)}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    rows, repeated = [], []
    status, error = "PASS", None
    try:
        if args.phase == "check":
            jobs = [("r16", seed, 3600) for seed in (2114013, 2114004, 2114002)]
        elif args.phase in ("prefix", "route"):
            jobs = [(model, seed, 3600) for model in ("r16", "certified") for seed in SEEDS[:16]]
        else:
            old_rows = {(r["model"], r["seed"], r["farm_trigger_microsteps"]): r for r in reused}
            jobs = [("r16", seed, trigger) for seed in SEEDS for trigger in (3600, 1800)]
            jobs += [("certified", seed, trigger) for seed in SEEDS[:16] for trigger in (3600, 1800)]
        payloads = [{"job": (model, seed, trigger), "calibration_id": args.calibration_id,
            "prefix": args.phase == "prefix", "baseline": baseline.get((model, seed)),
            "reused_row": old_rows.get((model, seed, trigger)) if args.phase == "timing" else None}
            for model, seed, trigger in jobs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_policy_process,
                initargs=(str(args.worker), str(args.certified_worker))) as executor, \
                (args.output_dir / "rows.jsonl").open("x") as stream:
            for row in ordered_bounded_results(executor, execute_calibration_job,
                                                payloads, args.workers):
                model, seed, trigger = row["model"], row["seed"], row["farm_trigger_microsteps"]
                rows.append(row)
                stream.write(json.dumps(row, sort_keys=True)+"\n")
                stream.flush()
                if args.phase == "prefix":
                    differences = compare_prefix(row, baseline[(model, seed)])
                    if differences:
                        raise RuntimeError(f"600/6000 baseline mismatch {(model, seed, trigger)}: {differences}")
                print(f"{args.phase} {model} seed={seed} trigger={trigger} "
                      f"return={row['actual_return_observed']} ready={row['ready_on_return']} "
                      f"steps={row['service_microsteps_observed']} stop={row['stop_reason']}", flush=True)
        if args.phase == "route":
            repeat_policy = audit.load_policy(args.worker)
            with (args.output_dir / "repeat_rows.jsonl").open("x") as stream:
                for original in (r for r in rows if r["model"] == "r16"):
                    row = run_episode(repeat_policy, original["seed"], "r16", 3600,
                                      args.calibration_id, baseline=baseline[("r16", original["seed"])])
                    repeated.append(row)
                    stream.write(json.dumps(row, sort_keys=True)+"\n")
                    stream.flush()
                    if row != original:
                        raise RuntimeError(f"route deterministic repeat mismatch seed={row['seed']}")
            if any(not any(r["actual_return_observed"] for r in rows if r["model"] == model)
                   for model in ("r16", "certified")):
                status, error = "INCOMPLETE_COVERAGE", "no measured real town return for one or both models"
    except Exception as exc:
        status, error = "FAIL", f"{type(exc).__name__}: {exc}"
        if isinstance(exc, audit.PilotEpisodeFailure):
            (args.output_dir / "failure.json").write_text(json.dumps(exc.evidence, indent=2))
    if before != identities(args.worker, args.certified_worker):
        status, error = "FAIL", "calibration source/native/weights identity changed during run"
    paired = paired_trigger_comparison(rows)
    if any(group["common_policy_prefix_mismatches"] for group in paired.values()):
        status, error = "FAIL", "paired policies differ before their common 1800-microstep prefix"
    report = {**config, "status": status, "error": error, "summary": summarize(rows),
        "paired_trigger_comparison": paired,
        "rows_sha256": audit.canonical_hash(rows), "repeat_count": len(repeated),
        "repeated_rows_sha256": audit.canonical_hash(repeated),
        "limitations": ["No L2 survival or formal 600-budget success is measured.",
            "Worker observations retain horizon 6000; the base cap expands to 10000 only after policy stops.",
            "Observed route durations exclude censored routes; report both counts.",
            "Economic claims use only each episode's observed quotes; unvisited merchants remain unknown.",
            "The trigger comparison is diagnostic and cannot establish a safe readiness threshold."]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": status, "summary": report["summary"], "error": error}), flush=True)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
