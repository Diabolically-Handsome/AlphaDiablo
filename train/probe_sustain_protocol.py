"""Zero-training sustain-v2 evaluation with real 6000 + 1800 exposure.

This new driver compares current-native legacy-v1 with sustain-v2. It does not
load a calibration, stop on a town return, alter readiness, or train a policy.
Check is a bounded engineering diagnostic; smoke exercises both fixed workers
and independent serial repeats; pilot requires the same-identity PASS smoke.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))
import probe_resource_protocol as audit

VERSION = "sustain-fixed-exposure-probe-v1"
POLICIES = ("legacy-v1", "sustain-v2")
MODELS = ("r16", "certified")
SEEDS = audit.SEEDS
ARRIVAL_BUDGET = 6000
FOLLOWUP_BEATS = 1800
KNOWN_CHECK_JOBS = (("r16",2114004,"sustain-v2"),
                    ("r16",2114006,"sustain-v2"),
                    ("certified",2114005,"sustain-v2"))
VOLATILE_KEYS = frozenset({"wall_time_seconds", "pid", "process_id", "job_archive"})


def identities(worker, certified):
    # Bind current binaries/content, not an archived native hash. The two
    # service implementations are compared inside this SAME new runtime.
    result = audit.identities(worker, certified)
    result["exposure_audit_driver_sha256"] = result.pop("driver_sha256")
    result["sustain_driver_sha256"] = audit.sha256(Path(__file__))
    result["sustain_sources"] = {name:audit.sha256(ROOT / name) for name in (
        "python/diablogym/resource_navigation.py",
        "python/diablogym/resource_sustain.py")}
    return result


def comparable_row(value):
    """Strip only explicitly volatile execution metadata, never state fields."""
    if isinstance(value, dict):
        return {key:comparable_row(child) for key,child in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [comparable_row(child) for child in value]
    return value


def compact_combat(raw):
    fields=("id","type","x","y","future_x","future_y","hp","max_hp",
            "hp_fixed_hi","hp_fixed_lo","visible","is_invalid",
            "rnd_item_seed_hi","rnd_item_seed_lo")
    return {"player_mode":raw.get("player_mode"),
            "player_future":[raw.get("future_x"),raw.get("future_y")],
            "monster_kill_total":raw.get("monster_kill_total"),
            "visible_monsters":[{key:monster.get(key) for key in fields}
                                for monster in raw.get("monsters",()) if monster.get("visible")]}


def economic_snapshot(raw, beat, stage):
    return {"beat":int(beat),"stage":stage,"gold":int(raw.get("gold",0)),
            "hp":raw.get("hp"),"max_hp":raw.get("max_hp"),
            "position":[raw.get("player_x"),raw.get("player_y")],
            "dungeon_level":raw.get("dungeon_level"),
            "equipped_items":deepcopy(raw.get("equipped_items",[])),
            "belt_heal_kinds":deepcopy(raw.get("belt_heal_kinds",[])),
            "resource_state":deepcopy(raw.get("resource_state",{}))}


class SustainLedger(audit.ExposureLedger):
    """Native clock and economic facts; town return is an observation, not stop."""
    def __init__(self, arrival_budget=ARRIVAL_BUDGET, followup=FOLLOWUP_BEATS):
        super().__init__(arrival_budget=arrival_budget,followup=followup)
        self.economic_snapshots=[]
        self.economic_events=[]
        self.resource_commands=[]
        self._economic_signature=None
        self._economic_snapshot_id=None
        self.service_object=None
        self.first_town_return_beat=None
        self.readiness_on_first_return=None
        self.service_start_beat=None
        self.policy_calls=0
        self.policy_calls_during_service=0
        self.policy_calls_while_service_queued=0
        self.script_started_beat=None

    def capture_economic(self, raw, beat, stage, *, marker=False):
        state=raw.get("resource_state") or {}
        signature=audit.canonical_hash((raw.get("gold"),raw.get("hp"),raw.get("max_hp"),
            raw.get("equipped_items"),raw.get("belt_heal_kinds"),raw.get("dungeon_level"),
            state.get("readiness"),state.get("inventory_items"),state.get("inventory_equipment"),
            state.get("town"),state.get("town_seed"),state.get("town_restock_sequence")))
        changed=signature != self._economic_signature
        if changed:
            self._economic_signature=signature
            self._economic_snapshot_id=len(self.economic_snapshots)
            snapshot=economic_snapshot(raw,beat,stage)
            snapshot["snapshot_id"]=self._economic_snapshot_id
            self.economic_snapshots.append(snapshot)
        if changed or marker:
            self.economic_events.append({"beat":int(beat),"stage":stage,
                "snapshot_id":self._economic_snapshot_id,
                "position":[raw.get("player_x"),raw.get("player_y")]})
        return self._economic_snapshot_id

    def __call__(self, env, raw, beat):
        old_arrival=self.first_arrival
        old_trips=self.completed_town_trips
        super().__call__(env,raw,beat)
        if old_arrival is None and self.first_arrival is not None:
            transition=raw["resource_state"]["transition"]
            if (transition.get("accepted") is not True
                    or transition.get("source_depth") != 1 or transition.get("target_depth") != 2
                    or transition.get("source_is_set") or transition.get("target_is_set")):
                raise RuntimeError("L2 exposure requires accepted canonical main L1-to-L2 transition")
        if self.completed_town_trips > old_trips and self.first_town_return_beat is None:
            transition=raw["resource_state"].get("transition",{})
            if (transition.get("accepted") is not True or transition.get("source_depth") != 0
                    or transition.get("target_depth") != 1 or transition.get("source_is_set")
                    or transition.get("target_is_set")):
                raise RuntimeError("town return lacks accepted native main 0-to-1 receipt")
            self.first_town_return_beat=int(beat)
            self.readiness_on_first_return=deepcopy(raw["resource_state"]["readiness"])
            # Intentionally DO NOT modify env.max_steps here.
        if self.service_object is not None and self.service_object.attempted:
            self.service_start_beat=int(self.service_object.start_steps)
        self.capture_economic(raw,beat,"observed_economic_change")

    def finish(self, raw, settled_steps):
        result=super().finish(raw,settled_steps)
        deadline=(self.arrival_budget if self.first_arrival is None
                  else self.first_arrival+self.followup)
        if self.last_beat > deadline:
            raise RuntimeError("native episode exceeded its exact arrival/followup deadline")
        if sum(self.by_depth.values()) != self.last_beat:
            raise RuntimeError("native per-scene exposure does not sum to actual microsteps")
        return {**result,"first_town_return_beat":self.first_town_return_beat,
                "readiness_on_first_return":self.readiness_on_first_return,
                "service_start_beat":self.service_start_beat,
                "complete_service_microsteps":(
                    self.first_town_return_beat-self.service_start_beat
                    if self.first_town_return_beat is not None and self.service_start_beat is not None else None),
                "policy_calls":self.policy_calls,
                "policy_calls_during_service":self.policy_calls_during_service,
                "policy_calls_while_service_queued":self.policy_calls_while_service_queued,
                "script_started_beat":self.script_started_beat,
                "pre_service_death":bool(result["died"] and self.service_start_beat is None),
                "death_during_service":bool(result["died"] and self.service_start_beat is not None
                                            and self.first_town_return_beat is None),
                "economic_snapshots":self.economic_snapshots,
                "economic_events":self.economic_events,
                "resource_commands":self.resource_commands}


def capture_resource_commands(inner, ledger):
    original=inner._execute_resource_command
    sequence=0
    def execute(command):
        nonlocal sequence
        sequence+=1
        before=int(inner._resource_actual_microsteps)
        if ledger.script_started_beat is None:
            ledger.script_started_beat=before
        before_position=[inner._raw.get("player_x"),inner._raw.get("player_y")]
        before_combat=compact_combat(inner._raw)
        before_snapshot=ledger.capture_economic(inner._raw,before,"before_command",marker=True)
        raw,beats,receipt=original(command)
        after=int(inner._resource_actual_microsteps)
        if after-before != int(beats):
            raise RuntimeError("resource command invented or omitted native microsteps")
        after_snapshot=ledger.capture_economic(raw,after,"after_command",marker=True)
        ledger.resource_commands.append({"command_id":sequence,"command":list(command),
            "before_beat":before,"after_beat":after,"micro_steps":int(beats),
            "before_position":before_position,
            "after_position":[raw.get("player_x"),raw.get("player_y")],
            "before_snapshot_id":before_snapshot,"after_snapshot_id":after_snapshot,
            "before_combat":before_combat,"after_combat":compact_combat(raw),
            "receipt":deepcopy(receipt)})
        return raw,beats,receipt
    inner._execute_resource_command=execute
    # The core command may be followed by the env's real settle beats. Keep
    # both clocks and expose the FULL step_resource interval as micro_steps.
    # All real DiabloGymEnv instances provide this method; the optional guard
    # allows the core-only deterministic ledger double used in unit tests.
    original_step=getattr(inner,"step_resource",None)
    if original_step is not None:
        def step_resource(command):
            before_records=len(ledger.resource_commands)
            result=original_step(command)
            if len(ledger.resource_commands)!=before_records+1:
                raise RuntimeError("resource step did not execute exactly one recorded command")
            event=ledger.resource_commands[-1]
            after=int(inner._resource_actual_microsteps)
            event["core_after_beat"]=event["after_beat"]
            event["core_micro_steps"]=event["micro_steps"]
            event["after_beat"]=after
            event["micro_steps"]=after-event["before_beat"]
            event["after_snapshot_id"]=ledger.capture_economic(inner._raw,after,"after_settle",marker=True)
            event["after_position"]=[inner._raw.get("player_x"),inner._raw.get("player_y")]
            event["after_combat"]=compact_combat(inner._raw)
            execution=result[-1].get("resource_action_audit") or {}
            if execution.get("micro_steps")!=event["micro_steps"]:
                raise RuntimeError("resource action audit disagrees with full command/settle native time")
            event["execution_audit"]=deepcopy(execution)
            return result
        inner.step_resource=step_resource


def record_policy_handoff(ledger, service, window_option):
    ledger.policy_calls+=1
    # action_masks may queue a service while building a final FARM gear-grace
    # observation. That is not yet script handoff. Preserve this legacy actor
    # decision; forbid policy calls only in a RESUPPLY window or during an
    # actually started, still-active script across its subsequent scene windows.
    if window_option==2 or (service.active and ledger.script_started_beat is not None):
        ledger.policy_calls_during_service+=1
        raise RuntimeError("policy called while resource script owns the actor")
    if service.active:
        ledger.policy_calls_while_service_queued+=1


def run_episode(cb, seed, model_name, service_policy):
    from diablogym.options_env import OptionsEnv,FARM,DIVE
    ledger=SustainLedger()
    options=None
    windows=[]
    last_info={}
    def worker(obs,mask):
        record_policy_handoff(ledger,options.resource_service,
                              (getattr(options,"_win",None) or {}).get("opt"))
        return cb(obs,mask)
    worker.diablogym_worker_observation_view=cb.diablogym_worker_observation_view
    worker.diablogym_worker_action12_mode=cb.diablogym_worker_action12_mode
    worker.on_beat=None
    try:
        options=OptionsEnv(max_steps=ARRIVAL_BUDGET,workers={FARM:worker,DIVE:worker},
            drink_sovereignty=True,worker_observation_view="dual-v4-asymmetric-v3",
            manager_observation_view="legacy-v3",dive_live_sovereignty=True,
            reward_economy="v4",explore_global_hunt=True,farm_scene_cap=3600,
            reset_layer_clock_on_window=True,resource_protocol="l2-town-v1",
            resource_purchase_mode="full",resource_service_policy=service_policy)
        options.reset(seed=seed)
        cb.episode_reseed(seed)
        if (options.max_steps != 6000 or options.farm_scene_cap != 3600
                or options.resource_service.calibration is not None):
            raise RuntimeError("formal worker horizon/denominator must remain 6000/3600 without calibration")
        ledger.service_object=options.resource_service
        ledger.service_state_provider=options.resource_service.telemetry
        options.env.resource_tick_callback=ledger
        ledger.capture_economic(options.env._raw,0,"reset",marker=True)
        capture_resource_commands(options.env,ledger)
        done=truncated=False
        empty_windows=0
        while not (done or truncated):
            before=int(options.env._steps)
            masks=options.action_masks()
            choice=options.resource_option_choice(masks)
            if choice is None or not masks[int(choice)]:
                raise RuntimeError("resource manager selected a missing or masked option")
            unused,reward,done,truncated,last_info=options.step(int(choice))
            after=int(options.env._steps)
            extra=last_info.get("option_extra") or {}
            windows.append({"option":int(choice),"before":before,"after":after,
                "reason":extra.get("reason"),"resource":deepcopy(extra.get("resource"))})
            empty_windows=empty_windows+1 if after==before else 0
            if empty_windows>8:
                raise RuntimeError("more than eight zero-native-tick option windows")
            if options.max_steps != 6000:
                raise RuntimeError("worker observation horizon changed during fixed-exposure evaluation")
        raw=options.env._raw
        result=ledger.finish(raw,options.env._steps)
        service=deepcopy(options.resource_service.telemetry())
        failure=(None if result["arrived_in_budget"] else
                 "death_before_arrival" if result["died"] else
                 (last_info.get("resource_protocol") or {}).get("terminal_reason")
                 or service.get("reason") or "arrival_budget_exhausted")
        row={"model":model_name,"seed":int(seed),"service_policy":service_policy,"mode":"full",
             **result,"service":service,"windows":windows,"resource_failure":failure,
             "final_resource":deepcopy(last_info.get("resource_protocol")),
             "victory":bool(raw.get("victory")),"char_level":int(raw.get("char_level",0)),
             "armor_class":int(raw.get("armor_class",0)),"gold":int(raw.get("gold",0)),
             "belt_heals":int(raw["resource_state"]["readiness"]["belt_heals"]),
             "service_microstep_cap":options.resource_service.service_microstep_cap,
             "arrival_budget":ARRIVAL_BUDGET,"followup_beats":FOLLOWUP_BEATS,
             "worker_observation_horizon":6000,"worker_farm_scene_denominator":3600,
             "resource_calibration":None}
        if row["followup_censored"]:
            raise audit.PilotEpisodeFailure("alive episode ended before complete followup exposure",row)
        return row
    except Exception as exc:
        if isinstance(exc,audit.PilotEpisodeFailure):
            raise
        inner=options.env if options is not None else None
        raw=(inner._raw or {}) if inner is not None else {}
        raise audit.PilotEpisodeFailure(str(exc),{
            "model":model_name,"seed":seed,"service_policy":service_policy,
            "error_type":type(exc).__name__,"error":str(exc),
            "native_microsteps":ledger.last_beat,
            "python_microsteps":getattr(inner,"_steps",None),
            "resource_state":deepcopy(raw.get("resource_state")),
            "service":deepcopy(options.resource_service.telemetry()) if options is not None else None,
            "economic_snapshots":ledger.economic_snapshots,"economic_events":ledger.economic_events,
            "resource_commands":ledger.resource_commands,"last_windows":windows[-5:]}) from exc
    finally:
        cb.on_beat=None
        if options is not None:options.close()


def build_jobs(phase, *, model=None, seeds=None, service_policy=None):
    if phase=="check":
        if model is None and seeds is None and service_policy is None:
            return list(KNOWN_CHECK_JOBS)
        selected_models=(model,) if model else MODELS
        selected_seeds=tuple(seeds) if seeds else tuple(dict.fromkeys(job[1] for job in KNOWN_CHECK_JOBS))
        return [(name,seed,service_policy or "sustain-v2") for name in selected_models for seed in selected_seeds]
    if any(value is not None for value in (model,seeds,service_policy)):
        raise ValueError("model/seed/service-policy overrides are check-only")
    selected=SEEDS[:16] if phase=="smoke" else SEEDS
    return [(name,seed,policy) for name in MODELS for seed in selected for policy in POLICIES]


def summarize(rows):
    result={}
    for model in MODELS:
        per_policy={}
        for policy in POLICIES:
            group=[r for r in rows if r["model"]==model and r["service_policy"]==policy]
            if not group:continue
            per_policy[policy]={r["seed"]:r for r in group}
            result[f"{model}:{policy}"]={"n":len(group),
                "arrived":sum(r["arrived_in_budget"] for r in group),
                "joint_successes":sum(r["joint_success"] for r in group),
                "died":sum(r["died"] for r in group),
                "followup_censored":sum(r["followup_censored"] for r in group),
                "completed_town_trips":sum(r["completed_town_trips"] for r in group),
                "arrival_beats":[r["first_l2_beat"] for r in group if r["arrived_in_budget"]],
                "failure_reasons":dict(Counter(r["resource_failure"] for r in group if r["resource_failure"]))}
        if len(per_policy)==2:
            common=sorted(set(per_policy[POLICIES[0]]) & set(per_policy[POLICIES[1]]))
            saved=sum(per_policy[POLICIES[1]][seed]["joint_success"] and not per_policy[POLICIES[0]][seed]["joint_success"] for seed in common)
            lost=sum(per_policy[POLICIES[0]][seed]["joint_success"] and not per_policy[POLICIES[1]][seed]["joint_success"] for seed in common)
            result[f"{model}:paired"]={"n":len(common),"saved":saved,"lost":lost,"net":saved-lost}
    return result


def smoke_coverage(rows,repeats):
    expected={(model,seed,policy) for model in MODELS for seed in SEEDS[:16] for policy in POLICIES}
    actual={(r["model"],r["seed"],r["service_policy"]) for r in rows}
    expected_repeats={key for key in expected if key[2]=="sustain-v2"}
    actual_repeats={(r["model"],r["seed"],r["service_policy"]) for r in repeats}
    returns={model:sum(r["completed_town_trips"]>0 for r in rows
                        if r["model"]==model and r["service_policy"]=="sustain-v2") for model in MODELS}
    return {"pass":actual==expected and len(rows)==len(expected)
                   and actual_repeats==expected_repeats and len(repeats)==len(expected_repeats)
                   and all(returns.values()),
            "rows":len(rows),"expected_rows":len(expected),"repeats":len(repeats),
            "expected_repeats":len(expected_repeats),"sustain_return_episodes":returns}


def load_engineering_receipt(path, identity):
    receipt=json.loads(path.read_text())
    if (receipt.get("version")!=VERSION or receipt.get("phase")!="smoke"
            or receipt.get("status")!="PASS" or receipt.get("identity")!=identity
            or not receipt.get("smoke_coverage",{}).get("pass")):
        raise ValueError("pilot requires a same-version, same-runtime/model PASS sustain smoke")
    rows_path=path.parent/"rows.jsonl"
    repeat_path=path.parent/"repeat_rows.jsonl"
    rows=[json.loads(line) for line in rows_path.read_text().splitlines() if line]
    repeats=[json.loads(line) for line in repeat_path.read_text().splitlines() if line]
    if (audit.canonical_hash(rows)!=receipt.get("rows_sha256")
            or audit.canonical_hash(repeats)!=receipt.get("repeated_rows_sha256")
            or not smoke_coverage(rows,repeats)["pass"]):
        raise ValueError("smoke row/repeat files differ from their engineering receipt")
    originals={(r["model"],r["seed"],r["service_policy"]):r for r in rows}
    if any(comparable_row(r)!=comparable_row(originals[(r["model"],r["seed"],r["service_policy"])]) for r in repeats):
        raise ValueError("smoke repeats are not independently equal to original rows")
    return receipt


_PROCESS_POLICIES=None


def initialize_policy_process(worker, certified):
    global _PROCESS_POLICIES
    _PROCESS_POLICIES={"r16":audit.load_policy(Path(worker)),"certified":audit.load_policy(Path(certified))}


def execute_job(payload):
    archive=Path(payload["archive"])
    archive.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    model,seed,policy=payload["job"]
    try:
        row=run_episode(_PROCESS_POLICIES[model],seed,model,policy)
        (archive/"row.json").write_text(json.dumps(row,sort_keys=True))
        return {"ok":True,"row":row}
    except Exception as exc:
        evidence=getattr(exc,"evidence",{"job":payload["job"],"error":str(exc)})
        (archive/"failure.json").write_text(json.dumps(evidence,indent=2))
        return {"ok":False,"error":f"{type(exc).__name__}: {exc}","evidence":evidence,
                "job_archive":str(archive)}
    finally:
        (archive/"execution.json").write_text(json.dumps({"pid":os.getpid(),
            "wall_time_seconds":time.monotonic()-started,"job":payload["job"]},indent=2))


def ordered_bounded_results(executor,function,payloads,max_in_flight):
    """Stop new submissions on any observed failure; archive in-flight jobs."""
    if type(max_in_flight) is not int or max_in_flight not in (1,2,3):
        raise ValueError("max_in_flight must be 1, 2, or 3")
    pending={}
    submitted=emitted=0
    def checked(future):
        result=future.result()
        if result.get("ok") is not True:
            raise audit.PilotEpisodeFailure(result.get("error","process job failed"),
                {"job_archive":result.get("job_archive"),"failure":result.get("evidence",{})})
        return result["row"]
    try:
        while emitted<len(payloads):
            for future in pending.values():
                if future.done():checked(future)
            while submitted<len(payloads) and len(pending)<max_in_flight:
                pending[submitted]=executor.submit(function,payloads[submitted])
                submitted+=1
                for future in pending.values():
                    if future.done():checked(future)
            target=pending[emitted]
            if not target.done():
                concurrent.futures.wait([f for f in pending.values() if not f.done()],
                                        return_when=concurrent.futures.FIRST_COMPLETED)
                continue
            row=checked(target)
            del pending[emitted]
            emitted+=1
            yield row
    finally:
        for future in pending.values():future.cancel()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",required=True,choices=("check","smoke","pilot"))
    parser.add_argument("--workers",type=int,choices=(1,2,3),default=1)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--worker",type=Path,default=audit.R16)
    parser.add_argument("--certified-worker",type=Path,default=audit.CERTIFIED)
    parser.add_argument("--engineering-receipt",type=Path)
    parser.add_argument("--model",choices=MODELS)
    parser.add_argument("--seed",type=int,action="append")
    parser.add_argument("--service-policy",choices=POLICIES)
    args=parser.parse_args(argv)
    try:
        jobs=build_jobs(args.phase,model=args.model,seeds=args.seed,service_policy=args.service_policy)
    except ValueError as exc:parser.error(str(exc))
    if args.output_dir.exists():parser.error("output-dir must be new; preserve every prior result")
    if args.phase=="pilot" and args.engineering_receipt is None:
        parser.error("pilot requires --engineering-receipt")
    before=identities(args.worker,args.certified_worker)
    if args.phase=="pilot":load_engineering_receipt(args.engineering_receipt,before)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    from eval_contract import resource_service_recipe
    config={"version":VERSION,"phase":args.phase,"identity":before,"training_steps":0,
        "development_evaluation":True,"certification":False,"resource_calibration":None,
        "arrival_budget":ARRIVAL_BUDGET,"followup_beats":FOLLOWUP_BEATS,
        "absolute_episode_limit":ARRIVAL_BUDGET+FOLLOWUP_BEATS,
        "worker_observation_horizon":6000,"worker_farm_scene_denominator":3600,
        "farm_trigger_microsteps":3600,"observation_view":"dual-v4-asymmetric-v3",
        "decoding":"sampled-reseed-per-episode","torch_threads":1,
        "workers":args.workers,"process_start_method":"spawn","max_in_flight":args.workers,
        "jobs":[list(job) for job in jobs],"policies":list(POLICIES),
        "recipes":{policy:resource_service_recipe("l2-town-v1","full",policy) for policy in POLICIES},
        "repeat_execution":"all 32 sustain smoke rows; serial independent main-process policies",
        "engineering_receipt_sha256":audit.sha256(args.engineering_receipt) if args.engineering_receipt else None}
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2))
    rows=[];repeats=[];status="PASS";error=None
    try:
        payloads=[{"job":job,"archive":str(args.output_dir/"jobs"/f"{i:04d}-{job[0]}-{job[1]}-{job[2]}")}
                  for i,job in enumerate(jobs)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),initializer=initialize_policy_process,
                initargs=(str(args.worker),str(args.certified_worker))) as executor, \
             (args.output_dir/"rows.jsonl").open("x") as stream:
            for row in ordered_bounded_results(executor,execute_job,payloads,args.workers):
                rows.append(row);stream.write(json.dumps(row,sort_keys=True)+"\n");stream.flush()
                print(f"{args.phase} {row['model']} {row['service_policy']} seed={row['seed']} "
                      f"return={row['completed_town_trips']} L2={row['first_l2_beat']} "
                      f"joint={row['joint_success']} beats={row['micro_steps']}",flush=True)
        if args.phase=="smoke":
            initialize_policy_process(str(args.worker),str(args.certified_worker))
            with (args.output_dir/"repeat_rows.jsonl").open("x") as stream:
                for i,original in enumerate(r for r in rows if r["service_policy"]=="sustain-v2"):
                    job=(original["model"],original["seed"],original["service_policy"])
                    result=execute_job({"job":job,"archive":str(args.output_dir/"repeats"/f"{i:04d}-{job[0]}-{job[1]}")})
                    if not result["ok"]:
                        raise audit.PilotEpisodeFailure(result["error"],result["evidence"])
                    repeated=result["row"];repeats.append(repeated)
                    stream.write(json.dumps(repeated,sort_keys=True)+"\n");stream.flush()
                    if comparable_row(repeated)!=comparable_row(original):
                        raise audit.PilotEpisodeFailure(f"strict repeat mismatch: {job}",{
                            "job":job,"original_sha256":audit.canonical_hash(comparable_row(original)),
                            "repeat_sha256":audit.canonical_hash(comparable_row(repeated)),
                            "original":original,"repeat":repeated})
            if not smoke_coverage(rows,repeats)["pass"]:
                status="INCOMPLETE_COVERAGE";error="smoke did not exercise a real sustain town return for both workers"
    except Exception as exc:
        status="FAIL";error=f"{type(exc).__name__}: {exc}"
        (args.output_dir/"failure.json").write_text(json.dumps(
            getattr(exc,"evidence",{"error":error}),indent=2))
    try:
        if before!=identities(args.worker,args.certified_worker):
            status="FAIL";error="source/native/content/weights changed during evaluation"
    except Exception as exc:
        status="FAIL";error=f"final identity validation failed: {exc}"
    report={**config,"status":status,"error":error,"summary":summarize(rows),
        "rows_sha256":audit.canonical_hash(rows),"rows_count":len(rows),
        "repeated_rows_sha256":audit.canonical_hash(repeats),"repeat_count":len(repeats),
        "smoke_coverage":smoke_coverage(rows,repeats) if args.phase=="smoke" else None,
        "limitations":["These are development seeds and results, not held-out certification.",
            "No L2 arrival is joint failure; an alive incomplete followup is censored and fails the engineering run.",
            "Legacy-v1 is paired under this current native runtime; it is not a claim of identity with an archived binary.",
            "The 1500/600 service caps and different shopping recipes are bundled policy interventions.",
            "After a process failure, new jobs stop; at most workers-1 already-running bounded jobs finish and keep independent archives.",
            "Economic snapshots reuse IDs only when economic contents match; commands retain full settled microsteps plus separate core clocks and exact combat/position facts.",
            "The default late 3600 trigger need not recreate historical 1800-trigger blocker geometry on the same seeds."]}
    (args.output_dir/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"status":status,"summary":report["summary"],"error":error}),flush=True)
    return 0 if status=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
