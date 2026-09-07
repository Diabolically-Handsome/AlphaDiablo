"""Pure clock, delivery-logging, receipt and scheduler tests; no game runs."""
from concurrent.futures import Future
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"train"))
import probe_sustain_protocol as probe


def state(depth=1, *, source=None, sequence=0, hp=50, dead=False):
    transition={"sequence":sequence,"accepted":source is not None,
                "source_depth":source,"target_depth":depth,"source_is_set":False,
                "target_is_set":False,"pretransition_ready":True}
    return {"player_x":10,"player_y":10,"future_x":10,"future_y":10,
            "hp":hp,"max_hp":70,"dead":dead,"dungeon_level":depth,"is_set_level":False,
            "gold":100,"belt_heals":4,"belt_heal_kinds":[1]*4+[0]*4,"char_level":3,
            "armor_class":10,"monster_kill_total":0,"equipped_items":[],"monsters":[],
            "resource_state":{"enabled":True,"readiness":{"ready":True,"belt_heals":4,"failures":[]},
                              "transition":transition,"town":{},"inventory_items":[],"inventory_equipment":[]}}


def fake_smoke_rows():
    return [{"model":model,"seed":seed,"service_policy":policy,"completed_town_trips":1}
            for model in probe.MODELS for seed in probe.SEEDS[:16] for policy in probe.POLICIES]


class LedgerTests(unittest.TestCase):
    def test_real_town_return_does_not_stop_or_extend_episode(self):
        ledger=probe.SustainLedger(arrival_budget=10,followup=3)
        env=SimpleNamespace(max_steps=10)
        ledger(env,state(0,source=1,sequence=1),1)
        ledger(env,state(1,source=0,sequence=2),2)
        self.assertEqual(env.max_steps,10)
        self.assertEqual(ledger.first_town_return_beat,2)
        ledger(env,state(1),3)
        self.assertEqual(ledger.last_beat,3)

    def test_first_canonical_l2_sets_full_followup_not_remaining_prearrival_budget(self):
        ledger=probe.SustainLedger(arrival_budget=10,followup=3);env=SimpleNamespace(max_steps=10)
        ledger(env,state(),1)
        ledger(env,state(2,source=1,sequence=1),2)
        self.assertEqual(env.max_steps,5)
        for beat in (3,4,5):ledger(env,state(2),beat)
        result=ledger.finish(state(2),5)
        self.assertTrue(result["joint_success"])
        self.assertEqual(result["l2_microsteps"],3)
        self.assertEqual(sum(result["beats_by_depth"].values()),5)

    def test_forged_accepted_or_readiness_receipts_cannot_start_success_clock(self):
        for field,value in (("accepted",False),("source_depth",0),("pretransition_ready",False)):
            with self.subTest(field=field):
                ledger=probe.SustainLedger(arrival_budget=10,followup=3)
                raw=state(2,source=1,sequence=1);raw["resource_state"]["transition"][field]=value
                with self.assertRaises(RuntimeError):ledger(SimpleNamespace(max_steps=10),raw,1)

    def test_alive_short_followup_is_censored_never_joint_success(self):
        ledger=probe.SustainLedger(arrival_budget=10,followup=3);env=SimpleNamespace(max_steps=10)
        ledger(env,state(2,source=1,sequence=1),1)
        ledger(env,state(2),2)
        result=ledger.finish(state(2),2)
        self.assertTrue(result["followup_censored"])
        self.assertIsNone(result["alive_at_followup"])
        self.assertFalse(result["joint_success"])

    def test_death_within_followup_is_observed_failure_not_censor(self):
        ledger=probe.SustainLedger(arrival_budget=10,followup=3);env=SimpleNamespace(max_steps=10)
        ledger(env,state(2,source=1,sequence=1),1)
        ledger(env,state(2,dead=True,hp=0),2)
        result=ledger.finish(state(2,dead=True,hp=0),2)
        self.assertFalse(result["followup_censored"])
        self.assertFalse(result["joint_success"])
        self.assertEqual(result["death_beat"],2)

    def test_no_arrival_is_failure_and_cannot_run_extra_followup_ticks(self):
        ledger=probe.SustainLedger(arrival_budget=2,followup=3);env=SimpleNamespace(max_steps=2)
        for beat in (1,2):ledger(env,state(),beat)
        self.assertFalse(ledger.finish(state(),2)["joint_success"])
        ledger(env,state(),3)
        with self.assertRaisesRegex(RuntimeError,"exact arrival/followup deadline"):
            ledger.finish(state(),3)

    def test_postarrival_overshoot_is_engineering_error(self):
        ledger=probe.SustainLedger(arrival_budget=10,followup=1);env=SimpleNamespace(max_steps=10)
        ledger(env,state(2,source=1,sequence=1),1)
        for beat in (2,3):ledger(env,state(2),beat)
        with self.assertRaisesRegex(RuntimeError,"exact arrival/followup deadline"):
            ledger.finish(state(2),3)

    def test_native_tick_gap_and_python_clock_mismatch_rejected(self):
        ledger=probe.SustainLedger();env=SimpleNamespace(max_steps=6000)
        with self.assertRaises(RuntimeError):ledger(env,state(),2)
        ledger(env,state(),1)
        with self.assertRaises(RuntimeError):ledger.finish(state(),2)

    def test_inventory_equipment_change_gets_new_snapshot_and_does_not_alias(self):
        ledger=probe.SustainLedger();raw=state()
        a=ledger.capture_economic(raw,0,"before")
        raw["resource_state"]["inventory_equipment"]=[{"index":0,"base_id":48,"seed_lo":22}]
        b=ledger.capture_economic(raw,1,"after")
        self.assertNotEqual(a,b)
        raw["resource_state"]["inventory_equipment"][0]["seed_lo"]=99
        self.assertEqual(ledger.economic_snapshots[b]["resource_state"]["inventory_equipment"][0]["seed_lo"],22)

    def test_all_new_commands_retain_before_after_physical_and_combat_facts(self):
        for command in (("unequip",0,1,2,3,4),("attack_monster",7,12),("drink",)):
            with self.subTest(command=command):
                ledger=probe.SustainLedger();inner=SimpleNamespace(_raw=state(),_resource_actual_microsteps=10)
                def execute(command):
                    raw=deepcopy(inner._raw);raw["hp"]=55;raw["belt_heal_kinds"][0]=0
                    raw["resource_state"]["inventory_equipment"]=[{"base_id":48,"seed_lo":2}]
                    inner._resource_actual_microsteps+=1
                    return raw,1,{"accepted":True,"price":0}
                inner._execute_resource_command=execute
                probe.capture_resource_commands(inner,ledger)
                inner._execute_resource_command(command)
                event=ledger.resource_commands[0]
                self.assertEqual(event["command"],list(command))
                self.assertEqual((event["before_beat"],event["after_beat"],event["micro_steps"]),(10,11,1))
                before=ledger.economic_snapshots[event["before_snapshot_id"]]
                after=ledger.economic_snapshots[event["after_snapshot_id"]]
                self.assertEqual((before["hp"],after["hp"]),(50,55))
                self.assertIn("before_combat",event);self.assertIn("after_combat",event)
                self.assertEqual(after["resource_state"]["inventory_equipment"][0]["base_id"],48)

    def test_outer_command_receipt_includes_real_settle_cost_and_final_physical_state(self):
        ledger=probe.SustainLedger();inner=SimpleNamespace(_raw=state(),_resource_actual_microsteps=10)
        def execute(command):
            inner._resource_actual_microsteps+=1
            raw=deepcopy(inner._raw);raw["hp"]=55
            return raw,1,{"accepted":True,"price":0}
        inner._execute_resource_command=execute
        def outer(command):
            inner._raw,unused,receipt=inner._execute_resource_command(command)
            inner._resource_actual_microsteps+=1
            inner._raw["hp"]=52
            return None,0,False,False,{"resource_action_audit":{"micro_steps":2,**receipt}}
        inner.step_resource=outer
        probe.capture_resource_commands(inner,ledger)
        inner.step_resource(("attack_monster",7,12))
        event=ledger.resource_commands[0]
        self.assertEqual((event["core_micro_steps"],event["micro_steps"]),(1,2))
        self.assertEqual((event["core_after_beat"],event["after_beat"]),(11,12))
        self.assertEqual(ledger.economic_snapshots[event["after_snapshot_id"]]["hp"],52)

    def test_command_cannot_invent_microsteps(self):
        inner=SimpleNamespace(_raw=state(),_resource_actual_microsteps=10,
                              _execute_resource_command=lambda cmd:(state(),1,{"accepted":True}))
        probe.capture_resource_commands(inner,probe.SustainLedger())
        with self.assertRaises(RuntimeError):inner._execute_resource_command(("wait",))


class DriverContractsTests(unittest.TestCase):
    def test_service_queue_is_not_confused_with_actual_script_handoff(self):
        ledger=probe.SustainLedger();service=SimpleNamespace(active=True)
        probe.record_policy_handoff(ledger,service,0)
        self.assertEqual(ledger.policy_calls_while_service_queued,1)
        self.assertEqual(ledger.policy_calls_during_service,0)
        ledger.script_started_beat=100
        with self.assertRaises(RuntimeError):probe.record_policy_handoff(ledger,service,0)
        service.active=False
        probe.record_policy_handoff(ledger,service,0)
        with self.assertRaises(RuntimeError):probe.record_policy_handoff(ledger,service,2)
        self.assertEqual(ledger.policy_calls_during_service,2)

    def test_frozen_job_counts_and_check_overrides(self):
        self.assertEqual(probe.build_jobs("check"),list(probe.KNOWN_CHECK_JOBS))
        self.assertEqual(len(probe.build_jobs("smoke")),64)
        self.assertEqual(len(probe.build_jobs("pilot")),192)
        self.assertEqual(probe.build_jobs("check",model="certified",seeds=[2114005]),
                         [("certified",2114005,"sustain-v2")])
        with self.assertRaises(ValueError):probe.build_jobs("pilot",model="r16")

    def test_repeat_comparison_removes_walltime_but_preserves_real_state(self):
        a={"seed":1,"wall_time_seconds":1,"service":{"steps":100},"hp":10,"pid":99}
        b={**a,"wall_time_seconds":2,"pid":88}
        self.assertEqual(probe.comparable_row(a),probe.comparable_row(b))
        b["hp"]=9
        self.assertNotEqual(probe.comparable_row(a),probe.comparable_row(b))

    def test_pass_smoke_requires_exact_paired_seed_sets_repeats_and_actual_returns(self):
        rows=fake_smoke_rows();repeats=[deepcopy(r) for r in rows if r["service_policy"]=="sustain-v2"]
        self.assertTrue(probe.smoke_coverage(rows,repeats)["pass"])
        self.assertFalse(probe.smoke_coverage(rows,repeats[:-1])["pass"])
        for row in rows:
            if row["model"]=="certified":row["completed_town_trips"]=0
        self.assertFalse(probe.smoke_coverage(rows,repeats)["pass"])

    def test_engineering_receipt_checks_native_identity_and_row_content(self):
        rows=fake_smoke_rows();repeats=[deepcopy(r) for r in rows if r["service_policy"]=="sustain-v2"]
        identity={"native":"new-candidate-hash"}
        receipt={"version":probe.VERSION,"phase":"smoke","status":"PASS","identity":identity,
                 "smoke_coverage":probe.smoke_coverage(rows,repeats),
                 "rows_sha256":probe.audit.canonical_hash(rows),
                 "repeated_rows_sha256":probe.audit.canonical_hash(repeats)}
        with TemporaryDirectory() as folder:
            root=Path(folder);summary=root/"summary.json"
            summary.write_text(json.dumps(receipt))
            (root/"rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (root/"repeat_rows.jsonl").write_text("\n".join(json.dumps(row) for row in repeats))
            self.assertEqual(probe.load_engineering_receipt(summary,identity)["status"],"PASS")
            with self.assertRaises(ValueError):probe.load_engineering_receipt(summary,{"native":"old"})
            rows[0]["completed_town_trips"]=2
            (root/"rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):probe.load_engineering_receipt(summary,identity)

    def test_run_episode_keeps_options_horizon_and_does_not_stop_at_town(self):
        # Entire Options/engine boundary is a deterministic double: no game load.
        captured={};created=[]
        class FakeOptions:
            def __init__(self,**kwargs):
                captured.update(kwargs);created.append(self)
                self.max_steps=6000;self.farm_scene_cap=3600;self.closed=False
                self.env=SimpleNamespace(_raw=state(),_steps=0,_resource_actual_microsteps=0,max_steps=6000,
                    _execute_resource_command=lambda cmd:(state(),0,{}))
                self.resource_service=SimpleNamespace(calibration=None,active=False,attempted=True,
                    start_steps=0,service_microstep_cap=1500,
                    telemetry=lambda:{"phase":"done","reason":"complete","steps":2})
            def reset(self,seed):self.seed=seed
            def action_masks(self):return [True,True,True]
            def resource_option_choice(self,masks):return 0
            def step(self,choice):
                beat=self.env._steps+1
                raw=(state(0,source=1,sequence=1) if beat==1 else
                     state(1,source=0,sequence=2) if beat==2 else
                     state(2,source=1,sequence=3) if beat==3 else state(2))
                self.env._steps=self.env._resource_actual_microsteps=beat;self.env._raw=raw
                self.env.resource_tick_callback(self.env,raw,beat)
                return None,0,False,beat>=self.env.max_steps,{"option_extra":{},"resource_protocol":{}}
            def close(self):self.closed=True
        cb=SimpleNamespace(episode_reseed=lambda seed:None,on_beat=None,
            diablogym_worker_observation_view="dual-v4-asymmetric-v3",
            diablogym_worker_action12_mode="environment-mask")
        original=probe.SustainLedger
        with patch("diablogym.options_env.OptionsEnv",FakeOptions), \
             patch.object(probe,"SustainLedger",side_effect=lambda:original(arrival_budget=10,followup=3)):
            row=probe.run_episode(cb,2114004,"r16","sustain-v2")
        self.assertNotIn("resource_calibration",captured)
        self.assertEqual(captured["max_steps"],6000)
        self.assertEqual(captured["resource_service_policy"],"sustain-v2")
        self.assertEqual((row["first_town_return_beat"],row["first_l2_beat"],row["micro_steps"]),(2,3,6))
        self.assertTrue(row["joint_success"]);self.assertTrue(created[0].closed)
        self.assertEqual(created[0].max_steps,6000)

    def test_episode_job_archives_are_exclusive_and_walltime_is_separate(self):
        with TemporaryDirectory() as folder:
            archive=Path(folder)/"job"
            with patch.object(probe,"_PROCESS_POLICIES",{"r16":object()}), \
                 patch.object(probe,"run_episode",return_value={"seed":1,"hp":50}):
                payload={"job":("r16",1,"sustain-v2"),"archive":str(archive)}
                result=probe.execute_job(payload)
                self.assertTrue(result["ok"])
                self.assertNotIn("wall_time_seconds",result["row"])
                self.assertTrue((archive/"execution.json").is_file())
                with self.assertRaises(FileExistsError):probe.execute_job(payload)
            failed=Path(folder)/"failed"
            with patch.object(probe,"_PROCESS_POLICIES",{"r16":object()}), \
                 patch.object(probe,"run_episode",side_effect=RuntimeError("broken")):
                result=probe.execute_job({"job":("r16",1,"sustain-v2"),"archive":str(failed)})
            self.assertFalse(result["ok"])
            self.assertTrue((failed/"failure.json").is_file())
            self.assertTrue((failed/"execution.json").is_file())


class SchedulerTests(unittest.TestCase):
    def test_later_completed_failure_stops_submissions_while_first_job_is_slow(self):
        first=Future();calls=[]
        class Executor:
            def submit(self,function,payload):
                calls.append(payload)
                if len(calls)==1:return first
                failed=Future();failed.set_result({"ok":False,"error":"blocked failure","evidence":{}})
                return failed
        with self.assertRaises(probe.audit.PilotEpisodeFailure):
            list(probe.ordered_bounded_results(Executor(),None,list(range(10)),2))
        self.assertEqual(calls,[0,1]);self.assertTrue(first.cancelled())

    def test_completed_results_emit_in_job_order_and_keep_bound(self):
        calls=[];first=Future()
        class Executor:
            def submit(self,function,payload):
                calls.append(payload)
                if payload==0:return first
                result=Future();result.set_result({"ok":True,"row":payload})
                if payload==1:first.set_result({"ok":True,"row":0})
                return result
        result=list(probe.ordered_bounded_results(Executor(),None,[0,1,2,3],2))
        self.assertEqual(result,[0,1,2,3]);self.assertEqual(calls,[0,1,2,3])

    def test_invalid_concurrency_never_submits(self):
        for limit in (0,4,True):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                list(probe.ordered_bounded_results(None,None,[1],limit))


if __name__=="__main__":unittest.main()
