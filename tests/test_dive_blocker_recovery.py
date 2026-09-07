"""Opt-in a11 blocker combat: actual ticks, observed targets, legacy identity."""
from copy import deepcopy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from diablogym import bridge
from diablogym.env import DiabloGymEnv, _scene_identity


def monster(**changes):
    value = dict(id=7, type=1, x=11, y=10, future_x=11, future_y=10,
                 hp=5, hp_fixed_hi=0, hp_fixed_lo=320, visible=True,
                 rnd_item_seed_hi=2, rnd_item_seed_lo=3)
    value.update(changes)
    return value


def raw(*, mode=None, monsters=None, **changes):
    value = dict(player_x=10, player_y=10, future_x=10, future_y=10,
                 dungeon_level=1, is_set_level=False, dead=False,
                 hp=70, max_hp=70, belt_heals=4,
                 player_mode=bridge.PM_STAND if mode is None else mode,
                 dest_action=bridge.ACTION_NONE, walkpath0=bridge.WALK_NONE,
                 triggers=[dict(x=13, y=10, msg=bridge.WM_DIABNEXTLVL)],
                 monsters=[monster()] if monsters is None else monsters,
                 resource_state={"readiness": {"ready": True}})
    value.update(changes)
    return value


class Rig:
    def __init__(self, before, frames, *, flag="adjacent-v1", now=100, maximum=6000,
                 path=None, callback=None):
        self.env = DiabloGymEnv.__new__(DiabloGymEnv)
        self.env.dive_blocker_recovery = flag
        self.env.resource_protocol = "l2-town-v1"
        self.env._raw = deepcopy(before)
        self.env._resource_actual_microsteps = now
        self.env._resource_pending_command = None
        self.env._resource_calibration = None
        self.env.max_steps = maximum
        self.env._record_visit = lambda _pos: None
        self.env._plan_descend_path = lambda *_args, **_kw: list(
            path if path is not None else [(11, 10, False), (12, 10, False)])
        self.env._descend_fallback_promotion = False
        self.last = deepcopy(before)
        self.frames = iter(frames)
        self.calls = []
        self.audit = {"attempts": 0, "accepts": 0}
        def step():
            self.env._resource_actual_microsteps += 1
            self.last = deepcopy(next(self.frames))
            self.calls.append(("tick", self.env._resource_actual_microsteps))
            if callback:
                callback(self.env, self.last)
            return self.last
        self.env._step_native = step

    def run(self, budget=12, attack_result=1):
        def attack(*args):
            self.calls.append(("attack", args))
            return attack_result
        def wait():
            self.calls.append(("wait",))
            if self.last["player_mode"] == bridge.PM_ATTACK:
                self.last["player_mode"] = bridge.PM_STAND
            self.last["dest_action"] = bridge.ACTION_NONE
            self.last["walkpath0"] = bridge.WALK_NONE
            return 1
        def walk(*args):
            self.calls.append(("walk", args))
            return 1
        with ExitStack() as stack:
            stack.enter_context(patch.object(bridge, "act_controller_attack_monster", side_effect=attack))
            stack.enter_context(patch.object(bridge, "act_explore_walk", side_effect=walk))
            stack.enter_context(patch.object(bridge, "act_wait", side_effect=wait))
            stack.enter_context(patch.object(bridge, "observe", side_effect=lambda: self.last))
            return self.env._macro_descend(max_beats=budget, execution_audit=self.audit)


class DiveBlockerRecoveryTests(unittest.TestCase):
    def test_constructor_rejects_unknown_and_protocol_off_before_engine(self):
        for flag, protocol in (("yes", "l2-town-v1"), ("adjacent-v1", "off"), (True, "l2-town-v1")):
            with self.subTest(flag=flag, protocol=protocol):
                with patch.object(bridge, "init") as init, self.assertRaises(ValueError):
                    DiabloGymEnv(dive_blocker_recovery=flag, resource_protocol=protocol)
                init.assert_not_called()

    def test_animation_reaches_actual_damage_without_first_tick_cancel(self):
        hurt = monster(hp=2, hp_fixed_lo=128)
        rig = Rig(raw(), [raw(mode=bridge.PM_ATTACK), raw(mode=bridge.PM_ATTACK), raw(monsters=[hurt])])
        observed, beats = rig.run()
        self.assertEqual(beats, 3)
        self.assertEqual(rig.calls[:4], [("attack", (7,10,10,1)), ("tick",101), ("tick",102), ("tick",103)])
        self.assertEqual(rig.audit, {"attempts": 1, "accepts": 1})
        self.assertTrue(rig.env._dive_blocker_audit["damage_observed"])
        self.assertFalse(rig.env._dive_blocker_audit["position_changed"])
        self.assertEqual(observed["monsters"][0]["hp"], 2)

    def test_miss_is_not_material_damage_or_movement(self):
        rig = Rig(raw(), [raw(mode=bridge.PM_ATTACK), raw()])
        rig.run()
        receipt = rig.env._dive_blocker_audit
        self.assertTrue(receipt["accepted"])
        self.assertFalse(receipt["damage_observed"])
        self.assertFalse(receipt["position_changed"])
        self.assertEqual(receipt["reason"], "swing_complete")

    def test_observed_kill_then_next_macro_walks_real_edge(self):
        dead = monster(hp=0, hp_fixed_lo=0)
        rig = Rig(raw(), [raw(monsters=[dead])])
        observed, beats = rig.run()
        self.assertEqual(beats, 1)
        self.assertFalse(rig.env._dive_blocker_audit["target_alive_observed"])
        self.assertTrue(rig.env._dive_blocker_audit["damage_observed"])
        walk = Rig(observed, [raw(monsters=[], player_x=11, future_x=11)], path=[(11,10,False)])
        after, unused = walk.run(budget=1)
        self.assertEqual(after["player_x"], 11)
        self.assertEqual([x[0] for x in walk.calls], ["walk", "tick", "wait"])

    def test_future_reservation_binds_only_native_adjacent_target(self):
        reserved = monster(x=12,y=10,future_x=11,future_y=10)
        rig = Rig(raw(monsters=[reserved]), [raw(monsters=[reserved])])
        rig.run(budget=1)
        self.assertEqual(rig.calls[0], ("attack", (7,10,10,1)))

    def test_nonblocking_unseen_dead_minion_and_out_of_range_never_attack(self):
        variants = [monster(visible=False), monster(hp=0,hp_fixed_lo=0),
                    monster(is_invalid=True), monster(is_player_minion=True), monster(type=109),
                    monster(x=10,y=11,future_x=10,future_y=11),
                    monster(future_x=13,future_y=10)]
        for target in variants:
            with self.subTest(target=target):
                rig = Rig(raw(monsters=[target]), [raw(monsters=[target])])
                rig.run(budget=1)
                self.assertFalse(any(c[0] == "attack" for c in rig.calls))
                self.assertEqual(rig.calls[0][0], "walk")

    def test_one_tick_tail_budget_after_previous_walk(self):
        before = raw(monsters=[])
        moved = raw(player_x=11, future_x=11, monsters=[monster(x=12,future_x=12)])
        rig = Rig(before, [moved, deepcopy(moved)])
        observed, beats = rig.run(budget=2)
        self.assertEqual(beats, 2)
        self.assertEqual(rig.env._resource_actual_microsteps, 102)
        self.assertEqual([c[0] for c in rig.calls], ["walk","tick","attack","tick","wait"])
        self.assertEqual(rig.calls[2], ("attack", (7,11,10,1)))
        self.assertEqual(rig.env._dive_blocker_audit["micro_steps"], 1)

    def test_rejected_attack_costs_one_real_wait(self):
        rig = Rig(raw(), [raw()])
        observed, beats = rig.run(attack_result=0)
        self.assertEqual(beats, 1)
        self.assertEqual(rig.audit, {"attempts":1,"accepts":0})
        self.assertEqual(rig.calls[:3], [("attack",(7,10,10,1)),("wait",),("tick",101)])
        self.assertFalse(rig.env._dive_blocker_audit["damage_observed"])

    def test_death_scene_health_loss_and_generation_reuse_stop_attack(self):
        cases = [(raw(dead=True), "terminal"), (raw(dungeon_level=2), "scene_changed"),
                 (raw(hp=54,resource_state={"readiness":{"ready":False}}), "readiness_lost"),
                 (raw(monsters=[monster(rnd_item_seed_lo=4)]), "target_changed")]
        for frame, reason in cases:
            with self.subTest(reason=reason):
                rig = Rig(raw(), [frame])
                observed, beats = rig.run()
                self.assertEqual(beats, 1)
                self.assertEqual(rig.env._dive_blocker_audit["reason"], reason)
                self.assertEqual(sum(c[0] == "attack" for c in rig.calls), 1)

    def test_disappearance_does_not_claim_kill(self):
        rig = Rig(raw(), [raw(monsters=[])])
        rig.run()
        receipt = rig.env._dive_blocker_audit
        self.assertIsNone(receipt["target_alive_observed"])
        self.assertFalse(receipt["damage_observed"])

    def test_live_callback_and_episode_deadline_stop_core(self):
        for maximum, callback in ((101,None),(6000,lambda e,r:setattr(e,"max_steps",e._resource_actual_microsteps))):
            with self.subTest(maximum=maximum):
                rig = Rig(raw(), [raw(mode=bridge.PM_ATTACK)], maximum=maximum, callback=callback)
                observed, beats = rig.run()
                self.assertEqual((beats,rig.env._resource_actual_microsteps),(1,101))

    def test_expired_global_deadline_performs_no_native_tick_or_attack(self):
        rig = Rig(raw(), [], maximum=100)
        observed, beats = rig.run()
        self.assertEqual(beats,0)
        self.assertFalse(any(c[0] in ("attack","tick") for c in rig.calls))

    def test_same_budget_miss_does_not_reset_or_extend_inside_macro(self):
        rig = Rig(raw(), [raw(mode=bridge.PM_ATTACK)]*12)
        observed, beats = rig.run(budget=12)
        self.assertEqual((beats,rig.env._resource_actual_microsteps),(12,112))
        self.assertEqual(sum(c[0] == "attack" for c in rig.calls),1)
        self.assertFalse(rig.env._dive_blocker_audit["damage_observed"])

    def test_new_flag_without_blocker_and_off_keep_legacy_action_trace(self):
        before=raw(monsters=[])
        frame=raw(monsters=[],player_x=11,future_x=11)
        outputs=[]
        for flag in ("off", "adjacent-v1", None):
            rig=Rig(before,[frame],flag=flag,path=[(11,10,False)])
            if flag is None:del rig.env.dive_blocker_recovery
            result=rig.run(budget=1)
            outputs.append((result,rig.calls,rig.audit,rig.env._resource_actual_microsteps))
        self.assertEqual(outputs[0],outputs[1])
        self.assertEqual(outputs[0],outputs[2])

    def test_off_with_blocker_keeps_legacy_walk_retries(self):
        traces=[]
        for flag in ("off",None):
            rig=Rig(raw(),[raw()]*6,flag=flag)
            if flag is None:del rig.env.dive_blocker_recovery
            rig.run()
            traces.append(rig.calls)
        self.assertEqual(traces[0],traces[1])
        self.assertEqual(sum(c[0] == "walk" for c in traces[0]),2)
        self.assertFalse(any(c[0] == "attack" for c in traces[0]))

    def test_normal_settle_is_charged_beyond_core_and_stops_at_live_horizon(self):
        rig=Rig(raw(), [raw(mode=bridge.PM_ATTACK)])
        observed, core=rig.run(budget=1)
        observed["player_mode"]=99  # native hit/block recovery, not a cancelable attack
        frames=iter([raw(mode=99),raw()])
        def step():
            rig.env._resource_actual_microsteps+=1
            return next(frames)
        rig.env._step_native=step
        with patch.object(bridge,"act_wait"):
            final,total=rig.env._settle_to_idle(observed,core,max_beats=20,start_scene=_scene_identity(observed))
        self.assertEqual((core,total,rig.env._resource_actual_microsteps),(1,3,103))
        self.assertTrue(rig.env._decision_idle(final))
        def deadline_step():
            rig.env._resource_actual_microsteps+=1
            rig.env.max_steps=rig.env._resource_actual_microsteps
            return raw(mode=99)
        rig.env._step_native=deadline_step
        with patch.object(bridge,"act_wait"):
            final,total=rig.env._settle_to_idle(raw(mode=99),0,max_beats=20,start_scene=_scene_identity(raw()))
        self.assertEqual(total,1)
        self.assertEqual(rig.env._resource_actual_microsteps,rig.env.max_steps)
        terminated,truncated,*unused=rig.env._episode_boundary(final,rig.env._resource_actual_microsteps,rig.env.max_steps)
        self.assertTrue(terminated)  # existing GLOBAL budget boundary only
        self.assertFalse(truncated)


    def test_effect_receipt_requires_both_native_acceptance_and_actual_target_damage(self):
        rig=Rig(raw(),[])
        for accepted, damage, expected in ((True,True,("blocker_damage",)),
                                            (True,False,()),(False,True,())):
            with self.subTest(accepted=accepted,damage=damage):
                rig.env._dive_blocker_audit={"accepted":accepted,"damage_observed":damage}
                reasons=rig.env._action_effect_reasons(
                    raw(),raw(),requested_action=11,engage_target_generation_key=None,
                    native_kills=0,exploration_before=0,softwalls_before=0,
                    drink_audit=None,action14_audit=None)
                self.assertEqual(reasons,expected)
        rig.env.dive_blocker_recovery="off"
        self.assertEqual(rig.env._action_effect_reasons(
            raw(),raw(),requested_action=11,engage_target_generation_key=None,
            native_kills=0,exploration_before=0,softwalls_before=0,
            drink_audit=None,action14_audit=None),())

    def test_settle_at_already_reached_live_horizon_cannot_advance_or_wait(self):
        rig=Rig(raw(),[],maximum=100)
        rig.env._dive_blocker_audit={"accepted":True}
        with patch.object(bridge,"act_wait") as wait:
            observed,beats=rig.env._settle_to_idle(
                raw(mode=99),0,max_beats=5000,start_scene=_scene_identity(raw()))
        self.assertEqual(beats,0)
        wait.assert_not_called()


    def test_unready_set_level_return_keeps_old_walk_and_permission(self):
        before=raw(is_set_level=True, set_level_id=3,
                   resource_state={"readiness":{"ready":False}},
                   triggers=[dict(x=13,y=10,msg=bridge.WM_DIABRTNLVL)])
        frame=deepcopy(before)
        frame.update(player_x=11,future_x=11)
        traces=[]
        for flag in ("off","adjacent-v1"):
            rig=Rig(before,[frame],flag=flag,path=[(11,10,False)])
            observed,beats=rig.run(budget=1)
            self.assertEqual((observed["player_x"],beats),(11,1))
            self.assertFalse(observed["resource_state"]["readiness"]["ready"])
            self.assertFalse(any(c[0] == "attack" for c in rig.calls))
            self.assertFalse(hasattr(rig.env,"_dive_blocker_audit"))
            traces.append(rig.calls)
        self.assertEqual(traces[0],traces[1])

    def test_town_next_trigger_keeps_old_walk(self):
        before=raw(dungeon_level=0)
        frame=deepcopy(before)
        frame.update(player_x=11,future_x=11)
        rig=Rig(before,[frame],path=[(11,10,False)])
        observed,beats=rig.run(budget=1)
        self.assertEqual((observed["player_x"],beats),(11,1))
        self.assertFalse(any(c[0] == "attack" for c in rig.calls))


if __name__ == "__main__":
    unittest.main()
