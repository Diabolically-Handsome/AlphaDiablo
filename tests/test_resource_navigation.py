"""Pure-state regression tests for the opt-in real-action obstacle recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "python/diablogym/resource_navigation.py"
SPEC = importlib.util.spec_from_file_location("resource_navigation_under_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Recovery = MODULE.ResourceNavigationRecovery


def state(x=10, y=10, hp=70, belt=0):
    return {"player_x":x, "player_y":y, "future_x":x, "future_y":y,
            "dungeon_level":1, "is_set_level":False, "hp":hp, "max_hp":70,
            "belt_heal_kinds":[1]*belt + [0]*(8-belt), "monsters":[]}


def monster(identifier, x, y, hp=5, **extra):
    return {"id":identifier,"x":x,"y":y,"future_x":x,"future_y":y,
            "hp":hp,"visible":True,"type":1,"is_invalid":False,**extra}


class ResourceNavigationTests(unittest.TestCase):
    def decision(self, planner, raw, now=0, deadline=1500, target=(20,20), **kw):
        return planner.next_command(raw,target,now=now,service_deadline=deadline,**kw)

    def test_r16_2114006_visible_one_hp_choke_produces_real_adjacent_attack(self):
        raw=state(43,38)
        raw["monsters"]=[monster(103,44,37,hp=1)]
        result=self.decision(Recovery(),raw,target=(86,44))
        self.assertEqual(result.command,("attack_monster",103,12))
        self.assertEqual(result.reason,"fight_adjacent_blocker")

    def test_certified_2114005_boxed_future_occupancy_targets_one_visible_monster(self):
        raw=state(51,83)
        raw["monsters"]=[monster(48,51,82,hp=7),
            monster(52,49,83,hp=1,future_x=50,future_y=83),
            monster(81,53,83,hp=9,future_x=52,future_y=83),
            monster(25,51,85,hp=8,future_x=51,future_y=84)]
        before=deepcopy(raw)
        result=self.decision(Recovery(),raw,target=(72,48))
        self.assertEqual(result.command,("attack_monster",52,12))
        self.assertEqual(raw,before)

    def test_no_unobserved_remote_invalid_or_minion_chase(self):
        raw=state()
        raw["monsters"]=[monster(1,11,10,visible=False),monster(2,12,10),
            monster(3,11,10,hp=0),monster(4,11,10,is_invalid=True),
            monster(5,11,10,type=109),monster(6,11,10,is_player_minion=True)]
        self.assertEqual(self.decision(Recovery(),raw).command,("wait",))

    def test_attack_geometry_uses_player_and_monster_future_positions(self):
        raw=state();raw["future_x"]=11
        raw["monsters"]=[monster(1,13,10,future_x=12),monster(2,10,9,hp=1,future_x=9)]
        self.assertEqual(self.decision(Recovery(),raw).command,("attack_monster",1,12))

    def test_native_deadline_last_tick_clamps_attack_and_never_returns_tick_1501(self):
        raw=state();raw["monsters"]=[monster(1,11,10)]
        planner=Recovery()
        result=self.decision(planner,raw,now=1499)
        self.assertEqual(result.command,("attack_monster",1,1))
        self.assertEqual(result.max_microsteps,1)
        end=self.decision(planner,raw,now=1500)
        self.assertIsNone(end.command)
        self.assertEqual(end.reason,"resource_service_cap")

    def test_budget_uses_actual_microsteps_not_calls(self):
        planner=Recovery();raw=state()
        for unused in range(100):
            self.assertEqual(self.decision(planner,raw,now=500).command,("wait",))
        self.assertEqual(planner.telemetry()["episodes"],1)
        self.assertEqual(planner.telemetry()["requested_commands"],{"wait":1})
        end=self.decision(planner,raw,now=548)
        self.assertIsNone(end.command)
        self.assertEqual(end.reason,"resource_navigation_no_progress")

    def test_transient_open_path_without_actual_movement_does_not_reset_block_budget(self):
        planner=Recovery();raw=state()
        self.decision(planner,raw,now=100)
        self.decision(planner,raw,now=120,path_available=True)
        result=self.decision(planner,raw,now=148)
        self.assertEqual(result.reason,"resource_navigation_no_progress")
        self.assertEqual(planner.episodes,1)
        self.assertEqual(planner.route_resumptions,0)

    def test_new_target_without_physical_progress_does_not_reset_block_budget(self):
        planner=Recovery();raw=state()
        self.decision(planner,raw,now=100,target=(20,20))
        result=self.decision(planner,raw,now=148,target=(5,5))
        self.assertEqual(result.reason,"resource_navigation_no_progress")

    def test_actual_damage_renews_progress_fence_but_not_total_local_budget(self):
        planner=Recovery(max_blocked_microsteps=60)
        raw=state();raw["monsters"]=[monster(1,11,10,hp=20)]
        self.decision(planner,raw,now=100)
        raw["monsters"][0]["hp"]=19
        self.assertEqual(self.decision(planner,raw,now=140).command,("attack_monster",1,12))
        last=self.decision(planner,raw,now=159)
        self.assertEqual(last.command,("attack_monster",1,1))
        end=self.decision(planner,raw,now=160)
        self.assertEqual(end.reason,"resource_navigation_recovery_cap")

    def test_fractional_native_hp_damage_is_real_progress(self):
        planner=Recovery();raw=state()
        raw["monsters"]=[monster(1,11,10,hp=2,hp_fixed_hi=0,hp_fixed_lo=150)]
        self.decision(planner,raw,now=0)
        raw["monsters"][0]["hp_fixed_lo"]=149
        self.assertIsNotNone(self.decision(planner,raw,now=47).command)
        self.assertIsNotNone(self.decision(planner,raw,now=60).command)

    def test_disappearing_monster_does_not_count_as_kill_or_success(self):
        planner=Recovery();raw=state();raw["monsters"]=[monster(1,11,10,hp=1)]
        self.decision(planner,raw)
        raw["monsters"]=[]
        result=self.decision(planner,raw,now=12)
        self.assertEqual(result.command,("wait",))
        self.assertEqual(planner.route_resumptions,0)

    def test_real_replan_and_player_movement_resume_route_and_allow_new_block(self):
        planner=Recovery();raw=state();raw["monsters"]=[monster(1,11,10,hp=1)]
        self.decision(planner,raw)
        raw["monsters"]=[]
        route=self.decision(planner,raw,now=12,path_available=True)
        self.assertIsNone(route.command)
        raw["player_x"]=raw["future_x"]=11
        self.decision(planner,raw,now=16,path_available=True)
        self.assertEqual(planner.route_resumptions,1)
        self.assertFalse(planner.active)
        self.assertEqual(self.decision(planner,raw,now=500).command,("wait",))
        self.assertEqual(planner.episodes,2)

    def test_heal_consumes_real_available_instant_kind_before_attack(self):
        planner=Recovery();raw=state(hp=35,belt=1)
        raw["monsters"]=[monster(1,11,10)]
        self.assertEqual(self.decision(planner,raw).command,("drink",))
        # Only the next real observation reports a consumed bottle / HP change.
        raw["belt_heal_kinds"]=[0]*8;raw["hp"]=55
        self.assertEqual(self.decision(planner,raw,now=1).command,("attack_monster",1,12))

    def test_scroll_only_legacy_count_never_schedules_healing(self):
        raw=state(hp=1);raw["belt_heals"]=99
        raw["monsters"]=[monster(1,11,10)]
        self.assertEqual(self.decision(Recovery(),raw).command,("attack_monster",1,12))

    def test_canonical_instant_heal_count_fallback(self):
        raw=state(hp=1);del raw["belt_heal_kinds"]
        raw["resource_state"]={"readiness":{"belt_heals":1}}
        self.assertEqual(self.decision(Recovery(),raw).command,("drink",))

    def test_sticky_live_target_does_not_switch_on_new_closer_weaker_monster(self):
        planner=Recovery();raw=state();raw["monsters"]=[monster(1,11,10,hp=5)]
        self.decision(planner,raw)
        raw["monsters"].append(monster(2,10,11,hp=1))
        self.assertEqual(self.decision(planner,raw,now=12).command,("attack_monster",1,12))

    def test_reused_monster_id_has_distinct_generation(self):
        planner=Recovery();raw=state()
        raw["monsters"]=[monster(1,11,10,hp=5,rnd_item_seed_lo=10)]
        self.decision(planner,raw)
        raw["monsters"]=[monster(1,11,10,hp=50,rnd_item_seed_lo=11),monster(2,10,11,hp=1)]
        self.assertEqual(self.decision(planner,raw,now=12).command,("attack_monster",2,12))

    def test_scene_transition_or_terminal_state_cannot_emit_combat(self):
        for change in ({"dungeon_level":0},{"is_set_level":True},{"dead":True},
                       {"game_over":True},{"victory":True},{"hp":0}):
            with self.subTest(change=change):
                planner=Recovery();raw=state();raw["monsters"]=[monster(1,11,10)]
                self.decision(planner,raw)
                raw.update(change)
                self.assertIsNone(self.decision(planner,raw,now=12).command)

    def test_clock_regression_and_invalid_config_are_rejected(self):
        planner=Recovery();raw=state();self.decision(planner,raw,now=5)
        with self.assertRaises(ValueError):self.decision(planner,raw,now=4)
        for value in (True,0,-1,1.5):
            with self.subTest(value=value),self.assertRaises(ValueError):
                Recovery(max_blocked_microsteps=value)
        with self.assertRaises(ValueError):self.decision(Recovery(),raw,now=True)
        with self.assertRaises(ValueError):self.decision(Recovery(),raw,deadline=-1)


if __name__ == "__main__":
    unittest.main()
