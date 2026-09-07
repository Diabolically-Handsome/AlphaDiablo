"""Actual-clock execution contracts for new script commands (native doubles)."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from diablogym import bridge
from diablogym.env import DiabloGymEnv, _scene_identity


def state(mode=None, belt=2, hp=25, **changes):
    value={"player_x":10,"player_y":10,"future_x":10,"future_y":10,
           "dungeon_level":1,"is_set_level":False,"hp":hp,"max_hp":70,
           "belt_heal_kinds":[1]*belt+[0]*(8-belt),
           "player_mode":bridge.PM_STAND if mode is None else mode,
           "dest_action":bridge.ACTION_NONE,"walkpath0":bridge.WALK_NONE,
           "monsters":[{"id":7,"x":11,"y":10,"future_x":11,"future_y":10,
                        "visible":True,"hp":5,"type":1}],
           "equipped_items":[]}
    value.update(changes)
    return value


def environment(frames, *, now=10, deadline=1510, max_steps=6000, mode="full", callback=None):
    env=DiabloGymEnv.__new__(DiabloGymEnv)
    env._raw=state()
    env._resource_actual_microsteps=now
    env._resource_service_deadline=deadline
    env.max_steps=max_steps
    env.resource_purchase_mode=mode
    env._resource_calibration=None
    env._resource_pending_command=None
    env._record_visit=lambda xy:None
    remaining=iter(frames)
    def step_native():
        env._resource_actual_microsteps+=1
        observed=deepcopy(next(remaining))
        if callback is not None:callback(env,observed)
        return observed
    env._step_native=step_native
    return env


class ResourceRecoveryCommandTests(unittest.TestCase):
    def test_accepted_swing_reaches_impact_before_any_wait_cancellation(self):
        env=environment([state(bridge.PM_ATTACK),state(bridge.PM_ATTACK),state()])
        with patch("diablogym.bridge.act_controller_attack_monster",return_value=1) as attack, \
             patch("diablogym.bridge.act_wait") as wait:
            raw,beats,receipt=env._execute_resource_command(("attack_monster",7,12))
        attack.assert_called_once_with(7,10,10,1)
        wait.assert_not_called()
        self.assertEqual((beats,env._resource_actual_microsteps),(3,13))
        self.assertTrue(receipt["accepted"])
        self.assertTrue(env._decision_idle(raw))

    def test_rejected_attack_is_exactly_one_real_explicit_wait(self):
        env=environment([state()])
        with patch("diablogym.bridge.act_controller_attack_monster",return_value=0), \
             patch("diablogym.bridge.act_wait") as wait:
            unused,beats,receipt=env._execute_resource_command(("attack_monster",7,12))
        wait.assert_called_once()
        self.assertEqual((beats,env._resource_actual_microsteps),(1,11))
        self.assertFalse(receipt["accepted"])

    def test_attack_clamps_command_episode_and_service_budgets(self):
        for budget,deadline,maximum,expected in ((1,100,100,1),(99,100,100,12),
                                               (12,12,100,2),(12,100,12,2)):
            with self.subTest(budget=budget,deadline=deadline,maximum=maximum):
                env=environment([state(bridge.PM_ATTACK)]*expected,
                                deadline=deadline,max_steps=maximum)
                with patch("diablogym.bridge.act_controller_attack_monster",return_value=1), \
                     patch("diablogym.bridge.act_wait") as wait:
                    unused,beats,receipt=env._execute_resource_command(("attack_monster",7,budget))
                self.assertEqual(beats,expected)
                self.assertEqual(env._resource_actual_microsteps,10+expected)
                wait.assert_not_called()

    def test_final_service_tick_executes_once_but_expired_deadline_never_mutates(self):
        for now,expected in ((1499,1),(1500,0)):
            with self.subTest(now=now):
                env=environment([state(bridge.PM_ATTACK)]*expected,now=now,deadline=1500)
                with patch("diablogym.bridge.act_controller_attack_monster",return_value=1) as attack, \
                     patch("diablogym.bridge.act_wait") as wait:
                    unused,beats,receipt=env._execute_resource_command(("attack_monster",7,12))
                self.assertEqual(beats,expected)
                self.assertEqual(env._resource_actual_microsteps,1500)
                self.assertEqual(attack.call_count,expected)
                wait.assert_not_called()

    def test_live_callback_can_stop_attack_and_outer_settle_without_calibration(self):
        for field in ("max_steps","_resource_service_deadline"):
            with self.subTest(field=field):
                def stop(env,raw):setattr(env,field,env._resource_actual_microsteps)
                env=environment([state(bridge.PM_ATTACK)],callback=stop)
                env._resource_pending_command=("attack_monster",7,12)
                with patch("diablogym.bridge.act_controller_attack_monster",return_value=1), \
                     patch("diablogym.bridge.act_wait") as wait:
                    raw,beats,receipt=env._execute_resource_command(env._resource_pending_command)
                    raw,beats=env._settle_to_idle(raw,beats,max_beats=12,start_scene=_scene_identity(env._raw))
                self.assertEqual((beats,env._resource_actual_microsteps),(1,11))
                wait.assert_not_called()

    def test_outer_settle_cannot_cross_local_attack_budget(self):
        env=environment([state(bridge.PM_ATTACK)])
        env._resource_pending_command=("attack_monster",7,1)
        with patch("diablogym.bridge.act_controller_attack_monster",return_value=1), \
             patch("diablogym.bridge.act_wait") as wait:
            raw,beats,receipt=env._execute_resource_command(env._resource_pending_command)
            raw,beats=env._settle_to_idle(raw,beats,max_beats=1,start_scene=_scene_identity(env._raw))
        self.assertEqual(beats,1)
        wait.assert_not_called()

    def test_stop_on_death_scene_change_or_actual_target_leaving_adjacency(self):
        variants=[state(bridge.PM_ATTACK,dead=True),state(bridge.PM_ATTACK,dungeon_level=0),
                  state(bridge.PM_ATTACK,monsters=[]),state(bridge.PM_ATTACK,monsters=[
                      {"id":7,"x":12,"y":10,"future_x":12,"future_y":10,
                       "hp":5,"visible":True,"type":1}])]
        for next_raw in variants:
            with self.subTest(next_raw=next_raw):
                env=environment([next_raw])
                with patch("diablogym.bridge.act_controller_attack_monster",return_value=1), \
                     patch("diablogym.bridge.act_wait"):
                    unused,beats,receipt=env._execute_resource_command(("attack_monster",7,12))
                self.assertEqual(beats,1)

    def test_hidden_or_remote_target_is_not_submitted_to_native_attack(self):
        for changes in ({"visible":False},{"future_x":12}):
            with self.subTest(changes=changes):
                env=environment([state()]);env._raw["monsters"][0].update(changes)
                with patch("diablogym.bridge.act_controller_attack_monster") as attack, \
                     patch("diablogym.bridge.act_wait") as wait:
                    unused,beats,receipt=env._execute_resource_command(("attack_monster",7,12))
                attack.assert_not_called();wait.assert_called_once()
                self.assertFalse(receipt["accepted"])
                self.assertEqual(beats,1)

    def test_drink_native_return_two_is_accepted_with_actual_belt_consumption(self):
        env=environment([state(belt=1,hp=55)])
        order=[]
        with patch("diablogym.bridge.act_wait",side_effect=lambda:order.append("wait")), \
             patch("diablogym.bridge.act_drink",side_effect=lambda:order.append("drink") or 2):
            unused,beats,receipt=env._execute_resource_command(("drink",))
        self.assertEqual(order,["wait","drink"])
        self.assertEqual((beats,env._resource_actual_microsteps),(1,11))
        self.assertTrue(receipt["accepted"])
        self.assertTrue(receipt["consumed"])
        self.assertTrue(receipt["belt_consumed_observed"])
        self.assertEqual((receipt["belt_before"],receipt["belt_after"]),(2,1))
        self.assertEqual((receipt["hp_before"],receipt["hp_after"]),(25,55))

    def test_drink_rejection_costs_one_tick_without_false_consumption(self):
        env=environment([state()])
        with patch("diablogym.bridge.act_wait") as wait, \
             patch("diablogym.bridge.act_drink",return_value=0):
            unused,beats,receipt=env._execute_resource_command(("drink",))
        wait.assert_called_once()
        self.assertEqual(beats,1)
        self.assertFalse(receipt["accepted"])
        self.assertFalse(receipt["consumed"])
        self.assertFalse(receipt["post_tick_effect_visible"])

    def test_autorefill_and_intervening_damage_keep_native_and_net_observation_distinct(self):
        env=environment([state(belt=2,hp=20)])
        with patch("diablogym.bridge.act_wait"),patch("diablogym.bridge.act_drink",return_value=2):
            unused,beats,receipt=env._execute_resource_command(("drink",))
        self.assertTrue(receipt["accepted"])
        self.assertFalse(receipt["belt_consumed_observed"])
        self.assertFalse(receipt["post_tick_effect_visible"])
        self.assertEqual(receipt["hp_after"],20)

    def test_scroll_legacy_count_cannot_validate_invalid_drink_receipt(self):
        env=environment([]);env._raw["belt_heal_kinds"]=[0]*8;env._raw["belt_heals"]=99
        with patch("diablogym.bridge.act_wait"),patch("diablogym.bridge.act_drink",return_value=99), \
             self.assertRaises(RuntimeError):
            env._execute_resource_command(("drink",))
        self.assertEqual(env._resource_actual_microsteps,10)

    def test_invalid_drink_receipt_type_or_count_fails_without_phantom_tick(self):
        for returned in (1,3,None,"2"):
            with self.subTest(returned=returned):
                env=environment([])
                with patch("diablogym.bridge.act_wait"),patch("diablogym.bridge.act_drink",return_value=returned), \
                     self.assertRaises(RuntimeError):
                    env._execute_resource_command(("drink",))
                self.assertEqual(env._resource_actual_microsteps,10)

    def test_unequip_uses_normal_identity_command_and_observes_only_after_real_tick(self):
        env=environment([state(dungeon_level=0,equipped_items=[{"present":False}])])
        env._raw["dungeon_level"]=0;env._raw["equipped_items"]=[{"present":True}]
        command=("unequip",0,11,22,33,44)
        with patch("diablogym.bridge.act_unequip_equipped_item",create=True,
                   return_value={"accepted":True,"reason":"unequipped","price":0}) as native, \
             patch("diablogym.bridge.act_wait") as wait:
            raw,beats,receipt=env._execute_resource_command(command)
        native.assert_called_once_with(0,11,22,33,44)
        wait.assert_not_called()
        self.assertTrue(receipt["accepted"])
        self.assertEqual((beats,env._resource_actual_microsteps),(1,11))
        self.assertFalse(raw["equipped_items"][0]["present"])

    def test_unequip_full_only_and_rejection_no_false_item_effect(self):
        for mode in ("none","heal","potions","armor"):
            with self.subTest(mode=mode):
                env=environment([],mode=mode)
                with patch("diablogym.bridge.act_unequip_equipped_item",create=True) as native, \
                     self.assertRaises(ValueError):
                    env._execute_resource_command(("unequip",0,1,2,3,4))
                native.assert_not_called();self.assertEqual(env._resource_actual_microsteps,10)
        env=environment([state()])
        with patch("diablogym.bridge.act_unequip_equipped_item",create=True,
                   return_value={"accepted":False,"reason":"cannot_fit","price":0}), \
             patch("diablogym.bridge.act_wait") as wait:
            unused,beats,receipt=env._execute_resource_command(("unequip",0,1,2,3,4))
        wait.assert_called_once();self.assertEqual(beats,1);self.assertFalse(receipt["accepted"])

    def test_new_commands_never_mutate_or_advance_an_already_terminal_state(self):
        for command in (("attack_monster",7,12),("drink",),("unequip",0,1,2,3,4)):
            with self.subTest(command=command):
                env=environment([]);env._raw["dead"]=True
                with patch("diablogym.bridge.act_drink") as drink, \
                     patch("diablogym.bridge.act_controller_attack_monster") as attack, \
                     patch("diablogym.bridge.act_unequip_equipped_item",create=True) as unequip, \
                     patch("diablogym.bridge.act_wait") as wait:
                    unused,beats,receipt=env._execute_resource_command(command)
                drink.assert_not_called();attack.assert_not_called()
                unequip.assert_not_called();wait.assert_not_called()
                self.assertEqual(beats,0)

    def test_attack_stops_on_same_id_but_different_monster_generation(self):
        future=state(bridge.PM_ATTACK)
        future["monsters"][0]["rnd_item_seed_lo"]=2
        env=environment([future]);env._raw["monsters"][0]["rnd_item_seed_lo"]=1
        with patch("diablogym.bridge.act_controller_attack_monster",return_value=1), \
             patch("diablogym.bridge.act_wait") as wait:
            unused,beats,receipt=env._execute_resource_command(("attack_monster",7,12))
        self.assertEqual(beats,1);wait.assert_not_called()

    def test_drink_and_unequip_after_deadline_cannot_touch_inventory(self):
        for command in (("drink",),("unequip",0,1,2,3,4)):
            with self.subTest(command=command):
                env=environment([],now=1500,deadline=1500)
                with patch("diablogym.bridge.act_drink") as drink, \
                     patch("diablogym.bridge.act_unequip_equipped_item",create=True) as unequip, \
                     patch("diablogym.bridge.act_wait") as wait:
                    unused,beats,receipt=env._execute_resource_command(command)
                drink.assert_not_called();unequip.assert_not_called();wait.assert_not_called()
                self.assertEqual(beats,0)


if __name__ == "__main__":
    unittest.main()
