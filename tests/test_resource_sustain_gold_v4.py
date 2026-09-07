"""V4 command-window identity and passive native-tail handoff evidence."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from diablogym.resource_sustain_gold import (
    SustainGoldExtendedService, SustainGoldMemoryService,
)


def gold(active_id=4):
    return dict(active_id=active_id, seed_hi=1, seed_lo=2, create_info=3,
                base_id=0, x=11, y=10, value=11)


def raw(items=(), *, x=10, cash=100, ready=False, busy=False):
    return dict(dungeon_level=1, is_set_level=False, player_x=x, player_y=10,
                future_x=x+int(busy), future_y=10, player_mode=2 if busy else 0,
                walkpath0=-1, dest_action=-1, hp=70, max_hp=70, gold=cash, monsters=[],
                resource_state=dict(enabled=True, gold_items=list(items),
                    readiness=dict(ready=ready, failures=[] if ready else ["armor"])))


def env(state, steps=1000, actual=None):
    result = SimpleNamespace(_raw=state, _steps=steps,
        _plan_descend_path=lambda raw, x, y, **kw: [(x, y, False)])
    if actual is not None:
        result._resource_actual_microsteps = actual
    return result


def bridge():
    return SimpleNamespace(configure_town_service=Mock(), WM_DIABPREVLVL=1,
                           act_wait=Mock(), step=Mock(), act_pickup_gold_at=Mock())


def service(cls):
    result = cls()
    result.start(raw(), 1000, "farm_cap", farm_scene_steps=3600)
    result._stairs = lambda env, message: ("walk", 9, 10)
    return result


def test_v3_schema_and_dynamic_diagnostic_constant_remain_unchanged():
    old = service(SustainGoldMemoryService)
    assert old.collect_microstep_cap == 300 and old.command_microstep_deadline == 1300
    info = old.telemetry()
    assert info["policy"] == "sustain-v3" and info["collect_microstep_cap"] == 300
    assert "collect_handoff" not in info and "collect_command_window_microsteps" not in info
    with patch("diablogym.resource_sustain_gold.OBSERVED_GOLD_COLLECT_CAP", 600):
        assert old.collect_microstep_cap == 600 and old.command_microstep_deadline == 1600
        assert old.telemetry()["collect_microstep_cap"] == 600
        extended = service(SustainGoldExtendedService)
        assert extended.collect_microstep_cap == 450
        assert extended.command_microstep_deadline == 1450


def test_v4_telemetry_names_command_window_without_claiming_physical_cap():
    extended = service(SustainGoldExtendedService)
    info = extended.telemetry()
    assert info["policy"] == "sustain-v4"
    assert info["collect_command_window_microsteps"] == 450
    assert "collect_microstep_cap" not in info
    assert info["collect_handoff"] is None
    assert info["service_microstep_cap"] == 1500


def test_v4_cutoff_records_busy_state_and_existing_tail_is_charged_outbound():
    extended = service(SustainGoldExtendedService)
    state = raw([gold()], busy=True); before = deepcopy(state)
    environment = env(state, 1450, actual=1450); native = bridge()
    assert extended.command(environment, native) == ("walk", 9, 10)
    assert environment._raw == before and environment._resource_actual_microsteps == 1450
    assert extended.phase == "outbound" and extended.command_microstep_deadline == 2500
    handoff = extended.telemetry()["collect_handoff"]
    assert handoff == dict(absolute_microstep=1450, elapsed_service_microsteps=450,
        collect_command_deadline=1450, reason="collect_budget", player_mode=2,
        tile=[10,10], future=[11,10], walkpath0=-1, dest_action=-1)
    native.configure_town_service.assert_called_once_with(True)
    native.act_wait.assert_not_called(); native.step.assert_not_called()
    native.act_pickup_gold_at.assert_not_called()
    # This is accounting for subsequent real beats supplied by the caller,
    # not permission for this module to step or finish native animations.
    extended.record_steps(1452, "outbound")
    assert extended.phase_steps["collect"] == 450
    assert extended.phase_steps["outbound"] == 2
    assert extended.steps == 452 and extended.service_microstep_cap == 1500
    handoff["future"][0] = 999
    state["future_x"] = 999
    assert extended.telemetry()["collect_handoff"]["future"] == [11,10]


def test_handoff_prefers_actual_native_clock_and_retains_unknown_raw_fields():
    extended = service(SustainGoldExtendedService)
    state = raw(); del state["future_x"]; del state["future_y"]
    environment = env(state, 1449, actual=1450)
    assert extended._leave_collect(environment, bridge(), "collect_budget") == ("walk", 9, 10)
    handoff = extended.telemetry()["collect_handoff"]
    assert handoff["absolute_microstep"] == 1450
    assert handoff["elapsed_service_microsteps"] == 450
    assert handoff["future"] == [None, None]
    fallback = service(SustainGoldExtendedService)
    fallback._leave_collect(env(raw(), 1450), bridge(), "collect_budget")
    assert fallback.telemetry()["collect_handoff"]["absolute_microstep"] == 1450


def test_v4_reuses_exact_diagnostic_450_command_and_receipt_sequence():
    def execute(cls):
        result = service(cls); native = bridge(); result.gold_memory.observe(raw([gold()]), 900)
        environment = env(raw()); commands = [result.command(environment, native)]
        environment._raw = raw([gold(active_id=8)], x=11); environment._steps=1001
        commands.append(result.command(environment, native))
        result.receipt(commands[-1], {"accepted":True, "reason":"gold", "price":0})
        assert result.gold_collected == 0
        environment._raw = raw([], x=11, cash=111); environment._steps=1002
        commands.append(result.command(environment, native))
        return commands, result.gold_collected, result.phase, result.gold_memory.candidates(), native.configure_town_service.call_args_list
    with patch("diablogym.resource_sustain_gold.OBSERVED_GOLD_COLLECT_CAP", 450):
        assert execute(SustainGoldMemoryService) == execute(SustainGoldExtendedService)


def test_v4_stops_at_total_budget_and_does_not_add_handoff_to_v3():
    extended = service(SustainGoldExtendedService)
    assert extended.command(env(raw([gold()]), 2500, actual=2500), bridge()) == ("finish", "resource_service_cap")
    assert extended.gold_collected == 0 and extended.steps == 1500
    assert extended.telemetry()["collect_handoff"] is None
    old = service(SustainGoldMemoryService)
    assert old.command(env(raw(), 1300), bridge()) == ("walk", 9, 10)
    assert "collect_handoff" not in old.telemetry()


def test_v4_ready_early_handoff_uses_live_verdict_without_native_actions():
    extended = service(SustainGoldExtendedService); native = bridge()
    assert extended.command(env(raw(ready=True), 1001, actual=1001), native) == ("complete",)
    assert extended.telemetry()["collect_handoff"]["reason"] == "observed_targets_exhausted"
    assert extended.reason == "ready_without_town"
    native.configure_town_service.assert_not_called(); native.act_wait.assert_not_called(); native.step.assert_not_called()


def test_options_v4_factory_and_real_reset_lifecycle_isolate_memory():
    import gymnasium as gym
    import numpy as np
    from diablogym.options_env import OptionsEnv
    base = SimpleNamespace(observation_space=gym.spaces.Box(-1,1,shape=(295,)),
                           _raw=None, _steps=0, _resource_actual_microsteps=0)
    def reset_base(*, seed, options):
        # The old episode must not receive observations from native reset.
        assert base.resource_observation_callback is None
        base._raw = raw([]); base._steps = base._resource_actual_microsteps = 0
        return np.zeros(295,dtype=np.float32), {"seed":seed}
    base.reset = Mock(side_effect=reset_base)
    with patch("diablogym.options_env.DiabloGymEnv", return_value=base) as constructor:
        options = OptionsEnv(resource_protocol="l2-town-v1", resource_service_policy="sustain-v4", max_steps=6000)
    options._mgr_obs = lambda obs: obs
    assert isinstance(options.resource_service, SustainGoldExtendedService)
    first = options.resource_service.gold_memory
    base.resource_observation_callback(base, raw([gold()]), 7)
    assert len(first.candidates()) == 1
    options.reset(seed=2114006)
    second = options.resource_service.gold_memory
    assert second is not first and second.candidates() == []
    base.resource_observation_callback(base, raw([gold(active_id=8)]), 1)
    assert len(second.candidates()) == 1 and first.candidates()[0]["active_id"] == 4
    options.reset(seed=2114006)  # Same seed still means a fresh episode.
    assert options.resource_service.gold_memory is not second
    assert options.resource_service.gold_memory.candidates() == []
    options.resource_service.start(base._raw, 0, "farm_cap", farm_scene_steps=3600)
    assert options.resource_service.command_microstep_deadline == 450
    assert options.resource_service.service_microstep_cap == 1500
    assert "resource_service_policy" not in constructor.call_args.kwargs


def test_options_v4_dispatch_installs_450_command_deadline():
    import gymnasium as gym
    import numpy as np
    from diablogym.options_env import OptionsEnv, RESUPPLY
    base = env(raw([gold()]), 1000, actual=1000)
    base.observation_space = gym.spaces.Box(-1,1,shape=(295,))
    base._ensure_active = lambda: None
    with patch("diablogym.options_env.DiabloGymEnv", return_value=base):
        options = OptionsEnv(resource_protocol="l2-town-v1", resource_service_policy="sustain-v4", max_steps=6000)
    options.resource_service.start(base._raw, 1000, "farm_cap", farm_scene_steps=3600)
    def begin(option):
        options._win = {"mode":"resupply", "last_info":{}}
    captured = []
    options._win_begin = begin
    options._consume_fuse_recovery = lambda: None
    options._win_beat = lambda action: (captured.append(base._resource_service_deadline)
                                       or SimpleNamespace(reason="cap"))
    options._win_end = lambda reason: ({"R": 0.0}, {}, False, False)
    options._last_base_obs = np.zeros(295,dtype=np.float32)
    options._mgr_obs = lambda obs: obs
    options.step(RESUPPLY)
    assert captured == [1450]
    assert options.resource_service.phase == "collect"
