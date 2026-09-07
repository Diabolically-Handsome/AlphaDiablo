"""Real wrapper/native wiring fixtures; no policy, training or effect episodes."""
from diablogym import bridge
from diablogym.options_env import OptionsEnv, WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC


def test_real_reset_and_accounted_wait_feed_new_service_without_changing_shape():
    owner = OptionsEnv(max_steps=6000, resource_protocol="l2-town-v1",
        resource_purchase_mode="full", resource_service_policy="sustain-loot-v1",
        worker_time_protocol="completion-l2-v1",
        worker_observation_view=WORKER_OBSERVATION_VIEW_DUAL_V4_ASYMMETRIC)
    try:
        obs, info = owner.reset(seed=424399)
        raw = owner.env._raw
        assert raw["dungeon_level"] == 1
        assert raw["resource_state"]["loot_economy"] is True
        assert raw["resource_state"]["service_trips_started"] == 0
        assert owner.env.action_space.n == 15
        assert owner.max_steps == 6000 and owner.env.max_steps == 12000
        assert owner.resource_service._latest["gold"] == raw["gold"]
        before = owner.env._resource_actual_microsteps
        owner.env.step_resource(("wait",))
        assert owner.env._resource_actual_microsteps > before
        assert owner.resource_service._observed_step == owner.env._steps
        assert owner._completion_clock.state.micro_step == owner.env._steps
        assert owner.resource_service.trip_count == 0
        assert owner.resource_service.gold_sold == 0
        assert owner.action_space.n == 3
    finally:
        owner.close()
        bridge.end_game()
        bridge.configure_resource_protocol(False)
