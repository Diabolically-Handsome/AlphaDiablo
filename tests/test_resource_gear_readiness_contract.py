"""Sustain-v6 identity/wiring tests; every game boundary is mocked."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "tests")]
import eval_contract as contract
import eval_assembled as evaluation
import migrate_resource_candidate as migration
import train_ppo as training
from test_eval_pipeline import _valid_v5_archive
from test_resource_sustain_contract import training_identity, DummyEnv
from test_resource_sustain_gold_v4 import raw
from test_env_v4_semantics import raw as controller_raw, controller_fixture
from diablogym import nav
from diablogym.env import DiabloGymEnv, _scene_identity
from diablogym.options_env import OptionsEnv
from diablogym.resource_protocol import (
    validate_equipment_readiness_preservation,
    validate_native_equipment_readiness_preservation,
)
from diablogym.resource_sustain_armor import SustainEquipmentReadinessService
from diablogym.worker_env import WorkerWindowEnv

V6 = dict(resource_protocol="l2-town-v1", resource_service_policy="sustain-v6")
PRESERVATION = {
    "version": "equipment-readiness-preservation-v1",
    "scope": "native-a14-eligibility-plan-and-commit",
    "equipment_failures": ["armor", "damage", "weapon", "durability"],
    "trigger": "current-equipment-subset-ready",
    "requirement": "next-equipment-subset-ready",
    "independent_failures": ["level", "health", "potions"],
    "unready_equipment": "legacy-upgrade-rule",
}


def marked_raw():
    state = raw()
    state["resource_state"].update(ordinary_armor_scope=True,
                                   preserve_equipment_readiness=True)
    return state


@pytest.mark.parametrize("protocol,flag", [("off", True), ("l2-town-v1", 1),
    ("l2-town-v1", "true"), ("l2-town-v1", None)])
def test_invalid_low_level_identity_rejected_before_engine_construction(protocol, flag):
    with pytest.raises(ValueError):
        validate_equipment_readiness_preservation(protocol, flag)
    with patch("diablogym.env.bridge.init") as init:
        with pytest.raises(ValueError):
            DiabloGymEnv(resource_protocol=protocol, resource_preserve_equipment_readiness=flag)
        init.assert_not_called()


def test_constructor_identity_is_readonly_and_old_instances_default_false():
    env = DiabloGymEnv.__new__(DiabloGymEnv)
    assert env.resource_preserve_equipment_readiness is False
    env._resource_preserve_equipment_readiness = True
    assert env.resource_preserve_equipment_readiness is True
    with pytest.raises(AttributeError):
        env.resource_preserve_equipment_readiness = False


@pytest.mark.parametrize("policy,override", [("sustain-v5", True), ("sustain-v4", True),
    ("sustain-v6", False), ("sustain-v6", 1)])
def test_options_rejects_policy_flag_conflicts_before_constructor(policy, override):
    with patch("diablogym.options_env.DiabloGymEnv") as constructor:
        with pytest.raises(ValueError, match="must match"):
            OptionsEnv(resource_protocol="l2-town-v1", resource_service_policy=policy,
                       resource_preserve_equipment_readiness=override)
        constructor.assert_not_called()


@pytest.mark.parametrize("policy", ["legacy-v1", "sustain-v4", "sustain-v5", "sustain-v6"])
def test_only_v6_derives_native_preservation_and_service_identity(policy):
    base = SimpleNamespace(observation_space=gym.spaces.Box(-1, 1, shape=(295,)), _raw=None)
    with patch("diablogym.options_env.DiabloGymEnv", return_value=base) as constructor:
        options = OptionsEnv(max_steps=6000, resource_protocol="l2-town-v1", resource_service_policy=policy)
    kwargs = constructor.call_args.kwargs
    if policy == "sustain-v6":
        assert kwargs["resource_ordinary_armor_scope"] is True
        assert kwargs["resource_preserve_equipment_readiness"] is True
        assert type(options.resource_service) is SustainEquipmentReadinessService
        assert options.resource_service.telemetry()["policy"] == "sustain-v6"
        assert options.resource_service.service_microstep_cap == 1500
    else:
        assert "resource_preserve_equipment_readiness" not in kwargs
    assert options.max_steps == 6000


@pytest.mark.parametrize("protocol,ordinary,preserve", [("off", False, False),
    ("l2-town-v1", False, False), ("l2-town-v1", True, False),
    ("l2-town-v1", True, True)])
def test_native_abi_remains_one_or_two_arguments_for_old_policies(protocol, ordinary, preserve):
    native = SimpleNamespace(end_game=Mock(), configure_resource_protocol=Mock())
    env = SimpleNamespace(resource_protocol=protocol, resource_ordinary_armor_scope=ordinary)
    if preserve:
        env.resource_preserve_equipment_readiness = True
    with patch("diablogym.env.bridge", native):
        DiabloGymEnv._configure_native_resource_protocol(env)
    native.end_game.assert_called_once_with()
    kwargs = ({"ordinary_armor_scope": True, "preserve_equipment_readiness": True}
              if preserve else {"ordinary_armor_scope": True} if ordinary else {})
    native.configure_resource_protocol.assert_called_once_with(protocol != "off", **kwargs)


def test_v6_cannot_fall_back_to_an_old_native_signature():
    called = []
    native = SimpleNamespace(end_game=Mock(), configure_resource_protocol=
                             lambda enabled, ordinary_armor_scope=False: called.append(enabled))
    with patch("diablogym.env.bridge", native):
        with pytest.raises(TypeError):
            DiabloGymEnv._configure_native_resource_protocol(SimpleNamespace(
                resource_protocol="l2-town-v1", resource_ordinary_armor_scope=True,
                resource_preserve_equipment_readiness=True))
    assert called == []


@pytest.mark.parametrize("expected,actual", [(True, None), (True, False), (True, 1),
    (False, True), (False, "false")])
def test_marker_mismatch_rejects_without_mutating_already_returned_raw(expected, actual):
    state = raw()
    if actual is not None:
        state["resource_state"]["preserve_equipment_readiness"] = actual
    before = deepcopy(state)
    with pytest.raises(RuntimeError, match="service version"):
        validate_native_equipment_readiness_preservation(state, expected)
    assert state == before


def test_tick_validates_existing_raw_before_observers_without_extra_native_calls():
    state = marked_raw(); before = deepcopy(state); calls = []
    env = SimpleNamespace(resource_protocol="l2-town-v1", resource_ordinary_armor_scope=True,
        resource_preserve_equipment_readiness=True, ticks_per_step=4,
        _resource_actual_microsteps=8,
        resource_observation_callback=lambda e, r, t: calls.append(("memory", r, t)),
        resource_tick_callback=lambda e, r, t: calls.append(("ledger", r, t)))
    with patch("diablogym.env.bridge.step", return_value=state) as step:
        assert DiabloGymEnv._step_native(env) is state
        step.assert_called_once_with(ticks=4)
    assert state == before and [entry[::2] for entry in calls] == [("memory", 9), ("ledger", 9)]
    state["resource_state"].pop("preserve_equipment_readiness")
    with patch("diablogym.env.bridge.step", return_value=state) as step:
        with pytest.raises(RuntimeError):
            DiabloGymEnv._step_native(env)
        step.assert_called_once_with(ticks=4)
    assert env._resource_actual_microsteps == 10 and len(calls) == 2


def test_finish_macro_checks_its_existing_synchronous_refresh_without_extra_ticks():
    state = marked_raw(); refreshed = deepcopy(state)
    refreshed["resource_state"].pop("preserve_equipment_readiness")
    with patch("diablogym.env.bridge.act_wait", return_value=1) as wait, \
         patch("diablogym.env.bridge.observe", return_value=refreshed) as observe, \
         patch("diablogym.env.bridge.step") as step:
        with pytest.raises(RuntimeError, match="service version"):
            DiabloGymEnv._finish_macro(state, 7, _scene_identity(state))
    wait.assert_called_once_with(); observe.assert_called_once_with(); step.assert_not_called()


def test_a14_commit_refresh_missing_marker_fails_before_next_step():
    state = controller_raw(floor_items=[dict(x=10, y=10, heal=False, gear=True,
                                            visible=True, reachable=True)])
    state["resource_state"] = marked_raw()["resource_state"]
    env = controller_fixture(state)
    env.resource_protocol = "l2-town-v1"
    env.resource_ordinary_armor_scope = True
    env._resource_preserve_equipment_readiness = True
    committed = deepcopy(state); committed["floor_items"] = []
    committed["gear_combat_utility"] = state["gear_combat_utility"] + 37
    committed["resource_state"].pop("preserve_equipment_readiness")
    with patch("diablogym.env.bridge.act_pickup_gear_at", return_value=1) as pickup, \
         patch("diablogym.env.bridge.observe", return_value=committed) as observe, \
         patch("diablogym.env.bridge.step") as step:
        with pytest.raises(RuntimeError, match="service version"):
            env._macro_pickup("gear", controller_snapshot=env._controller_snapshot)
    pickup.assert_called_once(); observe.assert_called_once_with(); step.assert_not_called()


def test_reset_navigation_validates_each_existing_observation_and_step():
    start = marked_raw(); start.update(dungeon_level=0, player_x=1, player_y=1,
                                     triggers=[dict(msg=42, x=2, y=2)])
    end = marked_raw(); end.update(player_x=2, player_y=2)
    def run(validator, after=end):
        bridge = SimpleNamespace(observe=Mock(return_value=start), step=Mock(return_value=after),
                                 act_walk=Mock(), WM_DIABNEXTLVL=42)
        result = nav.descend_to_dungeon(bridge, raw_validator=validator)
        assert bridge.observe.call_count == 2 and bridge.step.call_count == 1
        bridge.act_walk.assert_called_once_with(2, 2)
        return result
    seen = []
    def validator(state):
        seen.append(state)
        validate_native_equipment_readiness_preservation(state, True)
    assert run(validator) is end and seen == [start, start, end]
    assert run(None) is end
    bad = deepcopy(end); bad["resource_state"].pop("preserve_equipment_readiness")
    with pytest.raises(RuntimeError, match="service version"):
        run(validator, bad)


def test_v6_recipe_is_an_explicit_extension_without_relabelling_v5():
    old = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v5")
    digest = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # Exact recipe from the frozen candidate-sustain-v5, captured before v6.
    assert digest == "9656af53d3afedb6ca06e9dcee5afdf7af3753eae047c0a1c3eaf999413dbd3b"
    new = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v6")
    assert new["native_ordinary_armor_scope"] is True
    assert new["native_preserve_equipment_readiness"] is True
    assert new["equipment_readiness_preservation"] == PRESERVATION
    omitted = {"version", "service_policy", "native_preserve_equipment_readiness", "equipment_readiness_preservation"}
    assert {k: v for k, v in new.items() if k not in omitted} == {k: v for k, v in old.items() if k not in omitted}
    assert new["collect_command_window_microsteps"] == 450 and new["service_microstep_cap"] == 1500


@pytest.mark.parametrize("change", ["policy", "flag", "failures", "scope", "trigger", "early_growth"])
def test_eval_rejects_forged_or_relabelled_v6_recipe(change):
    archive = _valid_v5_archive()
    archive["meta"]["protocol"] = contract.make_protocol([7, 8], r16_environment=V6)
    contract.validate_eval_archive(archive)
    protocol = archive["meta"]["protocol"]; recipe = protocol["resource_service_recipe"]
    if change == "policy": protocol["r16_environment"]["resource_service_policy"] = "sustain-v5"
    elif change == "flag": recipe["native_preserve_equipment_readiness"] = False
    elif change == "failures": recipe["equipment_readiness_preservation"]["equipment_failures"].remove("durability")
    elif change == "scope": recipe["equipment_readiness_preservation"]["scope"] = "mask-only"
    elif change == "trigger": recipe["equipment_readiness_preservation"]["trigger"] = "fully-ready"
    else: recipe["equipment_readiness_preservation"]["unready_equipment"] = "reject-all"
    with pytest.raises(contract.EvalContractError):
        contract.validate_eval_archive(archive)


def test_v4_g0_and_v5_cannot_silently_resume_as_v6_with_any_override():
    v6 = training_identity(**V6)
    training._validate_resume_contract(v6, deepcopy(v6))
    for policy in ("sustain-v4", "sustain-v5"):
        old = training_identity(resource_protocol="l2-town-v1", resource_service_policy=policy)
        for before, after in ((old, v6), (v6, old), (None, v6)):
            with pytest.raises(ValueError, match="resource_service"):
                training._validate_resume_contract(before, after, allow_environment_restart=True,
                    allow_legacy_resume=True, allow_optimizer_reset=True, allow_manager_change=True,
                    allow_target_kl_change=True)
    bad = deepcopy(v6); bad["resource_service_recipe"]["native_preserve_equipment_readiness"] = False
    with pytest.raises(ValueError): training._validate_resource_resume_identity(bad, v6)


@pytest.mark.parametrize("mode,symbol", [("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")])
def test_training_factory_forwards_version_without_changing_actions(mode, symbol):
    with patch(f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
        env = training.make_env(**{mode: True}, **V6)
    assert constructor.call_args.kwargs["resource_service_policy"] == "sustain-v6"
    assert env.action_space.n == 15
    env.close()


def test_eval_and_worker_factories_forward_only_explicit_version():
    constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
    with patch.object(evaluation, "_native_runtime", return_value=(Mock(), constructor, None)):
        with pytest.raises(RuntimeError, match="constructor-boundary"):
            evaluation.evaluate(None, [], r16_environment=V6)
    assert constructor.call_args.kwargs["resource_service_policy"] == "sustain-v6"
    with patch("diablogym.worker_env.OptionsEnv", constructor):
        with pytest.raises(RuntimeError, match="constructor-boundary"):
            WorkerWindowEnv(manager_npz=None, manager_heuristic="readiness-v3", **V6)
    assert constructor.call_args.kwargs["resource_service_policy"] == "sustain-v6"
    assert "resource_preserve_equipment_readiness" not in constructor.call_args.kwargs


def test_v6_warmstart_is_explicit_exact_r16_migration_not_resume():
    parent = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
    if not parent.is_file(): pytest.skip("registered R16 source contract fixture unavailable")
    with zipfile.ZipFile(parent) as archive:
        source = json.loads(archive.read("data"))["diablogym_contract"]
    target = migration.target_contract(source, "a" * 64, "sustain-v6", "adjacent-v1")
    migration.validate_target_contract(source, target, "a" * 64)
    assert migration.operation_for("sustain-v6") == "r16-to-sustain-v6-weights-only-v1"
    assert target["action_n"] == source["action_n"] == 15
    assert target["observation_shape"] == source["observation_shape"]
    assert target["max_steps"] == source["max_steps"]
    assert migration.ALLOWED_CONTRACT_KEYS == migration.RESOURCE_KEYS | {"implementation_sha256", "dive_blocker_recovery"}
    wrong = deepcopy(target)
    wrong["resource_service_recipe"] = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v5")
    with pytest.raises(ValueError, match="contract drift"):
        migration.validate_target_contract(source, wrong, "a" * 64)
    with pytest.raises(ValueError, match="exact registered R16"):
        migration.target_contract(target, "a" * 64, "sustain-v6")


def test_training_and_eval_hash_all_python_raw_validation_paths():
    for name in ("python/diablogym/env.py", "python/diablogym/nav.py",
                 "python/diablogym/resource_protocol.py",
                 "python/diablogym/resource_sustain_armor.py"):
        assert name in training._IMPLEMENTATION_SOURCE_FILES
        assert name in contract.PROTOCOL_SOURCE_FILES
