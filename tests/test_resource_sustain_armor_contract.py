"""V5 wiring and version boundaries; every engine boundary is mocked."""
from copy import deepcopy
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
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import eval_contract as contract
import eval_assembled as evaluation
import migrate_resource_candidate as migration
import train_ppo as training
from test_eval_pipeline import _valid_v5_archive
from test_resource_sustain_contract import training_identity, DummyEnv
from test_resource_sustain_gold_v4 import raw, gold
from diablogym.env import DiabloGymEnv
from diablogym.options_env import OptionsEnv, RESUPPLY
from diablogym.resource_protocol import validate_native_armor_scope, validate_ordinary_armor_scope
from diablogym.resource_sustain_armor import SustainOrdinaryArmorService
from diablogym.worker_env import WorkerWindowEnv

V5 = dict(resource_protocol="l2-town-v1", resource_service_policy="sustain-v5")


@pytest.mark.parametrize("protocol,mode,scope", [("off", "full", True),
    ("l2-town-v1", "armor", True), ("l2-town-v1", "full", 1),
    ("l2-town-v1", "full", "true")])
def test_low_level_scope_rejects_nonboolean_disabled_or_nonfull(protocol, mode, scope):
    with pytest.raises(ValueError):
        validate_ordinary_armor_scope(protocol, mode, scope)


@pytest.mark.parametrize("policy,scope", [("sustain-v4", True), ("sustain-v5", False),
    ("sustain-v5", 1), ("legacy-v1", True)])
def test_service_policy_controls_scope_before_engine_construction(policy, scope):
    with patch("diablogym.options_env.DiabloGymEnv") as native_env:
        with pytest.raises(ValueError, match="must match"):
            OptionsEnv(resource_protocol="l2-town-v1", resource_service_policy=policy,
                       resource_ordinary_armor_scope=scope)
        native_env.assert_not_called()


@pytest.mark.parametrize("policy", ["legacy-v1", "sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5"])
def test_options_derives_true_scope_only_for_v5(policy):
    base = SimpleNamespace(observation_space=gym.spaces.Box(-1, 1, shape=(295,)), _raw=None)
    with patch("diablogym.options_env.DiabloGymEnv", return_value=base) as constructor:
        options = OptionsEnv(resource_protocol="l2-town-v1", resource_service_policy=policy)
    forwarded = constructor.call_args.kwargs
    assert "resource_service_policy" not in forwarded
    if policy == "sustain-v5":
        assert forwarded["resource_ordinary_armor_scope"] is True
        assert type(options.resource_service) is SustainOrdinaryArmorService
    else:
        assert "resource_ordinary_armor_scope" not in forwarded


def test_worker_validates_and_passes_v5_to_options_without_raw_flag_override():
    constructor = Mock(side_effect=RuntimeError("options-boundary"))
    with patch("diablogym.worker_env.OptionsEnv", constructor):
        with pytest.raises(RuntimeError, match="options-boundary"):
            WorkerWindowEnv(manager_npz=None, manager_heuristic="readiness-v3", **V5)
    assert constructor.call_args.kwargs["resource_service_policy"] == "sustain-v5"
    assert "resource_ordinary_armor_scope" not in constructor.call_args.kwargs


@pytest.mark.parametrize("protocol,scope,expected", [("off", False, False),
    ("l2-town-v1", False, True), ("l2-town-v1", True, True)])
def test_native_configuration_preserves_old_one_argument_abi(protocol, scope, expected):
    native = SimpleNamespace(end_game=Mock(), configure_resource_protocol=Mock())
    env = SimpleNamespace(resource_protocol=protocol, resource_ordinary_armor_scope=scope)
    with patch("diablogym.env.bridge", native):
        DiabloGymEnv._configure_native_resource_protocol(env)
    native.end_game.assert_called_once_with()
    if scope:
        native.configure_resource_protocol.assert_called_once_with(expected, ordinary_armor_scope=True)
    else:
        native.configure_resource_protocol.assert_called_once_with(expected)


def test_v5_cannot_silently_use_old_native_signature():
    calls = []
    native = SimpleNamespace(end_game=Mock(), configure_resource_protocol=lambda enabled: calls.append(enabled))
    with patch("diablogym.env.bridge", native):
        with pytest.raises(TypeError):
            DiabloGymEnv._configure_native_resource_protocol(SimpleNamespace(
                resource_protocol="l2-town-v1", resource_ordinary_armor_scope=True))
    assert calls == []


@pytest.mark.parametrize("expected,actual", [(True, None), (True, False), (True, 1),
                                          (False, True), (False, "false")])
def test_already_observed_native_scope_mismatch_fails_closed(expected, actual):
    state = raw()
    if actual is not None:
        state["resource_state"]["ordinary_armor_scope"] = actual
    before = deepcopy(state)
    with pytest.raises(RuntimeError, match="service version"):
        validate_native_armor_scope(state, expected)
    assert state == before


def test_tick_checks_scope_passively_without_extra_time_or_observation():
    state = raw(); state["resource_state"]["ordinary_armor_scope"] = True
    events = []
    env = SimpleNamespace(resource_protocol="l2-town-v1", resource_ordinary_armor_scope=True,
        ticks_per_step=4, _resource_actual_microsteps=8,
        resource_observation_callback=lambda e, r, t: events.append(("memory", t)),
        resource_tick_callback=lambda e, r, t: events.append(("audit", t)))
    before = deepcopy(state)
    with patch("diablogym.env.bridge.step", return_value=state) as native:
        assert DiabloGymEnv._step_native(env) is state
        native.assert_called_once_with(ticks=4)
    assert events == [("memory", 9), ("audit", 9)] and state == before
    state["resource_state"]["ordinary_armor_scope"] = False
    with patch("diablogym.env.bridge.step", return_value=state) as native:
        with pytest.raises(RuntimeError):
            DiabloGymEnv._step_native(env)
        native.assert_called_once_with(ticks=4)
    assert env._resource_actual_microsteps == 10 and len(events) == 2


def test_v5_reset_isolates_memory_and_dispatch_keeps_450_deadline():
    state = raw([gold()]); state["resource_state"]["ordinary_armor_scope"] = True
    base = SimpleNamespace(observation_space=gym.spaces.Box(-1, 1, shape=(295,)), _raw=None,
        _steps=1000, _resource_actual_microsteps=1000,
        _plan_descend_path=lambda r, x, y, **kw: [(x, y, False)], _ensure_active=lambda: None)
    with patch("diablogym.options_env.DiabloGymEnv", return_value=base):
        options = OptionsEnv(max_steps=6000, **V5)
    first = options.resource_service.gold_memory
    base.resource_observation_callback(base, state, 999)
    assert first.candidates()
    def reset_base(*, seed, options):
        assert base.resource_observation_callback is None
        base._raw = deepcopy(state)
        base._raw["resource_state"]["gold_items"] = []
        base._steps = base._resource_actual_microsteps = 0
        return np.zeros(295, dtype=np.float32), {}
    base.reset = reset_base
    options._mgr_obs = lambda obs: obs
    options.reset(seed=7)
    assert options.resource_service.gold_memory is not first
    assert options.resource_service.gold_memory.candidates() == []
    base._raw = state; base._steps = base._resource_actual_microsteps = 1000
    options.resource_service.gold_memory.observe(state, 1000)
    options.resource_service.start(state, 1000, "farm_cap", farm_scene_steps=3600)
    def begin(option):
        options._win = {"mode": "resupply", "last_info": {}}
    captured = []
    options._win_begin = begin
    options._consume_fuse_recovery = lambda: None
    options._win_beat = lambda action: (captured.append(base._resource_service_deadline)
                                      or SimpleNamespace(reason="cap"))
    options._win_end = lambda reason: ({"R": 0.0}, {}, False, False)
    options.step(RESUPPLY)
    assert captured == [1450] and options.resource_service.service_microstep_cap == 1500
    assert options.max_steps == 6000 and options.farm_scene_cap == 3600


def test_v5_recipe_binds_catalog_native_scope_and_same_v4_clocks():
    old = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v4")
    new = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v5")
    assert new["native_ordinary_armor_scope"] is True
    assert new["ordinary_armor_catalog"]["version"] == "ordinary-armor-v1"
    assert new["ordinary_armor_catalog"]["items"] == ["normal_helmet", "normal_shield", "normal_chest"]
    assert "native_ordinary_armor_scope" not in old and "ordinary_armor_catalog" not in old
    for field in ("stages", "service_microstep_cap", "potion_target", "gold_memory",
                  "collect_command_window_microsteps", "collect_tail", "collect_cutoff"):
        assert new[field] == old[field]


@pytest.mark.parametrize("change", ["policy", "scope", "version", "slots", "cap"])
def test_archive_rejects_relabelled_or_forged_v5_recipe(change):
    archive = _valid_v5_archive()
    archive["meta"]["protocol"] = contract.make_protocol([7, 8], r16_environment=V5)
    contract.validate_eval_archive(archive)
    protocol = archive["meta"]["protocol"]
    recipe = protocol["resource_service_recipe"]
    if change == "policy": protocol["r16_environment"]["resource_service_policy"] = "sustain-v4"
    elif change == "scope": recipe["native_ordinary_armor_scope"] = False
    elif change == "version": recipe["ordinary_armor_catalog"]["version"] = "ordinary-armor-v0"
    elif change == "slots": recipe["ordinary_armor_catalog"]["items"] = ["normal_chest"]
    else: recipe["collect_command_window_microsteps"] = 600
    with pytest.raises(contract.EvalContractError):
        contract.validate_eval_archive(archive)


def test_resume_rejects_v4_g0_version_even_with_broad_overrides_or_missing_contract():
    v4 = training_identity(resource_protocol="l2-town-v1", resource_service_policy="sustain-v4")
    v5 = training_identity(**V5)
    training._validate_resume_contract(v5, deepcopy(v5))
    for source, target in [(v4, v5), (v5, v4), (None, v5)]:
        with pytest.raises(ValueError, match="resource_service"):
            training._validate_resume_contract(source, target, allow_environment_restart=True,
                allow_legacy_resume=True, allow_optimizer_reset=True, allow_manager_change=True)
    forged = deepcopy(v5)
    forged["resource_service_recipe"]["native_ordinary_armor_scope"] = False
    with pytest.raises(ValueError, match="scope/version"):
        training._validate_resource_resume_identity(forged, v5)


@pytest.mark.parametrize("mode,symbol", [("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")])
def test_training_factory_passes_v5_as_identity_without_changing_model_actions(mode, symbol):
    with patch(f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
        env = training.make_env(**{mode: True}, **V5)
    assert constructor.call_args.kwargs["resource_service_policy"] == "sustain-v5"
    assert env.action_space.n == 15
    env.close()


def test_eval_factory_and_both_implementation_bundles_include_v5():
    constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
    with patch.object(evaluation, "_native_runtime", return_value=(Mock(), constructor, None)):
        with pytest.raises(RuntimeError, match="constructor-boundary"):
            evaluation.evaluate(None, [], r16_environment=V5)
    assert constructor.call_args.kwargs["resource_service_policy"] == "sustain-v5"
    name = "python/diablogym/resource_sustain_armor.py"
    assert name in training._IMPLEMENTATION_SOURCE_FILES and name in contract.PROTOCOL_SOURCE_FILES


def test_explicit_v5_initialization_keeps_registered_r16_source_and_separate_receipt():
    parent = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
    if not parent.is_file(): pytest.skip("registered R16 source fixture unavailable")
    with zipfile.ZipFile(parent) as archive:
        source = json.loads(archive.read("data"))["diablogym_contract"]
    target = migration.target_contract(source, "a" * 64, "sustain-v5")
    migration.validate_target_contract(source, target, "a" * 64)
    assert migration.operation_for("sustain-v5") == "r16-to-sustain-v5-weights-only-v1"
    assert target["action_n"] == source["action_n"] == 15
    assert target["observation_shape"] == source["observation_shape"]
    assert target["max_steps"] == source["max_steps"]
    with pytest.raises(ValueError, match="exact registered R16"):
        migration.target_contract(target, "a" * 64, "sustain-v5")
    wrong = deepcopy(target)
    wrong["resource_service_recipe"] = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v4")
    with pytest.raises(ValueError, match="contract drift"):
        migration.validate_target_contract(source, wrong, "a" * 64)
