"""R17.1 chairman ruling 3: the coach-v03 readiness law wired end to end.

Deterministic raw-state doubles and constructor boundaries only. No engine
reset, native step, optimizer update or model checkpoint is required.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import eval_contract as contract
import eval_assembled as evaluation
import train_ppo as training
from test_eval_pipeline import _valid_v5_archive
from test_resource_sustain_contract import training_identity

spec = importlib.util.spec_from_file_location(
    "readiness_law_under_test", ROOT / "python/diablogym/resource_protocol.py")
resource = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = resource
spec.loader.exec_module(resource)

COACH = {"resource_protocol": "l2-town-v1", "resource_readiness_law": "coach-v03"}


class DummyEnv(gym.Env):
    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.observation_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.action_space = spaces.Discrete(15)


def raw(depth=1, ready=False, ready_excluding_health=None,
        advisory=False, enabled=True):
    """Native payload double; coach-v03 needs the advisory + six-condition keys."""
    readiness = {"ready": ready, "failures": [] if ready else ["armor"]}
    if ready_excluding_health is not None:
        readiness["ready_excluding_health"] = ready_excluding_health
    state = {"enabled": enabled, "protocol": "l2-town-v1", "readiness": readiness}
    if advisory:
        state["readiness_advisory"] = True
    return {"dungeon_level": depth, "is_set_level": False,
            "player_x": 10, "player_y": 10, "hp": 70, "max_hp": 70,
            "gold": 100, "belt_heals": 4, "resource_state": state}


class ReadinessLawTests(unittest.TestCase):
    def test_law_identity_is_versioned_and_coach_requires_the_town_protocol(self):
        self.assertEqual(resource.RESOURCE_READINESS_LAWS, ("veto-v1", "coach-v03"))
        self.assertEqual(resource.validate_readiness_law("off", "veto-v1"), "veto-v1")
        self.assertEqual(resource.validate_readiness_law("l2-town-v1", "coach-v03"),
                         "coach-v03")
        for protocol, law in (("off", "coach-v03"), ("off", "coach-v3"),
                              ("l2-town-v1", "coach-v04"), ("l2-town-v1", None)):
            with self.subTest(protocol=protocol, law=law), self.assertRaises(ValueError):
                resource.validate_readiness_law(protocol, law)

    def test_native_acknowledgement_fails_closed_on_the_configured_flag(self):
        self.assertIsNone(resource.validate_native_readiness_law(raw(), "veto-v1"))
        self.assertIsNone(resource.validate_native_readiness_law({}, "veto-v1"))
        self.assertIsNone(resource.validate_native_readiness_law(
            raw(advisory=True, ready_excluding_health=True), "coach-v03"))
        # veto-v1 never tolerates an advisory bridge; coach-v03 never tolerates a
        # bridge that has not acknowledged the law or lacks the six-condition key.
        for payload, law in (
                (raw(advisory=True), "veto-v1"),
                (raw(ready_excluding_health=True), "coach-v03"),
                (raw(advisory=True), "coach-v03"),
                (raw(advisory=True, ready_excluding_health=True, enabled=False),
                 "coach-v03"),
                ({}, "coach-v03")):
            with self.subTest(law=law), self.assertRaises(RuntimeError):
                resource.validate_native_readiness_law(payload, law)

    def test_law_ready_reads_the_six_condition_verdict_only_under_coach(self):
        health_only_failure = raw(ready=False, ready_excluding_health=True)
        self.assertFalse(resource.law_ready(health_only_failure))
        self.assertFalse(resource.law_ready(health_only_failure, "veto-v1"))
        self.assertTrue(resource.law_ready(health_only_failure, "coach-v03"))
        gear_failure = raw(ready=True, ready_excluding_health=False)
        self.assertTrue(resource.law_ready(gear_failure, "veto-v1"))
        self.assertFalse(resource.law_ready(gear_failure, "coach-v03"))
        with self.assertRaises(RuntimeError):
            resource.law_ready(raw(ready=True), "coach-v03")

    def test_engine_transition_guard_only_records_under_coach(self):
        for depth in (1, 2):
            with self.subTest(depth=depth):
                unready = raw(depth, ready=False, ready_excluding_health=False)
                self.assertFalse(resource.progression_allowed(unready))
                self.assertFalse(resource.progression_allowed(unready, "veto-v1"))
                self.assertTrue(resource.progression_allowed(unready, "coach-v03"))
        self.assertTrue(resource.progression_allowed(raw(0), "veto-v1"))
        self.assertTrue(resource.progression_allowed(raw(1, ready=True), "veto-v1"))
        for key, value in (("is_set_level", True),
                           ("progression_targets", [{"kind": "diablo_switch"}])):
            for law in ("veto-v1", "coach-v03"):
                quest = raw(3)
                quest[key] = value
                with self.subTest(key=key, law=law):
                    self.assertTrue(resource.progression_allowed(quest, law))


class WrapperPassthroughTests(unittest.TestCase):
    """Both training wrappers only forward the law when it is not the default."""

    def _forwarded(self, module, symbol, factory):
        constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
        with patch.object(module, symbol, constructor):
            with self.assertRaisesRegex(RuntimeError, "constructor-boundary"):
                factory()
        return constructor.call_args.kwargs

    def test_options_env_forwards_only_the_explicit_law(self):
        from diablogym import options_env as module
        for law in ("veto-v1", "coach-v03"):
            with self.subTest(law=law):
                forwarded = self._forwarded(
                    module, "DiabloGymEnv",
                    lambda law=law: module.OptionsEnv(
                        resource_protocol="l2-town-v1", resource_readiness_law=law))
                if law == "veto-v1":
                    self.assertNotIn("resource_readiness_law", forwarded)
                else:
                    self.assertEqual(forwarded["resource_readiness_law"], law)
        with self.assertRaises(ValueError):
            module.OptionsEnv(resource_readiness_law="coach-v03")

    def test_worker_window_env_forwards_only_the_explicit_law(self):
        from diablogym import worker_env as module
        for law in ("veto-v1", "coach-v03"):
            with self.subTest(law=law):
                forwarded = self._forwarded(
                    module, "OptionsEnv",
                    lambda law=law: module.WorkerWindowEnv(
                        manager_npz=None, manager_heuristic="level-margin-1",
                        resource_protocol="l2-town-v1", resource_readiness_law=law))
                if law == "veto-v1":
                    self.assertNotIn("resource_readiness_law", forwarded)
                else:
                    self.assertEqual(forwarded["resource_readiness_law"], law)
        with self.assertRaises(ValueError):
            module.WorkerWindowEnv(manager_npz=None,
                                   manager_heuristic="level-margin-1",
                                   resource_readiness_law="coach-v03")


class DescendEscrowSettlementTests(unittest.TestCase):
    """R17.1: forced-unready descents are accepted by the engine but vest nothing."""

    def _settle(self, law, receipt):
        from diablogym.worker_env import WorkerWindowEnv
        env = SimpleNamespace(
            descend_escrow_fraction=0.5, descend_escrow_power=1.0,
            descend_escrow_readiness_gate=True,
            descend_escrow_readiness_table="v2",
            resource_protocol="l2-town-v1", resource_readiness_law=law,
            _descend_escrow=0.0, stats={},
            oe=SimpleNamespace(env=SimpleNamespace(
                _econ_episode_max_depth=2, _raw={},
                reward_economy=SimpleNamespace(descend_unit=48.0),
                _resource_transition_receipts=[receipt])))
        WorkerWindowEnv._descend_escrow_settlement(env, 1, "descend")
        return env

    def test_vest_ratio_reads_the_law_specific_pretransition_verdict(self):
        health_only_failure = {"target_depth": 2, "accepted": True,
                               "pretransition_ready": False,
                               "pretransition_ready_law": True}
        gear_failure = {"target_depth": 2, "accepted": True,
                        "pretransition_ready": True,
                        "pretransition_ready_law": False}
        for law, receipt, escrow in (("veto-v1", health_only_failure, 0.0),
                                     ("coach-v03", health_only_failure, 24.0),
                                     ("veto-v1", gear_failure, 24.0),
                                     ("coach-v03", gear_failure, 0.0)):
            with self.subTest(law=law, escrow=escrow):
                env = self._settle(law, receipt)
                self.assertEqual(env._descend_escrow, escrow)
                self.assertEqual(env.stats.get("descend_escrow_unready_denied", 0.0),
                                 0.0 if escrow else 24.0)
                entry = env.stats["descend_escrow_gate_log"][0]
                self.assertEqual(entry["table"], "native-l2-town-v1/coach-v03"
                                 if law == "coach-v03" else "native-l2-town-v1")
                self.assertEqual(entry["passed"], bool(escrow))

    def test_a_forced_descent_without_a_receipt_vests_nothing_under_either_law(self):
        for law in ("veto-v1", "coach-v03"):
            with self.subTest(law=law):
                env = self._settle(law, {"target_depth": 9, "accepted": True,
                                         "pretransition_ready": True,
                                         "pretransition_ready_law": True})
                self.assertEqual(env._descend_escrow, 0.0)


class EvalContractTests(unittest.TestCase):
    def test_default_law_is_omitted_and_coach_needs_the_town_protocol(self):
        self.assertEqual(contract.R16_ENVIRONMENT_DEFAULTS["resource_readiness_law"],
                         "veto-v1")
        self.assertEqual(contract.validate_r16_environment(COACH), COACH)
        for invalid in ({"resource_readiness_law": "coach-v03"},
                        {"resource_protocol": "l2-town-v1",
                         "resource_readiness_law": "veto-v1"},
                        {"resource_protocol": "l2-town-v1",
                         "resource_readiness_law": "coach-v04"}):
            with self.subTest(invalid=invalid), self.assertRaises(
                    contract.EvalContractError):
                contract.validate_r16_environment(invalid)

    def test_exam_identity_round_trips_and_the_default_archive_is_unchanged(self):
        self.assertNotIn("r16_environment", contract.make_protocol([2114000]))
        protocol = contract.make_protocol([7, 8], r16_environment=COACH)
        self.assertEqual(protocol["r16_environment"], COACH)
        self.assertEqual(protocol["resource_service_recipe"],
                         contract.resource_service_recipe("l2-town-v1", "full"))
        document = _valid_v5_archive()
        document["meta"]["protocol"] = protocol
        contract.validate_eval_archive(document)
        plain = _valid_v5_archive()
        plain["meta"]["protocol"] = contract.make_protocol(
            [7, 8], r16_environment={"resource_protocol": "l2-town-v1"})
        self.assertNotIn("resource_readiness_law", plain["meta"]["protocol"]
                         ["r16_environment"])
        contract.validate_eval_archive(plain)
        forged = deepcopy(document)
        forged["meta"]["protocol"]["r16_environment"]["resource_readiness_law"] = "veto-v1"
        with self.assertRaises(contract.EvalContractError):
            contract.validate_eval_archive(forged)


class _Parsed(Exception):
    pass


def parsed_args(case, *flags):
    """CLI defaults without any environment/model work (see test_resource_warm_start)."""
    captured = {}
    def capture(args):
        captured["args"] = args
        raise _Parsed()
    with patch.object(sys, "argv", ["train_ppo.py", *flags]), patch.object(
            training, "_validate_args", side_effect=capture):
        with case.assertRaises(_Parsed):
            training._main(SimpleNamespace())
    return captured["args"]


class TrainingWiringTests(unittest.TestCase):
    def test_cli_default_keeps_the_legacy_law_and_rejects_unknown_values(self):
        self.assertEqual(parsed_args(self).resource_readiness_law, "veto-v1")
        self.assertEqual(parsed_args(
            self, "--resource-readiness-law", "coach-v03").resource_readiness_law,
            "coach-v03")
        with patch.object(sys, "argv", ["train_ppo.py", "--resource-readiness-law",
                                        "coach-v04"]):
            with self.assertRaises(SystemExit):
                training._main(SimpleNamespace())

    def test_training_contract_writes_none_off_and_resume_drift_is_whitelisted(self):
        missing = training_identity()
        self.assertIsNone(missing["resource_readiness_law"])
        self.assertEqual(missing, training_identity(resource_readiness_law="veto-v1"))
        old = dict(missing)
        del old["resource_readiness_law"]
        training._validate_resume_contract(old, missing)
        saved = training_identity(resource_protocol="l2-town-v1")
        current = training_identity(**COACH)
        self.assertEqual(current["resource_readiness_law"], "coach-v03")
        with self.assertRaisesRegex(ValueError, "resource_readiness_law"):
            training._validate_resume_contract(saved, current)
        training._validate_resume_contract(saved, current, allow_environment_restart=True)

    def test_train_factory_forwards_only_the_explicit_law_to_both_modes(self):
        for mode, symbol in (("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")):
            for config in ({}, {"resource_protocol": "l2-town-v1"}, COACH):
                with self.subTest(mode=mode, config=config), patch(
                        f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
                    env = training.make_env(**{mode: True}, **config)
                    forwarded = constructor.call_args.kwargs
                    if "resource_readiness_law" in config:
                        self.assertEqual(forwarded["resource_readiness_law"],
                                         config["resource_readiness_law"])
                    else:
                        self.assertNotIn("resource_readiness_law", forwarded)
                    env.close()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                training.make_env(**{mode: True}, resource_readiness_law="coach-v03")

    def test_validate_args_rejects_the_coach_law_without_the_town_protocol(self):
        args = parsed_args(self)
        args.total_steps = 8192
        training._validate_args(args)
        args.resource_readiness_law = "coach-v03"
        with self.assertRaisesRegex(ValueError, "resource-readiness-law"):
            training._validate_args(args)
        args.resource_readiness_law = "veto-v9"
        with self.assertRaisesRegex(ValueError, "resource-readiness-law"):
            training._validate_args(args)


class EvalWiringTests(unittest.TestCase):
    def test_eval_forwards_the_law_and_omits_the_default_at_the_constructor(self):
        for config in (None, {"resource_protocol": "l2-town-v1"}, COACH):
            constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
            with self.subTest(config=config), patch.object(
                    evaluation, "_native_runtime",
                    return_value=(Mock(), constructor, None)):
                with self.assertRaisesRegex(RuntimeError, "constructor-boundary"):
                    evaluation.evaluate(None, [], r16_environment=config)
            forwarded = constructor.call_args.kwargs
            if config and "resource_readiness_law" in config:
                self.assertEqual(forwarded["resource_readiness_law"], "coach-v03")
            else:
                self.assertNotIn("resource_readiness_law", forwarded)


if __name__ == "__main__":
    unittest.main()
