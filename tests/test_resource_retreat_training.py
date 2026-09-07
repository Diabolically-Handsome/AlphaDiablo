"""R18-B: retreat-v1 wired into the training window (worker/contract/CLI).

Constructor and pure-function boundaries only. No engine reset, no native
step, no optimizer update, no checkpoint. WorkerWindowEnv is exercised with a
stubbed OptionsEnv (the ``test_resource_retreat.py`` pattern) and the escrow
settlement is called unbound against a synthetic ``self``.
"""
from types import SimpleNamespace
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "train"), str(ROOT / "python"), str(ROOT / "tests")]

import eval_contract as contract
import eval_assembled as evaluation
import train_ppo as training
from diablogym import worker_env
from diablogym.resource_protocol import (
    RESOURCE_RETREAT_PROTOCOLS,
    validate_retreat_protocol,
)

# The only classroom retreat-v1 is legal in (mirrors the engine-side guard).
RETREAT = {"resource_protocol": "l2-town-v1",
           "resource_readiness_law": "coach-v03",
           "resource_retreat": "retreat-v1"}


class DummyEnv(gym.Env):
    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.observation_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.action_space = spaces.Discrete(15)


class StubOptions:
    """Enough of OptionsEnv for the WorkerWindowEnv constructor; no engine."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.env = SimpleNamespace(
            observation_space=SimpleNamespace(shape=(295,)),
            max_steps=kwargs.get("max_steps", 3000))


def build_worker(**overrides):
    arguments = dict(manager_npz=None, manager_heuristic="readiness-v1")
    arguments.update(overrides)
    with patch.object(worker_env, "OptionsEnv",
                      side_effect=lambda **kw: StubOptions(**kw)) as options:
        return worker_env.WorkerWindowEnv(**arguments), options


class WorkerPassthroughTests(unittest.TestCase):
    """(a)/(b) the constructor now forwards instead of rejecting."""

    def test_retreat_v1_reaches_the_options_constructor_in_the_legal_classroom(self):
        env, options = build_worker(**RETREAT, resource_purchase_mode="full")
        self.assertEqual(env.resource_retreat, "retreat-v1")
        options.assert_called_once()
        self.assertEqual(env.oe.kwargs["resource_retreat"], "retreat-v1")
        # The readiness law must ride along; the engine reads both.
        self.assertEqual(env.oe.kwargs["resource_readiness_law"], "coach-v03")

    def test_the_default_and_explicit_off_add_no_keyword_at_all(self):
        for overrides in ({}, {"resource_retreat": "off"}):
            with self.subTest(overrides=overrides):
                env, _ = build_worker(**overrides)
                self.assertEqual(env.resource_retreat, "off")
                self.assertNotIn("resource_retreat", env.oe.kwargs)

    def test_retreat_v1_is_rejected_under_veto_v1_and_without_the_town_protocol(self):
        # The readiness law is validated first, so a coach-v03 request without
        # the town protocol dies on that older guard; either way OptionsEnv is
        # never reached and no env_kwargs key is produced.
        for overrides, pattern in (
                ({"resource_retreat": "retreat-v1"},
                 "retreat-v1 requires l2-town-v1 under coach-v03"),
                ({"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                  "resource_retreat": "retreat-v1"},
                 "retreat-v1 requires l2-town-v1 under coach-v03"),
                ({"resource_readiness_law": "coach-v03",
                  "resource_retreat": "retreat-v1"},
                 "coach-v03 requires l2-town-v1")):
            with self.subTest(overrides=overrides):
                with patch.object(worker_env, "OptionsEnv") as options:
                    with self.assertRaisesRegex(ValueError, pattern):
                        build_worker(**overrides)
                options.assert_not_called()

    def test_unknown_retreat_values_are_rejected_before_the_constructor(self):
        for value in ("retreat-v2", "on", True):
            with self.subTest(value=value):
                overrides = dict(RETREAT, resource_purchase_mode="full",
                                 resource_retreat=value)
                with patch.object(worker_env, "OptionsEnv") as options:
                    with self.assertRaisesRegex(ValueError, "Unknown resource_retreat"):
                        build_worker(**overrides)
                options.assert_not_called()

    def test_the_protocol_table_is_versioned(self):
        self.assertEqual(RESOURCE_RETREAT_PROTOCOLS, ("off", "retreat-v1"))
        self.assertEqual(validate_retreat_protocol("off", "veto-v1", "off"), "off")


def escrow_self(resource_retreat="off", pending=7.5, fraction=0.5, depth=1):
    """Synthetic ``self`` for the unbound settlement call (forfeit path only
    touches fraction/_descend_escrow/stats/resource_retreat; the vest path
    additionally reads oe.env)."""
    return SimpleNamespace(
        descend_escrow_fraction=fraction,
        descend_escrow_power=1.0,
        descend_escrow_readiness_gate=False,
        resource_retreat=resource_retreat,
        resource_protocol="l2-town-v1",
        resource_readiness_law="coach-v03",
        _descend_escrow=float(pending),
        stats={},
        oe=SimpleNamespace(env=SimpleNamespace(
            _econ_episode_max_depth=depth, reward_economy=None, _raw={})))


def settle(fake, d_before, reason):
    return worker_env.WorkerWindowEnv._descend_escrow_settlement(
        fake, d_before, reason)


class RetreatEscrowForfeitureTests(unittest.TestCase):
    """(e) R18-B 撤退触发的窗不发托管 —— retreat_trigger is death-equivalent."""

    def test_retreat_trigger_forfeits_exactly_like_death_when_the_law_is_on(self):
        for reason in ("death", "retreat_trigger"):
            with self.subTest(reason=reason):
                fake = escrow_self(resource_retreat="retreat-v1")
                self.assertEqual(settle(fake, 3, reason), 0.0)
                self.assertEqual(fake._descend_escrow, 0.0)
                self.assertEqual(fake.stats["descend_escrow_forfeited"], 7.5)
                self.assertNotIn("descend_escrow_vested", fake.stats)

    def test_every_other_close_reason_still_vests_under_the_retreat_law(self):
        for reason in ("scene", "stall", "descend", "retreat_complete", None):
            with self.subTest(reason=reason):
                fake = escrow_self(resource_retreat="retreat-v1")
                self.assertEqual(settle(fake, 3, reason), 7.5)
                self.assertEqual(fake.stats["descend_escrow_vested"], 7.5)
                self.assertNotIn("descend_escrow_forfeited", fake.stats)

    def test_with_the_law_off_retreat_trigger_is_an_ordinary_vesting_close(self):
        # House law: nothing about the off path may move. The reason cannot
        # actually occur when retreat is off, but the branch is explicit.
        fake = escrow_self(resource_retreat="off")
        self.assertEqual(settle(fake, 3, "retreat_trigger"), 7.5)
        self.assertEqual(fake.stats["descend_escrow_vested"], 7.5)
        self.assertNotIn("descend_escrow_forfeited", fake.stats)
        # And a missing attribute (older pickled envs) behaves like off.
        legacy = escrow_self()
        del legacy.resource_retreat
        self.assertEqual(settle(legacy, 3, "retreat_trigger"), 7.5)

    def test_the_flag_off_valve_still_returns_zero_without_touching_stats(self):
        fake = escrow_self(resource_retreat="retreat-v1", fraction=0.0)
        self.assertEqual(settle(fake, 3, "retreat_trigger"), 0.0)
        self.assertEqual(fake.stats, {})
        self.assertEqual(fake._descend_escrow, 7.5)


class WindowReasonCountersTests(unittest.TestCase):
    """(3) _log counts arbitrary reason strings; retreat needs no new code."""

    def test_retreat_reasons_are_counted_in_stats_and_ff_stats(self):
        fake = SimpleNamespace(
            log_windows=False, window_log=[],
            stats={"reasons": {}, "ff_reasons": {}, "ff_windows": 0})
        for reason, ff in (("retreat_trigger", False), ("retreat_complete", True),
                           ("retreat_complete", True)):
            worker_env.WorkerWindowEnv._log(fake, {"reason": reason}, ff)
        self.assertEqual(fake.stats["reasons"],
                         {"retreat_trigger": 1, "retreat_complete": 2})
        self.assertEqual(fake.stats["ff_reasons"], {"retreat_complete": 2})
        self.assertEqual(fake.stats["ff_windows"], 2)


class EvalContractTests(unittest.TestCase):
    """(c) the eval-side identity key."""

    def test_the_default_is_off_so_every_existing_archive_stays_byte_identical(self):
        self.assertEqual(contract.R16_ENVIRONMENT_DEFAULTS["resource_retreat"], "off")
        self.assertNotIn("r16_environment", contract.make_protocol([2114000]))

    def test_the_validator_accepts_retreat_v1_only_in_the_legal_classroom(self):
        self.assertEqual(contract.validate_r16_environment(dict(RETREAT)), RETREAT)
        for invalid in ({"resource_retreat": "off"},
                        {"resource_retreat": "retreat-v2",
                         "resource_protocol": "l2-town-v1",
                         "resource_readiness_law": "coach-v03"},
                        {"resource_retreat": "retreat-v1"},
                        {"resource_retreat": "retreat-v1",
                         "resource_protocol": "l2-town-v1"},
                        {"resource_retreat": "retreat-v1",
                         "resource_readiness_law": "coach-v03"}):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, contract.EvalContractError)):
                    contract.validate_r16_environment(invalid)

    def test_the_key_survives_the_protocol_round_trip(self):
        protocol = contract.make_protocol([7, 8], r16_environment=dict(RETREAT))
        self.assertEqual(protocol["r16_environment"]["resource_retreat"], "retreat-v1")


class _Parsed(Exception):
    pass


def parsed_args(case, *flags):
    """CLI defaults without any environment/model work."""
    captured = {}

    def capture(args):
        captured["args"] = args
        raise _Parsed()

    with patch.object(sys, "argv", ["train_ppo.py", *flags]), patch.object(
            training, "_validate_args", side_effect=capture):
        with case.assertRaises(_Parsed):
            training._main(SimpleNamespace())
    return captured["args"]


def training_identity(**overrides):
    args = SimpleNamespace(
        worker=False, options=True, flat_clock=False, arch="mlp",
        max_steps=6000, num_envs=1, n_steps=8, gamma=0.99, lr=3e-4,
        ent_coef=0.02, skip_dry=False, no_drink_sovereignty=False,
        dry_curriculum_schedule=None, bc_aux_lambda=0.0, bc_aux_demos=None,
        bc_aux_liveness_preflight=False, distill_beta=0.0,
        calib_record_only=False, **overrides)
    model = SimpleNamespace(max_grad_norm=0.5, action_space=spaces.Discrete(15),
                            observation_space=spaces.Box(-1, 1, (3,), dtype=np.float32),
                            device="cpu")
    return training._training_contract(args, model, batch_size=8)


class TrainingWiringTests(unittest.TestCase):
    """(d) the training-side contract key, CLI, validation and factory."""

    def test_the_training_contract_writes_none_when_off_and_the_literal_when_on(self):
        missing = training_identity()
        self.assertIn("resource_retreat", missing)
        self.assertIsNone(missing["resource_retreat"])
        self.assertEqual(missing, training_identity(resource_retreat="off"))
        # An old checkpoint that predates the key resumes without drift.
        old = dict(missing)
        del old["resource_retreat"]
        training._validate_resume_contract(old, missing)
        current = training_identity(**RETREAT)
        self.assertEqual(current["resource_retreat"], "retreat-v1")

    def test_a_retreat_change_is_drift_that_only_the_named_restart_may_ride(self):
        self.assertIn("resource_retreat", training._ENVIRONMENT_RESTART_ALLOWED_DRIFT)
        saved = training_identity(resource_protocol="l2-town-v1",
                                  resource_readiness_law="coach-v03")
        current = training_identity(**RETREAT)
        with self.assertRaisesRegex(ValueError, "resource_retreat"):
            training._validate_resume_contract(saved, current)
        training._validate_resume_contract(saved, current,
                                           allow_environment_restart=True)

    def test_the_resource_resume_identity_keys_cover_the_retreat_law(self):
        base = {"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                "resource_service_policy": "sustain-v6",
                "resource_service_recipe": training.resource_service_recipe(
                    "l2-town-v1", "full", "sustain-v6")}
        training._validate_resource_resume_identity(dict(base), dict(base))
        with self.assertRaisesRegex(ValueError, "separately identified initialization"):
            training._validate_resource_resume_identity(
                dict(base), dict(base, resource_retreat="retreat-v1"))

    def test_the_cli_defaults_to_off_and_rejects_unknown_values(self):
        self.assertEqual(parsed_args(self).resource_retreat, "off")
        self.assertEqual(parsed_args(
            self, "--resource-retreat", "retreat-v1").resource_retreat, "retreat-v1")
        with patch.object(sys, "argv",
                          ["train_ppo.py", "--resource-retreat", "retreat-v2"]):
            with self.assertRaises(SystemExit):
                training._main(SimpleNamespace())

    def test_validate_args_pins_retreat_to_the_town_protocol_and_coach_law(self):
        args = parsed_args(self)
        args.total_steps = 8192
        training._validate_args(args)
        args.resource_retreat = "retreat-v1"
        with self.assertRaisesRegex(ValueError, "resource-retreat"):
            training._validate_args(args)   # off protocol / veto law
        args.resource_protocol = "l2-town-v1"
        args.options, args.algo, args.gamma, args.max_steps = True, "mppo", 1.0, 6000
        with self.assertRaisesRegex(ValueError, "resource-retreat"):
            training._validate_args(args)   # still veto-v1
        args.resource_readiness_law = "coach-v03"
        training._validate_args(args)       # the legal classroom passes
        args.resource_retreat = "retreat-v9"
        with self.assertRaisesRegex(ValueError, "resource-retreat"):
            training._validate_args(args)

    def test_validate_args_refuses_retreat_without_worker_or_options(self):
        args = parsed_args(self)
        args.total_steps = 8192
        args.resource_protocol, args.resource_readiness_law = "l2-town-v1", "coach-v03"
        args.resource_retreat = "retreat-v1"
        with self.assertRaisesRegex(ValueError, "resource-retreat|resource-protocol"):
            training._validate_args(args)

    def test_the_env_factory_forwards_only_the_explicit_law_to_both_modes(self):
        for mode, symbol in (("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")):
            for config in ({}, {"resource_protocol": "l2-town-v1"}, RETREAT):
                with self.subTest(mode=mode, config=config), patch(
                        f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
                    env = training.make_env(**{mode: True}, **config)
                    forwarded = constructor.call_args.kwargs
                    if "resource_retreat" in config:
                        self.assertEqual(forwarded["resource_retreat"], "retreat-v1")
                    else:
                        self.assertNotIn("resource_retreat", forwarded)
                    env.close()
            with self.subTest(mode=mode), self.assertRaisesRegex(
                    ValueError, "resource_retreat"):
                training.make_env(**{mode: True}, resource_retreat="retreat-v1")

    def test_the_factory_refuses_retreat_outside_worker_or_options(self):
        with self.assertRaisesRegex(ValueError, "resource_retreat"):
            training.make_env(resource_protocol="l2-town-v1",
                              resource_readiness_law="coach-v03",
                              resource_retreat="retreat-v1")


class EvalAssembledWiringTests(unittest.TestCase):
    """(6) the certification eval must not silently rebuild a no-retreat env."""

    def test_the_evaluator_forwards_the_law_and_omits_the_default(self):
        for config in (None, {"resource_protocol": "l2-town-v1"}, RETREAT):
            constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
            with self.subTest(config=config), patch.object(
                    evaluation, "_native_runtime",
                    return_value=(Mock(), constructor, None)):
                with self.assertRaisesRegex(RuntimeError, "constructor-boundary"):
                    evaluation.evaluate(None, [], r16_environment=config)
            forwarded = constructor.call_args.kwargs
            if config and "resource_retreat" in config:
                self.assertEqual(forwarded["resource_retreat"], "retreat-v1")
            else:
                self.assertNotIn("resource_retreat", forwarded)

    def test_the_eval_cli_declares_the_flag_with_the_off_default(self):
        source = (ROOT / "train/eval_assembled.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--resource-retreat", default="off"', source)
        self.assertIn('"resource_retreat": args.resource_retreat,', source)


if __name__ == "__main__":
    unittest.main()
