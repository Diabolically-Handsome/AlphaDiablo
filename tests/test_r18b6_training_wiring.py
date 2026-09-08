"""R18-B6 (2026-09-07): sweep-v1, cain-v1 and smith-v1 wired into training.

Three deployment interfaces that already existed on the engine side (R18-H1,
R18-H2, R18-K2b) but had no training wiring.  The training world must equal the
tested world, so all three reach WorkerWindowEnv/OptionsEnv, the training
contract, the CLI, the warm-start migration and the eval identity -- and all
three stay byte-identical on their default ``off``.

Mirrors ``tests/test_r18b5_hunt_portal_training.py`` flag for flag.  Constructor
and pure-function boundaries only: no engine reset, no native step, no optimizer
update, no checkpoint.  WorkerWindowEnv is exercised with a stubbed OptionsEnv
(the ``test_resource_retreat_training.py`` pattern) and the escrow settlement is
called unbound against a synthetic ``self``.
"""
import ast
import contextlib
import io
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
import migrate_loot_candidate as loot
import train_ppo as training
from diablogym import worker_env
from diablogym.resource_identify import RESOURCE_IDENTIFY_PROTOCOLS
from diablogym.resource_protocol import RESOURCE_SWEEP_PROTOCOLS
from diablogym.resource_weapon_upgrade import RESOURCE_WEAPON_UPGRADES

# The only classroom the three laws are legal in: the loot economy's own town
# itinerary.  sweep-v1 only DROPS loot, cain-v1 and smith-v1 only spend the
# trip's gold, so all three are defined inside sustain-loot-v1 under l2-town-v1.
LOOT = {"resource_protocol": "l2-town-v1",
        "resource_purchase_mode": "full",
        "resource_service_policy": "sustain-loot-v1",
        "resource_readiness_law": "coach-v03",
        "resource_retreat": "retreat-v1"}
# The whole B6 world, as tonight's smoke launches it.
B6 = dict(LOOT, resource_sweep="sweep-v1", resource_identify="cain-v1",
          resource_weapon_upgrade="smith-v1")
# (key, on-value, deployment vocabulary) for the three laws, in flag order.
LAWS = (("resource_sweep", "sweep-v1", RESOURCE_SWEEP_PROTOCOLS),
        ("resource_identify", "cain-v1", RESOURCE_IDENTIFY_PROTOCOLS),
        ("resource_weapon_upgrade", "smith-v1", RESOURCE_WEAPON_UPGRADES))
LAW_KEYS = tuple(key for key, _value, _vocabulary in LAWS)
# sustain-loot-v1 exists only under a completion-l2 clock, and that clock exists
# only under an earned-dive-suffix-v1 WORKER (train_ppo._validate_worker_time_config).
# So the three laws are worker-only in training today even though their own gate
# says --worker/--options, exactly as retreat/portal's does.  This is the whole
# legal make_env classroom, prefix recipe included.
PARENT = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
CLOCK = "completion-l2-r18c"


def loot_make_env_kwargs(**overrides):
    import hashlib
    kwargs = dict(LOOT, worker=True, worker_time_protocol=CLOCK,
                  worker_learning_window_scope="earned-dive-suffix-v1",
                  worker_prefix_model=str(PARENT),
                  worker_prefix_sha256=hashlib.sha256(
                      PARENT.read_bytes()).hexdigest(),
                  worker_prefix_max_attempts=1000,
                  worker_prefix_max_microsteps=100_000_000,
                  max_steps=6000, farm_scene_cap=3600)
    kwargs.update(overrides)
    return kwargs


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
    """(a)/(b) the worker constructor forwards each law instead of dropping it."""

    def test_each_law_reaches_the_options_constructor_in_the_loot_classroom(self):
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                env, options = build_worker(**dict(LOOT, **{key: value}))
                self.assertEqual(getattr(env, key), value)
                options.assert_called_once()
                self.assertEqual(env.oe.kwargs[key], value)
                # The itinerary the leg rides on always travels with it.
                self.assertEqual(env.oe.kwargs["resource_service_policy"],
                                 "sustain-loot-v1")
                self.assertEqual(env.oe.kwargs["resource_protocol"], "l2-town-v1")

    def test_the_whole_b6_world_reaches_the_constructor_at_once(self):
        env, _ = build_worker(**B6)
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                self.assertEqual(env.oe.kwargs[key], value)

    def test_the_default_and_explicit_off_add_no_keyword_at_all(self):
        for key, _value, _vocabulary in LAWS:
            for overrides in ({}, {key: "off"}):
                with self.subTest(key=key, overrides=overrides):
                    env, _ = build_worker(**overrides)
                    self.assertEqual(getattr(env, key), "off")
                    self.assertNotIn(key, env.oe.kwargs)

    def test_the_loot_classroom_alone_produces_no_new_keyword(self):
        env, _ = build_worker(**LOOT)
        for key, _value, _vocabulary in LAWS:
            with self.subTest(key=key):
                self.assertEqual(getattr(env, key), "off")
                self.assertNotIn(key, env.oe.kwargs)

    def test_each_law_fails_closed_outside_the_loot_itinerary(self):
        for overrides, pattern in (
                # no resource protocol at all
                ({"resource_sweep": "sweep-v1"}, "sweep-v1 requires l2-town-v1"),
                ({"resource_identify": "cain-v1"}, "identify-v1 requires l2-town-v1"),
                ({"resource_weapon_upgrade": "smith-v1"},
                 "smith-v1 requires l2-town-v1"),
                # the protocol, but the legacy service policy (no loot economy)
                ({"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                  "resource_sweep": "sweep-v1"}, "sweep-v1 requires sustain-loot-v1"),
                ({"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                  "resource_identify": "cain-v1"},
                 "identify-v1 requires the sustain-loot-v1 town trip"),
                ({"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                  "resource_weapon_upgrade": "smith-v1"},
                 "smith-v1 requires the sustain-loot-v1 town itinerary"),
                # sustain-v6 is a first-class service law and still not the one
                # these three legs are defined inside.
                ({"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                  "resource_service_policy": "sustain-v6",
                  "resource_sweep": "sweep-v1"}, "sweep-v1 requires sustain-loot-v1"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, pattern):
                    build_worker(**overrides)

    def test_unknown_values_are_rejected_before_the_constructor(self):
        for key, _value, _vocabulary in LAWS:
            for bad in ("v1", "on", "", None, True):
                with self.subTest(key=key, bad=bad):
                    with self.assertRaises(ValueError):
                        build_worker(**dict(LOOT, **{key: bad}))

    def test_the_worker_reuses_the_deployment_validators_verbatim(self):
        # The training world must equal the tested world, so worker_env must not
        # restate the law: it calls the very functions OptionsEnv calls.
        text = (ROOT / "python/diablogym/worker_env.py").read_text(encoding="utf-8")
        for name in ("validate_sweep_protocol", "validate_identify_protocol",
                     "validate_weapon_upgrade"):
            with self.subTest(name=name):
                self.assertIn(name, text)
        options_text = (ROOT / "python/diablogym/options_env.py").read_text(
            encoding="utf-8")
        for name in ("validate_sweep_protocol", "validate_identify_protocol",
                     "validate_weapon_upgrade"):
            with self.subTest(name=name, side="deployment"):
                self.assertIn(name, options_text)

    def test_the_protocol_tables_are_versioned(self):
        self.assertEqual(RESOURCE_SWEEP_PROTOCOLS, ("off", "sweep-v1"))
        self.assertEqual(RESOURCE_IDENTIFY_PROTOCOLS, ("off", "cain-v1"))
        self.assertEqual(RESOURCE_WEAPON_UPGRADES, ("off", "dry-v1", "smith-v1"))


class NativeConfigurePathTests(unittest.TestCase):
    """K2b: the training env must call the SAME configure path the deployment
    env does.  It does, because there is only one -- the weapon scope is written
    by DiabloGymEnv._configure_native_resource_protocol, which every env reaches
    through the kwarg the worker now forwards."""

    @staticmethod
    def _code(relative):
        """The file with every comment line removed -- prose about a law is not
        a second implementation of it."""
        return "\n".join(
            line for line in (ROOT / relative).read_text(
                encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#"))

    def test_the_weapon_scope_has_exactly_one_native_writer(self):
        callers = []
        for relative in sorted(
                path.relative_to(ROOT).as_posix()
                for path in (ROOT / "python/diablogym").glob("*.py")):
            if "configure_native_weapon_purchase(" in self._code(relative):
                callers.append(relative)
        self.assertEqual(callers, ["python/diablogym/env.py",
                                   "python/diablogym/resource_weapon_upgrade.py"])
        env_code = self._code("python/diablogym/env.py")
        # env.py: the definition site is elsewhere, so exactly one CALL, and it
        # sits in the one configure path every env goes through.
        self.assertEqual(env_code.count("configure_native_weapon_purchase("), 1)
        self.assertIn("_configure_native_resource_protocol", env_code)

    def test_the_worker_never_reimplements_the_native_handshake(self):
        code = self._code("python/diablogym/worker_env.py")
        self.assertNotIn("configure_resource_weapon_purchase", code)
        self.assertNotIn("configure_native_weapon_purchase", code)
        # It forwards the kwarg instead, which is what reaches that one path.
        self.assertIn('env_kwargs["resource_weapon_upgrade"]', code)


def escrow_self(pending=7.5, fraction=0.5, depth=1, **overrides):
    """Synthetic ``self`` for the unbound settlement call (the B5 fixture with
    the B6 flags added; no portal service, so the portal clause is inert)."""
    fields = dict(
        descend_escrow_fraction=fraction,
        descend_escrow_power=1.0,
        descend_escrow_readiness_gate=False,
        resource_retreat="retreat-v1",
        resource_portal="off",
        resource_protocol="l2-town-v1",
        resource_readiness_law="coach-v03",
        resource_sweep="sweep-v1",
        resource_identify="cain-v1",
        resource_weapon_upgrade="smith-v1",
        _descend_escrow=float(pending),
        stats={},
        oe=SimpleNamespace(
            portal_service=None,
            env=SimpleNamespace(_econ_episode_max_depth=depth,
                                reward_economy=None, _steps=1000, _raw={})))
    fields.update(overrides)
    fake = SimpleNamespace(**fields)
    fake._portal_close_is_death_equivalent = (
        lambda: worker_env.WorkerWindowEnv._portal_close_is_death_equivalent(fake))
    return fake


def settle(fake, d_before, reason):
    return worker_env.WorkerWindowEnv._descend_escrow_settlement(
        fake, d_before, reason)


class SweepCloseReasonEscrowTests(unittest.TestCase):
    """The escrow ruling for the ONLY new close reasons these three laws add.

    sweep-v1 introduces ``sweep_trigger`` and ``sweep_complete``
    (options_env.py:1452/1462).  cain-v1 and smith-v1 introduce NONE: they are
    legs inside the RESUPPLY town trip, whose window still closes by the old law.
    """

    def test_the_sweep_reasons_vest_exactly_like_any_ordinary_close(self):
        # R18-B6 (2026-09-07) review round: only ONE of these two reasons can
        # ever reach the settlement.  "sweep_window" is set only when the option
        # is RESUPPLY (options_env.py:1267-1269), and WorkerWindowEnv
        # fast-forwards every RESUPPLY window, while the settlement is invoked
        # only at a LIVE window close -- so "sweep_complete" is structurally a
        # fast-forward-only reason (the 4096-step smoke: 14 in reasons, 14 in
        # ff_reasons, 0 live) and its subtest below pins a state the env cannot
        # produce.  It is kept deliberately: the ruling is that BOTH reasons
        # vest, and if a later pass ever makes a sweep window live, the branch
        # it lands in is already pinned.  "sweep_trigger" fires on a live
        # FARM/DIVE window (3 live in the same smoke) and is the real one.
        for reason, live in (("sweep_trigger", True), ("sweep_complete", False)):
            with self.subTest(reason=reason, reachable_live=live):
                fake = escrow_self()
                self.assertEqual(settle(fake, 3, reason), 7.5)
                self.assertEqual(fake.stats["descend_escrow_vested"], 7.5)
                self.assertNotIn("descend_escrow_forfeited", fake.stats)

    def test_sweep_complete_is_a_resupply_only_reason(self):
        """The structural half of the note above, pinned rather than asserted in
        prose: the sweep_window flag that gates "sweep_complete" is written only
        for a RESUPPLY option, and RESUPPLY windows never close live."""
        source = (ROOT / "python/diablogym/options_env.py").read_text(
            encoding="utf-8")
        block = source.split('self._win["sweep_window"] = bool(', 1)[1]
        block = block.split(")", 1)[0]
        self.assertIn("option == RESUPPLY", block)
        # and the settlement is only called from the live close path.
        worker_source = (ROOT / "python/diablogym/worker_env.py").read_text(
            encoding="utf-8")
        self.assertEqual(worker_source.count("self._descend_escrow_settlement("), 1)

    def test_the_two_forfeiting_reasons_are_untouched_by_the_new_flags(self):
        for reason in ("death", "retreat_trigger"):
            with self.subTest(reason=reason):
                fake = escrow_self()
                self.assertEqual(settle(fake, 3, reason), 0.0)
                self.assertEqual(fake.stats["descend_escrow_forfeited"], 7.5)
                self.assertNotIn("descend_escrow_vested", fake.stats)

    def test_the_settlement_never_reads_the_three_new_flags(self):
        # A sweep close is an ordinary close whether the sweep is on or off, so
        # the flag itself must not appear in the forfeit predicate.
        source = (ROOT / "python/diablogym/worker_env.py").read_text(encoding="utf-8")
        body = source.split("def _descend_escrow_settlement", 1)[1]
        body = body.split("def ", 1)[0]
        code = "\n".join(line for line in body.splitlines()
                         if not line.lstrip().startswith("#"))
        for key in LAW_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(key, code)
        for reason in ("sweep_trigger", "sweep_complete"):
            with self.subTest(reason=reason):
                self.assertNotIn(reason, code)

    def test_the_sweep_and_the_forfeiting_laws_cannot_share_a_beat(self):
        """The decision is provable, not merely argued: the sweep is main-L1
        only and both forfeiting closes are main-L2+ only, so the two sets are
        disjoint by depth and no sweep close can steal a dangerous one."""
        sweep_source = (ROOT / "python/diablogym/resource_sweep.py").read_text(
            encoding="utf-8")
        self.assertIn('dungeon_level', sweep_source)
        from diablogym.resource_retreat import RetreatService
        retreat = RetreatService()
        deep = {"dungeon_level": 5, "hp": 10, "max_hp": 100, "monsters": [],
                "belt_heal_kinds": [], "player_x": 50, "player_y": 50,
                "resource_state": {"retreats_started": 0}}
        self.assertIsNotNone(retreat.trigger_reason(deep, 10_000))
        for level in (0, 1):
            with self.subTest(dungeon_level=level):
                self.assertIsNone(retreat.trigger_reason(
                    dict(deep, dungeon_level=level), 10_000))
        # ... and the portal forfeit clause states the same depth gate.
        fake = escrow_self(resource_portal="portal-v1")
        fake.oe.env._raw = {"dungeon_level": 1, "hp": 10, "max_hp": 100}
        self.assertFalse(fake._portal_close_is_death_equivalent())

    def test_a_sweep_service_can_only_ever_fire_on_main_l1(self):
        from diablogym import resource_sweep as sweep_module
        service = sweep_module.SweepService()
        raw = {"dungeon_level": 1, "hp": 100, "max_hp": 100, "monsters": [],
               "objects": [], "player_x": 50, "player_y": 50, "player_mode": 0,
               "resource_state": {}}
        facts = dict(farm_scene_steps=10_000, cleared=True, farm_trigger=True,
                     town_trip_active=False, loot_trip_slots_left=1)
        for level in (0, 2, 5):
            with self.subTest(dungeon_level=level):
                self.assertIsNone(service.trigger_reason(
                    dict(raw, dungeon_level=level), 100, **facts))
        # ...and it refuses to START anywhere else either.
        self.assertIn("dungeon_level", (ROOT / "python/diablogym/resource_sweep.py")
                      .read_text(encoding="utf-8"))


class EvalContractTests(unittest.TestCase):
    """(f) the eval-side identity keys."""

    def test_the_defaults_keep_every_existing_archive_byte_identical(self):
        for key in LAW_KEYS:
            with self.subTest(key=key):
                self.assertEqual(contract.R16_ENVIRONMENT_DEFAULTS[key], "off")
        self.assertNotIn("r16_environment", contract.make_protocol([2114000]))

    def test_the_validator_accepts_each_law_only_inside_the_loot_itinerary(self):
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                for invalid in ({key: "off"},
                                {key: value},
                                {key: value, "resource_protocol": "l2-town-v1"},
                                {key: value,
                                 "resource_service_policy": "sustain-loot-v1"},
                                {key: value, "resource_protocol": "l2-town-v1",
                                 "resource_service_policy": "sustain-v6"},
                                {key: "v9", "resource_protocol": "l2-town-v1",
                                 "resource_service_policy": "sustain-loot-v1"}):
                    with self.subTest(invalid=invalid):
                        with self.assertRaises(
                                (ValueError, contract.EvalContractError)):
                            contract.validate_r16_environment(invalid)

    def test_dry_v1_is_not_an_archive_identity_either(self):
        with self.assertRaises((ValueError, contract.EvalContractError)):
            contract.validate_r16_environment(
                {"resource_weapon_upgrade": "dry-v1",
                 "resource_protocol": "l2-town-v1",
                 "resource_service_policy": "sustain-loot-v1"})

    def test_the_loot_archive_gate_is_the_outer_fail_closed_ring(self):
        # R18-B3: the archive schema still cannot name the completion clock, so
        # NO sustain-loot-v1 archive may be minted at all -- and therefore no
        # sweep/identify/weapon archive either.  The refusal is deliberate and
        # is pinned here so a later widening cannot pass unnoticed.
        with self.assertRaises((ValueError, contract.EvalContractError)):
            contract.validate_r16_environment(
                {"resource_protocol": "l2-town-v1",
                 "resource_service_policy": "sustain-loot-v1"})


class FingerprintTests(unittest.TestCase):
    """Every law module that exists in the tree is bound by both fingerprints.

    R18-B5 flagged the gap; R18-B6 closes it.  A law module that is not hashed
    can change under a resume (or between minting and evaluating an archive)
    while the digest still matches."""

    def _modules(self):
        return sorted(path.relative_to(ROOT).as_posix()
                      for path in (ROOT / "python/diablogym").glob("*.py"))

    def test_every_python_law_module_is_in_both_bundles(self):
        modules = self._modules()
        self.assertIn("python/diablogym/resource_sweep.py", modules)
        for relative in modules:
            with self.subTest(relative=relative):
                self.assertIn(relative, training._IMPLEMENTATION_SOURCE_FILES)
                self.assertIn(relative, contract.PROTOCOL_SOURCE_FILES)

    def test_neither_bundle_names_a_file_that_does_not_exist(self):
        for relative in (*training._IMPLEMENTATION_SOURCE_FILES,
                         *contract.PROTOCOL_SOURCE_FILES):
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_boss_avoidance_is_absent_because_r18_j_is_not_merged(self):
        self.assertFalse((ROOT / "python/diablogym/boss_avoidance.py").exists())
        for bundle in (training._IMPLEMENTATION_SOURCE_FILES,
                       contract.PROTOCOL_SOURCE_FILES):
            self.assertNotIn("python/diablogym/boss_avoidance.py", bundle)

    def test_the_bundles_have_no_duplicate_entries(self):
        for name, bundle in (("implementation", training._IMPLEMENTATION_SOURCE_FILES),
                             ("protocol", contract.PROTOCOL_SOURCE_FILES)):
            with self.subTest(name=name):
                self.assertEqual(len(bundle), len(set(bundle)))


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


class TrainingContractTests(unittest.TestCase):
    """(a)/(b) the training-side contract keys."""

    def test_the_contract_writes_none_on_the_defaults_and_the_literal_when_on(self):
        missing = training_identity()
        for key in LAW_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, missing)
                self.assertIsNone(missing[key])
        self.assertEqual(missing, training_identity(
            **{key: "off" for key in LAW_KEYS}))
        # An old checkpoint that predates all three keys resumes without drift.
        old = dict(missing)
        for key in LAW_KEYS:
            del old[key]
        training._validate_resume_contract(old, missing)
        current = training_identity(**{key: value for key, value, _v in LAWS})
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                self.assertEqual(current[key], value)

    def test_the_default_contract_gains_no_non_none_value(self):
        missing = training_identity()
        self.assertEqual({key: missing[key] for key in LAW_KEYS},
                         {key: None for key in LAW_KEYS})

    def test_each_change_is_drift_that_only_the_named_restart_may_ride(self):
        for key in LAW_KEYS:
            self.assertIn(key, training._ENVIRONMENT_RESTART_ALLOWED_DRIFT)
        # The legacy-v1 service policy keeps _validate_resource_resume_identity
        # out of the way, so the general contract-equality gate is the one under
        # test here (its message names the key).  The loot-classroom version of
        # the same drift is covered by the resource-identity test below.
        saved = training_identity()
        for key, value, _vocabulary in LAWS:
            current = training_identity(**{key: value})
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    training._validate_resume_contract(saved, current)
                training._validate_resume_contract(saved, current,
                                                   allow_environment_restart=True)

    def test_the_resource_resume_identity_keys_cover_all_three_laws(self):
        base = {"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                "resource_service_policy": "sustain-v6",
                "resource_service_recipe": training.resource_service_recipe(
                    "l2-town-v1", "full", "sustain-v6")}
        training._validate_resource_resume_identity(dict(base), dict(base))
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                        ValueError, "separately identified initialization"):
                    training._validate_resource_resume_identity(
                        dict(base), dict(base, **{key: value}))


class TrainingCliTests(unittest.TestCase):
    """(e) argparse choices and (c) the fail-closed combinations."""

    def test_the_cli_defaults_and_choices(self):
        args = parsed_args(self)
        for key, value, _vocabulary in LAWS:
            flag = "--" + key.replace("_", "-")
            with self.subTest(key=key):
                self.assertEqual(getattr(args, key), "off")
                self.assertEqual(getattr(parsed_args(self, flag, value), key), value)
        for flags in (("--resource-sweep", "sweep-v2"),
                      ("--resource-sweep", "on"),
                      ("--resource-identify", "cain-v2"),
                      ("--resource-identify", "identify-v1"),
                      ("--resource-weapon-upgrade", "smith-v2"),
                      # dry-v1 exists on the deployment side and is deliberately
                      # NOT a training arm: bit-identical rows, lying identity.
                      ("--resource-weapon-upgrade", "dry-v1")):
            with self.subTest(flags=flags):
                with patch.object(sys, "argv", ["train_ppo.py", *flags]):
                    with self.assertRaises(SystemExit):
                        training._main(SimpleNamespace())

    def _loot_args(self, **laws):
        """The REAL B3b/M2 launch argv, parsed by train_ppo's own argparse."""
        from test_r18b3b_loot_warm_start import LiveTrainingContractIdentityTests
        argv = list(LiveTrainingContractIdentityTests._launch_argv(
            CLOCK, "v2", "l1-only"))
        for key, value in laws.items():
            if value != "off":
                argv += ["--" + key.replace("_", "-"), value]
        return LiveTrainingContractIdentityTests._parse(argv)

    def test_validate_args_pins_each_law_to_the_loot_itinerary(self):
        for key, value, _vocabulary in LAWS:
            flag = "--" + key.replace("_", "-")
            with self.subTest(key=key):
                args = self._loot_args(**{key: value})
                training._validate_args(args)      # the legal classroom passes
                args.resource_service_policy = "sustain-v6"
                with self.assertRaisesRegex(ValueError, flag):
                    training._validate_args(args)
                # Removing the protocol trips an OUTER ring first (the loot
                # itinerary is guarded by a chain); the point is that the world
                # is refused, never silently rebuilt without the law.
                args = self._loot_args(**{key: value})
                args.resource_protocol = "off"
                with self.assertRaisesRegex(
                        ValueError, "l2-town-v1|resource-"):
                    training._validate_args(args)

    def test_validate_args_accepts_the_whole_b6_world(self):
        args = self._loot_args(**{key: value for key, value, _v in LAWS})
        training._validate_args(args)
        for key, value, _vocabulary in LAWS:
            self.assertEqual(getattr(args, key), value)

    def test_the_flat_mode_can_never_reach_a_law_and_names_the_ring_that_fires(self):
        """R18-B6 (2026-09-07) review round.

        The three ``--resource-* requires --worker/--options`` clauses B6 adds to
        ``_validate_args`` are DELIBERATELY UNREACHABLE defence in depth, exactly
        like retreat's and portal's: the general
        ``--resource-protocol requires --worker/--options`` clause sits ABOVE
        them and every law already requires ``l2-town-v1``.  In the real launch
        argv an even earlier ring fires -- the completion-l2 clock demands an
        earned-dive-suffix WORKER.  The earlier draft of this test used an
        alternation regex that accepted either message, which hid the fact that
        the B6 clause is never the one that speaks.  Pin the message that ACTUALLY
        fires, anchored, and pin the unreachable clause as source instead.
        """
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                args = self._loot_args(**{key: value})
                args.worker = args.options = False
                with self.assertRaisesRegex(
                        ValueError,
                        r"^completion-l2 protocols require an "
                        r"earned-dive-suffix-v1 Worker$"):
                    training._validate_args(args)

    def test_the_unreachable_worker_gates_are_still_written(self):
        """Defence in depth is only defence while it is present in the source."""
        source = (ROOT / "train/train_ppo.py").read_text(encoding="utf-8")
        body = source.split("def _validate_args", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"--resource-protocol requires --worker/--options"', body)
        for key in LAW_KEYS:
            flag = "--" + key.replace("_", "-")
            with self.subTest(key=key):
                self.assertIn(f'"{flag} requires --worker/--options"', body)
        self.assertIn("--resource-weapon-upgrade smith-v1 requires "
                      "--resource-purchase-mode full", body)

    def test_smith_v1_can_never_run_outside_the_full_purchase_mode(self):
        """R18-B6 (2026-09-07) review round: the B6 clause here is unreachable.

        ``resource_sustain.validate_service_policy`` (called at the very top of
        ``_validate_args`` through ``validate_resource_service_config``) already
        refuses ``sustain-loot-v1`` with any mode but ``full``, and smith-v1
        requires ``sustain-loot-v1``.  So no argv can make B6's own
        ``--resource-weapon-upgrade smith-v1 requires --resource-purchase-mode
        full`` speak.  The law still holds -- assert the ring that enforces it,
        anchored, instead of an alternation that would pass either way.
        """
        args = self._loot_args(resource_weapon_upgrade="smith-v1")
        args.resource_purchase_mode = "armor"
        with self.assertRaisesRegex(
                ValueError,
                r"^completion-l2 protocols require "
                r"l2-town-v1/full/sustain-v6 or sustain-loot-v1$"):
            training._validate_args(args)
        # ... and with the clock ring neutralised, the next ring out still
        # refuses, and still never reaches the B6 clause.
        args = self._loot_args(resource_weapon_upgrade="smith-v1")
        args.resource_purchase_mode = "armor"
        args.worker_time_protocol = "legacy"
        with self.assertRaisesRegex(
                ValueError,
                r"^sustain-loot-v1 requires an explicit completion-l2 "
                r"time protocol$"):
            training._validate_args(args)


class MakeEnvTests(unittest.TestCase):
    """(b)/(c) the env factory forwards only the explicit laws."""

    def test_the_factory_forwards_each_law_and_omits_the_defaults(self):
        for key, value, _vocabulary in LAWS:
            for on in (False, True):
                config = loot_make_env_kwargs(**({key: value} if on else {}))
                with self.subTest(key=key, on=on), patch(
                        "diablogym.WorkerWindowEnv",
                        side_effect=DummyEnv) as constructor:
                    env = training.make_env(**config)
                    forwarded = constructor.call_args.kwargs
                    if on:
                        self.assertEqual(forwarded[key], value)
                    else:
                        self.assertNotIn(key, forwarded)
                    env.close()

    def test_the_factory_forwards_the_whole_b6_world(self):
        with patch("diablogym.WorkerWindowEnv",
                   side_effect=DummyEnv) as constructor:
            env = training.make_env(**loot_make_env_kwargs(
                **{key: value for key, value, _v in LAWS}))
            forwarded = constructor.call_args.kwargs
            for key, value, _vocabulary in LAWS:
                with self.subTest(key=key):
                    self.assertEqual(forwarded[key], value)
            env.close()

    def test_the_factory_fails_closed_on_every_illegal_combination(self):
        for kwargs, pattern in (
                # no resource protocol / no loot policy at all
                ({"worker": True, "resource_sweep": "sweep-v1"},
                 "resource_sweep sweep-v1 requires l2-town-v1"),
                ({"worker": True, "resource_identify": "cain-v1"},
                 "resource_identify cain-v1 requires l2-town-v1"),
                ({"worker": True, "resource_weapon_upgrade": "smith-v1"},
                 "resource_weapon_upgrade smith-v1 requires l2-town-v1"),
                # the deployment-only observation twin
                ({"worker": True, "resource_weapon_upgrade": "dry-v1"},
                 "resource_weapon_upgrade must be off/smith-v1"),
                # the protocol, but sustain-v6 rather than the loot itinerary
                ({"worker": True, "resource_protocol": "l2-town-v1",
                  "resource_purchase_mode": "full",
                  "resource_service_policy": "sustain-v6",
                  "resource_sweep": "sweep-v1"},
                 "resource_sweep sweep-v1 requires sustain-loot-v1"),
                ({"worker": True, "resource_protocol": "l2-town-v1",
                  "resource_purchase_mode": "full",
                  "resource_service_policy": "sustain-v6",
                  "resource_identify": "cain-v1"},
                 "resource_identify cain-v1 requires sustain-loot-v1"),
                ({"worker": True, "resource_protocol": "l2-town-v1",
                  "resource_purchase_mode": "full",
                  "resource_service_policy": "sustain-v6",
                  "resource_weapon_upgrade": "smith-v1"},
                 "resource_weapon_upgrade smith-v1 requires sustain-loot-v1"),
                # outside WorkerWindowEnv/OptionsEnv entirely
                ({"resource_sweep": "sweep-v1"},
                 "resource_sweep sweep-v1 requires l2-town-v1"),
        ):
            with self.subTest(kwargs=sorted(kwargs)):
                with self.assertRaisesRegex(ValueError, pattern):
                    training.make_env(**kwargs)

    # The two rings that sit between the loot fixture and B6's own worker gate.
    # Unlike _validate_args, make_env puts the GENERAL protocol gate LAST, so
    # B6's per-law gate IS reachable -- but only once the clock recipe and the
    # retreat law (whose own identical gate is written earlier) are out of the
    # way.  R18-B6 (2026-09-07) review round: the earlier draft left both in and
    # matched an alternation, so the message it actually saw was the clock's.
    _CLOCK_KEYS = ("worker_time_protocol", "worker_learning_window_scope",
                   "worker_prefix_model", "worker_prefix_sha256",
                   "worker_prefix_max_attempts", "worker_prefix_max_microsteps")

    def test_each_law_still_needs_a_worker_or_options_env(self):
        # The flat (neither worker nor options) path: B6's OWN gate must be the
        # one that fires, and its message is asserted anchored.
        for key, value, _vocabulary in LAWS:
            config = loot_make_env_kwargs(**{key: value})
            config["worker"] = False
            for stale in self._CLOCK_KEYS:
                config.pop(stale, None)
            config.pop("resource_retreat", None)
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                        ValueError,
                        rf"^{key} requires WorkerWindowEnv/OptionsEnv$"):
                    training.make_env(**config)

    def test_the_clock_and_retreat_rings_shadow_that_gate_in_the_real_fixture(self):
        """Why the test above has to strip two keys: in the launch-shaped
        fixture the completion-l2 clock speaks first, and with the clock gone
        retreat's identical (older) gate speaks before B6's."""
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                config = loot_make_env_kwargs(**{key: value})
                config["worker"] = False
                with self.assertRaisesRegex(
                        ValueError,
                        r"^completion-l2 protocols require an "
                        r"earned-dive-suffix-v1 Worker$"):
                    training.make_env(**config)
                for stale in self._CLOCK_KEYS:
                    config.pop(stale, None)
                with self.assertRaisesRegex(
                        ValueError,
                        r"^resource_retreat requires WorkerWindowEnv/OptionsEnv$"):
                    training.make_env(**config)

    def test_smith_v1_can_never_run_outside_the_full_purchase_mode(self):
        """R18-B6 (2026-09-07) review round: as in ``_validate_args``, make_env's
        own ``resource_weapon_upgrade smith-v1 requires the full purchase mode``
        is unreachable -- ``validate_service_policy`` refuses sustain-loot-v1
        outside ``full`` first, and smith-v1 requires sustain-loot-v1.  It is kept
        as defence in depth; what is asserted here is the ring that speaks."""
        config = loot_make_env_kwargs(resource_weapon_upgrade="smith-v1",
                                      resource_purchase_mode="armor")
        with self.assertRaisesRegex(
                ValueError,
                r"^completion-l2 protocols require "
                r"l2-town-v1/full/sustain-v6 or sustain-loot-v1$"):
            training.make_env(**config)
        for stale in self._CLOCK_KEYS:
            config.pop(stale, None)
        with self.assertRaisesRegex(
                ValueError,
                r"^sustain-loot-v1 requires resource_protocol l2-town-v1 "
                r"and full purchase mode$"):
            training.make_env(**config)
        # The unreachable clause is nevertheless present.
        source = (ROOT / "train/train_ppo.py").read_text(encoding="utf-8")
        self.assertIn("resource_weapon_upgrade smith-v1 requires the full "
                      "purchase mode", source)


class EarnedSuffixPinTests(unittest.TestCase):
    """The earned-dive-suffix fixed dict pins none of the three -- retreat and
    portal are not pinned there either, and R18-B6 adds no new pin."""

    def _fixed_keys(self):
        source = (ROOT / "train/train_ppo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_validate_worker_prefix_args"):
                for statement in ast.walk(node):
                    if (isinstance(statement, ast.Assign)
                            and any(getattr(target, "id", None) == "fixed"
                                    for target in statement.targets)):
                        return [key.value for key in statement.value.keys]
        self.fail("_validate_worker_prefix_args fixed dict not found")

    def test_no_new_flag_is_pinned_under_the_earned_suffix(self):
        keys = self._fixed_keys()
        self.assertIn("resource_protocol", keys)
        for key in LAW_KEYS:
            with self.subTest(key=key):
                self.assertNotIn(key, keys)


class EvalAssembledWiringTests(unittest.TestCase):
    """(f) the certification eval must not silently rebuild a no-sweep env."""

    def test_the_evaluator_forwards_each_law_and_omits_the_defaults(self):
        for key, value, _vocabulary in LAWS:
            for config in (None, {key: value, "resource_protocol": "l2-town-v1",
                                  "resource_service_policy": "sustain-loot-v1",
                                  "resource_purchase_mode": "full"}):
                constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
                with self.subTest(key=key, config=config), patch.object(
                        evaluation, "_native_runtime",
                        return_value=(Mock(), constructor, None)), patch.object(
                        evaluation, "validate_r16_environment",
                        side_effect=lambda value: dict(value)):
                    with self.assertRaisesRegex(RuntimeError, "constructor-boundary"):
                        evaluation.evaluate(None, [], r16_environment=config)
                forwarded = constructor.call_args.kwargs
                if config:
                    self.assertEqual(forwarded[key], value)
                else:
                    self.assertNotIn(key, forwarded)

    def test_the_eval_cli_declares_all_three_flags_with_the_frozen_defaults(self):
        source = (ROOT / "train/eval_assembled.py").read_text(encoding="utf-8")
        for key in LAW_KEYS:
            flag = "--" + key.replace("_", "-")
            with self.subTest(key=key):
                self.assertIn(f'ap.add_argument("{flag}", default="off"', source)
                self.assertIn(f'"{key}": args.{key},', source)


class EvalAssembledCliGateTests(unittest.TestCase):
    """Drive the parser rather than assert on source strings (the B5 review
    round's lesson): argparse exits before any environment work."""

    def _cli(self, *flags):
        stderr = io.StringIO()
        with patch.object(sys, "argv",
                          ["eval_assembled.py", "--worker", "script", *flags]):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_case:
                    evaluation.main()
        self.assertEqual(exit_case.exception.code, 2)
        return " ".join(stderr.getvalue().split())

    def test_each_law_needs_the_resource_protocol(self):
        for key, value, _vocabulary in LAWS:
            flag = "--" + key.replace("_", "-")
            with self.subTest(key=key):
                self.assertIn(f"{flag} {value} requires --resource-protocol "
                              "l2-town-v1", self._cli(flag, value))

    def test_each_law_needs_the_loot_service_policy(self):
        for key, value, _vocabulary in LAWS:
            flag = "--" + key.replace("_", "-")
            with self.subTest(key=key):
                self.assertIn(
                    f"{flag} {value} requires --resource-service-policy "
                    "sustain-loot-v1",
                    self._cli(flag, value, "--resource-protocol", "l2-town-v1"))

    def test_the_eval_cli_still_cannot_name_the_loot_service_policy(self):
        # Known and deliberate (R18-B3): eval_assembled's own
        # --resource-service-policy choices stop at sustain-v6, so the three
        # gates above are unreachable-by-construction on the CLI today.  Pinned
        # so the day that changes, this test says so.
        message = self._cli("--resource-service-policy", "sustain-loot-v1")
        self.assertIn("invalid choice", message)


class MigrationVocabularyTests(unittest.TestCase):
    """The mint must pin the world being entered (R18-M2 §5's lesson)."""

    def test_the_three_laws_are_registered_in_the_loot_warm_start_world(self):
        self.assertEqual(loot.SWEEPS, ("off", "sweep-v1"))
        self.assertEqual(loot.IDENTIFIES, ("off", "cain-v1"))
        self.assertEqual(loot.WEAPON_UPGRADES, ("off", "smith-v1"))
        for key in LAW_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, loot.WORLD_KEYS)
                self.assertIn(key, loot.ALLOWED_CONTRACT_KEYS)

    def test_the_world_writes_none_for_off_and_the_literal_when_on(self):
        off = loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1",
                                "v2", "l1-only")
        full = loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1",
                                 "v2", "l1-only", "sweep-v1", "cain-v1", "smith-v1")
        self.assertEqual(set(off), set(loot.WORLD_KEYS))
        self.assertEqual(set(full), set(loot.WORLD_KEYS))
        for key, value, _vocabulary in LAWS:
            with self.subTest(key=key):
                self.assertIsNone(off[key])
                self.assertEqual(full[key], value)
        # world_of re-derives rather than trusting the stored value.
        self.assertEqual(loot.world_of(full), full)
        self.assertEqual(loot.world_of(off), off)

    def test_an_unregistered_law_is_refused_rather_than_silently_minted(self):
        for index, law, message in ((5, "sweep-v2", "registered sweep law"),
                                    (6, "cain-v2", "registered identify law"),
                                    (7, "dry-v1", "registered weapon upgrade law")):
            positional = ["completion-l2-r18c", "coach-v03", "retreat-v1",
                          "v2", "l1-only", "off", "off", "off"]
            positional[index] = law
            with self.subTest(law=law), self.assertRaisesRegex(ValueError, message):
                loot.target_world(*positional)

    def test_the_mint_cli_offers_exactly_the_registered_vocabulary(self):
        source = (ROOT / "train/migrate_loot_candidate.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "add_argument"
                    and node.args
                    and isinstance(getattr(node.args[0], "value", None), str)):
                for keyword in node.keywords:
                    if keyword.arg == "choices":
                        found[node.args[0].value] = keyword.value
        for flag, name in (("--resource-sweep", "SWEEPS"),
                           ("--resource-identify", "IDENTIFIES"),
                           ("--resource-weapon-upgrade", "WEAPON_UPGRADES")):
            with self.subTest(flag=flag):
                self.assertIn(flag, found)
                self.assertEqual(getattr(found[flag], "id", None), name)


class LiveWarmStartIdentityTests(unittest.TestCase):
    """The load-bearing one: the minted contract IS the live training contract
    for the FULL B6 world, and both consumption gates accept it.

    This is the class R18-M2 §5.3 identifies as the one that catches a world the
    mint does not pin; it is re-run here with the three new laws on.
    """

    @classmethod
    def setUpClass(cls):
        from test_r18b3b_loot_warm_start import (
            IMPL, POLICY_SHA, PARENT, _prefix)
        import io as _io
        from leashed_ppo import LeashedMaskablePPO
        if not PARENT.is_file():
            raise unittest.SkipTest("registered R16 parent checkpoint unavailable")
        payload = PARENT.read_bytes()
        from train_ppo import (_validate_checkpoint_bytes,
                               _validate_resumable_leashed_boundary)
        cls.impl, cls.policy_sha = IMPL, POLICY_SHA
        cls.parent_data = _validate_resumable_leashed_boundary(
            _validate_checkpoint_bytes(payload, str(PARENT), require_leashed=True))
        cls.model = LeashedMaskablePPO.load(_io.BytesIO(payload), device="cpu",
                                            teacher_path=None, teacher_sha256=None)
        cls.source = cls.parent_data["diablogym_contract"]
        cls.prefix = _prefix()
        cls.PARENT = PARENT

    def _argv(self, **laws):
        from test_r18b3b_loot_warm_start import LiveTrainingContractIdentityTests
        argv = list(LiveTrainingContractIdentityTests._launch_argv(
            "completion-l2-r18c", "v2", "l1-only"))
        for key, value in laws.items():
            if value != "off":
                argv += ["--" + key.replace("_", "-"), value]
        return argv

    def _parse(self, argv):
        from test_r18b3b_loot_warm_start import LiveTrainingContractIdentityTests
        return LiveTrainingContractIdentityTests._parse(argv)

    def test_the_full_b6_world_mints_and_consumes(self):
        for laws in ({key: "off" for key in LAW_KEYS},
                     {key: value for key, value, _v in LAWS}):
            with self.subTest(laws=laws):
                args = self._parse(self._argv(**laws))
                for key, value in laws.items():
                    self.assertEqual(getattr(args, key), value)
                batch = training._select_batch_size(args.n_steps, args.num_envs)
                current = training._training_contract(
                    args, self.model, batch, implementation_sha256=self.impl)
                target = loot.target_contract(
                    self.source, self.impl, self.prefix,
                    worker_time_protocol="completion-l2-r18c",
                    resource_readiness_law="coach-v03",
                    resource_retreat="retreat-v1",
                    worker_descend_escrow_readiness_table="v2",
                    hunt_scope="l1-only", **laws)
                self.assertEqual(set(target), set(current))
                self.assertEqual(target, current)
                self.assertEqual(loot.json_sha256(target),
                                 loot.json_sha256(current))
                for key, value in laws.items():
                    self.assertEqual(current[key],
                                     None if value == "off" else value)
                # Both consumption gates train_ppo runs on a warm start.
                training._validate_resume_contract(target, current)
                receipt = loot.make_receipt(self.parent_data, target,
                                            self.policy_sha)
                loot.validate_inherited_receipt(receipt, current)

    def test_a_receipt_minted_without_the_laws_refuses_the_world_with_them(self):
        """Fail closed, not silent corruption: an off-world receipt cannot be
        used to enter the sweep world."""
        off = {key: "off" for key in LAW_KEYS}
        on = {key: value for key, value, _v in LAWS}
        stale_target = loot.target_contract(
            self.source, self.impl, self.prefix,
            worker_time_protocol="completion-l2-r18c",
            worker_descend_escrow_readiness_table="v2",
            hunt_scope="l1-only", **off)
        stale_receipt = loot.make_receipt(self.parent_data, stale_target,
                                          self.policy_sha)
        live_target = loot.target_contract(
            self.source, self.impl, self.prefix,
            worker_time_protocol="completion-l2-r18c",
            worker_descend_escrow_readiness_table="v2",
            hunt_scope="l1-only", **on)
        with self.assertRaisesRegex(ValueError, "receipt|world|drift"):
            loot.validate_inherited_receipt(stale_receipt, live_target)

    def test_a_receipt_that_predates_the_three_keys_refuses_this_world(self):
        """R18-B6 (2026-09-07) review round: the KEY SET, not just the values.

        The test above varies what the three laws SAY; a receipt minted before
        this pass does not mention them at all -- its ``target_world`` carries
        the six pre-B6 keys where this tree writes nine (the real artifact is
        ``/home/laure/r17_work/r18/m2-smoke-4096/candidate/manifest.json``).
        ``validate_inherited_receipt`` compares ``target_world`` by equality
        against a freshly re-derived world, so it must refuse.  If a later pass
        ever makes ``world_of``/``target_world`` tolerant of the missing keys
        (a ``setdefault`` on the receipt side, the way ``resource_portal`` is
        handled), a six-key receipt would silently enter the B6 world under a
        receipt that never named the three laws -- and this is the test that
        fails first.
        """
        from copy import deepcopy
        off = {key: "off" for key in LAW_KEYS}
        target = loot.target_contract(
            self.source, self.impl, self.prefix,
            worker_time_protocol="completion-l2-r18c",
            worker_descend_escrow_readiness_table="v2",
            hunt_scope="l1-only", **off)
        receipt = loot.make_receipt(self.parent_data, target, self.policy_sha)
        # The receipt is good as minted, against the world it names.
        loot.validate_inherited_receipt(receipt, target)
        self.assertEqual(len(receipt["target_world"]), 9)
        pre_b6 = deepcopy(receipt)
        for key in LAW_KEYS:
            pre_b6["target_world"].pop(key)
        self.assertEqual(len(pre_b6["target_world"]), 6)
        with self.assertRaisesRegex(
                ValueError, "does not pin the world being entered"):
            loot.validate_inherited_receipt(pre_b6, target)
        # ... and one key short is refused just as hard as three.
        for key in LAW_KEYS:
            with self.subTest(missing=key):
                short = deepcopy(receipt)
                short["target_world"].pop(key)
                with self.assertRaisesRegex(
                        ValueError, "does not pin the world being entered"):
                    loot.validate_inherited_receipt(short, target)

    def test_the_measured_target_contract_sha_moved_for_every_schema2_arm(self):
        """R18-B6 (2026-09-07) review round: registering three keys moves
        ``target_contract_sha256`` for EVERY schema/2 arm, the all-off one
        included, because ``json_sha256`` is key-presence sensitive.  The B6
        report used to claim the frozen-default arm was unchanged; it is not."""
        off = {key: "off" for key in LAW_KEYS}
        target = loot.target_contract(
            self.source, self.impl, self.prefix,
            worker_time_protocol="completion-l2-r18c",
            worker_descend_escrow_readiness_table="v2",
            hunt_scope="l1-only", **off)
        for key in LAW_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, target)      # written, and written as None
                self.assertIsNone(target[key])
        from copy import deepcopy
        pre_b6 = deepcopy(target)
        for key in LAW_KEYS:
            pre_b6.pop(key)
        self.assertNotEqual(loot.json_sha256(target), loot.json_sha256(pre_b6))


if __name__ == "__main__":
    unittest.main()
