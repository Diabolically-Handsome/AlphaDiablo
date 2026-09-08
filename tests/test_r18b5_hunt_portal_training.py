"""R18-B5: portal-v1 and hunt_scope wired into the training window.

Two deployment interfaces that already existed on the engine side but had no
training wiring: ``resource_portal`` (the Scroll of Town Portal vehicle for the
retreat) and ``hunt_scope`` (the a10 global-hunt gate per floor).  The training
world must equal the tested world, so both reach WorkerWindowEnv/OptionsEnv,
the training contract, the CLI and the eval identity -- and both stay
byte-identical on their defaults (``off`` / ``all``).

Constructor and pure-function boundaries only: no engine reset, no native step,
no optimizer update, no checkpoint.  WorkerWindowEnv is exercised with a stubbed
OptionsEnv (the ``test_resource_retreat_training.py`` pattern) and the escrow
settlement is called unbound against a synthetic ``self``.
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
import train_ppo as training
from diablogym import env as env_module
from diablogym import options_env
from diablogym import worker_env
from diablogym.resource_portal import PortalService
from diablogym.resource_protocol import (
    RESOURCE_PORTAL_PROTOCOLS,
    validate_portal_protocol,
)

# The only classroom retreat-v1 is legal in (R18-B).
RETREAT = {"resource_protocol": "l2-town-v1",
           "resource_readiness_law": "coach-v03",
           "resource_retreat": "retreat-v1"}
# The portal is the retreat vehicle, so its classroom is the retreat classroom
# plus retreat-v1 itself -- never beside it.
PORTAL = dict(RETREAT, resource_portal="portal-v1")
HUNT = {"hunt_scope": "l1-only"}
# Review round: l1-only only scopes the a10 global hunt, so the training and
# archive identities may only carry it beside the hunt it scopes.
HUNT_ON = dict(HUNT, explore_global_hunt=True)


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


class WorkerPortalPassthroughTests(unittest.TestCase):
    """(a)/(b) the worker constructor forwards portal-v1 instead of dropping it."""

    def test_portal_v1_reaches_the_options_constructor_in_the_legal_classroom(self):
        env, options = build_worker(**PORTAL, resource_purchase_mode="full")
        self.assertEqual(env.resource_portal, "portal-v1")
        options.assert_called_once()
        self.assertEqual(env.oe.kwargs["resource_portal"], "portal-v1")
        # The vehicle never travels without the law it is a vehicle for.
        self.assertEqual(env.oe.kwargs["resource_retreat"], "retreat-v1")
        self.assertEqual(env.oe.kwargs["resource_readiness_law"], "coach-v03")

    def test_the_default_and_explicit_off_add_no_keyword_at_all(self):
        for overrides in ({}, {"resource_portal": "off"}):
            with self.subTest(overrides=overrides):
                env, _ = build_worker(**overrides)
                self.assertEqual(env.resource_portal, "off")
                self.assertNotIn("resource_portal", env.oe.kwargs)

    def test_retreat_only_still_produces_no_portal_keyword(self):
        env, _ = build_worker(**RETREAT, resource_purchase_mode="full")
        self.assertEqual(env.resource_portal, "off")
        self.assertNotIn("resource_portal", env.oe.kwargs)
        self.assertEqual(env.oe.kwargs["resource_retreat"], "retreat-v1")

    def test_portal_v1_fails_closed_outside_its_classroom(self):
        for overrides, pattern in (
                # (c) portal without retreat -- the cash-out channel the
                # retreat clause closes must not reopen through the portal.
                (dict(RETREAT, resource_purchase_mode="full",
                      resource_retreat="off", resource_portal="portal-v1"),
                 "portal-v1 requires retreat-v1"),
                # (c) portal under veto-v1: the readiness law is validated
                # first, so the older guard fires.
                ({"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                  "resource_retreat": "retreat-v1", "resource_portal": "portal-v1"},
                 "retreat-v1 requires l2-town-v1 under coach-v03"),
                # (c) portal with the resource protocol off.
                ({"resource_portal": "portal-v1"},
                 "portal-v1 requires l2-town-v1 under coach-v03"),
                ({"resource_readiness_law": "coach-v03",
                  "resource_portal": "portal-v1"},
                 "coach-v03 requires l2-town-v1")):
            with self.subTest(overrides=overrides):
                with patch.object(worker_env, "OptionsEnv") as options:
                    with self.assertRaisesRegex(ValueError, pattern):
                        build_worker(**overrides)
                options.assert_not_called()

    def test_unknown_portal_values_are_rejected_before_the_constructor(self):
        for value in ("portal-v2", "on", True):
            with self.subTest(value=value):
                overrides = dict(PORTAL, resource_purchase_mode="full",
                                 resource_portal=value)
                with patch.object(worker_env, "OptionsEnv") as options:
                    with self.assertRaisesRegex(ValueError, "Unknown resource_portal"):
                        build_worker(**overrides)
                options.assert_not_called()

    def test_the_protocol_table_is_versioned(self):
        self.assertEqual(RESOURCE_PORTAL_PROTOCOLS, ("off", "portal-v1"))
        self.assertEqual(
            validate_portal_protocol("off", "veto-v1", "off", "off"), "off")


class WorkerHuntScopePassthroughTests(unittest.TestCase):
    """(a)/(b) hunt_scope is a DiabloGymEnv switch, forwarded through env_kwargs."""

    def test_l1_only_reaches_the_options_constructor(self):
        env, options = build_worker(**HUNT)
        self.assertEqual(env.hunt_scope, "l1-only")
        options.assert_called_once()
        self.assertEqual(env.oe.kwargs["hunt_scope"], "l1-only")

    def test_the_default_and_explicit_all_add_no_keyword_at_all(self):
        for overrides in ({}, {"hunt_scope": "all"}):
            with self.subTest(overrides=overrides):
                env, _ = build_worker(**overrides)
                self.assertEqual(env.hunt_scope, "all")
                self.assertNotIn("hunt_scope", env.oe.kwargs)

    def test_l1_only_needs_no_resource_protocol(self):
        # env.py:709 validates hunt_scope on its own; the training side must
        # not over-restrict what the deployed environment already accepts.
        env, _ = build_worker(**HUNT)
        self.assertEqual(env.resource_protocol, "off")
        self.assertEqual(env.oe.kwargs["hunt_scope"], "l1-only")

    def test_unknown_hunt_scope_values_are_rejected_before_the_constructor(self):
        # (c) an invalid string must die at the worker boundary, not silently
        # ride **env_kwargs down to DiabloGymEnv after the engine is up.
        for value in ("l1", "L1-ONLY", "l2-only", "off", "", True):
            with self.subTest(value=value):
                with patch.object(worker_env, "OptionsEnv") as options:
                    with self.assertRaisesRegex(ValueError, "Unknown hunt_scope"):
                        build_worker(hunt_scope=value)
                options.assert_not_called()

    def test_the_wording_matches_the_engine_side_law_verbatim(self):
        source = (ROOT / "python/diablogym/env.py").read_text(encoding="utf-8")
        self.assertIn(
            'raise ValueError(f"Unknown hunt_scope {hunt_scope!r}; '
            'expected all or l1-only")', source)
        worker_source = (ROOT / "python/diablogym/worker_env.py").read_text(
            encoding="utf-8")
        self.assertIn(
            'raise ValueError(f"Unknown hunt_scope {hunt_scope!r}; '
            'expected all or l1-only")', worker_source)

    def test_the_deployment_side_reads_one_single_vocabulary(self):
        # Review round: the tuple itself must not be copied. env.py owns it and
        # worker_env imports that very object, so a third scope cannot reach the
        # deployed world while the training world still refuses it.
        self.assertEqual(env_module.HUNT_SCOPES, ("all", "l1-only"))
        self.assertIs(worker_env.HUNT_SCOPES, env_module.HUNT_SCOPES)
        for path in ("python/diablogym/env.py", "python/diablogym/worker_env.py"):
            with self.subTest(path=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                self.assertIn("if hunt_scope not in HUNT_SCOPES:", text)
                self.assertNotIn('hunt_scope not in ("all", "l1-only")', text)


def _hunt_scope_membership_tuples(path):
    """Every ``hunt_scope in (...)`` literal in a train-side source file."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "hunt_scope"
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.In)
                and isinstance(node.comparators[0], ast.Tuple)):
            found.append(tuple(item.value for item in node.comparators[0].elts))
    return found


def _argparse_choices(path, flag):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args
                and getattr(node.args[0], "value", None) == flag):
            for keyword in node.keywords:
                if keyword.arg == "choices":
                    return tuple(item.value for item in keyword.value.elts)
    return None


class HuntScopeVocabularyTests(unittest.TestCase):
    """Review round: the value set, not only the error wording, is pinned in
    every place that copies it -- train/ must not import diablogym (eval_contract
    is a stdlib-only contract module), so the copies are compared here instead."""

    def test_every_training_side_copy_agrees_with_the_deployment_vocabulary(self):
        scopes = env_module.HUNT_SCOPES
        found = _hunt_scope_membership_tuples("train/train_ppo.py")
        self.assertEqual(len(found), 2)   # _validate_args and make_env
        for literal in found:
            with self.subTest(literal=literal):
                self.assertEqual(literal, scopes)
        for path in ("train/train_ppo.py", "train/eval_assembled.py"):
            with self.subTest(path=path):
                self.assertEqual(_argparse_choices(path, "--hunt-scope"), scopes)

    def test_the_archive_vocabulary_can_express_every_deployed_scope(self):
        scopes = env_module.HUNT_SCOPES
        # "all" is the archive default (expressed by omission); every other
        # scope needs its own accepted literal in validate_r16_environment.
        self.assertEqual(contract.R16_ENVIRONMENT_DEFAULTS["hunt_scope"], scopes[0])
        source = (ROOT / "train/eval_contract.py").read_text(encoding="utf-8")
        for scope in scopes[1:]:
            with self.subTest(scope=scope):
                self.assertIn(f'item == "{scope}"', source)


def dungeon_raw(hp=20, max_hp=100, depth=5, **extra):
    """A main-dungeon state the portal law can read (danger by default)."""
    raw = {"dungeon_level": depth, "hp": hp, "max_hp": max_hp,
           "player_x": 50, "player_y": 50, "monsters": [],
           "belt_heal_kinds": [1, 1, 1, 1],
           "resource_state": {"portal_scrolls": [{"spell_from": 0}],
                              "portals_started": 0,
                              "readiness": {"required_belt_heals": 4,
                                            "belt_heals": 4}}}
    raw.update(extra)
    return raw


def town_raw():
    """Town, full HP, gold in hand, no scroll carried -- the shopping errand."""
    return {"dungeon_level": 0, "hp": 100, "max_hp": 100, "gold": 5000,
            "player_x": 57, "player_y": 40, "monsters": [],
            "belt_heal_kinds": [1, 1, 1, 1],
            "resource_state": {"portal_scrolls": [], "portals_started": 0,
                               "readiness": {"required_belt_heals": 4,
                                             "belt_heals": 4}}}


def standing_portal_raw():
    """Main L5 at full HP with a portal already open here (no danger at all)."""
    return dungeon_raw(hp=100, resource_state={
        "portal_scrolls": [{"spell_from": 0}], "portals_started": 0,
        "portal_open": True, "portal_level": 5,
        "readiness": {"required_belt_heals": 4, "belt_heals": 4}})


def escrow_self(resource_retreat="off", resource_portal="off",
                pending=7.5, fraction=0.5, depth=1, raw=None,
                portal_service=None):
    """Synthetic ``self`` for the unbound settlement call (forfeit path only
    touches fraction/_descend_escrow/stats/resource_retreat/resource_portal
    plus, since the review round, oe.portal_service + oe.env._raw; the vest
    path additionally reads oe.env)."""
    if portal_service is None and resource_portal != "off":
        portal_service = PortalService()
    fake = SimpleNamespace(
        descend_escrow_fraction=fraction,
        descend_escrow_power=1.0,
        descend_escrow_readiness_gate=False,
        resource_retreat=resource_retreat,
        resource_portal=resource_portal,
        resource_protocol="l2-town-v1",
        resource_readiness_law="coach-v03",
        _descend_escrow=float(pending),
        stats={},
        oe=SimpleNamespace(
            portal_service=portal_service,
            env=SimpleNamespace(
                _econ_episode_max_depth=depth, reward_economy=None,
                _steps=1000,
                _raw=dungeon_raw() if raw is None else raw)))
    # The settlement asks the real danger predicate; SimpleNamespace cannot
    # bind a method, so hand it the genuine unbound implementation.
    fake._portal_close_is_death_equivalent = (
        lambda: worker_env.WorkerWindowEnv._portal_close_is_death_equivalent(fake))
    return fake


def settle(fake, d_before, reason):
    return worker_env.WorkerWindowEnv._descend_escrow_settlement(
        fake, d_before, reason)


class PortalEscrowForfeitureTests(unittest.TestCase):
    """(d) a DANGEROUS portal_trigger is death-equivalent exactly like
    retreat_trigger; an errand one is not (review round)."""

    def test_portal_trigger_forfeits_exactly_like_death_and_retreat(self):
        for reason in ("death", "retreat_trigger", "portal_trigger"):
            with self.subTest(reason=reason):
                fake = escrow_self(resource_retreat="retreat-v1",
                                   resource_portal="portal-v1")
                self.assertEqual(settle(fake, 3, reason), 0.0)
                self.assertEqual(fake._descend_escrow, 0.0)
                self.assertEqual(fake.stats["descend_escrow_forfeited"], 7.5)
                self.assertNotIn("descend_escrow_vested", fake.stats)

    def test_portal_complete_and_every_other_reason_still_vest(self):
        for reason in ("scene", "stall", "descend", "retreat_complete",
                       "portal_complete", None):
            with self.subTest(reason=reason):
                fake = escrow_self(resource_retreat="retreat-v1",
                                   resource_portal="portal-v1")
                self.assertEqual(settle(fake, 3, reason), 7.5)
                self.assertEqual(fake.stats["descend_escrow_vested"], 7.5)
                self.assertNotIn("descend_escrow_forfeited", fake.stats)

    def test_with_the_portal_off_portal_trigger_is_an_ordinary_vesting_close(self):
        # House law: nothing about the off path may move. The reason cannot
        # actually occur when the portal is off, but the branch is explicit.
        for retreat in ("off", "retreat-v1"):
            with self.subTest(retreat=retreat):
                fake = escrow_self(resource_retreat=retreat, resource_portal="off")
                self.assertEqual(settle(fake, 3, "portal_trigger"), 7.5)
                self.assertEqual(fake.stats["descend_escrow_vested"], 7.5)
                self.assertNotIn("descend_escrow_forfeited", fake.stats)
        # And a missing attribute (older pickled envs) behaves like off.
        legacy = escrow_self(resource_retreat="retreat-v1")
        del legacy.resource_portal
        self.assertEqual(settle(legacy, 3, "portal_trigger"), 7.5)
        self.assertEqual(legacy.stats["descend_escrow_vested"], 7.5)

    def test_the_retreat_clause_is_untouched_by_the_portal_flag(self):
        # retreat_trigger keeps forfeiting on the retreat flag alone.
        fake = escrow_self(resource_retreat="retreat-v1", resource_portal="off")
        self.assertEqual(settle(fake, 3, "retreat_trigger"), 0.0)
        self.assertEqual(fake.stats["descend_escrow_forfeited"], 7.5)
        # and never forfeits when the retreat law itself is off.
        off = escrow_self(resource_retreat="off", resource_portal="portal-v1")
        self.assertEqual(settle(off, 3, "retreat_trigger"), 7.5)

    def test_the_flag_off_valve_still_returns_zero_without_touching_stats(self):
        fake = escrow_self(resource_retreat="retreat-v1",
                           resource_portal="portal-v1", fraction=0.0)
        self.assertEqual(settle(fake, 3, "portal_trigger"), 0.0)
        self.assertEqual(fake.stats, {})
        self.assertEqual(fake._descend_escrow, 7.5)


class PortalErrandCloseTests(unittest.TestCase):
    """Review round (blocking): OptionsEnv._win_term compresses EVERY non-None
    PortalService.trigger_reason into the single string "portal_trigger", and
    three of the four triggers carry no danger whatsoever.  Hazard pay may only
    be destroyed by the danger law, never by a shopping errand."""

    def _win_term(self, raw, opt=options_env.FARM, portal=None):
        """Drive the real seven-rung ladder with a real PortalService."""
        portal = PortalService() if portal is None else portal
        fake = SimpleNamespace(
            _win={"t0": 0, "opt": opt, "dlvl0": int(raw["dungeon_level"]),
                  "scene0": env_module._scene_identity(raw),
                  "resource_depth0": 0, "floor": 10**9},
            resource_protocol="l2-town-v1",
            portal_service=portal,
            retreat_service=None,
            env=SimpleNamespace(_raw=raw, _steps=1000,
                                _resource_max_main_depth=0))
        return options_env.OptionsEnv._win_term(fake, False, False, 0)

    def test_the_town_shopping_errand_really_does_close_with_portal_trigger(self):
        raw = town_raw()
        self.assertEqual(PortalService().trigger_reason(raw, 1000), "buy_scroll")
        self.assertEqual(self._win_term(raw), "portal_trigger")

    def test_a_standing_portal_at_full_hp_also_closes_with_portal_trigger(self):
        raw = standing_portal_raw()
        self.assertEqual(PortalService().trigger_reason(raw, 1000),
                         "portal_standing")
        self.assertEqual(self._win_term(raw), "portal_trigger")

    def test_the_danger_law_is_the_only_one_that_forfeits(self):
        for label, raw, forfeited in (
                # town, full HP, buying a scroll: an errand, keep the escrow
                ("buy_scroll", town_raw(), False),
                # main L5, full HP, a door already open here: still no danger
                ("portal_standing", standing_portal_raw(), False),
                # main L5 at 20/100 HP: the retreat law's own evidence
                ("low_hp", dungeon_raw(hp=20), True),
                # main L5, healthy but the belt is empty below 75%
                ("empty_belt", dungeon_raw(hp=70, belt_heal_kinds=[]), True),
                # main L1: the portal law never fires below L2 anyway
                ("l1", dungeon_raw(hp=20, depth=1), False)):
            with self.subTest(label=label):
                fake = escrow_self(resource_retreat="retreat-v1",
                                   resource_portal="portal-v1", raw=raw)
                vested = settle(fake, 3, "portal_trigger")
                if forfeited:
                    self.assertEqual(vested, 0.0)
                    self.assertEqual(fake.stats["descend_escrow_forfeited"], 7.5)
                else:
                    self.assertEqual(vested, 7.5)
                    self.assertEqual(fake.stats["descend_escrow_vested"], 7.5)
                    self.assertNotIn("descend_escrow_forfeited", fake.stats)

    def test_a_quest_set_level_is_never_a_portal_forfeit(self):
        # portal legs belong to the main dungeon; a set level cannot be one.
        fake = escrow_self(resource_retreat="retreat-v1",
                           resource_portal="portal-v1",
                           raw=dungeon_raw(hp=20, is_set_level=True))
        self.assertEqual(settle(fake, 3, "portal_trigger"), 7.5)

    def test_without_a_portal_service_or_a_state_the_escrow_survives(self):
        # Fail open on hazard pay: an unreadable state must not destroy it.
        for label in ("no service", "no raw"):
            with self.subTest(label=label):
                fake = escrow_self(resource_retreat="retreat-v1",
                                   resource_portal="portal-v1")
                if label == "no service":
                    fake.oe.portal_service = None
                else:
                    fake.oe.env._raw = None
                self.assertEqual(settle(fake, 3, "portal_trigger"), 7.5)

    def test_the_danger_clauses_have_exactly_one_home(self):
        # trigger_reason delegates to danger_reason; the thresholds are not
        # copied into the worker-side settlement.
        service = PortalService()
        self.assertEqual(service.danger_reason(dungeon_raw(hp=20)), "low_hp")
        self.assertIsNone(service.danger_reason(dungeon_raw(hp=100)))
        self.assertEqual(service.trigger_reason(dungeon_raw(hp=20), 1000),
                         service.danger_reason(dungeon_raw(hp=20)))
        source = (ROOT / "python/diablogym/worker_env.py").read_text(
            encoding="utf-8")
        self.assertIn("portal.danger_reason(raw) is not None", source)
        self.assertNotIn("hp_fraction", source)


class WindowReasonCountersTests(unittest.TestCase):
    """_log counts arbitrary reason strings; the portal needs no new code."""

    def test_portal_reasons_are_counted_in_stats_and_ff_stats(self):
        fake = SimpleNamespace(
            log_windows=False, window_log=[],
            stats={"reasons": {}, "ff_reasons": {}, "ff_windows": 0})
        for reason, ff in (("portal_trigger", False), ("portal_complete", True),
                           ("portal_complete", True)):
            worker_env.WorkerWindowEnv._log(fake, {"reason": reason}, ff)
        self.assertEqual(fake.stats["reasons"],
                         {"portal_trigger": 1, "portal_complete": 2})
        self.assertEqual(fake.stats["ff_reasons"], {"portal_complete": 2})
        self.assertEqual(fake.stats["ff_windows"], 2)


class EvalContractTests(unittest.TestCase):
    """(f) the eval-side identity keys."""

    def test_the_defaults_keep_every_existing_archive_byte_identical(self):
        self.assertEqual(contract.R16_ENVIRONMENT_DEFAULTS["resource_portal"], "off")
        self.assertEqual(contract.R16_ENVIRONMENT_DEFAULTS["hunt_scope"], "all")
        self.assertNotIn("r16_environment", contract.make_protocol([2114000]))

    def test_the_validator_accepts_portal_v1_only_on_top_of_retreat_v1(self):
        self.assertEqual(contract.validate_r16_environment(dict(PORTAL)), PORTAL)
        for invalid in ({"resource_portal": "off"},
                        {"resource_portal": "portal-v2",
                         "resource_protocol": "l2-town-v1",
                         "resource_readiness_law": "coach-v03",
                         "resource_retreat": "retreat-v1"},
                        {"resource_portal": "portal-v1"},
                        {"resource_portal": "portal-v1",
                         "resource_protocol": "l2-town-v1"},
                        {"resource_portal": "portal-v1",
                         "resource_readiness_law": "coach-v03"},
                        # the portal without the retreat it is a vehicle for
                        dict(RETREAT, resource_retreat="off",
                             resource_portal="portal-v1")):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, contract.EvalContractError)):
                    contract.validate_r16_environment(invalid)

    def test_the_validator_accepts_l1_only_only_beside_the_hunt_it_scopes(self):
        self.assertEqual(contract.validate_r16_environment(dict(HUNT_ON)), HUNT_ON)
        for invalid in ({"hunt_scope": "all"}, {"hunt_scope": "l1"},
                        {"hunt_scope": True},
                        # Review round: l1-only without the global hunt is a
                        # guaranteed no-op; the identity must not testify to a
                        # law the evaluated world never ran.
                        dict(HUNT),
                        dict(HUNT, explore_global_hunt=False)):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, contract.EvalContractError)):
                    contract.validate_r16_environment(invalid)

    def test_the_keys_survive_the_protocol_round_trip(self):
        protocol = contract.make_protocol(
            [7, 8], r16_environment=dict(PORTAL, **HUNT_ON))
        self.assertEqual(protocol["r16_environment"]["resource_portal"], "portal-v1")
        self.assertEqual(protocol["r16_environment"]["hunt_scope"], "l1-only")


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
        for key in ("resource_portal", "hunt_scope"):
            with self.subTest(key=key):
                self.assertIn(key, missing)
                self.assertIsNone(missing[key])
        self.assertEqual(missing, training_identity(resource_portal="off",
                                                    hunt_scope="all"))
        # An old checkpoint that predates both keys resumes without drift.
        old = dict(missing)
        del old["resource_portal"], old["hunt_scope"]
        training._validate_resume_contract(old, missing)
        current = training_identity(**PORTAL, **HUNT)
        self.assertEqual(current["resource_portal"], "portal-v1")
        self.assertEqual(current["hunt_scope"], "l1-only")

    def test_the_default_contract_gains_no_non_none_value(self):
        missing = training_identity()
        self.assertEqual(
            {key: value for key, value in missing.items()
             if key in ("resource_portal", "hunt_scope", "resource_retreat")},
            {"resource_portal": None, "hunt_scope": None, "resource_retreat": None})

    def test_both_changes_are_drift_that_only_the_named_restart_may_ride(self):
        for key in ("resource_portal", "hunt_scope"):
            self.assertIn(key, training._ENVIRONMENT_RESTART_ALLOWED_DRIFT)
        saved = training_identity(**RETREAT)
        for current, pattern in ((training_identity(**PORTAL), "resource_portal"),
                                 (training_identity(**RETREAT, **HUNT), "hunt_scope")):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    training._validate_resume_contract(saved, current)
                training._validate_resume_contract(saved, current,
                                                   allow_environment_restart=True)

    def test_the_resource_resume_identity_keys_cover_the_portal_law(self):
        base = {"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                "resource_service_policy": "sustain-v6",
                "resource_service_recipe": training.resource_service_recipe(
                    "l2-town-v1", "full", "sustain-v6")}
        training._validate_resource_resume_identity(dict(base), dict(base))
        with self.assertRaisesRegex(ValueError, "separately identified initialization"):
            training._validate_resource_resume_identity(
                dict(base), dict(base, resource_portal="portal-v1"))

    def test_hunt_scope_is_not_smuggled_into_the_resource_service_identity(self):
        # hunt_scope is a DiabloGymEnv exploration switch, not part of the
        # resource_service version; its drift is governed by the environment
        # restart whitelist instead.
        base = {"resource_protocol": "l2-town-v1", "resource_purchase_mode": "full",
                "resource_service_policy": "sustain-v6",
                "resource_service_recipe": training.resource_service_recipe(
                    "l2-town-v1", "full", "sustain-v6")}
        training._validate_resource_resume_identity(
            dict(base), dict(base, hunt_scope="l1-only"))


class TrainingCliTests(unittest.TestCase):
    """(e) argparse choices and (c) the fail-closed combinations."""

    def test_the_cli_defaults_and_choices(self):
        args = parsed_args(self)
        self.assertEqual(args.resource_portal, "off")
        self.assertEqual(args.hunt_scope, "all")
        self.assertEqual(parsed_args(
            self, "--resource-portal", "portal-v1").resource_portal, "portal-v1")
        self.assertEqual(parsed_args(
            self, "--hunt-scope", "l1-only").hunt_scope, "l1-only")
        for flags in (("--resource-portal", "portal-v2"),
                      ("--resource-portal", "on"),
                      ("--hunt-scope", "l1"),
                      ("--hunt-scope", "off")):
            with self.subTest(flags=flags):
                with patch.object(sys, "argv", ["train_ppo.py", *flags]):
                    with self.assertRaises(SystemExit):
                        training._main(SimpleNamespace())

    def _legal_options_args(self):
        args = parsed_args(self)
        args.total_steps = 8192
        args.options, args.algo, args.gamma, args.max_steps = True, "mppo", 1.0, 6000
        return args

    def test_validate_args_pins_the_portal_to_the_retreat_and_the_classroom(self):
        args = self._legal_options_args()
        training._validate_args(args)
        args.resource_portal = "portal-v1"
        with self.assertRaisesRegex(ValueError, "resource-portal"):
            training._validate_args(args)   # off protocol / veto law
        args.resource_protocol = "l2-town-v1"
        with self.assertRaisesRegex(ValueError, "resource-portal"):
            training._validate_args(args)   # still veto-v1
        args.resource_readiness_law = "coach-v03"
        with self.assertRaisesRegex(ValueError, "resource-portal"):
            training._validate_args(args)   # still no retreat-v1
        args.resource_retreat = "retreat-v1"
        training._validate_args(args)       # the legal classroom passes
        args.resource_portal = "portal-v9"
        with self.assertRaisesRegex(ValueError, "resource-portal"):
            training._validate_args(args)

    def test_validate_args_refuses_the_portal_without_worker_or_options(self):
        args = parsed_args(self)
        args.total_steps = 8192
        args.resource_protocol, args.resource_readiness_law = "l2-town-v1", "coach-v03"
        args.resource_retreat, args.resource_portal = "retreat-v1", "portal-v1"
        with self.assertRaisesRegex(
                ValueError, "resource-portal|resource-retreat|resource-protocol"):
            training._validate_args(args)

    def test_validate_args_accepts_l1_only_and_refuses_it_outside_the_classroom(self):
        args = self._legal_options_args()
        args.hunt_scope = "l1-only"
        # Review round: l1-only is read only by the a10 global-hunt gate, so
        # without --explore-global-hunt it is a guaranteed no-op that would
        # still mint a non-default contract value.
        self.assertFalse(args.explore_global_hunt)
        with self.assertRaisesRegex(ValueError, "explore-global-hunt"):
            training._validate_args(args)
        args.explore_global_hunt = True
        training._validate_args(args)       # no resource protocol required
        args.hunt_scope = "l2-only"
        with self.assertRaisesRegex(ValueError, "hunt-scope"):
            training._validate_args(args)
        flat = parsed_args(self)
        flat.total_steps = 8192
        flat.hunt_scope = "l1-only"
        flat.explore_global_hunt = True
        with self.assertRaisesRegex(ValueError, "hunt-scope"):
            training._validate_args(flat)


class MakeEnvTests(unittest.TestCase):
    """(b)/(c) the env factory forwards only the explicit laws."""

    def test_the_env_factory_forwards_the_portal_to_both_modes(self):
        for mode, symbol in (("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")):
            for config in ({}, RETREAT, PORTAL):
                with self.subTest(mode=mode, config=config), patch(
                        f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
                    env = training.make_env(**{mode: True}, **config)
                    forwarded = constructor.call_args.kwargs
                    if "resource_portal" in config:
                        self.assertEqual(forwarded["resource_portal"], "portal-v1")
                    else:
                        self.assertNotIn("resource_portal", forwarded)
                    env.close()

    def test_the_env_factory_forwards_hunt_scope_to_both_modes(self):
        for mode, symbol in (("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")):
            for config in ({}, {"hunt_scope": "all"}, HUNT_ON):
                with self.subTest(mode=mode, config=config), patch(
                        f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
                    env = training.make_env(**{mode: True}, **config)
                    forwarded = constructor.call_args.kwargs
                    if config.get("hunt_scope") == "l1-only":
                        self.assertEqual(forwarded["hunt_scope"], "l1-only")
                    else:
                        self.assertNotIn("hunt_scope", forwarded)
                    env.close()

    def test_the_factory_fails_closed_on_every_illegal_combination(self):
        for kwargs, pattern in (
                ({"worker": True, "resource_portal": "portal-v1"},
                 "resource_portal"),
                ({"worker": True, **RETREAT, "resource_portal": "portal-v2"},
                 "resource_portal"),
                # portal without retreat, inside an otherwise legal classroom
                ({"worker": True, "resource_protocol": "l2-town-v1",
                  "resource_readiness_law": "coach-v03",
                  "resource_portal": "portal-v1"},
                 "resource_portal portal-v1 requires resource_retreat"),
                # portal outside WorkerWindowEnv/OptionsEnv
                (dict(PORTAL), "resource_portal|resource_retreat|resource_protocol"),
                ({"worker": True, "hunt_scope": "l2-only"}, "hunt_scope"),
                ({"hunt_scope": "l1-only"},
                 "hunt_scope l1-only requires WorkerWindowEnv/OptionsEnv"),
                # Review round: the scope without the hunt it scopes.
                ({"worker": True, "hunt_scope": "l1-only"},
                 "hunt_scope l1-only requires explore_global_hunt")):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, pattern):
                    training.make_env(**kwargs)


class EarnedSuffixPinTests(unittest.TestCase):
    """The earned-dive-suffix fixed dict pins neither flag -- retreat is not
    pinned there either, and R18-B5 adds no new pin."""

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

    def test_neither_new_flag_is_pinned_under_the_earned_suffix(self):
        keys = self._fixed_keys()
        self.assertIn("resource_protocol", keys)
        self.assertNotIn("resource_retreat", keys)
        self.assertNotIn("resource_portal", keys)
        self.assertNotIn("hunt_scope", keys)


class EvalAssembledWiringTests(unittest.TestCase):
    """(f) the certification eval must not silently rebuild a no-portal env."""

    def test_the_evaluator_forwards_both_laws_and_omits_the_defaults(self):
        for config, key, value in ((None, "resource_portal", "portal-v1"),
                                   (dict(RETREAT), "resource_portal", "portal-v1"),
                                   (dict(PORTAL), "resource_portal", "portal-v1"),
                                   (None, "hunt_scope", "l1-only"),
                                   (dict(HUNT_ON), "hunt_scope", "l1-only")):
            constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
            with self.subTest(config=config, key=key), patch.object(
                    evaluation, "_native_runtime",
                    return_value=(Mock(), constructor, None)):
                with self.assertRaisesRegex(RuntimeError, "constructor-boundary"):
                    evaluation.evaluate(None, [], r16_environment=config)
            forwarded = constructor.call_args.kwargs
            if config and key in config:
                self.assertEqual(forwarded[key], value)
            else:
                self.assertNotIn(key, forwarded)

    def test_the_eval_cli_declares_both_flags_with_the_frozen_defaults(self):
        source = (ROOT / "train/eval_assembled.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--resource-portal", default="off"', source)
        self.assertIn('"resource_portal": args.resource_portal,', source)
        self.assertIn('ap.add_argument("--hunt-scope", default="all"', source)
        self.assertIn('"hunt_scope": args.hunt_scope,', source)


class EvalAssembledCliGateTests(unittest.TestCase):
    """Review round: the evaluator's own ap.error gates were pinned only by a
    source-string assertion, so deleting them kept the volume green.  Drive the
    parser instead -- argparse exits before any environment work."""

    CLASSROOM = ("--resource-protocol", "l2-town-v1",
                 "--resource-readiness-law", "coach-v03")

    def _cli(self, *flags):
        stderr = io.StringIO()
        with patch.object(sys, "argv",
                          ["eval_assembled.py", "--worker", "script", *flags]):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_case:
                    evaluation.main()
        self.assertEqual(exit_case.exception.code, 2)
        return stderr.getvalue()

    def test_the_portal_needs_its_classroom(self):
        self.assertIn(
            "--resource-portal portal-v1 requires --resource-protocol "
            "l2-town-v1 and --resource-readiness-law coach-v03",
            " ".join(self._cli("--resource-portal", "portal-v1").split()))

    def test_the_portal_needs_the_retreat_it_is_a_vehicle_for(self):
        message = " ".join(self._cli(
            "--resource-portal", "portal-v1", *self.CLASSROOM).split())
        self.assertIn("--resource-portal portal-v1 requires "
                      "--resource-retreat retreat-v1", message)

    def test_l1_only_needs_the_global_hunt_it_scopes(self):
        self.assertIn(
            "--hunt-scope l1-only requires --explore-global-hunt",
            " ".join(self._cli("--hunt-scope", "l1-only").split()))

    def test_the_legal_combinations_get_past_every_gate(self):
        # No SystemExit from the gates themselves: the legal combinations run
        # on to the bridge-preload guard, which is as far as a unit volume may
        # follow the certification CLI.
        for flags in (("--hunt-scope", "l1-only", "--explore-global-hunt"),
                      ("--resource-portal", "portal-v1", "--resource-retreat",
                       "retreat-v1") + self.CLASSROOM):
            with self.subTest(flags=flags):
                stderr = io.StringIO()
                with patch.object(sys, "argv", ["eval_assembled.py", "--worker",
                                                "script", *flags]):
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaisesRegex(contract.EvalContractError,
                                                    "bridge"):
                            evaluation.main()


if __name__ == "__main__":
    unittest.main()
