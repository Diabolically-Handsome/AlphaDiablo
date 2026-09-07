"""Explicit sustain service identity and unchanged legacy/resume behavior.

No game reset, native step, optimizer update or model checkpoint is required.
"""
from copy import deepcopy
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

SUSTAIN = {"resource_protocol": "l2-town-v1",
           "resource_service_policy": "sustain-v2"}
SUSTAIN_V3 = {**SUSTAIN, "resource_service_policy": "sustain-v3"}
SUSTAIN_V4 = {**SUSTAIN, "resource_service_policy": "sustain-v4"}


class DummyEnv(gym.Env):
    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.observation_space = spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.action_space = spaces.Discrete(15)


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


class SustainContractTests(unittest.TestCase):
    def test_legacy_recipe_and_default_eval_identity_remain_exact(self):
        self.assertIsNone(contract.resource_service_recipe())
        self.assertEqual(contract.resource_service_recipe("l2-town-v1", "full"), {
            "version": "l2-town-paid-repair-v2", "mode": "full",
            "stages": ["collect_gold", "town_trip", "armor_if_needed",
                       "normal_paid_repair", "native_heal",
                       "potions_to_capacity", "return_l1"],
            "service_microstep_cap": 600, "potion_target": 8})
        self.assertNotIn("r16_environment", contract.make_protocol([2114000]))
        legacy = contract.make_protocol([2114000], r16_environment={
            "resource_protocol": "l2-town-v1"})
        self.assertNotIn("service_policy", legacy["resource_service_recipe"])

    def test_new_recipe_records_exact_service_scope_and_native_target(self):
        protocol = contract.make_protocol([2114000], r16_environment=SUSTAIN)
        self.assertEqual(protocol["r16_environment"], SUSTAIN)
        recipe = protocol["resource_service_recipe"]
        self.assertEqual(recipe["service_microstep_cap"], 1500)
        self.assertEqual(recipe["potion_target"], 4)
        self.assertEqual(recipe["potion_target_source"], "native_readiness.required_belt_heals")
        self.assertEqual(recipe["service_policy"], "sustain-v2")
        self.assertEqual(recipe["stages"], ["collect_gold", "town_trip", "observe_smith",
            "observe_healer_and_native_heal", "joint_minimum_readiness_budget",
            "normal_unequip_repair_or_replace", "potions_to_readiness", "return_l1"])
        self.assertEqual(recipe["planning_scope"],
                         "retain_or_one_armor_change_plus_repairs_live_replan")
        self.assertEqual(recipe["navigation_recovery"], "bounded-combat-heal-v1")

    def test_v3_gold_memory_identity_changes_only_explicit_collection_recipe(self):
        v2 = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v2")
        v2_before = deepcopy(v2)
        v3 = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v3")
        self.assertEqual(v2, v2_before)
        self.assertNotIn("gold_memory", v2)
        self.assertNotIn("collect_microstep_cap", v2)
        self.assertEqual(v3["version"], "l2-town-joint-sustain-v3")
        self.assertEqual(v3["stages"], ["collect_observed_l1_gold", *v2["stages"][1:]])
        self.assertEqual(v3["service_microstep_cap"], 1500)
        self.assertEqual(v3["collect_microstep_cap"], 300)
        self.assertEqual(v3["collect_budget_exhausted"], "continue_outbound")
        self.assertEqual(v3["gold_memory"]["version"], "observed-l1-gold-memory-v1")
        self.assertEqual(v3["gold_memory"]["identity_fields"],
                         ["seed_hi", "seed_lo", "create_info", "base_id"])
        self.assertFalse(v3["gold_memory"]["remembered_value_is_wallet"])
        for key in ("planning_scope", "service_microstep_cap", "potion_target",
                    "potion_target_source", "navigation_recovery", "mode"):
            self.assertEqual(v2[key], v3[key])
        self.assertEqual(contract.make_protocol([7], r16_environment=SUSTAIN_V3)
                         ["resource_service_recipe"], v3)

    def test_v3_archive_cannot_mix_v2_or_changed_collect_budget(self):
        document = _valid_v5_archive()
        document["meta"]["protocol"] = contract.make_protocol([7, 8], r16_environment=SUSTAIN_V3)
        contract.validate_eval_archive(document)
        for mutation in ("v2", "cap", "wallet"):
            changed = deepcopy(document)
            protocol = changed["meta"]["protocol"]
            if mutation == "v2":
                protocol["resource_service_recipe"] = contract.resource_service_recipe("l2-town-v1")
            elif mutation == "cap":
                protocol["resource_service_recipe"]["collect_microstep_cap"] = 600
            else:
                protocol["resource_service_recipe"]["gold_memory"]["remembered_value_is_wallet"] = True
            with self.subTest(mutation=mutation), self.assertRaises(contract.EvalContractError):
                contract.validate_eval_archive(changed)

    def test_v4_command_window_is_distinct_from_v3_physical_budget_label(self):
        v3 = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v3")
        before = deepcopy(v3)
        v4 = contract.resource_service_recipe("l2-town-v1", "full", "sustain-v4")
        self.assertEqual(v3, before)
        self.assertEqual(v3["collect_microstep_cap"], 300)
        self.assertNotIn("collect_command_window_microsteps", v3)
        self.assertNotIn("collect_tail", v3)
        self.assertEqual(v4["collect_command_window_microsteps"], 450)
        self.assertNotIn("collect_microstep_cap", v4)
        self.assertEqual(v4["collect_budget_scope"],
                         "new-collect-commands-within-actual-microstep-window")
        self.assertEqual(v4["collect_tail"],
                         "finish-existing-animation-in-outbound-charged-to-service-budget")
        self.assertEqual(v4["collect_cutoff"],
                         "record-current-native-state-at-command-window-cutoff")
        for key in ("service_microstep_cap", "planning_scope", "potion_target",
                    "potion_target_source", "navigation_recovery", "stages", "gold_memory"):
            self.assertEqual(v3[key], v4[key])
        self.assertEqual(v4["service_microstep_cap"], 1500)
        document = _valid_v5_archive()
        document["meta"]["protocol"] = contract.make_protocol([7, 8], r16_environment=SUSTAIN_V4)
        contract.validate_eval_archive(document)
        for key, value in (("collect_command_window_microsteps", 300),
                           ("collect_microstep_cap", 450),
                           ("collect_tail", "complete-physical-collection-within-450")):
            changed = deepcopy(document)
            changed["meta"]["protocol"]["resource_service_recipe"][key] = value
            with self.subTest(key=key), self.assertRaises(contract.EvalContractError):
                contract.validate_eval_archive(changed)

    def test_v3_and_v4_cannot_be_relabelled_in_archive_or_ordinary_resume(self):
        for source, target in ((SUSTAIN_V3, SUSTAIN_V4), (SUSTAIN_V4, SUSTAIN_V3)):
            document = _valid_v5_archive()
            document["meta"]["protocol"] = contract.make_protocol([7, 8], r16_environment=source)
            document["meta"]["protocol"]["r16_environment"] = target
            with self.assertRaises(contract.EvalContractError):
                contract.validate_eval_archive(document)
            with self.assertRaisesRegex(ValueError, "resource_service"):
                training._validate_resume_contract(training_identity(**source),
                    training_identity(**target), allow_environment_restart=True)

    def test_unknown_disabled_or_nonfull_sustain_is_rejected(self):
        for protocol, mode, policy in (("off", "full", "sustain-v2"),
                ("off", "full", "sustain-v3"),
                ("l2-town-v1", "armor", "sustain-v3"),
                ("l2-town-v1", "armor", "sustain-v2"),
                ("l2-town-v1", "potions", "sustain-v2"),
                ("l2-town-v1", "heal", "sustain-v2"),
                ("l2-town-v1", "none", "sustain-v2"),
                ("off", "full", "sustain-v4"),
                ("l2-town-v1", "armor", "sustain-v4"),
                ("l2-town-v1", "full", "sustain-v999"),
                ("off", "full", None)):
            with self.subTest(protocol=protocol, mode=mode, policy=policy):
                with self.assertRaises(ValueError):
                    contract.resource_service_recipe(protocol, mode, policy)
                with self.assertRaises(ValueError):
                    training.make_env(options=True, resource_protocol=protocol,
                        resource_purchase_mode=mode, resource_service_policy=policy)

    def test_eval_canonical_form_rejects_explicit_legacy_and_nonfull_sustain(self):
        for config in ({"resource_service_policy": "sustain-v2"},
                       {"resource_service_policy": "legacy-v1"},
                       {**SUSTAIN, "resource_purchase_mode": "armor"},
                       {**SUSTAIN, "resource_purchase_mode": "full"}):
            with self.subTest(config=config), self.assertRaises(contract.EvalContractError):
                contract.validate_r16_environment(config)

    def test_archive_cannot_claim_new_policy_with_legacy_or_forged_recipe(self):
        document = _valid_v5_archive()
        document["meta"]["protocol"] = contract.make_protocol([7, 8], r16_environment=SUSTAIN)
        contract.validate_eval_archive(document)
        for mutation in ("legacy", "cap", "potions", "omit"):
            invalid = deepcopy(document)
            protocol = invalid["meta"]["protocol"]
            if mutation == "legacy":
                protocol["resource_service_recipe"] = contract.resource_service_recipe("l2-town-v1")
            elif mutation == "omit":
                del protocol["resource_service_recipe"]
            else:
                protocol["resource_service_recipe"][
                    "service_microstep_cap" if mutation == "cap" else "potion_target"] = 600 if mutation == "cap" else 8
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    contract.EvalContractError, "service recipe"):
                contract.validate_eval_archive(invalid)

    def test_training_missing_policy_matches_explicit_legacy(self):
        missing = training_identity()
        explicit = training_identity(resource_service_policy="legacy-v1")
        self.assertEqual(missing, explicit)
        self.assertIsNone(missing["resource_service_policy"])
        old = dict(missing)
        del old["resource_service_policy"]
        training._validate_resume_contract(old, explicit)
        enabled_legacy = training_identity(resource_protocol="l2-town-v1")
        self.assertIsNone(enabled_legacy["resource_service_policy"])
        self.assertEqual(enabled_legacy["resource_service_recipe"]["service_microstep_cap"], 600)

    def test_new_training_contract_is_explicit_and_broad_resume_still_rejects(self):
        new = training_identity(**SUSTAIN)
        self.assertEqual(new["resource_service_policy"], "sustain-v2")
        self.assertEqual(new["resource_service_recipe"],
                         contract.resource_service_recipe("l2-town-v1", "full", "sustain-v2"))
        for saved in (training_identity(), training_identity(resource_protocol="l2-town-v1")):
            with self.assertRaisesRegex(ValueError, "resource_service_policy"):
                training._validate_resume_contract(saved, new, allow_environment_restart=True)
        training._validate_resume_contract(new, deepcopy(new))

    def test_train_factory_forwards_only_explicit_policy_to_both_modes(self):
        for mode, symbol in (("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")):
            for config in ({}, {"resource_protocol": "l2-town-v1"}, SUSTAIN, SUSTAIN_V3, SUSTAIN_V4):
                with self.subTest(mode=mode, config=config), patch(
                        f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
                    env = training.make_env(**{mode: True}, **config)
                    forwarded = constructor.call_args.kwargs
                    if "resource_service_policy" in config:
                        self.assertEqual(forwarded["resource_service_policy"], config["resource_service_policy"])
                    else:
                        self.assertNotIn("resource_service_policy", forwarded)
                    if not config:
                        self.assertNotIn("resource_protocol", forwarded)
                    env.close()

    def test_eval_forwards_new_policy_and_omits_default_at_constructor(self):
        for config in (None, SUSTAIN, SUSTAIN_V3, SUSTAIN_V4):
            constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
            with self.subTest(config=config), patch.object(evaluation, "_native_runtime",
                    return_value=(Mock(), constructor, None)):
                with self.assertRaisesRegex(RuntimeError, "constructor-boundary"):
                    evaluation.evaluate(None, [], r16_environment=config)
            forwarded = constructor.call_args.kwargs
            if config:
                self.assertEqual(forwarded["resource_service_policy"], config["resource_service_policy"])
            else:
                self.assertNotIn("resource_service_policy", forwarded)

    def test_new_code_is_bound_to_both_implementation_bundles(self):
        for name in ("resource_navigation.py", "resource_sustain.py", "resource_gold_memory.py",
                     "resource_sustain_gold.py"):
            relative = "python/diablogym/" + name
            self.assertIn(relative, contract.PROTOCOL_SOURCE_FILES)
            self.assertIn(relative, training._IMPLEMENTATION_SOURCE_FILES)
            self.assertTrue((ROOT / relative).is_file())


if __name__ == "__main__":
    unittest.main()
