"""Configuration-only DIVE recovery tests; no engine reset/step or runtime claim."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "tests"))
import eval_contract as contract
import eval_assembled as evaluation
import train_ppo as training
from test_eval_pipeline import _valid_v5_archive
from test_resource_sustain_contract import DummyEnv, training_identity


class DiveRecoveryContractTests(unittest.TestCase):
    def test_off_preserves_exact_default_training_and_eval_dictionaries(self):
        self.assertEqual(training_identity(), training_identity(dive_blocker_recovery="off"))
        self.assertNotIn("dive_blocker_recovery", training_identity())
        self.assertIsNone(contract.dive_blocker_recovery_recipe())
        self.assertNotIn("dive_blocker_recovery_recipe", contract.make_protocol([7]))
        for policy in ("sustain-v2", "sustain-v3", "sustain-v4"):
            config = dict(resource_protocol="l2-town-v1", resource_service_policy=policy)
            self.assertEqual(training_identity(**config),
                             training_identity(**config, dive_blocker_recovery="off"))

    def test_named_flag_is_the_only_contract_change_and_service_recipe_does_not_change(self):
        for policy in ("sustain-v2", "sustain-v3", "sustain-v4"):
            config = dict(resource_protocol="l2-town-v1", resource_service_policy=policy)
            off = training_identity(**config)
            on = training_identity(**config, dive_blocker_recovery="adjacent-v1")
            self.assertEqual({key for key in set(off) | set(on) if off.get(key) != on.get(key)},
                             {"dive_blocker_recovery"})
            self.assertEqual(off["resource_service_recipe"], on["resource_service_recipe"])
            with self.assertRaisesRegex(ValueError, "dive_blocker_recovery"):
                training._validate_resume_contract(off, on, allow_environment_restart=True)

    def test_separate_eval_recipe_binds_core_and_charged_settle_without_claiming_full_cap12(self):
        for policy in ("sustain-v2", "sustain-v3", "sustain-v4"):
            config = dict(resource_protocol="l2-town-v1", resource_service_policy=policy)
            off = contract.make_protocol([7, 8], r16_environment=config)
            on = contract.make_protocol([7, 8], r16_environment={**config, "dive_blocker_recovery": "adjacent-v1"})
            self.assertEqual(off["resource_service_recipe"], on["resource_service_recipe"])
            self.assertNotIn("dive_blocker_recovery_recipe", off)
            recipe = on["dive_blocker_recovery_recipe"]
            self.assertEqual(recipe["core_macro_microstep_cap"], 12)
            self.assertNotIn("macro_microstep_cap", recipe)
            self.assertEqual(recipe["settle"], "normal-idle-settle-charged-each-native-microstep")
            self.assertEqual(recipe["deadline"], "live-episode-and-followup-cutoff")
            document = _valid_v5_archive()
            document["meta"]["protocol"] = on
            contract.validate_eval_archive(document)
            for mutation in ("missing", "budget"):
                invalid = deepcopy(document)
                target = invalid["meta"]["protocol"]
                if mutation == "missing":
                    del target["dive_blocker_recovery_recipe"]
                else:
                    target["dive_blocker_recovery_recipe"]["core_macro_microstep_cap"] = 24
                with self.subTest(policy=policy, mutation=mutation), self.assertRaises(contract.EvalContractError):
                    contract.validate_eval_archive(invalid)

    def test_resource_off_unknown_modes_and_noncanonical_eval_defaults_are_rejected(self):
        for protocol, recovery in (("off", "adjacent-v1"), ("l2-town-v1", "adjacent-v2"),
                                   ("l2-town-v1", True), ("l2-town-v1", None)):
            with self.subTest(protocol=protocol, recovery=recovery), self.assertRaises(ValueError):
                training.make_env(worker=True, resource_protocol=protocol, dive_blocker_recovery=recovery)
        for config in ({"dive_blocker_recovery": "adjacent-v1"},
                       {"dive_blocker_recovery": "off"},
                       {"resource_protocol": "l2-town-v1", "dive_blocker_recovery": "off"}):
            with self.subTest(config=config), self.assertRaises(contract.EvalContractError):
                contract.validate_r16_environment(config)
        document = _valid_v5_archive()
        document["meta"]["protocol"]["dive_blocker_recovery_recipe"] = None
        with self.assertRaises(contract.EvalContractError):
            contract.validate_eval_archive(document)

    def test_factory_forwards_on_only_and_eval_uses_the_same_flag(self):
        for mode, symbol in (("worker", "WorkerWindowEnv"), ("options", "OptionsEnv")):
            for recovery in ("off", "adjacent-v1"):
                with self.subTest(mode=mode, recovery=recovery), patch(
                        f"diablogym.{symbol}", side_effect=DummyEnv) as constructor:
                    env = training.make_env(**{mode: True}, resource_protocol="l2-town-v1",
                        resource_service_policy="sustain-v3", dive_blocker_recovery=recovery)
                    if recovery == "off":
                        self.assertNotIn("dive_blocker_recovery", constructor.call_args.kwargs)
                    else:
                        self.assertEqual(constructor.call_args.kwargs["dive_blocker_recovery"], recovery)
                    env.close()
        constructor = Mock(side_effect=RuntimeError("constructor-boundary"))
        with patch.object(evaluation, "_native_runtime", return_value=(Mock(), constructor, None)):
            with self.assertRaisesRegex(RuntimeError, "constructor-boundary"):
                evaluation.evaluate(None, [], r16_environment={"resource_protocol": "l2-town-v1",
                    "resource_service_policy": "sustain-v3", "dive_blocker_recovery": "adjacent-v1"})
        self.assertEqual(constructor.call_args.kwargs["dive_blocker_recovery"], "adjacent-v1")


if __name__ == "__main__":
    unittest.main()
