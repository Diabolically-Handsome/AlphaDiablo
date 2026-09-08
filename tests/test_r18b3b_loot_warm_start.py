"""R18-B3b (2026-09-07): the sustain-loot-v1 warm-start path, end to end.

Everything about schema/2 (``migrate_loot_candidate``) plus proof that schema/1
-- the five frozen sustain-v2..v6 operations, their receipts and their allowed
contract vocabulary -- is byte-identical to the tree before this change.  The
literals in ``FROZEN_*`` were captured from the main tree
(``~/AlphaDiablo/diablogym``) and from ``b3-tree.pre-b3b`` BEFORE the R18-B3b
edit; the two agree on every one of them (the only pre-existing difference
between those trees is R18-B3's reserved, unmintable loot name).
"""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "python"))

import migrate_resource_candidate as legacy  # noqa: E402
import migrate_loot_candidate as loot  # noqa: E402
import eval_contract  # noqa: E402
from eval_contract import worker_prefix_recipe, EARNED_DIVE_SUFFIX_SCOPE  # noqa: E402

PARENT = ROOT / "train/runs/r16-arm-a-constitution/model_candidate.zip"
IMPL = "a" * 64
POLICY_SHA = "b" * 64

# ---- literals captured before the R18-B3b edit -----------------------------
FROZEN_SCHEMA = "diablogym-resource-warm-start/1"
FROZEN_OPERATIONS = {
    "sustain-v2": "r16-to-sustain-v2-weights-only-v1",
    "sustain-v3": "r16-to-sustain-v3-weights-only-v1",
    "sustain-v4": "r16-to-sustain-v4-weights-only-v1",
    "sustain-v5": "r16-to-sustain-v5-weights-only-v1",
    "sustain-v6": "r16-to-sustain-v6-weights-only-v1",
}
FROZEN_OPERATION_FOR = {
    ("sustain-v2", "off"): "r16-to-sustain-v2-weights-only-v1",
    ("sustain-v2", "adjacent-v1"): "r16-to-sustain-v2-dive-adjacent-v1-weights-only-v1",
    ("sustain-v3", "off"): "r16-to-sustain-v3-weights-only-v1",
    ("sustain-v3", "adjacent-v1"): "r16-to-sustain-v3-dive-adjacent-v1-weights-only-v1",
    ("sustain-v4", "off"): "r16-to-sustain-v4-weights-only-v1",
    ("sustain-v4", "adjacent-v1"): "r16-to-sustain-v4-dive-adjacent-v1-weights-only-v1",
    ("sustain-v5", "off"): "r16-to-sustain-v5-weights-only-v1",
    ("sustain-v5", "adjacent-v1"): "r16-to-sustain-v5-dive-adjacent-v1-weights-only-v1",
    ("sustain-v6", "off"): "r16-to-sustain-v6-weights-only-v1",
    ("sustain-v6", "adjacent-v1"): "r16-to-sustain-v6-dive-adjacent-v1-weights-only-v1",
}
FROZEN_EARNED_OPERATION = (
    "r16-to-sustain-v6-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v1")
FROZEN_DEPTH_SIGNAL_OPERATION = (
    "r16-to-sustain-v6-earned-dive-suffix-v1-depth24-dive-adjacent-v1-weights-only-v1")
FROZEN_BASE_ALLOWED_CONTRACT_KEYS = {
    "implementation_sha256", "resource_protocol", "resource_purchase_mode",
    "resource_service_policy", "resource_service_recipe"}
FROZEN_ALLOWED_CONTRACT_KEYS = FROZEN_BASE_ALLOWED_CONTRACT_KEYS | {"dive_blocker_recovery"}
FROZEN_EARNED_ALLOWED_CONTRACT_KEYS = FROZEN_ALLOWED_CONTRACT_KEYS | {
    "worker_learning_window_scope", "worker_prefix"}
# json_sha256 of the schema/1 target contracts and receipts built from the exact
# registered R16 parent with implementation "a"*64 and policy_sha256 "b"*64.
FROZEN_TARGET_SHA256 = {
    "sustain-v6-plain": "3ebdf8b472468fa705f64e5e5ef10c19310fed13151abda4a937cf3891d9f2ba",
    "sustain-v6-adjacent": "9f8b38b4edab33a1af6627a8bd6e845c14c06286e1592d97694baf9d866b7573",
    "sustain-v6-earned": "9e5a0686da5f4b059557d3485639c9e54de353548abb5fd95f84618bc81b5487",
}
FROZEN_RECEIPT_SHA256 = {
    "sustain-v6-plain": "524c3075b5f05300837177c0ec6623948c5a2192e86847a2f07bb8c9596542b4",
    "sustain-v6-adjacent": "7d89152e11a16b24f63b60e1760323f57238cadb23d4c55792e62235fb49c4b0",
    "sustain-v6-earned": "74792a2a7006f2d24c1da7efcd93dbf734604ee987b8ab6f8392ea5c20b66f47",
}

# ---- the two new, clock-bearing operation names ----------------------------
NEW_OPERATIONS = {
    "completion-l2-v1":
        "r16-to-sustain-loot-v1-completion-l2-v1"
        "-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v2",
    "completion-l2-r18c":
        "r16-to-sustain-loot-v1-completion-l2-r18c"
        "-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v2",
}
RECEIPT_KEYS = {
    "schema", "operation", "parent_checkpoint_sha256", "parent_contract_sha256",
    "parent_num_timesteps", "source_contract", "target_contract_sha256",
    "target_implementation_sha256", "policy_sha256", "historical_critic_warmup",
    "parent_resume_lineage", "optimizer_state", "current_world_training_steps",
    "historical_warmup_is_current_training", "exact_trajectory_continuation",
    "environment_state_mode", "publication_status", "resource_protocol",
    "resource_purchase_mode", "resource_service_policy", "resource_service_recipe",
    "worker_time_protocol", "worker_time_recipe", "worker_learning_window_scope",
    "worker_prefix", "dive_blocker_recovery", "dive_blocker_recovery_recipe",
    "target_world",
}
# R18-M2 (2026-09-07): hunt_scope joined the registered world (B5 made it a
# training flag and _training_contract emits the key unconditionally).
# R18-B6 (2026-09-07): the three loot-itinerary laws joined it for the same
# reason -- _training_contract emits all three keys unconditionally.
WORLD_KEYS = {"worker_time_protocol", "worker_time_recipe", "resource_readiness_law",
              "resource_retreat", "worker_descend_escrow_readiness_table",
              "hunt_scope", "resource_sweep", "resource_identify",
              "resource_weapon_upgrade"}
# The exact contract keys a warm start from the R16 parent into the R18-B3b arm
# may change; measured against the real training contract train_ppo builds.
MIGRATED_KEYS = {
    "implementation_sha256", "resource_protocol", "resource_purchase_mode",
    "resource_service_policy", "resource_service_recipe", "dive_blocker_recovery",
    "worker_learning_window_scope", "worker_prefix", "worker_time_protocol",
    "worker_time_recipe", "resource_readiness_law", "resource_retreat",
}


def _prefix(sha=legacy.PARENT_SHA256, attempts=1000, microsteps=100_000_000):
    return worker_prefix_recipe(EARNED_DIVE_SUFFIX_SCOPE, sha, attempts, microsteps)


class LootMigrationVocabularyTests(unittest.TestCase):
    """Names, schemas and registries; no model, no engine, no parent needed."""

    def test_the_new_operations_are_new_clock_bearing_and_disjoint(self):
        self.assertEqual(loot.OPERATIONS, NEW_OPERATIONS)
        self.assertEqual(loot.SCHEMA, "diablogym-resource-warm-start/2")
        self.assertEqual(legacy.SCHEMA_V2, loot.SCHEMA)
        for clock, name in NEW_OPERATIONS.items():
            with self.subTest(clock=clock):
                self.assertIn(clock, name)
                self.assertTrue(name.endswith("-weights-only-v2"))
                self.assertEqual(loot.operation_for(clock), name)
        self.assertEqual(len(set(NEW_OPERATIONS.values())), 2)
        old = set(legacy.OPERATIONS.values()) | {FROZEN_EARNED_OPERATION,
                                                 FROZEN_DEPTH_SIGNAL_OPERATION,
                                                 legacy.DEPTH_SIGNAL_OPERATION}
        self.assertFalse(old & set(NEW_OPERATIONS.values()))

    def test_the_registered_clocks_are_exactly_the_registered_loot_recipe_clocks(self):
        self.assertEqual(tuple(loot.OPERATIONS),
                         tuple(eval_contract.LOOT_SERVICE_RECIPE_VERSIONS))
        from diablogym.completion_clock import COMPLETION_PROTOCOLS
        self.assertEqual(tuple(loot.OPERATIONS), tuple(COMPLETION_PROTOCOLS))

    def test_unregistered_clocks_fail_closed(self):
        for clock in (None, "legacy", "completion-l2-v2", "completion-l2", 1, ""):
            with self.subTest(clock=clock), self.assertRaisesRegex(
                    ValueError, "registered completion-l2 clock"):
                loot.operation_for(clock)

    def test_the_time_recipe_is_the_exact_immutable_clock_recipe(self):
        from diablogym.completion_clock import COMPLETION_RECIPES
        for clock, recipe in COMPLETION_RECIPES.items():
            with self.subTest(clock=clock):
                self.assertEqual(loot.time_recipe(clock), recipe.as_dict())
        self.assertEqual(loot.time_recipe("completion-l2-r18c")["followup_microsteps"], 9000)
        self.assertEqual(loot.time_recipe("completion-l2-v1")["followup_microsteps"], 1800)

    def test_only_the_registered_laws_of_the_arm_are_mintable(self):
        self.assertEqual(loot.READINESS_LAWS, ("coach-v03",))
        self.assertEqual(loot.RETREATS, ("retreat-v1",))
        self.assertEqual(loot.ESCROW_READINESS_TABLES, ("v1", "v2"))
        # R18-M2 (2026-09-07): the fourth registered law of the arm.
        self.assertEqual(loot.HUNT_SCOPES, ("all", "l1-only"))
        for law in ("veto-v1", "coach-v04", None, "coach-v3"):
            with self.subTest(law=law), self.assertRaisesRegex(
                    ValueError, "registered readiness law"):
                loot.target_world("completion-l2-r18c", law, "retreat-v1", "v1")
        for retreat in ("off", "retreat-v2", None):
            with self.subTest(retreat=retreat), self.assertRaisesRegex(
                    ValueError, "registered retreat law"):
                loot.target_world("completion-l2-r18c", "coach-v03", retreat, "v1")
        for table in ("v3", None, "V2"):
            with self.subTest(table=table), self.assertRaisesRegex(
                    ValueError, "descend-escrow readiness table"):
                loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1", table)
        # R18-M2 (2026-09-07): same fail-closed shape for the hunt scope.
        for scope in ("l2-only", None, "L1-ONLY", "off"):
            with self.subTest(hunt_scope=scope), self.assertRaisesRegex(
                    ValueError, "registered hunt scope"):
                loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1",
                                  "v1", scope)
        # R18-B6 (2026-09-07): the fifth, sixth and seventh registered laws,
        # same fail-closed shape again.  dry-v1 is deliberately unregistered.
        self.assertEqual(loot.SWEEPS, ("off", "sweep-v1"))
        self.assertEqual(loot.IDENTIFIES, ("off", "cain-v1"))
        self.assertEqual(loot.WEAPON_UPGRADES, ("off", "smith-v1"))
        for law, index, message in (("sweep-v2", 5, "registered sweep law"),
                                    (None, 5, "registered sweep law"),
                                    ("SWEEP-V1", 5, "registered sweep law"),
                                    ("cain-v2", 6, "registered identify law"),
                                    (None, 6, "registered identify law"),
                                    ("sweep-v1", 6, "registered identify law"),
                                    ("dry-v1", 7, "registered weapon upgrade law"),
                                    (None, 7, "registered weapon upgrade law"),
                                    ("smith-v2", 7, "registered weapon upgrade law")):
            positional = ["completion-l2-r18c", "coach-v03", "retreat-v1",
                          "v1", "all", "off", "off", "off"]
            positional[index] = law
            with self.subTest(law=law, index=index), self.assertRaisesRegex(
                    ValueError, message):
                loot.target_world(*positional)

    def test_the_world_writes_none_for_the_legacy_escrow_ruler(self):
        world = loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1", "v1")
        self.assertEqual(set(world), WORLD_KEYS)
        self.assertIsNone(world["worker_descend_escrow_readiness_table"])
        self.assertEqual(world["worker_time_protocol"], "completion-l2-r18c")
        self.assertEqual(world["resource_readiness_law"], "coach-v03")
        self.assertEqual(world["resource_retreat"], "retreat-v1")
        v2 = loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1", "v2")
        self.assertEqual(v2["worker_descend_escrow_readiness_table"], "v2")
        # R18-M2 (2026-09-07): "all" is the frozen scope and is written as None,
        # exactly like the v1 escrow ruler; only l1-only is literal.
        self.assertIsNone(world["hunt_scope"])
        l1 = loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1",
                               "v1", "l1-only")
        self.assertEqual(l1["hunt_scope"], "l1-only")
        self.assertEqual(set(l1), WORLD_KEYS)
        # R18-B6 (2026-09-07): "off" is the frozen value of each new law and is
        # written as None; only the versioned law is literal.
        for key in ("resource_sweep", "resource_identify", "resource_weapon_upgrade"):
            with self.subTest(key=key):
                self.assertIsNone(world[key])
        full = loot.target_world("completion-l2-r18c", "coach-v03", "retreat-v1",
                                 "v2", "l1-only", "sweep-v1", "cain-v1", "smith-v1")
        self.assertEqual(set(full), WORLD_KEYS)
        self.assertEqual(full["resource_sweep"], "sweep-v1")
        self.assertEqual(full["resource_identify"], "cain-v1")
        self.assertEqual(full["resource_weapon_upgrade"], "smith-v1")

    def test_the_allowed_contract_vocabulary_is_the_earned_set_plus_the_world(self):
        self.assertEqual(set(loot.ALLOWED_CONTRACT_KEYS),
                         FROZEN_EARNED_ALLOWED_CONTRACT_KEYS | WORLD_KEYS)
        self.assertEqual(set(loot.WORLD_KEYS), WORLD_KEYS)
        self.assertEqual(loot.PARENT_SHA256, legacy.PARENT_SHA256)
        self.assertEqual(loot.PARENT_CONTRACT_SHA256, legacy.PARENT_CONTRACT_SHA256)
        self.assertEqual(loot.PARENT_STEPS, legacy.PARENT_STEPS)


class FrozenSchemaOneTests(unittest.TestCase):
    """schema/1 is untouched: names, vocabulary and receipt bytes."""

    def test_the_frozen_operation_names_are_byte_identical(self):
        self.assertEqual(legacy.SCHEMA, FROZEN_SCHEMA)
        for policy, name in FROZEN_OPERATIONS.items():
            with self.subTest(policy=policy):
                self.assertEqual(legacy.OPERATIONS[policy], name)
        for (policy, recovery), name in FROZEN_OPERATION_FOR.items():
            with self.subTest(policy=policy, recovery=recovery):
                self.assertEqual(legacy.operation_for(policy, recovery), name)
        self.assertEqual(legacy.operation_for("sustain-v6", "adjacent-v1",
                                              EARNED_DIVE_SUFFIX_SCOPE),
                         FROZEN_EARNED_OPERATION)
        self.assertEqual(legacy.operation_for("sustain-v6", "adjacent-v1",
                                              EARNED_DIVE_SUFFIX_SCOPE, 24.0),
                         FROZEN_DEPTH_SIGNAL_OPERATION)
        self.assertEqual(legacy.DEPTH_SIGNAL_OPERATION, FROZEN_DEPTH_SIGNAL_OPERATION)
        # R18-B3 reserved the schema/1 loot name and refuses to mint it; R18-B3b
        # does not resurrect it, it registers separate schema/2 names instead.
        self.assertEqual(legacy.OPERATIONS["sustain-loot-v1"],
                         "r16-to-sustain-loot-v1-weights-only-v1")
        with self.assertRaisesRegex(ValueError, "completion-l2 clock in the migration"):
            legacy.operation_for("sustain-loot-v1", "adjacent-v1", EARNED_DIVE_SUFFIX_SCOPE)

    def test_the_frozen_contract_vocabulary_is_byte_identical(self):
        self.assertEqual(set(legacy.BASE_ALLOWED_CONTRACT_KEYS),
                         FROZEN_BASE_ALLOWED_CONTRACT_KEYS)
        self.assertEqual(set(legacy.ALLOWED_CONTRACT_KEYS), FROZEN_ALLOWED_CONTRACT_KEYS)
        self.assertEqual(set(legacy.EARNED_ALLOWED_CONTRACT_KEYS),
                         FROZEN_EARNED_ALLOWED_CONTRACT_KEYS)


@unittest.skipUnless(PARENT.is_file(), "registered R16 fixture is unavailable")
class LootTargetAndReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        payload = PARENT.read_bytes()
        if hashlib.sha256(payload).hexdigest() != legacy.PARENT_SHA256:
            raise RuntimeError("Registered R16 test fixture SHA changed")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            cls.parent_data = json.loads(archive.read("data"))
        cls.source = cls.parent_data["diablogym_contract"]
        cls.prefix = _prefix()
        cls.target = loot.target_contract(cls.source, IMPL, cls.prefix,
            worker_time_protocol="completion-l2-r18c",
            worker_descend_escrow_readiness_table="v2")
        cls.receipt = loot.make_receipt(cls.parent_data, cls.target, POLICY_SHA)

    # ---- the frozen schema/1 receipts, rebuilt from the same parent ---------
    def test_the_schema_one_targets_and_receipts_are_byte_identical(self):
        cases = {
            "sustain-v6-plain": dict(service_policy="sustain-v6"),
            "sustain-v6-adjacent": dict(service_policy="sustain-v6",
                                        dive_blocker_recovery="adjacent-v1"),
            "sustain-v6-earned": dict(service_policy="sustain-v6",
                                      dive_blocker_recovery="adjacent-v1",
                                      worker_learning_window_scope=EARNED_DIVE_SUFFIX_SCOPE,
                                      worker_prefix=_prefix()),
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                target = legacy.target_contract(self.source, IMPL, **kwargs)
                receipt = legacy.make_receipt(self.parent_data, target, POLICY_SHA)
                self.assertEqual(legacy.json_sha256(target), FROZEN_TARGET_SHA256[name])
                self.assertEqual(legacy.json_sha256(receipt), FROZEN_RECEIPT_SHA256[name])
                self.assertEqual(receipt["schema"], FROZEN_SCHEMA)
                legacy.validate_inherited_receipt(receipt, target)

    def test_the_prefix_recipe_is_unchanged(self):
        self.assertEqual(self.prefix["source_sha256"], legacy.PARENT_SHA256)
        self.assertEqual(self.prefix["version"], "earned-dive-suffix-prefix-v1")
        self.assertEqual(self.prefix["max_attempts"], 1000)
        self.assertEqual(self.prefix["max_microsteps"], 100_000_000)

    # ---- the new target contract -------------------------------------------
    def test_the_target_changes_exactly_the_registered_world_keys(self):
        changed = {key for key in set(self.source) | set(self.target)
                   if self.source.get(key) != self.target.get(key)}
        self.assertEqual(changed,
                         MIGRATED_KEYS | {"worker_descend_escrow_readiness_table"})
        self.assertTrue(changed <= set(loot.ALLOWED_CONTRACT_KEYS))
        self.assertEqual(self.target["resource_service_recipe"]["version"],
                         "l1-two-trip-loot-economy-v1-r18c")
        self.assertEqual(self.target["resource_service_recipe"]["time_protocol"],
                         "completion-l2-r18c")
        self.assertEqual(self.target["worker_time_recipe"],
                         loot.time_recipe("completion-l2-r18c"))
        self.assertEqual(self.target["worker_learning_window_scope"],
                         EARNED_DIVE_SUFFIX_SCOPE)
        self.assertEqual(self.target["worker_prefix"], self.prefix)
        self.assertEqual(self.target["dive_blocker_recovery"], "adjacent-v1")
        self.assertEqual(self.target["num_envs"], self.source["num_envs"])
        self.assertEqual(self.target["max_steps"], 6000)

    def test_the_legacy_escrow_ruler_is_written_as_an_explicit_none(self):
        # R18-B3b (2026-09-07) review round: the v1 ruler changes no value, but
        # the KEY must be present with None, because _training_contract always
        # writes it and json_sha256 is key-presence sensitive.
        target = loot.target_contract(self.source, IMPL, self.prefix,
            worker_time_protocol="completion-l2-r18c")
        changed = {key for key in set(self.source) | set(target)
                   if self.source.get(key) != target.get(key)}
        self.assertEqual(changed, MIGRATED_KEYS)
        self.assertIn("worker_descend_escrow_readiness_table", target)
        self.assertIsNone(target["worker_descend_escrow_readiness_table"])
        self.assertNotIn("worker_descend_escrow_readiness_table", self.source)

    def test_the_two_clocks_are_two_different_targets_and_receipts(self):
        other = loot.target_contract(self.source, IMPL, self.prefix,
            worker_time_protocol="completion-l2-v1",
            worker_descend_escrow_readiness_table="v2")
        self.assertEqual({key for key in set(other) | set(self.target)
                          if other.get(key) != self.target.get(key)},
                         {"resource_service_recipe", "worker_time_protocol",
                          "worker_time_recipe"})
        receipt = loot.make_receipt(self.parent_data, other, POLICY_SHA)
        self.assertEqual(receipt["operation"], NEW_OPERATIONS["completion-l2-v1"])
        self.assertNotEqual(legacy.json_sha256(receipt),
                            legacy.json_sha256(self.receipt))
        loot.validate_inherited_receipt(receipt, other)

    # ---- the receipt shape --------------------------------------------------
    def test_the_receipt_pins_the_parent_the_world_and_the_implementation(self):
        self.assertEqual(set(self.receipt), RECEIPT_KEYS)
        self.assertEqual(self.receipt["schema"], "diablogym-resource-warm-start/2")
        self.assertEqual(self.receipt["operation"], NEW_OPERATIONS["completion-l2-r18c"])
        self.assertEqual(self.receipt["parent_checkpoint_sha256"], legacy.PARENT_SHA256)
        self.assertEqual(self.receipt["parent_contract_sha256"],
                         legacy.PARENT_CONTRACT_SHA256)
        self.assertEqual(self.receipt["parent_num_timesteps"], 4089856)
        self.assertEqual(self.receipt["target_contract_sha256"],
                         legacy.json_sha256(self.target))
        self.assertEqual(self.receipt["target_implementation_sha256"], IMPL)
        self.assertEqual(self.receipt["policy_sha256"], POLICY_SHA)
        self.assertEqual(self.receipt["resource_service_recipe"],
                         self.target["resource_service_recipe"])
        self.assertEqual(self.receipt["worker_time_protocol"], "completion-l2-r18c")
        self.assertEqual(self.receipt["worker_time_recipe"],
                         loot.time_recipe("completion-l2-r18c"))
        self.assertEqual(self.receipt["worker_learning_window_scope"],
                         EARNED_DIVE_SUFFIX_SCOPE)
        self.assertEqual(self.receipt["worker_prefix"], self.prefix)
        self.assertEqual(self.receipt["dive_blocker_recovery_recipe"],
                         eval_contract.dive_blocker_recovery_recipe("adjacent-v1"))
        self.assertEqual(self.receipt["target_world"], {
            "worker_time_protocol": "completion-l2-r18c",
            "worker_time_recipe": loot.time_recipe("completion-l2-r18c"),
            "resource_readiness_law": "coach-v03",
            "resource_retreat": "retreat-v1",
            "worker_descend_escrow_readiness_table": "v2",
            # R18-M2 (2026-09-07): the registered world now names the hunt scope;
            # this fixture mints the frozen "all", which is written as None.
            "hunt_scope": None,
            # R18-B6 (2026-09-07): three more registered laws, all minted at
            # their frozen "off", all written as None.
            "resource_sweep": None,
            "resource_identify": None,
            "resource_weapon_upgrade": None})
        self.assertEqual(self.receipt["historical_critic_warmup"],
                         loot.PARENT_HISTORICAL_CRITIC_WARMUP)
        self.assertEqual(legacy.json_sha256(self.receipt["parent_resume_lineage"]),
                         loot.PARENT_LINEAGE_SHA256)
        for key, value in (("optimizer_state", "reset-empty"),
                           ("current_world_training_steps", 0),
                           ("historical_warmup_is_current_training", False),
                           ("exact_trajectory_continuation", False),
                           ("environment_state_mode", "fresh-normal-l1-no-snapshot"),
                           ("publication_status", "INITIALIZATION_ONLY_NOT_TRAINED")):
            with self.subTest(key=key):
                self.assertEqual(self.receipt[key], value)
        loot.validate_inherited_receipt(self.receipt, self.target)
        legacy.validate_inherited_receipt(self.receipt, self.target)

    # ---- fail-closed --------------------------------------------------------
    def test_a_clock_mismatch_between_receipt_and_target_is_refused(self):
        other = loot.target_contract(self.source, IMPL, self.prefix,
            worker_time_protocol="completion-l2-v1",
            worker_descend_escrow_readiness_table="v2")
        with self.assertRaises(ValueError):
            loot.validate_inherited_receipt(self.receipt, other)
        with self.assertRaises(ValueError):
            legacy.validate_inherited_receipt(self.receipt, other)
        # A receipt whose own clock claim is edited to match the other target.
        forged = deepcopy(self.receipt)
        forged["worker_time_protocol"] = "completion-l2-v1"
        forged["worker_time_recipe"] = loot.time_recipe("completion-l2-v1")
        forged["target_world"]["worker_time_protocol"] = "completion-l2-v1"
        forged["target_world"]["worker_time_recipe"] = loot.time_recipe("completion-l2-v1")
        forged["operation"] = NEW_OPERATIONS["completion-l2-v1"]
        forged["target_contract_sha256"] = legacy.json_sha256(other)
        with self.assertRaises(ValueError):
            loot.validate_inherited_receipt(forged, other)

    def test_a_loot_receipt_cannot_be_consumed_by_a_sustain_v6_world(self):
        v6 = legacy.target_contract(self.source, IMPL, "sustain-v6", "adjacent-v1",
                                    EARNED_DIVE_SUFFIX_SCOPE, self.prefix)
        for validator in (loot.validate_inherited_receipt,
                          legacy.validate_inherited_receipt):
            with self.subTest(validator=validator.__module__), self.assertRaises(ValueError):
                validator(self.receipt, v6)
        v6_receipt = legacy.make_receipt(self.parent_data, v6, POLICY_SHA)
        for validator in (loot.validate_inherited_receipt,
                          legacy.validate_inherited_receipt):
            with self.subTest(validator=validator.__module__), self.assertRaises(ValueError):
                validator(v6_receipt, self.target)
        relabelled = {**v6_receipt, "schema": loot.SCHEMA,
                      "operation": NEW_OPERATIONS["completion-l2-r18c"]}
        with self.assertRaises(ValueError):
            legacy.validate_inherited_receipt(relabelled, self.target)

    def test_a_parent_sha_mismatch_is_refused(self):
        for key in ("parent_checkpoint_sha256", "parent_contract_sha256",
                    "parent_num_timesteps", "schema", "operation"):
            forged = deepcopy(self.receipt)
            forged[key] = ("c" * 64) if isinstance(forged[key], str) else 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                loot.validate_inherited_receipt(forged, self.target)
        # policy_sha256 is deliberately NOT self-evident inside the receipt: it
        # is bound to the actual tensors by validate_inherited_runtime and by
        # load_initialization's probe (LootMintTests covers that binding).  A
        # malformed one is still refused here.
        for bad in ("c" * 63, "C" * 64, None, 1):
            forged = deepcopy(self.receipt)
            forged["policy_sha256"] = bad
            with self.subTest(policy_sha256=bad), self.assertRaisesRegex(
                    ValueError, "policy tensor SHA256"):
                loot.validate_inherited_receipt(forged, self.target)
        forged = deepcopy(self.receipt)
        forged["source_contract"] = {**self.source, "max_steps": 3000}
        with self.assertRaises(ValueError):
            loot.validate_inherited_receipt(forged, self.target)
        forged = deepcopy(self.receipt)
        forged["historical_critic_warmup"]["_critic_warmup_optimizer_steps_completed"] = 641
        with self.assertRaisesRegex(ValueError, "critic warmup evidence differs"):
            loot.validate_inherited_receipt(forged, self.target)
        forged = deepcopy(self.receipt)
        forged["parent_resume_lineage"]["generation"] = 3
        with self.assertRaisesRegex(ValueError, "resume lineage differs"):
            loot.validate_inherited_receipt(forged, self.target)
        with self.assertRaisesRegex(ValueError, "exact registered R16 parent SHA"):
            loot.migrate(str(PARENT), parent_sha256="d" * 64,
                         output_dir="/nonexistent/should-not-be-created",
                         implementation_sha256=IMPL, seed=1,
                         worker_time_protocol="completion-l2-r18c", worker_prefix=self.prefix)

    def test_a_tampered_recipe_is_refused(self):
        tampered_targets = [
            {**self.target, "resource_service_recipe": {
                **self.target["resource_service_recipe"],
                "version": "l1-two-trip-loot-economy-v1"}},
            {**self.target, "resource_service_recipe": {
                **self.target["resource_service_recipe"], "max_town_trips": 3}},
            {**self.target, "worker_time_recipe": {
                **self.target["worker_time_recipe"], "followup_microsteps": 1800}},
            {**self.target, "resource_readiness_law": "veto-v1"},
            {**self.target, "resource_retreat": "off"},
            {**self.target, "resource_purchase_mode": "minimal"},
            {**self.target, "worker_learning_window_scope": "farm-dive-v1"},
            {**self.target, "worker_depth_shaping_unit": 24.0},
            {**self.target, "max_steps": 10000},
            {**self.target, "num_envs": 2},
        ]
        for index, target in enumerate(tampered_targets):
            with self.subTest(case=index), self.assertRaises(ValueError):
                loot.validate_target_contract(self.source, target, IMPL)
            with self.subTest(case=index, stage="receipt"), self.assertRaises(ValueError):
                loot.validate_inherited_receipt(self.receipt, target)
        # A different prefix budget is a legal target on its own, but never the
        # world this frozen receipt was minted for.
        with self.assertRaises(ValueError):
            loot.validate_inherited_receipt(
                self.receipt,
                {**self.target, "worker_prefix": _prefix(attempts=999)})
        for key in ("resource_service_recipe", "worker_time_recipe", "worker_prefix",
                    "target_world", "dive_blocker_recovery_recipe"):
            forged = deepcopy(self.receipt)
            forged[key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                loot.validate_inherited_receipt(forged, self.target)
        forged = deepcopy(self.receipt)
        forged["target_implementation_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            loot.validate_inherited_receipt(forged, self.target)

    def test_an_unregistered_implementation_or_prefix_is_refused(self):
        for bad in ("A" * 64, "a" * 63, None, 1):
            with self.subTest(impl=bad), self.assertRaises(ValueError):
                loot.target_contract(self.source, bad, self.prefix,
                                     worker_time_protocol="completion-l2-r18c")
        with self.assertRaisesRegex(ValueError, "exact registered R16 prefix"):
            loot.target_contract(self.source, IMPL,
                                 {**self.prefix, "source_sha256": "f" * 64},
                                 worker_time_protocol="completion-l2-r18c")
        with self.assertRaisesRegex(ValueError, "exact prefix recipe"):
            loot.target_contract(self.source, IMPL, None,
                                 worker_time_protocol="completion-l2-r18c")


@unittest.skipUnless(PARENT.is_file(), "registered R16 fixture is unavailable")
class LootMintTests(unittest.TestCase):
    """One real mint: fresh-dir-only, manifest shape and the reload path."""

    def test_the_cli_mints_into_a_fresh_directory_only(self):
        import torch
        import train_ppo as training
        torch.set_num_threads(1)
        prefix = _prefix()
        with tempfile.TemporaryDirectory() as directory, patch.object(
                training, "_implementation_bundle_sha256", return_value=IMPL):
            output = Path(directory) / "loot-initialization"
            manifest = loot.migrate(str(PARENT), parent_sha256=legacy.PARENT_SHA256,
                output_dir=output, implementation_sha256=IMPL, seed=2164000,
                worker_time_protocol="completion-l2-r18c", worker_prefix=prefix,
                worker_descend_escrow_readiness_table="v2")
            self.assertEqual(manifest["schema"], "diablogym-resource-warm-start/2")
            self.assertEqual(manifest["operation"], NEW_OPERATIONS["completion-l2-r18c"])
            self.assertEqual(manifest["status"], "INITIALIZATION_ONLY_NOT_TRAINED")
            self.assertIs(manifest["ordinary_resume_eligible"], False)
            self.assertIs(manifest["trained_in_target_world"], False)
            self.assertEqual(manifest["parent_sha256"], legacy.PARENT_SHA256)
            self.assertEqual(manifest["receipt"]["target_contract_sha256"],
                             legacy.json_sha256(manifest["target_contract"]))
            # The shared capture/reload path accepts the new schema and revalidates.
            payload, data, captured = legacy.capture_initialization(
                output / "manifest.json", IMPL)
            self.assertEqual(captured, manifest)
            self.assertEqual(data["diablogym_contract"], manifest["target_contract"])
            self.assertEqual(data["_resource_warm_start_receipt"], manifest["receipt"])
            self.assertEqual(data["num_timesteps"], 0)
            model = legacy.load_initialization(payload, manifest)
            self.assertEqual(legacy.policy_sha256(model), manifest["policy_sha256"])
            self.assertEqual(model._resume_lineage["operation"], manifest["operation"])
            self.assertEqual(model._resume_lineage["immediate_parent_num_timesteps"],
                             legacy.PARENT_STEPS)
            # R18-B3b (2026-09-07) review round: the manifest may name either
            # registered schema, never one the receipt in the checkpoint denies.
            relabelled = Path(directory) / "relabelled"
            relabelled.mkdir()
            (relabelled / "model_warm_start.zip").write_bytes(
                (output / "model_warm_start.zip").read_bytes())
            forged = deepcopy(manifest)
            forged["schema"] = legacy.SCHEMA
            (relabelled / "manifest.json").write_text(
                json.dumps(forged, indent=2, sort_keys=True, ensure_ascii=False))
            with self.assertRaisesRegex(ValueError, "false lineage/training claims"):
                legacy.capture_initialization(relabelled / "manifest.json", IMPL)
            # A second mint into the same directory is refused before any write.
            with self.assertRaisesRegex(ValueError, "output directory already exists"):
                loot.migrate(str(PARENT), parent_sha256=legacy.PARENT_SHA256,
                    output_dir=output, implementation_sha256=IMPL, seed=2164000,
                    worker_time_protocol="completion-l2-r18c", worker_prefix=prefix,
                    worker_descend_escrow_readiness_table="v2")
            # An unregistered world never reaches the filesystem.
            fresh = Path(directory) / "never-created"
            with self.assertRaises(ValueError):
                loot.migrate(str(PARENT), parent_sha256=legacy.PARENT_SHA256,
                    output_dir=fresh, implementation_sha256=IMPL, seed=2164000,
                    worker_time_protocol="completion-l2-r18c", worker_prefix=prefix,
                    resource_readiness_law="veto-v1")
            self.assertFalse(fresh.exists())
            # A drifted implementation is refused, again without a directory.
            other = Path(directory) / "also-never-created"
            with patch.object(training, "_implementation_bundle_sha256",
                              return_value="f" * 64):
                with self.assertRaisesRegex(ValueError, "implementation changed"):
                    loot.migrate(str(PARENT), parent_sha256=legacy.PARENT_SHA256,
                        output_dir=other, implementation_sha256=IMPL, seed=2164000,
                        worker_time_protocol="completion-l2-r18c", worker_prefix=prefix)
            self.assertFalse(other.exists())


class TrainingConsumerTests(unittest.TestCase):
    """train_ppo's argument law and resume identity for the loot warm start."""

    @staticmethod
    def _args(**overrides):
        from types import SimpleNamespace
        base = dict(resource_warm_start=str(Path(__file__).resolve()), worker=True, algo="mppo",
                    device="cpu", seed=22, resume_from=None, bc_init=None,
                    teacher_override=None, reset_optimizer=False,
                    reset_worker_critic=False, allow_legacy_resume=False,
                    allow_manager_change=False, allow_environment_restart_resume=False,
                    freeze_policy_steps=0, deep_start_curriculum=None,
                    dry_curriculum_schedule=None, skip_dry=False, distill_beta=0.0,
                    bc_aux_demos=None, bc_aux_lambda=0.0,
                    resource_protocol="l2-town-v1", resource_purchase_mode="full",
                    resource_service_policy="sustain-loot-v1",
                    worker_time_protocol="completion-l2-r18c",
                    worker_learning_window_scope="earned-dive-suffix-v1",
                    dive_blocker_recovery="adjacent-v1")
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_the_registered_loot_arm_passes_the_warm_start_argument_law(self):
        import train_ppo as training
        training._validate_resource_warm_start_args(self._args())
        for policy in ("sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6"):
            with self.subTest(policy=policy):
                training._validate_resource_warm_start_args(self._args(
                    resource_service_policy=policy, worker_time_protocol="legacy",
                    worker_learning_window_scope="farm-dive-v1",
                    dive_blocker_recovery="off"))

    def test_the_warm_start_argument_law_fails_closed(self):
        import train_ppo as training
        cases = {
            "legacy clock": dict(worker_time_protocol="legacy"),
            "farm-only scope": dict(worker_learning_window_scope="farm-only"),
            "farm-dive scope": dict(worker_learning_window_scope="farm-dive-v1"),
            "no dive recovery": dict(dive_blocker_recovery="off"),
        }
        for name, overrides in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(
                    ValueError, "sustain-loot-v1 warm start requires"):
                training._validate_resource_warm_start_args(self._args(**overrides))
        for policy in ("legacy-v1", "sustain-loot", "sustain-loot-v2"):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                    ValueError, "resource warm-start requires an explicit"):
                training._validate_resource_warm_start_args(
                    self._args(resource_service_policy=policy))
        with self.assertRaisesRegex(ValueError, "resource warm-start requires an explicit"):
            training._validate_resource_warm_start_args(self._args(resource_protocol="off"))
        with self.assertRaisesRegex(ValueError, "cannot combine with resume"):
            training._validate_resource_warm_start_args(self._args(resume_from="x.zip"))
        # No warm start at all: the law is silent, exactly as before.
        training._validate_resource_warm_start_args(self._args(resource_warm_start=None))

    def test_the_new_module_is_part_of_the_implementation_bundle(self):
        import train_ppo as training
        self.assertIn("train/migrate_loot_candidate.py",
                      training._IMPLEMENTATION_SOURCE_FILES)
        self.assertIn("train/migrate_resource_candidate.py",
                      training._IMPLEMENTATION_SOURCE_FILES)
        self.assertIn("train/migrate_completion_candidate.py",
                      training._IMPLEMENTATION_SOURCE_FILES)

    def test_the_first_update_diagnostic_accepts_both_registered_schemas(self):
        import training_diagnostics
        source = Path(training_diagnostics.__file__).read_text()
        self.assertIn('"diablogym-resource-warm-start/1"', source)
        self.assertIn('"diablogym-resource-warm-start/2"', source)


@unittest.skipUnless(PARENT.is_file(), "registered R16 fixture is unavailable")
class ResumeIdentityTests(unittest.TestCase):
    """The minted target contract is exactly what training would write."""

    @classmethod
    def setUpClass(cls):
        with zipfile.ZipFile(io.BytesIO(PARENT.read_bytes())) as archive:
            cls.source = json.loads(archive.read("data"))["diablogym_contract"]
        cls.prefix = _prefix()
        cls.target = loot.target_contract(cls.source, IMPL, cls.prefix,
            worker_time_protocol="completion-l2-r18c",
            worker_descend_escrow_readiness_table="v2")

    def test_the_target_contract_resumes_against_itself(self):
        import train_ppo as training
        training._validate_resume_contract(self.target, deepcopy(self.target))

    def test_a_clock_or_law_change_is_not_an_ordinary_resume(self):
        import train_ppo as training
        other = loot.target_contract(self.source, IMPL, self.prefix,
            worker_time_protocol="completion-l2-v1",
            worker_descend_escrow_readiness_table="v2")
        with self.assertRaisesRegex(ValueError, "worker time protocol changes"):
            training._validate_resume_contract(self.target, other,
                                               allow_environment_restart=True)
        v6 = legacy.target_contract(self.source, IMPL, "sustain-v6", "adjacent-v1",
                                    EARNED_DIVE_SUFFIX_SCOPE, self.prefix)
        with self.assertRaises(ValueError):
            training._validate_resume_contract(self.target, v6,
                                               allow_environment_restart=True)
        for key, value in (("resource_readiness_law", "veto-v1"),
                           ("resource_retreat", None),
                           ("worker_descend_escrow_readiness_table", None)):
            drifted = {**self.target}
            if value is None:
                drifted.pop(key, None)
            else:
                drifted[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                training._validate_resume_contract(self.target, drifted)

    def test_the_loot_service_recipe_is_protected_at_resume(self):
        import train_ppo as training
        forged = deepcopy(self.target)
        forged["resource_service_recipe"] = {**forged["resource_service_recipe"],
                                             "version": "l1-two-trip-loot-economy-v1"}
        with self.assertRaisesRegex(ValueError, "mismatched native armor"):
            training._validate_resume_contract(forged, deepcopy(forged))


@unittest.skipUnless(PARENT.is_file(), "registered R16 fixture is unavailable")
class LiveTrainingContractIdentityTests(unittest.TestCase):
    """R18-B3b (2026-09-07) review round: the minted target contract is compared
    against a real ``train_ppo._training_contract`` built from the launch argv,
    for every registered clock x escrow ruler -- not against a copy of itself.

    The v1 ruler arm was mintable but unconsumable before this round: the
    migrated contract omitted ``worker_descend_escrow_readiness_table`` while
    training always writes it as None, and ``json_sha256`` counts keys.
    """

    @classmethod
    def setUpClass(cls):
        import torch
        from leashed_ppo import LeashedMaskablePPO
        torch.set_num_threads(1)
        payload = PARENT.read_bytes()
        if hashlib.sha256(payload).hexdigest() != legacy.PARENT_SHA256:
            raise RuntimeError("Registered R16 test fixture SHA changed")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            cls.parent_data = json.loads(archive.read("data"))
        cls.model = LeashedMaskablePPO.load(io.BytesIO(payload), device="cpu",
                                            teacher_path=None, teacher_sha256=None)
        cls.source = cls.parent_data["diablogym_contract"]
        cls.prefix = _prefix()

    @staticmethod
    def _launch_argv(time_protocol, escrow_table, hunt_scope="all"):
        """B3B-LAUNCH-COMMAND.sh's train_ppo invocation, verbatim.

        ``--resource-warm-start`` only has to be truthy here: the contract asks
        whether a warm start is in play, and the file itself is read by _main,
        which this test stops at ``parse_args``.
        """
        return [
            "--worker", "--algo", "mppo", "--gamma", "1.0", "--max-steps", "6000",
            "--num-envs", "4", "--n-steps", "512", "--total-steps", "4096",
            "--seed", "22", "--device", "cpu",
            "--run-name", "r18-b3b-contract-probe", "--artifact-scope", "candidate",
            "--resource-warm-start", str(Path(__file__).resolve()),
            "--gradient-clip-mode", "separate-root-context-critic-v2",
            "--manager-heuristic", "readiness-v3",
            "--worker-action14-logit-bonus", "2.5",
            "--worker-dive-action11-logit-bonus", "2.0",
            "--worker-potion-action13-logit-bonus", "2.0",
            "--worker-descend-escrow-fraction", "0.5",
            "--worker-descend-escrow-power", "1.6",
            "--worker-descend-escrow-readiness-gate",
            "--worker-descend-escrow-readiness-table", escrow_table,
            "--worker-hp-loss-price", "0.1",
            "--worker-potion-pickup-bonus", "2.0",
            "--worker-no-progress-timeout-credit", "zero",
            "--explore-global-hunt",
            # R18-M2 (2026-09-07): "all" is the frozen default and adds no flag,
            # so every pre-existing subtest keeps its exact argv.
            *(("--hunt-scope", hunt_scope) if hunt_scope != "all" else ()),
            "--farm-scene-cap", "3600",
            "--reset-layer-clock-on-window",
            "--ckpt-every-steps", "63488", "--sentinel-every", "63488",
            "--lr", "0.0001", "--ent-coef", "0.005", "--target-kl", "0.01",
            "--drink-sovereignty",
            "--worker-policy-observation-view", "dual-v4-asymmetric-v3",
            "--worker-fast-forward-reward-credit", "terminal-death-only",
            "--worker-additional-terminal-death-cost", "0.0",
            "--reward-economy", "v4",
            "--worker-learning-window-scope", "earned-dive-suffix-v1",
            "--worker-prefix-model", str(PARENT),
            "--worker-prefix-max-attempts", "1000",
            "--worker-prefix-max-microsteps", "100000000",
            "--worker-time-protocol", time_protocol,
            "--resource-protocol", "l2-town-v1",
            "--resource-purchase-mode", "full",
            "--resource-service-policy", "sustain-loot-v1",
            "--resource-readiness-law", "coach-v03",
            "--resource-retreat", "retreat-v1",
            "--dive-blocker-recovery", "adjacent-v1",
        ]

    @staticmethod
    def _parse(argv):
        """train_ppo's own argparse, stopped the instant it has the namespace."""
        import argparse
        import train_ppo as training

        class _Captured(Exception):
            def __init__(self, namespace):
                self.namespace = namespace

        original = argparse.ArgumentParser.parse_args

        def _stop(self, *args, **kwargs):
            raise _Captured(original(self, *args, **kwargs))

        saved = sys.argv
        argparse.ArgumentParser.parse_args = _stop
        sys.argv = ["train_ppo.py"] + list(argv)
        try:
            training._main(training._TrainingResources())
        except _Captured as captured:
            return captured.namespace
        finally:
            argparse.ArgumentParser.parse_args = original
            sys.argv = saved
        raise AssertionError("train_ppo._main never reached parse_args")

    def test_every_registered_clock_and_ruler_equals_the_training_contract(self):
        import train_ppo as training
        for clock in sorted(loot.OPERATIONS):
          for table in loot.ESCROW_READINESS_TABLES:
            # R18-M2 (2026-09-07): the third dimension.  The l1-only arm is the
            # one the R18-M2 smoke launches, and before hunt_scope was registered
            # it failed here with {"hunt_scope": (None, "l1-only")} -- which is
            # exactly how train_ppo._validate_resume_contract refused the live run.
            for scope in loot.HUNT_SCOPES:
                with self.subTest(clock=clock, escrow_table=table, hunt_scope=scope):
                    args = self._parse(self._launch_argv(clock, table, scope))
                    self.assertEqual(args.worker_descend_escrow_readiness_table, table)
                    self.assertEqual(args.worker_time_protocol, clock)
                    self.assertEqual(getattr(args, "hunt_scope", "all"), scope)
                    batch = training._select_batch_size(args.n_steps, args.num_envs)
                    current = training._training_contract(
                        args, self.model, batch, implementation_sha256=IMPL)
                    target = loot.target_contract(self.source, IMPL, self.prefix,
                        worker_time_protocol=clock,
                        resource_readiness_law="coach-v03",
                        resource_retreat="retreat-v1",
                        worker_descend_escrow_readiness_table=table,
                        hunt_scope=scope)
                    # Key sets first: json_sha256 (and therefore the receipt) is
                    # key-presence sensitive where .get()-based comparators are not.
                    self.assertEqual(set(target), set(current))
                    self.assertEqual(target, current)
                    self.assertEqual(loot.json_sha256(target),
                                     loot.json_sha256(current))
                    self.assertEqual(
                        current["worker_descend_escrow_readiness_table"],
                        "v2" if table == "v2" else None)
                    self.assertEqual(current["hunt_scope"],
                                     "l1-only" if scope == "l1-only" else None)
                    # R18-B5 also emits resource_portal unconditionally and it is
                    # deliberately NOT registered: the frozen None must match.
                    self.assertIsNone(current["resource_portal"])
                    # Both consumption gates train_ppo runs on a warm start.
                    training._validate_resume_contract(target, current)
                    receipt = loot.make_receipt(self.parent_data, target, POLICY_SHA)
                    loot.validate_inherited_receipt(receipt, current)


if __name__ == "__main__":
    unittest.main()
