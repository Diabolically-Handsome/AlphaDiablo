"""Exact R16 parent -> sustain-loot-v1 earned-suffix weight initialization under an
explicit completion-l2 clock; never a resumed trajectory.

R18-B3b (2026-09-07).  R18-B3 registered the loot service recipe for training but
left ``migrate_resource_candidate.operation_for`` refusing ``sustain-loot-v1``:
the schema/1 migration vocabulary has no clock key, and a frozen receipt that
cannot name the clock it was minted under says something nobody verified.  The
loot recipe is versioned per clock (``eval_contract.LOOT_SERVICE_RECIPE_VERSIONS``),
so the clock is part of this migration's identity, not decoration.

This module is that missing schema version.  It is a NEW operation family with a
NEW schema string (``diablogym-resource-warm-start/2``); the schema/1 operations,
receipts and manifests are untouched and remain byte-identical.  Everything that
is not the registered R18-B3b world fails closed here, not later in the engine.

The registered world, all of it pinned in the receipt and re-verified by every
consumer:

* parent  = the exact registered R16 worker (checkpoint sha / contract sha /
  4,089,856 steps / completed critic warmup evidence / resume lineage),
* target  = l2-town-v1 / full / sustain-loot-v1 with the clock-versioned service
  recipe, dive_blocker_recovery adjacent-v1, worker_learning_window_scope
  earned-dive-suffix-v1 with the exact prefix identity, an explicit
  completion-l2 protocol and its exact recipe dict, the readiness law, the
  retreat law, the descend-escrow readiness table, and the implementation sha.

Importing this module or validating JSON metadata does not load a model, and the
clock recipe is read from the standalone stdlib module without executing
``diablogym.__init__`` (no native extension needed for forensics).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import migrate_resource_candidate as legacy

ROOT = Path(__file__).resolve().parents[1]

# The parent is the same registered R16 worker as schema/1; its constants are
# imported, never re-declared, so the two schemas can never drift apart.
PARENT_SHA256 = legacy.PARENT_SHA256
PARENT_CONTRACT_SHA256 = legacy.PARENT_CONTRACT_SHA256
PARENT_STEPS = legacy.PARENT_STEPS
# R18-B3b: the exact registered parent evidence.  schema/1 checked these values
# field by field inside validate_inherited_receipt; here the whole receipt is
# re-derived and compared by digest, so the parent evidence has to be pinned as
# literals or a tampered receipt would round-trip against itself.
PARENT_HISTORICAL_CRITIC_WARMUP = {
    "_critic_warmup_start_timesteps": 3497984,
    "_critic_warmup_until_timesteps": 3514368,
    "_critic_warmup_expected_rollouts": 8,
    "_critic_warmup_rollouts_completed": 8,
    "_critic_warmup_optimizer_steps_completed": 640,
    "_critic_warmup_completed": True,
    "_critic_warmup_actor_sha256":
        "5e870fd86392644dc9c3bcbccee113e8d4ba46e30beecca158ea9bfa175cb7e2",
}
PARENT_LINEAGE_SHA256 = "c272fcb541c51d7a75779708a6d953c377d345b6b195bb0d85b73b8b12fd7c0f"

SCHEMA = legacy.SCHEMA_V2
SERVICE_POLICY = "sustain-loot-v1"
RESOURCE_PROTOCOL = "l2-town-v1"
RESOURCE_PURCHASE_MODE = "full"
DIVE_BLOCKER_RECOVERY = "adjacent-v1"
LEARNING_WINDOW_SCOPE = "earned-dive-suffix-v1"
# One frozen operation name per completion clock.  New names only; the schema/1
# strings (including the reserved r16-to-sustain-loot-v1-*-weights-only-v1) are
# never renamed or reused.
OPERATIONS = {
    "completion-l2-v1": "r16-to-sustain-loot-v1-completion-l2-v1"
                        "-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v2",
    "completion-l2-r18c": "r16-to-sustain-loot-v1-completion-l2-r18c"
                          "-earned-dive-suffix-v1-dive-adjacent-v1-weights-only-v2",
}
# The registered laws of the R18-B3b arm.  A law the chairman has not registered
# for a loot warm start (veto-v1, retreat off, a future coach-v04) is refused
# here rather than silently minted; adding one is a one-line registration plus a
# new test, and never a rename.
READINESS_LAWS = ("coach-v03",)
RETREATS = ("retreat-v1",)
ESCROW_READINESS_TABLES = ("v1", "v2")
# R18-M2 (2026-09-07): R18-B5 turned hunt_scope into a training flag and made
# train_ppo._training_contract emit the key UNCONDITIONALLY ("all" spelled
# None, l1-only literal -- train/train_ppo.py:2739-2741).  It is registered
# here on B3b's own terms so a warm start may actually enter the l1-only
# world; R18-M wrote the key as a frozen None outside the world, which kept
# the default arm consumable but refused the flag itself (the refusal was
# reproduced: "resume 训练/环境契约漂移: {'hunt_scope': (None, 'l1-only')}").
# resource_portal stays UNregistered -- no launch tonight sets it, so it keeps
# the fail-closed setdefault in target_contract() below.
HUNT_SCOPES = ("all", "l1-only")
# R18-B6 (2026-09-07): the three loot-itinerary laws B6 wires into training.
# train_ppo._training_contract emits all three keys UNCONDITIONALLY (off spelled
# None, the versioned law literal), so a warm start that does not register them
# is mintable but refuses to be consumed the moment a launch turns one on --
# exactly the refusal R18-M2 §5.1 reproduced for hunt_scope.  The migration must
# pin the world being entered, so each is registered here on B3b's own terms:
# "off" (the frozen default, written as None) plus the one law that exists.
# dry-v1 is NOT registered: train_ppo refuses it too (it is an observation twin
# whose rows are bit-identical to the control arm).
# R18-B6 (2026-09-07) review round -- MEASURED, so nobody has to guess: adding
# these three keys moves target_contract_sha256 for EVERY schema/2 arm, the
# all-off one included, because json_sha256 is key-presence sensitive (see the
# WORLD_KEYS note below) and target_contract() now writes three more keys even
# when all three are None.  Held at a fixed target_implementation_sha256, the
# R18-M2 4096-step smoke arm moves 5cf839f331d2840425b52bb6ff3e5a8cc0322d4b02
# 657bc63f01c63f75ef8188 -> 5436c29c7c7401c2a62ff82da869b346e0241db4d8566b4fd
# 831b31098e1359d (74 -> 77 keys).  Every pre-B6 schema/2 receipt must therefore
# be re-minted on this tree; validate_inherited_receipt refuses the old ones
# with "Loot warm-start receipt does not pin the world being entered", which is
# the intended fail-closed outcome, not a regression.
SWEEPS = ("off", "sweep-v1")
IDENTIFIES = ("off", "cain-v1")
WEAPON_UPGRADES = ("off", "smith-v1")
# The contract delta this schema adds on top of the schema/1 earned-suffix
# vocabulary.  worker_descend_escrow_readiness_table v1 is the legacy ruler and
# _training_contract writes None for it, so the world dict carries the contract
# value (None), not the CLI spelling.  R18-B3b (2026-09-07) review round: that
# None is WRITTEN into the target contract, key and all -- _training_contract
# emits the key unconditionally and json_sha256 is key-presence sensitive, so an
# omitted key makes the receipt's target_contract_sha256 unmatchable.
WORLD_KEYS = ("worker_time_protocol", "worker_time_recipe", "resource_readiness_law",
              "resource_retreat", "worker_descend_escrow_readiness_table",
              # R18-M2 (2026-09-07): "all" is the frozen default and is written
              # as None, exactly like the v1 escrow ruler above.
              "hunt_scope",
              # R18-B6 (2026-09-07): "off" is the frozen default of each and is
              # written as None, same accounting again.
              "resource_sweep", "resource_identify", "resource_weapon_upgrade")
ALLOWED_CONTRACT_KEYS = legacy.EARNED_ALLOWED_CONTRACT_KEYS | frozenset(WORLD_KEYS)

require = legacy.require
json_sha256 = legacy.json_sha256
policy_sha256 = legacy.policy_sha256
policy_probe = legacy.policy_probe


@lru_cache(maxsize=1)
def _clock_recipes():
    # Import the standalone stdlib recipes without executing diablogym.__init__.
    name = "_diablogym_loot_migration_clock"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "python/diablogym/completion_clock.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.COMPLETION_RECIPES


def time_recipe(time_protocol):
    """The exact registered clock recipe dict; unregistered clocks fail closed."""
    recipes = _clock_recipes()
    require(isinstance(time_protocol, str) and time_protocol in recipes
            and time_protocol in OPERATIONS,
            "sustain-loot-v1 migration requires a registered completion-l2 clock "
            f"({sorted(OPERATIONS)}): {time_protocol!r}")
    return recipes[time_protocol].as_dict()


def operation_for(time_protocol):
    """One registered operation name per clock; nothing else is mintable."""
    time_recipe(time_protocol)
    return OPERATIONS[time_protocol]


def target_world(time_protocol, resource_readiness_law="coach-v03",
                 resource_retreat="retreat-v1",
                 worker_descend_escrow_readiness_table="v1",
                 hunt_scope="all", resource_sweep="off",
                 resource_identify="off", resource_weapon_upgrade="off"):
    """The exact contract delta of the registered arm, as training would write it."""
    recipe = time_recipe(time_protocol)
    require(resource_readiness_law in READINESS_LAWS,
            f"sustain-loot-v1 warm start requires a registered readiness law "
            f"{list(READINESS_LAWS)}: {resource_readiness_law!r}")
    require(resource_retreat in RETREATS,
            f"sustain-loot-v1 warm start requires a registered retreat law "
            f"{list(RETREATS)}: {resource_retreat!r}")
    require(worker_descend_escrow_readiness_table in ESCROW_READINESS_TABLES,
            "sustain-loot-v1 warm start requires descend-escrow readiness table "
            f"{list(ESCROW_READINESS_TABLES)}: {worker_descend_escrow_readiness_table!r}")
    # R18-M2 (2026-09-07): same shape as the three laws above -- an unregistered
    # scope is refused here rather than silently minted.
    require(hunt_scope in HUNT_SCOPES,
            f"sustain-loot-v1 warm start requires a registered hunt scope "
            f"{list(HUNT_SCOPES)}: {hunt_scope!r}")
    # R18-B6 (2026-09-07): same shape again, one require per registered law.
    require(resource_sweep in SWEEPS,
            f"sustain-loot-v1 warm start requires a registered sweep law "
            f"{list(SWEEPS)}: {resource_sweep!r}")
    require(resource_identify in IDENTIFIES,
            f"sustain-loot-v1 warm start requires a registered identify law "
            f"{list(IDENTIFIES)}: {resource_identify!r}")
    require(resource_weapon_upgrade in WEAPON_UPGRADES,
            f"sustain-loot-v1 warm start requires a registered weapon upgrade law "
            f"{list(WEAPON_UPGRADES)}: {resource_weapon_upgrade!r}")
    return {
        "worker_time_protocol": recipe["protocol"],
        "worker_time_recipe": recipe,
        "resource_readiness_law": resource_readiness_law,
        "resource_retreat": resource_retreat,
        # R17.0 修正案一 accounting: v1 (the legacy ruler) is written as None by
        # _training_contract, so the migrated contract carries the key with the
        # value None -- exactly what train_ppo writes (R18-B3b 2026-09-07).
        "worker_descend_escrow_readiness_table":
            ("v2" if worker_descend_escrow_readiness_table == "v2" else None),
        # R18-M2 (2026-09-07): "all" is the frozen behaviour and _training_contract
        # writes None for it (train/train_ppo.py:2739-2741); only l1-only is literal.
        "hunt_scope": (hunt_scope if hunt_scope == "l1-only" else None),
        # R18-B6 (2026-09-07): "off" is the frozen behaviour and _training_contract
        # writes None for it (train/train_ppo.py:2787-2795); only the law is literal.
        "resource_sweep": (resource_sweep if resource_sweep != "off" else None),
        "resource_identify": (resource_identify if resource_identify != "off" else None),
        "resource_weapon_upgrade": (
            resource_weapon_upgrade if resource_weapon_upgrade != "off" else None),
    }


def world_of(target):
    """Re-derive (and thereby re-validate) the world a target contract claims."""
    require(isinstance(target, dict), "Invalid loot warm-start target contract")
    return target_world(target.get("worker_time_protocol"),
                        target.get("resource_readiness_law"),
                        target.get("resource_retreat"),
                        target.get("worker_descend_escrow_readiness_table") or "v1",
                        # R18-M2 (2026-09-07): None (the frozen default) re-derives to "all".
                        target.get("hunt_scope") or "all",
                        # R18-B6 (2026-09-07): None (the frozen default) re-derives to "off".
                        target.get("resource_sweep") or "off",
                        target.get("resource_identify") or "off",
                        target.get("resource_weapon_upgrade") or "off")


def target_contract(source, implementation_sha256, worker_prefix, *, worker_time_protocol,
                    resource_readiness_law="coach-v03", resource_retreat="retreat-v1",
                    worker_descend_escrow_readiness_table="v1", hunt_scope="all",
                    resource_sweep="off", resource_identify="off",
                    resource_weapon_upgrade="off"):
    from eval_contract import resource_service_recipe, worker_prefix_recipe
    legacy.validate_source_contract(source)
    require(isinstance(implementation_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", implementation_sha256),
            "Invalid target implementation SHA256")
    world = target_world(worker_time_protocol, resource_readiness_law, resource_retreat,
                         worker_descend_escrow_readiness_table, hunt_scope,
                         resource_sweep, resource_identify, resource_weapon_upgrade)
    require(isinstance(worker_prefix, dict), "Earned suffix requires an exact prefix recipe")
    expected_prefix = worker_prefix_recipe(LEARNING_WINDOW_SCOPE,
        worker_prefix.get("source_sha256"), worker_prefix.get("max_attempts"),
        worker_prefix.get("max_microsteps"))
    require(worker_prefix == expected_prefix, "Earned suffix prefix recipe drift")
    target = deepcopy(source)
    target.update(resource_protocol=RESOURCE_PROTOCOL,
        resource_purchase_mode=RESOURCE_PURCHASE_MODE,
        resource_service_policy=SERVICE_POLICY,
        resource_service_recipe=resource_service_recipe(
            RESOURCE_PROTOCOL, RESOURCE_PURCHASE_MODE, SERVICE_POLICY,
            time_protocol=world["worker_time_protocol"]),
        implementation_sha256=implementation_sha256,
        dive_blocker_recovery=DIVE_BLOCKER_RECOVERY,
        worker_learning_window_scope=LEARNING_WINDOW_SCOPE,
        worker_prefix=deepcopy(expected_prefix))
    # R18-B3b (2026-09-07) review round: every world key is written, None
    # included.  train_ppo._training_contract emits
    # worker_descend_escrow_readiness_table unconditionally (None for the legacy
    # v1 ruler); dropping the key here left the v1 arm mintable but unusable --
    # json_sha256 is key-presence sensitive, so validate_inherited_receipt could
    # never match the receipt's target_contract_sha256 against the live contract.
    target.update({key: deepcopy(value) for key, value in world.items()})
    # R18-M (2026-09-07) merge resolution, B3b x B5.  B5 gave
    # train_ppo._training_contract two more UNCONDITIONAL keys -- "hunt_scope"
    # (train/train_ppo.py:2739-2741) and "resource_portal"
    # (train/train_ppo.py:2782-2784) -- both spelled None at their frozen
    # defaults ("all" / "off").  B3b's own law is that the minted target contract
    # IS the live training contract (tests/test_r18b3b_loot_warm_start.py:796-800)
    # and json_sha256 is key-presence sensitive, so without these two keys every
    # schema/2 warm start dies in validate_inherited_receipt below with "Loot
    # warm-start receipt does not pin the world being entered".  Neither patch
    # alone is wrong; the two laws only meet here.  The keys are added OUTSIDE
    # `world` on purpose: B3b's world vocabulary (WORLD_KEYS, receipt
    # ["target_world"], ALLOWED_CONTRACT_KEYS) stays byte-for-byte what B3b
    # registered, and the registered value of each new key stays the frozen
    # default.  A launch that sets --hunt-scope or --resource-portal is still
    # refused, by validate_target_contract() below: its re-derived `expected`
    # carries None while the live contract carries the literal, so the
    # "drift outside the exact registered world" gate fires.  setdefault, not
    # assignment, so a source contract that already names either key is left
    # alone and judged by that same gate.
    # R18-M2 (2026-09-07): hunt_scope was REGISTERED (see HUNT_SCOPES above), so
    # `world` now writes that key and its setdefault here is gone -- it would be
    # a no-op after the world update, and leaving it would suggest the key is
    # still unregistered.  resource_portal keeps the M resolution verbatim: no
    # launch registers portal-v1 for a loot warm start, so it stays fail-closed.
    target.setdefault("resource_portal", None)
    return target


def validate_target_contract(source, target, implementation_sha256):
    require(isinstance(target, dict), "Invalid loot warm-start target contract")
    require(target.get("resource_service_policy") == SERVICE_POLICY,
            "Only sustain-loot-v1 targets belong to this migration schema")
    world = world_of(target)
    expected = target_contract(source, implementation_sha256, target.get("worker_prefix"),
        worker_time_protocol=world["worker_time_protocol"],
        resource_readiness_law=world["resource_readiness_law"],
        resource_retreat=world["resource_retreat"],
        worker_descend_escrow_readiness_table=(
            target.get("worker_descend_escrow_readiness_table") or "v1"),
        hunt_scope=(target.get("hunt_scope") or "all"),
        # R18-B6 (2026-09-07): the registered world re-derives these too.
        resource_sweep=(target.get("resource_sweep") or "off"),
        resource_identify=(target.get("resource_identify") or "off"),
        resource_weapon_upgrade=(target.get("resource_weapon_upgrade") or "off"))
    require(all(target.get(key) == expected.get(key)
                for key in set(target) | set(expected)),
            "Loot warm-start target contract drift outside the exact registered world")
    changed = {key for key in set(source) | set(target) if source.get(key) != target.get(key)}
    require(changed <= ALLOWED_CONTRACT_KEYS, "Unexpected loot warm-start contract changes")


def make_receipt(parent_data, target, parameters_sha):
    from eval_contract import dive_blocker_recovery_recipe
    source = parent_data["diablogym_contract"]
    legacy.validate_source_contract(source)
    require(parent_data.get("num_timesteps") == PARENT_STEPS,
            "Parent training count differs from the registered R16")
    historical = {name: deepcopy(parent_data.get(name)) for name in legacy.WARMUP_FIELDS}
    require(historical == PARENT_HISTORICAL_CRITIC_WARMUP
            and all(type(historical[key]) is type(value)
                    for key, value in PARENT_HISTORICAL_CRITIC_WARMUP.items()),
            "Registered parent critic warmup evidence differs")
    lineage = deepcopy(parent_data.get("_resume_lineage"))
    require(json_sha256(lineage) == PARENT_LINEAGE_SHA256, "Parent resume lineage differs")
    validate_target_contract(source, target, target.get("implementation_sha256"))
    require(isinstance(parameters_sha, str) and re.fullmatch(r"[0-9a-f]{64}", parameters_sha),
            "Invalid preserved policy tensor SHA256")
    world = world_of(target)
    return {
        "schema": SCHEMA,
        "operation": operation_for(world["worker_time_protocol"]),
        "parent_checkpoint_sha256": PARENT_SHA256,
        "parent_contract_sha256": PARENT_CONTRACT_SHA256,
        "parent_num_timesteps": PARENT_STEPS,
        "source_contract": deepcopy(source),
        "target_contract_sha256": json_sha256(target),
        "target_implementation_sha256": target["implementation_sha256"],
        "policy_sha256": parameters_sha,
        "historical_critic_warmup": historical,
        "parent_resume_lineage": lineage,
        "optimizer_state": "reset-empty",
        "current_world_training_steps": 0,
        "historical_warmup_is_current_training": False,
        "exact_trajectory_continuation": False,
        "environment_state_mode": "fresh-normal-l1-no-snapshot",
        "publication_status": "INITIALIZATION_ONLY_NOT_TRAINED",
        # The whole target world, spelled out, so the frozen receipt never needs
        # the target contract to explain what it was minted for.
        "resource_protocol": RESOURCE_PROTOCOL,
        "resource_purchase_mode": RESOURCE_PURCHASE_MODE,
        "resource_service_policy": SERVICE_POLICY,
        "resource_service_recipe": deepcopy(target["resource_service_recipe"]),
        "worker_time_protocol": world["worker_time_protocol"],
        "worker_time_recipe": deepcopy(world["worker_time_recipe"]),
        "worker_learning_window_scope": LEARNING_WINDOW_SCOPE,
        "worker_prefix": deepcopy(target["worker_prefix"]),
        "dive_blocker_recovery": DIVE_BLOCKER_RECOVERY,
        "dive_blocker_recovery_recipe": dive_blocker_recovery_recipe(DIVE_BLOCKER_RECOVERY),
        "target_world": deepcopy(world),
    }


def validate_inherited_receipt(receipt, target):
    """Re-derive the whole receipt from its own evidence and compare by digest."""
    require(isinstance(receipt, dict) and isinstance(target, dict),
            "Invalid loot warm-start lineage")
    require(receipt.get("schema") == SCHEMA
            and receipt.get("operation") in set(OPERATIONS.values())
            and receipt.get("parent_checkpoint_sha256") == PARENT_SHA256
            and receipt.get("parent_contract_sha256") == PARENT_CONTRACT_SHA256
            and receipt.get("parent_num_timesteps") == PARENT_STEPS,
            "Invalid registered loot warm-start lineage")
    validate_target_contract(receipt.get("source_contract"), target,
                             target.get("implementation_sha256"))
    world = world_of(target)
    # The clock, the laws and the service recipe are claims in the frozen
    # receipt; they must equal the world actually being entered.
    require(receipt.get("operation") == operation_for(world["worker_time_protocol"])
            and receipt.get("worker_time_protocol") == world["worker_time_protocol"]
            and receipt.get("worker_time_recipe") == world["worker_time_recipe"]
            and receipt.get("target_world") == world
            and receipt.get("resource_service_policy") == target.get("resource_service_policy")
            and receipt.get("resource_service_recipe") == target.get("resource_service_recipe")
            and receipt.get("worker_learning_window_scope")
                == target.get("worker_learning_window_scope")
            and receipt.get("worker_prefix") == target.get("worker_prefix")
            and receipt.get("dive_blocker_recovery") == target.get("dive_blocker_recovery")
            and receipt.get("target_implementation_sha256")
                == target.get("implementation_sha256")
            and receipt.get("target_contract_sha256") == json_sha256(target),
            "Loot warm-start receipt does not pin the world being entered")
    parent = {
        "diablogym_contract": receipt.get("source_contract"),
        "num_timesteps": PARENT_STEPS,
        "_resume_lineage": receipt.get("parent_resume_lineage"),
        **(receipt.get("historical_critic_warmup")
           if isinstance(receipt.get("historical_critic_warmup"), dict) else {}),
    }
    expected = make_receipt(parent, target, receipt.get("policy_sha256"))
    require(json_sha256(receipt) == json_sha256(expected),
            "Loot warm-start receipt identity or semantics drift")


def migrate(parent, *, parent_sha256, output_dir, implementation_sha256, seed,
            worker_time_protocol, worker_prefix, resource_readiness_law="coach-v03",
            resource_retreat="retreat-v1", worker_descend_escrow_readiness_table="v1",
            hunt_scope="all", resource_sweep="off", resource_identify="off",
            resource_weapon_upgrade="off"):
    """Explicit-only model operation; all validation also remains in the reload path."""
    import torch
    from leashed_ppo import LeashedMaskablePPO
    from train_ppo import (_validate_checkpoint_bytes, _validate_resumable_leashed_boundary,
        _atomic_save_model, _implementation_bundle_sha256)
    require(parent_sha256 == PARENT_SHA256,
            "Only the exact registered R16 parent SHA is accepted")
    require(type(seed) is int and 0 <= seed < 2**32, "Explicit uint32 seed required")
    source_path = Path(parent).resolve(strict=True)
    destination = Path(output_dir).absolute()
    require(not destination.exists(), "Warm-start output directory already exists")
    payload = source_path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == PARENT_SHA256,
            "Parent checkpoint SHA mismatch")
    parent_data = _validate_resumable_leashed_boundary(_validate_checkpoint_bytes(
        payload, str(source_path), require_leashed=True))
    source = parent_data["diablogym_contract"]
    target = target_contract(source, implementation_sha256, worker_prefix,
        worker_time_protocol=worker_time_protocol,
        resource_readiness_law=resource_readiness_law,
        resource_retreat=resource_retreat,
        worker_descend_escrow_readiness_table=worker_descend_escrow_readiness_table,
        hunt_scope=hunt_scope, resource_sweep=resource_sweep,
        resource_identify=resource_identify,
        resource_weapon_upgrade=resource_weapon_upgrade)
    require(_implementation_bundle_sha256() == implementation_sha256,
            "Target implementation changed before migration")
    torch.set_num_threads(1)
    model = LeashedMaskablePPO.load(io.BytesIO(payload), device="cpu",
                                    teacher_path=None, teacher_sha256=None)
    before = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
    parameters_sha = policy_sha256(model)
    probe = policy_probe(model)
    receipt = make_receipt(parent_data, target, parameters_sha)
    legacy.reset_to_initialization(model, receipt, target, seed)
    for name, value in model.policy.state_dict().items():
        require(torch.equal(before[name], value.detach()), f"Changed policy tensor: {name}")
    require(policy_probe(model) == probe, "Reset changed fixed logits or critic values")
    destination.mkdir(parents=True, exist_ok=False)
    model_path = _atomic_save_model(model, destination / "model_warm_start.zip")
    manifest = {"schema": SCHEMA, "operation": receipt["operation"],
        "status": "INITIALIZATION_ONLY_NOT_TRAINED", "seed": seed,
        "parent_path": str(source_path), "parent_sha256": PARENT_SHA256,
        "source_contract": source, "target_contract": target, "receipt": receipt,
        "model_file": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "policy_tensor_count": len(before), "policy_sha256": parameters_sha,
        "policy_probe": probe, "ordinary_resume_eligible": False,
        "trained_in_target_world": False,
        "dive_blocker_recovery_recipe": deepcopy(receipt["dive_blocker_recovery_recipe"])}
    restored = legacy.load_initialization(model_path.read_bytes(), manifest)
    for name, value in restored.policy.state_dict().items():
        require(torch.equal(before[name], value.detach()),
                f"Save/load changed policy tensor: {name}")
    require(hashlib.sha256(source_path.read_bytes()).hexdigest() == PARENT_SHA256,
            "Parent checkpoint changed during migration")
    require(_implementation_bundle_sha256() == implementation_sha256,
            "Target implementation changed during migration")
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    legacy.capture_initialization(destination / "manifest.json", implementation_sha256)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--implementation-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resource-service-policy", choices=(SERVICE_POLICY,),
                        default=SERVICE_POLICY,
                        help="this schema mints only the sustain-loot-v1 world")
    parser.add_argument("--worker-time-protocol", required=True, choices=tuple(OPERATIONS))
    parser.add_argument("--resource-readiness-law", choices=READINESS_LAWS,
                        default=READINESS_LAWS[0])
    parser.add_argument("--resource-retreat", choices=RETREATS, default=RETREATS[0])
    parser.add_argument("--worker-descend-escrow-readiness-table",
                        choices=ESCROW_READINESS_TABLES, default=ESCROW_READINESS_TABLES[0])
    # R18-M2 (2026-09-07): the registered hunt scope of the arm being minted.
    parser.add_argument("--hunt-scope", choices=HUNT_SCOPES, default=HUNT_SCOPES[0])
    # R18-B6 (2026-09-07): the registered loot-itinerary laws of the arm being minted.
    parser.add_argument("--resource-sweep", choices=SWEEPS, default=SWEEPS[0])
    parser.add_argument("--resource-identify", choices=IDENTIFIES, default=IDENTIFIES[0])
    parser.add_argument("--resource-weapon-upgrade", choices=WEAPON_UPGRADES,
                        default=WEAPON_UPGRADES[0])
    parser.add_argument("--worker-prefix-model", required=True)
    parser.add_argument("--worker-prefix-max-attempts", required=True, type=int)
    parser.add_argument("--worker-prefix-max-microsteps", required=True, type=int)
    args = parser.parse_args()
    from eval_contract import worker_prefix_recipe
    prefix = worker_prefix_recipe(LEARNING_WINDOW_SCOPE,
        hashlib.sha256(Path(args.worker_prefix_model).read_bytes()).hexdigest(),
        args.worker_prefix_max_attempts, args.worker_prefix_max_microsteps)
    result = migrate(args.parent, parent_sha256=args.parent_sha256,
        output_dir=args.output_dir, implementation_sha256=args.implementation_sha256,
        seed=args.seed, worker_time_protocol=args.worker_time_protocol, worker_prefix=prefix,
        resource_readiness_law=args.resource_readiness_law,
        resource_retreat=args.resource_retreat,
        worker_descend_escrow_readiness_table=args.worker_descend_escrow_readiness_table,
        hunt_scope=args.hunt_scope, resource_sweep=args.resource_sweep,
        resource_identify=args.resource_identify,
        resource_weapon_upgrade=args.resource_weapon_upgrade)
    print(json.dumps({key: result[key]
                      for key in ("status", "operation", "model_sha256", "policy_sha256")}))


if __name__ == "__main__":
    main()
