"""Evaluation archive schema v5 / protocol v4: identity binding and the concurrent-write contract.

A new archive must answer three questions: who was evaluated, which binaries/game content were used, and whether the
per-seed detail really yields the declared aggregate. Old archives lack this evidence and are rejected by default; forensic code
may read them only when it supplies the full SHA-256 of the historical file explicitly.
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import pathlib
import platform
import re
import stat as stat_module
import statistics
import sys
import sysconfig
import zipfile
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 5
PROTOCOL_NAME = "diablogym.eval_assembled"
# Protocol v4 is the first environment contract with explicit command
# cancellation, validity masks for combat/support actions, conservative damage
# credit, and non-terminal worker-window continuation.  Those changes alter the
# MDP itself, so v3 archives and calibrated thresholds must fail closed instead
# of being appended to the same leaderboards.
PROTOCOL_VERSION = 4
PROTOCOL_MAX_STEPS = 3000
PUBLISHED_WORKER_RECEIPT_NAME = "bc_aux_behavior_receipt.json"
UINT32_MAX = 2**32 - 1
DEFAULT_MANAGER_RELATIVE = "train/models/v22-h-manager/policy.npz"
DEFAULT_MANAGER_SHA256 = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"

RUNTIME_PACKAGE_VERSIONS = {
    "numpy": "2.5.0",
    "gymnasium": "1.3.0",
    "torch": "2.12.1",
    "stable-baselines3": "2.9.0",
    "sb3-contrib": "2.9.0",
    "tensorboard": "2.21.0",
}

PROTOCOL_SOURCE_FILES = (
    "train/eval_assembled.py",
    "train/migrate_resource_candidate.py",
    "train/migrate_completion_candidate.py",
    "train/prefix_worker.py",
    "train/eval_contract.py",
    "train/leashed_ppo.py",
    "python/diablogym/__init__.py",
    "python/diablogym/controller_wire.py",
    "python/diablogym/env.py",
    "python/diablogym/nav.py",
    "python/diablogym/options_env.py",
    "python/diablogym/worker_env.py",
    "python/diablogym/completion_clock.py",
    "python/diablogym/resource_protocol.py",
    "python/diablogym/resource_navigation.py",
    "python/diablogym/resource_sustain.py",
    "python/diablogym/resource_gold_memory.py",
    "python/diablogym/resource_sustain_gold.py",
    "python/diablogym/resource_sustain_armor.py",
    "python/diablogym/resource_sustain_completion.py",
    "python/diablogym/resource_sustain_combinations.py",
    "python/diablogym/resource_sustain_loot.py",
    # R18-B6 (2026-09-07): every law module that exists in the tree is bound.
    # R18-B5 flagged the gap: retreat/portal/aggro/engagement/sweep/identify/
    # weapon-upgrade all decide what the EVALUATED world does, so an archive
    # whose python_protocol digest omits them cannot testify to the world it
    # was minted in.  resource_sustain_combat.py / resource_sustain_protection.py
    # were missing from both lists for the same reason.  boss_avoidance.py is
    # deliberately absent: R18-J is not merged and the file does not exist,
    # and a missing path is a fail-closed EvalContractError here.
    "python/diablogym/resource_sustain_combat.py",
    "python/diablogym/resource_sustain_protection.py",
    "python/diablogym/resource_retreat.py",
    "python/diablogym/resource_portal.py",
    "python/diablogym/resource_sweep.py",
    "python/diablogym/resource_identify.py",
    "python/diablogym/resource_weapon_upgrade.py",
    "python/diablogym/aggro_cap.py",
    "python/diablogym/engagement.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_BASE_AGG_KEYS = {
    "n", "ret_mean", "ret_median", "died", "depth_median", "l3",
    "kills_mean", "farm_tau_mean", "farm_descend_rate", "override_rate",
    "cap_rate", "micro_steps_mean",
    "farm_r_mean", "farm_w_mean", "farm_bonus_mean",
    "farm_worker_wage_mean", "farm_kills_mean",
    "farm_worker_kills_mean", "nonfarm_r_mean", "nonfarm_kills_mean",
    "farm_dry_n_mean", "farm_fresh_n_mean",
    "farm_dry_worker_wage_mean", "farm_fresh_worker_wage_mean",
    "farm_dry_worker_kills_mean", "farm_fresh_worker_kills_mean",
    "farm_voluntary_drinks_mean", "farm_reflex_drain_attempts_mean",
    "farm_reflex_drains_mean",
    "farm_multi_drink_window_rate", "ending_belt_heals_mean",
    "victories", "game_over", "time_limit_idle", "time_limit_unsettled",
}
_ENGAGEMENT_KEYS = {
    "worker_calls", "worker_action_hist", "worker_divergences",
    "script_divergence_rate",
}
_GEAR_ENGAGEMENT_KEYS = {
    "worker_action14_mask_opportunities",
    "worker_action14_requests",
    "worker_action14_native_successes",
    "worker_action14_gear_utility_delta",
}
_ROW_KEYS = {
    "seed", "ret", "depth", "died", "kills", "farm_n", "farm_tau_mean",
    "farm_tau_sum", "farm_descend", "windows", "beats", "overrides", "cap",
    "mode_seq", "micro_steps", "terminal_kind",
    "farm_r", "farm_w", "farm_bonus", "farm_worker_wage",
    "farm_kills", "farm_worker_kills", "nonfarm_r", "nonfarm_kills",
    "farm_dry_n", "farm_fresh_n",
    "farm_dry_worker_wage", "farm_fresh_worker_wage",
    "farm_dry_worker_kills", "farm_fresh_worker_kills",
    "farm_voluntary_drinks", "farm_reflex_drain_attempts",
    "farm_reflex_drains",
    "farm_multi_drink_windows", "farm_max_voluntary_drinks_per_window",
    "ending_belt_heals",
}
_TERMINAL_KINDS = {
    "death", "victory", "game_over",
    "time_limit_idle", "time_limit_unsettled",
}
# Old-law defaults of the R16 new-law environment/classroom switches (= the OptionsEnv/DiabloGymEnv constructor defaults;
# eval_assembled._native_runtime guards FARM_SCENE_CAP against drift). The archive identity records only
# non-default values: with all defaults meta.protocol has no r16_environment key (old archives/default-flag archives
# stay byte-identical); writing a default value explicitly is a malformed identity and the validator rejects it.
R16_ENVIRONMENT_DEFAULTS = {
    "explore_global_hunt": False,
    # R18-B5 (2026-09-07): scope of the a10 global hunt (all = frozen behavior, expressed by omission).
    "hunt_scope": "all",
    "explore_global_fallback": False,
    "progress_far_tiles": 0,
    "farm_scene_cap": 1800,
    "reset_layer_clock_on_window": False,
    "resource_protocol": "off",
    "resource_purchase_mode": "full",
    "resource_service_policy": "legacy-v1",
    "resource_readiness_law": "veto-v1",
    "resource_retreat": "off",
    # R18-B5 (2026-09-07): the town-portal scroll carrier (off = frozen behavior, expressed by omission).
    "resource_portal": "off",
    # R18-B6 (2026-09-07): the three loot-trip laws: chest sweep / identify / weapon upgrade
    # (off = frozen behavior, expressed by omission; omission keeps old archives byte-identical).
    "resource_sweep": "off",
    "resource_identify": "off",
    "resource_weapon_upgrade": "off",
    "dive_blocker_recovery": "off",
}


RESOURCE_SERVICE_RECIPE_VERSION = "l2-town-paid-repair-v2"
_RESOURCE_SERVICE_STAGES = {
    "none": ("collect_gold",),
    "heal": ("collect_gold", "town_trip", "native_heal", "return_l1"),
    "potions": ("collect_gold", "town_trip", "native_heal", "potions_to_capacity", "return_l1"),
    "armor": ("collect_gold", "town_trip", "armor_if_needed", "native_heal", "return_l1"),
    "full": ("collect_gold", "town_trip", "armor_if_needed", "normal_paid_repair",
             "native_heal", "potions_to_capacity", "return_l1"),
}


WORKER_PREFIX_R16_SHA256 = "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"
EARNED_DIVE_SUFFIX_SCOPE = "earned-dive-suffix-v1"


def worker_prefix_recipe(scope, model_sha256=None, max_attempts=None, max_microsteps=None):
    """Pure identity for excluded fixed-policy prefixes; no model or engine load."""
    if scope != EARNED_DIVE_SUFFIX_SCOPE:
        if scope not in (None, "farm-only", "farm-dive-v1"):
            raise ValueError("Unknown worker learning window scope")
        if any(value is not None for value in (model_sha256, max_attempts, max_microsteps)):
            raise ValueError("Prefix parameters require earned-dive-suffix-v1")
        return None
    if model_sha256 != WORKER_PREFIX_R16_SHA256:
        raise ValueError("earned-dive-suffix-v1 requires the exact registered R16 prefix")
    for name, value in (("max_attempts", max_attempts), ("max_microsteps", max_microsteps)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Prefix {name} must be an explicit positive integer")
    return {"version": "earned-dive-suffix-prefix-v1", "source_sha256": model_sha256,
        "observation_view": "dual-v4-asymmetric-v3", "action12_mode": "environment-mask",
        "sampling": "original-sampled-predict",
        "rng_isolation": "persistent-private-python-numpy-torch-cpu-v1",
        "episode_reseed": "actual-episode-seed", "max_attempts": max_attempts,
        "max_microsteps": max_microsteps, "budget_scope": "per-env-lifetime-across-resets",
        "physical_clock_start": "normal-reset-return-main-l1-formal-beat0",
        "initial_town_navigation": "excluded-from-formal-microsteps-inherited-reset-boundary",
        "wall_clock_scope": "complete-reset-and-prefix",
        "handoff": "first-main-l1-live-dive-ready-idle-after-normal-drain",
        "prefix_training": "excluded", "on_beat": None}


def validate_resource_service_config(protocol="off", mode="full",
                                     service_policy="legacy-v1"):
    """Validate the versioned service without importing or starting the engine."""
    if protocol not in ("off", "l2-town-v1") or mode not in _RESOURCE_SERVICE_STAGES:
        raise ValueError("Invalid resource protocol/arm for service recipe")
    if protocol == "off" and mode != "full":
        raise ValueError("resource_purchase_mode requires resource_protocol l2-town-v1")
    if service_policy not in ("legacy-v1", "sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"):
        raise ValueError(f"Invalid resource_service_policy: {service_policy!r}")
    if service_policy in ("sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1") and (protocol != "l2-town-v1" or mode != "full"):
        raise ValueError(f"{service_policy} requires resource_protocol l2-town-v1 and full purchase mode")


def validate_dive_blocker_recovery(protocol="off", recovery="off"):
    if recovery not in ("off", "adjacent-v1"):
        raise ValueError(f"Invalid dive_blocker_recovery: {recovery!r}")
    if recovery != "off" and protocol != "l2-town-v1":
        raise ValueError("dive_blocker_recovery requires resource_protocol l2-town-v1")


def dive_blocker_recovery_recipe(recovery="off"):
    if recovery == "off":
        return None
    if recovery != "adjacent-v1":
        raise ValueError(f"Invalid dive_blocker_recovery: {recovery!r}")
    return {"version": "adjacent-v1", "scope": "a11-main-dungeon-descent-only",
            "trigger": "visible-adjacent-live-monster-current-or-reserved-next-path-tile",
            "action": "native-controller-radius1-attack",
            "core_macro_microstep_cap": 12,
            "core_budget": "shared-a11-remaining",
            "settle": "normal-idle-settle-charged-each-native-microstep",
            "deadline": "live-episode-and-followup-cutoff",
            "continuation": "replan-from-current-native-state"}


# R18-B3 (2026-09-07): once the loot economy enters worker training, the recipe must state which
# completion clock it runs under: "two town trips + selling gear" under completion-l2-v1 (followup 1800) and under completion-l2-r18c
# (followup 9000) are not the same thing. Each clock gets one registered version
# number; an unregistered clock fails closed. The key set is deliberately identical to diablogym.completion_clock.
# COMPLETION_PROTOCOLS (pinned byte for byte by tests/test_r18b3_loot_training_contract.py),
# so this module still does not import the engine (archive forensics must also run on machines without the native extension).
LOOT_SERVICE_RECIPE_VERSIONS = {
    "completion-l2-v1": "l1-two-trip-loot-economy-v1",
    "completion-l2-r18c": "l1-two-trip-loot-economy-v1-r18c",
}
LOOT_SERVICE_DEFAULT_TIME_PROTOCOL = "completion-l2-v1"


def resource_service_recipe(protocol="off", mode="full",
                            service_policy="legacy-v1",
                            time_protocol=None):
    """R18-B3: ``time_protocol`` only matters for sustain-loot-v1; when omitted the old recipe stays byte-identical.

    R18-B3 review (2026-09-07): the sentinel changed from "the default clock value" to ``None`` (= the caller passed
    no clock). Previously the real default served as the sentinel, so ``resource_service_recipe(...,"sustain-v6",
    time_protocol="completion-l2-v1")`` was silently accepted and ignored, while the same call with
    completion-l2-r18c raised: callers that uniformly thread the run clock into the recipe were half silent and
    half crashing. Now **any** explicit time_protocol on a non-loot service law fails closed;
    the default output (including key order) is byte-identical.
    """
    validate_resource_service_config(protocol, mode, service_policy)
    if service_policy == "sustain-loot-v1":
        if time_protocol is None:
            time_protocol = LOOT_SERVICE_DEFAULT_TIME_PROTOCOL
        if time_protocol not in LOOT_SERVICE_RECIPE_VERSIONS:
            raise ValueError(
                "sustain-loot-v1 requires a registered completion-l2 time protocol "
                f"({sorted(LOOT_SERVICE_RECIPE_VERSIONS)}): {time_protocol!r}")
    elif time_protocol is not None:
        raise ValueError(
            f"time_protocol only applies to sustain-loot-v1, not {service_policy!r}")
    if protocol == "off":
        return None
    if service_policy == "sustain-loot-v1":
        recipe = resource_service_recipe(protocol, mode, "sustain-v6")
        recipe.update(version=LOOT_SERVICE_RECIPE_VERSIONS[time_protocol],
            service_policy=service_policy,
            time_protocol=time_protocol, service_microstep_cap=3000,
            collect_command_window_microsteps=900, max_town_trips=2,
            native_loot_economy=True,
            second_trip_trigger="native-deficit-and-real-resource-or-growth-change-after-return",
            loot_collection="observed-non-upgrade-smith-equipment-normal-inventory",
            sale="on-site-identity-and-price-bound-native-smith-inventory-sale",
            sale_income="actual-personal-inventory-gold-receipt-only",
            gear_replacement="retain-replaced-items-or-reject-no-room",
            stages=["collect_observed_gold_and_loot", "town_trip", "sell_idle_smith_equipment",
                    "observe_smith", "observe_healer_and_native_heal",
                    "joint_minimum_readiness_budget", "normal_unequip_repair_or_replace",
                    "potions_to_readiness", "return_original_l1"])
        return recipe
    if service_policy in ("sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"):
        recipe = {
            "version": "l2-town-joint-" + service_policy,
            "service_policy": service_policy,
            "mode": mode,
            "stages": ["collect_gold", "town_trip", "observe_smith",
                       "observe_healer_and_native_heal",
                       "joint_minimum_readiness_budget",
                       "normal_unequip_repair_or_replace",
                       "potions_to_readiness", "return_l1"],
            "planning_scope": "retain_or_one_armor_change_plus_repairs_live_replan",
            "service_microstep_cap": 1500,
            "potion_target": 4,
            "potion_target_source": "native_readiness.required_belt_heals",
            "navigation_recovery": "bounded-combat-heal-v1",
        }
        if service_policy in ("sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6"):
            recipe["stages"][0] = "collect_observed_l1_gold"
            recipe.update(
                gold_memory={
                    "version": "observed-l1-gold-memory-v1",
                    "source": "existing-native-main-l1-observations",
                    "identity_fields": ["seed_hi", "seed_lo", "create_info", "base_id"],
                    "pickup": "normal-revisit-current-active-id-revalidation",
                    "remembered_value_is_wallet": False,
                },
                collect_microstep_cap=300,
                collect_budget_scope="movement-combat-and-pickup-actual-microsteps",
                collect_budget_exhausted="continue_outbound",
            )
            if service_policy in ("sustain-v4", "sustain-v5", "sustain-v6"):
                # A command-admission window, not a promise that existing
                # native animation has physically settled by microstep 450.
                del recipe["collect_microstep_cap"]
                recipe.update(
                    collect_command_window_microsteps=450,
                    collect_budget_scope="new-collect-commands-within-actual-microstep-window",
                    collect_tail="finish-existing-animation-in-outbound-charged-to-service-budget",
                    collect_cutoff="record-current-native-state-at-command-window-cutoff",
                )
        if service_policy in ("sustain-v5", "sustain-v6"):
            recipe.update(
                planning_scope="observed_single_ordinary_armor_retained_repairs_minimum_medicine",
                native_ordinary_armor_scope=True,
                ordinary_armor_catalog={
                    "version": "ordinary-armor-v1",
                    "items": ["normal_helmet", "normal_shield", "normal_chest"],
                    "projection": "native_full_readiness_actual_replaced_slots",
                    "projection_refresh": "known_smith_identities_live_player_before_each_joint_plan",
                    "projection_refresh_native_microsteps": 0,
                    "selection": "effective_blocking_if_complete_affordable_then_lowest_total",
                    "blocking_is_native_readiness_gate": False,
                    "missing_block_projection": "unknown",
                    "partial_free_equip_fallback": False,
                    "missing_quote_or_projection": "unresolved_not_physical_impossibility",
                },
            )
        if service_policy == "sustain-v6":
            recipe.update(
                native_preserve_equipment_readiness=True,
                equipment_readiness_preservation={
                    "version": "equipment-readiness-preservation-v1",
                    "scope": "native-a14-eligibility-plan-and-commit",
                    "equipment_failures": ["armor", "damage", "weapon", "durability"],
                    "trigger": "current-equipment-subset-ready",
                    "requirement": "next-equipment-subset-ready",
                    "independent_failures": ["level", "health", "potions"],
                    "unready_equipment": "legacy-upgrade-rule",
                },
            )
        return recipe
    # Preserve the exact historical legacy recipe, including omitted policy.
    return {"version": RESOURCE_SERVICE_RECIPE_VERSION, "mode": mode,
            "stages": list(_RESOURCE_SERVICE_STAGES[mode]),
            "service_microstep_cap": 600,
            "potion_target": 8 if mode in ("potions", "full") else None}


class EvalContractError(ValueError):
    """The evaluation archive violates the identity, schema or numeric contract."""


class OutputReservationError(FileExistsError):
    """The target archive already exists, or the same tag is being produced by another process."""


class OperationalFailure(RuntimeError):
    """A known training, export or evaluation infrastructure failure in the driver."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvalContractError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
             f"{label} must be a full lowercase SHA-256")
    return value


def source_bundle_sha256(files: Mapping[str, str]) -> str:
    """Build a stable protocol-source bundle SHA from relative paths and each file's full SHA."""
    h = hashlib.sha256()
    for name, digest in sorted(files.items()):
        _validate_sha256(digest, f"runtime.python_protocol.files[{name!r}]")
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def runtime_versions_identity() -> dict[str, Any]:
    """Bind numerical dependencies without importing their heavy modules."""
    try:
        packages = {
            name: importlib.metadata.version(name)
            for name in RUNTIME_PACKAGE_VERSIONS
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise EvalContractError(f"evaluation runtime dependency missing: {exc.name}") from exc
    # Cross-platform note (2026-07-27 WSL2 port): the same upstream release carries a local
    # version segment on Linux wheels (e.g. 2.12.1+cpu). The gate compares the public version segment; archives still
    # record the full local version in packages, so the identity stays exact. The CUDA (+cu130) wheel imports
    # unreliably under WSL, so the CPU (+cpu) wheel is pinned there.
    mismatches = {
        name: (packages[name], expected)
        for name, expected in RUNTIME_PACKAGE_VERSIONS.items()
        if packages[name] != expected
        and packages[name].split("+", 1)[0] != expected
    }
    _require(not mismatches,
             f"evaluation runtime version drift (an upgrade requires redoing the numeric regression): {mismatches}")
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "packages": packages,
    }


def default_game_data_dir() -> pathlib.Path:
    """Keep the same platform contract as DiabloGymEnv's default data_dir."""
    return (pathlib.Path.home()
            / "Library" / "Application Support" / "diasurgical" / "devilution")


def default_assets_dir(root: pathlib.Path) -> pathlib.Path:
    return (root / "build" / "engine" / "devilutionx.app"
            / "Contents" / "Resources")


def _absolute_path(path: str | pathlib.Path) -> pathlib.Path:
    """Get the absolute lexical path without resolving symlinks, so the path choice itself can be re-verified."""
    return pathlib.Path(os.path.abspath(os.fspath(pathlib.Path(path).expanduser())))


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _stable_file_sha256(path: pathlib.Path, label: str) -> str:
    """Hash a regular file that the same path points to throughout and that does not change while being read."""
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    try:
        with open(path, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat_module.S_ISREG(before.st_mode),
                     f"{label} is not a regular file: {path}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise EvalContractError(f"{label} cannot be read stably: {path}: {exc}") from exc
    signature = _stat_signature(before)
    _require(signature == _stat_signature(after)
             and signature == _stat_signature(current),
             f"{label} changed while being hashed: {path}")
    return digest.hexdigest()


def game_data_identity(data_dir: str | pathlib.Path | None = None
                       ) -> dict[str, Any]:
    """Bind the main MPQ actually used in the default data_dir, in LoadGameArchives priority order."""
    directory = _absolute_path(data_dir or default_game_data_dir())
    _require(directory.is_dir(), f"game data_dir does not exist: {directory}")
    for name in ("DIABDAT.MPQ", "diabdat.mpq", "spawn.mpq"):
        candidate = directory / name
        if candidate.is_file():
            return {
                "path": str(candidate),
                "sha256": _stable_file_sha256(candidate, "game main MPQ"),
            }
    raise EvalContractError(
        f"game data_dir lacks DIABDAT.MPQ/diabdat.mpq/spawn.mpq: {directory}")


def _resource_files(directory: pathlib.Path) -> list[pathlib.Path]:
    try:
        entries = list(directory.rglob("*"))
    except OSError as exc:
        raise EvalContractError(f"resource tree cannot be traversed: {directory}: {exc}") from exc
    files: list[pathlib.Path] = []
    for entry in entries:
        _require(not entry.is_symlink(), f"resource tree must not contain symlinks: {entry}")
        if entry.is_file():
            files.append(entry)
        else:
            _require(entry.is_dir(), f"resource tree contains a special file: {entry}")
    return sorted(files, key=lambda path: path.relative_to(directory).as_posix())


def assets_tree_identity(assets_dir: str | pathlib.Path) -> dict[str, Any]:
    """Compute a deterministic Resources tree hash from relative paths and each file's raw content."""
    directory = _absolute_path(assets_dir)
    _require(directory.is_dir(), f"engine Resources directory does not exist: {directory}")
    _require(not directory.is_symlink(),
             f"engine Resources directory must not be a symlink: {directory}")
    files = _resource_files(directory)
    _require(bool(files), f"engine Resources directory is empty: {directory}")
    digest = hashlib.sha256(b"diablogym-resources-tree-v1\0")
    signatures: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        for path in files:
            relative = path.relative_to(directory).as_posix()
            relative_bytes = relative.encode("utf-8")
            with open(path, "rb") as stream:
                before = os.fstat(stream.fileno())
                _require(stat_module.S_ISREG(before.st_mode),
                         f"resource tree entry is not a regular file: {path}")
                _require(0 <= before.st_size < 2**64,
                         f"resource tree file size cannot be encoded: {path}")
                digest.update(len(relative_bytes).to_bytes(8, "big"))
                digest.update(relative_bytes)
                digest.update(before.st_size.to_bytes(8, "big"))
                read_size = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    read_size += len(chunk)
                after = os.fstat(stream.fileno())
            signature = _stat_signature(before)
            _require(read_size == before.st_size
                     and signature == _stat_signature(after),
                     f"resource file changed during the tree hash: {path}")
            signatures[relative] = signature
    except OSError as exc:
        raise EvalContractError(f"resource tree cannot be read stably: {exc}") from exc

    final_files = _resource_files(directory)
    final_relatives = [path.relative_to(directory).as_posix()
                       for path in final_files]
    _require(final_relatives == list(signatures),
             "resource tree file set changed during hashing")
    try:
        stable = all(_stat_signature(path.stat()) == signatures[relative]
                     for path, relative in zip(final_files, final_relatives))
    except OSError as exc:
        raise EvalContractError(f"resource tree read-back re-verification failed: {exc}") from exc
    _require(stable, "resource tree files changed before hashing finished")
    return {
        "path": str(directory),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def content_identity(root: pathlib.Path, *,
                     data_dir: str | pathlib.Path | None = None,
                     assets_dir: str | pathlib.Path | None = None
                     ) -> dict[str, Any]:
    root = root.resolve()
    return {
        "game_data": game_data_identity(data_dir),
        "assets": assets_tree_identity(assets_dir or default_assets_dir(root)),
    }


def engine_binary_path(root: pathlib.Path) -> pathlib.Path:
    """Locate the DevilutionX shared engine the bridge actually links; fail closed on ambiguity."""
    engine_root = root.resolve() / "build" / "engine"
    if sys.platform == "darwin":
        names = ("liblibdevilutionx_so.dylib",)
    elif sys.platform == "win32":
        names = ("libdevilutionx_so.dll", "liblibdevilutionx_so.dll")
    else:
        names = ("liblibdevilutionx_so.so",)
    directories = (
        engine_root,
        *(engine_root / config for config in
          ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")),
    )
    matches = {
        candidate.resolve()
        for directory in directories
        for name in names
        if (candidate := directory / name).is_file()
    }
    _require(len(matches) == 1,
             "cannot uniquely locate the DevilutionX engine shared library: "
             f"root={engine_root}, matches={sorted(map(str, matches))}")
    return next(iter(matches))


def loaded_engine_binary_path(expected_path: str | pathlib.Path) -> pathlib.Path:
    """Resolve the engine image actually mapped in this process, fail closed."""
    expected = pathlib.Path(expected_path).resolve()
    candidates: set[pathlib.Path] = set()
    if sys.platform == "darwin":
        process = ctypes.CDLL(None)
        image_count = process._dyld_image_count
        image_count.argtypes = []
        image_count.restype = ctypes.c_uint32
        image_name = process._dyld_get_image_name
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
        for index in range(image_count()):
            raw = image_name(index)
            if raw:
                path = pathlib.Path(os.fsdecode(raw))
                if path.name == expected.name:
                    candidates.add(path.resolve())
    elif sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_handle = kernel32.GetModuleHandleW
        get_handle.argtypes = [ctypes.c_wchar_p]
        get_handle.restype = ctypes.c_void_p
        get_filename = kernel32.GetModuleFileNameW
        get_filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                 ctypes.c_uint32]
        get_filename.restype = ctypes.c_uint32
        handle = get_handle(expected.name)
        if handle:
            buffer = ctypes.create_unicode_buffer(32768)
            if get_filename(handle, buffer, len(buffer)):
                candidates.add(pathlib.Path(buffer.value).resolve())
    else:
        maps = pathlib.Path("/proc/self/maps")
        if maps.is_file():
            for line in maps.read_text(encoding="utf-8", errors="strict").splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and not fields[5].endswith(" (deleted)"):
                    path = pathlib.Path(fields[5])
                    if path.name == expected.name:
                        candidates.add(path.resolve())
    _require(len(candidates) == 1,
             "cannot uniquely locate the DevilutionX engine actually mapped by this process: "
             f"expected={expected}, mapped={sorted(map(str, candidates))}")
    actual = next(iter(candidates))
    _require(actual == expected,
             f"actually mapped engine path differs from the frozen identity: {actual} != {expected}")
    return actual


def runtime_identity(root: pathlib.Path, bridge_path: pathlib.Path,
                     engine_path: pathlib.Path | None = None, *,
                     data_dir: str | pathlib.Path | None = None,
                     assets_dir: str | pathlib.Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    bridge_path = bridge_path.resolve()
    engine_path = (engine_path or engine_binary_path(root)).resolve()
    files: dict[str, str] = {}
    for relative in PROTOCOL_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise EvalContractError(f"protocol source missing: {path}")
        files[relative] = sha256_file(path)
    return {
        "bridge": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
        },
        "engine": {
            "path": str(engine_path),
            "sha256": sha256_file(engine_path),
        },
        "content": content_identity(
            root, data_dir=data_dir, assets_dir=assets_dir),
        "versions": runtime_versions_identity(),
        "python_protocol": {
            "sha256": source_bundle_sha256(files),
            "files": files,
        },
    }


def bridge_binary_path(root: pathlib.Path) -> pathlib.Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    _require(isinstance(suffix, str) and bool(suffix), "cannot determine the current Python extension ABI")
    path = root.resolve() / "build" / f"_diablogym{suffix}"
    _require(path.is_file(), f"bridge binary for the current ABI does not exist: {path}")
    return path


def resolve_checkpoint_file(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix.lower() != ".zip":
        zipped = pathlib.Path(f"{candidate}.zip")
        if zipped.is_file():
            return zipped.resolve()
    raise EvalContractError(f"checkpoint does not exist: {candidate}")


def checkpoint_num_timesteps_bytes(payload: bytes, label: str = "checkpoint") -> int:
    """Read SB3 progress from the same immutable bytes used by the loader."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            data = json.loads(archive.read("data"))
        value = data["num_timesteps"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            zipfile.BadZipFile) as exc:
        raise EvalContractError(f"checkpoint num_timesteps unreadable: {label}") from exc
    _require(_is_int(value) and value >= 0,
             f"checkpoint num_timesteps is illegal: {value!r}")
    return value


def checkpoint_num_timesteps(path: str | pathlib.Path) -> int:
    checkpoint = resolve_checkpoint_file(path)
    try:
        payload = checkpoint.read_bytes()
    except OSError as exc:
        raise EvalContractError(
            f"checkpoint num_timesteps unreadable: {checkpoint}") from exc
    return checkpoint_num_timesteps_bytes(payload, str(checkpoint))


def file_identity(kind: str, path: str | pathlib.Path,
                  *, num_timesteps: int | None = None,
                  gate_report_path: str | pathlib.Path | None = None) -> dict[str, Any]:
    resolved = pathlib.Path(path).resolve()
    _require(resolved.is_file(), f"policy file does not exist: {resolved}")
    report_sha = None
    if gate_report_path is not None:
        report = pathlib.Path(gate_report_path).resolve()
        _require(report.is_file(), f"policy gate report does not exist: {report}")
        report_sha = sha256_file(report)
    return {
        "kind": kind,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "num_timesteps": num_timesteps,
        "gate_report_sha256": report_sha,
    }


def script_worker_identity(protocol_bundle_sha256: str) -> dict[str, Any]:
    _validate_sha256(protocol_bundle_sha256, "protocol bundle SHA")
    return {
        "kind": "script",
        "path": None,
        "sha256": _sha256_text(
            f"diablogym-script-farm-v1\0{protocol_bundle_sha256}"),
        "num_timesteps": None,
        "gate_report_sha256": None,
    }


def freeze_eval_identity(root: pathlib.Path, worker_spec: str | pathlib.Path,
                         manager_path: str | pathlib.Path | None = None
                         ) -> dict[str, Any]:
    """Freeze all inputs and the current runtime identity before starting the evaluation subprocess."""
    root = root.resolve()
    runtime = runtime_identity(root, bridge_binary_path(root))
    spec = str(worker_spec)
    if spec == "script":
        worker = script_worker_identity(runtime["python_protocol"]["sha256"])
    elif spec == "bc":
        path = root / "train" / "runs" / "bc-worker" / "policy_sd.pt"
        worker = file_identity(
            "bc_state_dict", path, gate_report_path=path.with_name("bc_report.json"))
    elif pathlib.Path(spec).suffix.lower() == ".npz":
        worker = file_identity("numpy_policy", pathlib.Path(spec))
    else:
        checkpoint = resolve_checkpoint_file(spec)
        try:
            checkpoint_payload = checkpoint.read_bytes()
        except OSError as exc:
            raise EvalContractError(f"checkpoint unreadable: {checkpoint}") from exc
        worker = {
            "kind": "sb3_checkpoint", "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "num_timesteps": checkpoint_num_timesteps_bytes(
                checkpoint_payload, str(checkpoint)),
            "gate_report_sha256": (
                sha256_file(checkpoint.with_name(
                    PUBLISHED_WORKER_RECEIPT_NAME))
                if checkpoint.with_name(
                    PUBLISHED_WORKER_RECEIPT_NAME).is_file()
                else None),
        }
    manager_file = (pathlib.Path(manager_path) if manager_path is not None else
                    root / DEFAULT_MANAGER_RELATIVE)
    manager = file_identity("numpy_policy", manager_file)
    if manager_path is None:
        _require(manager["sha256"] == DEFAULT_MANAGER_SHA256,
                 "default v22-H manager SHA drift: "
                 f"{manager['sha256']} != {DEFAULT_MANAGER_SHA256}")
    return {"worker": worker, "manager": manager, "runtime": runtime}


def expected_eval_identity(snapshot: Mapping[str, Any], *, tag: str,
                           seeds: Iterable[int]) -> dict[str, Any]:
    worker, manager, runtime = (
        snapshot["worker"], snapshot["manager"], snapshot["runtime"])
    game_data = runtime["content"]["game_data"]
    assets = runtime["content"]["assets"]
    return {
        "expected_tag": tag,
        "expected_seeds": list(seeds),
        "expected_worker_kind": worker["kind"],
        "expected_worker_sha256": worker["sha256"],
        "expected_worker_gate_report_sha256": worker["gate_report_sha256"],
        "expected_manager_kind": manager["kind"],
        "expected_manager_sha256": manager["sha256"],
        "expected_worker_num_timesteps": worker["num_timesteps"],
        "expected_bridge_sha256": runtime["bridge"]["sha256"],
        "expected_engine_sha256": runtime["engine"]["sha256"],
        "expected_game_data_path": game_data["path"],
        "expected_game_data_sha256": game_data["sha256"],
        "expected_assets_path": assets["path"],
        "expected_assets_sha256": assets["sha256"],
        "expected_assets_file_count": assets["file_count"],
        "expected_runtime_versions": runtime["versions"],
        "expected_protocol_bundle_sha256": runtime["python_protocol"]["sha256"],
    }


def verify_eval_identity(snapshot: Mapping[str, Any], root: pathlib.Path) -> None:
    """Re-hash after the subprocess exits; any input swapped during evaluation voids the archive."""
    verify_file_identity(snapshot["worker"])
    verify_file_identity(snapshot["manager"])
    try:
        content = snapshot["runtime"]["content"]
        data_dir = pathlib.Path(content["game_data"]["path"]).parent
        assets_dir = pathlib.Path(content["assets"]["path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalContractError("frozen content identity structure is malformed") from exc
    current = runtime_identity(
        root.resolve(), bridge_binary_path(root.resolve()),
        data_dir=data_dir, assets_dir=assets_dir)
    _require(current == snapshot["runtime"],
             "bridge, engine, game data, resource tree or Python protocol source changed during evaluation")


def verify_file_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("kind") == "script":
        return
    path = identity.get("path")
    _require(isinstance(path, str) and pathlib.Path(path).is_file(),
             f"policy identity path is invalid: {path!r}")
    actual = sha256_file(path)
    _require(actual == identity.get("sha256"),
             f"policy file changed during evaluation: {actual} != {identity.get('sha256')!r}")
    expected_report = identity.get("gate_report_sha256")
    if expected_report is not None:
        report_name = (
            PUBLISHED_WORKER_RECEIPT_NAME
            if identity.get("kind") == "sb3_checkpoint"
            else "bc_report.json")
        report_path = pathlib.Path(path).with_name(report_name)
        _require(report_path.is_file(),
                 f"policy release/gate report disappeared during evaluation: {report_path}")
        actual_report = sha256_file(report_path)
        _require(actual_report == expected_report,
                 "policy release/gate report changed during evaluation: "
                 f"{actual_report} != {expected_report!r}")


def validate_r16_environment(value: Any) -> dict[str, Any]:
    """R16 environment-switch identity sub-dict: non-empty, known keys only, every key a legal non-default value."""
    _require(isinstance(value, dict) and bool(value),
             "meta.protocol.r16_environment must be a non-empty object")
    unknown = set(value) - set(R16_ENVIRONMENT_DEFAULTS)
    _require(not unknown,
             f"meta.protocol.r16_environment has unknown keys: {sorted(unknown)}")
    for key, item in value.items():
        default = R16_ENVIRONMENT_DEFAULTS[key]
        if key == "resource_protocol":
            _require(item == "l2-town-v1",
                     "resource_protocol must be l2-town-v1 (off is omitted)")
        elif key == "resource_purchase_mode":
            _require(item in ("none", "heal", "potions", "armor"),
                     "resource_purchase_mode must be none/heal/potions/armor "
                     "(full is omitted)")
            _require(value.get("resource_protocol") == "l2-town-v1",
                     "resource_purchase_mode requires resource_protocol")
        elif key == "resource_service_policy":
            _require(item in ("sustain-v2", "sustain-v3", "sustain-v4", "sustain-v5", "sustain-v6", "sustain-loot-v1"),
                     "resource_service_policy must be sustain-v2/sustain-v3/sustain-v4/sustain-v5/sustain-v6 (legacy-v1 is omitted)")
            # R18-B3 review (2026-09-07): from this version on, the loot recipe is versioned by completion clock
            # (LOOT_SERVICE_RECIPE_VERSIONS), yet the archive identity vocabulary
            # (R16_ENVIRONMENT_DEFAULTS) has no clock key at all: an archive cannot say
            # which clock it ran under. If we kept building the recipe from the default completion-l2-v1,
            # the archive of a worker trained under completion-l2-r18c would record a false statement about
            # its clock and economy version, while the honest -r18c recipe would be rejected by the validator.
            # Until the archive schema grows a clock key, fail closed: better unable to evaluate
            # than let a default value testify for the world under test.
            _require(item != "sustain-loot-v1",
                     "sustain-loot-v1 archives require the completion-l2 clock in the "
                     "archive identity, which this schema version does not carry yet "
                     "(R18-B3); no loot eval archive may be minted under the default clock")
            _require(value.get("resource_protocol") == "l2-town-v1"
                     and value.get("resource_purchase_mode", "full") == "full",
                     "sustain service requires resource_protocol l2-town-v1 and full purchase mode")
        elif key == "resource_readiness_law":
            _require(item == "coach-v03",
                     "resource_readiness_law must be coach-v03 (veto-v1 is omitted)")
            _require(value.get("resource_protocol") == "l2-town-v1",
                     "resource_readiness_law coach-v03 requires resource_protocol l2-town-v1")
        elif key == "resource_retreat":
            _require(item == "retreat-v1",
                     "resource_retreat must be retreat-v1 (off is omitted)")
            _require(value.get("resource_protocol") == "l2-town-v1"
                     and value.get("resource_readiness_law") == "coach-v03",
                     "resource_retreat retreat-v1 requires resource_protocol "
                     "l2-town-v1 and resource_readiness_law coach-v03")
        elif key == "resource_portal":
            # R18-B5 (2026-09-07): the portal is the retreat's carrier, so the archive identity must also record
            # that retreat law; otherwise a "portal without retreat" archive cannot explain how the window was closed.
            _require(item == "portal-v1",
                     "resource_portal must be portal-v1 (off is omitted)")
            _require(value.get("resource_protocol") == "l2-town-v1"
                     and value.get("resource_readiness_law") == "coach-v03"
                     and value.get("resource_retreat") == "retreat-v1",
                     "resource_portal portal-v1 requires resource_protocol "
                     "l2-town-v1, resource_readiness_law coach-v03 and "
                     "resource_retreat retreat-v1")
        elif key in ("resource_sweep", "resource_identify",
                     "resource_weapon_upgrade"):
            # R18-B6 (2026-09-07): all three laws live only inside the loot economy's town trips
            # (same as the deployment side's validate_sweep_protocol / validate_identify_protocol /
            # validate_weapon_upgrade), so the archive identity must also record
            # l2-town-v1 and sustain-loot-v1; smith-v1 additionally needs the full purchase mode.
            # Note this chain currently always fails closed: the resource_service_policy
            # branch above rejects every sustain-loot-v1 archive (R18-B3: the archive schema has no
            # clock key yet). This is deliberate: until an archive can say which clock it ran under,
            # an archive "with chest sweep but no clock" must not be minted.
            # Review fix (same day): for resource_weapon_upgrade this table is one value narrower than the deployment-side
            # vocabulary: RESOURCE_WEAPON_UPGRADES also has dry-v1, which is not
            # accepted here. This is deliberate, for the same reason as on the training side (_validate_args / make_env):
            # dry-v1 sends no command, takes no micro-step and makes no native call, so its row data are bit-identical
            # to the control arm; an archive identity stating dry-v1 would falsely testify for an environment bit-identical
            # to the default world (the same decision as the hunt_scope l1-only review fix). So
            # the B6 narrowing sits not only in the training ring but **explicitly** also in the archive identity ring and eval_assembled's
            # CLI ring; to examine a dry-v1 counterfactual arm it first needs its own identity decision
            # (B6-REPORT.md open question 2), not a pass here.
            expected = {"resource_sweep": "sweep-v1",
                        "resource_identify": "cain-v1",
                        "resource_weapon_upgrade": "smith-v1"}[key]
            _require(item == expected,
                     f"{key} must be {expected} (off is omitted)")
            _require(value.get("resource_protocol") == "l2-town-v1"
                     and value.get("resource_service_policy") == "sustain-loot-v1",
                     f"{key} {expected} requires resource_protocol l2-town-v1 "
                     "and resource_service_policy sustain-loot-v1")
            if key == "resource_weapon_upgrade":
                _require(value.get("resource_purchase_mode", "full") == "full",
                         "resource_weapon_upgrade smith-v1 requires the full "
                         "purchase mode")
        elif key == "hunt_scope":
            # R18-B5 (2026-09-07): a DiabloGymEnv-level switch; needs no resource protocol.
            _require(item == "l1-only",
                     "hunt_scope must be l1-only (all is omitted)")
            # Review fix (same day): the only consumer of l1-only is the gate of the a10 whole-map monster search,
            # so with the search off it is a guaranteed no-op. An archive that declares this law without enabling the search
            # would mint a second identity for an environment bit-identical to the default world; an identity must only
            # testify for laws that really ran (same decision as R18-B3), hence fail closed.
            _require(value.get("explore_global_hunt") is True,
                     "hunt_scope l1-only requires explore_global_hunt")
        elif key == "dive_blocker_recovery":
            _require(item == "adjacent-v1", "dive_blocker_recovery must be adjacent-v1 (off is omitted)")
            _require(value.get("resource_protocol") == "l2-town-v1",
                     "dive_blocker_recovery requires resource_protocol l2-town-v1")
        elif isinstance(default, bool):
            _require(item is True,
                     f"meta.protocol.r16_environment.{key} only allows true"
                     " (default false is expressed by omission)")
        else:
            _require(_is_int(item) and item > 0,
                     f"meta.protocol.r16_environment.{key} must be a positive integer")
            _require(item != default,
                     f"meta.protocol.r16_environment.{key} explicitly writes the default value "
                     f"{default}; malformed identity, omit it")
    return {key: value[key] for key in R16_ENVIRONMENT_DEFAULTS if key in value}


def make_protocol(seeds: Iterable[int], *,
                  worker_decoding: str = "argmax",
                  r16_environment: Mapping[str, Any] | None = None
                  ) -> dict[str, Any]:
    protocol = {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "environment": "OptionsEnv",
        "max_steps": PROTOCOL_MAX_STEPS,
        "action_selection": "argmax_with_action_masks",
        "manager_forward": "numpy_tanh_mlp",
        "reward": "undiscounted_manager_ledger",
        "deterministic": True,
        "seeds": list(seeds),
    }
    # R16 (C3): the worker decoding mode enters the archive identity. argmax is the only historical definition and is expressed
    # by omission (old archives/default-flag archives stay byte-identical); sample = the worker samples from its training distribution
    # with a per-episode seed (torch.manual_seed(seed), single thread) and stays bit-reproducible, so deterministic
    # stays True. action_selection describes the manager-side numpy argmax and is unchanged.
    if worker_decoding != "argmax":
        _require(worker_decoding == "sample",
                 f"worker_decoding only allows argmax/sample: {worker_decoding!r}")
        protocol["worker_decoding"] = worker_decoding
    # R16 new-law anchor: write the sub-dict only when some environment/classroom switch is non-default (listing only non-default keys).
    if r16_environment:
        protocol["r16_environment"] = validate_r16_environment(
            dict(r16_environment))
        if r16_environment.get("resource_protocol") == "l2-town-v1":
            protocol["resource_service_recipe"] = resource_service_recipe(
                "l2-town-v1", r16_environment.get("resource_purchase_mode", "full"),
                r16_environment.get("resource_service_policy", "legacy-v1"))
        if r16_environment.get("dive_blocker_recovery") == "adjacent-v1":
            protocol["dive_blocker_recovery_recipe"] = dive_blocker_recovery_recipe("adjacent-v1")
    return protocol


def make_meta(*, tag: str, seeds: Iterable[int], worker: Mapping[str, Any],
              manager: Mapping[str, Any], runtime: Mapping[str, Any],
              worker_decoding: str = "argmax",
              r16_environment: Mapping[str, Any] | None = None
              ) -> dict[str, Any]:
    return {
        "tag": tag,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": make_protocol(seeds, worker_decoding=worker_decoding,
                                  r16_environment=r16_environment),
        "worker": dict(worker),
        "manager": dict(manager),
        "runtime": dict(runtime),
    }


def recompute_agg(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    _require(n > 0, "evaluation rows must not be empty")
    rets = sorted(row["ret"] for row in rows)
    farm_n = sum(row["farm_n"] for row in rows)
    return {
        "n": n,
        # schema-v5 keeps full floats in the archive; only the CLI/leaderboard formats them for display.
        # This removes gate flips at the line caused by round(2) per episode followed by another round after aggregation.
        "ret_mean": sum(rets) / n,
        "ret_median": statistics.median(rets),
        "died": sum(row["died"] for row in rows),
        "depth_median": statistics.median(row["depth"] for row in rows),
        "l3": sum(row["depth"] >= 3 for row in rows),
        "kills_mean": sum(row["kills"] for row in rows) / n,
        "farm_tau_mean": (sum(row["farm_tau_sum"] for row in rows)
                          / max(1, farm_n)),
        "farm_descend_rate": (sum(row["farm_descend"] for row in rows)
                              / max(1, farm_n)),
        "override_rate": (sum(row["overrides"] for row in rows)
                          / max(1, sum(row["beats"] for row in rows))),
        "cap_rate": (sum(row["cap"] for row in rows)
                     / max(1, sum(row["windows"] for row in rows))),
        "micro_steps_mean": sum(row["micro_steps"] for row in rows) / n,
        "farm_r_mean": sum(row["farm_r"] for row in rows) / n,
        "farm_w_mean": sum(row["farm_w"] for row in rows) / n,
        "farm_bonus_mean": sum(row["farm_bonus"] for row in rows) / n,
        "farm_worker_wage_mean": (
            sum(row["farm_worker_wage"] for row in rows) / n),
        "farm_kills_mean": sum(row["farm_kills"] for row in rows) / n,
        "farm_worker_kills_mean": (
            sum(row["farm_worker_kills"] for row in rows) / n),
        "nonfarm_r_mean": sum(row["nonfarm_r"] for row in rows) / n,
        "nonfarm_kills_mean": sum(row["nonfarm_kills"] for row in rows) / n,
        "farm_dry_n_mean": sum(row["farm_dry_n"] for row in rows) / n,
        "farm_fresh_n_mean": sum(row["farm_fresh_n"] for row in rows) / n,
        "farm_dry_worker_wage_mean": (
            sum(row["farm_dry_worker_wage"] for row in rows) / n),
        "farm_fresh_worker_wage_mean": (
            sum(row["farm_fresh_worker_wage"] for row in rows) / n),
        "farm_dry_worker_kills_mean": (
            sum(row["farm_dry_worker_kills"] for row in rows) / n),
        "farm_fresh_worker_kills_mean": (
            sum(row["farm_fresh_worker_kills"] for row in rows) / n),
        "farm_voluntary_drinks_mean": (
            sum(row["farm_voluntary_drinks"] for row in rows) / n),
        "farm_reflex_drain_attempts_mean": (
            sum(row["farm_reflex_drain_attempts"] for row in rows) / n),
        "farm_reflex_drains_mean": (
            sum(row["farm_reflex_drains"] for row in rows) / n),
        "farm_multi_drink_window_rate": (
            sum(row["farm_multi_drink_windows"] for row in rows)
            / max(1, farm_n)),
        "ending_belt_heals_mean": (
            sum(row["ending_belt_heals"] for row in rows) / n),
        "victories": sum(row["terminal_kind"] == "victory" for row in rows),
        "game_over": sum(row["terminal_kind"] == "game_over" for row in rows),
        "time_limit_idle": sum(
            row["terminal_kind"] == "time_limit_idle" for row in rows),
        "time_limit_unsettled": sum(
            row["terminal_kind"] == "time_limit_unsettled" for row in rows),
    }


def _validate_identity(identity: Any, label: str,
                       runtime_bundle_sha256: str) -> None:
    _require(isinstance(identity, dict), f"meta.{label} must be an object")
    _require(set(identity) == {
        "kind", "path", "sha256", "num_timesteps", "gate_report_sha256"},
        f"meta.{label} fields are malformed")
    kind = identity["kind"]
    allowed = ({"script", "bc_state_dict", "numpy_policy", "sb3_checkpoint"}
               if label == "worker" else {"numpy_policy"})
    _require(kind in allowed, f"meta.{label}.kind is illegal: {kind!r}")
    _validate_sha256(identity["sha256"], f"meta.{label}.sha256")
    if kind == "script":
        _require(identity["path"] is None and identity["num_timesteps"] is None
                 and identity["gate_report_sha256"] is None,
                 "script worker must not declare a file, step count or gate report")
        expected = script_worker_identity(runtime_bundle_sha256)["sha256"]
        _require(identity["sha256"] == expected,
                 "script worker SHA is not bound to the current protocol source bundle")
    else:
        _require(isinstance(identity["path"], str) and bool(identity["path"]),
                 f"meta.{label}.path missing")
        if kind == "sb3_checkpoint":
            _require(_is_int(identity["num_timesteps"])
                     and identity["num_timesteps"] >= 0,
                     "SB3 worker must declare a non-negative num_timesteps")
        else:
            _require(identity["num_timesteps"] is None,
                     f"{kind} must not declare num_timesteps")
        report_sha = identity["gate_report_sha256"]
        if kind == "bc_state_dict":
            _validate_sha256(report_sha, "meta.worker.gate_report_sha256")
        elif kind == "sb3_checkpoint":
            if report_sha is not None:
                _validate_sha256(
                    report_sha, "meta.worker.gate_report_sha256")
        else:
            _require(report_sha is None,
                     f"{kind} must not declare gate_report_sha256")


def _validate_rows(rows: Any, seeds: list[int]) -> list[Mapping[str, Any]]:
    _require(isinstance(rows, list) and rows, "rows must be a non-empty array")
    _require(len(rows) == len(seeds), "rows count differs from the protocol seeds")
    seen: list[int] = []
    for index, row in enumerate(rows):
        label = f"rows[{index}]"
        _require(isinstance(row, dict), f"{label} must be an object")
        _require(set(row) == _ROW_KEYS, f"{label} fields are malformed")
        for key in (
                "seed", "depth", "kills", "micro_steps",
                "farm_kills", "farm_worker_kills", "nonfarm_kills",
                "farm_dry_n", "farm_fresh_n",
                "farm_dry_worker_kills", "farm_fresh_worker_kills",
                "farm_voluntary_drinks", "farm_reflex_drain_attempts",
                "farm_reflex_drains",
                "farm_multi_drink_windows",
                "farm_max_voluntary_drinks_per_window",
                "ending_belt_heals",
                "farm_n", "farm_tau_sum", "farm_descend",
                "windows", "beats", "overrides", "cap"):
            _require(_is_int(row[key]), f"{label}.{key} must be an integer")
        _require(0 <= row["seed"] <= UINT32_MAX,
                 f"{label}.seed must be in uint32 range")
        _require(1 <= row["depth"] <= 16 and row["kills"] >= 0,
                 f"{label} depth must be in [1,16] and kills must not be negative")
        _require(0 < row["micro_steps"] <= PROTOCOL_MAX_STEPS,
                 f"{label}.micro_steps out of range")
        # beats is the number of executed base action/macro calls; each call consumes at least one
        # micro-step. A fuse rejection neither adds beats nor consumes a micro-step.
        _require(row["farm_n"] >= 0 and 0 < row["windows"] <= row["beats"]
                 <= row["micro_steps"],
                 f"{label} window/action-beat/micro-step counts out of range")
        _require(row["kills"] <= PROTOCOL_MAX_STEPS,
                 f"{label}.kills exceeds the per-episode micro-step budget")
        _require(
            0 <= row["farm_worker_kills"] <= row["farm_kills"]
            and row["nonfarm_kills"] >= 0
            and row["kills"] == row["farm_kills"] + row["nonfarm_kills"],
            f"{label} FARM/non-FARM/worker kill split does not balance",
        )
        _require(0 <= row["farm_descend"] <= row["farm_n"] <= row["windows"],
                 f"{label} FARM count relation is malformed")
        _require(
            row["farm_dry_n"] >= 0
            and row["farm_fresh_n"] >= 0
            and row["farm_dry_n"] + row["farm_fresh_n"] == row["farm_n"],
            f"{label} FARM dry/fresh window split does not balance",
        )
        _require(
            row["farm_dry_worker_kills"] >= 0
            and row["farm_fresh_worker_kills"] >= 0
            and (
                row["farm_dry_worker_kills"]
                + row["farm_fresh_worker_kills"]
                == row["farm_worker_kills"]
            ),
            f"{label} FARM dry/fresh worker kill split does not balance",
        )
        _require(0 <= row["overrides"] <= row["beats"],
                 f"{label}.overrides exceeds beats")
        _require(0 <= row["cap"] <= row["windows"],
                 f"{label}.cap exceeds windows")
        _require(
            row["farm_voluntary_drinks"] >= 0
            and row["farm_reflex_drain_attempts"]
            >= row["farm_reflex_drains"] >= 0
            and row["farm_voluntary_drinks"] + row["farm_reflex_drains"]
            <= row["beats"],
            f"{label} active-drink/reflex-attempt/successful-drain counts out of range")
        _require(
            0 <= row["farm_multi_drink_windows"] <= row["farm_n"]
            and 0 <= row["farm_max_voluntary_drinks_per_window"]
            <= row["farm_voluntary_drinks"]
            and 2 * row["farm_multi_drink_windows"]
            <= row["farm_voluntary_drinks"],
            f"{label} multi-drink window/per-window max active-drink counts out of range")
        _require(
            (row["farm_multi_drink_windows"] == 0
             and row["farm_max_voluntary_drinks_per_window"] <= 1)
            or (row["farm_multi_drink_windows"] > 0
                and row["farm_max_voluntary_drinks_per_window"] >= 2),
            f"{label} multi-drink windows disagree with the per-window max active drinks")
        farm_windows = row["farm_n"]
        voluntary_drinks = row["farm_voluntary_drinks"]
        multi_drink_windows = row["farm_multi_drink_windows"]
        max_voluntary_drinks = row[
            "farm_max_voluntary_drinks_per_window"
        ]
        if voluntary_drinks == 0:
            drink_distribution_feasible = (
                multi_drink_windows == 0 and max_voluntary_drinks == 0
            )
        elif multi_drink_windows == 0:
            # Without a multi-drink window each FARM window has at most one potion; if any active drink happened,
            # the per-window maximum must be exactly 1.
            drink_distribution_feasible = (
                max_voluntary_drinks == 1
                and voluntary_drinks <= farm_windows
            )
        else:
            # Of M multi-drink windows at least one reaches max and the other M-1 drink at least two each;
            # non-multi-drink windows drink at most one each. T must lie between these two achievable bounds.
            minimum_total = (
                max_voluntary_drinks + 2 * (multi_drink_windows - 1)
            )
            maximum_total = (
                multi_drink_windows * max_voluntary_drinks
                + (farm_windows - multi_drink_windows)
            )
            drink_distribution_feasible = (
                minimum_total <= voluntary_drinks <= maximum_total
            )
        _require(
            drink_distribution_feasible,
            f"{label} FARM active drinks N/T/M/max are jointly unachievable")
        _require(0 <= row["ending_belt_heals"] <= 8,
                 f"{label}.ending_belt_heals out of range")
        _require(isinstance(row["died"], bool), f"{label}.died must be a bool")
        ret = _finite_number(row["ret"], f"{label}.ret")
        farm_r = _finite_number(row["farm_r"], f"{label}.farm_r")
        farm_w = _finite_number(row["farm_w"], f"{label}.farm_w")
        farm_bonus = _finite_number(
            row["farm_bonus"], f"{label}.farm_bonus")
        farm_worker_wage = _finite_number(
            row["farm_worker_wage"], f"{label}.farm_worker_wage")
        farm_dry_worker_wage = _finite_number(
            row["farm_dry_worker_wage"],
            f"{label}.farm_dry_worker_wage")
        farm_fresh_worker_wage = _finite_number(
            row["farm_fresh_worker_wage"],
            f"{label}.farm_fresh_worker_wage")
        nonfarm_r = _finite_number(row["nonfarm_r"], f"{label}.nonfarm_r")
        _require(farm_bonus >= 0, f"{label}.farm_bonus must not be negative")
        _require(math.isclose(
            ret, farm_r + nonfarm_r, rel_tol=1e-12, abs_tol=1e-12),
            f"{label} total return does not balance with the FARM/non-FARM split")
        _require(math.isclose(
            farm_r, farm_w + farm_bonus,
            rel_tol=1e-12, abs_tol=1e-12),
            f"{label} FARM R/W/bonus split does not balance")
        _require(
            farm_worker_wage
            == farm_dry_worker_wage + farm_fresh_worker_wage,
            f"{label} FARM dry/fresh worker wage split does not balance",
        )
        for stratum, windows, wage, kills in (
                ("dry", row["farm_dry_n"], farm_dry_worker_wage,
                 row["farm_dry_worker_kills"]),
                ("fresh", row["farm_fresh_n"], farm_fresh_worker_wage,
                 row["farm_fresh_worker_kills"])):
            _require(
                windows > 0 or (wage == 0.0 and kills == 0),
                f"{label} FARM {stratum} has worker entries but no windows",
            )
        tau_mean = _finite_number(row["farm_tau_mean"], f"{label}.farm_tau_mean")
        tau_sum = _finite_number(row["farm_tau_sum"], f"{label}.farm_tau_sum")
        _require(tau_mean >= 0 and tau_sum >= 0, f"{label} FARM tau must not be negative")
        _require(tau_sum <= row["micro_steps"],
                 f"{label}.farm_tau_sum exceeds this episode's micro_steps")
        expected_tau = tau_sum / max(1, row["farm_n"])
        _require(math.isclose(
            tau_mean, expected_tau, rel_tol=1e-12, abs_tol=1e-12),
                 f"{label}.farm_tau_mean disagrees with farm_tau_sum")
        if row["farm_n"] == 0:
            _require(
                farm_r == farm_w == farm_bonus == 0.0
                and row["farm_worker_wage"] == 0.0
                and row["farm_kills"] == row["farm_worker_kills"] == 0
                and row["farm_voluntary_drinks"]
                == row["farm_reflex_drain_attempts"]
                == row["farm_reflex_drains"]
                == row["farm_multi_drink_windows"]
                == row["farm_max_voluntary_drinks_per_window"]
                == 0,
                f"{label} has FARM entries but no FARM window",
            )
        terminal = row["terminal_kind"]
        _require(isinstance(terminal, str) and terminal in _TERMINAL_KINDS,
                 f"{label}.terminal_kind is illegal: {terminal!r}")
        _require(row["died"] == (terminal == "death"),
                 f"{label}.died disagrees with terminal_kind")
        if terminal in {"time_limit_idle", "time_limit_unsettled"}:
            _require(row["micro_steps"] == PROTOCOL_MAX_STEPS,
                     f"{label} time-limit terminal did not exactly exhaust the budget")
        sequence = row["mode_seq"]
        _require(isinstance(sequence, str), f"{label}.mode_seq must be a string")
        death_marker = sequence.endswith("†")
        core_sequence = sequence[:-1] if death_marker else sequence
        _require("†" not in core_sequence and set(core_sequence) <= {"F", "D", "R"},
                 f"{label}.mode_seq has illegal/misplaced markers")
        _require(death_marker == row["died"],
                 f"{label}.mode_seq death marker disagrees with died")
        _require(len(core_sequence) == row["windows"],
                 f"{label}.mode_seq length disagrees with windows")
        _require(core_sequence.count("F") == row["farm_n"],
                 f"{label}.mode_seq FARM count disagrees with farm_n")
        seen.append(row["seed"])
    _require(seen == seeds, "rows seed order/set does not exactly match the protocol seeds")
    _require(len(seen) == len(set(seen)), "rows contain a duplicate seed")
    return rows


def _validate_agg(agg: Any, rows: list[Mapping[str, Any]], worker_kind: str) -> None:
    _require(isinstance(agg, dict), "agg must be an object")
    if worker_kind == "script":
        expected_key_sets = {frozenset(_BASE_AGG_KEYS)}
    else:
        # Preserve validation of immutable pre-gear-ledger schema-v5 archives.
        # Current evaluators always emit the complete second set, and R7
        # separately requires it before scientific analysis.  Making the
        # four-field group all-or-none prevents a partially forged ledger.
        expected_key_sets = {
            frozenset(_BASE_AGG_KEYS | _ENGAGEMENT_KEYS),
            frozenset(
                _BASE_AGG_KEYS | _ENGAGEMENT_KEYS
                | _GEAR_ENGAGEMENT_KEYS),
        }
    _require(
        frozenset(agg) in expected_key_sets,
        "agg fields disagree with the worker type/schema",
    )
    integer_keys = {
        "n", "died", "l3", "victories", "game_over",
        "time_limit_idle", "time_limit_unsettled",
    }
    for key in integer_keys:
        _require(_is_int(agg[key]), f"agg.{key} must be an integer")
    for key in _BASE_AGG_KEYS - integer_keys:
        _finite_number(agg[key], f"agg.{key}")
    n = len(rows)
    _require(0 <= agg["died"] <= n and 0 <= agg["l3"] <= n,
             "agg death/L3 counts out of range")
    _require(
        agg["died"] + agg["victories"] + agg["game_over"]
        + agg["time_limit_idle"] + agg["time_limit_unsettled"] == n,
        "agg terminal categories do not exactly cover all episodes",
    )
    for key in ("farm_descend_rate", "override_rate", "cap_rate"):
        _require(0 <= float(agg[key]) <= 1, f"agg.{key} must be in [0,1]")
    recomputed = recompute_agg(rows)
    for key, expected in recomputed.items():
        _require(agg[key] == expected,
                 f"agg.{key} disagrees with the recomputation from rows: {agg[key]!r} != {expected!r}")
    if worker_kind == "script":
        _require(
            all(row["farm_worker_wage"] == 0.0
                and row["farm_worker_kills"] == 0
                and row["farm_dry_worker_wage"] == 0.0
                and row["farm_fresh_worker_wage"] == 0.0
                and row["farm_dry_worker_kills"] == 0
                and row["farm_fresh_worker_kills"] == 0
                for row in rows),
            "script worker archive must not declare learned-worker wage/kills",
        )
    if worker_kind != "script":
        calls = agg["worker_calls"]
        _require(_is_int(calls) and calls >= 0, "agg.worker_calls must be a non-negative integer")
        # The worker callback counts before every proposal; a proposal rejected by the fuse adds to
        # overrides but, per protocol, neither executes a base action nor adds beats. So the legal upper bound
        # is executed beats + explicitly rejected beats; capping at beats alone would wrongly fail real fuse episodes.
        _require(
            calls <= sum(row["beats"] + row["overrides"] for row in rows),
            "agg.worker_calls exceeds all executed/explicitly rejected proposals",
        )
        histogram = agg["worker_action_hist"]
        _require(isinstance(histogram, dict), "agg.worker_action_hist must be an object")
        normalized: dict[int, int] = {}
        for raw_key, value in histogram.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise EvalContractError("worker_action_hist action keys must be integers") from exc
            _require(str(key) == str(raw_key) or raw_key == key,
                     f"worker_action_hist action key is not canonical: {raw_key!r}")
            _require(0 <= key < 15 and key not in normalized,
                     f"worker_action_hist action key is illegal/duplicate: {raw_key!r}")
            _require(_is_int(value) and value >= 0,
                     f"worker_action_hist[{raw_key!r}] must be a non-negative integer")
            normalized[key] = value
        _require(sum(normalized.values()) == calls,
                 "worker_action_hist total disagrees with worker_calls")
        divergences = agg["worker_divergences"]
        _require(_is_int(divergences) and 0 <= divergences <= calls,
                 "agg.worker_divergences must be in [0, worker_calls]")
        divergence = _finite_number(
            agg["script_divergence_rate"], "agg.script_divergence_rate")
        _require(0 <= divergence <= 1,
                 "agg.script_divergence_rate must be in [0,1]")
        _require(agg["script_divergence_rate"]
                 == divergences / max(1, calls),
                 "script_divergence_rate disagrees with the raw divergence count")
        if _GEAR_ENGAGEMENT_KEYS <= set(agg):
            opportunities = agg[
                "worker_action14_mask_opportunities"]
            requests = agg["worker_action14_requests"]
            successes = agg["worker_action14_native_successes"]
            utility_delta = agg[
                "worker_action14_gear_utility_delta"]
            for key, value in (
                ("worker_action14_mask_opportunities", opportunities),
                ("worker_action14_requests", requests),
                ("worker_action14_native_successes", successes),
                ("worker_action14_gear_utility_delta", utility_delta),
            ):
                _require(
                    _is_int(value) and value >= 0,
                    f"agg.{key} must be a non-negative integer",
                )
            _require(
                opportunities <= calls,
                "action14 mask opportunities exceed worker_calls",
            )
            _require(
                requests == normalized.get(14, 0),
                "action14 request count disagrees with worker_action_hist[14]",
            )
            _require(
                successes <= requests <= opportunities,
                "action14 opportunity/request/native-success count ordering is malformed",
            )
            _require(
                (successes == 0) == (utility_delta == 0),
                "action14 native success count disagrees with the zero-ness of the gear utility gain",
            )


def validate_eval_archive(document: Any, *, expected_tag: str | None = None,
                          expected_seeds: Iterable[int] | None = None,
                          expected_worker_kind: str | None = None,
                          expected_worker_sha256: str | None = None,
                          expected_worker_gate_report_sha256: str | None = None,
                          expected_manager_kind: str | None = None,
                          expected_manager_sha256: str | None = None,
                          expected_worker_num_timesteps: int | None = None,
                          expected_bridge_sha256: str | None = None,
                          expected_engine_sha256: str | None = None,
                          expected_game_data_path: str | None = None,
                          expected_game_data_sha256: str | None = None,
                          expected_assets_path: str | None = None,
                          expected_assets_sha256: str | None = None,
                          expected_assets_file_count: int | None = None,
                          expected_runtime_versions: Mapping[str, Any] | None = None,
                          expected_protocol_bundle_sha256: str | None = None,
                          expected_worker_decoding: str | None = None,
                          expected_r16_environment: Mapping[str, Any] | None = None
                          ) -> dict[str, Any]:
    """Strictly validate one schema v5 archive; legacy is never silently let through here."""
    _require(isinstance(document, dict), "evaluation archive must be a JSON object")
    _require(set(document) == {"schema_version", "meta", "agg", "rows"},
             "evaluation archive top-level fields are malformed or belong to an unauthorized legacy schema")
    _require(document["schema_version"] == SCHEMA_VERSION,
             f"evaluation schema must be v{SCHEMA_VERSION}")
    meta = document["meta"]
    _require(isinstance(meta, dict), "meta must be an object")
    _require(set(meta) == {
        "tag", "created_at_utc", "protocol", "worker", "manager", "runtime"},
        "meta fields are malformed")
    tag = meta["tag"]
    _require(isinstance(tag, str) and _TAG_RE.fullmatch(tag) is not None,
             "meta.tag is illegal")
    _require(isinstance(meta["created_at_utc"], str) and bool(meta["created_at_utc"]),
             "meta.created_at_utc missing")
    try:
        created_at = dt.datetime.fromisoformat(meta["created_at_utc"])
    except ValueError as exc:
        raise EvalContractError("meta.created_at_utc is not an ISO-8601 time") from exc
    _require(created_at.tzinfo is not None
             and created_at.utcoffset() == dt.timedelta(0),
             "meta.created_at_utc must be a UTC time")
    if expected_tag is not None:
        _require(tag == expected_tag, f"evaluation tag mislabeled: {tag!r} != {expected_tag!r}")

    protocol = meta["protocol"]
    _require(isinstance(protocol, dict), "meta.protocol must be an object")
    protocol_keys = {
        "name", "version", "environment", "max_steps", "action_selection",
        "manager_forward", "reward", "deterministic", "seeds"}
    # R16 (C3): worker_decoding is an optional key; omitted = argmax (all historical archives);
    # when present only sample is allowed: one definition must not have two spellings, so the identity stays canonical and unique.
    # R16 new-law anchor: r16_environment is an optional key (listing only non-default switches) and may appear together with
    # worker_decoding.
    optional_keys = {"worker_decoding", "r16_environment", "resource_service_recipe",
                     "dive_blocker_recovery_recipe"}
    _require(protocol_keys <= set(protocol)
             and set(protocol) - protocol_keys <= optional_keys,
             "meta.protocol fields are malformed")
    _require("worker_decoding" not in protocol
             or protocol["worker_decoding"] == "sample",
             "meta.protocol.worker_decoding only allows omission (argmax) or sample")
    worker_decoding = protocol.get("worker_decoding", "argmax")
    if expected_worker_decoding is not None:
        _require(worker_decoding == expected_worker_decoding,
                 "worker decoding mode disagrees with the call contract: "
                 f"{worker_decoding!r} != {expected_worker_decoding!r}")
    r16_environment: dict[str, Any] = {}
    if "r16_environment" in protocol:
        r16_environment = validate_r16_environment(protocol["r16_environment"])
    if r16_environment.get("resource_protocol") == "l2-town-v1":
        _require(protocol.get("resource_service_recipe") == resource_service_recipe(
            "l2-town-v1", r16_environment.get("resource_purchase_mode", "full"),
            r16_environment.get("resource_service_policy", "legacy-v1")),
            "Resource service recipe missing or inconsistent with the current implementation")
    else:
        _require("resource_service_recipe" not in protocol,
                 "Disabled resource protocol must omit the service recipe")
    if r16_environment.get("dive_blocker_recovery") == "adjacent-v1":
        _require(protocol.get("dive_blocker_recovery_recipe") == dive_blocker_recovery_recipe("adjacent-v1"),
                 "DIVE blocker recovery recipe missing or inconsistent")
    else:
        _require("dive_blocker_recovery_recipe" not in protocol,
                 "Disabled DIVE blocker recovery must omit its recipe")
    if expected_r16_environment is not None:
        _require(r16_environment == dict(expected_r16_environment),
                 "R16 environment-switch identity disagrees with the call contract: "
                 f"{r16_environment!r} != {dict(expected_r16_environment)!r}")
    _require(protocol["name"] == PROTOCOL_NAME
             and protocol["version"] == PROTOCOL_VERSION,
             "evaluation protocol name/version mismatch")
    _require(protocol["environment"] == "OptionsEnv"
             and protocol["max_steps"] == PROTOCOL_MAX_STEPS
             and protocol["action_selection"] == "argmax_with_action_masks"
             and protocol["manager_forward"] == "numpy_tanh_mlp"
             and protocol["reward"] == "undiscounted_manager_ledger"
             and protocol["deterministic"] is True,
             "evaluation protocol knob drift")
    seeds = protocol["seeds"]
    _require(isinstance(seeds, list) and seeds
             and all(_is_int(seed) and 0 <= seed <= UINT32_MAX for seed in seeds),
             "meta.protocol.seeds must be a non-empty array of uint32 integers")
    _require(len(seeds) == len(set(seeds)), "meta.protocol.seeds contains duplicates")
    if expected_seeds is not None:
        _require(seeds == list(expected_seeds), "evaluation seed list disagrees with the call contract")

    runtime = meta["runtime"]
    _require(isinstance(runtime, dict)
             and set(runtime) == {
                 "bridge", "engine", "content", "versions", "python_protocol"},
             "meta.runtime fields are malformed")
    bridge = runtime["bridge"]
    _require(isinstance(bridge, dict) and set(bridge) == {"path", "sha256"}
             and isinstance(bridge["path"], str) and bool(bridge["path"]),
             "meta.runtime.bridge fields are malformed")
    bridge_sha = _validate_sha256(bridge["sha256"], "meta.runtime.bridge.sha256")
    engine = runtime["engine"]
    _require(isinstance(engine, dict) and set(engine) == {"path", "sha256"}
             and isinstance(engine["path"], str) and bool(engine["path"]),
             "meta.runtime.engine fields are malformed")
    engine_sha = _validate_sha256(engine["sha256"], "meta.runtime.engine.sha256")
    content = runtime["content"]
    _require(isinstance(content, dict)
             and set(content) == {"game_data", "assets"},
             "meta.runtime.content fields are malformed")
    game_data = content["game_data"]
    _require(isinstance(game_data, dict)
             and set(game_data) == {"path", "sha256"}
             and isinstance(game_data["path"], str)
             and pathlib.Path(game_data["path"]).is_absolute()
             and pathlib.Path(game_data["path"]).name
             in {"DIABDAT.MPQ", "diabdat.mpq", "spawn.mpq"},
             "meta.runtime.content.game_data fields are malformed")
    game_data_sha = _validate_sha256(
        game_data["sha256"], "meta.runtime.content.game_data.sha256")
    assets = content["assets"]
    _require(isinstance(assets, dict)
             and set(assets) == {"path", "sha256", "file_count"}
             and isinstance(assets["path"], str)
             and pathlib.Path(assets["path"]).is_absolute()
             and _is_int(assets["file_count"])
             and assets["file_count"] > 0,
             "meta.runtime.content.assets fields are malformed")
    assets_sha = _validate_sha256(
        assets["sha256"], "meta.runtime.content.assets.sha256")
    versions = runtime["versions"]
    _require(isinstance(versions, dict)
             and set(versions) == {"python", "packages"},
             "meta.runtime.versions fields are malformed")
    python_runtime = versions["python"]
    _require(isinstance(python_runtime, dict)
             and set(python_runtime) == {"implementation", "version", "cache_tag"}
             and all(isinstance(value, str) and bool(value)
                     for value in python_runtime.values()),
             "meta.runtime.versions.python fields are malformed")
    packages = versions["packages"]
    # Same cross-platform revision as the collection gate (runtime_versions_identity): compare the public version segment;
    # the archive still stores the full local version (2026-07-27 WSL2 port).
    _require(isinstance(packages, dict)
             and set(packages) == set(RUNTIME_PACKAGE_VERSIONS)
             and all(isinstance(v, str)
                     and (v == RUNTIME_PACKAGE_VERSIONS[k]
                          or v.split("+", 1)[0] == RUNTIME_PACKAGE_VERSIONS[k])
                     for k, v in packages.items()),
             "meta.runtime.versions.packages disagrees with the frozen numeric stack")
    py_protocol = runtime["python_protocol"]
    _require(isinstance(py_protocol, dict)
             and set(py_protocol) == {"sha256", "files"}
             and isinstance(py_protocol["files"], dict) and py_protocol["files"],
             "meta.runtime.python_protocol fields are malformed")
    _require(set(py_protocol["files"]) == set(PROTOCOL_SOURCE_FILES),
             "protocol source bundle file set is incomplete/contains unknown files")
    for name in py_protocol["files"]:
        path = pathlib.PurePosixPath(name)
        _require(not path.is_absolute() and ".." not in path.parts,
                 f"illegal protocol source path: {name!r}")
    bundle_sha = _validate_sha256(
        py_protocol["sha256"], "meta.runtime.python_protocol.sha256")
    _require(source_bundle_sha256(py_protocol["files"]) == bundle_sha,
             "protocol source bundle SHA disagrees with the per-file SHAs")
    if expected_bridge_sha256 is not None:
        _require(bridge_sha == expected_bridge_sha256, "bridge SHA disagrees with the call contract")
    if expected_engine_sha256 is not None:
        _require(engine_sha == expected_engine_sha256,
                 "engine SHA disagrees with the call contract")
    if expected_game_data_path is not None:
        _require(game_data["path"] == expected_game_data_path,
                 "game main MPQ path disagrees with the call contract")
    if expected_game_data_sha256 is not None:
        _require(game_data_sha == expected_game_data_sha256,
                 "game main MPQ SHA disagrees with the call contract")
    if expected_assets_path is not None:
        _require(assets["path"] == expected_assets_path,
                 "Resources path disagrees with the call contract")
    if expected_assets_sha256 is not None:
        _require(assets_sha == expected_assets_sha256,
                 "Resources tree SHA disagrees with the call contract")
    if expected_assets_file_count is not None:
        _require(assets["file_count"] == expected_assets_file_count,
                 "Resources file count disagrees with the call contract")
    if expected_runtime_versions is not None:
        _require(versions == expected_runtime_versions,
                 "Python/numeric dependency versions disagree with the call contract")
    if expected_protocol_bundle_sha256 is not None:
        _require(bundle_sha == expected_protocol_bundle_sha256,
                 "protocol source bundle SHA disagrees with the call contract")

    _validate_identity(meta["worker"], "worker", bundle_sha)
    _validate_identity(meta["manager"], "manager", bundle_sha)
    if expected_worker_kind is not None:
        _require(meta["worker"]["kind"] == expected_worker_kind,
                 "worker type disagrees with the call inputs")
    if expected_worker_sha256 is not None:
        _require(meta["worker"]["sha256"] == expected_worker_sha256,
                 "worker SHA disagrees with the call inputs (suspected copied/mislabeled archive)")
    if expected_worker_gate_report_sha256 is not None:
        _require(meta["worker"]["gate_report_sha256"]
                 == expected_worker_gate_report_sha256,
                 "worker gate report SHA disagrees with the call inputs")
    if expected_manager_kind is not None:
        _require(meta["manager"]["kind"] == expected_manager_kind,
                 "manager type disagrees with the call inputs")
    if expected_manager_sha256 is not None:
        _require(meta["manager"]["sha256"] == expected_manager_sha256,
                 "manager SHA disagrees with the call inputs (suspected copied/mislabeled archive)")
    if expected_worker_num_timesteps is not None:
        _require(meta["worker"]["num_timesteps"] == expected_worker_num_timesteps,
                 "worker num_timesteps disagrees with the call inputs")

    rows = _validate_rows(document["rows"], seeds)
    _validate_agg(document["agg"], rows, meta["worker"]["kind"])
    return document


def strict_json_loads(payload: str | bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise EvalContractError(f"JSON contains a non-standard/non-finite constant: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvalContractError(f"JSON object has a duplicate key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload, parse_constant=reject_constant,
                          object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise EvalContractError(f"evaluation JSON cannot be parsed: {exc}") from exc


def read_eval_archive(path: str | pathlib.Path, *,
                      trusted_legacy_sha256: str | None = None,
                      **expected: Any) -> dict[str, Any]:
    """Read and validate an archive; legacy is let through only explicitly, against the full file SHA the caller supplies."""
    archive_path = pathlib.Path(path)
    payload = archive_path.read_bytes()
    document = strict_json_loads(payload)
    if isinstance(document, dict) and document.get("schema_version") == SCHEMA_VERSION:
        return validate_eval_archive(document, **expected)
    _validate_sha256(trusted_legacy_sha256, "trusted_legacy_sha256")
    actual = hashlib.sha256(payload).hexdigest()
    _require(actual == trusted_legacy_sha256,
             f"legacy archive SHA mismatch: {actual} != {trusted_legacy_sha256}")
    _require(isinstance(document, dict) and set(document) == {"agg", "rows"},
             "legacy archive basic structure is malformed")
    return document


@contextlib.contextmanager
def exclusive_lock(lock_path: str | pathlib.Path, purpose: str = "resource"):
    """Take a cross-process, non-blocking exclusive lock on a persistent lock file."""
    lock_path = pathlib.Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OutputReservationError(
                f"{purpose} is held by another evaluation process: {lock_path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def reserve_output(path: str | pathlib.Path):
    """Make a cross-process, non-blocking exclusive reservation of one target via a persistent lock file."""
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with exclusive_lock(lock_path, "same tag"):
        if output.exists():
            raise OutputReservationError(f"evaluation archive already exists, refusing to overwrite: {output}")
        yield output
